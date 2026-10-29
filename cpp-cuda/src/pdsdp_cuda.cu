#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace pdsdp {

constexpr double kSqrt2 = 1.41421356237309504880168872420969808;
constexpr int kThreads = 256;
constexpr const char* kVersion = "0.1.0";
static_assert(sizeof(int) == sizeof(std::int32_t), "checkpoint requires 32-bit int");

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    fail(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* call) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(std::string(call) + " failed with status " + std::to_string(status));
  }
}

void check_cusparse(cusparseStatus_t status, const char* call) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    fail(std::string(call) + ": " + cusparseGetErrorString(status));
  }
}

void check_cusolver(cusolverStatus_t status, const char* call) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    fail(std::string(call) + " failed with status " + std::to_string(status));
  }
}

#define CUDA_CHECK(call) ::pdsdp::check_cuda((call), #call)
#define CUBLAS_CHECK(call) ::pdsdp::check_cublas((call), #call)
#define CUSPARSE_CHECK(call) ::pdsdp::check_cusparse((call), #call)
#define CUSOLVER_CHECK(call) ::pdsdp::check_cusolver((call), #call)

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t size) { reset(size); }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      swap(other);
    }
    return *this;
  }
  void reset(std::size_t size) {
    release();
    size_ = size;
    if (size_) CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
  }
  void zero() {
    if (size_) CUDA_CHECK(cudaMemset(data_, 0, size_ * sizeof(T)));
  }
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  void release() {
    if (data_) cudaFree(data_);
    data_ = nullptr;
    size_ = 0;
  }
  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

struct CooEntry {
  int row;
  int col;
  double value;
};

struct CsrHost {
  int rows = 0;
  int cols = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<double> values;
};

CsrHost make_csr(int rows, int cols, std::vector<CooEntry> entries) {
  std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.row, lhs.col) < std::tie(rhs.row, rhs.col);
  });
  CsrHost out;
  out.rows = rows;
  out.cols = cols;
  out.row_ptr.assign(static_cast<std::size_t>(rows) + 1, 0);
  out.col_idx.reserve(entries.size());
  out.values.reserve(entries.size());
  for (std::size_t i = 0; i < entries.size();) {
    const int row = entries[i].row;
    const int col = entries[i].col;
    double value = 0.0;
    std::size_t j = i;
    while (j < entries.size() && entries[j].row == row && entries[j].col == col) {
      value += entries[j].value;
      ++j;
    }
    if (value != 0.0) {
      out.col_idx.push_back(col);
      out.values.push_back(value);
      ++out.row_ptr[static_cast<std::size_t>(row) + 1];
    }
    i = j;
  }
  std::partial_sum(out.row_ptr.begin(), out.row_ptr.end(), out.row_ptr.begin());
  return out;
}

CsrHost transpose_csr(const CsrHost& matrix) {
  std::vector<CooEntry> entries;
  entries.reserve(matrix.values.size());
  for (int row = 0; row < matrix.rows; ++row) {
    for (int p = matrix.row_ptr[row]; p < matrix.row_ptr[row + 1]; ++p) {
      entries.push_back({matrix.col_idx[p], row, matrix.values[p]});
    }
  }
  return make_csr(matrix.cols, matrix.rows, std::move(entries));
}

struct ScalingInfo {
  double b_scale = 1.0;
  double c_scale = 1.0;
  double original_b_inf = 0.0;
  double original_c_inf = 0.0;
  std::vector<double> row_norm;
  std::vector<double> col_norm;
};

struct Problem {
  int m = 0;
  int n = 0;
  std::vector<int> raw_dims;
  std::vector<int> original_cone_dims;
  std::vector<int> cone_dims;
  std::vector<int> old_to_new;
  std::vector<int> var_ptr;
  std::vector<double> b;
  std::vector<double> c;
  CsrHost a;
  CsrHost at;
  ScalingInfo scaling;
};

std::vector<std::string> numeric_tokens(const std::string& line) {
  const auto first = line.find_first_not_of(" \t\r\n");
  if (first == std::string::npos || line[first] == '*' || line[first] == '"' ||
      line[first] == '#') {
    return {};
  }
  std::string normalized = line;
  for (char& ch : normalized) {
    if (ch == ',' || ch == '{' || ch == '}' || ch == '(' || ch == ')' ||
        ch == '=' || ch == ':') {
      ch = ' ';
    }
  }
  std::istringstream stream(normalized);
  std::vector<std::string> tokens;
  for (std::string token; stream >> token;) tokens.push_back(std::move(token));
  return tokens;
}

Problem load_sdpa(const fs::path& path) {
  const auto start = Clock::now();
  std::ifstream input(path);
  if (!input) fail("Cannot open SDPA file: " + path.string());

  std::vector<std::string> header;
  std::size_t target = std::numeric_limits<std::size_t>::max();
  bool have_block_dimensions = false;
  std::string line;
  while (header.size() < target && std::getline(input, line)) {
    auto tokens = numeric_tokens(line);
    header.insert(header.end(), tokens.begin(), tokens.end());
    if (header.size() >= 2 && target == std::numeric_limits<std::size_t>::max()) {
      const int blocks = std::stoi(header[1]);
      target = static_cast<std::size_t>(2 + blocks);
    }
    if (!have_block_dimensions &&
        target != std::numeric_limits<std::size_t>::max() &&
        header.size() >= target) {
      have_block_dimensions = true;
      target = static_cast<std::size_t>(2 + std::stoi(header[1]) + std::stoi(header[0]));
    }
  }
  if (target == std::numeric_limits<std::size_t>::max() || header.size() != target) {
    fail("Invalid SDPA header or RHS/matrix boundary in " + path.string());
  }

  Problem problem;
  problem.m = std::stoi(header[0]);
  const int nblocks = std::stoi(header[1]);
  problem.raw_dims.resize(nblocks);
  for (int i = 0; i < nblocks; ++i) {
    problem.raw_dims[i] = std::stoi(header[static_cast<std::size_t>(2 + i)]);
    if (problem.raw_dims[i] == 0) fail("Zero-dimensional SDPA block");
  }
  problem.b.resize(problem.m);
  for (int i = 0; i < problem.m; ++i) {
    problem.b[i] = std::stod(header[static_cast<std::size_t>(2 + nblocks + i)]);
  }

  std::vector<int> raw_psd_to_original(nblocks, -1);
  std::vector<int> raw_lp_offset(nblocks, -1);
  for (int block = 0; block < nblocks; ++block) {
    if (problem.raw_dims[block] > 0) {
      raw_psd_to_original[block] = static_cast<int>(problem.original_cone_dims.size());
      problem.original_cone_dims.push_back(problem.raw_dims[block]);
    }
  }
  for (int block = 0; block < nblocks; ++block) {
    if (problem.raw_dims[block] < 0) {
      raw_lp_offset[block] = static_cast<int>(problem.original_cone_dims.size());
      problem.original_cone_dims.insert(
          problem.original_cone_dims.end(), -problem.raw_dims[block], 1);
    }
  }

  const int cones = static_cast<int>(problem.original_cone_dims.size());
  std::vector<int> new_to_old(cones);
  std::iota(new_to_old.begin(), new_to_old.end(), 0);
  std::stable_sort(new_to_old.begin(), new_to_old.end(), [&](int lhs, int rhs) {
    return problem.original_cone_dims[lhs] < problem.original_cone_dims[rhs];
  });
  problem.old_to_new.resize(cones);
  problem.cone_dims.resize(cones);
  for (int index = 0; index < cones; ++index) {
    problem.old_to_new[new_to_old[index]] = index;
    problem.cone_dims[index] = problem.original_cone_dims[new_to_old[index]];
  }
  problem.var_ptr.assign(static_cast<std::size_t>(cones) + 1, 0);
  for (int cone = 0; cone < cones; ++cone) {
    const std::int64_t dim = problem.cone_dims[cone];
    const std::int64_t next = static_cast<std::int64_t>(problem.var_ptr[cone]) +
                              dim * (dim + 1) / 2;
    if (next > std::numeric_limits<int>::max()) fail("Packed SDP exceeds int32 capacity");
    problem.var_ptr[cone + 1] = static_cast<int>(next);
  }
  problem.n = problem.var_ptr.back();
  problem.c.assign(problem.n, 0.0);

  auto packed_col = [&](int original_cone, int row, int col) {
    if (row < col) std::swap(row, col);
    const int sorted_cone = problem.old_to_new[original_cone];
    return problem.var_ptr[sorted_cone] + row * (row + 1) / 2 + col;
  };

  std::vector<CooEntry> entries;
  bool in_comment = false;
  std::uint64_t input_entries = 0;
  while (std::getline(input, line)) {
    const auto first = line.find_first_not_of(" \t\r\n");
    const std::string stripped = first == std::string::npos ? "" : line.substr(first);
    if (stripped.rfind("BEGIN.COMMENT", 0) == 0) {
      in_comment = true;
      continue;
    }
    if (stripped.rfind("END.COMMENT", 0) == 0) {
      in_comment = false;
      continue;
    }
    if (in_comment) continue;
    auto tokens = numeric_tokens(line);
    if (tokens.empty()) continue;
    if (tokens.size() != 5) fail("Invalid SDPA matrix line: " + line);
    const int matrix_id = std::stoi(tokens[0]);
    const int raw_block = std::stoi(tokens[1]) - 1;
    int row = std::stoi(tokens[2]) - 1;
    int col = std::stoi(tokens[3]) - 1;
    double value = std::stod(tokens[4]);
    if (matrix_id < 0 || matrix_id > problem.m || raw_block < 0 || raw_block >= nblocks) {
      fail("SDPA matrix/block index out of range");
    }
    int original_cone;
    const int dim = problem.raw_dims[raw_block];
    if (dim > 0) {
      if (row < 0 || row >= dim || col < 0 || col >= dim) fail("PSD index out of range");
      original_cone = raw_psd_to_original[raw_block];
      if (row != col) value *= kSqrt2;
    } else {
      if (row != col || row < 0 || row >= -dim) fail("Invalid diagonal block entry");
      original_cone = raw_lp_offset[raw_block] + row;
      row = col = 0;
    }
    const int variable = packed_col(original_cone, row, col);
    if (matrix_id == 0) {
      problem.c[variable] -= value;
    } else {
      entries.push_back({matrix_id - 1, variable, value});
    }
    ++input_entries;
  }
  if (in_comment) fail("Unterminated BEGIN.COMMENT block");
  problem.a = make_csr(problem.m, problem.n, std::move(entries));
  std::cout << "PDSDP_CUDA_SDPA_LOAD m=" << problem.m << " n=" << problem.n
            << " raw_blocks=" << nblocks << " cones=" << cones
            << " input_entries=" << input_entries << " nnzA=" << problem.a.values.size()
            << " time_sec=" << seconds_since(start) << '\n';
  return problem;
}

double vector_norm2(const std::vector<double>& values) {
  long double sum = 0.0;
  for (double value : values) sum += static_cast<long double>(value) * value;
  return std::sqrt(static_cast<double>(sum));
}

void scale_problem(Problem& problem) {
  const auto start = Clock::now();
  auto& scaling = problem.scaling;
  scaling.row_norm.assign(problem.m, 1.0);
  scaling.col_norm.assign(problem.n, 1.0);
  for (double value : problem.b) scaling.original_b_inf = std::max(scaling.original_b_inf, std::abs(value));
  for (std::size_t cone = 0; cone < problem.cone_dims.size(); ++cone) {
    const int dim = problem.cone_dims[cone];
    int cursor = problem.var_ptr[cone];
    for (int row = 0; row < dim; ++row) {
      for (int col = 0; col <= row; ++col, ++cursor) {
        const double entry = std::abs(problem.c[cursor]) / (row == col ? 1.0 : kSqrt2);
        scaling.original_c_inf = std::max(scaling.original_c_inf, entry);
      }
    }
  }

  std::vector<double> row_scale(problem.m);
  std::vector<double> col_scale(problem.n);
  auto equilibration_step = [&](bool one_norm) {
    std::fill(row_scale.begin(), row_scale.end(), 0.0);
    std::fill(col_scale.begin(), col_scale.end(), 0.0);
    for (int row = 0; row < problem.m; ++row) {
      double aggregate = 0.0;
      for (int p = problem.a.row_ptr[row]; p < problem.a.row_ptr[row + 1]; ++p) {
        const double magnitude = std::abs(problem.a.values[p]);
        aggregate = one_norm ? aggregate + magnitude : std::max(aggregate, magnitude);
        if (one_norm) {
          col_scale[problem.a.col_idx[p]] += magnitude;
        } else {
          col_scale[problem.a.col_idx[p]] =
              std::max(col_scale[problem.a.col_idx[p]], magnitude);
        }
      }
      row_scale[row] = std::sqrt(aggregate);
    }
    for (double& value : col_scale) value = std::sqrt(value);
    for (std::size_t cone = 0; cone < problem.cone_dims.size(); ++cone) {
      double block_max = 0.0;
      for (int col = problem.var_ptr[cone]; col < problem.var_ptr[cone + 1]; ++col) {
        block_max = std::max(block_max, col_scale[col]);
      }
      block_max = block_max == 0.0 ? 1.0 : block_max;
      for (int col = problem.var_ptr[cone]; col < problem.var_ptr[cone + 1]; ++col) {
        col_scale[col] = block_max;
      }
    }
    for (double& value : row_scale) if (value == 0.0) value = 1.0;
    for (double& value : col_scale) if (value == 0.0) value = 1.0;

#pragma omp parallel for schedule(static)
    for (int row = 0; row < problem.m; ++row) {
      problem.b[row] /= row_scale[row];
      scaling.row_norm[row] *= row_scale[row];
      for (int p = problem.a.row_ptr[row]; p < problem.a.row_ptr[row + 1]; ++p) {
        problem.a.values[p] /= row_scale[row] * col_scale[problem.a.col_idx[p]];
      }
    }
#pragma omp parallel for schedule(static)
    for (int col = 0; col < problem.n; ++col) {
      problem.c[col] /= col_scale[col];
      scaling.col_norm[col] *= col_scale[col];
    }
  };

  for (int iteration = 0; iteration < 20; ++iteration) equilibration_step(false);
  equilibration_step(true);
  scaling.b_scale = 1.0 + vector_norm2(problem.b);
  scaling.c_scale = 1.0 + vector_norm2(problem.c);
#pragma omp parallel for schedule(static)
  for (int row = 0; row < problem.m; ++row) problem.b[row] /= scaling.b_scale;
#pragma omp parallel for schedule(static)
  for (int col = 0; col < problem.n; ++col) problem.c[col] /= scaling.c_scale;
  problem.at = transpose_csr(problem.a);
  std::cout << "PDSDP_CUDA_SCALING time_sec=" << seconds_since(start)
            << " b_scale=" << scaling.b_scale << " c_scale=" << scaling.c_scale << '\n';
}

class CsrDevice {
 public:
  CsrDevice(cusparseHandle_t handle, const CsrHost& host)
      : rows_(host.rows), cols_(host.cols), nnz_(static_cast<int>(host.values.size())),
        row_ptr_(host.row_ptr.size()), col_idx_(host.col_idx.size()), values_(host.values.size()) {
    CUDA_CHECK(cudaMemcpy(row_ptr_.data(), host.row_ptr.data(), host.row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(col_idx_.data(), host.col_idx.data(), host.col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(values_.data(), host.values.data(), host.values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matrix_, rows_, cols_, nnz_, row_ptr_.data(), col_idx_.data(), values_.data(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&x_desc_, cols_, nullptr, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&y_desc_, rows_, nullptr, CUDA_R_64F));
    double alpha = 1.0, beta = 0.0;
    std::size_t bytes = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_, x_desc_, &beta,
        y_desc_, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &bytes));
    workspace_.reset(bytes);
  }
  ~CsrDevice() {
    if (x_desc_) cusparseDestroyDnVec(x_desc_);
    if (y_desc_) cusparseDestroyDnVec(y_desc_);
    if (matrix_) cusparseDestroySpMat(matrix_);
  }
  CsrDevice(const CsrDevice&) = delete;
  CsrDevice& operator=(const CsrDevice&) = delete;
  void multiply(cusparseHandle_t handle, const double* x, double* y) {
    CUSPARSE_CHECK(cusparseDnVecSetValues(x_desc_, const_cast<double*>(x)));
    CUSPARSE_CHECK(cusparseDnVecSetValues(y_desc_, y));
    const double alpha = 1.0, beta = 0.0;
    CUSPARSE_CHECK(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_, x_desc_, &beta,
        y_desc_, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, workspace_.data()));
  }

 private:
  int rows_, cols_, nnz_;
  DeviceBuffer<int> row_ptr_, col_idx_;
  DeviceBuffer<double> values_;
  DeviceBuffer<unsigned char> workspace_;
  cusparseSpMatDescr_t matrix_ = nullptr;
  cusparseDnVecDescr_t x_desc_ = nullptr;
  cusparseDnVecDescr_t y_desc_ = nullptr;
};

__global__ void add_kernel(double* x, const double* y, double alpha, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += alpha * y[i];
}

__global__ void dual_step_kernel(double* y, const double* product, const double* b,
                                 double step, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += step * (product[i] - b[i]);
}

__global__ void extrapolate_kernel(double* out, const double* current,
                                   const double* last, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 2.0 * current[i] - last[i];
}

__global__ void difference_kernel(double* out, const double* lhs,
                                  const double* rhs, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = lhs[i] - rhs[i];
}

__global__ void halpern_kernel(double* out, const double* candidate,
                              const double* last, const double* anchor,
                              double alpha1, double alpha2, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = alpha1 * (1.9 * candidate[i] - 0.9 * last[i]) + alpha2 * anchor[i];
  }
}

__global__ void scale_inplace_kernel(double* x, const double* scale,
                                     double scalar, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= scalar * scale[i];
}

__global__ void original_residual_kernel(double* out, const double* product,
                                         const double* b, const double* row_scale,
                                         double b_scale, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = b_scale * row_scale[i] * (product[i] - b[i]);
}

__global__ void unpack_svec_kernel(const double* packed, double* dense,
                                   int dim, int batch, int packed_len) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = packed_len * batch;
  if (index >= total) return;
  const int matrix = index / packed_len;
  const int k = index - matrix * packed_len;
  const int row = static_cast<int>((sqrt(8.0 * static_cast<double>(k) + 1.0) - 1.0) * 0.5);
  const int col = k - row * (row + 1) / 2;
  const double value = packed[index] / (row == col ? 1.0 : kSqrt2);
  double* target = dense + static_cast<std::size_t>(matrix) * dim * dim;
  target[row + col * dim] = value;
  target[col + row * dim] = value;
}

__global__ void pack_svec_kernel(double* packed, const double* dense,
                                 int dim, int batch, int packed_len) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = packed_len * batch;
  if (index >= total) return;
  const int matrix = index / packed_len;
  const int k = index - matrix * packed_len;
  const int row = static_cast<int>((sqrt(8.0 * static_cast<double>(k) + 1.0) - 1.0) * 0.5);
  const int col = k - row * (row + 1) / 2;
  const double* source = dense + static_cast<std::size_t>(matrix) * dim * dim;
  packed[index] = source[row + col * dim] * (row == col ? 1.0 : kSqrt2);
}

__global__ void clamp_scalar_kernel(double* x, int start, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) x[start + i] = fmax(x[start + i], 0.0);
}

__global__ void eigen_sqrt_scale_kernel(double* vectors, const double* values,
                                        int dim, int batch) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = dim * dim * batch;
  if (index >= total) return;
  const int matrix_offset = index / (dim * dim);
  const int local = index - matrix_offset * dim * dim;
  const int column = local / dim;
  const double lambda = fmax(values[matrix_offset * dim + column], 0.0);
  vectors[index] *= sqrt(lambda);
}

struct ProjectionTask {
  int start = 0;
  int dim = 0;
  int batch = 0;
  int packed_len = 0;
  bool batched = false;
  DeviceBuffer<double> dense;
  DeviceBuffer<double> result;
  DeviceBuffer<double> eigenvalues;
  DeviceBuffer<double> workspace;
  DeviceBuffer<int> info;
};

class ProjectionEngine {
 public:
  ProjectionEngine(cusolverDnHandle_t solver, cublasHandle_t blas,
                   const std::vector<int>& dims, const std::vector<int>& var_ptr)
      : solver_(solver), blas_(blas) {
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&jacobi_info_));
    CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(jacobi_info_, 1e-7));
    CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(jacobi_info_, 100));
    int cone = 0;
    while (cone < static_cast<int>(dims.size())) {
      const int dim = dims[cone];
      int end = cone + 1;
      while (end < static_cast<int>(dims.size()) && dims[end] == dim) ++end;
      if (dim == 1) {
        scalar_ranges_.push_back({var_ptr[cone], end - cone});
      } else if (dim <= 100) {
        for (int first = cone; first < end; first += 128) {
          create_task(var_ptr[first], dim, std::min(128, end - first), true);
        }
      } else {
        for (int first = cone; first < end; ++first) {
          create_task(var_ptr[first], dim, 1, false);
        }
      }
      cone = end;
    }
  }
  ~ProjectionEngine() {
    if (jacobi_info_) cusolverDnDestroySyevjInfo(jacobi_info_);
  }
  ProjectionEngine(const ProjectionEngine&) = delete;
  ProjectionEngine& operator=(const ProjectionEngine&) = delete;

  void project(double* x) {
    for (const auto& range : scalar_ranges_) {
      clamp_scalar_kernel<<<(range.second + kThreads - 1) / kThreads, kThreads>>>(
          x, range.first, range.second);
    }
    for (auto& task : tasks_) {
      const int total_packed = task.packed_len * task.batch;
      unpack_svec_kernel<<<(total_packed + kThreads - 1) / kThreads, kThreads>>>(
          x + task.start, task.dense.data(), task.dim, task.batch, task.packed_len);
      if (task.batched) {
        CUSOLVER_CHECK(cusolverDnDsyevjBatched(
            solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, task.dim,
            task.dense.data(), task.dim, task.eigenvalues.data(), task.workspace.data(),
            static_cast<int>(task.workspace.size()), task.info.data(), jacobi_info_, task.batch));
      } else {
        CUSOLVER_CHECK(cusolverDnDsyevd(
            solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, task.dim,
            task.dense.data(), task.dim, task.eigenvalues.data(), task.workspace.data(),
            static_cast<int>(task.workspace.size()), task.info.data()));
      }
      const int dense_count = task.dim * task.dim * task.batch;
      eigen_sqrt_scale_kernel<<<(dense_count + kThreads - 1) / kThreads, kThreads>>>(
          task.dense.data(), task.eigenvalues.data(), task.dim, task.batch);
      const double alpha = 1.0, beta = 0.0;
      if (task.batch == 1) {
        CUBLAS_CHECK(cublasDgemm(
            blas_, CUBLAS_OP_N, CUBLAS_OP_T, task.dim, task.dim, task.dim, &alpha,
            task.dense.data(), task.dim, task.dense.data(), task.dim, &beta,
            task.result.data(), task.dim));
      } else {
        const long long stride = static_cast<long long>(task.dim) * task.dim;
        CUBLAS_CHECK(cublasDgemmStridedBatched(
            blas_, CUBLAS_OP_N, CUBLAS_OP_T, task.dim, task.dim, task.dim, &alpha,
            task.dense.data(), task.dim, stride, task.dense.data(), task.dim, stride,
            &beta, task.result.data(), task.dim, stride, task.batch));
      }
      pack_svec_kernel<<<(total_packed + kThreads - 1) / kThreads, kThreads>>>(
          x + task.start, task.result.data(), task.dim, task.batch, task.packed_len);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  void verify_last_projection() {
    for (std::size_t task_index = 0; task_index < tasks_.size(); ++task_index) {
      const auto& task = tasks_[task_index];
      std::vector<int> status(task.batch);
      CUDA_CHECK(cudaMemcpy(status.data(), task.info.data(),
                            status.size() * sizeof(int), cudaMemcpyDeviceToHost));
      for (int matrix = 0; matrix < task.batch; ++matrix) {
        if (status[matrix] != 0) {
          fail("cuSOLVER projection failed for task " + std::to_string(task_index) +
               ", matrix " + std::to_string(matrix) +
               ", info=" + std::to_string(status[matrix]));
        }
      }
    }
  }

 private:
  void create_task(int start, int dim, int batch, bool batched) {
    ProjectionTask task;
    task.start = start;
    task.dim = dim;
    task.batch = batch;
    task.packed_len = dim * (dim + 1) / 2;
    task.batched = batched;
    const std::size_t dense_count = static_cast<std::size_t>(dim) * dim * batch;
    task.dense.reset(dense_count);
    task.result.reset(dense_count);
    task.eigenvalues.reset(static_cast<std::size_t>(dim) * batch);
    task.info.reset(batch);
    int workspace_elements = 0;
    if (batched) {
      CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(
          solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, dim,
          task.dense.data(), dim, task.eigenvalues.data(), &workspace_elements,
          jacobi_info_, batch));
    } else {
      CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
          solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, dim,
          task.dense.data(), dim, task.eigenvalues.data(), &workspace_elements));
    }
    task.workspace.reset(workspace_elements);
    tasks_.push_back(std::move(task));
  }

  cusolverDnHandle_t solver_;
  cublasHandle_t blas_;
  syevjInfo_t jacobi_info_ = nullptr;
  std::vector<std::pair<int, int>> scalar_ranges_;
  std::vector<ProjectionTask> tasks_;
};

struct Options {
  double time_limit = 3600.0;
  std::int64_t max_iterations = 1000000000LL;
  double tolerance = 1e-4;
  int evaluate_frequency = 40;
  int print_frequency = 1000;
};

struct Metrics {
  double primal = std::numeric_limits<double>::infinity();
  double dual = std::numeric_limits<double>::infinity();
  double gap = std::numeric_limits<double>::infinity();
  double primal_obj = std::numeric_limits<double>::infinity();
  double dual_obj = -std::numeric_limits<double>::infinity();
  double residual() const { return std::max({primal, dual, gap}); }
};

class Solver {
 public:
  Solver(Problem& problem, Options options)
      : p_(problem), options_(options),
        d_b_(p_.m), d_c_(p_.n), d_row_norm_(p_.m), d_col_norm_(p_.n),
        x_(p_.n), last_x_(p_.n), temp_x_(p_.n), avg_x_(p_.n), restart_x_(p_.n),
        delta_x_(p_.n), gradient_(p_.n), projected_slack_(p_.n),
        y_(p_.m), last_y_(p_.m), temp_y_(p_.m), avg_y_(p_.m), restart_y_(p_.m),
        delta_y_(p_.m), ax_(p_.m), last_ax_(p_.m), avg_ax_(p_.m),
        delta_ax_(p_.m), extrapolated_ax_(p_.m), residual_buffer_(p_.m),
        power_q_(p_.m), power_z_(p_.m), power_atq_(p_.n) {
    CUSPARSE_CHECK(cusparseCreate(&sparse_));
    CUBLAS_CHECK(cublasCreate(&blas_));
    CUSOLVER_CHECK(cusolverDnCreate(&dense_solver_));
    a_ = std::make_unique<CsrDevice>(sparse_, p_.a);
    at_ = std::make_unique<CsrDevice>(sparse_, p_.at);
    projection_ = std::make_unique<ProjectionEngine>(dense_solver_, blas_, p_.cone_dims, p_.var_ptr);
    copy_to_device(d_b_, p_.b);
    copy_to_device(d_c_, p_.c);
    copy_to_device(d_row_norm_, p_.scaling.row_norm);
    copy_to_device(d_col_norm_, p_.scaling.col_norm);
    zero_state();
  }
  ~Solver() {
    projection_.reset();
    at_.reset();
    a_.reset();
    if (dense_solver_) cusolverDnDestroy(dense_solver_);
    if (blas_) cublasDestroy(blas_);
    if (sparse_) cusparseDestroy(sparse_);
  }

  void solve() {
    solver_start_ = Clock::now();
    const double operator_norm = estimate_operator_norm();
    if (!std::isfinite(operator_norm) ||
        operator_norm <= std::numeric_limits<double>::epsilon()) {
      fail("Constraint operator norm is zero or non-finite");
    }
    step_size_ = 0.5 / operator_norm;
    const double b_norm = vector_norm2(p_.b);
    const double c_norm = vector_norm2(p_.c);
    primal_weight_ = (std::isfinite(b_norm) && std::isfinite(c_norm) &&
                      b_norm > std::numeric_limits<double>::epsilon() &&
                      c_norm > std::numeric_limits<double>::epsilon())
                         ? c_norm / b_norm
                         : 1.0;
    std::cout << "PDSDP_CUDA_INIT operator_norm=" << operator_norm
              << " step_size=" << step_size_ << " primal_weight=" << primal_weight_
              << " elapsed_sec=" << elapsed() << '\n';

    while (true) {
      const double now = elapsed();
      if (iterations_ >= options_.max_iterations) {
        status_ = "OUT_OF_ITER";
        break;
      }
      if (now > options_.time_limit) {
        status_ = "OUT_OF_TIME";
        break;
      }
      ++iterations_;
      ++interval_iterations_;
      iteration();
      if (numerical_error_) {
        status_ = "NUMERICAL_ERROR";
        break;
      }

      const double fp_residual = fixed_point_residual_;
      int restart_condition = 0;
      if (interval_iterations_ <= 1) {
        restart_baseline_ = fp_residual;
      } else if (fp_residual <= 0.2 * restart_baseline_) {
        restart_condition = 1;
      } else if (fp_residual <= 0.9 * restart_baseline_ && fp_residual > last_kkt_residual_) {
        restart_condition = 2;
      } else if (interval_iterations_ >= 0.2 * static_cast<double>(iterations_)) {
        restart_condition = 3;
      }
      if (interval_iterations_ > 10000) restart_condition = 4;
      if (restart_condition && iterations_ > 40) apply_restart(restart_condition);

      if (iterations_ % options_.evaluate_frequency == 0) {
        metrics_ = evaluate();
        last_kkt_residual_ = metrics_.residual();
        if (metrics_.residual() <= options_.tolerance) {
          status_ = "CONVERGED";
          break;
        }
      }
      if (iterations_ == 1 || iterations_ % options_.print_frequency == 0) {
        std::cout << "PDSDP_CUDA_ITER iter=" << iterations_ << " p=" << metrics_.primal
                  << " d=" << metrics_.dual << " g=" << metrics_.gap
                  << " step=" << step_size_ << " weight=" << primal_weight_
                  << " time_sec=" << elapsed() << '\n';
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    termination_time_ = elapsed();
    metrics_ = evaluate();
    CUDA_CHECK(cudaDeviceSynchronize());
    final_time_ = elapsed();
    if (status_ != "CONVERGED" && termination_time_ <= options_.time_limit &&
        metrics_.residual() <= options_.tolerance) {
      status_ = "CONVERGED";
    }
  }

  void save_checkpoint(const fs::path& path) {
    std::vector<double> x(p_.n), y(p_.m);
    CUDA_CHECK(cudaMemcpy(x.data(), x_.data(), x.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(), y_.data(), y.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < p_.n; ++i) x[i] = x[i] / p_.scaling.col_norm[i] * p_.scaling.b_scale;
    for (int i = 0; i < p_.m; ++i) y[i] = y[i] / p_.scaling.row_norm[i] * p_.scaling.c_scale;

    std::vector<int> original_ptr(p_.original_cone_dims.size() + 1, 0);
    for (std::size_t cone = 0; cone < p_.original_cone_dims.size(); ++cone) {
      const std::int64_t dim = p_.original_cone_dims[cone];
      original_ptr[cone + 1] = original_ptr[cone] + static_cast<int>(dim * (dim + 1) / 2);
    }
    std::vector<double> original_x(x.size());
    for (std::size_t old_cone = 0; old_cone < p_.original_cone_dims.size(); ++old_cone) {
      const int new_cone = p_.old_to_new[old_cone];
      std::copy(x.begin() + p_.var_ptr[new_cone], x.begin() + p_.var_ptr[new_cone + 1],
                original_x.begin() + original_ptr[old_cone]);
    }

    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    if (fs::exists(path)) fail("Refusing to overwrite checkpoint: " + path.string());
    fs::path temporary_path = path;
    temporary_path += ".tmp";
    if (fs::exists(temporary_path)) {
      fail("Refusing to overwrite temporary checkpoint: " + temporary_path.string());
    }
    std::ofstream output(temporary_path, std::ios::binary);
    if (!output) fail("Cannot create checkpoint: " + temporary_path.string());
    const char magic[8] = {'P', 'D', 'S', 'D', 'P', 'C', '1', '\0'};
    const std::uint32_t version = 1, endian = 0x01020304;
    const std::int32_t cones = static_cast<std::int32_t>(p_.original_cone_dims.size());
    const std::int32_t m = p_.m;
    const std::int64_t primal_length = static_cast<std::int64_t>(original_x.size());
    output.write(magic, sizeof(magic));
    output.write(reinterpret_cast<const char*>(&version), sizeof(version));
    output.write(reinterpret_cast<const char*>(&endian), sizeof(endian));
    output.write(reinterpret_cast<const char*>(&cones), sizeof(cones));
    output.write(reinterpret_cast<const char*>(&m), sizeof(m));
    output.write(reinterpret_cast<const char*>(&primal_length), sizeof(primal_length));
    output.write(reinterpret_cast<const char*>(p_.original_cone_dims.data()),
                 p_.original_cone_dims.size() * sizeof(int));
    output.write(reinterpret_cast<const char*>(original_x.data()), original_x.size() * sizeof(double));
    output.write(reinterpret_cast<const char*>(y.data()), y.size() * sizeof(double));
    output.close();
    if (!output) fail("Checkpoint write failed: " + temporary_path.string());
    fs::rename(temporary_path, path);
  }

  const std::string& status() const { return status_; }
  std::int64_t iterations() const { return iterations_; }
  double termination_time() const { return termination_time_; }
  double final_time() const { return final_time_; }
  const Metrics& metrics() const { return metrics_; }

 private:
  template <typename T>
  void copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    CUDA_CHECK(cudaMemcpy(destination.data(), source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  void device_copy(double* destination, const double* source, int size) {
    CUBLAS_CHECK(cublasDcopy(blas_, size, source, 1, destination, 1));
  }
  double dot(const double* lhs, const double* rhs, int size) {
    double value = 0.0;
    CUBLAS_CHECK(cublasDdot(blas_, size, lhs, 1, rhs, 1, &value));
    return value;
  }
  double norm(const double* values, int size) {
    double value = 0.0;
    CUBLAS_CHECK(cublasDnrm2(blas_, size, values, 1, &value));
    return value;
  }
  int blocks(int size) const { return (size + kThreads - 1) / kThreads; }
  double elapsed() const { return seconds_since(solver_start_); }

  void zero_state() {
    x_.zero(); last_x_.zero(); temp_x_.zero(); avg_x_.zero(); restart_x_.zero();
    delta_x_.zero(); gradient_.zero(); projected_slack_.zero();
    y_.zero(); last_y_.zero(); temp_y_.zero(); avg_y_.zero(); restart_y_.zero();
    delta_y_.zero(); ax_.zero(); last_ax_.zero(); avg_ax_.zero(); delta_ax_.zero();
    extrapolated_ax_.zero(); residual_buffer_.zero();
  }

  double estimate_operator_norm() {
    std::mt19937 generator(1);
    std::normal_distribution<double> normal;
    std::vector<double> initial(p_.m);
    for (double& value : initial) value = normal(generator) + 1e-8;
    copy_to_device(power_z_, initial);
    double lambda = 1.0;
    for (int iteration = 0; iteration < 5000; ++iteration) {
      device_copy(power_q_.data(), power_z_.data(), p_.m);
      const double q_norm = norm(power_q_.data(), p_.m);
      const double inv = 1.0 / q_norm;
      CUBLAS_CHECK(cublasDscal(blas_, p_.m, &inv, power_q_.data(), 1));
      at_->multiply(sparse_, power_q_.data(), power_atq_.data());
      a_->multiply(sparse_, power_atq_.data(), power_z_.data());
      lambda = dot(power_q_.data(), power_z_.data(), p_.m);
      device_copy(residual_buffer_.data(), power_z_.data(), p_.m);
      add_kernel<<<blocks(p_.m), kThreads>>>(residual_buffer_.data(), power_q_.data(), -lambda, p_.m);
      if (norm(residual_buffer_.data(), p_.m) < 1e-4) return std::sqrt(lambda);
    }
    return std::sqrt(lambda);
  }

  void iteration() {
    device_copy(last_x_.data(), x_.data(), p_.n);
    device_copy(last_y_.data(), y_.data(), p_.m);
    device_copy(last_ax_.data(), ax_.data(), p_.m);
    at_->multiply(sparse_, last_y_.data(), gradient_.data());
    add_kernel<<<blocks(p_.n), kThreads>>>(gradient_.data(), d_c_.data(), 1.0, p_.n);

    double trial_step = step_size_;
    double raw_interaction = 0.0;
    double norm_primal_sq = 0.0;
    double norm_dual_sq = 0.0;
    for (int trace = 0; trace < 10; ++trace) {
      device_copy(temp_x_.data(), last_x_.data(), p_.n);
      device_copy(temp_y_.data(), last_y_.data(), p_.m);
      add_kernel<<<blocks(p_.n), kThreads>>>(temp_x_.data(), gradient_.data(),
                                             -trial_step / primal_weight_, p_.n);
      projection_->project(temp_x_.data());
      a_->multiply(sparse_, temp_x_.data(), ax_.data());
      extrapolate_kernel<<<blocks(p_.m), kThreads>>>(
          extrapolated_ax_.data(), ax_.data(), last_ax_.data(), p_.m);
      dual_step_kernel<<<blocks(p_.m), kThreads>>>(
          temp_y_.data(), extrapolated_ax_.data(), d_b_.data(),
          trial_step * primal_weight_, p_.m);
      difference_kernel<<<blocks(p_.n), kThreads>>>(
          delta_x_.data(), temp_x_.data(), last_x_.data(), p_.n);
      difference_kernel<<<blocks(p_.m), kThreads>>>(
          delta_y_.data(), temp_y_.data(), last_y_.data(), p_.m);
      difference_kernel<<<blocks(p_.m), kThreads>>>(
          delta_ax_.data(), ax_.data(), last_ax_.data(), p_.m);

      raw_interaction = dot(delta_y_.data(), delta_ax_.data(), p_.m);
      const double interaction = std::abs(raw_interaction);
      norm_primal_sq = dot(delta_x_.data(), delta_x_.data(), p_.n);
      norm_dual_sq = dot(delta_y_.data(), delta_y_.data(), p_.m);
      const double movement = 0.5 *
          (primal_weight_ * norm_primal_sq + norm_dual_sq / primal_weight_);
      double limit = std::numeric_limits<double>::infinity();
      if (interaction > 0.0) {
        limit = movement / interaction;
        if (movement == 0.0 || !std::isfinite(movement)) {
          numerical_error_ = true;
          return;
        }
      }
      const double interval = static_cast<double>(interval_iterations_ + 1);
      const double first = (1.0 - 1.0 / std::pow(interval, 0.3)) * limit;
      const double second = (1.0 + 1.0 / std::pow(interval, 0.9)) * trial_step;
      trial_step = std::min(first, second);
      if (trial_step <= limit) break;
    }
    step_size_ = trial_step;

    const double alpha2 = 1.0 / static_cast<double>(interval_iterations_ + 2);
    const double alpha1 = 1.0 - alpha2;
    halpern_kernel<<<blocks(p_.n), kThreads>>>(
        x_.data(), temp_x_.data(), last_x_.data(), avg_x_.data(), alpha1, alpha2, p_.n);
    halpern_kernel<<<blocks(p_.m), kThreads>>>(
        y_.data(), temp_y_.data(), last_y_.data(), avg_y_.data(), alpha1, alpha2, p_.m);
    halpern_kernel<<<blocks(p_.m), kThreads>>>(
        ax_.data(), ax_.data(), last_ax_.data(), avg_ax_.data(), alpha1, alpha2, p_.m);
    fixed_point_residual_ = norm_primal_sq * primal_weight_ / step_size_ +
                            norm_dual_sq / (primal_weight_ * step_size_) +
                            2.0 * raw_interaction;
    if (!std::isfinite(fixed_point_residual_) || !std::isfinite(step_size_) || step_size_ <= 0.0) {
      numerical_error_ = true;
    }
  }

  void apply_restart(int condition) {
    device_copy(avg_x_.data(), x_.data(), p_.n);
    device_copy(avg_y_.data(), y_.data(), p_.m);
    device_copy(avg_ax_.data(), ax_.data(), p_.m);
    difference_kernel<<<blocks(p_.n), kThreads>>>(
        delta_x_.data(), x_.data(), restart_x_.data(), p_.n);
    difference_kernel<<<blocks(p_.m), kThreads>>>(
        delta_y_.data(), y_.data(), restart_y_.data(), p_.m);
    const double primal_distance = norm(delta_x_.data(), p_.n);
    const double dual_distance = norm(delta_y_.data(), p_.m);
    if (primal_distance > 0.0 && dual_distance > 0.0) {
      const double estimate = dual_distance / primal_distance;
      primal_weight_ = std::exp(0.1 * std::log(estimate) + 0.9 * std::log(primal_weight_));
    }
    if (primal_weight_ < 1e-4 || primal_weight_ > 1e4 || !std::isfinite(primal_weight_)) {
      const double b_norm = vector_norm2(p_.b), c_norm = vector_norm2(p_.c);
      primal_weight_ = (b_norm > 0.0 && c_norm > 0.0) ? c_norm / b_norm : 1.0;
    }
    device_copy(restart_x_.data(), x_.data(), p_.n);
    device_copy(restart_y_.data(), y_.data(), p_.m);
    interval_iterations_ = 0;
    std::cout << "PDSDP_CUDA_RESTART iter=" << iterations_ << " condition=" << condition
              << " fp=" << fixed_point_residual_ << " weight=" << primal_weight_ << '\n';
  }

  Metrics evaluate() {
    Metrics metrics;
    at_->multiply(sparse_, y_.data(), gradient_.data());
    add_kernel<<<blocks(p_.n), kThreads>>>(gradient_.data(), d_c_.data(), 1.0, p_.n);
    scale_inplace_kernel<<<blocks(p_.n), kThreads>>>(
        gradient_.data(), d_col_norm_.data(), p_.scaling.c_scale, p_.n);
    device_copy(projected_slack_.data(), gradient_.data(), p_.n);
    projection_->project(projected_slack_.data());
    projection_->verify_last_projection();
    difference_kernel<<<blocks(p_.n), kThreads>>>(
        projected_slack_.data(), gradient_.data(), projected_slack_.data(), p_.n);
    metrics.dual = norm(projected_slack_.data(), p_.n) / (1.0 + p_.scaling.original_c_inf);

    original_residual_kernel<<<blocks(p_.m), kThreads>>>(
        residual_buffer_.data(), ax_.data(), d_b_.data(), d_row_norm_.data(),
        p_.scaling.b_scale, p_.m);
    metrics.primal = norm(residual_buffer_.data(), p_.m) / (1.0 + p_.scaling.original_b_inf);
    const double scale = p_.scaling.b_scale * p_.scaling.c_scale;
    metrics.primal_obj = dot(x_.data(), d_c_.data(), p_.n) * scale;
    metrics.dual_obj = dot(y_.data(), d_b_.data(), p_.m) * scale;
    metrics.gap = std::abs(metrics.primal_obj + metrics.dual_obj) /
                  (1.0 + std::abs(metrics.primal_obj) + std::abs(metrics.dual_obj));
    if (!std::isfinite(metrics.primal) || !std::isfinite(metrics.dual) ||
        !std::isfinite(metrics.gap) || !std::isfinite(metrics.primal_obj) ||
        !std::isfinite(metrics.dual_obj)) {
      fail("Non-finite convergence metric");
    }
    return metrics;
  }

  Problem& p_;
  Options options_;
  cusparseHandle_t sparse_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  cusolverDnHandle_t dense_solver_ = nullptr;
  std::unique_ptr<CsrDevice> a_, at_;
  std::unique_ptr<ProjectionEngine> projection_;
  DeviceBuffer<double> d_b_, d_c_, d_row_norm_, d_col_norm_;
  DeviceBuffer<double> x_, last_x_, temp_x_, avg_x_, restart_x_, delta_x_;
  DeviceBuffer<double> gradient_, projected_slack_;
  DeviceBuffer<double> y_, last_y_, temp_y_, avg_y_, restart_y_, delta_y_;
  DeviceBuffer<double> ax_, last_ax_, avg_ax_, delta_ax_, extrapolated_ax_, residual_buffer_;
  DeviceBuffer<double> power_q_, power_z_, power_atq_;
  Clock::time_point solver_start_;
  std::string status_ = "NOT_STARTED";
  std::int64_t iterations_ = 0;
  int interval_iterations_ = 0;
  double step_size_ = 0.001;
  double primal_weight_ = 0.001;
  double restart_baseline_ = 100.0;
  double last_kkt_residual_ = std::numeric_limits<double>::infinity();
  double fixed_point_residual_ = std::numeric_limits<double>::infinity();
  bool numerical_error_ = false;
  double termination_time_ = 0.0;
  double final_time_ = 0.0;
  Metrics metrics_;
};

}  // namespace pdsdp

int main(int argc, char** argv) {
  using namespace pdsdp;
  const auto process_start = Clock::now();
  try {
    if (argc < 3 || argc > 6) {
      std::cerr << "usage: pdsdp-cuda PROBLEM.dat-s OUTPUT.pdsdpbin "
                   "[TIME_SEC] [MAX_ITER] [TOL]\n";
      return 2;
    }
    Options options;
    if (argc >= 4) options.time_limit = std::stod(argv[3]);
    if (argc >= 5) options.max_iterations = std::stoll(argv[4]);
    if (argc >= 6) options.tolerance = std::stod(argv[5]);
    if (!std::isfinite(options.time_limit) || options.time_limit <= 0.0) {
      fail("TIME_SEC must be finite and positive");
    }
    if (options.max_iterations <= 0) fail("MAX_ITER must be positive");
    if (!std::isfinite(options.tolerance) || options.tolerance <= 0.0) {
      fail("TOL must be finite and positive");
    }
    const fs::path problem_path = fs::absolute(argv[1]).lexically_normal();
    const fs::path output_path = fs::absolute(argv[2]).lexically_normal();
    if (problem_path == output_path) fail("Problem and output paths must differ");
    if (fs::exists(output_path)) {
      fail("Refusing to overwrite checkpoint: " + output_path.string());
    }
    std::cout << std::setprecision(17);
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    std::cout << "PDSDP_CUDA_BUILD version=" << kVersion
              << " driver=" << driver_version << " runtime=" << runtime_version << '\n';
    std::cout << "PDSDP_CUDA_DEVICE name=" << properties.name
              << " cc=" << properties.major << '.' << properties.minor
              << " memory_bytes=" << properties.totalGlobalMem << '\n';

    Problem problem = load_sdpa(problem_path);
    scale_problem(problem);
    Solver solver(problem, options);
    solver.solve();
    solver.save_checkpoint(output_path);
    const auto& metrics = solver.metrics();
    std::cout << "PDSDP_CUDA_RESULT status=" << solver.status()
              << " iterations=" << solver.iterations()
              << " solver_time_sec=" << solver.termination_time()
              << " final_time_sec=" << solver.final_time()
              << " process_time_sec=" << seconds_since(process_start)
              << " primal=" << metrics.primal << " dual=" << metrics.dual
              << " gap=" << metrics.gap << " primal_obj=" << metrics.primal_obj
              << " dual_obj=" << metrics.dual_obj << " output=" << output_path
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PDSDP_CUDA_ERROR " << error.what() << '\n';
    return 1;
  }
}
