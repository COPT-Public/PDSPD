# PDSDP

GPU-oriented primal-dual solvers for semidefinite programming, with a
Julia/CUDA.jl implementation and a native C++17/CUDA implementation.

PDSDP combines Halpern restarts and adaptive rAPDHG steps with sparse
matrix-vector products, problem scaling, and projections onto positive
semidefinite cones. The supplied command-line runners read SDPA-sparse
(`.dat-s`) problems containing PSD blocks and diagonal/nonnegative blocks,
and export solution checkpoints for subsequent evaluation.

The code is hosted in [COPT-Public/PDSPD](https://github.com/COPT-Public/PDSPD).
The solver and source directories retain the name **PDSDP** used in the
original implementation.

## Implementations

| Directory | Implementation | Main entry point | Solution output |
| --- | --- | --- | --- |
| [`julia-original/`](julia-original/) | Julia with CUDA.jl; original research implementation | `mittelmann_runner.jl` | MAT checkpoint |
| [`cpp-cuda/`](cpp-cuda/) | Native C++17/CUDA; cuSPARSE, cuSOLVER, and cuBLAS | `build/pdsdp-cuda` after compilation | `.pdsdpbin` checkpoint |

Both documented solver entry points require an NVIDIA GPU. The native
executable uses one GPU per solve. Its multi-GPU campaign scripts schedule
different problem instances across GPUs.

## Quick start with C++ and CUDA

The supplied Makefile targets a Linux environment with GNU Make, a
CUDA-compatible C++17 host compiler, OpenMP, and the CUDA Toolkit. The toolkit
must provide the CUDA Runtime, cuBLAS, cuSPARSE, and cuSOLVER.

From the repository root:

```bash
git clone https://github.com/COPT-Public/PDSPD.git
cd PDSPD

# H100 target; adjust CUDA_HOME for your toolkit installation.
make -C cpp-cuda CUDA_HOME=/usr/local/cuda CUDA_ARCH=90

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 \
  ./cpp-cuda/build/pdsdp-cuda \
  /path/to/problem.dat-s /path/to/new-solution.pdsdpbin \
  120 1000000000 1e-4
```

`CUDA_ARCH` selects the compilation target; for example, use `89` for an RTX
4090. This setting does not constitute a validation result for that GPU.
The output checkpoint must not already exist. The three optional arguments
after the output path are the time limit in seconds, maximum iterations, and
termination tolerance. Native defaults are 3600 seconds, 1,000,000,000
iterations, and `1e-4`.

Supply your own SDPA-sparse input file; benchmark datasets are not included.
See the [native implementation README](cpp-cuda/README.md) for checkpoint
format, stopping criteria, and implementation details.

## Quick start with Julia

The archived package environment records Julia **1.12.5** and a CUDA runtime
preference of **12.8**. An NVIDIA driver compatible with the selected runtime
is required. From the repository root:

```bash
cd julia-original
julia --project=. -e 'using Pkg; Pkg.instantiate()'

CUDA_VISIBLE_DEVICES=0 PDSDP_BLAS_THREADS=32 \
  julia --project=. mittelmann_runner.jl \
  /path/to/problem.dat-s /path/to/solution.mat \
  120 1000000000 1e-4
```

The Julia runner has the same argument order as the native runner; its default
iteration limit is **20,000** when omitted. The example sets it explicitly.
Use `mittelmann_runner.jl` for the documented SDPA interface. The older
`main.jl` and auxiliary research scripts contain experiment-specific paths
and device settings. See the [Julia implementation README](julia-original/README.md).

## Inspecting results

The native solver prints a `PDSDP_CUDA_RESULT` line; the Julia runner prints
`PDSDP_RESULT`. Check the reported status, primal residual, dual residual,
and primal-dual gap. A process exit code of zero alone does not establish
convergence: the native executable can successfully write a checkpoint after
reaching a time or iteration limit.

The archived independent metric evaluator can be called directly for a native
checkpoint after installing NumPy and SciPy:

```bash
python cpp-cuda/independent-validator/dimacs.py \
  --problem /path/to/problem.dat-s \
  --solution-kind pdsdp-cuda-bin \
  --checkpoint /path/to/new-solution.pdsdpbin \
  --output /path/to/validation.json
```

It reports DIMACS metrics and checkpoint-validity information. Inspect those
values and flags against your required tolerances; its exit code is not a
pass/fail acceptance test. MAT checkpoint loading may additionally require
`h5py`, and optional GPU eigenvalue evaluation requires PyTorch with CUDA.

The separate `validate_mittelmann.py` and `validate_cuda_checkpoint.py`
wrappers import an external `sdp_bench/src/gpu_sdp_bench` package that is
**not included** in this repository. The standalone evaluator above is the
archived `dimacs.py` file; it does not supply the rest of that package.

## Archived experiments and reproducibility

This repository includes the supplied source and small-result snapshot from
the September 14, 2026 Mittelmann-75 experiments:

- [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md): experimental protocol,
  implementation comparison, reported results, and limitations (in Chinese).
- [`julia-original/benchmark-summary/`](julia-original/benchmark-summary/):
  Julia campaign manifests, summaries, and logs.
- [`cpp-cuda/benchmark-summary/`](cpp-cuda/benchmark-summary/): native campaign
  manifests, summaries, and logs.
- [`ARCHIVE_README.md`](ARCHIVE_README.md): the original archive overview.

These files document the original experiments; publishing this snapshot is
not a new reproduction of their results. The reported three-metric
feasibility test and the additional transport-strict test are distinct and
should not be combined when comparing solved counts.

The benchmark datasets, multi-gigabyte solution checkpoints, detailed
per-instance outputs, external `sdp_bench` package, and tested native binary
are not bundled. Historical campaign scripts and reports retain their
original machine-specific paths. Adapt those paths and provide the external
dependencies before replaying a campaign. Some auxiliary Julia comparison
and tuning scripts also require additional packages beyond the main runner.

## License and support

See the repository's [MIT license](LICENSE). Third-party dependencies retain
their own licenses.

Report bugs or request improvements through
[GitHub Issues](https://github.com/COPT-Public/PDSPD/issues). Include the
implementation, commit, GPU and driver, CUDA/Julia versions, invocation,
and a minimal reproducible input when possible.
