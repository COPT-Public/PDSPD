#!/usr/bin/env python3
"""Independent DIMACS validation for SDPA sparse problems and low-rank solutions.

The validator always reparses the original problem and reconstructs the dual
slack in original coordinates.  Solver-reported residuals are metadata only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


GPU_LANCZOS_NNZ_THRESHOLD = int(
    os.environ.get("GPU_SDP_BENCH_GPU_LANCZOS_NNZ_THRESHOLD", "5000000")
)
GPU_LANCZOS_DIM_THRESHOLD = int(
    os.environ.get("GPU_SDP_BENCH_GPU_LANCZOS_DIM_THRESHOLD", "2000")
)
GPU_LANCZOS_MAXITER = int(
    os.environ.get("GPU_SDP_BENCH_GPU_LANCZOS_MAXITER", "2048")
)
GPU_LANCZOS_CHECK_INTERVAL = 64
GPU_LANCZOS_SEED = 20260826
EIGEN_TOL = 1e-7
EIGEN_MAXITER = 5000
CPU_DENSE_EXACT_DIM_MIN = int(
    os.environ.get("GPU_SDP_BENCH_CPU_DENSE_EXACT_DIM_MIN", "1024")
)
CPU_DENSE_EXACT_DIM_MAX = int(
    os.environ.get("GPU_SDP_BENCH_CPU_DENSE_EXACT_DIM_MAX", "2048")
)
CPU_DENSE_EXACT_DENSITY_MIN = float(
    os.environ.get("GPU_SDP_BENCH_CPU_DENSE_EXACT_DENSITY_MIN", "0.25")
)
PDSDP_PRIMAL_TRANSPORT_DIMACS_TOL = float(
    os.environ.get("GPU_SDP_BENCH_PDSDP_PRIMAL_TRANSPORT_DIMACS_TOL", "1e-10")
)


@dataclass
class SdpaProblem:
    m: int
    block_dims: list[int]
    raw_block_dims: list[int]
    psd_block_indices: list[int]
    lp_dim: int
    lp_block_offsets: dict[int, int]
    b: np.ndarray
    mat: np.ndarray
    blk: np.ndarray
    row: np.ndarray
    col: np.ndarray
    val: np.ndarray


def _open_text(path: Path):
    path = _resolve_migrated_benchmark_path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="ascii", errors="strict")
    return path.open("rt", encoding="ascii", errors="strict")


def _numbers(line: str) -> list[str]:
    stripped = line.lstrip()
    if not stripped or stripped[0] in '"*#':
        return []
    for char in ",{}()=:":
        line = line.replace(char, " ")
    return line.split()


def read_sdpa(path: Path) -> SdpaProblem:
    """Read sparse SDPA with PSD blocks and diagonal nonnegative LP blocks."""
    with _open_text(path) as stream:
        header: list[str] = []
        while len(header) < 2:
            header.extend(_numbers(stream.readline()))
        m, nblocks = int(header[0]), int(header[1])

        block_tokens = header[2:]
        while len(block_tokens) < nblocks:
            block_tokens.extend(_numbers(stream.readline()))
        raw_block_dims = [int(x) for x in block_tokens[:nblocks]]
        if any(n == 0 for n in raw_block_dims):
            raise NotImplementedError("Zero-dimensional SDPA blocks are unsupported")
        block_dims = [n for n in raw_block_dims if n > 0]
        psd_block_indices = [i for i, n in enumerate(raw_block_dims) if n > 0]
        lp_block_offsets = {}
        lp_dim = 0
        for i, n in enumerate(raw_block_dims):
            if n < 0:
                lp_block_offsets[i] = lp_dim
                lp_dim += -n

        b_tokens = block_tokens[nblocks:]
        while len(b_tokens) < m:
            b_tokens.extend(_numbers(stream.readline()))
        b = np.asarray(b_tokens[:m], dtype=np.float64)
        if len(b_tokens) != m:
            raise ValueError("RHS and matrix entries must be separated at a line boundary")

        # Sparse SDPA uses five numeric fields per matrix entry. Reading the
        # tail in one pass avoids millions of tiny Python objects on G40_mb.
        matrix_text = stream.read()
    flat = np.fromstring(matrix_text, dtype=np.float64, sep=" ")
    if flat.size % 5:
        raise ValueError(f"Matrix token count {flat.size} is not divisible by 5")
    entries = flat.reshape(-1, 5)
    return SdpaProblem(
        m=m,
        block_dims=block_dims,
        raw_block_dims=raw_block_dims,
        psd_block_indices=psd_block_indices,
        lp_dim=lp_dim,
        lp_block_offsets=lp_block_offsets,
        b=b,
        mat=entries[:, 0].astype(np.int32),
        blk=entries[:, 1].astype(np.int32) - 1,
        row=entries[:, 2].astype(np.int32) - 1,
        col=entries[:, 3].astype(np.int32) - 1,
        val=entries[:, 4],
    )


def load_cardal(path: Path):
    with path.open("rb") as stream:
        if stream.read(8) != b"CARDALS1":
            raise ValueError("Not a CARDAL solution checkpoint")
        version, endian = struct.unpack("<II", stream.read(8))
        if version != 1 or endian != 0x01020304:
            raise ValueError(f"Unsupported CARDAL checkpoint version/endian: {version}/{endian:#x}")
        ncones, m, lp_dim = struct.unpack("<iii", stream.read(12))
        (factor_length,) = struct.unpack("<q", stream.read(8))
        dims = np.fromfile(stream, dtype="<i4", count=ncones)
        ranks = np.fromfile(stream, dtype="<i4", count=ncones)
        factor = np.fromfile(stream, dtype="<f8", count=factor_length)
        dual = np.fromfile(stream, dtype="<f8", count=m)
        if dual.size != m:
            raise ValueError("Truncated CARDAL checkpoint")

    factors = []
    offset = 0
    for n, rank in zip(dims, ranks):
        count = int(n) * int(rank)
        factors.append(factor[offset : offset + count].reshape((int(n), int(rank)), order="F"))
        offset += count
    if offset + lp_dim != factor_length:
        raise ValueError("Unexpected CARDAL factor length")
    lp_primal = factor[offset : offset + lp_dim].copy()
    return factors, -dual, lp_primal, {
        "stored_dual_convention": "lambda=-y",
        "ranks": ranks.tolist(),
    }


def _resolve_migrated_benchmark_path(path: Path) -> Path:
    """Relocate a missing path recorded under an older benchmark checkout."""
    path = Path(path)
    if path.exists():
        return path

    try:
        benchmark_index = path.parts.index("sdp_bench")
    except ValueError:
        return path

    benchmark_root = Path(
        os.environ.get(
            "GPU_SDP_BENCH_ROOT",
            Path(__file__).resolve().parents[2],
        )
    ).resolve()
    candidate = benchmark_root.joinpath(*path.parts[benchmark_index + 1 :])
    return candidate if candidate.exists() else path


def load_cardal_text(path: Path, block_dims: list[int], m: int, lp_dim: int = 0):
    requested_path = Path(path)
    path = _resolve_migrated_benchmark_path(requested_path)
    suffix = "_primal_factor.txt"
    if not path.name.endswith(suffix):
        raise ValueError(f"CARDAL text factor file must end with {suffix}")
    instance = path.name[: -len(suffix)]
    ranks_path = path.with_name(f"{instance}_rank_list.txt")
    dual_path = path.with_name(f"{instance}_dual_solution.txt")
    lp_path = path.with_name(f"{instance}_lp_primal.txt")

    ranks = np.atleast_1d(np.loadtxt(ranks_path, dtype=np.int64))
    factor = np.atleast_1d(np.loadtxt(path, dtype=np.float64))
    dual = np.atleast_1d(np.loadtxt(dual_path, dtype=np.float64))
    if ranks.size != len(block_dims):
        raise ValueError("CARDAL rank count does not match the SDPA block count")
    if dual.size != m:
        raise ValueError("CARDAL dual length does not match the SDPA constraint count")
    if lp_dim:
        lp_primal = np.atleast_1d(np.loadtxt(lp_path, dtype=np.float64))
        if lp_primal.size != lp_dim:
            raise ValueError("CARDAL LP primal length does not match the SDPA LP dimension")
    else:
        lp_primal = np.empty(0, dtype=np.float64)

    factors = []
    offset = 0
    for n, rank in zip(block_dims, ranks):
        count = int(n) * int(rank)
        factors.append(factor[offset : offset + count].reshape((int(n), int(rank)), order="F"))
        offset += count
    if offset != factor.size:
        raise ValueError(
            f"CARDAL factor length mismatch: expected {offset}, found {factor.size}"
        )
    metadata = {
        "stored_dual_convention": "lambda=-y",
        "ranks": ranks.tolist(),
        "format": "CARDAL CLI text solution",
    }
    if path != requested_path:
        metadata["relocated_from"] = str(requested_path)
        metadata["resolved_path"] = str(path)
    return factors, -dual, lp_primal, metadata


def _hdf5_vector(mat, name: str) -> np.ndarray:
    """Read either a numeric MATLAB vector or a one-cell referenced vector."""
    dataset = mat[name]
    if bool(np.asarray(dataset.attrs.get("MATLAB_empty", 0)).reshape(-1)[0]):
        return np.empty(0, dtype=np.float64)
    raw = np.asarray(dataset)
    if raw.dtype.kind == "O":
        refs = raw.reshape(-1)
        if refs.size != 1:
            raise ValueError(f"Expected one MATLAB cell in {name}, found {refs.size}")
        dataset = mat[refs[0]]
        if bool(np.asarray(dataset.attrs.get("MATLAB_empty", 0)).reshape(-1)[0]):
            return np.empty(0, dtype=np.float64)
        raw = np.asarray(dataset)
    return np.asarray(raw, dtype=np.float64).reshape(-1)


def load_culorads(path: Path):
    import h5py

    with h5py.File(path, "r") as mat:
        cells = np.asarray(mat["primal_sdp"]).reshape(-1)
        factors = [np.asarray(mat[ref], dtype=np.float64).T.copy() for ref in cells]
        dual = _hdf5_vector(mat, "dual")
        lp_primal = (
            _hdf5_vector(mat, "primal_lp")
            if "primal_lp" in mat
            else np.empty(0, dtype=np.float64)
        )
    return factors, dual, lp_primal, {
        "stored_dual_convention": "y",
        "ranks": [u.shape[1] for u in factors],
        "format": "cuLoRADS MATLAB v7.3 output",
    }


def _unpack_svec_block(values: np.ndarray, dim: int) -> np.ndarray:
    """Convert PDSDP's orthonormal lower-triangular svec to a dense matrix."""
    expected = dim * (dim + 1) // 2
    if values.size != expected:
        raise ValueError(
            f"PDSDP svec block length mismatch: expected {expected}, found {values.size}"
        )
    matrix = np.zeros((dim, dim), dtype=np.float64)
    cursor = 0
    for row in range(dim):
        for col in range(row + 1):
            value = float(values[cursor])
            if row == col:
                matrix[row, col] = value
            else:
                value /= np.sqrt(2.0)
                matrix[row, col] = value
                matrix[col, row] = value
            cursor += 1
    return matrix


def _factor_pdsdp_full_primal(
    primal: np.ndarray,
    pdsdp_dual: np.ndarray,
    problem: SdpaProblem,
    *,
    checkpoint_format: str,
):
    """Convert a PDSDP original-coordinate full-primal checkpoint."""
    if pdsdp_dual.size != problem.m:
        raise ValueError(
            "PDSDP dual length mismatch: "
            f"expected {problem.m}, found {pdsdp_dual.size}"
        )

    factors = []
    ranks = []
    minimum_primal_eigenvalue = np.inf
    cursor = 0
    for dim in problem.block_dims:
        packed_length = dim * (dim + 1) // 2
        block = _unpack_svec_block(
            primal[cursor : cursor + packed_length], dim
        )
        eigenvalues, eigenvectors = np.linalg.eigh((block + block.T) * 0.5)
        minimum_primal_eigenvalue = min(
            minimum_primal_eigenvalue, float(eigenvalues[0])
        )
        positive = eigenvalues > 0.0
        factor = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
        factors.append(factor)
        ranks.append(int(np.count_nonzero(positive)))
        cursor += packed_length

    lp_primal = primal[cursor : cursor + problem.lp_dim].copy()
    cursor += problem.lp_dim
    if cursor != primal.size:
        raise ValueError(
            f"PDSDP primal length mismatch: consumed {cursor}, found {primal.size}"
        )

    b_norm_inf = float(np.max(np.abs(problem.b), initial=0.0))
    raw_primal_psd_violation = (
        0.0
        if minimum_primal_eigenvalue == np.inf
        else max(0.0, -minimum_primal_eigenvalue)
    )
    raw_primal_psd_dimacs_error = raw_primal_psd_violation / (1.0 + b_norm_inf)
    return factors, -pdsdp_dual, lp_primal, {
        "stored_dual_convention": "lambda=-y",
        "ranks": ranks,
        "format": checkpoint_format,
        "full_matrix_primal_min_eigenvalue_before_clipping": (
            None
            if minimum_primal_eigenvalue == np.inf
            else minimum_primal_eigenvalue
        ),
        "primal_factorization_note": (
            "full matrices are eigendecomposed; negative roundoff eigenvalues "
            "are recorded above and clipped only for UU' factor transport"
        ),
        "raw_primal_psd_dimacs_error_before_clipping": raw_primal_psd_dimacs_error,
        "primal_transport_tolerance": PDSDP_PRIMAL_TRANSPORT_DIMACS_TOL,
        "primal_transport_valid": (
            raw_primal_psd_dimacs_error <= PDSDP_PRIMAL_TRANSPORT_DIMACS_TOL
        ),
    }


def load_pdsdp(path: Path, problem: SdpaProblem):
    """Load the original-coordinate full-matrix checkpoint emitted by PDSDP.

    The Mittelmann adapter stores genuine PSD blocks first and represents every
    coordinate of an SDPA diagonal/nonnegative block as a final 1x1 cone.  The
    latter coordinates are returned as LP variables so the independent
    evaluator applies the original problem's cone convention.
    """
    import h5py

    with h5py.File(path, "r") as mat:
        primal = _hdf5_vector(
            mat, "primal_solution_recovered_original_order"
        )
        pdsdp_dual = _hdf5_vector(
            mat, "dual_solution_recovered_original_order"
        )

    return _factor_pdsdp_full_primal(
        primal,
        pdsdp_dual,
        problem,
        checkpoint_format="PDSDP MATLAB v7.3 full-matrix output",
    )


def load_pdsdp_cuda(path: Path, problem: SdpaProblem):
    """Load the checked native binary emitted by the C++/CUDA PDSDP solver."""
    payload = Path(path).read_bytes()
    header_format = "<8sIIiiq"
    header_size = struct.calcsize(header_format)
    if len(payload) < header_size:
        raise ValueError("PDSDP CUDA checkpoint is truncated before its header")
    magic, version, endian, cone_count, m, primal_length = struct.unpack_from(
        header_format, payload, 0
    )
    if magic != b"PDSDPC1\x00":
        raise ValueError(f"Invalid PDSDP CUDA checkpoint magic: {magic!r}")
    if version != 1:
        raise ValueError(f"Unsupported PDSDP CUDA checkpoint version: {version}")
    if endian != 0x01020304:
        raise ValueError(f"Invalid PDSDP CUDA endian marker: 0x{endian:08x}")
    if cone_count < 0 or m < 0 or primal_length < 0:
        raise ValueError("PDSDP CUDA checkpoint contains a negative array length")

    expected_dims = problem.block_dims + [1] * problem.lp_dim
    if cone_count != len(expected_dims):
        raise ValueError(
            "PDSDP CUDA cone-count mismatch: "
            f"expected {len(expected_dims)}, found {cone_count}"
        )
    if m != problem.m:
        raise ValueError(
            f"PDSDP CUDA dual length mismatch: expected {problem.m}, found {m}"
        )
    expected_primal_length = sum(dim * (dim + 1) // 2 for dim in expected_dims)
    if primal_length != expected_primal_length:
        raise ValueError(
            "PDSDP CUDA primal length mismatch: "
            f"expected {expected_primal_length}, found {primal_length}"
        )

    expected_size = header_size + 4 * cone_count + 8 * (primal_length + m)
    if len(payload) != expected_size:
        raise ValueError(
            "PDSDP CUDA checkpoint byte-length mismatch: "
            f"expected {expected_size}, found {len(payload)}"
        )
    dims = np.frombuffer(payload, dtype="<i4", count=cone_count, offset=header_size)
    if dims.tolist() != expected_dims:
        raise ValueError(
            "PDSDP CUDA cone dimensions mismatch: "
            f"expected {expected_dims}, found {dims.tolist()}"
        )
    primal_offset = header_size + 4 * cone_count
    primal = np.frombuffer(
        payload, dtype="<f8", count=primal_length, offset=primal_offset
    ).copy()
    pdsdp_dual = np.frombuffer(
        payload,
        dtype="<f8",
        count=m,
        offset=primal_offset + 8 * primal_length,
    ).copy()
    return _factor_pdsdp_full_primal(
        primal,
        pdsdp_dual,
        problem,
        checkpoint_format="PDSDP C++/CUDA binary v1 full-matrix output",
    )


def symmetric_sparse(n: int, row: np.ndarray, col: np.ndarray, val: np.ndarray) -> csr_matrix:
    off = row != col
    rows = np.concatenate((row, col[off]))
    cols = np.concatenate((col, row[off]))
    vals = np.concatenate((val, val[off]))
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def _torch_cuda_smallest_eigenvalue(
    matrix: csr_matrix,
    *,
    gpu: int,
    block: int,
    tol: float = EIGEN_TOL,
    maxiter: int = GPU_LANCZOS_MAXITER,
) -> tuple[float, dict]:
    """Find a conservative smallest eigenvalue with independent GPU Lanczos.

    The basis is fully reorthogonalized, and convergence is checked using the
    explicit Ritz residual.  Returning ``theta-residual`` makes the DIMACS
    cone violation conservative with respect to the computed Ritz pair.
    PyTorch is imported lazily so CPU-only validation keeps its small
    dependency and startup footprint.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports that CUDA is unavailable")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(
            f"eigen GPU {gpu} is outside the visible range "
            f"[0, {torch.cuda.device_count()})"
        )
    if matrix.shape[0] < 2:
        value = float(matrix[0, 0])
        return value, {
            "backend": "direct-1x1",
            "iterations": 0,
            "ritz_value": value,
            "ritz_residual": 0.0,
            "conservative_value": value,
        }

    device = torch.device(f"cuda:{gpu}")
    n = int(matrix.shape[0])
    # CUDA sparse CSR accepts 32-bit indices and they halve the workspace for
    # dense SDPA slacks such as biomedP.  Promote only when SciPy produced a
    # wider index type already.
    index_dtype = np.int64 if matrix.indices.dtype.itemsize > 4 else np.int32
    crow = torch.from_numpy(
        matrix.indptr.astype(index_dtype, copy=False)
    ).to(device=device)
    columns = torch.from_numpy(
        matrix.indices.astype(index_dtype, copy=False)
    ).to(device=device)
    values = torch.from_numpy(matrix.data).to(device=device)
    operator = torch.sparse_csr_tensor(
        crow,
        columns,
        values,
        size=matrix.shape,
        dtype=torch.float64,
        device=device,
    )

    basis = torch.empty(
        (n, maxiter + 1), dtype=torch.float64, device=device
    )
    diagonal = torch.empty(maxiter, dtype=torch.float64, device=device)
    off_diagonal = torch.empty(maxiter, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    seed = GPU_LANCZOS_SEED + int(block)
    generator.manual_seed(seed)
    vector = torch.randn(
        n, dtype=torch.float64, device=device, generator=generator
    )
    vector /= torch.linalg.vector_norm(vector)
    basis[:, 0] = vector
    previous = torch.zeros_like(vector)
    previous_beta = torch.zeros((), dtype=torch.float64, device=device)

    theta_value = None
    residual_value = None
    converged = False
    breakdown = False
    iterations = 0
    for iteration in range(maxiter):
        vector = basis[:, iteration]
        work = torch.mv(operator, vector) - previous_beta * previous
        alpha = torch.dot(vector, work)
        work -= alpha * vector

        # A second modified Gram-Schmidt pass controls loss of orthogonality
        # in the large near-zero clusters produced by low-rank SDP optima.
        active_basis = basis[:, : iteration + 1]
        for _ in range(2):
            coefficients = active_basis.T @ work
            work -= active_basis @ coefficients

        beta = torch.linalg.vector_norm(work)
        diagonal[iteration] = alpha
        off_diagonal[iteration] = beta
        iterations = iteration + 1
        beta_value = float(beta)
        breakdown = beta_value <= np.finfo(np.float64).eps
        if not breakdown:
            basis[:, iteration + 1] = work / beta
        previous = vector
        previous_beta = beta

        check = (
            breakdown
            or iterations % GPU_LANCZOS_CHECK_INTERVAL == 0
            or iterations == maxiter
        )
        if not check:
            continue
        tridiagonal = torch.diag(diagonal[:iterations])
        if iterations > 1:
            tridiagonal += torch.diag(
                off_diagonal[: iterations - 1], diagonal=1
            )
            tridiagonal += torch.diag(
                off_diagonal[: iterations - 1], diagonal=-1
            )
        eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
        theta = eigenvalues[0]
        residual = off_diagonal[iterations - 1] * torch.abs(
            eigenvectors[-1, 0]
        )
        theta_value = float(theta)
        residual_value = float(residual)
        convergence_tolerance = tol * (1.0 + abs(theta_value))
        if breakdown or residual_value <= convergence_tolerance:
            converged = True
            break

    if theta_value is None or residual_value is None:
        raise RuntimeError("GPU Lanczos produced no Ritz estimate")
    if not converged:
        raise RuntimeError(
            "GPU Lanczos did not converge in "
            f"{iterations} iterations (theta={theta_value:.9e}, "
            f"residual={residual_value:.3e})"
        )

    # Verify the recurrence-based residual with an explicit sparse matvec.
    ritz_vector = basis[:, :iterations] @ eigenvectors[:, 0]
    explicit_residual = float(
        torch.linalg.vector_norm(
            torch.mv(operator, ritz_vector) - theta * ritz_vector
        )
    )
    residual_value = max(residual_value, explicit_residual)
    conservative_value = theta_value - residual_value
    detail = {
        "backend": "torch.cuda.full-reorthogonalized-lanczos",
        "gpu": gpu,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "device_name": torch.cuda.get_device_name(device),
        "iterations": iterations,
        "maxiter": maxiter,
        "check_interval": GPU_LANCZOS_CHECK_INTERVAL,
        "seed": seed,
        "tol": tol,
        "ritz_value": theta_value,
        "ritz_residual": residual_value,
        "conservative_value": conservative_value,
        "breakdown": bool(breakdown),
        "block_dim": n,
        "nnz": int(matrix.nnz),
    }
    del operator, basis, diagonal, off_diagonal
    torch.cuda.empty_cache()
    return conservative_value, detail


def evaluate(
    problem: SdpaProblem,
    factors: list[np.ndarray],
    y: np.ndarray,
    lp_primal: np.ndarray | None = None,
    *,
    eigen_gpu: int | None = None,
) -> dict:
    if len(factors) != len(problem.block_dims):
        raise ValueError("Checkpoint cone count does not match the SDPA file")
    if y.size != problem.m:
        raise ValueError("Checkpoint dual length does not match the SDPA file")
    if lp_primal is None:
        lp_primal = np.empty(0, dtype=np.float64)
    if lp_primal.size != problem.lp_dim:
        raise ValueError("Checkpoint LP length does not match the SDPA file")

    constraint_values = np.zeros(problem.m, dtype=np.float64)
    primal_obj = 0.0
    slacks: list[csr_matrix] = []
    complementarity = 0.0
    min_x = np.inf
    min_z = np.inf
    c_norm_inf = 0.0
    c_abs_sum = 0.0
    eigen_ncv_by_block: list[int] = []
    eigen_backend_by_block: list[str] = []
    eigen_details_by_block: list[dict] = []

    for block, (n, u, raw_block) in enumerate(
        zip(problem.block_dims, factors, problem.psd_block_indices)
    ):
        if u.shape[0] != n:
            raise ValueError(f"Factor row count mismatch in block {block}")
        mask = problem.blk == raw_block
        mat = problem.mat[mask]
        row = problem.row[mask]
        col = problem.col[mask]
        raw = problem.val[mask]
        weights = np.where(row == col, 1.0, 2.0)
        x_at_entry = np.einsum("ij,ij->i", u[row], u[col])
        inner_terms = raw * weights * x_at_entry

        obj_mask = mat == 0
        primal_obj += float(np.sum(-inner_terms[obj_mask]))
        con_mask = ~obj_mask
        constraint_values += np.bincount(
            mat[con_mask] - 1, weights=inner_terms[con_mask], minlength=problem.m
        )

        # SDPA F0 is negated by both solvers: C=-F0 and Z=C-A^*y.
        coeff = np.empty_like(raw)
        coeff[obj_mask] = -raw[obj_mask]
        coeff[con_mask] = -raw[con_mask] * y[mat[con_mask] - 1]
        z = symmetric_sparse(n, row, col, coeff)
        slacks.append(z)
        c_norm_inf = max(c_norm_inf, float(np.max(np.abs(raw[obj_mask]), initial=0.0)))
        c_abs_sum += float(
            np.sum(np.abs(raw[obj_mask]) * np.where(row[obj_mask] == col[obj_mask], 1.0, 2.0))
        )
        complementarity += float(np.sum(u * (z @ u)))
        # X=UU^T is PSD analytically; report a conservative roundoff value.
        min_x = min(min_x, 0.0)
        # The returned BM factor normally spans an approximate nullspace of
        # the dual slack.  Its rank is therefore a useful lower estimate of
        # the tightly clustered eigenvalues around zero.  ARPACK's default
        # ncv (about 20 for k=1) can be smaller than that cluster: foot/rank67
        # then needed hundreds of seconds, while ncv=140 reproduced the same
        # eigenvalue in 1.2 seconds.  Cover twice the factor rank plus a small
        # margin.  This changes only Krylov workspace, not the tolerance or
        # reconstructed original-coordinate operator.
        eigen_ncv = min(n, max(40, 2 * int(u.shape[1]) + 6))
        eigen_ncv_by_block.append(eigen_ncv)
        gpu_error = None
        use_gpu = eigen_gpu is not None and (
            n >= GPU_LANCZOS_DIM_THRESHOLD
            or z.nnz >= GPU_LANCZOS_NNZ_THRESHOLD
        )
        if use_gpu:
            try:
                z_min, eigen_detail = _torch_cuda_smallest_eigenvalue(
                    z, gpu=eigen_gpu, block=block
                )
            except Exception as error:
                gpu_error = f"{type(error).__name__}: {error}"
            else:
                eigen_backend_by_block.append(eigen_detail["backend"])
                eigen_details_by_block.append(eigen_detail)
        if not use_gpu or gpu_error is not None:
            matrix_density = float(z.nnz) / float(n * n)
            use_cpu_dense_exact = (
                CPU_DENSE_EXACT_DIM_MIN <= n <= CPU_DENSE_EXACT_DIM_MAX
                and matrix_density >= CPU_DENSE_EXACT_DENSITY_MIN
            )
            if use_cpu_dense_exact:
                # Dense, medium-sized slacks are a poor fit for shift-free
                # ARPACK when the BM factor leaves a large cluster near zero.
                # A full symmetric eigensolve has bounded work here and is
                # exact up to the LAPACK backward error.  Keep sparse ARPACK
                # for smaller or genuinely sparse blocks, and prefer the GPU
                # path above whenever it was explicitly assigned and worked.
                z_min = float(np.linalg.eigvalsh(z.toarray())[0])
                eigen_detail = {
                    "backend": "numpy.linalg.eigvalsh-structural-dense",
                    "block_dim": n,
                    "nnz": int(z.nnz),
                    "density": matrix_density,
                    "dimension_range": [
                        CPU_DENSE_EXACT_DIM_MIN,
                        CPU_DENSE_EXACT_DIM_MAX,
                    ],
                    "density_min": CPU_DENSE_EXACT_DENSITY_MIN,
                }
            else:
                try:
                    # 1e-7 is tighter than the 1e-6 accuracy track. Large
                    # clustered graph slacks retain a 5000-update safety budget,
                    # although the rank-aware Krylov space normally converges
                    # much earlier.
                    z_min = float(
                        eigsh(
                            z,
                            k=1,
                            which="SA",
                            tol=EIGEN_TOL,
                            maxiter=EIGEN_MAXITER,
                            ncv=eigen_ncv,
                            return_eigenvectors=False,
                        )[0]
                    )
                    eigen_detail = {
                        "backend": "scipy.sparse.linalg.eigsh",
                        "tol": EIGEN_TOL,
                        "maxiter": EIGEN_MAXITER,
                        "ncv": eigen_ncv,
                        "block_dim": n,
                        "nnz": int(z.nnz),
                    }
                except ArpackNoConvergence as error:
                    if error.eigenvalues is not None and error.eigenvalues.size:
                        z_min = float(np.min(error.eigenvalues))
                        eigen_detail = {
                            "backend": "scipy.sparse.linalg.eigsh-partial",
                            "tol": EIGEN_TOL,
                            "maxiter": EIGEN_MAXITER,
                            "ncv": eigen_ncv,
                            "block_dim": n,
                            "nnz": int(z.nnz),
                        }
                    elif n <= 5000:
                        z_min = float(np.linalg.eigvalsh(z.toarray())[0])
                        eigen_detail = {
                            "backend": "numpy.linalg.eigvalsh",
                            "block_dim": n,
                            "nnz": int(z.nnz),
                        }
                    else:
                        raise
                except Exception:
                    if n > 5000:
                        raise
                    z_min = float(np.linalg.eigvalsh(z.toarray())[0])
                    eigen_detail = {
                        "backend": "numpy.linalg.eigvalsh",
                        "block_dim": n,
                        "nnz": int(z.nnz),
                    }
            if gpu_error is not None:
                eigen_detail["gpu_fallback_error"] = gpu_error
            eigen_backend_by_block.append(eigen_detail["backend"])
            eigen_details_by_block.append(eigen_detail)
        min_z = min(min_z, z_min)

    if problem.lp_dim:
        lp_slack = np.zeros(problem.lp_dim, dtype=np.float64)
        for raw_block, offset in problem.lp_block_offsets.items():
            mask = problem.blk == raw_block
            mat = problem.mat[mask]
            row = problem.row[mask]
            col = problem.col[mask]
            raw = problem.val[mask]
            if np.any(row != col):
                raise ValueError(f"Non-diagonal entry in SDPA LP block {raw_block}")
            var = offset + row
            obj_mask = mat == 0
            if np.any(obj_mask):
                np.add.at(lp_slack, var[obj_mask], -raw[obj_mask])
                primal_obj += float(np.sum(-raw[obj_mask] * lp_primal[var[obj_mask]]))
                c_norm_inf = max(
                    c_norm_inf, float(np.max(np.abs(raw[obj_mask]), initial=0.0))
                )
                c_abs_sum += float(np.sum(np.abs(raw[obj_mask])))
            con_mask = ~obj_mask
            if np.any(con_mask):
                constr = mat[con_mask] - 1
                values = raw[con_mask]
                variables = var[con_mask]
                constraint_values += np.bincount(
                    constr,
                    weights=values * lp_primal[variables],
                    minlength=problem.m,
                )
                np.add.at(lp_slack, variables, -values * y[constr])
        complementarity += float(lp_primal @ lp_slack)
        min_x = min(min_x, float(np.min(lp_primal)))
        min_z = min(min_z, float(np.min(lp_slack)))

    primal_residual = float(np.linalg.norm(constraint_values - problem.b))
    dual_obj = float(problem.b @ y)
    denom_obj = 1.0 + abs(primal_obj) + abs(dual_obj)
    b_norm_inf = float(np.max(np.abs(problem.b), initial=0.0))
    b_abs_sum = float(np.sum(np.abs(problem.b)))
    primal_cone_violation = max(0.0, -min_x)
    dual_cone_violation = max(0.0, -min_z)
    err1 = primal_residual / (1.0 + b_norm_inf)
    err2 = primal_cone_violation / (1.0 + b_norm_inf)
    err3 = 0.0  # Z is explicitly defined as C-A^*y.
    err4 = dual_cone_violation / (1.0 + c_norm_inf)
    err5 = (primal_obj - dual_obj) / denom_obj
    err6 = complementarity / denom_obj
    l1_err1 = primal_residual / (1.0 + b_abs_sum)
    l1_err2 = primal_cone_violation / (1.0 + b_abs_sum)
    l1_err4 = dual_cone_violation / (1.0 + c_abs_sum)
    return {
        "primal_objective": primal_obj,
        "dual_objective": dual_obj,
        "absolute_primal_residual": primal_residual,
        "lambda_min_x": min_x,
        "lambda_min_z": min_z,
        "complementarity": complementarity,
        "lp_dim": problem.lp_dim,
        "eigen_solver": {
            "name": "hybrid independent smallest-eigenvalue validation",
            "tol": EIGEN_TOL,
            "maxiter": EIGEN_MAXITER,
            "ncv_by_block": eigen_ncv_by_block,
            "ncv_rule": "min(block_dim, max(40, 2*factor_rank+6))",
            "backend_by_block": eigen_backend_by_block,
            "details_by_block": eigen_details_by_block,
            "gpu": eigen_gpu,
            "gpu_lanczos_nnz_threshold": GPU_LANCZOS_NNZ_THRESHOLD,
            "gpu_lanczos_dim_threshold": GPU_LANCZOS_DIM_THRESHOLD,
            "cpu_dense_exact_dim_range": [
                CPU_DENSE_EXACT_DIM_MIN,
                CPU_DENSE_EXACT_DIM_MAX,
            ],
            "cpu_dense_exact_density_min": CPU_DENSE_EXACT_DENSITY_MIN,
            "cpu_dense_exact_rule": (
                "when no assigned GPU eigensolver succeeds, use dense exact "
                "LAPACK for structurally dense medium-sized blocks; otherwise "
                "use sparse ARPACK with dense fallback on failure"
            ),
            "gpu_lanczos_rule": (
                "use fully reorthogonalized CUDA Lanczos when eigen_gpu is "
                "set and block dimension or nnz reaches its threshold; "
                "return ritz_value-ritz_residual; otherwise use CPU ARPACK"
            ),
        },
        "normalization": {
            "b_max_abs": b_norm_inf,
            "b_abs_sum": b_abs_sum,
            "c_max_abs": c_norm_inf,
            "c_abs_sum": c_abs_sum,
            "objective_gap_denominator": denom_obj,
        },
        "cone_violation": {
            "primal": primal_cone_violation,
            "dual": dual_cone_violation,
        },
        "dimacs": {
            "err1": err1,
            "err2": err2,
            "err3": err3,
            "err4": err4,
            "err5": err5,
            "err6": err6,
            "err6_valid": err2 == 0.0 and err4 == 0.0,
        },
        "entrywise_l1_normalized": {
            "err1": l1_err1,
            "err2": l1_err2,
            "err3": 0.0,
            "err4": l1_err4,
            "err5": err5,
            "err6": err6,
            "err6_valid": l1_err2 == 0.0 and l1_err4 == 0.0,
            "note": (
                "Diagnostic normalization using 1+sum(abs(b)) and the "
                "entrywise symmetric-matrix 1+sum(abs(C)); not the "
                "Mittelmann/DIMACS acceptance metric"
            ),
        },
    }


def evaluate_solution_file(
    problem_path: Path,
    solution_kind: str,
    solution_path: Path,
    expected_dual_objective: float | None = None,
    eigen_gpu: int | None = None,
) -> dict:
    """Evaluate one saved solution and return a JSON-serializable record."""
    problem_path = Path(problem_path)
    solution_path = Path(solution_path)
    problem = read_sdpa(problem_path)
    if solution_kind == "cardal-binary":
        factors, y, lp_primal, metadata = load_cardal(solution_path)
    elif solution_kind == "cardal-text":
        factors, y, lp_primal, metadata = load_cardal_text(
            solution_path, problem.block_dims, problem.m, problem.lp_dim
        )
    elif solution_kind == "culorads-mat":
        factors, y, lp_primal, metadata = load_culorads(solution_path)
    elif solution_kind == "pdsdp-mat":
        factors, y, lp_primal, metadata = load_pdsdp(solution_path, problem)
    elif solution_kind == "pdsdp-cuda-bin":
        factors, y, lp_primal, metadata = load_pdsdp_cuda(solution_path, problem)
    else:
        raise ValueError(f"Unsupported solution kind: {solution_kind}")

    result = evaluate(problem, factors, y, lp_primal, eigen_gpu=eigen_gpu)
    checkpoint_dual = result["dual_objective"]
    dual_scale_mismatch = False
    if expected_dual_objective is not None:
        dual_scale_mismatch = abs(checkpoint_dual - expected_dual_objective) > (
            1e-6 * (1.0 + abs(expected_dual_objective))
        )
    result["dual_checkpoint_valid"] = not dual_scale_mismatch
    if dual_scale_mismatch:
        # Some cuLoRADS MAT files store the internally scaled dual without the
        # row-scaling vector. Preserve primal metrics, use the logged scalar
        # dual objective for err5, and mark slack-dependent metrics unavailable.
        result["dual_objective_checkpoint"] = checkpoint_dual
        result["dual_objective"] = expected_dual_objective
        denom = 1.0 + abs(result["primal_objective"]) + abs(expected_dual_objective)
        result["dimacs"]["err3"] = None
        result["dimacs"]["err4"] = None
        result["dimacs"]["err5"] = (
            result["primal_objective"] - expected_dual_objective
        ) / denom
        result["dimacs"]["err6"] = None
        result["dimacs"]["err6_valid"] = False
        result["dual_checkpoint_note"] = (
            "stored dual objective does not match the solver-reported objective; "
            "per-constraint unscaling metadata is unavailable"
        )
    result.update(
        {
            "solution_kind": solution_kind,
            "problem": problem_path.name,
            "solution": str(solution_path),
            "m": problem.m,
            "block_dims": problem.block_dims,
            "raw_block_dims": problem.raw_block_dims,
            "lp_dim": problem.lp_dim,
            **metadata,
        }
    )
    result["validator_artifacts"] = {
        "dimacs_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "numpy_version": str(np.__version__),
        "scipy_version": __import__("scipy").__version__,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument(
        "--solution-kind",
        required=True,
        choices=(
            "cardal-binary",
            "cardal-text",
            "culorads-mat",
            "pdsdp-mat",
            "pdsdp-cuda-bin",
        ),
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-dual-objective", type=float)
    parser.add_argument(
        "--eigen-gpu",
        type=int,
        help=(
            "GPU index for independent Lanczos on large reconstructed slacks; "
            "small blocks and unavailable CUDA fall back to CPU ARPACK"
        ),
    )
    args = parser.parse_args(argv)

    result = evaluate_solution_file(
        args.problem,
        args.solution_kind,
        args.checkpoint,
        args.expected_dual_objective,
        args.eigen_gpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
