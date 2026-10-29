# Native C++/CUDA PDSDP

This directory contains the native C++17/CUDA reimplementation of the PDSDP
Halpern/rAPDHG solver used in the 2026-09-14 Mittelmann-75 experiment. It reads
SDPA-sparse problems, supports PSD blocks and negative SDPA diagonal blocks,
uses Float64 throughout the stopping path, and exports an original-order binary
checkpoint for independent validation.

## Requirements

- CUDA toolkit with cuSPARSE, cuSOLVER, cuBLAS, and CUDA Runtime;
- a C++17 host compiler;
- OpenMP;
- for the Python runner and independent validator: Python with NumPy, SciPy,
  h5py, and the adjacent `sdp_bench/src` package.

The default build targets H100 (`sm_90`). Override `CUDA_HOME` or `CUDA_ARCH`
when necessary.

## Build

From this directory:

```bash
make
```

The executable is written to `build/pdsdp-cuda`. Fast math is deliberately not
enabled because convergence is checked in Float64.

The tested source and build recipe have these SHA-256 values:

| file | SHA-256 |
|---|---|
| `src/pdsdp_cuda.cu` | `54a03efffa68035dc574775201d5282bd578884685fb142b7176542d65652818` |
| `Makefile` | `74478f1f6a37a0944aa2ec2bccd9aba05b5ea3087ac8f793658f3cd2674124ab` |
| tested binary | `5d6bd9d92044b8025f635f6c76e2ba8c5678405841e6993c915d4ca2ee1049bc` |

The tested binary is not duplicated in this source snapshot. It remains at
`/DATA/disk2/wyc/sdp_project/PDSDP-cuda/build/pdsdp-cuda`.

## Direct solver interface

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 ./build/pdsdp-cuda \
  PROBLEM.dat-s OUTPUT.pdsdpbin TIME_SEC MAX_ITER TOL
```

Defaults are 3600 seconds, 1,000,000,000 iterations, and tolerance `1e-4`.
The program refuses to overwrite an existing checkpoint.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=32 ./build/pdsdp-cuda \
  /path/to/ros_2000.dat-s /tmp/ros_2000.pdsdpbin 120 1000000000 1e-4
```

## Reproducible single run

`run_benchmark.py` records the command, environment, solver log, `/usr/bin/time`
statistics, checkpoint, independent validation result, source/artifact hashes,
and a joined `record.json`:

```bash
python run_benchmark.py \
  --problem /path/to/problem.dat-s \
  --output-dir /path/to/new-empty-output-dir \
  --gpu 0 \
  --time-limit 120 \
  --max-iterations 1000000000 \
  --tolerance 1e-4 \
  --omp-threads 32
```

The output directory must be empty. Add `--eigen-gpu 1` to use a physical GPU
for the independent validator's large smallest-eigenvalue calculation. Add
`--skip-validation` only when a separate validation queue will consume the
checkpoint.

The Python wrappers locate `sdp_bench` as a sibling of their own project
directory. In this archive layout, either run the original working copy under
`/DATA/disk2/wyc/sdp_project/PDSDP-cuda`, place `sdp_bench` next to this
directory, or update `BENCH_ROOT` explicitly.

## Stopping contract

Every 40 iterations, the solver evaluates the following quantities in original
problem coordinates:

```text
p = ||A(X)-b||_2 / (1+||b||_inf)
d = dist_F(C+A'*lambda, product PSD cone) / (1+||C||_inf)
g = |<C,X>+<b,lambda>| / (1+|<C,X>|+|<b,lambda>|)
```

It reports `CONVERGED` only when `max(p,d,g) <= TOL`. The internal dual
residual uses the Frobenius norm of all negative spectral parts, which is at
least as conservative as the independent DIMACS worst-eigenvalue check.

`validate_cuda_checkpoint.py` reparses the original SDPA file through the
independent `sdp_bench` validator. It records:

- `three_metric_strict_at_tolerance`: finite independent `p`, `d`, and `g` no
  larger than the requested tolerance;
- `strict_at_tolerance`: the three-metric test plus valid dual transport and no
  primal PSD projection larger than the `1e-10` transport guard.

The report presents both definitions. At tolerance `1e-4`, the transport guard
is intentionally much stricter than the numerical feasibility threshold.

The exact `dimacs.py` used by the C++ full-75 campaign is preserved for audit
at [`independent-validator/dimacs.py`](independent-validator/dimacs.py), with
SHA-256
`81b87fb9a4905ba8eebc007b042e33a7f99aabefcb9c98cf814e8ad19bd19613`.

## Checkpoint format

Version 1 is little-endian and contains:

1. eight-byte magic `PDSDPC1\0`;
2. `uint32 version`, `uint32 endian_marker`;
3. `int32 cone_count`, `int32 m`, `int64 primal_length`;
4. original-order `int32[cone_count]` cone dimensions;
5. original-scale `float64[primal_length]` orthonormal lower-row-major svec;
6. original-scale `float64[m]` PDSDP dual `lambda`.

The independent loader checks the header, endian marker, exact byte length,
primal/dual lengths, and cone layout before evaluating the solution.

## Implementation choices

- one `A'` and one `A` sparse product per rAPDHG trial, with extrapolated and
  delta products obtained by linear reuse;
- persistent cuSPARSE descriptors/workspaces and persistent device vectors;
- chunked `syevjBatched` projection for cones up to dimension 100 and `syevd`
  for larger cones;
- scalar CUDA kernels for 1x1/LP cones and cuBLAS reconstruction;
- fused vector kernels and no allocation in the iteration loop;
- original-coordinate residual kernels and per-projection cuSOLVER status
  checks;
- deterministic power-iteration seed and atomic checkpoint publication.

## Full-campaign scheduling

`run_full75_decoupled_campaign.py` separates work into:

- a dynamic GPU solve queue, where each GPU immediately takes another problem
  after publishing a checkpoint;
- a bounded CPU-only independent-validation queue.

This prevents long validation from leaving GPUs idle. The frozen campaign
scripts include workspace-specific dataset, baseline-manifest, and Python paths;
they are retained verbatim for provenance and should be parameterized before
reuse in another workspace.

## Full-75 result summary

On the common H100 setup and the relaxed independent `p/d/g <= 1e-4` rule:

- 35 of 75 C++/CUDA endpoints were feasible and all 35 terminated as
  `CONVERGED`;
- the Julia implementation produced 16 feasible endpoints, of which 14
  terminated as `CONVERGED`;
- C++/CUDA was faster on all 16 problems feasible for both implementations;
- median solver-time speedup over Julia was 2.48x and geometric-mean speedup
  was 3.21x;
- 24 C++/CUDA endpoints also passed the conservative transport-strict rule;
- compared with Cardal v294 on the 35 mutually feasible problems, C++/CUDA was
  faster on 25 by solver time and 22 by endpoint-generation wall time.

See [`../BENCHMARK_REPORT.md`](../BENCHMARK_REPORT.md) and
[`benchmark-summary/comparison.csv`](benchmark-summary/comparison.csv) for the
complete results.

## Current limitations

- Only 35 of 75 problems reached `p/d/g <= 1e-4`; Cardal v294 reached the
  comparison standard on 70, so this is currently a specialized or portfolio
  solver rather than a general replacement.
- Eleven converged C++ endpoints pass `p/d/g <= 1e-4` but fail the additional
  `1e-10` primal-transport guard because the stored full matrix has very small
  negative eigenvalues. A PSD-factor checkpoint or an explicitly projected and
  revalidated endpoint would remove this ambiguity.
- The default Makefile emits only `sm_90`; other GPU generations require an
  explicit `CUDA_ARCH` setting.
