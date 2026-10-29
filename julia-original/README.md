# Original Julia PDSDP

This is the Julia/CUDA.jl PDSDP source snapshot used by the full
Mittelmann-75 experiment on 2026-09-14. It implements Halpern/rAPDHG for
semidefinite programs and includes the SDPA-sparse adapter used in the
experiment.

## Environment

The recorded experiment used:

- Julia 1.12.5;
- CUDA 12.8 through CUDA.jl;
- NVIDIA H100 80GB;
- the package environment frozen by `Project.toml`, `Manifest.toml`, and
  `LocalPreferences.toml`.

Instantiate the environment from this directory with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Mittelmann entry point

```bash
CUDA_VISIBLE_DEVICES=0 PDSDP_BLAS_THREADS=32 \
  julia --project=. mittelmann_runner.jl \
  PROBLEM.dat-s OUTPUT.mat TIME_SEC MAX_ITER TOL
```

The full experiment used `MAX_ITER=1000000000`, `TOL=1e-4`, and an
instance-specific limit equal to twice Cardal v294's solver time.

Important files:

- `src/PDSDP.jl`: module assembly;
- `src/halpern_pdhg.jl` and `src/pdhg_core.jl`: solver logic;
- `src/projection.jl`: PSD projections;
- `src/scaling.jl`: problem scaling;
- `mittelmann_runner.jl`: SDPA-sparse loading, experiment settings, and MAT
  solution export;
- `validate_mittelmann.py`: independent solution validation;
- `run_full75_campaign.py`: frozen full-campaign driver.

The campaign scripts retain workspace-specific paths to `sdp_bench` and the
Mittelmann dataset. They are included as provenance-exact experiment tools;
adjust those constants before using this snapshot in another layout.

## Provenance

Selected SHA-256 values, matching `benchmark-summary/manifest.json`:

| file | SHA-256 |
|---|---|
| `mittelmann_runner.jl` | `6461b69cae19066de622f75c2b7250a50360b19d5f0efc76197b64bebe2da164` |
| `src/halpern_pdhg.jl` | `215dbc06c2f4628236ad10b9cc8ef1e15a83788e08d0995de56b46fdb15ab614` |
| `src/pdhg_core.jl` | `2ea68c79cfb3bda5433d5c57515a2114adfcaacac1b1f959b10f3ee315d6300e` |
| `src/scaling.jl` | `8f52fa483539fe8693fc39a9cf587cc4889cc81923f391f51bb3aea6e7230f4a` |
| `Manifest.toml` | `394f1564a10568f8c21146cb422e119d843e9d257940e37d113171758331a10c` |

See [`../BENCHMARK_REPORT.md`](../BENCHMARK_REPORT.md) for results.

