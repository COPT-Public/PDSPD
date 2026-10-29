# PDSDP implementations and Mittelmann-75 comparison

This directory collects the two PDSDP implementations used in the 2026-09-14
Mittelmann-75 experiments:

- [`julia-original/`](julia-original/): the original Julia/CUDA.jl
  implementation and its campaign tools;
- [`cpp-cuda/`](cpp-cuda/): the native C++17/CUDA reimplementation, its
  independent-validation wrapper, and its decoupled campaign scheduler;
- [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md): the detailed experimental
  protocol, Julia-versus-C++ results, Cardal v294 comparison, structural
  analysis, and limitations.

Each implementation contains a `benchmark-summary/` directory with the frozen
manifest, flat comparison table, final progress/status, and campaign journal.
The multi-gigabyte solution checkpoints and per-instance validation files are
not duplicated here. Their authoritative locations are:

- Julia:
  `/DATA/disk2/wyc/sdp_project/PDSDP-main/benchmark_runs/full75_2x_v294_cuda128_20260914`
- C++/CUDA:
  `/DATA/disk2/wyc/sdp_project/PDSDP-cuda/benchmark_runs/full75_2x_v294_gpu0_1_20260914`

The original working directories were preserved. The two implementation
directories here are source and small-result snapshots, not relocated working
trees.

