# PDSDP Mittelmann-75 full campaign

This directory is the immutable result root for the PDSDP full-75 comparison
against the formally promoted CARDAL v294 baseline.

## Protocol

- Dataset: all 75 local Mittelmann SDPA instances listed in `manifest.json`.
- Hardware: physical GPU 6, CUDA 12.8, Julia 1.12.5.
- Per-instance PDSDP limit: exactly `2 * CARDAL v294 solver_time_sec`.
- Iteration limit: 1,000,000,000, so the assigned time is the operative limit.
- Accuracy: native convergence plus independent validation on the original
  problem with all three DIMACS metrics `p`, `d`, and `g` at most `1e-4`.
- Existing formal time gate: 3600 seconds. Any PDSDP endpoint obtained after
  3600 seconds is retained but explicitly classified as exploratory.
- Timing: both PDSDP termination time and complete solver-process wall time are
  retained. Comparisons using either definition are reported separately.

The run order puts instances with at least 32 PSD cones and maximum PSD-cone
dimension at most 32 first. This tests the many-small-cones hypothesis early;
all remaining instances still run afterward.

## Live and final artifacts

- `manifest.json`: frozen inputs, ordering, budgets, environment, and hashes.
- `campaign_status.json`: current process and instance, or final status.
- `campaign.log`: one-line start/completion journal for every instance.
- `progress.json`: live machine-readable aggregate.
- `comparison.csv`: live flat comparison table.
- `PARALLELIZATION_AMENDMENT.md`: audited transition from the initial
  single-GPU queue to disjoint multi-GPU shards for the final 15 instances.
- `workers/gpu*/`: per-GPU shard status and scheduler logs.
- `runs/<instance>/r001/solver.log`: complete native solver output.
- `runs/<instance>/r001/solution.mat`: exported original-order solution.
- `runs/<instance>/r001/validation.json`: full independent validation result.
- `runs/<instance>/r001/record.json`: joined provenance, timing,
  classification, and SHA-256 hashes for that attempt.

The campaign is resumable. Completed records are never rerun; an interrupted
instance receives a new numbered attempt directory.
