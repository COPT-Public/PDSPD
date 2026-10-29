# PDSDP Mittelmann interface and validation

## Acceptance contract

PDSDP is accepted under the same project gate used for CARDAL when all of the
following hold:

1. native status is `CONVERGED`;
2. the solver reaches that status within the 3600 second budget;
3. the saved solution is reparsed against the original SDPA file and satisfies
   `p <= 1e-4`, `d <= 1e-4`, and `abs(g) <= 1e-4`;
4. the process exits normally and the input, source, command, checkpoint, and
   validation output remain traceable.

The independent metrics are

```text
p = ||A(X)-b||_2 / (1+||b||_inf)
d = max(PSD violation, LP violation, free stationarity) / (1+||C||_inf)
g = |<C,X>-b^T y| / (1+|<C,X>|+|b^T y|).
```

The uploaded PDSDP stopping test was not equivalent: its primal and dual tests
used infinity norms and solution-dependent denominators. Its gap formula was
already equivalent. The patched origin-coordinate test uses the exact project
formula for `p` and `g`. Its internal `d` is the Euclidean svec distance to the
product PSD cone over the same `1+||C||_inf` denominator. This is conservative:
the Frobenius norm of all negative spectral parts is at least the worst negative
eigenvalue used by the independent validator. The final independent validation
remains authoritative.

The uploaded time limit was ineffective because `total_time` was only assigned
after the iteration loop. The patched loop refreshes it before every iteration,
starts the budget before power iteration, and reports exactly `max_iterations`
iterations rather than one extra. A 120 second `mater-6` run terminated with
native status `OUT_OF_TIME`, confirming the live budget path.

## Interfaces

`mittelmann_runner.jl` reads an SDPA sparse `.dat-s` file, translates the SDPA
objective sign, packs symmetric matrices using orthonormal svec coordinates,
and emits a MATLAB v7.3 checkpoint in restored scale and original cone order.
Positive SDPA blocks stay PSD cones. Every coordinate of a negative SDPA block
is represented as a 1x1 PSD cone so it retains the original nonnegative/LP
semantics rather than becoming one of PDSDP's free trailing variables.

Example solve (physical GPU 6):

```bash
env -u LD_LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=6 JULIA_NUM_THREADS=32 \
  JULIA_DEPOT_PATH="$PWD/.tools/julia-depot" \
  PDSDP_STREAMS_PER_DEVICE=8 \
  .tools/julia-1.12.5/bin/julia --project=PDSDP-main \
  PDSDP-main/mittelmann_runner.jl \
  sdp_bench/.cache/datasets/mittelmann-75/ros_2000.dat-s \
  PDSDP-main/benchmark_runs/ros_2000/solution.mat \
  3600 20000 1e-4
```

The existing independent validator now accepts `--solution-kind pdsdp-mat`:

```bash
PYTHONPATH=sdp_bench/src \
  /DATA/disk2/wyc/conda/envs/sdp-cu128/bin/python \
  sdp_bench/src/gpu_sdp_bench/dimacs.py \
  --problem sdp_bench/.cache/datasets/mittelmann-75/ros_2000.dat-s \
  --solution-kind pdsdp-mat \
  --checkpoint PDSDP-main/benchmark_runs/ros_2000_r005_final/solution.mat \
  --output PDSDP-main/benchmark_runs/ros_2000_r005_final/validation_core_cli.json
```

The loader restores each svec block, converts the PDSDP multiplier convention
`lambda` to benchmark convention `y=-lambda`, transports full PSD matrices as
factors, and passes the original LP tail unchanged. It records the minimum
eigenvalue before clipping numerical roundoff during factor transport.

## Measured result

The formal stable comparison baseline is CARDAL v294. On `ros_2000`, v294 was
independently strict in 670.3701 solver seconds. The final CUDA 12.8 PDSDP run
`ros_2000_r005_final` was natively `CONVERGED` at iteration 2920 in 17.1392
internal seconds and 61.51 full-process wall seconds. Independent validation returned

```text
p = 1.2640080151970114e-6
d = 9.22688491583356e-6
g = 1.3736848352760474e-5
```

Thus the measured speedups are 39.11x by reported solver time and 10.90x even
when all Julia startup, parsing, scaling, GPU setup, plotting, and MAT export
overhead is charged to PDSDP. Two earlier independent repetitions produced the
same iteration count and validation metrics.

`rose15` exercised a real negative SDPA block (`135`, `-2`) and successfully
round-tripped it as one PSD block plus two LP scalars; the independent validator
correctly rejected that nonconverged endpoint because `g=0.1837`. `mater-6` and
`biggs` were also not promising under the uploaded default algorithm. The clear
current PDSDP niche is therefore the many-small-PSD-cone structure exemplified
by `ros_2000`, not a blanket replacement for CARDAL.
