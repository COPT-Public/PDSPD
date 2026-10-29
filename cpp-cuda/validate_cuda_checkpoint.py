#!/usr/bin/env python3
"""Independently validate a native PDSDP C++/CUDA checkpoint on SDPA data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1] / "sdp_bench"
sys.path.insert(0, str(BENCH_ROOT / "src"))

from gpu_sdp_bench.dimacs import evaluate_solution_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eigen-gpu", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    result = evaluate_solution_file(
        args.problem,
        "pdsdp-cuda-bin",
        args.checkpoint,
        eigen_gpu=args.eigen_gpu,
    )
    dimacs = result["dimacs"]
    metrics = {
        "p": float(dimacs["err1"]),
        "d": float(dimacs["err4"]),
        "g": abs(float(dimacs["err5"])),
    }
    result["strict_metrics"] = metrics
    result["three_metric_strict_at_tolerance"] = all(
        math.isfinite(value) and value <= args.tolerance
        for value in metrics.values()
    )
    result["strict_at_tolerance"] = bool(
        result["three_metric_strict_at_tolerance"]
        and result.get("dual_checkpoint_valid", False)
        and result.get("primal_transport_valid", False)
    )
    result["tolerance"] = args.tolerance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"strict": result["strict_at_tolerance"], **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
