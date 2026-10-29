#!/usr/bin/env python3
"""Run one native PDSDP solve and preserve a self-contained validation record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BINARY = ROOT / "build" / "pdsdp-cuda"
DEFAULT_VALIDATOR = ROOT / "validate_cuda_checkpoint.py"
DIMACS_SOURCE = ROOT.parent / "sdp_bench" / "src" / "gpu_sdp_bench" / "dimacs.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_result(line: str) -> dict:
    prefix = "PDSDP_CUDA_RESULT "
    if not line.startswith(prefix):
        raise ValueError("not a PDSDP_CUDA_RESULT line")
    result: dict[str, object] = {}
    integer_keys = {"iterations"}
    float_keys = {
        "solver_time_sec",
        "final_time_sec",
        "process_time_sec",
        "primal",
        "dual",
        "gap",
        "primal_obj",
        "dual_obj",
    }
    for token in shlex.split(line[len(prefix) :]):
        key, value = token.split("=", 1)
        if key in integer_keys:
            result[key] = int(value)
        elif key in float_keys:
            result[key] = float(value)
        else:
            result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int, help="physical solver GPU")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--max-iterations", type=int, default=1_000_000_000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--omp-threads", type=int, default=32)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--validator-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--eigen-gpu",
        type=int,
        help="optional physical GPU for independent large-slack validation",
    )
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    problem = args.problem.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    validator = args.validator.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = output_dir / "solution.pdsdpbin"
    solver_log = output_dir / "solver.log"
    time_file = output_dir / "time.txt"
    solver_argv = [
        str(binary),
        str(problem),
        str(checkpoint),
        repr(args.time_limit),
        str(args.max_iterations),
        repr(args.tolerance),
    ]
    timed_argv = [
        "/usr/bin/time",
        "-f",
        "wall_sec=%e user_sec=%U sys_sec=%S max_rss_kib=%M",
        "-o",
        str(time_file),
        "--",
        *solver_argv,
    ]
    command = {
        "argv": timed_argv,
        "cwd": str(ROOT),
        "environment": {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OMP_NUM_THREADS": str(args.omp_threads),
        },
        "physical_gpu": args.gpu,
        "time_limit_sec": args.time_limit,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
    }
    (output_dir / "command.json").write_text(
        json.dumps(command, indent=2, sort_keys=True) + "\n"
    )

    environment = os.environ.copy()
    environment.update(command["environment"])
    start = time.monotonic()
    result_line = None
    with solver_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            timed_argv,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            if line.startswith("PDSDP_CUDA_RESULT "):
                result_line = line.rstrip("\n")
        return_code = process.wait()
    solver_wall_sec = time.monotonic() - start

    native_result = parse_result(result_line) if result_line is not None else None
    validation = None
    validation_return_code = None
    if return_code == 0 and checkpoint.exists() and not args.skip_validation:
        validation_path = output_dir / "validation.json"
        validation_argv = [
            str(args.validator_python),
            str(validator),
            "--problem",
            str(problem),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(validation_path),
            "--tolerance",
            repr(args.tolerance),
        ]
        validation_environment = os.environ.copy()
        if args.eigen_gpu is not None:
            validation_environment["CUDA_VISIBLE_DEVICES"] = str(args.eigen_gpu)
            validation_argv.extend(["--eigen-gpu", "0"])
        validation_command = {
            "argv": validation_argv,
            "physical_eigen_gpu": args.eigen_gpu,
        }
        (output_dir / "validation_command.json").write_text(
            json.dumps(validation_command, indent=2, sort_keys=True) + "\n"
        )
        with (output_dir / "validation.log").open("w", encoding="utf-8") as log:
            validation_process = subprocess.run(
                validation_argv,
                cwd=ROOT,
                env=validation_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        validation_return_code = validation_process.returncode
        if validation_return_code == 0:
            validation = json.loads(validation_path.read_text())

    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "record.json":
            artifacts[path.name] = sha256(path)
    sources = {
        "binary": sha256(binary),
        "benchmark_driver": sha256(Path(__file__)),
        "pdsdp_cuda": sha256(ROOT / "src" / "pdsdp_cuda.cu"),
        "makefile": sha256(ROOT / "Makefile"),
        "validator_wrapper": sha256(validator),
        "dimacs_validator": sha256(DIMACS_SOURCE),
        "problem": sha256(problem),
    }
    record = {
        "schema_version": 1,
        # A solver-only checkpoint is intentionally incomplete until the
        # decoupled campaign validator attaches an independent validation.
        "complete": (
            return_code == 0
            and native_result is not None
            and not args.skip_validation
        ),
        "native_converged": bool(
            native_result is not None and native_result.get("status") == "CONVERGED"
        ),
        "independently_strict": (
            None if validation is None else bool(validation["strict_at_tolerance"])
        ),
        "solver_return_code": return_code,
        "validation_return_code": validation_return_code,
        "solver_process_wall_sec": solver_wall_sec,
        "native_result": native_result,
        "validation": validation,
        "command": command,
        "artifact_sha256": artifacts,
        "source_sha256": sources,
    }
    (output_dir / "record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "record": str(output_dir / "record.json"),
                "solver_return_code": return_code,
                "native_converged": record["native_converged"],
                "independently_strict": record["independently_strict"],
                "solver_process_wall_sec": solver_wall_sec,
            },
            sort_keys=True,
        )
    )
    return return_code if return_code != 0 else (validation_return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
