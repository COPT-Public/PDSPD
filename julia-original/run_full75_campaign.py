#!/usr/bin/env python3
"""Run a resumable PDSDP Mittelmann-75 campaign with per-instance budgets."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PDSDP_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = WORKSPACE / "sdp_bench"
DATASET_ROOT = BENCH_ROOT / ".cache/datasets/mittelmann-75"
BASELINE_REPORT = BENCH_ROOT / "reports/v294-vs-culorads-75-comparison-0911.json"
JULIA = WORKSPACE / ".tools/julia-1.12.5/bin/julia"
JULIA_DEPOT = WORKSPACE / ".tools/julia-depot"
PYTHON = Path("/DATA/disk2/wyc/conda/envs/sdp-cu128/bin/python")
SOLVER = PDSDP_ROOT / "mittelmann_runner.jl"
VALIDATOR = PDSDP_ROOT / "validate_mittelmann.py"
TOLERANCE = 1e-4
MAX_ITERATIONS = 1_000_000_000
FORMAL_TIME_CAP_SEC = 3600.0
SOLVER_WALL_GRACE_SEC = 900.0
VALIDATION_TIMEOUT_SEC = 3600.0


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_tokens(line: str) -> list[str]:
    stripped = line.lstrip()
    if not stripped or stripped[0] in '*"#':
        return []
    for char in ",{}()=:":
        line = line.replace(char, " ")
    return line.split()


def sdpa_structure(path: Path) -> dict:
    tokens: list[str] = []
    with path.open("rt", encoding="ascii", errors="strict") as stream:
        while len(tokens) < 2:
            tokens.extend(numeric_tokens(stream.readline()))
        block_count = int(tokens[1])
        while len(tokens) < 2 + block_count:
            tokens.extend(numeric_tokens(stream.readline()))
    m = int(tokens[0])
    raw_dims = [int(value) for value in tokens[2 : 2 + block_count]]
    psd_dims = [value for value in raw_dims if value > 0]
    lp_dim = sum(-value for value in raw_dims if value < 0)
    packed_n = sum(value * (value + 1) // 2 for value in psd_dims) + lp_dim
    return {
        "m": m,
        "raw_block_dims": raw_dims,
        "psd_cone_count": len(psd_dims),
        "lp_dim": lp_dim,
        "effective_pdsdp_cone_count": len(psd_dims) + lp_dim,
        "maximum_psd_cone_dim": max(psd_dims, default=0),
        "packed_primal_dimension": packed_n,
        "dense_projection_work_proxy": sum(value**3 for value in psd_dims) + lp_dim,
    }


def source_identity() -> dict[str, str]:
    paths = {
        "uploaded_archive": WORKSPACE / "PDSDP-main.zip",
        "campaign_driver": Path(__file__).resolve(),
        "mittelmann_runner": SOLVER,
        "validator_wrapper": VALIDATOR,
        "halpern_pdhg": PDSDP_ROOT / "src/halpern_pdhg.jl",
        "pdhg_core": PDSDP_ROOT / "src/pdhg_core.jl",
        "scaling": PDSDP_ROOT / "src/scaling.jl",
        "dimacs_validator": BENCH_ROOT / "src/gpu_sdp_bench/dimacs.py",
        "julia_manifest": PDSDP_ROOT / "Manifest.toml",
        "cuda_preferences": PDSDP_ROOT / "LocalPreferences.toml",
        "cardal_baseline_report": BASELINE_REPORT,
    }
    return {name: sha256(path) for name, path in paths.items()}


def make_manifest() -> dict:
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    rows = []
    for row in baseline["rows"]:
        instance = row["instance"]
        problem = DATASET_ROOT / f"{instance}.dat-s"
        if not problem.is_file():
            raise FileNotFoundError(problem)
        cardal = row["v294"]
        cardal_time = float(cardal["solver_time_sec"])
        structure = sdpa_structure(problem)
        many_small_cones = (
            structure["psd_cone_count"] >= 32
            and structure["maximum_psd_cone_dim"] <= 32
        )
        rows.append(
            {
                "instance": instance,
                "problem": str(problem.resolve()),
                "problem_sha256": sha256(problem),
                "cardal": cardal,
                "cardal_solver_time_sec": cardal_time,
                "assigned_pdsdp_time_sec": 2.0 * cardal_time,
                "formal_time_cap_sec": FORMAL_TIME_CAP_SEC,
                "budget_beyond_formal_cap_is_exploratory": (
                    2.0 * cardal_time > FORMAL_TIME_CAP_SEC
                ),
                "many_small_cones": many_small_cones,
                "structure": structure,
            }
        )
    if len(rows) != 75 or len({row["instance"] for row in rows}) != 75:
        raise ValueError("Expected exactly 75 unique baseline rows")
    rows.sort(
        key=lambda row: (
            not row["many_small_cones"],
            row["structure"]["maximum_psd_cone_dim"],
            -row["structure"]["psd_cone_count"],
            row["instance"],
        )
    )
    return {
        "schema_version": 1,
        "campaign": "pdsdp-full75-2x-v294-cuda128-20260914",
        "created_at_unix": time.time(),
        "baseline_version": "CARDAL_evo_v294_progress_qualified_reopt_budget",
        "baseline_reason": (
            "v294 is the current formally promoted stable version; the v313 "
            "full-75 audit is still incomplete and is not eligible as baseline"
        ),
        "budget_rule": "2 * per-instance CARDAL v294 solver_time_sec",
        "formal_interpretation": (
            "PDSDP endpoints after 3600 seconds are exploratory and cannot pass "
            "the existing formal time gate"
        ),
        "tolerance": TOLERANCE,
        "max_iterations": MAX_ITERATIONS,
        "solver_wall_grace_sec": SOLVER_WALL_GRACE_SEC,
        "validation_timeout_sec": VALIDATION_TIMEOUT_SEC,
        "gpu_physical_index": 6,
        "cuda_runtime": "12.8 artifact selected by CUDA.jl LocalPreferences",
        "julia_version": "1.12.5",
        "python": str(PYTHON),
        "source_sha256": source_identity(),
        "summary": {
            "instances": len(rows),
            "sum_cardal_solver_time_sec": sum(
                row["cardal_solver_time_sec"] for row in rows
            ),
            "sum_assigned_pdsdp_time_sec": sum(
                row["assigned_pdsdp_time_sec"] for row in rows
            ),
            "many_small_cone_instances": sum(
                row["many_small_cones"] for row in rows
            ),
            "exploratory_beyond_3600_instances": sum(
                row["budget_beyond_formal_cap_is_exploratory"] for row in rows
            ),
        },
        "instances": rows,
    }


def parse_result(log_path: Path) -> dict | None:
    result_line = None
    with log_path.open("rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("PDSDP_RESULT "):
                result_line = line.strip()
    if result_line is None:
        return None
    raw = dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", result_line))
    result: dict[str, object] = dict(raw)
    for key in (
        "solver_time_sec",
        "termination_time_sec",
        "benchmark_time_sec",
        "primal",
        "dual",
        "gap",
        "primal_obj",
        "dual_obj",
    ):
        if key in result:
            result[key] = float(result[key])
    if "iterations" in result:
        result["iterations"] = int(result["iterations"])
    return result


def parse_external_time(path: Path) -> dict:
    if not path.is_file():
        return {}
    values = dict(re.findall(r"([a-z_]+)=([^\s]+)", path.read_text()))
    parsed: dict[str, object] = {}
    for key, value in values.items():
        parsed[key] = int(value) if key == "max_rss_kib" else float(value)
    return parsed


def terminate_process_group(process: subprocess.Popen, log) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    log.write(b"\nCAMPAIGN_DRIVER: external timeout; process group terminated\n")
    log.flush()


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    timeout_sec: float,
    env: dict[str, str],
) -> tuple[int | None, float, bool]:
    start = time.monotonic()
    timed_out = False
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=WORKSPACE,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process, log)
            return_code = None
    return return_code, time.monotonic() - start, timed_out


def next_attempt(campaign_dir: Path, instance: str) -> tuple[Path | None, dict | None]:
    instance_dir = campaign_dir / "runs" / instance
    attempts = sorted(instance_dir.glob("r[0-9][0-9][0-9]"))
    for attempt in reversed(attempts):
        record_path = attempt / "record.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("complete"):
                return None, record
    number = len(attempts) + 1
    attempt = instance_dir / f"r{number:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt, None


def file_hashes(attempt: Path) -> dict[str, str]:
    names = (
        "command.json",
        "solver.log",
        "time.txt",
        "solution.mat",
        "validation_command.json",
        "validation.log",
        "validation.json",
    )
    return {
        name: sha256(attempt / name)
        for name in names
        if (attempt / name).is_file()
    }


def classify(record: dict) -> dict:
    cardal = record["cardal"]
    pdsdp = record.get("pdsdp_result") or {}
    validation = record.get("validation") or {}
    termination_time = pdsdp.get("termination_time_sec")
    assigned_budget = record["assigned_pdsdp_time_sec"]
    within_assigned_budget = (
        isinstance(termination_time, (int, float))
        and math.isfinite(termination_time)
        and termination_time <= assigned_budget + max(1.0, assigned_budget * 1e-3)
    )
    native_converged = (
        record.get("solver_return_code") == 0
        and pdsdp.get("status") == "CONVERGED"
    )
    independently_strict = bool(validation.get("strict_at_tolerance", False))
    pdsdp_accepted_assigned = (
        native_converged and independently_strict and within_assigned_budget
    )
    within_formal_cap = (
        isinstance(termination_time, (int, float))
        and termination_time <= FORMAL_TIME_CAP_SEC
    )
    pdsdp_formally_accepted = pdsdp_accepted_assigned and within_formal_cap
    cardal_strict = bool(cardal.get("independently_strict", False))
    cardal_time = float(cardal["solver_time_sec"])
    solve_wall = record.get("solver_process_wall_sec")
    return {
        "native_converged": native_converged,
        "independently_strict": independently_strict,
        "within_assigned_budget": within_assigned_budget,
        "within_formal_3600_sec": within_formal_cap,
        "pdsdp_accepted_assigned_budget": pdsdp_accepted_assigned,
        "pdsdp_formally_accepted": pdsdp_formally_accepted,
        "new_formal_solve_over_cardal": pdsdp_formally_accepted and not cardal_strict,
        "same_quality_solver_time_faster": (
            pdsdp_formally_accepted
            and cardal_strict
            and termination_time < cardal_time
        ),
        "same_quality_end_to_end_faster": (
            pdsdp_formally_accepted
            and cardal_strict
            and isinstance(solve_wall, (int, float))
            and solve_wall < cardal_time
        ),
        "solver_time_speedup_over_cardal": (
            cardal_time / termination_time
            if pdsdp_formally_accepted and termination_time > 0
            else None
        ),
        "end_to_end_speedup_over_cardal_solver_time": (
            cardal_time / solve_wall
            if pdsdp_formally_accepted
            and isinstance(solve_wall, (int, float))
            and solve_wall > 0
            else None
        ),
    }


def write_progress(campaign_dir: Path, manifest: dict, current: str | None) -> None:
    records = []
    for row in manifest["instances"]:
        instance_dir = campaign_dir / "runs" / row["instance"]
        candidates = sorted(instance_dir.glob("r*/record.json"))
        if candidates:
            records.append(json.loads(candidates[-1].read_text(encoding="utf-8")))
    summary = {
        "total_instances": 75,
        "complete_records": len(records),
        "remaining": 75 - len(records),
        "current_instance": current,
        "pdsdp_native_converged": sum(
            record["classification"]["native_converged"] for record in records
        ),
        "pdsdp_independently_strict": sum(
            record["classification"]["independently_strict"] for record in records
        ),
        "pdsdp_formally_accepted": sum(
            record["classification"]["pdsdp_formally_accepted"] for record in records
        ),
        "same_quality_solver_time_faster": sum(
            record["classification"]["same_quality_solver_time_faster"]
            for record in records
        ),
        "same_quality_end_to_end_faster": sum(
            record["classification"]["same_quality_end_to_end_faster"]
            for record in records
        ),
        "new_formal_solves_over_cardal": sum(
            record["classification"]["new_formal_solve_over_cardal"]
            for record in records
        ),
        "updated_at_unix": time.time(),
    }
    rows = []
    for record in records:
        pdsdp = record.get("pdsdp_result") or {}
        validation = record.get("validation") or {}
        metrics = validation.get("strict_metrics") or {}
        rows.append(
            {
                "instance": record["instance"],
                "many_small_cones": record["many_small_cones"],
                "psd_cones": record["structure"]["psd_cone_count"],
                "max_cone_dim": record["structure"]["maximum_psd_cone_dim"],
                "lp_dim": record["structure"]["lp_dim"],
                "cardal_time": record["cardal"]["solver_time_sec"],
                "cardal_strict": record["cardal"]["independently_strict"],
                "assigned_budget": record["assigned_pdsdp_time_sec"],
                "pdsdp_status": pdsdp.get("status"),
                "pdsdp_time": pdsdp.get("termination_time_sec"),
                "pdsdp_wall": record.get("solver_process_wall_sec"),
                "p": metrics.get("p"),
                "d": metrics.get("d"),
                "g": metrics.get("g"),
                **record["classification"],
            }
        )
    atomic_json(
        campaign_dir / "progress.json",
        {"schema_version": 1, "summary": summary, "rows": rows},
    )
    columns = list(rows[0]) if rows else ["instance"]
    csv_tmp = campaign_dir / "comparison.csv.tmp"
    with csv_tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    csv_tmp.replace(campaign_dir / "comparison.csv")


def run_instance(campaign_dir: Path, row: dict, gpu: int) -> dict:
    attempt, existing = next_attempt(campaign_dir, row["instance"])
    if existing is not None:
        print(f"SKIP complete instance={row['instance']}", flush=True)
        return existing
    assert attempt is not None
    budget = float(row["assigned_pdsdp_time_sec"])
    solution = attempt / "solution.mat"
    time_path = attempt / "time.txt"
    solver_log = attempt / "solver.log"
    command = [
        "/usr/bin/time",
        "-f",
        "wall_sec=%e user_sec=%U sys_sec=%S max_rss_kib=%M",
        "-o",
        str(time_path),
        "--",
        str(JULIA),
        f"--project={PDSDP_ROOT}",
        str(SOLVER),
        row["problem"],
        str(solution),
        repr(budget),
        str(MAX_ITERATIONS),
        repr(TOLERANCE),
    ]
    atomic_json(
        attempt / "command.json",
        {
            "argv": command,
            "cwd": str(WORKSPACE),
            "physical_gpu": gpu,
            "assigned_solver_time_sec": budget,
        },
    )
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JULIA_NUM_THREADS": "32",
            "JULIA_DEPOT_PATH": str(JULIA_DEPOT),
            "PDSDP_BLAS_THREADS": "32",
            "PDSDP_STREAMS_PER_DEVICE": "8",
            "PDSDP_PRINT_FREQUENCY": "1000",
            "PDSDP_MINIMAL_CHECKPOINT": "true",
            "PDSDP_SAVE_RESIDUAL_PLOT": "false",
        }
    )
    print(
        f"START instance={row['instance']} budget={budget:.6f}s "
        f"cones={row['structure']['psd_cone_count']} "
        f"max_dim={row['structure']['maximum_psd_cone_dim']}",
        flush=True,
    )
    return_code, solve_wall, solver_timed_out = run_logged(
        command,
        log_path=solver_log,
        timeout_sec=budget + SOLVER_WALL_GRACE_SEC,
        env=env,
    )
    pdsdp_result = parse_result(solver_log)

    validation = None
    validation_return_code = None
    validation_timed_out = False
    validation_wall = None
    validation_path = attempt / "validation.json"
    if solution.is_file():
        validation_command = [
            str(PYTHON),
            str(VALIDATOR),
            "--problem",
            row["problem"],
            "--checkpoint",
            str(solution),
            "--output",
            str(validation_path),
            "--eigen-gpu",
            "0",
            "--tolerance",
            repr(TOLERANCE),
        ]
        atomic_json(
            attempt / "validation_command.json",
            {"argv": validation_command, "physical_gpu": gpu},
        )
        validation_env = os.environ.copy()
        validation_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        validation_return_code, validation_wall, validation_timed_out = run_logged(
            validation_command,
            log_path=attempt / "validation.log",
            timeout_sec=VALIDATION_TIMEOUT_SEC,
            env=validation_env,
        )
        if validation_path.is_file():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))

    record = {
        "schema_version": 1,
        "complete": True,
        "instance": row["instance"],
        "attempt": attempt.name,
        "problem": row["problem"],
        "problem_sha256": row["problem_sha256"],
        "many_small_cones": row["many_small_cones"],
        "structure": row["structure"],
        "cardal": row["cardal"],
        "assigned_pdsdp_time_sec": budget,
        "formal_time_cap_sec": FORMAL_TIME_CAP_SEC,
        "solver_return_code": return_code,
        "solver_external_timeout": solver_timed_out,
        "solver_process_wall_sec": solve_wall,
        "external_time": parse_external_time(time_path),
        "pdsdp_result": pdsdp_result,
        "validation_return_code": validation_return_code,
        "validation_external_timeout": validation_timed_out,
        "validation_wall_sec": validation_wall,
        "validation": validation,
        "source_sha256": source_identity(),
        "finished_at_unix": time.time(),
    }
    record["classification"] = classify(record)
    record["artifact_sha256"] = file_hashes(attempt)
    atomic_json(attempt / "record.json", record)
    print(
        f"DONE instance={row['instance']} status="
        f"{(pdsdp_result or {}).get('status')} strict="
        f"{record['classification']['independently_strict']} time="
        f"{(pdsdp_result or {}).get('termination_time_sec')} wall={solve_wall:.3f}",
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    campaign_dir = args.output.resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (campaign_dir / "campaign.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another campaign driver holds campaign.lock", file=sys.stderr)
        return 2

    manifest_path = campaign_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = make_manifest()
        atomic_json(manifest_path, manifest)
    write_progress(campaign_dir, manifest, None)
    if args.prepare_only:
        print(json.dumps(manifest["summary"], sort_keys=True))
        return 0

    rows = manifest["instances"]
    if args.limit is not None:
        rows = rows[: args.limit]
    atomic_json(
        campaign_dir / "campaign_status.json",
        {
            "status": "running",
            "pid": os.getpid(),
            "gpu_physical_index": args.gpu,
            "started_at_unix": time.time(),
        },
    )
    exit_code = 0
    for row in rows:
        atomic_json(
            campaign_dir / "campaign_status.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "gpu_physical_index": args.gpu,
                "current_instance": row["instance"],
                "updated_at_unix": time.time(),
            },
        )
        try:
            run_instance(campaign_dir, row, args.gpu)
        except Exception as error:
            exit_code = 1
            print(
                f"ERROR instance={row['instance']} {type(error).__name__}: {error}",
                flush=True,
            )
        write_progress(campaign_dir, manifest, row["instance"])
    write_progress(campaign_dir, manifest, None)
    progress = json.loads((campaign_dir / "progress.json").read_text())
    atomic_json(
        campaign_dir / "campaign_status.json",
        {
            "status": "complete" if progress["summary"]["remaining"] == 0 else "partial",
            "pid": os.getpid(),
            "gpu_physical_index": args.gpu,
            "exit_code": exit_code,
            "finished_at_unix": time.time(),
            "summary": progress["summary"],
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
