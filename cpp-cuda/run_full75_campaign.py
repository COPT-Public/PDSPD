#!/usr/bin/env python3
"""Run a resumable two-GPU native PDSDP Mittelmann-75 campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
BENCH_ROOT = WORKSPACE / "sdp_bench"
SOURCE_CAMPAIGN = (
    WORKSPACE
    / "PDSDP-main/benchmark_runs/full75_2x_v294_cuda128_20260914/manifest.json"
)
PYTHON = Path("/DATA/disk2/wyc/conda/envs/sdp-cu128/bin/python")
SINGLE_RUNNER = ROOT / "run_benchmark.py"
BINARY = ROOT / "build/pdsdp-cuda"
TOLERANCE = 1e-4
MAX_ITERATIONS = 1_000_000_000
FORMAL_TIME_CAP_SEC = 3600.0
SOLVER_GRACE_SEC = 900.0
VALIDATION_GRACE_SEC = 3600.0


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


def source_identity() -> dict[str, str]:
    paths = {
        "binary": BINARY,
        "pdsdp_cuda": ROOT / "src/pdsdp_cuda.cu",
        "makefile": ROOT / "Makefile",
        "single_benchmark_driver": SINGLE_RUNNER,
        "campaign_driver": Path(__file__).resolve(),
        "validator_wrapper": ROOT / "validate_cuda_checkpoint.py",
        "dimacs_validator": BENCH_ROOT / "src/gpu_sdp_bench/dimacs.py",
        "cardal_v294_pdsdp_manifest": SOURCE_CAMPAIGN,
    }
    return {name: sha256(path) for name, path in paths.items()}


def make_manifest(gpus: list[int]) -> dict:
    source = json.loads(SOURCE_CAMPAIGN.read_text(encoding="utf-8"))
    rows = []
    for original in source["instances"]:
        row = {
            "instance": original["instance"],
            "problem": original["problem"],
            "problem_sha256": original["problem_sha256"],
            "cardal": original["cardal"],
            "cardal_solver_time_sec": original["cardal_solver_time_sec"],
            "assigned_pdsdp_time_sec": original["assigned_pdsdp_time_sec"],
            "formal_time_cap_sec": FORMAL_TIME_CAP_SEC,
            "budget_beyond_formal_cap_is_exploratory": original[
                "budget_beyond_formal_cap_is_exploratory"
            ],
            "many_small_cones": original["many_small_cones"],
            "structure": original["structure"],
        }
        problem = Path(row["problem"])
        if not problem.is_file():
            raise FileNotFoundError(problem)
        if sha256(problem) != row["problem_sha256"]:
            raise ValueError(f"Problem hash changed: {problem}")
        rows.append(row)
    if len(rows) != 75 or len({row["instance"] for row in rows}) != 75:
        raise ValueError("Expected exactly 75 unique instances")
    rows.sort(
        key=lambda row: (
            -float(row["assigned_pdsdp_time_sec"]),
            -int(row["structure"]["dense_projection_work_proxy"]),
            row["instance"],
        )
    )
    return {
        "schema_version": 1,
        "campaign": "pdsdp-cuda-full75-2x-v294-gpu0-1-20260914",
        "created_at_unix": time.time(),
        "baseline_version": source["baseline_version"],
        "baseline_reason": source["baseline_reason"],
        "budget_rule": source["budget_rule"],
        "formal_interpretation": source["formal_interpretation"],
        "tolerance": TOLERANCE,
        "max_iterations": MAX_ITERATIONS,
        "physical_gpus": gpus,
        "scheduling": "dynamic shared queue, descending assigned budget",
        "solver_grace_sec": SOLVER_GRACE_SEC,
        "validation_grace_sec": VALIDATION_GRACE_SEC,
        "source_sha256": source_identity(),
        "summary": {
            "instances": len(rows),
            "sum_cardal_solver_time_sec": sum(
                float(row["cardal_solver_time_sec"]) for row in rows
            ),
            "sum_assigned_pdsdp_time_sec": sum(
                float(row["assigned_pdsdp_time_sec"]) for row in rows
            ),
            "many_small_cone_instances": sum(row["many_small_cones"] for row in rows),
            "exploratory_beyond_3600_instances": sum(
                row["budget_beyond_formal_cap_is_exploratory"] for row in rows
            ),
        },
        "instances": rows,
    }


def newest_record(campaign_dir: Path, instance: str) -> tuple[Path | None, dict | None]:
    records = sorted((campaign_dir / "runs" / instance).glob("r*/record.json"))
    if not records:
        return None, None
    path = records[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def next_attempt(campaign_dir: Path, instance: str) -> tuple[Path | None, dict | None]:
    path, record = newest_record(campaign_dir, instance)
    if record is not None and record.get("complete"):
        return None, record
    attempts = sorted((campaign_dir / "runs" / instance).glob("r[0-9][0-9][0-9]"))
    attempt = campaign_dir / "runs" / instance / f"r{len(attempts) + 1:03d}"
    return attempt, None


def classify(record: dict) -> dict:
    native = record.get("native_result") or {}
    validation = record.get("validation") or {}
    solver_time = native.get("solver_time_sec")
    assigned = float(record["assigned_pdsdp_time_sec"])
    within_assigned = (
        isinstance(solver_time, (int, float))
        and math.isfinite(solver_time)
        and solver_time <= assigned + max(1.0, assigned * 1e-3)
    )
    native_converged = bool(
        record.get("solver_return_code") == 0 and native.get("status") == "CONVERGED"
    )
    independently_strict = bool(validation.get("strict_at_tolerance", False))
    accepted = native_converged and independently_strict and within_assigned
    within_formal = bool(
        isinstance(solver_time, (int, float)) and solver_time <= FORMAL_TIME_CAP_SEC
    )
    formally_accepted = accepted and within_formal
    cardal = record["cardal"]
    cardal_time = float(cardal["solver_time_sec"])
    solve_wall = record.get("solver_process_wall_sec")
    cardal_strict = bool(cardal.get("independently_strict", False))
    return {
        "native_converged": native_converged,
        "independently_strict": independently_strict,
        "within_assigned_budget": within_assigned,
        "within_formal_3600_sec": within_formal,
        "pdsdp_accepted_assigned_budget": accepted,
        "pdsdp_formally_accepted": formally_accepted,
        "new_formal_solve_over_cardal": formally_accepted and not cardal_strict,
        "same_quality_solver_time_faster": bool(
            formally_accepted
            and cardal_strict
            and isinstance(solver_time, (int, float))
            and solver_time < cardal_time
        ),
        "same_quality_end_to_end_faster": bool(
            formally_accepted
            and cardal_strict
            and isinstance(solve_wall, (int, float))
            and solve_wall < cardal_time
        ),
        "solver_time_speedup_over_cardal": (
            cardal_time / solver_time
            if formally_accepted and isinstance(solver_time, (int, float)) and solver_time > 0
            else None
        ),
        "end_to_end_speedup_over_cardal_solver_time": (
            cardal_time / solve_wall
            if formally_accepted and isinstance(solve_wall, (int, float)) and solve_wall > 0
            else None
        ),
    }


def enrich_record(record: dict, row: dict, gpu: int, attempt: Path) -> dict:
    record.update(
        {
            "instance": row["instance"],
            "attempt": attempt.name,
            "problem": row["problem"],
            "problem_sha256": row["problem_sha256"],
            "many_small_cones": row["many_small_cones"],
            "structure": row["structure"],
            "cardal": row["cardal"],
            "assigned_pdsdp_time_sec": row["assigned_pdsdp_time_sec"],
            "formal_time_cap_sec": FORMAL_TIME_CAP_SEC,
            "execution": {
                "scheduler": "dynamic_shared_queue",
                "physical_gpu": gpu,
                "campaign_driver_sha256": sha256(Path(__file__).resolve()),
            },
            "finished_at_unix": time.time(),
        }
    )
    # A completed experiment includes an exported endpoint and an independent
    # validation record, whether or not that endpoint is strict.
    record["complete"] = bool(
        record.get("complete")
        and record.get("native_result") is not None
        and record.get("validation") is not None
        and record.get("validation_return_code") == 0
    )
    record["classification"] = classify(record)
    return record


def fallback_record(row: dict, gpu: int, attempt: Path, message: str) -> dict:
    record = {
        "schema_version": 1,
        "complete": False,
        "solver_return_code": None,
        "validation_return_code": None,
        "native_result": None,
        "validation": None,
        "campaign_error": message,
    }
    return enrich_record(record, row, gpu, attempt)


def write_progress(campaign_dir: Path, manifest: dict, active: dict[int, str]) -> None:
    records = []
    for row in manifest["instances"]:
        _, record = newest_record(campaign_dir, row["instance"])
        if record is not None and record.get("complete"):
            records.append(record)
    classification = [record["classification"] for record in records]
    summary = {
        "total_instances": 75,
        "complete_records": len(records),
        "remaining": 75 - len(records),
        "active_by_gpu": {str(gpu): name for gpu, name in sorted(active.items())},
        "native_converged": sum(item["native_converged"] for item in classification),
        "independently_strict": sum(
            item["independently_strict"] for item in classification
        ),
        "formally_accepted": sum(
            item["pdsdp_formally_accepted"] for item in classification
        ),
        "same_quality_solver_time_faster": sum(
            item["same_quality_solver_time_faster"] for item in classification
        ),
        "same_quality_end_to_end_faster": sum(
            item["same_quality_end_to_end_faster"] for item in classification
        ),
        "new_formal_solves_over_cardal": sum(
            item["new_formal_solve_over_cardal"] for item in classification
        ),
        "updated_at_unix": time.time(),
    }
    rows = []
    for record in records:
        native = record.get("native_result") or {}
        validation = record.get("validation") or {}
        metrics = validation.get("strict_metrics") or {}
        rows.append(
            {
                "instance": record["instance"],
                "gpu": record["execution"]["physical_gpu"],
                "many_small_cones": record["many_small_cones"],
                "psd_cones": record["structure"]["psd_cone_count"],
                "max_cone_dim": record["structure"]["maximum_psd_cone_dim"],
                "lp_dim": record["structure"]["lp_dim"],
                "cardal_time": record["cardal"]["solver_time_sec"],
                "cardal_strict": record["cardal"]["independently_strict"],
                "assigned_budget": record["assigned_pdsdp_time_sec"],
                "pdsdp_status": native.get("status"),
                "pdsdp_time": native.get("solver_time_sec"),
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
    temporary = campaign_dir / "comparison.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(campaign_dir / "comparison.csv")


def terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class Campaign:
    def __init__(self, campaign_dir: Path, manifest: dict, gpus: list[int]):
        self.campaign_dir = campaign_dir
        self.manifest = manifest
        self.gpus = gpus
        self.queue: queue.Queue[dict] = queue.Queue()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.active: dict[int, str] = {}
        self.children: dict[int, subprocess.Popen] = {}
        self.errors: list[dict] = []

    def update_progress(self) -> None:
        with self.lock:
            write_progress(self.campaign_dir, self.manifest, dict(self.active))

    def request_stop(self, signum: int) -> None:
        print(f"STOP_REQUEST signal={signum}", flush=True)
        self.stop.set()
        with self.lock:
            children = list(self.children.values())
        for process in children:
            terminate_process_group(process)

    def run_one(self, row: dict, gpu: int) -> dict:
        attempt, existing = next_attempt(self.campaign_dir, row["instance"])
        if existing is not None:
            return existing
        assert attempt is not None
        worker_log = (
            self.campaign_dir / "worker_logs" / gpu.__str__() / row["instance"]
            / f"{attempt.name}.log"
        )
        worker_log.parent.mkdir(parents=True, exist_ok=True)
        budget = float(row["assigned_pdsdp_time_sec"])
        command = [
            str(PYTHON),
            str(SINGLE_RUNNER),
            "--problem",
            row["problem"],
            "--output-dir",
            str(attempt),
            "--gpu",
            str(gpu),
            "--time-limit",
            repr(budget),
            "--max-iterations",
            str(MAX_ITERATIONS),
            "--tolerance",
            repr(TOLERANCE),
            "--omp-threads",
            "32",
            "--validator-python",
            str(PYTHON),
            "--eigen-gpu",
            str(gpu),
        ]
        print(
            f"START gpu={gpu} instance={row['instance']} budget={budget:.3f} "
            f"cones={row['structure']['psd_cone_count']} "
            f"max_dim={row['structure']['maximum_psd_cone_dim']}",
            flush=True,
        )
        started = time.monotonic()
        with worker_log.open("wb") as output:
            process = subprocess.Popen(
                command,
                cwd=WORKSPACE,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self.lock:
                self.children[gpu] = process
            status_path = self.campaign_dir / "workers" / f"gpu{gpu}.json"
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "worker_pid": os.getpid(),
                    "child_pid": process.pid,
                    "physical_gpu": gpu,
                    "instance": row["instance"],
                    "attempt": attempt.name,
                    "started_at_unix": time.time(),
                },
            )
            timed_out = False
            try:
                return_code = process.wait(
                    timeout=budget + SOLVER_GRACE_SEC + VALIDATION_GRACE_SEC
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process_group(process)
                return_code = None
            finally:
                with self.lock:
                    self.children.pop(gpu, None)
        record_path = attempt / "record.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record = enrich_record(record, row, gpu, attempt)
            record["campaign_runner_return_code"] = return_code
            record["campaign_runner_external_timeout"] = timed_out
            record["campaign_runner_wall_sec"] = time.monotonic() - started
        else:
            record = fallback_record(
                row,
                gpu,
                attempt,
                "single runner produced no record; "
                f"return_code={return_code}, external_timeout={timed_out}",
            )
            record["campaign_runner_return_code"] = return_code
            record["campaign_runner_external_timeout"] = timed_out
            record["campaign_runner_wall_sec"] = time.monotonic() - started
        atomic_json(record_path, record)
        atomic_json(
            self.campaign_dir / "workers" / f"gpu{gpu}.json",
            {
                "status": "idle",
                "worker_pid": os.getpid(),
                "physical_gpu": gpu,
                "last_instance": row["instance"],
                "last_complete": record["complete"],
                "updated_at_unix": time.time(),
            },
        )
        result = record.get("native_result") or {}
        print(
            f"DONE gpu={gpu} instance={row['instance']} complete={record['complete']} "
            f"status={result.get('status')} strict="
            f"{record['classification']['independently_strict']} "
            f"solver_sec={result.get('solver_time_sec')} "
            f"wall_sec={record.get('solver_process_wall_sec')}",
            flush=True,
        )
        return record

    def worker(self, gpu: int) -> None:
        while not self.stop.is_set():
            try:
                row = self.queue.get_nowait()
            except queue.Empty:
                break
            with self.lock:
                self.active[gpu] = row["instance"]
            self.update_progress()
            try:
                record = self.run_one(row, gpu)
                if not record.get("complete"):
                    with self.lock:
                        self.errors.append(
                            {"instance": row["instance"], "gpu": gpu, "error": "incomplete"}
                        )
            except Exception as error:
                print(
                    f"ERROR gpu={gpu} instance={row['instance']} "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                with self.lock:
                    self.errors.append(
                        {
                            "instance": row["instance"],
                            "gpu": gpu,
                            "type": type(error).__name__,
                            "error": str(error),
                        }
                    )
            finally:
                with self.lock:
                    self.active.pop(gpu, None)
                self.queue.task_done()
                self.update_progress()
        atomic_json(
            self.campaign_dir / "workers" / f"gpu{gpu}.json",
            {
                "status": "stopped" if self.stop.is_set() else "complete",
                "worker_pid": os.getpid(),
                "physical_gpu": gpu,
                "finished_at_unix": time.time(),
            },
        )

    def run(self, limit: int | None) -> int:
        pending = []
        for row in self.manifest["instances"]:
            _, record = newest_record(self.campaign_dir, row["instance"])
            if record is None or not record.get("complete"):
                pending.append(row)
        if limit is not None:
            pending = pending[:limit]
        for row in pending:
            self.queue.put(row)
        self.update_progress()
        threads = [
            threading.Thread(target=self.worker, args=(gpu,), name=f"gpu{gpu}")
            for gpu in self.gpus
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.update_progress()
        progress = json.loads((self.campaign_dir / "progress.json").read_text())
        remaining = progress["summary"]["remaining"]
        atomic_json(
            self.campaign_dir / "campaign_status.json",
            {
                "status": "complete" if remaining == 0 else "partial",
                "pid": os.getpid(),
                "physical_gpus": self.gpus,
                "errors": self.errors,
                "finished_at_unix": time.time(),
                "summary": progress["summary"],
            },
        )
        return 0 if remaining == 0 and not self.errors else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", required=True, nargs="+", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("GPU indices must be unique")
    campaign_dir = args.output.resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (campaign_dir / "campaign.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another driver holds campaign.lock", file=sys.stderr)
        return 2

    manifest_path = campaign_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["physical_gpus"] != args.gpus:
            raise ValueError("Resume GPU list differs from manifest")
    else:
        manifest = make_manifest(args.gpus)
        atomic_json(manifest_path, manifest)
    if args.prepare_only:
        print(json.dumps(manifest["summary"], sort_keys=True))
        return 0

    atomic_json(
        campaign_dir / "campaign_status.json",
        {
            "status": "running",
            "pid": os.getpid(),
            "physical_gpus": args.gpus,
            "started_at_unix": time.time(),
        },
    )
    campaign = Campaign(campaign_dir, manifest, args.gpus)
    signal.signal(signal.SIGTERM, lambda signum, _frame: campaign.request_stop(signum))
    signal.signal(signal.SIGINT, lambda signum, _frame: campaign.request_stop(signum))
    return campaign.run(args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
