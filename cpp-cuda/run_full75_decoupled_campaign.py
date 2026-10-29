#!/usr/bin/env python3
"""Run the Mittelmann campaign with independent solver and validator queues.

GPU slots are released as soon as ``run_benchmark.py --skip-validation``
returns.  Completed checkpoints are validated by a bounded CPU-only pool, so
long independent validation never prevents the next GPU solve from starting.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import run_benchmark as single
import run_full75_campaign as base


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/DATA/disk2/wyc/conda/envs/sdp-cu128/bin/python")
VALIDATOR = ROOT / "validate_cuda_checkpoint.py"


@dataclass(frozen=True)
class ValidationTask:
    row: dict
    gpu: int
    attempt: Path


@dataclass(frozen=True)
class ExternalTask:
    instance: str
    gpu: int
    pid: int


def pid_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return state != "Z"


class DecoupledCampaign:
    def __init__(
        self,
        campaign_dir: Path,
        manifest: dict,
        gpus: list[int],
        validation_workers: int,
        external: list[ExternalTask],
    ) -> None:
        self.campaign_dir = campaign_dir
        self.manifest = manifest
        self.gpus = gpus
        self.validation_workers = validation_workers
        self.external = {task.instance: task for task in external}
        self.rows = {row["instance"]: row for row in manifest["instances"]}
        self.solve_queue: queue.Queue[dict] = queue.Queue()
        self.validation_queue: queue.Queue[ValidationTask | None] = queue.Queue()
        self.stop = threading.Event()
        self.external_done = threading.Event()
        if not external:
            self.external_done.set()
        self.state_lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.active_solvers: dict[int, str] = {}
        self.active_validators: dict[str, str] = {}
        self.children: dict[str, subprocess.Popen] = {}
        self.errors: list[dict] = []
        self.driver_sha256 = base.sha256(Path(__file__).resolve())

    def execution(self, gpu: int, scheduler: str) -> dict:
        return {
            "scheduler": scheduler,
            "physical_gpu": gpu,
            "campaign_driver_sha256": self.driver_sha256,
        }

    def finalize_record(
        self, record: dict, row: dict, gpu: int, attempt: Path, scheduler: str
    ) -> dict:
        record = base.enrich_record(record, row, gpu, attempt)
        record["execution"] = self.execution(gpu, scheduler)
        return record

    def update_progress(self) -> None:
        with self.progress_lock:
            with self.state_lock:
                active = dict(self.active_solvers)
                validators = dict(self.active_validators)
                external = {
                    name: {"gpu": task.gpu, "pid": task.pid}
                    for name, task in self.external.items()
                }
                errors = list(self.errors)
            base.write_progress(self.campaign_dir, self.manifest, active)
            progress = json.loads(
                (self.campaign_dir / "progress.json").read_text(encoding="utf-8")
            )
            base.atomic_json(
                self.campaign_dir / "decoupled_status.json",
                {
                    "schema_version": 1,
                    "status": "stopping" if self.stop.is_set() else "running",
                    "pid": os.getpid(),
                    "physical_gpus": self.gpus,
                    "active_solvers_by_gpu": {
                        str(gpu): name for gpu, name in sorted(active.items())
                    },
                    "active_cpu_validators": validators,
                    "queued_validations": self.validation_queue.qsize(),
                    "external_handoff": external,
                    "errors": errors,
                    "summary": progress["summary"],
                    "updated_at_unix": time.time(),
                },
            )

    def request_stop(self, signum: int) -> None:
        print(f"DECOUPLED_STOP_REQUEST signal={signum}", flush=True)
        self.stop.set()
        with self.state_lock:
            children = list(self.children.values())
        for process in children:
            base.terminate_process_group(process)

    def reconcile_existing(self) -> None:
        """Enrich old raw records and queue solver-only checkpoints."""
        for row in self.manifest["instances"]:
            path, record = base.newest_record(self.campaign_dir, row["instance"])
            if path is None or record is None:
                continue
            attempt = path.parent
            gpu = int((record.get("command") or {}).get("physical_gpu", 0))
            validation = record.get("validation")
            native = record.get("native_result")
            if native is not None and validation is None:
                record["complete"] = False
                base.atomic_json(path, record)
                if row["instance"] not in self.external:
                    self.validation_queue.put(ValidationTask(row, gpu, attempt))
                continue
            if validation is not None and record.get("validation_return_code") == 0:
                record = self.finalize_record(
                    record, row, gpu, attempt, "handoff_or_existing"
                )
                base.atomic_json(path, record)

    def prepare_queues(self) -> None:
        self.reconcile_existing()
        for row in self.manifest["instances"]:
            name = row["instance"]
            if name in self.external:
                continue
            _, record = base.newest_record(self.campaign_dir, name)
            if record is None:
                self.solve_queue.put(row)
            elif record.get("complete"):
                continue
            elif record.get("native_result") is not None:
                # The corresponding checkpoint is already in the validation queue.
                continue
            else:
                self.solve_queue.put(row)

    def solver_worker(self, gpu: int) -> None:
        while not self.stop.is_set():
            try:
                row = self.solve_queue.get_nowait()
            except queue.Empty:
                break
            name = row["instance"]
            attempt, existing = base.next_attempt(self.campaign_dir, name)
            if existing is not None:
                self.solve_queue.task_done()
                continue
            assert attempt is not None
            budget = float(row["assigned_pdsdp_time_sec"])
            command = [
                str(PYTHON),
                str(base.SINGLE_RUNNER),
                "--problem",
                row["problem"],
                "--output-dir",
                str(attempt),
                "--gpu",
                str(gpu),
                "--time-limit",
                repr(budget),
                "--max-iterations",
                str(base.MAX_ITERATIONS),
                "--tolerance",
                repr(base.TOLERANCE),
                "--omp-threads",
                "32",
                "--validator-python",
                str(PYTHON),
                "--skip-validation",
            ]
            worker_log = (
                self.campaign_dir
                / "worker_logs_decoupled"
                / f"gpu{gpu}"
                / name
                / f"{attempt.name}.log"
            )
            worker_log.parent.mkdir(parents=True, exist_ok=True)
            with self.state_lock:
                self.active_solvers[gpu] = name
            self.update_progress()
            print(
                f"SOLVE_START gpu={gpu} instance={name} budget={budget:.3f}",
                flush=True,
            )
            started = time.monotonic()
            with worker_log.open("wb") as output:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                with self.state_lock:
                    self.children[f"solve:{gpu}"] = process
                timed_out = False
                try:
                    return_code = process.wait(timeout=budget + base.SOLVER_GRACE_SEC)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    base.terminate_process_group(process)
                    return_code = None
                finally:
                    with self.state_lock:
                        self.children.pop(f"solve:{gpu}", None)
            record_path = attempt / "record.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record["complete"] = False
                record["campaign_solver_runner_return_code"] = return_code
                record["campaign_solver_external_timeout"] = timed_out
                record["campaign_solver_runner_wall_sec"] = time.monotonic() - started
                base.atomic_json(record_path, record)
            else:
                record = base.fallback_record(
                    row,
                    gpu,
                    attempt,
                    "decoupled solver produced no record; "
                    f"return_code={return_code}, external_timeout={timed_out}",
                )
                base.atomic_json(record_path, record)
            native = record.get("native_result")
            if return_code == 0 and native is not None:
                self.validation_queue.put(ValidationTask(row, gpu, attempt))
            else:
                with self.state_lock:
                    self.errors.append(
                        {
                            "stage": "solver",
                            "instance": name,
                            "gpu": gpu,
                            "return_code": return_code,
                        }
                    )
            print(
                f"SOLVE_DONE gpu={gpu} instance={name} "
                f"status={(native or {}).get('status')} "
                f"solver_sec={(native or {}).get('solver_time_sec')} "
                "gpu_released=True",
                flush=True,
            )
            with self.state_lock:
                self.active_solvers.pop(gpu, None)
            self.solve_queue.task_done()
            self.update_progress()
        base.atomic_json(
            self.campaign_dir / "workers" / f"gpu{gpu}.json",
            {
                "status": "stopped" if self.stop.is_set() else "solver_queue_complete",
                "worker_pid": os.getpid(),
                "physical_gpu": gpu,
                "finished_at_unix": time.time(),
                "scheduler": "decoupled_solver_validation_queues",
            },
        )

    def run_validation(self, task: ValidationTask, worker_index: int) -> None:
        row, gpu, attempt = task.row, task.gpu, task.attempt
        name = row["instance"]
        checkpoint = attempt / "solution.pdsdpbin"
        output_path = attempt / "validation.json"
        argv = [
            str(PYTHON),
            str(VALIDATOR),
            "--problem",
            row["problem"],
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output_path),
            "--tolerance",
            repr(base.TOLERANCE),
        ]
        base.atomic_json(
            attempt / "validation_command.json",
            {
                "argv": argv,
                "physical_eigen_gpu": None,
                "mode": "cpu_only_decoupled",
            },
        )
        key = f"cpu{worker_index}"
        with self.state_lock:
            self.active_validators[key] = name
        self.update_progress()
        print(f"VALIDATE_START worker={key} instance={name}", flush=True)
        with (attempt / "validation.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with self.state_lock:
                self.children[f"validate:{key}"] = process
            try:
                return_code = process.wait(timeout=base.VALIDATION_GRACE_SEC)
            except subprocess.TimeoutExpired:
                base.terminate_process_group(process)
                return_code = None
            finally:
                with self.state_lock:
                    self.children.pop(f"validate:{key}", None)
        record_path = attempt / "record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        validation = None
        if return_code == 0 and output_path.is_file():
            validation = json.loads(output_path.read_text(encoding="utf-8"))
        record["validation"] = validation
        record["validation_return_code"] = return_code
        record["independently_strict"] = (
            None if validation is None else bool(validation["strict_at_tolerance"])
        )
        artifacts = {}
        for path in sorted(attempt.iterdir()):
            if path.is_file() and path.name != "record.json":
                artifacts[path.name] = single.sha256(path)
        record["artifact_sha256"] = artifacts
        record["complete"] = bool(
            record.get("solver_return_code") == 0
            and record.get("native_result") is not None
            and validation is not None
            and return_code == 0
        )
        record = self.finalize_record(
            record, row, gpu, attempt, "decoupled_solver_validation_queues"
        )
        base.atomic_json(record_path, record)
        if not record["complete"]:
            with self.state_lock:
                self.errors.append(
                    {
                        "stage": "validation",
                        "instance": name,
                        "gpu": gpu,
                        "return_code": return_code,
                    }
                )
        print(
            f"VALIDATE_DONE worker={key} instance={name} "
            f"complete={record['complete']} "
            f"strict={record['classification']['independently_strict']}",
            flush=True,
        )
        with self.state_lock:
            self.active_validators.pop(key, None)
        self.update_progress()

    def validation_worker(self, worker_index: int) -> None:
        # Existing synchronous validators can be exceptionally memory hungry.
        # Let them drain before starting the bounded decoupled CPU pool.
        while not self.external_done.wait(timeout=1.0):
            if self.stop.is_set():
                return
        while not self.stop.is_set():
            task = self.validation_queue.get()
            if task is None:
                self.validation_queue.task_done()
                break
            try:
                self.run_validation(task, worker_index)
            except Exception as error:
                print(
                    f"VALIDATE_ERROR worker=cpu{worker_index} "
                    f"instance={task.row['instance']} "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                with self.state_lock:
                    self.errors.append(
                        {
                            "stage": "validation",
                            "instance": task.row["instance"],
                            "gpu": task.gpu,
                            "type": type(error).__name__,
                            "error": str(error),
                        }
                    )
            finally:
                self.validation_queue.task_done()

    def external_watcher(self) -> None:
        pending = dict(self.external)
        while pending and not self.stop.is_set():
            for name, task in list(pending.items()):
                path, record = base.newest_record(self.campaign_dir, name)
                if record is None or record.get("validation") is None:
                    if pid_running(task.pid):
                        continue
                    # Recover from an inherited wrapper that exited before it
                    # could publish validation.json.  Its solver checkpoint is
                    # still usable, so hand only validation to the CPU queue.
                    if (
                        path is not None
                        and record is not None
                        and record.get("native_result") is not None
                    ):
                        record["complete"] = False
                        base.atomic_json(path, record)
                        self.validation_queue.put(
                            ValidationTask(self.rows[name], task.gpu, path.parent)
                        )
                        print(
                            f"EXTERNAL_REQUEUE_VALIDATION instance={name} "
                            f"gpu={task.gpu} dead_pid={task.pid}",
                            flush=True,
                        )
                        pending.pop(name)
                        with self.state_lock:
                            self.external.pop(name, None)
                        self.update_progress()
                        continue
                    # The wrapper may be between atomic artifact writes; retry.
                    continue
                row = self.rows[name]
                assert path is not None
                record = self.finalize_record(
                    record,
                    row,
                    task.gpu,
                    path.parent,
                    "handoff_external_synchronous",
                )
                record["campaign_runner_return_code"] = 0
                record["campaign_runner_external_timeout"] = False
                base.atomic_json(path, record)
                print(
                    f"EXTERNAL_DONE instance={name} gpu={task.gpu} "
                    f"complete={record['complete']}",
                    flush=True,
                )
                pending.pop(name)
                with self.state_lock:
                    self.external.pop(name, None)
                self.update_progress()
            time.sleep(1.0)
        self.external_done.set()

    def run(self) -> int:
        self.prepare_queues()
        print(
            f"DECOUPLED_START gpus={self.gpus} pending_solves={self.solve_queue.qsize()} "
            f"queued_validations={self.validation_queue.qsize()} "
            f"external={sorted(self.external)}",
            flush=True,
        )
        external_thread = threading.Thread(
            target=self.external_watcher, name="external-watcher"
        )
        external_thread.start()
        validators = [
            threading.Thread(
                target=self.validation_worker, args=(index,), name=f"validator-{index}"
            )
            for index in range(self.validation_workers)
        ]
        for thread in validators:
            thread.start()
        solvers = [
            threading.Thread(target=self.solver_worker, args=(gpu,), name=f"solver-{gpu}")
            for gpu in self.gpus
        ]
        for thread in solvers:
            thread.start()
        self.update_progress()
        for thread in solvers:
            thread.join()
        self.solve_queue.join()
        external_thread.join()
        self.external_done.set()
        self.validation_queue.join()
        for _ in validators:
            self.validation_queue.put(None)
        for thread in validators:
            thread.join()
        self.update_progress()
        progress = json.loads(
            (self.campaign_dir / "progress.json").read_text(encoding="utf-8")
        )
        remaining = progress["summary"]["remaining"]
        status = "complete" if remaining == 0 and not self.errors else "partial"
        base.atomic_json(
            self.campaign_dir / "campaign_status.json",
            {
                "status": status,
                "pid": os.getpid(),
                "physical_gpus": self.gpus,
                "scheduler": "decoupled_solver_validation_queues",
                "errors": self.errors,
                "finished_at_unix": time.time(),
                "summary": progress["summary"],
            },
        )
        print(
            f"DECOUPLED_DONE status={status} remaining={remaining} "
            f"errors={len(self.errors)}",
            flush=True,
        )
        return 0 if status == "complete" else 1


def parse_external(value: str) -> ExternalTask:
    try:
        instance, gpu, pid = value.rsplit(":", 2)
        return ExternalTask(instance, int(gpu), int(pid))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "external must be INSTANCE:GPU:PID"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", required=True, nargs="+", type=int)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--external", action="append", type=parse_external, default=[])
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("GPU indices must be unique")
    if args.validation_workers < 1:
        parser.error("validation-workers must be positive")
    campaign_dir = args.output.resolve()
    lock_stream = (campaign_dir / "campaign.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another driver holds campaign.lock", file=sys.stderr)
        return 2
    manifest = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["physical_gpus"] != args.gpus:
        raise ValueError("Resume GPU list differs from manifest")
    campaign = DecoupledCampaign(
        campaign_dir,
        manifest,
        args.gpus,
        args.validation_workers,
        args.external,
    )
    signal.signal(signal.SIGTERM, lambda signum, _frame: campaign.request_stop(signum))
    signal.signal(signal.SIGINT, lambda signum, _frame: campaign.request_stop(signum))
    base.atomic_json(
        campaign_dir / "campaign_status.json",
        {
            "status": "running",
            "pid": os.getpid(),
            "physical_gpus": args.gpus,
            "scheduler": "decoupled_solver_validation_queues",
            "validation_workers": args.validation_workers,
            "started_at_unix": time.time(),
        },
    )
    return campaign.run()


if __name__ == "__main__":
    raise SystemExit(main())
