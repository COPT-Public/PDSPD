#!/usr/bin/env python3
"""Resume a paused campaign after its shards finish and target GPU is free."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


TERMINAL_WORKER_STATES = {"complete", "partial", "dead"}


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def gpu_compute_pids(gpu: int) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if value.isdigit():
            pids.append(int(value))
    return pids


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-pid", required=True, type=int)
    parser.add_argument("--resume-gpu", required=True, type=int)
    parser.add_argument("--gpus", required=True, nargs="+", type=int)
    args = parser.parse_args()

    output = args.output.resolve()
    own_status = output / "workers" / "resume_when_gpu_free.json"
    started = time.time()
    while True:
        worker_states = {}
        all_terminal = True
        for gpu in args.gpus:
            path = output / "workers" / f"gpu{gpu}" / "status.json"
            if not path.is_file():
                state = "missing"
            else:
                status = json.loads(path.read_text(encoding="utf-8"))
                state = status.get("status", "unknown")
                pid = status.get("pid")
                if (
                    state == "running"
                    and isinstance(pid, int)
                    and not process_exists(pid)
                ):
                    state = "dead"
            worker_states[str(gpu)] = state
            if state not in TERMINAL_WORKER_STATES:
                all_terminal = False

        blocking_gpu_pids = gpu_compute_pids(args.resume_gpu)
        ready = all_terminal and not blocking_gpu_pids
        common = {
            "pid": os.getpid(),
            "resume_pid": args.resume_pid,
            "resume_gpu": args.resume_gpu,
            "blocking_gpu_pids": blocking_gpu_pids,
            "worker_states": worker_states,
            "started_at_unix": started,
        }
        atomic_json(
            own_status,
            {
                **common,
                "status": "resuming" if ready else "waiting",
                "updated_at_unix": time.time(),
            },
        )
        if ready:
            if process_exists(args.resume_pid):
                os.kill(args.resume_pid, signal.SIGCONT)
                result = "resumed"
            else:
                result = "resume_process_missing"
            atomic_json(
                own_status,
                {
                    **common,
                    "status": result,
                    "finished_at_unix": time.time(),
                },
            )
            return 0 if result == "resumed" else 1
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
