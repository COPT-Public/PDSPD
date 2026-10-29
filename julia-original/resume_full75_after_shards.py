#!/usr/bin/env python3
"""Resume the paused sequential campaign when all static shards terminate."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


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
    parser.add_argument("--gpus", required=True, nargs="+", type=int)
    args = parser.parse_args()

    output = args.output.resolve()
    own_status = output / "workers" / "resume_supervisor.json"
    started = time.time()
    while True:
        worker_states = {}
        all_terminal = True
        for gpu in args.gpus:
            path = output / "workers" / f"gpu{gpu}" / "status.json"
            if not path.is_file():
                worker_states[str(gpu)] = "missing"
                all_terminal = False
                continue
            status = json.loads(path.read_text(encoding="utf-8"))
            state = status.get("status", "unknown")
            pid = status.get("pid")
            if state == "running" and isinstance(pid, int) and not process_exists(pid):
                state = "dead"
            worker_states[str(gpu)] = state
            if state not in {"complete", "partial", "dead"}:
                all_terminal = False

        atomic_json(
            own_status,
            {
                "status": "waiting" if not all_terminal else "resuming",
                "pid": os.getpid(),
                "resume_pid": args.resume_pid,
                "worker_states": worker_states,
                "started_at_unix": started,
                "updated_at_unix": time.time(),
            },
        )
        if all_terminal:
            if process_exists(args.resume_pid):
                os.kill(args.resume_pid, signal.SIGCONT)
                result = "resumed"
            else:
                result = "resume_process_missing"
            atomic_json(
                own_status,
                {
                    "status": result,
                    "pid": os.getpid(),
                    "resume_pid": args.resume_pid,
                    "worker_states": worker_states,
                    "started_at_unix": started,
                    "finished_at_unix": time.time(),
                },
            )
            return 0 if result == "resumed" else 1
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
