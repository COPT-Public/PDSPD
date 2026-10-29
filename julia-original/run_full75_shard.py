#!/usr/bin/env python3
"""Run a disjoint static shard inside an existing full-75 campaign."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import run_full75_campaign as campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--instances", required=True, nargs="+")
    args = parser.parse_args()

    campaign_dir = args.output.resolve()
    manifest = json.loads(
        (campaign_dir / "manifest.json").read_text(encoding="utf-8")
    )
    rows_by_name = {row["instance"]: row for row in manifest["instances"]}
    unknown = sorted(set(args.instances) - set(rows_by_name))
    if unknown:
        raise ValueError(f"Unknown instances: {unknown}")

    worker_dir = campaign_dir / "workers" / f"gpu{args.gpu}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    status_path = worker_dir / "status.json"
    shard_source = campaign.sha256(Path(__file__).resolve())
    base_status = {
        "schema_version": 1,
        "pid": os.getpid(),
        "physical_gpu": args.gpu,
        "instances": args.instances,
        "shard_source_sha256": shard_source,
        "started_at_unix": time.time(),
    }
    campaign.atomic_json(status_path, {**base_status, "status": "running"})

    errors: list[dict[str, str]] = []
    for name in args.instances:
        campaign.atomic_json(
            status_path,
            {
                **base_status,
                "status": "running",
                "current_instance": name,
                "errors": errors,
                "updated_at_unix": time.time(),
            },
        )
        try:
            record = campaign.run_instance(
                campaign_dir, rows_by_name[name], args.gpu
            )
            record_path = (
                campaign_dir
                / "runs"
                / name
                / record["attempt"]
                / "record.json"
            )
            record["execution"] = {
                "scheduler": "static_disjoint_shard",
                "physical_gpu": args.gpu,
                "worker_pid": os.getpid(),
                "shard_source_sha256": shard_source,
            }
            campaign.atomic_json(record_path, record)
        except Exception as error:  # the sequential driver will retry later
            errors.append(
                {
                    "instance": name,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            print(
                f"ERROR instance={name} {type(error).__name__}: {error}",
                flush=True,
            )

    campaign.atomic_json(
        status_path,
        {
            **base_status,
            "status": "complete" if not errors else "partial",
            "errors": errors,
            "finished_at_unix": time.time(),
        },
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
