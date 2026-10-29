# Parallelization amendment

At 2026-09-14 15:56 CST, after 60 of 75 records had completed, the user
authorized using additional GPUs. All physical GPUs 0--5 and 7 were confirmed
idle immediately before launch. The active `1tc.2048` solve was allowed to
finish on GPU 6.

The sequential scheduler (PID 2085022) was paused, without stopping its active
child, so it could not collide with the static shards. The remaining instances
other than `1tc.2048` were assigned disjointly as follows:

- GPU 0: `diamond_patch`
- GPU 1: `shmup5`
- GPU 2: `G60mc`
- GPU 3: `G60_mb`
- GPU 4: `torusg3-15`
- GPU 5: `foot`, `p_auss2_3.0`, `ice_2.0`, `G55mc`
- GPU 7: `biomedP`, `G48_mb`, `G59mc`, `checker_1.5`, `G48mc`

Each instance still has exclusive use of one H100 and retains exactly the
budget frozen in `manifest.json`. No solver, stopping, export, validation, or
classification setting changed. Every sharded `record.json` includes its
physical GPU and the shard scheduler hash.

The resume supervisor restores the original sequential scheduler after all
shards terminate. That scheduler performs the delayed validation for
`1tc.2048`, incorporates every completed shard into the aggregate files, and
automatically retries any shard that failed before producing a complete
record.

Scheduler source identities:

- `run_full75_shard.py`:
  `44dccb7339d5eb9aa2b87a4042728dbf45dff5352d919f4233c9074113e92d2f`
- `resume_full75_after_shards.py`:
  `ab21fbc8b6273978ee71d563f4dfcafd1dbe4ca3cccae99d833770a79ed78ac5`

Live worker state and logs are under `workers/gpu*/` and the resume state is
in `workers/resume_supervisor.json`.

## Resume guard update

At 2026-09-14 16:13 CST, an unrelated VLLM job began using physical GPUs 3--6
after this campaign's shards on GPUs 3--5 had already completed. The original
resume supervisor was replaced before it resumed the sequential scheduler.
The active guard, `resume_full75_when_gpu_free.py`, requires both all shard
workers to terminate and physical GPU 6 to have no compute processes. This
prevents the delayed `1tc.2048` validation and any retries from overlapping the
unrelated workload. Its live state is `workers/resume_when_gpu_free.json`.
