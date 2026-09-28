# Milestone 3 operational wrappers (verbatim records)

Exact copies of the shell/Python wrappers executed on the nodes during decoder
Milestone 3, kept as text so the job sequence is reviewable.

- Every job they started also left a launch receipt beside its output: full docker
  argv, image, commit, start/end UTC and exit status.
- Formal runs left the eval runner's `COLLECT.json` / `GPU-TIME.json` instead.
- Node aliases resolve from a private file; no address or credential appears here.

`m3-n4lk.sh` is the launcher whose command ran twice. That launch was aborted
before any training step (see the results record). `m3-n4lkr.sh` is its
lock-guarded replacement. `m3-e8v.sh` was stopped before any E8V job and
superseded by the queues in `m3-lux.sh`.
