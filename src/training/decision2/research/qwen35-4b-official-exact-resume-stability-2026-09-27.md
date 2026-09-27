# Official Qwen3.5 4B: exact continuation through update 128

The single
[preregistered continuation](qwen35-4b-official-exact-resume-after-probe-2026-09-27.md)
loaded the original update-64 adapter, optimizer state, data cursor and random
state under the same source/data/code contract. It moved past both earlier
native-failure boundaries (updates 84 and 125) and completed update 128 on
the newly reserved accelerator. At the handoff inspection, the process was
still running, Docker reported `OOMKilled=false`, and no new matching native
kernel fault appeared. This is **short-run infrastructure stability**, not a
completed training arm or release result.

The update-128 checkpoint directory contains the metadata, trainer state,
adapter and decision head. Metadata says `step=128`; `LATEST.json` and
`BEST.json` both point to `checkpoint-0000128`; no incomplete checkpoint
directory was present. SHA-256 for those four files in that order:

```text
ebfedf5380a3719cc59996c55a405de4d0da786842aa5afc0593309feea42c0c
9b0941fbb2ae76b433e1c51858c447e25fd5ea8bd5cb082562f4f120c5dae359
bce6331b9d91be64537f440c5903faef5284615974237047786f897c3bb3450a
a31e381266bec17b2b7c66835e4447946f649eff9316b696b5aba99ef2fd5434
```

The update-128 SELECT result is 568/700, family macro accuracy 0.76722;
this is an opened **development** reading. The original SELECT selector
currently prefers update 128, but later preregistered checkpoints may replace
it. No DEV, CSS pilot, public benchmark or formal panel was used here.

The 4B process remains under its original total 3.0 GPU-hour cap and one-failure
HOLD rule. Continue monitoring its native exit, finite losses and durable
checkpoints through the fixed update-466 goal. If it fails again, stop further
automatic resumes. If it completes, the exact-reload and full development
gate still precede any post-key same-panel formal evaluation.
