# 9B Milestone 6, amendment 1: the AutoJev wave's prompt format (before any training step and before any target existed)

Amends the [preregistration](lux9b-m6-prereg-2026-09-30.md) (`f41402e68`). Written 2026-09-30 ~02:45 UTC+8. No training
step, development readout or teacher target exists yet.

## What happened

- The first wave attempt (chain `m6-aj`, 18:08–18:22Z) ran cleanly up to conversion:
  - the target guard passed;
  - both shards exited 0 (10,635 + 10,918 rows, every prompt answered), kept the autotune digest `cff772eb…` and used
    the FLA path;
  - the 128-prompt production repeat check passed.
- `v2.data.m3.teacher_targets` then refused the receipts ("teacher answered a different prompt"). No target file was
  written, KA's teacher file was not built, and the chain stopped.

## Cause

- M6's split step wrote the wave prompts as canonical JSON, with sorted keys.
- The qualified collector digests each prompt in its stored key order. The converter re-derives that digest from the
  training row's native prompt in its natural key order, which is the production waves' `prompt_line` format.
- The two orders differ, so the per-row attestation fails by design. The collector's answers were not read.

## Fix and amended rule

- **Fix** (commit in the state file): the split step now writes prompt lines with `v2.data.m3.waves.prompt_line`,
  byte-identical to the production format. A unit test binds each line's collector digest to its row. S, the wave rows
  and the length cap are unchanged: the rebuilt `S.ids.txt` and `wave.rows.jsonl` must hash as before (`4fa9063b…`,
  `ff975f38…`).
- **The prereg's stop rule** ("if the wave or its conversion fails, KA stops and is not retried") is amended as follows:
  - A failure of the qualified runtime still stops KA without retry: the guard, the qualification identity, an autotune
    change, a collector error or the repeat check.
  - A fault in M6's own wave-input writer that the converter catches before any target exists is fixed by a new commit.
    The wave is then repeated **once** from the fixed inputs.
- **Records and budget:** the first attempt stays on node A as `m6/data/m6-split-r1` and `m6/aj-wave-r1`. Its GPU-hours
  (two shards 0.408 + the repeat check) count against the 24 GPU-h cap; `m6_gpu_hours` counts every `aj-wave*` attempt.
- Nothing else changes: arms, seeds, rules, finalists and caps are as preregistered.
