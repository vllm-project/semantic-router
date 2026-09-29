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

## Data freeze (appended 2026-09-30 ~03:00 UTC+8, before any training step)

- **Repeat wave** (chain `m6-aj2`, mirror `7380a3cbf`, 18:31–18:45Z): guard PASS; shards 10,635 + 10,918 rows, exit 0,
  autotune digest `cff772eb…` unchanged, FLA path; repeat check 128 / 128 identical answers, drift 0.0.
  - Targets `249b1906d25a1f2f85c8013ff22415cba723c2ce5918c353e81d4676f2c93037` (21,553 rows).
  - Attestation `5e6f19d8…`; report `f54ac115…`; prompts `5a9182ab…` (rows `ff975f38…`, as before).
- **KA teacher** `3f0aabe0cccdde1af6a79b4d0a926dce47f4666a8494fdfa98dc29ea2a030356` (122,651 rows; manifest `65bf8fa3…`;
  train.jsonl `a66131b1…`, byte-identical to x60).
  - AutoJev on all 30,792 S rows: production `aj-a0s-strict` 1,107, AJ-M 6,004, AJ-SL 2,128, M6 wave 21,553.
  - Own-Lux on the other 91,859 rows.
- **Agreement with gold on S** (argmax, report only):

  | Teacher | All | Choice | Noul | Score |
  | --- | ---: | ---: | ---: | ---: |
  | AutoJev-27B | .608 | .892 | .840 | .481 |
  | own-Lux | .585 | .851 | .823 | .460 |

- **AutoJev GPU-hours:** 0.868 over both attempts (0.424 + 0.444), within the 1.5 cap.
