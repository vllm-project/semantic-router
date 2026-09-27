# Qwen3.5-9B Base initialization contrast: technical HOLD

This is the outcome of the [prospectively frozen Base versus Posttrained
initialization contrast](qwen35-9b-base-init-contrast-prereg-2026-09-28.md).
The 458-update official Base arm **did not complete**. Its process exited 139
after update 106, with a kernel `pt_autograd_0` segmentation fault. It did not
reach a SELECT-chosen complete checkpoint or the prespecified BEST reload and
typed DEV/CSS pilot gate. No CAL, typed FINAL, CSS FINAL, public JevBench, or HF
release evaluation was run for this arm. Its step-64 SELECT receipt is a
diagnostic partial-run artifact, never a candidate score.

## Frozen inputs and admission

- Source: official `Qwen/Qwen3.5-9B-Base@68c46c4b3498877f3ef123c856ecfde50c39f404`;
  the read-only Posttrained control is the separately finished
  `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` arm.
  Both use the exact signed trainer at `3d29a41f6` and the 7,324-row
  group-filtered rights-clean TRAIN SHA-256
  `fe9c419a3e751e4a4173b5c90c49e0a83bba7e1c35c683441bb036138acb971c`.
- The official tokenizers differ as files, but the native segmented-option
  encoder emitted identical ordered token IDs on every frozen TRAIN, SELECT,
  and CAL row; this controls the actual training-input tokenizer confound.
- Zero-step native SELECT: 700/700 valid, 254/700 correct, family macro
  accuracy 0.322101 and normalized Brier 0.347913.
- Independent one-update preflight: finite optimizer update, 700/700 SELECT
  valid, 239/700 correct, family macro 0.334758 and Brier 0.493900.
  Fresh-process 32-row reload covered 13 Choice, 14 Noul, and 5 Score inputs:
  zero category changes and zero p99/maximum probability drift. This was a
  technical admission check, not a quality result or full-arm initializer.

## Full-arm failure and decision

| Item | Observation |
| --- | --- |
| Frozen target | One fresh official-Base 458-update arm; same optimizer, seed, data, length 4,096, checkpoints and SELECT rule as the Posttrained control |
| Execution | 2026-09-27 18:05:49–18:17:01 UTC, 672 seconds on one accelerator |
| Last complete update | 106/458; no `COMPLETE.json` |
| Exit | 139 (`SIGSEGV`); kernel reported `pt_autograd_0` and Python segmentation faults near 18:16:55 UTC, without an application-level traceback or recorded OOM |
| Only scheduled SELECT reached | Update 64: 700/700 evaluated, 519/700 correct, family macro 0.694074, Brier 0.162969 |
| Final BEST reload / development gate | **Not run**, because the required complete arm does not exist |
| Required action | **HOLD** this Base initialization cell; do not resume, repeat it as if identical, or promote the partial checkpoint |

Source and data identities were checked before launch. The failure is a native
backward/runtime fault, not evidence that Base underperforms the Posttrained
source on the full matched contrast. It also does not invalidate the
Posttrained arm's separately observed development result. Diagnosis should
isolate the faulting operator/input on a distinct, preregistered technical
cell before any new full optimizer run. The own Lux1 development receipt of
approximately 70.326 remains a historical same-input reference; the released
Lux package has an unresolved parity qualification and is not a current
matched release comparator.

## Time and evidence

| GPU stage | Duration | One-GPU hours |
| --- | ---: | ---: |
| Zero-step | 55 s | 0.015278 |
| Independent one-update | 88 s | 0.024444 |
| One-update fresh reload | 31 s | 0.008611 |
| Failed full arm | 672 s | 0.186667 |
| **Total** | **846 s** | **0.235000** |

The private execution directory retains all logs and partial checkpoints.
Publicly safe identity receipts: full-arm provenance SHA-256
`21a544b82a3b98a5bcb5a123ae37d5802ecf5fa301e2ce0e03572ea8fdc6d94c`,
train metrics SHA-256
`10ff2bd1934304350e68e12751f9d70eef91709569a66599192dfd7435024e79`,
full console log SHA-256
`59e2aecf8e2c6d684598d0cbd994c305cd38c9ed83834ba2673f5bd559f13ded`,
and step-64 SELECT predictions SHA-256
`0fff70d7f2707ad1ecf4272ffdd122b58c588baee4e57f87aa97df4c8ced388f`.
The complete source revision, partition hashes, budget, and stop criteria remain
in the linked preregistration. No scores from this partial run enter model
selection or publication tables.
