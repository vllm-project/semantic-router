# Own-Nox 4B Choice-weight screen: runtime-corrected v2

**New prospective attempt after a failed v1 gate.** The previous arm
[failed zero-step parity](nox4b-choice-human-weight-preflight-result-2026-09-27.md)
and produced only a one-update, noncausal smoke. None of its checkpoint or
scores enters this v2 protocol. The historical own-Nox clean-v2 control
remains read-only.

The hypothesis, exact weighted 2,240-row Choice cohort (1.5×), source,
TRAIN/SELECT/CAL bytes, model initialization, 466-update schedule,
fixed step-128 early observation, SELECT and DEV/CSS pilot thresholds,
and 1.0-GPU-hour cap are **unchanged** from the
[original preregistration](nox4b-choice-human-weight-prereg-2026-09-27.md).
This v2 changes only the effective runtime import path and execution order
needed to make zero-step parity testable before optimization. It is not a
threshold relaxation or a re-selection of data or weight.

## Runtime identity and pre-optimizer sequence

Use the same pinned runtime image
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`
as the completed control, with the image's original
`PYTHONPATH=/opt/decision-fla` intact and workdir at the exact read-only
code mirror. Python adds that workdir to `sys.path`, so both
`training.model.train` and optimized `fla` import without overriding
`PYTHONPATH`. Before any GPU inference, log and compare the effective
`PYTHONPATH`, `training.model.train.__file__`,
`fla.__file__`, image ID, package versions and key source hashes.
Missing optimized `fla` is a STOP.

The local treatment trainer SHA-256 is
`349fd6b47dc961930bd6b3bea63a66dc34f1a516877babb478a36468521967b5`.
It adds a `--zero-step-only` mode that runs the trainer's ordinary fresh
source/load/encode/SELECT baseline path and exits before `model.train()`,
backward or optimizer step. The full candidate uses the **same source code**
without that flag. Verify the mirror hash before either process. The
SELECT baseline source receipt remains
`eabd788e4e656974e3f380cdff0d32f1ad0887ddb2034235b2c43b9c3f994de3`.

Run the following **sequentially**, not concurrently:

1. Freshly confirm the assigned physical GPU is unoccupied, correctly
   reserved, and the container exposes exactly one BF16 GPU. Use only
   `ROCR_VISIBLE_DEVICES` for the assigned physical index; do not combine
   multiple overlapping ROCm visibility filters. Check all source/data
   file SHA-256 values against v1, 7,455/700/700 rows, 4,194,465 TRAIN
   tokens, 6,596 maximum length and exactly 2,240 weighted Choice rows.
2. Launch **only** `--zero-step-only` in a fresh v2 probe output. Wait for
   exit code zero and verify zero optimizer entries or checkpoints. Then
   compare all 700 predictions against the archived control: ordered IDs,
   native prompt/token hashes, task types, option domains, zero categorical
   changes and maximum absolute option probability drift `<=1e-4`.
   Write a private PASS/FAIL receipt with both prediction hashes. FAIL
   means stop this v2 attempt; no one-step smoke is authorized.
3. Only after an independent PASS receipt, run a separate one-step
   numerical smoke with `max-steps=1` in a new directory. It must have
   finite loss/gradients, a nonzero number of weighted Choice examples,
   exact first-window input-token count (5,131), one durable checkpoint
   and projected step-128 cost <=1.0 GPU-hour. It is never a model
   comparison because its LR schedule differs.
4. Only after that gate, start the fresh **466-step planned-horizon**
   treatment with the original v1 flags and save interval 128. At the
   durable checkpoint128, stop externally. Compare its SELECT receipt to
   the archived matched checkpoint128 using the original fixed thresholds:
   GoEmotions Choice >=150/200, all SELECT >=574/700, family macro
   >=.805185, Score >=62/90, complete validity. Only a pass opens one
   same-native, uncalibrated DEV1600/CSS pilot1430 paired diagnostic for
   both saved step128 weights. All original advancement thresholds apply.

An exact parity pass is necessary but does not prove broad transfer. No
calibration, formal v3 panel, protected labels, public benchmark or HF
publication is authorized by this short screen. Preserve all failed output
and GPU-hour receipts. In particular, never reinterpret the v1 failure as
a success after observing v2.
