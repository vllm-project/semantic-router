# Lux 1.0 v3 control: prospective runtime repair r2

**LOCKED before any r2 GPU use.** The r1 attempt stopped during ROCm device
initialization with no model weights loaded, no typed FINAL/CSS15 answer, no
gold read and no score. Its full failed receipt remains separate. This r2
changes only the GPU visibility environment and adds two gold-free gates; it
does not choose a model, checkpoint, answer threshold or evaluator after
seeing formal performance. It is a technical repair, not a rescoring of r1.

All fixed identities, exact 8,147 v3 original items, 8,547 answer slots,
current Lux package, native collector, sealer, scorers, model revision, input
hashes, qualified image and post-key disclosure are as in
`lux9b-current-package-v3-control-prereg-2026-09-28.md`. They do not change.
The local source commit for the r1 protocol is `0abc76a93`; the r1 stop
receipt SHA-256 is
`469a5301edba4d20e0089d21cc7285a1c3893fcdda42b01cb18fd6d8e57ec039`.

## One fixed technical intervention

- Choose the already live-confirmed idle physical GPU index 7. The r1
  container set both `HIP_VISIBLE_DEVICES=7` and
  `ROCR_VISIBLE_DEVICES=7`. In r2, set **only** `ROCR_VISIBLE_DEVICES=7`;
  do not set `HIP_VISIBLE_DEVICES`. Inside the container the single exposed
  device is addressed as `cuda:0`. All Docker device mounts, image, package,
  source and prompts remain the same. If no single MI325X device is visible,
  stop r2; do not try a second mask or device.
- Before model load, run one bounded gold-free image probe requiring
  `torch.cuda.is_available()`, `torch.cuda.device_count() == 1`, and the
  exposed device architecture `gfx942`. The probe has a 60-second cap.
- Then run the unchanged native collector on the already frozen two-request,
  five-answer current-package example *requests* under the r2 mask. Compare
  its answers to the previously sealed current-package first process with
  unchanged `jev_arena/verify_lux9b_current_package.py` SHA-256
  `37abe239c5b48c335e980539180c1e4f47db9f29161a06dc63bc6faf08426ee8`.
  Require five complete valid answers, exact package/runtime and zero
  category changes with maximum numeric drift at most `1e-6`. Do not read
  the stale published answer fields. This native gate has a 120-second cap.
- Only if both gates pass, collect complete typed FINAL and CSS15 once each
  with the same mask, no gold mount and the previous fixed source and panel
  hashes. A missing, malformed or over-budget answer counts as wrong; a
  process crash or missing row ends the r2 run incomplete. Seal both entire
  prediction files and create a joint gold-free freeze before reading either
  key. Then score once. No alternate mask, model, image, checkpoint,
  calibration or prompt and no retry after an r2 failure.

The total cap is **0.6 GPU-hour / 36 minutes elapsed** for one device,
including probe, native parity and both panels. Record exact UTC times,
GPU-seconds, model/source/image/input/prediction/report digests and failure
cause. Keep raw inputs, outputs, protected keys, logs, host paths and container
details private. The same-panel score, if obtained, is explicitly post-key;
it cannot be described as a fresh independent test or transferred to a 2.0
weight package. JevBench public231 remains separate and is not part of this
r2 GPU run.
