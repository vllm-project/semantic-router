# Official Qwen3 0.6B: TRAIN-only external-teacher KL arm

Status: **prospective, before optimizer update**. This one-arm experiment tests
whether reliable TRAIN-only teacher distributions improve the existing native
Decision 0.6B student. It is development evidence, not a release score. The
teacher is never a weight initializer. No new TRAIN examples are added.

## Fixed source and matched control

- Student start: official `Qwen/Qwen3-0.6B-Base`, revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; complete local source
  files are checked against pinned per-file SHA-256 in `external_teacher.py`.
- Control: archived official-Qwen full-model, shared-head, hard-label
  `ce_brier` run. Its final/BEST SELECT is **562/700**, family macro accuracy
  **0.7725925925925926**, Brier **0.14263445265**. Control provenance SHA-256
  `d3d5291bbe8989de5cddc0d3615b52571ea3915fcd220da88fbd7916366b8ec3`;
  final SELECT metrics SHA-256
  `7a588b60a7bb95b99c8a265548c6ad2e5f0412b83760584be3111be37edfe253`.
  Reuse this control; do not rerun it.
- Original rights-clean v2 TRAIN 7,455 SHA-256
  `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`;
  SELECT 700 SHA-256
  `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`;
  CAL 700 SHA-256
  `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a`.
  TRAIN, SELECT and CAL are isolated by existing partition checks. This arm
  does not consume CAL labels and never consumes keyed DEV, CSS, formal or
  public benchmark labels during selection.
- Private external-teacher TRAIN Choice/Noul distribution SHA-256
  `8cf211e5af88920de556dc84aa0fcb8b16677fdfd5209abb04db0ffe8e8b195c`.
  The artifact includes original row identity, native input hash, option keys,
  source/model/revision and rights-manifest provenance; all are validated before
  any TRAIN row is mutated. Private row vectors stay outside public artifacts.

## Single fixed treatment

Attach unchanged teacher probability vectors only to Choice/Noul TRAIN rows
where the teacher has a **unique** top option matching the original TRAIN gold,
gold probability is at least **0.7**, and the row is outside the known weak
`legacy:stage4-general-composition-v2` source. The rule uses TRAIN gold only.
The resulting fixed eligible roster is **3,690 rows**, Choice **1,841** and
Noul **1,849**; ordered eligible-roster SHA-256 is
`0e7aeee17f2fd78acbbfd710921fb98cebb992a8558e49a32485065eeff842c5`.
The other Choice/Noul rows and all **516 Score** rows retain the control's
hard-label loss only. This source-aware mask avoids the weak arithmetic and
automaton slices rather than tuning eligibility on SELECT.

All 7,455 examples retain the control's one-epoch exposure: 4,094,489 native
tokens, maximum observed encoded length 6,559, max length 8,192, one GPU,
BF16 capable, full-model shared head, 466 optimizer steps, microbatch 1,
accumulation 16, head dimension 256, AdamW weight decay 0.01, backbone LR
2e-5, head LR 2e-4, warmup 0.05, gradient checkpointing, seed 20260926,
`ce_brier` with Brier weight 0.5. Only change: add teacher-to-student KL at
fixed weight **0.05** to eligible rows; no temperature search, no extra rows,
no altered prompt or native Choice/Noul/Score output. Planned SELECT
checkpoints: steps **64, 128, 192, 256, 320, 384, 448, 466**. Selection is
by the existing family-macro rule, with the final step separately reported.

## Preflight and stop gates

1. CPU preflight must match every source/data/teacher SHA and validate every
   original TRAIN row, option key, source, 4,094,489 token exposure, 466-step
   plan and Score hard-label-only mask. The completed CPU receipt reports PASS;
   receipt SHA-256
   `3869864a5ef3ddcd1e1c87fa3110d6c2cc1af9233bf6b16131ac88c60227503e`.
2. Recheck exact idle device and live containers, reserve only one available
   GPU, and pin runtime image digest. No source file may change after this
   prereg without a new prospective record. Overall GPU budget is at most
   **0.5 GPU-hours**, including zero-step and one-step preflights. Stop on
   unavailable memory, timeout, nonfinite loss or source mismatch; retain
   failure evidence and do not silently retry or change sampler/checkpoint.
3. Native zero-step inference must process all 700 SELECT rows from the
   student start and match the archived hard-label control's baseline
   prediction file SHA-256
   `e62731df7a8b9f35c6ecda17aa763b1dd30700522b3cd8c082f8063f2829f5c3`:
   same IDs/order, prompt/token hashes and candidate domain, **0** top-category
   changes and maximum probability drift **≤1e-4**.
4. A separate one-step trial must produce finite total/CE/Brier/KL, actual KL
   exposure for an eligible row, and a checkpoint that reloads to reproduce
   the first 32 in-process SELECT native outputs with **0** category changes
   and maximum probability drift **≤1e-4**. A failure ends the arm.
5. Only then run the single frozen 466-step arm within the remaining cap.
   SELECT is the only outcome read. The treatment advances beyond SELECT only
   if its fixed-rule BEST checkpoint has **≥562/700** correct and family macro
   accuracy **≥0.7725925925925926**, and the overall GPU/resource/integrity
   gates pass. Below either threshold is HOLD; no DEV/CSS/formal/JevBench run.
   Passing SELECT licenses a later separately frozen evaluation, not a claim
   of independent or released improvement.

Report per-type SELECT counts and Brier, Score retention, chosen checkpoint,
valid/invalid rates, elapsed GPU-hours, output/provenance hashes and all stop
reasons. Do not report a teacher audit score as a student score. Training data
source and downstream weight rights remain subject to separate release audit.

## Local code identity at preregistration

Local branch `xunzhuo/decision2-autojev-kl06-train-arm`, code HEAD before this
document `d086e73fbda1b2d24c7133e1fd666c3c38205db4`. Runtime core SHA-256:

| File | SHA-256 |
|---|---|
| `training/model/train.py` | `0111bbfd0e6a372661a88c44e0c716c18a94b78f651c0662fa2b18bbe59ad96d` |
| `training/model/external_teacher.py` | `ccca48bcca3adc20068499a2e8c350b81f9d28d0a0004a2a4a7569aef78ac437` |
| `training/model/loss.py` | `c6fc39edbf03cd4dfad3f97911ffa0c6bc3b28808c373548fd1db27429b9bcaa` |
| `research/autojev_kl06_preflight.py` | `4d2f311c9213c0423d265f7f979b9bd94207382eb508d5ecfb2c7f15e6ff1e97` |
| `research/autojev_kl06_zero_compare.py` | `a3c1a2db71ed522b062dfcfc430ec73be6d04d736b9c97926634ca6fc9a8a637` |
| `research/qwen3_06_checkpoint_smoke.py` | `c63da3c0c1c58efc8ea14aee6a2a73ef5f7ab56b77788b69b2aa23f9fabd1ec7` |
