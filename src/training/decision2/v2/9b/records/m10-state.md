# 9B M10 state (running log, newest first; prereg `lux9b-m10-prereg-2026-10-02.md`)

Index values stay private (node private run directories and the coordinator's private folder); this file has none.

## 2026-10-02 08:15Z (16:15 UTC+8), worker 7e1c9ce8

- **Amendment 4 (arm KX)** committed `c8ef116f9` before any KX row was built, then mirrored to nodes A and B.
  The KX TRAIN was built on node B (CPU). The base build `4f12c94b…` has 156,687 rows: 60,364,987 native tokens
  (IB share .2149) with a multilingual share of .3285. The ML block adds 19,472 `~m2` copies (12,429 groups, 33
  languages, 8,761,244 tokens), giving TRAIN `f1d9ecf8…` / teacher `2a7ac626…`: 176,159 rows and 69,126,231 tokens.
  The multilingual share is .4136, x60's .4136. The data went node B → C → A (relay removed), was re-hashed and
  locked on node A (`READY-m10.json` arm KX).
- Node A's Lux zero-step member (`m9-KIB-s1-zero`) has the same SHA-256 list as node B's (list `6b1d31f3…`). The
  Triton cache `f83b1d10` was copied from node A's M9 cache.
- **Training, node A:** KX-s1 (pre-warm) started 07:57Z on GPU1. KX-s2 / s3 started 08:03Z on GPU2 / 7, and all three
  passed preflight. They should finish ≈ 11:15–11:25Z (≈ 15% more tokens than K-a13IB). The KX post chain waits on
  node A.
- **Training, node B:** unchanged. KIBM-s3 is on GPU3, KIB4-s1 / s2 on GPU6 / 7 and KSW-s1 / s2 on GPU2 / 4 (all
  ≈ 10:05–10:30Z). Node B M10 GPU-h was 14.4 at 07:38Z.
- **Index:** KUP-a33 runs on node C, greedy over pool GPU1–7 (parity passed; shards placing). KUP-a25 / a40 were
  restaged on node C (`cedbf102…` / `79da5e4d…`) and copied to node A (`ix.sh pkgcopy`). Panel-7 and the reference run
  were copied C → A. A second chain runs on node A pool GPU3–6 (parity passed for both; shards placing). The first
  node A launch used the short names `KUP-a25` / `KUP-a40`, so the IX1 launcher printed usage and exited 2. Nothing
  was measured. The relaunch used the registered names `M10-KUP-a25-bf16` / `M10-KUP-a40-bf16`.
- **`ix.sh ship`** now takes `M10_FROM=a` for KX's soups, which are built on node A.
- **Budget (90 GPU-h):** ≈ 30 for node B training plus labeling, ≈ 10 for KX, and ≈ 2.7 per Index run. If all 15
  points are measured, the total is ≈ 80.

## 2026-10-02 07:45Z (15:45 UTC+8), worker 7e1c9ce8

- **Done:** KUP-s1 / s2 / s3 and KIBM-s1 / s2 (07:0x–07:21Z). KUP soup (`61b5bf75…`) and points built on node B:
  KUP-a33 `e5f3dd51…`, KUP-a25 `ff73f997…`, KUP-a40.
- **KSW labeling** done (two shards, ≈ 15 min each; agreement with gold: choice .91, noul .87, score .63);
  `teacher-sd.jsonl` `bb710e91…`; KSW locked; KSW-s1 / s2 started 07:38Z on GPU2 / 4. KIB4-s1 / s2 started 07:21Z on
  GPU6 / 7. KIBM-s3 runs on GPU3.
- **Index:** KUP-a33 shipped to node C, BF16 copy `6bcff807…` restaged (loaded 7,940,895,744, T = 1; manifest
  `f0f09ba5…`). The node C chain (mirror `29a52205d`) uses greedy shard placement over the shared pool GPU1–7
  (`M10_SHARDS=7`), since the pool never had seven idle GPUs at once. KUP-a25 / a40 ship next.
- Node B M10 GPU-h ≈ 14.4 at 07:38Z.

## 2026-10-02 06:45Z (14:45 UTC+8), worker 7e1c9ce8

- Seeds at updates 1,495–1,543 (06:37Z); no failure. Index GPUs: node C GPU4 / 5 idle, GPU1–3 / 6–7 and node E's
  fast lane busy with other IX1 runs (node E until ≈ 10:10Z by its leases); the M10 chain takes node C GPUs as they
  go idle.

## 2026-10-02 06:20Z (14:20 UTC+8), worker 7e1c9ce8

- Seeds at updates 1,194–1,266 (06:17Z), ≈ 13–14 per minute; no failure. KUP seeds end ≈ 07:15Z, KIBM-s1 / s2
  ≈ 07:35Z.

## 2026-10-02 05:55Z (13:55 UTC+8), worker 7e1c9ce8

- **Arm (d) KIB4** (amendment 3; IB4 phase 1 `76cea510`, TRAIN `6045b456…` checked against its `final.json`):
  TRAIN `2e72bcfd…` / teacher `377f8878…` locked (151,039 rows, 99,545 x60, 60,332,200 native tokens, IB share .1898;
  6,808 `sentfin` rows out). Phase-2 chains `b6-p2` / `b7-p2` (KIB4-s1 / s2) wait for GPU6 / 7's flocks.
- KSW / KIB4 are two-seed arms; the labeling chain, post-KSW and post-KIB4 run from mirror `c07aee766`.
- A KIB4 release needs the C1 recheck r3 PASS (IB4 is release-safe pending C1).

## 2026-10-02 05:45Z (13:45 UTC+8), worker 7e1c9ce8

- Seeds at update 570–724 of 2,083 (KUP) / 2,365 (KIBM), ≈ 15 updates per minute: KUP ends ≈ 07:05–07:20Z, KIBM-s1 /
  s2 ≈ 07:35Z. No failure.
- KSW labeling now waits for GPU2 / 4 only (mirror `8266426aa`; two shards), so those GPUs go straight from KUP to
  labeling and KSW seeds; KSW-s3 follows KIBM-s1 on GPU6. GPU7's phase-1 chain idles its lease when KIBM-s2 ends.

## 2026-10-02 05:15Z (13:15 UTC+8), worker 7e1c9ce8

- **Arm (c) KSW** (amendments 1 / 2): data built (TRAIN `e5cc44bb…`, 146,600 rows, 60,272,054 native tokens, IB share
  .1567); teacher K-a13IB rebuilt on node B with identity `4701ba41…` (second build; the first differed only in the
  member paths recorded in `decision_config.json`). The labeling chain (`ksw.sh teach`, mirror `7ff21a506`) waits
  for the GPU2 / 4 / 6 / 7 phase-1 flocks, then labels, locks KSW and starts phase-2 seeds on GPU2 / 4 / 6.
- **Post chains relaunched** from `7ff21a506` for KUP, KIBM and KSW (the 04:52Z launch passed the mirror as a full
  path, which the chain does not resolve; nothing had been built).
- Seeds' full runs started ≈ 04:50Z; expected end ≈ 07:25Z (KIBM-s3 ≈ 10:05Z, KSW seeds ≈ 10:30Z).
- Arm (d): the IB4 record has no release-safe phase yet (amendment 2 of IB4, re-audit pending).
- Integrity notes for the gate: KUP's TRAIN is K-a13IB's (audited for its release); KSW's TRAIN rows are a subset
  of K-a13IB's; KIBM adds IB3-r2 `mqa` rows, which need the row-level Index audit if KIBM passes.

## 2026-10-02 04:55Z (12:55 UTC+8), worker 7e1c9ce8

- **Training (node B, image `f83b1d10`, leases `track=9b-m10`):** chains launched 04:40Z from mirror `240c8ca79`.
  - KUP-s1 GPU3 (pre-warm done 04:45Z, preflight PASS 04:46:53Z), KUP-s2 GPU2, KUP-s3 GPU4, KIBM-s1 GPU6,
    KIBM-s2 GPU7: zero-step and one-step done 04:47–04:50Z, full runs in progress. KIBM-s3 is queued on GPU3 after
    KUP-s1.
  - Expected ≈ 2.6 GPU-h per seed: the first five end ≈ 07:30Z, KIBM-s3 ≈ 10:10Z.
- **Data locked** in `$M/data/READY-m10.json`: KUP weights `ae696d93…` (x60 weight share .677 → .758); KIBM TRAIN
  `aadca52f…` / teacher `b4cfc3be…` (151,375 rows, 60,391,810 tokens, IB share .1815; 6,808 `sentfin` rows out, 8,752
  IB3-r2 `mqa` rows in).
- **Lux member pinned:** M10's zero-step checkpoint (KUP-s1) is byte-identical to M9's `m9-KIB-s1-zero` (10 weight
  files; `$M/inputs/lux-zero-m9-KIB-s1.sha256`), K-a13IB's Lux member.
- **Post chains (node B, CPU) launched 04:52Z from mirror `ed69b9657`:** `post.sh KUP` / `post.sh KIBM` wait for their
  seeds, then build the 3-seed soup and the a33 / a25 / a40 points.
- **Index measurement:** node C (package `DEV2.0-9B-e51f9881`, panel-7, reference run `K-a13IB-bf16` local); the pool
  GPU1–7 is busy with other Index runs until ≈ 08:10Z. Driver `m10/ix.sh` (ship → bf16 → chain → status → fetch).
- **GPU-h:** ≈ 1 so far (5 seeds started); projection ≈ 16 training + ≈ 2.7 per Index run.
- **Next:** at ≈ 07:30Z ship KUP's points to node C; BF16 copies; Index chain on the free node C GPUs; then KIBM.
  Arms (c) 9B swap and (d) + IB4 are amendments, not yet written.
