# 9B M10 state (running log, newest first; prereg `lux9b-m10-prereg-2026-10-02.md`)

Index values stay private (node private run directories and the coordinator's private folder); this file has none.

## 2026-10-02 12:45Z (20:45 UTC+8), M10 continuation worker

- **Measured, once each** (values private):
  - KIB4-a33, KSW-a33 and KX-a33 pass the Index gate (lower bound > 0 vs K-a13IB-bf16).
  - KIBM-a33 and KIBM-a40 do not; the KIBM family is closed.
- **Release choice: KIB4-a33** (largest lower bound).
  - Formal path done (typed FINAL: no collapsed type), release inputs, card Index input and assets, spec and
    decision.
  - Its prerelease on node A GPU7 passed (build, examples, card, parity, `verify_bundle`) against the card4 base.
- **Base moved twice.**
  - The organization rename made Lux `main` the card-only revision `6af07f36`. The ops now supersede it: spec from
    `dev2-9b-org.json`, the org gate and decision, `vllm-sr` repo and collection. Integration merged at `e0c92f51a`.
  - Runtime phase A's Lux runtime-only `release.sh` started at 12:27Z on node A. A successor must carry its runtime
    (spec `dev2-9b-ra.json` and its frozen cache `runtime-a-9b`), so the KIB4-a33 upload waits until that release
    and its branch's merge into integration. Then the spec is re-derived, the prerelease rerun and the upload made.
- **Index runs in flight:**
  - Node C: KIB4-a40, then X1-a33 (KX + KIB4) and X2-a33 (KX + KIB).
  - Node B, GPU2 / 4: X3-a33 (KIB4 + KIB).
  - Node A, GPU1–6: X5-a33 (KIB4 + KX + KSW; the three best arms by lower bound, amendment 5).
  - X4 equals X1, because the two best arms are KIB4 and KX.
- **Extra seeds (amendment 6)** run on node B GPU3 / 6 / 7. `post.sh KIB4` with `M10_POST_NAME=KIB4P` builds the
  three-seed KIB4 soup when KIB4-s3 ends.
- **Budget:** KXP is not built, because the 85 GPU-h stop rule leaves room for one more Index run (KIB4P-a33).
  KX-s4 / s5 are spent unmeasured unless budget frees.
- **GPU-h:** ≈ 65 used: training 42.2 (node A 8.9, node B 33.3), Index 20.7, formal and prerelease ≈ 2. About 14 more
  are committed to running work.

## 2026-10-02 10:15Z (18:15 UTC+8), M10 continuation worker

- **Built (node B, CPU):** the KIBM soup (3 seeds) and the KIB4 soup (2 seeds), each with its a33 / a25 / a40
  points. The KSW soup (2 seeds) is building, and so is X3-a33 (amendment 5: KIB4 + K-a13IB's KIB arm soup).
- **Index runs** (mirror `58bad21d9`, panel-7, base K-a13IB-bf16):
  - Node A, GPU3–5: M10-KIB4-a33-bf16, with parity running.
  - Node C, greedy over GPU1–7, shared with 4B runs: M10-KIBM-a33-bf16.
  - Node B, GPU2 / 4 (leases handed to `track=eval-ix1`; IX1 environment copied from node C, SHA-256 lists equal):
    M10-KSW-a33-bf16, then M10-X3-a33-bf16, once their BF16 packages are copied.
  - KIBM's first parity on node A was aborted when the queue was reordered to put KIB4 first, and it was moved to
    `parity-aborted/`. KIBM is measured on node C instead, once.
- **Speculative formal:** M10-KIB4-a33 on node A GPU6 (`m10/formal.sh`), started 10:08Z.
- **Extra seeds** (amendment 6, COORDINATOR WATCHDOG 18:00): node B GPU3 runs KIB4-s3, GPU6 / 7 run KX-s4 / s5.
  All three started 10:04Z, and their preflights run in the chain. KX entered node B's lock with node A's hashes. The
  `prep.sh kx-lock` log line reads "on node A" because that text is fixed; the lock was written on node B.
- **Node A KX-s1–s3** (GPU1 / 2 / 7) end ≈ 11:20Z. Then the KX soup is built, KX-a33's Index runs on node A, and its
  formal starts.
- **GPU-h:** ≈ 37 (node B training 26.2, node A KX ≈ 5, Index ≈ 6.9 on KUP). Of the 90 total, ≈ 30 are committed
  ahead: 3 seeds ≈ 9, 4 Index runs ≈ 11, 2 formals ≈ 4, X points ≈ 5.

## 2026-10-02 09:35Z (17:35 UTC+8), M10 continuation worker

- **KUP dropped** (COORDINATOR UPDATE 17:25): M10-KUP-a25-bf16 is measured (scored, both bootstraps) and does not
  pass the gate, like KUP-a33. **M10-KUP-a40-bf16 was stopped unmeasured** at 09:22Z (4 of 7 shards running, 0.9
  run GPU-h spent); node A GPU3–6 leases released.
- **Speculative formal path** (17:25): `m10/formal.sh` runs M9's `formal.sh` (unchanged runner since K-a13IB's formal
  mirror `787abdc54`) for a point next to its Index run, on any node A GPU under lease `track=9b-m10`
  (`M9_FORMAL_GPUS` / `M9_FORMAL_TRACK` / `M9_A_GPUS` / `M9_LEASE_TRACK`, defaults unchanged). Node A only: the
  comparators, CAL698 inputs, frozen formal cache and Lux 1.0 package are node A paths. Planned: KIB4-a33 when the
  KIB4 soup is built (≈ 10:15Z), KX-a33 when KX's is (≈ 11:25Z). `ix.sh ckcopy` moves a shipped FP32 point C → A.
- Integration merged (includes `cd565a588`).
- **GPU-h at 09:20Z:** training 22.8 (node B) + 3.8 (node A, running); Index 3.3 + 2.7 + 0.9 (KUP).

## 2026-10-02 09:05Z (17:05 UTC+8), worker 7e1c9ce8 (handoff; see `m10-handoff-2026-10-02.md`)

- **M10-KUP-a33-bf16 measured** on node C: parity passed, 7 shards, 120,224 rows merged (2 unsupported), both
  bootstraps run, 3.29 run GPU-h. Its summaries are in the private folder. It does **not** pass the gate.
- M10-KUP-a25-bf16 is running on node A (5 of 7 shards done). M10-KUP-a40-bf16 is next on node A.
- All training is unchanged and on schedule.

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
