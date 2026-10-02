# 9B M10 state (running log, newest first; prereg `lux9b-m10-prereg-2026-10-02.md`)

Index values stay private (node private run directories and the coordinator's private folder); this file has none.

## 2026-10-02 19:50Z (10-03 03:50 UTC+8), M10 continuation

- **Gate verdicts vs `M10-KIB4-a40-bf16`** (values private):
  - `M10-KIB4W2-a40-bf16`: **FAIL**, significantly below.
  - `M10-KIB4L2-a40-bf16`: **FAIL**. It is significantly below even K-a13IB; its bootstrap vs KIB4-a40 is
    finishing.
  - `M10-X7-a40-bf16`: scored. Its point is level with KIB4-a40, so a FAIL is expected; the gate bootstrap is
    running on node B.
- **Amendment 10** (`6c0c3d195`): the factory's α ladder. KIB4-a50 runs on node A GPU1–4 (linked, model
  `319ca812…`; BF16 copy on node A). KIB4Q-a60 and KF-a60 are dropped.
- **Amendment 11:** KIB4W3-a40 is dropped, Y1 / Y2 become conditional, and no extra KIB4-family arm is planned (see
  the record).
- Node C GPU1–4 and node B GPU7 were released when their chains finished.
- **GPU-h (continuation):** ≈ 12.0 used (Index: X7 2.70, X8 2.70, W2 2.79, L2 2.77; parity gates ≈ 0.6; formals
  0.47), plus KIB4-a50 running (≈ 2.7).

## 2026-10-02 19:05Z (10-03 03:05 UTC+8), M10 continuation

- **Gate verdicts vs `M10-KIB4-a40-bf16`** (IF1; values private):
  - `AF-KF-a40-bf16` (the factory's nine-seed KIB4-family soup at α = 2/5): **FAIL**, significantly below the current
    release.
  - `M10-X8-a40-bf16` (KIB4 + KSW at α = 2/5): **FAIL**. Its lower bound is ≤ 0, though its point delta is
    positive. That meets amendment 7's condition for planning one more KIB4-family arm, which the W2 / L2 results
    will inform first.
- **KF-a50 dropped before measurement.** It is the same nine-seed soup as KF-a40, and α 1/2 vs 2/5 has moved
  KIB-family points by about 0.1 (K-a12IB vs K-a13IB, KIB4-a40 vs KIB4-a33). It cannot plausibly pass. Its parity
  gate (node A) had already run; the node A chain was restarted for KIB4L2-a40 alone.
- **Index:** X7-a40 is at shard 6 of 7 (node B GPU7), KIB4W2-a40 at shards 4–6 (node C), and KIB4L2-a40 at shards 0–3
  (node A GPU1–4).
- **Arm factory:** its amendment 6 stages an α ladder for the 9B publisher (KIB4-a50, KIB4Q-a50 / -a60, KF-a60).
  Its node B batch-2 soups KIB4W3-a40 / KIB4R-a40 wait for their seeds (≈ 19:40Z).
- **GPU-h (continuation):** ≈ 9.5 used, ≈ 12.2 with the running shards.

## 2026-10-02 18:25Z (10-03 02:25 UTC+8), M10 continuation

- **Arm-factory hand-off** (COORDINATION 02:00 / 02:22). The factory measured only `AF-KF-a40-bf16`, and M10 now
  measures its other 9B points on node A GPU1–4. **Amendment 9** (`a35efba33` with the IX1 entries; mirror on A, B and
  C) sets the order: KF-a40 (gate) → KIB4W2-a40 / KIB4L2-a40 → KF-a50 → the batch-2 points KIB4W3-a40 / KIB4R-a40 →
  Y1 / Y2 → KFxKIB-a40 (last, budget permitting).
- **Index:**
  - X8-a40 is scored on node C (120,224 rows + 2 unsupported), and its bootstraps are running.
  - X7-a40 is at shard 5 of 7 on node B GPU7.
  - KIB4W2-a40 is on node C GPU1–4 (shards 0–3). The chain was restarted for W2 alone, and KIB4L2-a40 moves to node A.
  - On node A GPU1–4: KIB4L2-a40 (package C → A), then KF-a50 (linked from the factory, model `2a1c8624…`).
  - AF-KF-a40 is scored on node A, and its gate bootstraps vs KIB4-a40 are running.
- **Formal:** AF-KF-a40 done (node A GPU7, 17:43–17:57Z, exit 0; choice / noul / score OK). Node A GPU7 is now the
  4B owner's (COORDINATION 02:00).
- **GPU-h (continuation):** ≈ 7.0 (Index: X8 2.70, X7 2.3 so far, W2 0.5 so far; parity 0.4; formal 0.5).

## 2026-10-02 17:55Z (10-03 01:55 UTC+8), M10 continuation

- **Node E went to the user** (COORDINATION 01:25), during X8-a40's run. Shards 0–2 had finished (exit 0) and
  shards 3–4 were partial. The coordinator moved the shard directories to node F.
  - Shards 0–2 were relayed F → A → C, with equal SHA-256 lists. The partial shards 3 / 4 and their launcher files
    were left out.
  - X8's package was copied B → C. Its parity gate and the remaining shards 3–6 run on node C GPU1–4 (lease
    `eval-ix1`, from the factory's `released` leases; COORDINATION 01:38).
  - This completes X8's single measurement. Nothing is measured twice.
- **Public release policy** (USER 01:11, COORDINATION 01:36): integration merged (`47dd06be7`, the hub visibility
  policy). The Hub unit tests pass (16). The continuation decision text now says public repository and public
  collection.
- **Factory points linked on node A** (`xpts.sh link`, equal lists): `AF-KF-a40` (model `f555d2f7…`), and for
  amendment 8 `KIB4W2-a40` (`9f467afe…`) and `KIB4L2-a40` (`1cacc2c4…`). The last two are shipped to node C for
  their BF16 copies and Index runs.
- **Formal:** AF-KF-a40 on node A GPU7 since 17:43Z (speculative; the factory's KF-a40 Index run started 17:18Z on
  node A GPU1–6).
- **X7-a40:** node B GPU7, shard 3 of 7.
- **GPU-h (continuation):** ≈ 3.0.

## 2026-10-02 17:15Z (10-03 01:15 UTC+8), M10 continuation

- **Amendment 8** (`667202519`): the factory's node A queue measures only KF-a40, KF-a50 and KFxKIB-a40. M10 therefore
  measures the factory's KIB4W2-a40 and KIB4L2-a40 itself, once each, as `M10-KIB4W2-a40-bf16` and
  `M10-KIB4L2-a40-bf16` (IX1 entries `f31039025`; mirror `f31039025` on nodes A, B, C and E).
- **X7-a40 formal done** (node A GPU7, 16:28–16:42Z, exit 0). The typed FINAL is choice / noul / score OK, with no
  type collapsed. The formal path takes about 14 min per point now, not the planned 2 h.
- **Index:** X7-a40 shard 2 of 7 (node B GPU7); X8-a40 shards 3–4 of 7 (node E GPU3 / 7).
- **Reference bootstraps:** X5-a33 and KIB4-a33 were bootstrapped vs KIB4-a40 (CPU, node A). They are not
  candidates, and they calibrate the paired interval width vs the current release. Values are private.
- **Soups on node A for Y1 / Y2:** KIB4-a40 and X5-a33, copied B → C → A with equal lists.
- `m10/gate.sh` and `ix.sh gate` gained `M10_GATE_WAIT=1`, which waits for a run's own scoring and bootstraps.
- **GPU-h (continuation):** ≈ 2.0 (Index 1.7 so far, parity 0.1, formal 0.23).

## 2026-10-02 16:35Z (10-03 00:35 UTC+8), M10 continuation

- **Index runs** (mirror `96c1bd3ff`; parity 86 / 86, max |Δp| 0.0 for both):
  - X7-a40 on node B GPU7, 7 shards in sequence. The arm factory took node B GPU2 / 3 / 4 / 6 at 16:09–16:17Z for
    its second 9B batch (KIB4W3, KIB4R; COORDINATION 00:00). The first node B chain was stopped before any shard
    and restarted for X7 alone; its log records this.
  - X8-a40 on node E GPU3 / 7 (released by dec-m17 at 14:07Z, leased `track=eval-ix1`). Panel-7, the 9B package
    and both reference runs were copied there with equal SHA-256 lists.
  - Each candidate is still measured once. The gate bootstraps vs KIB4-a40 follow scoring (`ix.sh gate`).
- **Formal:** X7-a40 on node A GPU7 since 16:28Z (`m10/formal.sh`; FP32 point copied B → C → A, equal lists). It
  went first because node A GPU7 was idle and may be taken after 15 min (COORDINATION 00:00), and KF-a40 is not
  built before ≈ 18:20Z. KF-a40 follows on the first free node A GPU.
- **Audit `m10c` PASS** (node C CPU, driver `0ae8eaa34`, mirror `96c1bd3ff`): 120,226 Index rows, planted control
  200 / 200 (0 missed). Every set has 0 item rows: KIB (K-a13IB's TRAIN `2cd09292…`, 151,015 lines; 212
  familiar-text rows), KIB4 (151,039; 196), KIB4R (the factory's, 151,088; 210), KSW (146,600; 166) and KX
  (176,159; 189). It covers every continuation candidate, including the points that contain KIB.
- **GPU-h (continuation):** ≈ 0.2 so far (two parity gates; the X7 shards and formal are running).

## 2026-10-02 16:15Z (10-03 00:15 UTC+8), M10 continuation (amendment 7; 30 GPU-h)

- **Amendment 7** (`d802c97c4`): every candidate is now gated vs `M10-KIB4-a40-bf16`. The candidates are the arm
  factory's 9B hand-offs, `Y1` / `Y2` (averages of the best factory point with KIB4-a40 / X5-a33) and the cheap
  weightings `X7-a40` (KIB4 + KX + KSW at α = 2/5) and `X8-a40` (KIB4 + KSW at α = 2/5).
- **Ops** (mirror `96c1bd3ff` on nodes A, B and C): `m10/ix.sh kref | gate | gstatus | gfetch` (gate bootstraps vs
  KIB4-a40, `m10/gate.sh`), `m10/xpts.sh` (averages of points; `link` for arm-factory points); IX1 names
  (`259b6fc2a`, a separate commit to `v2/eval/ix1/launch.sh`). The continuation release ops are in
  `v2/release/records/dev2-9b-m10c-2026-10-03/ops/`: they supersede `f3122c7c`, take KIB4-a40's gate and
  decision, and purge with KIB4-a40's node A package as the node copy.
- **Built on node B (CPU):** X8-a40 (model `310228cb…`) and X7-a40 (building). KX-a40 was copied A → C → B with
  equal lists. KIB4-a40's run is on nodes A and B (`kref`, equal lists).
- **Next:** stage X7 / X8 (BF16 copy, restage), then their Index chain on node B GPU2 / 3 / 4 / 6 / 7 (7 shards,
  greedy) and their gate bootstraps. The factory's 9B seeds end ≈ 17:45–18:05Z, and its KF soups follow. The KF-a40
  formal starts on node A GPU7 as soon as KF-a40 exists.
- **GPU-h (continuation):** 0 so far.

## 2026-10-02 15:50Z (23:50 UTC+8), M10 continuation worker

- **RELEASED: Lux-9B `main` = `f3122c7c8abd302326c22220aac4095eb1799f37`** (KIB4-a40, user 23:02 UTC+8), superseding
  `f77b41f5`; private; K-a13IB's weights purged. Record `v2/release/records/dev2-9b-m10-2026-10-02.md` (with the
  23:20 fast-path waiver note). The upload waited for the 27B prerelease's `release.sh` to end (15:14Z).
- **All M10 Index runs are in** (values private): X2-a33 and KIB4P-a33 pass IF1 vs K-a13IB but are below KIB4-a40.
  Later candidates (arm factory hand-offs) must now beat `M10-KIB4-a40-bf16`.
- **Leases:** node A GPU7 released; node A GPU3 is the arm factory's. KIB4P-a33's release inputs on node A are kept
  (unused).

## 2026-10-02 14:10Z (22:10 UTC+8), M10 continuation worker

- **Index runs in:** KIB4-a40, X1-a33, X3-a33 and X5-a33 all pass IF1 vs K-a13IB-bf16 (values private). Still running:
  X2-a33 (node C GPU1-2), KIB4P-a33 (node B GPU2/3/4/6/7, restarted 13:53Z with 7 shards; the 13:21Z start asked
  for 6 on the 7-shard panel and exited at once).
- **Release choice KIB4-a40** (the highest passer so far), committed at `6fa6693b3`; its formal typed-FINAL run is
  in (no collapsed type), inputs, card Index, assets, spec and decision derived on node A (`make_m10.py --check`
  passes at the integration merge `e842eed8b`). Prerelease running on node A GPU7. The upload waits for KIB4P-a33's
  Index result (a later candidate must beat the released point, so it is measured first); KIB4P-a33's formal run is
  started speculatively on node A GPU3.
- **Integration merged** (`85419d98d`: runtime phase A record, 27B ra spec); no 9B file changed.
- **COORDINATION 21:20 (new KIB4-family arms) not started:** M10's budget is 90 GPU-h (stop rule 85 for Index and
  formal runs) and the projection with everything in flight is ≈ 82; one 3-seed arm is ≈ 8 GPU-h before it can be
  measured. New arms need a follow-on milestone budget; node A GPU1/2/4/5/6 and node C GPU3-7 are released.

## 2026-10-02 13:20Z (21:20 UTC+8), M10 continuation worker

- **Release base moved to the runtime phase A revision `f77b41f5`** (Lux-9B runtime-only, released while the
  KIB4-a33 release waited). Merged `origin/xunzhuo/decision-2-runtime-a`; the M10 ops now derive from
  `specs/dev2-9b-ra.json` with the ra gate receipt and decision, supersede `f77b41f5`, and run panel parity with a
  copy of the frozen cache `release/triton/runtime-a-9b` (digest `0e0b7342…`, 4,881 files, checked on node A).
  KIB4-a33 spec and decision re-derived on node A (decided 12:57:13Z), `make_m10.py --check` passes at `a27aa81ac`;
  card assets unchanged. Prerelease running on node A GPU7 (the stale card4-base decision on node A was moved aside).
- **KIB4-a40 Index run finished; it passes the Index gate.** Its speculative formal run started 13:08Z on node A GPU3
  (FP32 point copied from node C). The release candidate is chosen once X1 / X2 / X3 / X5 and the formal run are in.
- **KX-s4 / KX-s5 stopped at 13:15Z** (budget rule of amendment 5: KXP is not built, so the seeds have no use);
  node B GPU3 / 6 / 7 released, then leased to the IX1 queue for KIB4P-a33 (soup built; ship → BF16 → chain).
- **In flight:** Index X1-a33 / X2-a33 (node C), X3-a33 (node B GPU2 / 4), X5-a33 (node A), KIB4P-a33 (node B
  GPU3 / 6 / 7); formal KIB4-a40 (node A GPU3); prerelease KIB4-a33 (node A GPU7).
- **GPU-h:** node B training final 35.18, node A training 8.93; projected M10 total ≈ 81 with everything in flight.
  No further Index or formal runs after these (85 stop rule).

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
