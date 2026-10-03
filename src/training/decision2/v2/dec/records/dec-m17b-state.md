# Decoder M17b (4B owner and the only Nox-4B publisher; M17 continuation) — state

Branch `xunzhuo/decision-2-training-dec-m17`, worktree `vllm-sr-dev2-dec-m17`. M17's record:
[`dec-m17-state.md`](dec-m17-state.md). Release in force: `vllm-sr/Decision-2.0-Nox-4B@d55528d1` = `4b-SDMLxALL`
(public since 2026-10-03 01:17 UTC+8). Index values stay private (node private stores and
`decision2-program/private/m17b/`); this file has none. Times UTC.

## 2026-10-03 01:45Z — Nox-4B `ce1bdc9d` RELEASED (`4b-LRHxALL`); wave 7 continues against it

- **`vllm-sr/Decision-2.0-Nox-4B@ce1bdc9d` is `main` and public.**
  - It is `4b-LRHxALL`, the uniform soup of four half-LR arm soups.
  - IF1 vs `c60d3b5c`: lower bound above 0 (values private).
  - R3 types OK, IF3 audit6, parity 86 / 86; gate evaluate and every fast-path post-check pass
    (`post_checks=ok` 01:35Z).
  - The `c60d3b5c` weights are purged (9.70 GB, `rewrite_history=False`).
  - Record: [`dev2-4b-w7-2026-10-03.md`](../../release/records/dev2-4b-w7-2026-10-03.md).
- **The next successor is gated against `AF-4b-LRHxALL-bf16`'s run** (amendment 4; results `852e9928…`, pinned
  as `ix1/af/refs/AF-4b-LRHxALL-bf16` on nodes B and C; node C `ix1/runs` holds the run). Nox's next release record
  pins `CURRENT` to `ce1bdc9d`:
  - gate `c350b3be…`, decision `8aee8644…`, manifest `0818bbec…`, weights `31d4ee92…`;
  - `BASE_SPEC` = `specs/dev2-4b-lrhxall.json`;
  - `CURRENT_RUN` = `formal/m17/m17-4b-LRHxALL`;
  - purge node copy = node A `/data/dev2/runs/release/dev2-4b-lrhxall-release-20261003T003955Z/package/Decision-2.0-Nox-4B`.
- **Amendment 4** (`1f5c8ec73`): every remaining candidate gates against that run, and a conditional quarter-LR
  cross-arm soup `4b-LRQxALL` is added.
- **Amendment 5** (`c602f1e11`): `4b-LRHxXALL-m75` is dropped unread, and the rules are read against the
  single-arm release.
- **Scored, gate re-bootstraps running** (CPU; `runs/<model>/paired-boot-full-vs-lrhxall.json`):
  `4b-LRHxALL-L2`, `4b-SDML-lrh` (node C), `4b-LHS17ML-lrh` (node B). Still in shards: `4b-LHS17IB4X-lrh`
  (node B), `4b-LRH2` (node C GPU2–4), `4b-LHS17IB4-lrq` (node C GPU5 / 6, chain C3c against the new reference).
- **GPUs:** node C GPU7 went to the Index submission worker (09:38). Node A GPU0's lease is released. I hold node B
  GPU4 / 6 and node C GPU2–6 until their shards end.
- **GPU-h:** ≈ 45.

## 2026-10-03 00:00Z — first wave-7 reads: no successor; amendment 3; five reads on three lanes

Nox-4B stays `c60d3b5c`. Index values are private (`decision2-program/private/m17b/`).

- **Read once** (full-panel paired bootstrap vs `AF-4b-LHS17IB4-lrh-bf16`, 2,000 replicates):
  - `AF-4b-LRHxXALL-m50-bf16`: not a successor. Point slightly above the release, lower bound below 0. It trades
    the release's deficit gains for the full-LR soup's strongholds. Amendment 2's rule therefore builds
    `4b-LRHxXALL-m75` (node F, `c79ed93d…` → BF16 `ccb3f110…`).
  - `AF-4b-SDMLIB4-lrh-bf16`: not a successor. Point above the release, lower bound below 0. Its profile complements
    the release's arm.
- **Amendment 3** ([`dec-m17b-wave7-amendment-3-2026-10-03.md`](dec-m17b-wave7-amendment-3-2026-10-03.md),
  `e43b58252`; disclosed: written after these two reads):
  - `4b-LRH2` = ½ `4b-LHS17IB4-lrh` + ½ `4b-SDMLIB4-lrh` (node C `6381a8fc…` → BF16 `35c7ae3d…`);
  - `4b-LRHxTOP3`;
  - `4b-LRHxALL7`. The factory was asked for `4b-LHS17SD-lrh` and `4b-LHS17UP-lrh`.

  IX1 entries are at `bdff32333`, mirrored on A / B / C / F.
- **`4b-LHS17IB4-lrq`:**
  - The factory's soup hit a lease precheck at 23:40:57Z: I had leased GPU6, which its `AF_MERGE_SHARED` merges
    used. No merge had started.
  - The FAILED marker went to `soup/void/4b-LHS17IB4-lrq-lease-precheck-*`, disclosed in OPERATIONS.log and
    COORDINATION 08:00. The soup was rebuilt with the merges on GPU1 (released): `f95c15e6…` → BF16 `9eb03c12…`.
- **Lanes:**
  - node C GPU2–4: chain C2 runs `4b-LRHxALL` (shards ending), then `4b-LRHxALL-L2`. Chain C4, `4b-LRH2`, starts
    once C2 has launched its last shard.
  - node C GPU5–7 (idle > 20 min after the factory released them; COORDINATION 07:38 assigns them): chain C3b runs
    `4b-SDML-lrh`, `4b-LHS17IB4-lrq`, then `4b-LRHxXALL-m75`. C3 was stopped while it waited, so as to read `-lrq`
    before m75. The running shards were untouched, and the restart is on the same mirror.
  - node B GPU4 / 6: chain B2 runs `4b-LHS17ML-lrh`, then `4b-LHS17IB4X-lrh`.
- **Incident, no harm:** chains C1 and C2 overlapped on pool 2 3 4 for one wave, and two GPUs each ran two shards
  (peak 228 of 255 GiB, every shard exit 0). Since then, a lane's next chain starts only after the previous chain
  has launched its last shard.
- **R3 blocker:** M17's readouts need a node F GPU whose lease names `dec-m17`, and node F is all 27B. I asked for a
  co-tenant window on node F GPU6 (COORDINATION 07:00). `a15cb55a6` adds `M17_COTENANT_ANY` (≥ 60 GB free VRAM,
  side entry).
- **Leases:** node B GPU4 / 6 and node C GPU2–7 (`track=eval-ix1`).
- **GPU-h:** ≈ 33.

## 2026-10-02 22:50Z — 4B owner #3 (2d3664f4) took over; wave 7 runs on two lanes

Nox-4B stays `c60d3b5c` (`4b-LHS17IB4-lrh`). Took over at 22:12Z. The coordinator approved +30 GPU-h (70 in total).

- **Amendment 2** ([`dec-m17b-wave7-amendment-2-2026-10-03.md`](dec-m17b-wave7-amendment-2-2026-10-03.md),
  `3d5e86cbf`, 22:24Z) was written before any wave-7 result. It adds:
  - single-arm reads of every half-LR arm;
  - `4b-LRHxALL-L2` and the rule-based `4b-LRHxQ`;
  - the conditional `4b-LRHxXALL-m75` and `4b-LRQxLRH`.

  IX1 entries are at `75a3e3974`, mirrored on A / B / C / F.
- **Soups:**
  - the factory's, on node C (merges beside its seed on GPU6): `4b-SDMLIB4-lrh` `de1ab06b…`, `4b-LHS17ML-lrh`
    `a9ab8c81…` and `4b-LHS17IB4X-lrh` `77f3dba1…`, plus the full-LR `4b-SDMLIB4W2`, `4b-SDMLIB4-UP` and
    `4b-LHS17IB4-UP`;
  - mine, on node C (CPU, `af-soup.sh`): `4b-LRHxALL` `2fdee0bc…` (the four half-LR arm soups that existed) and
    `4b-LRHxALL-L2` `72ba598c…` (`4b-LHS17IB4-lrh` listed twice). Both were copied to node F for the formal path.
- **Staged on node C** (`af-stage.sh`, BF16): `4b-SDMLIB4-lrh` `9eefa000…`, `4b-LHS17ML-lrh` `9168ea03…`,
  `4b-LHS17IB4X-lrh` `e89760aa…`, `4b-LRHxALL` `31d4ee92…`, `4b-LRHxALL-L2` `3d11d7f1…`.
- **Index lanes** (reference `AF-4b-LHS17IB4-lrh-bf16`; results hash pinned on B and C):
  - lane B (node B GPU4 / 6): `4b-LRHxXALL-m50` (shards end ≈ 23:16Z), then chain B2 with `4b-LHS17ML-lrh` and
    `4b-LHS17IB4X-lrh`;
  - lane C (node C GPU2–4, leased 22:36–22:37Z from the factory's released leases): chain C1 `4b-SDMLIB4-lrh`
    (parity PASS 22:39Z), chain C2 `4b-LRHxALL`, then `4b-LRHxALL-L2`.
- **Release ops** `v2/release/records/dev2-4b-w7-2026-10-03/ops/` (`f6c9a5f4f`, `588903aca`; mirrored on A). They
  pin `c60d3b5c` as current, write the weight-origin summary with the true seed count, and stop `--release` before
  the download's IX1 gate, which needs the `-hub` entry first.
- **Formal path:** `4b-LRHxXALL-m50` is linked into M17's tree on node F.
- **Leases:** node B GPU4 / 6 and node C GPU2–4, all `track=eval-ix1`, held by my chains.
- **GPU-h:** ≈ 26.

## 2026-10-02 22:15Z — HANDOFF (continuation needed): wave 7 running

**Current release:** `vllm-sr/Decision-2.0-Nox-4B@c60d3b5c` = `4b-LHS17IB4-lrh` (public). The next successor is gated
against `AF-4b-LHS17IB4-lrh-bf16`'s run (results `2d1f1b03…`; node A `ix1/runs/`, node B `ix1/af/refs/`).

**Running (mine):**

- `AF-4b-LRHxXALL-m50-bf16` (BF16 `16baee3d…`) runs on node B GPU4 / 6 under `ixchain.sh`, from mirror
  `cb9bc22b7` with reference `ix1/af/refs/AF-4b-LHS17IB4-lrh-bf16`. It started 22:08Z and should end around 23:20Z,
  followed by its bootstraps. The chain releases both leases (`track=eval-ix1`) at its end.

**Leases held:** node B GPU4, GPU6 (`track=eval-ix1`, 4B owner) for that chain only. No others.

**Next commands** (worktree `vllm-sr-dev2-dec-m17`; `H` = the mirrored head):

1. Read the result:
   `AF_LOCAL=~/code/decision2-program/private/m17b bash src/training/decision2/v2/af/ops/af-ix.sh $H fetch b 4b-LRHxXALL-m50`,
   then look at `paired-boot-full-vs-ref.json` (headline `ci95[0]` > 0 passes).
2. Low-LR arms. The factory was asked for the two-seed soups on node F (COORDINATION 06:15). `4b-SDMLIB4-lrh` and
   `4b-SDMLIB4W2` seeds are DONE on node C; `4b-LHS17ML-lrh`, `4b-LHS17IB4X-lrh`, `4b-SDML-lrh` and
   `4b-LHS17IB4-lrq` started 21:27Z on node C GPU1–7.
   - Read each two-seed soup once: stage on F with `af-ix.sh $H stage f <name>`, `pkgcopy` to the lane node, then
     `ixchain.sh` with the reference above. Their IX1 entries already exist (`a4bed246e`); a new name needs a launcher
     commit on the factory's 4B line.
3. `4b-LRHxALL` = uniform soup of the half-LR arm soups that exist by then (wave 7 amendment 1). Build it on F with
   `af-soup.sh`, then measure it.
4. Release path for a passing candidate: `dev2-4b-w6-2026-10-03/ops/` is per-candidate (`CAND`). Copy it to a new
   record `dev2-4b-w7-…` with `CURRENT` = `c60d3b5c`:
   - `gate_sha256` = `sha256(release/receipts/gate.json)` of this release;
   - decision `40317389…`;
   - manifest `0736f6ca…`;
   - weights `1cef169d…`;
   - `BASE_SPEC` = `specs/dev2-4b-lhs17ib4-lrh.json`;
   - `CURRENT_RUN` = `formal/m17/m17-4b-LHS17IB4-lrh`;
   - `PURGE_NODE_COPY` = node A `/data/dev2/runs/release/dev2-4b-lhs17ib4-lrh-release-20261002T213803Z/package/Decision-2.0-Nox-4B`.

   Also fix `origin.summary` (it still says "three seeds"). The formal path is `m17b-formal.sh`: run `mlx` from the
   SAME mirror as `formal`. Before `--release` the GPU lease must be `track=eval-ix1` for the IX1 gate, and the
   download's IX1 `-hub` entry is committed after the upload, then run `--post WORK`.

**Budget:** about 24 of 40 GPU-h used.

## 2026-10-02 22:05Z — Nox-4B `c60d3b5c` RELEASED (`4b-LHS17IB4-lrh`); wave 7 (low-LR arms) next

- **`vllm-sr/Decision-2.0-Nox-4B@c60d3b5c` is `main` and public.**
  - Gate evaluate and all fast-path post-checks pass (`post_checks=ok` 21:58Z).
  - The `d55528d1` weights are purged (9.70 GB, `rewrite_history=False`).
  - Record: [`dev2-4b-w6-2026-10-03.md`](../../release/records/dev2-4b-w6-2026-10-03.md).
  - Integration fast-forwarded to `fee91afd0`.
- **The next 4B successor is gated against `AF-4b-LHS17IB4-lrh-bf16`'s run** (node A `ix1/runs`, results
  `merged/results.jsonl`).
- **Wave 7, amendment 1 (`d7277c8dd`):**
  - `4b-LRHxXALL-m50` (`7e256b09…`): ½ the new release + ½ `4b-SDMLxALL`. Built on node F, not staged yet.
  - `4b-SDMLIB4-lrh`: the factory's seeds are DONE on node C; the soup is not built yet (merges need a node C GPU).
  - `4b-LRHxALL`: built over the half-LR arms as they finish. The factory has `4b-LHS17ML-lrh`, `4b-LHS17IB4X-lrh`
    and `4b-SDML-lrh` (two seeds each) and `4b-LHS17IB4-lrq` on node C GPU1–7, from 21:27Z.
  - `4b-LHS17IB4-lrq`.
- **Leases:** none held (node A GPU0, node F GPU6–7 released; node C GPU1–4 went to the factory after wave 6b).
- GPU-h so far ≈ 24.

## 2026-10-02 21:40Z — the half-LR arm `4b-LHS17IB4-lrh` passes; its Nox-4B release is running

- **Wave 6, every candidate measured once** (full-panel paired bootstrap vs `DEV2.0-4B-SDMLxALL-bf16`, 2,000
  replicates; values private):
  - not successors: `4b-AFxALL3` (CI below 0), `4b-XALLx` (CI below 0), `4b-AFxALL2`, `4b-XALLU2` and `4b-AFxALL`
    (the last two at the release's level, lower bounds below 0);
  - information points: `4b-LHS23IB4` and `4b-LHS17IB4ML` below the release;
  - **`4b-LHS17IB4-lrh` (the factory's two-seed half-LR arm) passes: lower bound above 0** (amendment 2,
    `b60c1835a`, disclosed as written after its read).
- **Its release inputs:**
  - parity 86 / 86;
  - IF3 = audit6 (TRAIN `dfed3944…`, 0 item rows);
  - R3: formal `m17-4b-LHS17IB4-lrh` (node F GPU7, mirror `b60c1835a`), types choice / Noul / Score OK, mlx-diag
    scored;
  - the card's Index input: each other tier at its current main (Kai `51b7b474`, Eos `1d380452`, Sol `6a62b319`,
    Lux `f3122c7c`, Vega `5c85c127`), and the 4B point from this run;
  - assets rendered; spec `2495f460…` and decision `40317389…` committed at `9bde2b7c4`, with `--check` byte-equal on
    node A.
- `release_w6.sh LHS17IB4-lrh --release --gpu 0` started 21:38Z on node A (Hub `main` = `d55528d1` checked).
  References (not gates, disclosed): post-key v3 below the current weights'; mlx-diag and public 231 above.
- mlx-diag attempts of `m17-4b-XALLx` / `-XALLU2` from a mirror other than their collection's exited at the adapter
  spec (no inference). They were moved to `formal/m17/void/mlx-mirror-mismatch-m17b-*` and relaunched from the
  collection mirror. `AF-4b-LHS17IB4ML-bf16`'s shards 6 / 7 were re-run whole after their partial directories and
  launch records went to `ix1/void/`.
- GPU-h so far ≈ 22.

## 2026-10-02 20:05Z — wave 6a: no successor; wave 6b on node C

- **Not successors** (full-panel paired bootstrap vs `DEV2.0-4B-SDMLxALL-bf16`, 2,000 replicates; values private):
  `AF-4b-AFxALL3-bf16` (95% CI entirely below 0) and `AF-4b-AFxALL2-bf16` (point below 0, lower bound below 0).
  Both lose most on RAGTruth, then GPQA Diamond, When2Call and MuSR; RAGTruth follows `4b-LHS17UP`'s share of a soup.
- **Amendment 1** (`0e444cb1a`, written before any further result): `4b-XALLx` (`7ae47bc2…`, the release's seven
  arms at their most seeds) and `4b-XALLU2` (`58fa0585…`, UP listed twice); `4b-AFxALL4` dropped. Both on node C
  GPU1–4 (leased 19:30Z, idle since 19:07Z), parity PASS 19:44Z. Their formal runs are collected and scored (node F
  co-tenant, node A); mlx-diag collecting.
- `AF-4b-AFxALL-bf16` (node F GPU6–7) and the information points `AF-4b-LHS23IB4-bf16` (scored) and
  `AF-4b-LHS17IB4-lrh-bf16` (node A GPU5–7) follow the prereg.
- Merged `9d90afd10` (shared-context switch, COORDINATION 02:23): the wave-6 spec's `runtime_source` is this
  branch's mirror `b50e8650b` (phase A plus the opt-in switch); 40 release tests pass on node A (vendor source,
  shared ctx, bf16 copy, hub upload / collect).
- Fixes: `af-stage.sh` now finds an IX1 entry that is the first name of its list (`b3a917279`); the launcher's
  `AF-4b-XALLx` / `AF-4b-XALLU2` entries sit on the factory's 4B line (`247440f37`).
- GPU-h so far ≈ 14 (Index ≈ 11.5, formal and readouts ≈ 1.5, parity and merges ≈ 1).

## 2026-10-02 19:05Z — wave 6a on the Index (two lanes); release inputs staged

- **Soups built** (uniform FP32, `af-soup.sh` from this branch's mirror; node C merges on GPU5, SELECT agreement
  checked): `4b-LHS17IB4X-s34` `9c9e9355…`, `4b-LHS17ML-s34` `e4056234…` (node C, copied to F, lists equal); on node F
  `4b-LHS17UP-x4` `4422a353…`, `4b-AFxALL` `8f282287…` (12), `4b-AFxALL2` `70612d1c…` (10), `4b-LHS17IB4X-x4`
  `6edd6165…`, `4b-LHS17ML-x4` `8c69cd51…`, `4b-AFxALL3` `df95b286…` (10).
- **Index (reference `DEV2.0-4B-SDMLxALL-bf16`, results `54389c5a…` pinned on F and A):**
  - `AF-4b-AFxALL2-bf16` (BF16 `3cfe2816…`): node F GPU6–7, parity PASS 18:12Z, shards running.
  - `AF-4b-AFxALL3-bf16` (BF16 `06b3633f…`): node A GPU5–7 (02:00 assignment; panel-8 copied C → A, lists equal),
    parity PASS 18:22Z, last two shards running.
  - Queued: `AF-4b-AFxALL` on node F after AFxALL2; information point `AF-4b-LHS23IB4-bf16` on node A.
- **IF3:** `audit6` (node C CPU, one audit of the ten distinct member TRAIN files of every wave-6 candidate): 120,226
  Index rows, planted 200 / 200, **0 item rows in all ten**.
- **Formal path (R3, card reports) run ahead of the Index results**, since it costs about 15 GPU-minutes per
  candidate: `m17-4b-AFxALL2` and `m17-4b-AFxALL3` collected on node F (co-tenant, T = 1), scored on node A: types
  choice / Noul / Score OK for both; mlx-diag collected and scored.
- **Release ops** `v2/release/records/dev2-4b-w6-2026-10-03/ops/` (`ca2b8fe94`). The card's Index input takes each
  other tier's point from the released input that scored its current main (9B Lux `f3122c7c`, 27B Vega `5c85c127`).
  Already staged on node A for both candidates: current-revision receipts, the gate files, the BF16 checkpoints.
- GPU-h so far ≈ 4.5 (Index ≈ 3.5, formal and readouts ≈ 0.6, merges ≈ 0.1, parity ≈ 0.3).

## 2026-10-02 18:05Z — started; wave 6 preregistered

- Merged integration (`47dd06be7`, the central Hub visibility policy) and the arm-factory branch (its soup, staging
  and measurement tools, its IX1 entries and M10's `ixchain.sh`).
- Wave 6 prereg: [`dec-m17b-wave6-prereg-2026-10-03.md`](dec-m17b-wave6-prereg-2026-10-03.md). Candidates in order:
  `4b-AFxALL2`, `4b-AFxALL3`, `4b-AFxALL`, contingent `4b-AFxALL4`.
- Inventory:
  - M17's 21 soups and their Index runs (node F soups, node C runs and scores) are intact. The factory's built 4B
    soups: `4b-LHS17IB4-x5`, `4b-SDMLIB4-x5`, `4b-LHS17IB4ML`, `4b-LHS23IB4` (node F); `4b-LHS17IB4-lrh`,
    `4b-LHS17UP-s34` (node C). Batch-2 seeds `4b-LHS17IB4X` s3 / s4 and `4b-LHS17ML` s3 / s4 are DONE (node C,
    not merged). `4b-SDMLIB4-lrh` s1 / s2 failed at the node F return (the factory re-runs them).
  - The factory's node F soup pipeline (`soups-all.sh`: `4b-LHS17UP-x4`, `4b-AFxALL`, `4b-AFxALL2`) survived the
    return and waits only for `4b-LHS17UP-s34` and `4b-LHS17IB4-lrh`: both are being copied C → F (node A relay,
    SHA-256 lists compared).
  - `AF-4b-LHS17IB4ML-bf16`: shards 0–5 exit 0, shards 6–7 died at the return (not scored).
  - Per-case results for the paired bootstrap against `d55528d1`: the release's run `DEV2.0-4B-SDMLxALL-bf16` on
    node C (`ix1/runs`) and node F (`ix1/af/refs`, results hash pinned).
- Leases: node F GPU6–7 (`track=eval-ix1`, 4B owner Index runs).
