# Decoder M17b (4B owner and the only Nox-4B publisher; M17 continuation) — state

Branch `xunzhuo/decision-2-training-dec-m17`, worktree `vllm-sr-dev2-dec-m17`. M17's record:
[`dec-m17-state.md`](dec-m17-state.md). Release in force: `vllm-sr/Decision-2.0-Nox-4B@d55528d1` = `4b-SDMLxALL`
(public since 2026-10-03 01:17 UTC+8). Index values stay private (node private stores and
`decision2-program/private/m17b/`); this file has none. Times UTC.

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
