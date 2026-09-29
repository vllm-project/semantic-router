# HT-DEV v2 (`ht-dev2`) — build and validation: PASS (eval & peers, 2026-09-30)

Preregistered in [`htdev2-prereg-2026-09-30.md`](htdev2-prereg-2026-09-30.md) (`22265ee24`), amendments
[1](htdev2-prereg-amendment-1-2026-09-30.md) to [5](htdev2-prereg-amendment-5-2026-09-30.md), each committed before the
step it changed. Code `v2/eval/htdev2/`; aggregates in [`htdev2/`](htdev2/) (`validation/analysis.json`,
`validation/features.json`, `build/`, `design/`, `gpu-time.json`, `sources.json`). Formal H comes from the stored
post-key `REPORT.json` files; no formal item was re-run and nothing from JevArena-C1 was read (the informational C1
check uses stored event aggregates). HT-DEV v2 numbers are development readouts, never release scores.

## Result

- **HT-DEV v2 tracks formal within-tier human transfer clearly better than the CSS-pilot three-task mean. Both
  preregistered conditions hold.**
  - (i) Within-tier sign agreement with ΔH_formal on decidable pairs (|ΔH_formal| ≥ 0.02): **145/172 = 0.843** vs
    **113/172 = 0.657** for the pilot mean. Paired model bootstrap (5,000 draws): Δ +0.186 [0.000, +0.391],
    **P(better) = 0.972** (bar 0.90).
  - (ii) Within-tier Pearson r with formal H: **0.647 vs 0.206** (P(better) 0.999).
- The gain is largest where tracks need it: **sibling pairs** (same-lineage 2.0 candidates) agree on **74/78 = 0.949**
  vs 0.756, and all pairs of our own models (2.0 candidates and own 1.0) on 0.935 vs 0.685.
- **Use it as the human-transfer screen in development gates** (usage below). The formal paired CSS15 CI still decides
  human transfer; HT-DEV v2 does not predict JevArena-C1 deltas (informational check below).

## Design (why a CSS15 parallel form)

- HT-DEV v1 (13 dataset-isolated human-labelled tasks, design (b)) failed because formal H is dataset-specific, not
  because the target is noisy: a split of the formal CSS15 items shows within-tier ΔH is highly reproducible (a
  parallel form drawn from the formal items at 35% size agrees 0.90 with disjoint formal items; design check,
  `design/ceiling*.json`).
- So HT-DEV v2 is **design (a)**: the same CSS15 tasks, sources, SALT formatting, prompt templates and authors'
  option maps, with items from the held-out portions of each source. Design (b) was not rebuilt; (c) is reported as a
  secondary (it adds nothing).
- Primary functional: the **mean** of task macro-F1 (more robust than the median at this size in the design check).

## Panel

- **1,944 items: 9 tasks × 216**, class-balanced like the formal sample (emotion 6 × 36; ibc, media_ideology,
  talklife 3 × 72; conv_go_awry, mrf, reddit_humor, tempowic, wiki_corpus 2 × 108). All Choice, in the formal CSS
  item format; every task's instructions, criteria and option order are identical to its formal items.
- **Sources** (`htdev2/sources.json`): SALT `LLMs_for_CSS@55183a64` pools (ibc, media_ideology, talklife, mrf),
  `dair-ai/emotion@cab853a1` (unsplit), TempoWiC `@c83ccbe5` (train/test), RedditHumorDetection `@09c86e22`
  (test remainder, train), ConvoKit CGA and wiki-corpus. Internal evaluation use only, as for CSS15.
- **Reconstruction check:** the builder rebuilt each SALT pool and mapped the formal items onto it by (normalised
  context hash, raw prompt hash): 100% mapped for every task except wiki_corpus (99.4%), with 100% label agreement.
- **Tasks dropped (6 of 15):** indian_english_dialect (no held-out rows), tropes (114 singleton labels cannot be
  reproduced), persuasion and raop (the formal sample used every item of one class), flute and wiki_politeness
  (their held-out rows are in prior-session CSS training pools, `css_flute_1k_v1`, `combined_6k_v1`,
  `css_wiki_politeness_v1`).
- **Isolation** (counts in `build/`):
  - formal CSS15 and pilot: every pool item with a formal or pilot context, and every item in a group (conversation,
    seeker post, article, tweet pair, premise) touched by a formal item, was removed first;
  - training: 43,889 pre-scan candidates against 122,485 files of the C1 v1.3 rescan's training inventory plus PN1
    and HS1 (8-token spans, containment): 12,382 OVERLAP + 2,720 REVIEW dropped, conservatively including raw source
    copies in that inventory;
  - protected panels (typed FINAL/DEV, public 231 = JevBench, mlx-diag, HT-DEV v1, Score5/Score5t, HS1-DEV, SELECT,
    CAL): 90 OVERLAP + 105 REVIEW dropped;
  - JevArena-C1: no source is a C1 registry dataset; the sealed directory was never read;
  - leak audit: CLEAN on option and state surfaces, 0 id mentions of gold.
- **Rules:** panel items are never training data. A new training arm drawn from any of the nine sources must exclude
  the panel's items and groups (hashes in the gold file) and pass the training scan.

## Validation

**Models:** 50 unique-weight 0.6B–9B models with stored node-A formal runs, siblings included (0.6B 8, 0.8B 12, 2B 5,
4B 16, 9B 9). Not collected: four 0.6B 1.0/peer models at the 1.5 GPU-h cap (amendment 5) and four decoder packages
that exist only on node B (amendment 4). Each model ran once on its formal runtime (formal adapter spec, image,
input limit and a copy of its frozen autotune cache). The pilot mean comes from the same-runtime pilot readouts (HT-DEV
v1 collections, track readouts, or the same job).

**Primary and secondary metrics** (260 within-tier pairs; target formal CSS15 H):

| Metric | HT-DEV v2 (mean) | HT-DEV v2 median | **Pilot three-task mean (bar)** | Pilot median |
| --- | ---: | ---: | ---: | ---: |
| **Agreement, \|ΔH_formal\| ≥ 0.02 (172 pairs)** | **0.843** | 0.762 | **0.657** | 0.581 |
| Agreement, all pairs | 0.781 | 0.665 | 0.588 | 0.550 |
| Agreement, ≥ 0.01 / ≥ 0.03 | 0.827 / 0.849 | 0.686 / 0.799 | 0.623 / 0.669 | 0.555 / 0.597 |
| **Within-tier Pearson r** | **0.647** | 0.574 | **0.206** | 0.114 |
| Pearson r of within-tier pair deltas | 0.692 | 0.600 | 0.231 | 0.170 |
| Cross-tier Spearman | 0.906 | 0.752 | 0.847 | 0.817 |
| Agreement vs CSS15 task mean (114 pairs) | 0.921 | — | 0.711 | — |

- **Per tier** (decidable pairs right, HT-DEV v2 vs pilot mean): 0.6B 21/21 vs 10/21; 0.8B 37/45 vs 26/45; 2B 4/9 vs
  6/9; 4B 71/84 vs 62/84; 9B 12/13 vs 9/13. It loses only at 2B (5 models).
- **Lineage:** own-model pairs 0.935 (92 pairs) vs 0.685; pairs involving a third-party peer 0.738 (80) vs 0.625. The
  misses concentrate on peers that score higher on HT-DEV v2 than on formal (Decider 2B, This-That 1.2, JPT, Kev,
  Intern); their training data are unknown, so they may have seen these sources' training portions.
- **Agreement by |ΔH_dev2|:** < 0.01: 0.63 (86 pairs); 0.01–0.02: 0.74 (69); 0.02–0.04: 0.91 (67); ≥ 0.04: 0.97 (38).
  Preregistered normal-residual 10%-risk band: **0.045** (σ 0.028, slope 1.17, tier fixed effects).
- **Combination (c)** (9 HT-DEV v2 tasks + 3 pilot tasks): 0.837, r 0.614; no gain over HT-DEV v2 alone.
- **HT-DEV v1 subset** (33 models, 68 decidable pairs): HT-DEV v2 0.721, pilot mean 0.500, HT-DEV v1 0.456.
- **JevArena-C1 post-key, informational** (17 event models, 25 within-tier pairs; never tuned to): HT-DEV v2 11/25, the
  pilot mean 14/25, formal H itself 14/25. C1 measures different human-labelled sources; HT-DEV v2 predicts formal H,
  not C1, and the C1 guard (item 8) stays necessary.
- **Motivating cases:**
  - Decoder M6 0.8B finalist `E6K-b1_3` vs the E8F soup: formal −0.003, HT-DEV v2 +0.002 (TIE: no human-transfer change).
  - DEV2.0-4B (N4XF) vs Nox 1.0: formal +0.061, HT-DEV v2 +0.036. Its C1 deficit (−1.32) is not a human-transfer
    signal on either panel.
  - DEV2.0-4B vs Decider 4B (formal +0.024, HT-DEV v2 −0.015) and DEV2.0-2B vs Decider 2B (+0.105 vs −0.027) are
    peer misses.

**Limitations.** Nine of the fifteen tasks; the rest had no admissible held-out portion. Some held-out items come from
other splits (TempoWiC train/test, RedditHumor train, the unsplit emotion set) than the formal sample; the class
design and templates are the formal ones. The 0.6B tier here has the 0.6B-track soups only. Peers are weaker.

## Use (paste-ready for "Eval runners")

### HT-DEV v2 (`ht-dev2`): human-transfer screen (eval track, 2026-09-30; code at `7a6bd7986` or later)

Record `v2/eval/records/htdev2-validation-2026-09-30.md` (prereg `htdev2-prereg-2026-09-30.md`, amendments 1–5).

- **Validated (PASS).** Within-tier sign agreement with formal CSS15 ΔH 0.843 vs 0.657 for the CSS-pilot three-task
  mean (172 decidable pairs, 50 models, P(better) 0.972); within-tier r 0.647 vs 0.206; sibling pairs 74/78.
- **Panel:** 1,944 items, nine CSS15 tasks × 216 from held-out portions of the same sources, formal templates and
  option maps; disjoint from the formal items at item and group level; screened against training data and every
  protected panel. Never training data.
- **Rule for every track (replaces the CSS-pilot three-task-mean non-decrease screen in development gates):**
  - Collect `ht-dev2` for each shortlisted checkpoint at its formal input limit and kernel path, and compare it with the
    tier's reference (the current release, or the milestone's baseline) with `--htdev2-reference`.
  - **FLAG** (ΔH_dev2 ≤ −0.02): do not send it to the formal runner as a successor without a recorded reason. Across
    within-tier pairs with |ΔH_dev2| ≥ 0.02, formal ΔH has the same sign 93% of the time.
  - **TIE** (|ΔH_dev2| < 0.02): uninformative; it neither passes nor blocks human transfer.
  - **GAIN** (≥ +0.02): a likely formal human-transfer gain.
  - ±0.045 is the preregistered 10%-risk band: beyond it, the formal order should match with ≥ 90% probability.
  - It is a screen, not a score to optimize. Human transfer is decided only by the formal paired CSS15 CI. Treat
    comparisons with third-party peers with caution (0.74), and keep the C1 guard (HT-DEV v2 does not predict C1).
- **Run** (node A; about 0.016 GPU-h per 0.6B checkpoint and 0.036 GPU-h per 4B/9B checkpoint):

  ```bash
  SRC=<full-sha>-src_training_decision2; S=/data/dev2/src/$SRC/src/training/decision2; export PYTHONPATH=$S
  $S/v2/eval/run_same_panel.sh --gpu <N> --track <track> --src $SRC --run-dir <run> --model-dir <pkg> [--shared-lease <name>] \
    [--env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR=<rw copy> --mount-rw <rw copy>] \
    -- --adapter-spec <adapter.json> --model-path <pkg> --revision <id> --panels ht-dev2   # or typed-dev,css-pilot,ht-dev2
  python3 -m v2.eval.dev_readout --run-dir <run> --label <ckpt> --output <readout.json> \
    --htdev2-reference /data/dev2/runs/eval/htdev2/collect/<reference key>/output/ht-dev2.predictions.jsonl
  ```

  - The readout JSON gets `htdev2`: H_dev2 (mean task macro-F1), the median, per-task macro-F1, the item-bootstrap SD,
    and `vs_reference` with delta, paired 95% CI and verdict. Stdout adds `htdev2.H_dev2`, `htdev2.delta`,
    `htdev2.ci95` and `htdev2.verdict`.
  - Stored reference collections (`/data/dev2/runs/eval/htdev2/collect/<key>`) for the current releases:
    DEV2.0-0.6B `06b-m6-mxcx-soup` (the same weights as `b2131337`, whose Score offsets do not touch Choice items),
    DEV2.0-0.8B `dec-m2-E8F-soup`, DEV2.0-2B `dec-m3-S2T-soup`, DEV2.0-4B `dec-m4-N4XF-soup`, DEV2.0-9B
    `9b-m4-K-a13`. Own 1.0: `eos1`, `sol1`, `nox1`, `lux1` (Kai1 not collected). There is no 27B collection yet (node B).
  - Scorer only: `python3 -m v2.eval.htdev.score seal --prompts .../goldfree/ht-dev2.prompts.jsonl --predictions <p>
    --output <seal>`, then `python3 -m v2.eval.htdev2.score score|compare ...`.
- **Files:** `/data/dev2/private/panels/{goldfree,gold}/ht-dev2.*` (prompts `90cd409a…`, gold `659c92b4…`, gold mode
  600). Backup: private eval-artifacts `htdev2/v1/` at `6f6acf5d` (panel, build and scan receipts, 50 collections,
  validation; 350 files verified by readback).

## Compute

- **GPU: 1.454 GPU-h** on node A GPU0 (0.733) and GPU1 (0.721), shared lease `owner.eval-htdev2`: collections 1.414
  (50 jobs), smoke runs 0.040 (4 runs; two failed first attempts of about 0.0001 GPU-h each were replaced). Per job 0.016 GPU-h (0.6B) to 0.036 GPU-h (4B, 9B). Ledger `htdev2/gpu-time.json`.
- CPU only: design check (4 runs, about 15 s each), pool build (28 s), overlap scans (training 704 s with 48 workers;
  panels 11 s), freeze, leak audit, scoring (50 models, 4 s with 25 workers), validation (6 s).

## Artifacts and hashes

- Node A: panel build `/data/dev2/private/htdev2/{work,scans,panel}`; collections, validation and jobs
  `/data/dev2/runs/eval/htdev2/{collect,validation,ops,usage}`.
- sha256: prompts `90cd409a1e091a623362c0e5b227d13b7301bf13fe266f7905e09233cf815f74`, gold
  `659c92b45d2a5b37e250ccb728cdbf5580ba361b3fc4b2f2ab505a13e0a556cc`, `LEAK-AUDIT.json` `029eb8cd…`.
- Commits (branch `xunzhuo/decision-2-training-eval-devpanel`): design check `fe5ba61bf`, `4b6917838`; prereg
  `22265ee24`; builder + amendment 1 `24f3170df`; amendment 2 `1e4689e3e`; scorer, jobs and lane `30774cd97`;
  validation `501e077fc`, `19c931a56`; amendment 3 `5c1d50428`; **shared-module changes:** panel registry
  `ee32c827b`, leak audit `6f72b8689`, `dev_readout` `htdev2` block `7a6bd7986`; job fix `8097e859b`; amendment 4
  `bcd1ddc8c`; scoring driver `6709902ae`; amendment 5 `ab6e15a12`.
