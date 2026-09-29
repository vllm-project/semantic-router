# DEV2.0-4B release candidate — N4XF soup (decoder Milestone 4; 2026-09-29)

> **Erratum (2026-09-29, per coordinator 05:25 note).** The identity of this checkpoint rests on the per-file SHA-256
> list (node B `m4/hf-staging/N4XF-soup.sha256`, `cea4cc9a2a5997e02ebb55ed3cb653b348f2067855a51b13bbbba7f95da984f3`, 14 files,
> 16,853,623,514 B) and the verified node copies, not on the staging revision. `e8656221` was created after every
> history-rewriting cleanup. It is `main` of `dev2-dec-staging`, and its files were downloadable when checked on
> 2026-09-29 ≈00:15Z. Details: [dec-staging-citation-errata-2026-09-29.md](dec-staging-citation-errata-2026-09-29.md).

**Qualifies under the preregistered 4B bar** ([prereg](dec-m4-prereg-2026-09-29.md),
[amendment 1](dec-m4-amendment-1-2026-09-29.md), [selection](dec-m4-selection-2026-09-29.md), chosen on
development panels before any formal run). The paired 95% CI lower bound vs the adopted Nox 1.0 run is > 0;
CSS15 H .580 ≥ .519; typed-FINAL types are all above Nox 1.0's; no type collapsed.

## Post-key same-panel results (JevArena v3, node A, 16,384 tokens)

Scored run: `/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA` (REPORT `87ec711c…`, SEAL `12b44de9…`, PAIRED vs
adopted `01764e8f…`); `mlx-diag`: `/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-mlx`. Image
`decision20-train-fast:host2` `f83b1d10…` (the adopted Nox 1.0 run's), per-run persisted autotune cache (the M3
procedure).

| Comparison (v3 = 100·√(T·H)) | Candidate | Comparator | Δ [95% CI] |
| --- | ---: | ---: | --- |
| **adopted Nox 1.0** (`m1-adopt/nox1`, the bar) | **63.151** | 56.470 | **+6.68 [+0.99, +9.64]** |
| Nox 1.0 16K same-limit (`formal/m3/nox1-16k`) | | 55.689 | +7.46 [+1.80, +10.45] |
| Decider 4B (`m1-adopt/decider4b`, best 4B peer) | | 61.882 | +1.27 [−5.71, +4.31] |
| Jet v6.2 (`m2/q5b-jet62`) | | 60.375 | +2.78 [−3.33, +5.94] |
| M3 N4LKr soup (previous best 4B) | | 59.539 | +3.61 [+1.22, +12.13] |

- **Axes vs Nox 1.0:** T .6881 vs .6144 (+.074); H (CSS15 median task macro-F1) .5795 vs .5190 (+.061).
- **Typed-FINAL:** Choice / Noul / Score 582 / 734 / 185 vs 552 / 653 / 178. By family: exception_stack .835 vs
  .632, constraint_competition .455 vs .380, resource_ledger .463 vs .445, evidence_join 1.0 vs 1.0.
- **Type gate** (`v2.eval.gates types`): no type collapsed. Choice .728 [.696, .757], Noul .918 [.896, .935],
  Score .463 [.414, .511]; top answer shares .27 / .52 / .36.
- **CSS15:** 11 of 15 tasks up. Largest gains reddit_humor +.166, emotion +.084, persuasion +.070.
- **public 231:** 171 vs 173 (easy 48 / 48, standard 67 vs 66, hard 56 vs 59).
- **Loaded parameters:** 4,208,383,488 (safetensors header count).

## Disclosures for the card

- **Multilingual (`mlx-diag`) −2.4 points.** Type macro .770 vs Nox 1.0 .795; non-English .763 vs .789.
  - The loss is non-English Noul: .727 vs .800. Multilingual Score is level (.811 vs .808).
  - By language: ko .63 vs .69, ja .72 vs .77, es .78 vs .82, zh .78 vs .82, de .78 vs .81. Ahead on ar and ru.
- **CSS15 regressions:** wiki_corpus −.065, mrf −.045, media_ideology −.016, talklife −.001.
- **public 231:** hard tier −3 items; long inputs 16 vs 17 of 37. CSS15 long inputs are level (.411 vs .415).
- **Invalid answers:** 4 invalid CSS15 Choice answers (of 6,547 items), counted as failures.
- **Weight origin:** own Nox 1.0 (`llm-semantic-router/Decision-1.0-Nox-4B@cde2a68d`), full fine-tuning.
- **Teacher:** own Lux 1.0 (`bd45a30a`) soft targets on every row except the gold-only H7 / H8 gap rows (6.4% of
  rows, which dilute the KL term by about that share). No third-party teacher. The published own-Lux XL targets
  were produced on the node-B Lux image without causal-conv1d (research & data's disclosure applies).
- **Training data:** a token-matched 14.42% whole-group subsample of research & data's XL r2 full recipe (plus
  the M3 A0s rows). 58,742 rows / 29.41M native tokens, 35 languages, English 58%. None of r2's 305
  evaluation-panel-excluded groups, no A7x, no C1-registry dataset.
  - **C1 event 3 needs its independence recheck** against XL r2, the H7 / H8 gap arms and Lux-XL waves 3–5 (the
    eval track's standing plan).

## Identities for release engineering

| Item | Value |
| --- | --- |
| Staged checkpoint | private `llm-semantic-router/dev2-dec-staging@e86562218928fbb77f8071b45646962f63049fad`, folder `m4/N4XF-soup/` (14 files, hash list `/data/dev2/runs/dec/m4/hf-staging/N4XF-soup.sha256` on node B, verified on node A) |
| Soup | uniform FP32 average of N4XF s1 `checkpoint-0000690`, s2 `checkpoint-0000676`, s3 `checkpoint-0000492` (SELECT700 BEST); `model_sha256` `11b5ca1c22f8cd38d67d718cbfa59a3fb4dc8f4e7bfd75b3e988f34e35fad0ad`, decision head `bd324aa6…` |
| Node copies | node B `/data/dev2/runs/dec/m4/soup/N4XF/build/N4XF-soup` (+ `hf-staging/N4XF-soup`); node A `/data/dev2/runs/dec/m4/formal-candidates/staging-e8656221/m4/N4XF-soup` |
| Calibration | CAL698 at 16,384 tokens `cal698-16k/calibration.json` `176a2f0e…` (Choice 0.667 / Noul 0.496 / Score 0.212), used for the formal run; the 23:15 rule (adopt only if typed-DEV and CSS-pilot ECE and Brier do not worsen, else T = 1) is still to be applied |
| Packaging | profile `qwen-full`, 16,384 tokens, adapter `v2/dec/adapter-spec-infer-dec.json` |
| Training | mixture `m4-xl-full-29m` `c7d51219…` (spec `cf311178…`); teacher `e2ff27ce…`; recipe M3F, KL 1.0, `--teacher-partial`; seeds 20260926 / 7 / 8; node B image `dbe5f32b`; code `0b8ebfc54` (data lock `42f26c846`) |
