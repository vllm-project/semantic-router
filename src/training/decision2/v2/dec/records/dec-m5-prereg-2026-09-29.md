# Decoder Milestone 5 — preregistration (4B multilingual-Noul successor study; 2026-09-29)

Written and pushed before any Milestone 5 mixture, teacher file or development panel is built and before any arm is
launched or read out. Development readouts (SELECT700, CAL698, typed DEV, CSS pilot, MLX-DEV below) are never
release scores; JevArena v3 / public JevBench 231 numbers are post-key same-panel comparisons; `mlx-diag` is a
diagnostic panel and drives no Milestone 5 decision.

## Goal and bar

Remove the multilingual-Noul loss that the 0.8B / 2B / 4B 2.0 models share, without giving up their typed-reasoning
and human-transfer gains, on the 4B N4XF recipe first (then 2B, conditionally). Reference: the 4B release candidate
**N4XF soup** (post-key v3 63.151; `mlx-diag` .770 vs Nox 1.0 .795, non-English Noul .727 vs .800).

- **Successor** (21:30 post-release rule): paired post-key v3 95% CI lower bound > 0 against the N4XF soup
  reference run (below) AND human transfer not worse (paired CSS15 H difference ≥ 0) AND typed-FINAL types each
  ≥ 75% of Nox 1.0's, no type collapsed.
- **Card-only multilingual fix:** v3 paired CI includes 0 (point ≥ −1.0) and the diagnostic `mlx-diag` non-English
  Noul closes at least half of the N4XF − Nox 1.0 gap (≥ .764) with non-English Choice and Score each within −.01
  of N4XF. Recorded for the coordinator; not a successor.
- Anything else is negative and recorded as such.

## Inputs decided before this record (disclosed)

1. **Diagnosis of the loss** (aggregates of the already-run `mlx-diag` scorings of released / candidate models; no
   item-level use): the whole N4XF − Nox 1.0 gap is Noul, i.e. PAWS-X "same meaning?" pairs. It is a yes-bias:
   non-English predicted-yes rate .760 vs .663 (gold .50), recall on gold-No .467 vs .637, worst in ja (.34 vs .60)
   and ko (.30 vs .48); English is level. The 2B (S2T .822 vs Sol 1.0 .790) and 0.8B (E8F .948 vs Eos 1.0 .878)
   drift the same way. No Milestone 5 artifact is read on `mlx-diag` before its v3 formal report is sealed.
2. **Training-data fact.** Every non-English human Noul row in `mx-xl-full-r2` is answerability / relevance style
   (H5 evidence-removal and TyDi removal / relevance twins; H8 MIRACL relevance and TyDi page-window removal); no
   paraphrase or NLI Noul exists in any non-English language, and none may be added here (MASSIVE / PAWS-X / XNLI
   are excluded in every split; the only Spanish NLI candidate, esnli-R, is a C1 registry source). Filed for research
   & data; Milestone 5 stays within the admitted multilingual sub-arms A7q / A7k / A7s, H8 and data-v2 H5.
3. **Matched tokens without diluting typed / transfer data.** In N4XF every token outside the five multilingual
   sub-arms is a typed curriculum pool (generated A7 / G / V1 pools), a human transfer pool (H1, H3, H6, H7, E11) or
   the A0s anchor, which itself mixes typed generators (programmatic, stage-4 composition, stage-3 replay) with
   GoEmotions and SNLI. Raising the multilingual share at the matched budget would therefore displace typed /
   transfer tokens. **Decision:** every row outside the non-English multilingual block stays identical to N4XF (same
   ids), and the "controlled share" is applied inside the block's fixed token total; added retention comes from
   targets (own-Lux on H8, own-Nox replay).
4. **Held-out development slices.** The published AHO slices are "diagnostic only, never used for checkpoint
   selection" under the data-v1 and A7 preregistrations, so Milestone 5 builds its own MLX-DEV panel from TRAIN
   groups that N4XF never saw and that are then excluded from every Milestone 5 mixture (below).
5. **HT-DEV v1** (eval-devpanel track) is frozen but not yet cleared for decisions and is English-only; it is not
   used. If it is cleared before the last Milestone 5 soup readout, the artifacts' HT-DEV scores are reported only.
6. **Own-Lux targets for H8** now exist (wave `h-w1`, private revision `75e557f170979bdbc428b6ea698a2047e2d2a5cd`,
   `m3/teachers/lux1/xl/`, `h-w1.targets.jsonl` `679ef009…`, coverage `ecb6dc36…`); N4XF trained H8 on gold only.
7. **Formal runs stay on node B.** Under the 04:55 storage policy non-approved models are never uploaded to HF, so
   the finalists' weights stay on node B and the N4XF soup is re-collected there as the same-image, same-cache
   reference (comparability rule, 23:40).

## Frozen for every arm (identical to N4XF unless stated)

- **Start:** own `llm-semantic-router/Decision-1.0-Nox-4B@cde2a68dbaa557ea65dc458104d410a0802ee259`.
- **Recipe M3F:** full fine-tuning, backbone LR 5e-6 / head LR 5e-5, AdamW (weight decay 0.01, clip 1.0), 5% linear
  warmup then cosine to 10%, one epoch, token batching (32,768 tokens / 64 rows per micro-batch, ≥ 64 rows per
  update), 8,192-token limit without truncation, per row CE + 0.5·Brier + 1.0·KL(target ‖ student) where a target
  exists, checkpoints on `even8`, SELECT700 matrix-v1 selection, seeds 20260926 / 20260927 / 20260928 (N4XF's, so
  every contrast is seed-paired), one GPU per seed, uniform FP32 soup of the three selected checkpoints.
- **Rows outside the block:** exactly N4XF's (`m4/data/m4-xl-full-29m/train.jsonl`, `c7d51219…`, spec
  `specs/m4-xl-full-29m.json` `cf311178…`), including its English A7q / H8 rows. H7 stays gold-only in every arm.
- **Lux sources:** N4XF's composed file (`m4/teacher/m4-xl-full-29m/lux-teacher.jsonl`, `e2ff27ce…`, own
  `Decision-1.0-Lux-9B@bd45a30a…`), the XL-r2 own-Lux waves for added rows, and `h-w1` for H8 rows only.
- **Budget:** N4XF's 29,407,326 native tokens ±1% (whole groups); outside ±1% is a failed preflight.
- **Runtime:** node B GPU3–4, image `sha256:dbe5f32b2263…`, frozen autotune cache
  `/data/dev2/runs/dec/triton-cache/dbe5f32b2263`; SELECT700 / CAL698 from `/data/dev2/runs/dec/m3/data-sel700-cal698`.
- **Code:** exact mirror of the commit adding the Milestone 5 launchers (`v2/dec/ops/m5/`), after this record.

## The multilingual block and the arms

**Block** = the non-English rows (language ≠ `en`; code-mixed hi-en / es-en count as non-English) of A7q, A7k, A7s,
H8 and H5. In N4XF: 14,298 rows, 5,513,920 tokens (18.75%); Noul 71.5% of the block, all answerability / relevance.

| Arm | Rows | Targets on block rows | Targets elsewhere | Contrast |
| --- | --- | --- | --- | --- |
| N4XF (existing control) | N4XF | own-Lux; H8 gold-only | own-Lux; H7 gold | — |
| **N5N** | N4XF (identical ids) | **own-Nox 1.0 replay** | own-Lux (+ `h-w1` on English H8); H7 gold | replay vs N4XF |
| **N5B** | N4XF outside the block; **re-composed block** | own-Lux incl. H8 (`h-w1`) | own-Lux; H7 gold | block composition (+ H8 Lux) vs N4XF |
| **N5BN** | N5B | **own-Nox 1.0 replay** | as N5B | replay vs N5B; with N5N a 2×2 |

**N5B block quotas** (shares of the block total 5,513,920 tokens; each within ±5% relative; the build within ±1%):
answerability / relevance Noul (H5 Noul + H8 Noul) **45%** (N4XF 71.5%); A7q non-English 20%; A7k 6% (if fewer
rows are available after the holdout, the remainder goes to A7q); A7s 9%; H8 Choice (JCommonsenseQA) 9%; H8 Score
(SentiMix Spanglish) 5%; H5 Choice (MTOP) 6%. Construction: per cell, start from N4XF's rows (a shrinking cell keeps
a language-stratified subset of N4XF's whole groups in sha256(`dec-m5-block-v1\0<cell>\0<group_id>`) order); a
growing cell keeps all of N4XF's rows and adds XL-r2 groups N4XF never saw, in the same hash order, stratified by
the cell's XL-r2 language mix, excluding MLX-DEV groups and any row sharing a segment hash with MLX-DEV.

**Own-Nox replay targets:** `v2.dec.teacher_label` with the start model itself (Nox 1.0 above, 8,192 tokens, same
image and cache) on every block row of N5N and N5BN; KL weight 1.0 (as Lux elsewhere); gold CE + Brier unchanged.
Composed first-source-wins (Nox on block rows first, then the Lux sources).

## MLX-DEV (mlx-diag-free multilingual development panel)

Held-out slices of the multilingual training arms: XL-r2 TRAIN groups of H5, H8, A7q, A7k and A7s that are absent
from N4XF, non-English rows only, taken whole in sha256(`dec-m5-mlxdev-v1\0<cell>\0<group_id>`) order. Excluded
from every Milestone 5 mixture. A candidate group is dropped if any of its normalized input segments (fields split on
newlines; NFKC, lower-case, whitespace-collapsed; ≥ 20 characters) occurs in any N4XF row; Milestone 5 builds drop
added rows sharing a segment with MLX-DEV. Targets: Noul — H5 up to 100 groups per language (15 languages, both
twins) and H8 MIRACL up to 80 groups per language (es / fa / fr / hi / zh); Choice — JCommonsenseQA 400 and MTOP
400 rows; Score — A7q 600, A7k 400, A7s 400, H8 Spanglish 300 rows. Rows keep their training rendering and gold.

Metrics (report per language and per source): **Noul-ML** = balanced accuracy (mean of recall on gold-yes and gold-No),
macro over languages, plus predicted-yes rate and gold-No recall; **Choice-ML** / **Score-ML** = accuracy (Score also
±1-level accuracy), macro over sources; **M_dev** = mean of the three. Paired differences use a group-clustered
bootstrap (2,000 replicates, seed 20260929). Baselines read once with the same procedure: Nox 1.0, N4XF seeds and
soup. Training-data diagnostic (report only): Lux and Nox predicted-yes rates and gold agreement on the block's
Noul rows, per language. Caveat, stated with every result: MLX-DEV Noul is answerability / relevance, not paraphrase,
and it is in-distribution for arms that train more on these sub-arms.

## Preflights (stop and record on failure; no rerun to fill GPUs)

1. Builds: tokens within ±1%; ids outside the block identical to N4XF; N5B quotas within tolerance; zero MLX-DEV
   groups or segment conflicts in any mixture; zero rows of the 305 r2-excluded groups.
2. Nox labels: a 300-row re-label is bitwise identical; where rows overlap the M3 own-Nox file (same revision and
   input hash), argmax agreement ≥ 0.97. A failed label check stops N5N and N5BN.
3. Teacher coverage: every row except H7 (and nothing else) carries a target; the trainer re-validates hashes.
4. Per seed (`drive_arm.sh`): zero-step, one-step + reload, preflight receipt PASS before the full run.
5. Cost cap 1.6 GPU-h per seed.

## Selection (development panels only; v3 and `mlx-diag` never used)

1. **Artifact per arm:** the uniform soup if its R is ≥ the seed mean of R, else the median seed by R.
2. **Eligibility:** typed-DEV correct counts per type ≥ 75% of Nox 1.0's (Choice ≥ 345, Noul ≥ 171, Score ≥ 284),
   every answer valid, no type answered with one constant value.
3. **R** = proxy v2 P = 100·√(T_dev·H_pilot) (pilot median), band 8; P_mean3 is reported. An artifact ≥ 8 below
   the N4XF soup's R (same readout) is dropped, as is one whose Noul-ML paired difference vs the N4XF soup has a 95%
   upper bound < 0.
4. **Finalists:** every eligible, non-dropped artifact (at most three; no cross-arm soup). MLX-DEV defines the
   development verdict "multilingual-Noul fix (dev)" = Noul-ML vs the N4XF soup with a paired lower bound > 0 and
   typed-DEV per type not below the N4XF soup by more than 5%; it never ranks siblings.

## Formal runs (finalists only)

Frozen formal runner (`v2/eval/run_same_panel.sh`, gold never mounted), node B, image `dbe5f32b`, package `qwen-full`
at 16,384 tokens, CAL698 16K temperatures under the 23:15 rule. **Reference:** the N4XF soup (node-B build, files
verified against its staged hash list) collected first on a fresh cache copy; every finalist runs on a copy of that
frozen cache. The reference's answers are compared with the node-A formal N4XF run (differences reported). Panels:
v3, public 231, then `mlx-diag`. Predictions are sealed, reported and compared on node A. Paired comparisons: N4XF
reference (the bar), adopted Nox 1.0 and Decider 4B (context). `mlx-diag` is scored only after the v3 report is sealed.

## Conditional 2B port (development trigger only)

Runs only if a 4B finalist earns "multilingual-Noul fix (dev)". If that arm is N5N alone, no 2B run: S2T already
trains own-Sol targets on every row, so it embodies the replay treatment (recorded). Otherwise the block operation is
applied to S2T's M3 recipe (Sol 1.0 start, own-Sol targets, its seeds), fixed in an amendment before any 2B training,
with the released DEV2.0-2B (S2T) formal run as the reference. Three seeds + soup, at most one 2B finalist.

## Budget and storage

Planned ≈ 10.3 GPU-h at 4B (9 seeds × ≈ 0.96, Nox labels ≈ 0.4, soups / readouts / baselines ≈ 0.6, formal incl.
reference ≈ 0.7) plus ≤ 2.5 for the 2B port; cap 15 GPU-h. Node B GPU3–4 only; node A GPU5 stays free for the
eval and release shared leases unless the 2B port needs it. No HF uploads (04:55 policy): seeds, soups and receipts
stay on node B with sha256 lists; any coordinator-approved successor is uploaded directly to its release repo after
the headroom check.
