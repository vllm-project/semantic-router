# Decoder Milestone 9 — preregistration (HR2 efficacy pilot at 4B; PILOT, NOT RELEASABLE; 2026-10-01)

Written 2026-10-01 ≈11:10 UTC+8, before any M9 GPU job. Assignment: COORDINATION 2026-10-01 10:30 ("4B HR2 efficacy
pilot, NOT releasable: full seeds from Nox + N4XF + an HR2 block vs a matched control, screened with HT-DEV v2;
node A GPU6–7; 16 GPU-h") and the 2026-09-30 20:20 program conclusion (stop teacher and recipe-top-up arms; HR2 is
the remaining lever; add it as a block with matched controls, screened with HT-DEV v2). Development readouts are
never release scores; v3 / public 231 are post-key same-panel comparisons; the HR2 DEV slice, the CSS pilot and
`hs1-dev` are diagnostics. A data lock with every SHA-256 follows before any training launch. Nothing goes to HF;
C1 is never opened by this track.

**Status of every model trained here: pilot, NOT a release candidate.** HR2 is flagged `release_safe: false` (its
blind review found 13 / 216 = 6.02% label errors; data record `hr2-results-2026-10-01.md`). No M9 model is handed off
as a successor, sent to the C1 custodian or uploaded. If HR2 helps, a release-candidate retrain follows on HR2-r2
(the data track's fixed rebuild) under its own preregistration.

## Question and endpoints

Does adding HR2 (new licence-clean human-rated TRAIN rows, gold labels) as one block to the released 4B recipe move
human transfer at matched tokens?

- **Primary (development):** ΔH_dev2 = HT-DEV v2 H(H9 soup) − H(C9 soup) at α = 1, paired bootstrap 95% CI (the
  matched-token HR2 effect), and ΔH_dev2(H9 soup − DEV2.0-4B) with the 04:10 verdict (FLAG ≤ −0.02 < TIE < +0.02 ≤
  GAIN).
- **Secondary (development):** the same contrasts at α = ½; typed DEV per type and family; Score5-typed-DEV flags and
  top share; CSS pilot (three-task mean, median) and the proxy P; the HR2 DEV slice per family; `hs1-dev` quote
  adoption and unmet-condition false yes.
- **Formal (one development passer only):** post-key same-panel CSS15 human transfer H, typed FINAL and v3 of the
  H9 pick vs DEV2.0-4B (paired 95% CIs), labelled **"M9 pilot, not a release candidate"**.

## Why this design (evidence)

- 4B has the weakest generalisation: level with Nox 1.0 on C1 and below Jet v6.2 there; teacher distillation and
  recipe top-ups all failed (M6, M6b, M7, M8). M8: continuing the released weights at the recipe's peak LR costs
  typed FINAL and human transfer (the matched top-up control was the worst finalist, −5.20), and teacher soft targets
  on human rows lowered HT-DEV v2 at every size. So M9 trains **full seeds from Nox 1.0** (the recipe that produced
  DEV2.0-4B) rather than top-ups, puts **gold labels on HR2 rows with no teacher**, and keeps the recipe's own-Lux KL
  exactly where the recipe has it.
- **Add, don't substitute** (9B M6: HS1 substitution hurt Noul). HR2 is added whole; the control adds the same tokens
  of the recipe's own next rows, so "new human-rated data" is separated from "more tokens" (M6b / M7: more recipe
  tokens alone are not a lever).
- **Three seeds per arm** with the same seed triple as N4XF: the 4B seed-to-soup variance is as large as every M5–M7
  arm effect, and a shared triple pairs the arms by seed.
- **HT-DEV v2** is the human-transfer screen (04:10: within-tier sign agreement 0.843 with formal ΔH; FLAG at −0.02
  agrees in sign 93% of the time). It does not predict C1, so no C1 claim is made from it.

## Data

- **Base:** N4XF's mixture `m4-xl-full-29m` (`c7d51219…`, 58,742 rows) minus the quarantined 4B near-match group
  (3 HotpotQA rows; `ops/m7/specs/m7-quarantine-groups.json`, `29dadde1…`) = M7's 4B base, 58,739 rows / 29,404,539
  tokens (Nox / Eos tokenizer `363c4a5e`).
- **HR2 block:** `m5/hr2/hr2.train.jsonl` at private dataset revision `afc3bc1e1d6849058f6fafdfbc3dfe007d067400`
  (`0fd9b2db…`; 27,697 rows, 22,336 groups; Choice 10,207 / Noul 10,346 / Score 7,144; ≈16.75M tokens by the data
  track's count, recounted with the decoder tokenizer at build). Whole, once, gold only (no teacher record; pool
  label `hr2`).
- **Control filler:** whole groups of `m6-xl-full-59m` (`160812e2…`, the released XL r2 recipe's next rows) that share
  no id, group or input hash with the base, chosen by `build_template_s.select_groups` (strata source × task type ×
  language) with M7's 4B filler seed `dec-m7-fill-4b-v1`, the budget bisected until the C9 file's tokens are closest
  to H9's. With the same seed the chosen set grows monotonically with the budget, so C9's filler contains M7's N7C
  filler (checked in the lock); N7C (M7, node B, same seeds, 43.5M tokens) is therefore a reference sibling of C9.
- **Arms:**

  | Arm | TRAIN | Targets |
  | --- | --- | --- |
  | **H9** | base + HR2 block | own-Lux 1.0 KL on base rows (N4XF's composed targets `e2ff27ce…`), H7 / H8 rows gold only, **HR2 rows gold only** |
  | **C9** (matched-token control) | base + filler of \|HR2\| tokens | own-Lux KL on base rows and on filler rows (`lux-all-59m` `a1bafad5…`), H7 / H8 rows gold only — the N4XF recipe at equal tokens |

- **Lock checks** (`ops/m9/m9_lock.py`, before any training): TRAIN hashes equal the composer's; unique ids;
  quarantine absent; |C9 − H9| / H9 ≤ 0.5%; C9's filler contains N7C's filler groups (reported; not a gate); every
  HR2 row of H9 is a row of the pinned HR2 file and every HR2 TRAIN row is in H9; the composed teacher covers every
  row outside the gold-only pools (`base-gold`, `fill-gold`, `hr2`) and no `hr2` row has a target; empty exposure
  receipt (`v2.eval.overlap_effects exposure`, r2 payload `2194716a…`) per TRAIN file; the C1 registry source guard
  (no TRAIN source names a C1 registry source); `v2.common.eval_only` on every input; no TRAIN row's canonical state
  equals a prompt state of SELECT700, CAL698, typed DEV, the CSS pilot, HT-DEV v2, Score5-typed-DEV, `hs1-dev` or the
  HR2 DEV slice (counts reported per panel; a hit in an HR2 row fails the lock). No typed-FINAL-family generator draws
  and no panel items are added (16:40 rule). A failed lock stops the milestone before any training.

## Training (both arms identical except the TRAIN / teacher files)

`v2.dec.train_dec` from **Nox 1.0** `llm-semantic-router/Decision-1.0-Nox-4B@cde2a68d…` (`--init decision1`), the
N4XF hyper-parameters: full fine-tuning, one epoch, CE + 0.5·Brier on gold for every row + 1.0·KL(teacher ‖ student)
on rows with a teacher record (`--teacher-partial --teacher-kl-weight 1.0`), backbone / head LR 5e-6 / 5e-5, AdamW wd
.01, warmup .05 then cosine, token batching ≤ 32,768 tokens / ≤ 64 rows, 64 rows per update, max length 8,192,
`even8` checkpoints with SELECT700 `matrix-v1` selection per seed (SELECT700 `32a4352d…`, CAL698 `19cc1a8c…` from
`m3/data-sel700-cal698`, hash-checked on node A), seeds **20260926 / 20260927 / 20260928**, `drive_arm.sh` per seed
(zero-step, one-step, `preflight_dec` load parity + one-step reload, full run, postrun), and the uniform FP32 soup of
the seeds' BEST checkpoints as the arm artifact (`v2.dec.soup`).

- **Node / image:** node A GPU6 (`renderD177`) and GPU7 (`renderD185`), image `dbe5f32b` (the 4B lineage's training
  and formal image; present on node A), node-A training autotune cache. `drive_arm.sh` gains the node-A GPU6–7 render
  nodes and an image override; `launch.sh` gains an opt-in `HIP_FORCE_DEV_KERNARG=1` (used for node-A readouts and
  formal, as in the node-A A20r label and 2B formal precedents). Both are small decoder-only changes with tests.
- **Schedule** (seeds interleaved so each arm's seeds use both GPUs): GPU6 H9-s1 → C9-s2 → H9-s3; GPU7 C9-s1 → H9-s2
  → C9-s3. The soup is built by whichever chain finishes an arm's last seed.

## Early stop (after seed 1 of both arms; `ops/m9/m9_rules.py early`, read by the chains on node A)

Both chains wait for H9-s1 and C9-s1 to reach a terminal marker; each reads HT-DEV v2 of its own s1 BEST (16K,
T = 1, the M9 readout path below), then the rule runs once:

- **E1-typed:** H9 stops if H9-s1's BEST SELECT700 family-macro accuracy < C9-s1's − 0.03.
- **E1-human:** H9 stops if ΔH_dev2(H9-s1 − C9-s1) ≤ −0.03 **and** ΔH_dev2(H9-s1 − DEV2.0-4B) ≤ −0.02 (FLAG).
- If H9 stops, C9 stops too (a control without its arm has no purpose); the pilot then reports the seed-1 pair and
  ends without lines or formal. If C9-s1 or H9-s1 did not finish, the rule is not applied and the other arm continues.
- Otherwise both arms run seeds 2–3.

## Development readout path (node A) and references

- Every point is read with `v2.dec.infer_dec` through `launch.sh`, node A GPU6 / GPU7, image `dbe5f32b`,
  `HIP_FORCE_DEV_KERNARG=1`, 16,384 tokens, T = 1, on one persistent M9 readout cache: a copy of node B's decoder
  autotune cache for `dbe5f32b` (the cache every M6–M8 4B node-B readout used), moved over the direct node link and
  checked by content manifest. Readouts are co-tenants (refused below 60 GB free VRAM); wall seconds are recorded.
- **`4b-I` = DEV2.0-4B's weights:** the N4XF soup staged on node A (`m4/formal-candidates/staging-e8656221/m4/
  N4XF-soup/checkpoint`, per-file list `df602ca9…`, identical to node B's soup and to the current revision's
  weights). It is read fresh on the M9 path on every panel. **Parity (reported, not a gate):** its typed DEV, CSS
  pilot, HT-DEV v2 (M7 node-B readouts), Score5-typed-DEV (M8) and `hs1-dev` predictions are compared with the stored
  node-B readouts (answers and maximum probability drift), and its HT-DEV v2 with the eval track's reference
  `dec-m4-N4XF-soup`. Every M9 contrast is paired within the M9 path, whatever the parity outcome.
- Panels (gold-free prompts in the node-A decoder panel directory; gold stays under `/data/dev2/private`): typed DEV
  (1,600), CSS pilot (1,430), HT-DEV v2 (1,944), Score5-typed-DEV (800), `hs1-dev` (2,396; `4b-I` and the α 1 soups
  only), and the **HR2 DEV slice** (`m5/hr2/hr2.dev.jsonl@afc3bc1e`, `697c3142…`, 1,615 rows, 9 families): each row
  becomes one gold-free prompt whose question renders exactly as the row (`ops/m8/m8_data.to_prompt`), gold to
  `/data/dev2/private/dec/m9/` (mode 600).

## Lines and the development gates (`ops/m9/m9_rules.py`; never v3, C1, mlx-diag or public 231)

- Points per arm X ∈ {H9, C9}: α 1 = the X soup; α ½ = `v2.dec.soup` of [I, X soup] (`4b-X-a1`, `4b-X-a1_2`).
- **A point is eligible** if, against `4b-I` on the M9 path (the M8 gates):
  1. **typed floors:** every type c_t ≥ c_t,I − 0.03·n_t and every typed-DEV family F_f ≥ F_f,I − 0.10;
  2. **Noul floor:** `rule_precedence` c ≥ c_I − 0.01·400;
  3. **Score floor:** Score5-typed-DEV check half without COLLAPSE, and without WARN unless `4b-I`'s check half has
     WARN;
  4. **HT-DEV v2 vs DEV2.0-4B: GAIN or TIE** (ΔH_dev2 > −0.02).
- **Formal candidate = the H9 line's pick:** among eligible H9 points, a GAIN first, then the larger α. At most one
  formal run; the control line is read and gated for the report but is never a formal candidate. No pick → no formal
  run, and the pilot reports development only. The newest COORDINATION notes are re-read before the pick is fixed.

## Formal (the H9 pick only; pilot, not a release candidate)

The M8 formal procedure on node A instead of node B: CAL698 16K fit with the 23:15 calibration rule, an 8-item smoke,
then typed FINAL, CSS15 and public 231 at 16,384 tokens with the frozen runner, `mlx-diag`; image `dbe5f32b`,
`HIP_FORCE_DEV_KERNARG=1`, node A GPU6 / GPU7, each run on its own copy of node B's `formal/m5/cache-frozen`
(`f6d0f920…`; `cache-frozen-mlx` `65d7d38f…` for `mlx-diag`) moved over the direct node link; gold-free seal, then
scoring and paired CIs (5,000 draws) on node A.

- **Formal-path parity first:** the incumbent's weights (`4b-I`, T = 1) are collected once on that path (typed FINAL,
  CSS15, public 231) and compared with DEV2.0-4B's scored T = 1 run `runs/release/dev2-4b-t1-derived` (node B). With
  0 answer differences the stored run is the bar, as for every earlier 4B candidate; otherwise the node-A incumbent
  run is the paired reference and both comparisons are reported.
- **Reported (information only; nothing gates a release):** v3, T, H with paired CIs vs DEV2.0-4B (the bar run or the
  node-A reference), vs the adopted Nox 1.0, Decider 4B and Jet v6.2; human transfer per CSS15 task; typed FINAL C / N
  / S; public 231 by tier (`gates public231`, never selected on); card-eligible `mlx-diag` vs the node-B N4XF
  reference; the successor-rule items 1–7 as a read-out. Item 8 (C1) is not run: the model is not a candidate, and
  HR2 would need the custodian's content recheck first.

## Recommendation rule (fixed now)

- **HR2 is a lever → recommend an HR2-r2 release-candidate milestone at 4B** if the α 1 matched contrast ΔH_dev2(H9 −
  C9) ≥ +0.02 with its 95% CI above 0, an H9 point passes the development gates, and the formal ΔH vs DEV2.0-4B (if
  run) has a point estimate ≥ 0.
- **Weak / conditional** if the matched contrast is positive with its CI above 0 but below +0.02, or the formal ΔH is
  negative despite a development gain, or H9 fails only a typed / Score / Noul floor while its human-transfer contrast
  is positive: an HR2-r2 milestone only with a design change aimed at the failure (dose, proportional Score rows).
- **Not a lever** if ΔH_dev2(H9 − C9) ≤ 0, or H9's soup FLAGs vs DEV2.0-4B without a positive matched contrast.
- Other tiers (2B, 0.8B, 9B) inherit the 4B verdict as a prior, adjusted by each tier's recorded failure mode (2B:
  Score-head shift under human Score rows; 9B: PAWS-X yes-bias; 0.8B: development gains that did not carry to
  formal).

## Budget, GPUs, chains and stop rules

- **Cap 16 GPU-h** (wall-clock × GPUs, including preflights, postruns, soups, readouts, early-rule reads, parity and
  formal).

  | Attempt | Cap (GPU-h) |
  | --- | ---: |
  | H9 (3 seeds; ≈ 46M tokens each, ≈ 1.45 GPU-h per seed from M7's 1.35 at 43.5M) | 4.8 |
  | C9 (3 seeds) | 4.8 |
  | Early-rule reads, references, lines, diagnostics | 1.5 |
  | Formal (parity run, CAL fit, smoke, collection, mlx-diag) | 1.2 |
  | Contingency (never for reruns) | 2.0 |

- At **14 GPU-h** cumulative no new seed starts; formal work finishes within the cap.
- A failed preflight stops that arm (no rerun, no replacement seed); an arm stops at its cap; an arm with fewer than
  two finished seeds has no soup (a two-seed soup is disclosed); a failed soup, readout, smoke, calibration or
  collection stops that step and is recorded. Failed arms are never rerun.
- **GPUs:** node A GPU6–7 only (9B-owned, lent), lease `owner` files kept current; node A CPU builds data, scores and
  applies the rules. Node B is read only (inputs and stored readouts, moved over the direct node link).
- **Chain rule:** every script runs from an exact mirror (`mirror_to_node.sh`); chains are launched in a separate step
  after the mirror is verified; liveness by PID and container name, never `pgrep -f`; the first log line is
  confirmed. Poll every ≤ 30 min with a state commit (`records/m9-state.md`); hand off near 5 h.

## Code (committed before use)

`v2/dec/ops/m9/`: `m9_compose.py`, `m9_lock.py`, `m9-prep.sh` (node-A data build: pinned HR2 / XL id fetches,
compose, teachers, exposure, lock), `m9-chains.sh` (seeds, early rule, soups), `m9-soup.sh`, `m9_gpuh.py`,
`m9-lines.sh` (references, early reads, lines, diagnostics, scoring on node A), `m9_hr2dev.py` (HR2 DEV prompts and
per-family scoring), `m9_rules.py` (early, HT-DEV v2 pairs, Score5-typed-DEV, gates, pick); tests
`v2/dec/tests/test_m9.py`. The formal step reuses the M6 / M8 formal scripts with a node-A 4B option, committed
before use. Changes to `drive_arm.sh` / `launch.sh` are small separate commits with tests; no shared module changes.
