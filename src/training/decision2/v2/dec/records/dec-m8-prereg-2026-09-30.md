# Decoder Milestone 8 — preregistration (DEV2.0-4B: cross-size distillation from DEV2.0-27B; 2026-09-30)

Written 2026-09-30 ≈17:45 UTC+8, before any M8 GPU job. Assignment: coordinator note 2026-09-30 17:00 (the user's
"aggressive" plan: decoder M8 at 4B starts now with cross-size distillation from DEV2.0-27B and matched controls; it
does not wait for HR2). Development readouts are never release scores; v3 / public 231 / C1 are post-key same-panel
comparisons; `hs1-dev` is a diagnostic. A data lock with every SHA-256 follows before any training launch. Nothing
goes to HF during the milestone; C1 is never opened by this track.

## Goal and bars

A successor to **DEV2.0-4B** that passes successor-rule items 1–8 against the current revision `fadbba4f` (BF16 storage
copy of the N4XF soup, answer-identical to its scored T = 1 run `runs/release/dev2-4b-t1-derived`: post-key v3 63.151,
T .688, H .580; C1 v1.2 baseline 48.38), with the gain coming from generalization rather than typed gains alone.
Paired CIs are also reported against Decider 4B (61.882, `m1-adopt/decider4b`) and Jet v6.2 (60.375, `m2/q5b-jet62`).

## Why this lever

- 4B has the program's largest generalization gap: level with Nox 1.0 on C1, below Jet v6.2 there (−2.06), and −21
  items against Decider 4B on JevBench (17:15 note; a guard, item 7).
- The teacher is much stronger than the student: DEV2.0-27B = A20r (`llm-semantic-router/DEV2.0-27B@5323310327e5…`,
  identity `2e07451107a2…`) scores v3 72.36 / C1 57.56 against 4B's 63.15 / 48.38. Earlier teacher attempts (AutoJev
  → 9B, own-Lux or AutoJev → 27B) failed with a much smaller teacher–student gap (06:40, 06:10 notes); this is a new
  test, and its matched controls separate "a stronger teacher" from "more steps".
- DEV2.0-4B already came from distillation: N4XF = full fine-tuning from own Nox 1.0 with CE + 0.5·Brier +
  1.0·KL(own Lux 1.0 ‖ student) on the XL r2 mixture (M4). M8 changes the teacher, not the recipe.

## Design choice: a continuation top-up of each released soup member (not new seeds)

- **Evidence.** 9B M7 showed that a ~6.2M-token continuation of each soup member, then the re-soup, moves behaviour
  strongly at about a tenth of a new seed's cost (PN1 dev accuracy .656 → .965 at member level, 0.35 GPU-h per member
  at 9B). At 4B a new 29.4M-token seed costs ≈ 1.0 GPU-h and a 43.5M one 1.35 (M7); retraining would also re-draw the
  4B seed-to-soup variance, which is as large as every M5–M7 arm effect, while a top-up keeps the released soup's three
  members and pairs every arm by member and seed.
- **Members:** the three members of the released N4XF soup (node B `m4/arms/full/m4-N4XF-s{1,2,3}`, SELECT BEST
  checkpoints 690 / 676 / 492; model SHA-256 `670ba59b…` / `993a5ba1…` / `d2399ce3…`). Member k continues on the same
  top-up slice in every arm with seed 20260930 + k, so D1, D2 and C are paired by member, seed and row order.
- **Dose:** one pass over a ~6.2M-native-token slice (≈21% of the recipe's 29.4M tokens; ≈ 190 updates, about a
  quarter of a seed's updates). One dose only; a larger dose is a later milestone's question.

## Teacher targets: A20r's decision distributions (scored runtime, T = 1)

- **Rows:** exactly the top-up slice (below): typed and human rows of the 4B training mixture. Only the slice is
  labeled: the unchanged scored collector reads one prompt at a time (A20r's formal run: 8,378 prompts in 1,459 s), so
  the slice costs ≈ 0.8 GPU-h and the whole 58.7k-row mixture ≈ 3.3; no arm uses rows outside the slice.
- **Runtime = A20r's scored runtime, unchanged:** `v2.27b.typed_collect_kernel` (the formal and C1 collector; it runs
  `training.model.infer`), image `dbe5f32b`, `HIP_FORCE_DEV_KERNARG=1`, a fresh copy of the formal run's persisted
  autotune cache (`f474e2e9…`), 32,768 tokens, the package's T = 1 `calibration.json` (`518e19cd…`), node A: the frozen
  C1 package (`c1pkg/package/DEV2.0-27B`, manifest `0d7953cc…`, identity `2e07451107a2…`; the released revision
  `5323310` differs from it only in `README.md`) on the pinned base `Qwen/Qwen3.8-27B@1d4bf0f2`.
- **Rows as prompts:** each slice row becomes one gold-free prompt `{id, state, questions: {q: {type, instructions,
  criteria}}}` (Choice / Noul criteria = the row's options in order; Score criteria = the level descriptions, keys
  0..n−1). The build checks on CPU that `question_to_row` of every prompt renders **the same prompt text and option
  keys** as the TRAIN row (`training.model.decision_model.segments`); a row that does not convert is not labeled and is
  reported. Gold never enters a prompt.
- **Runtime parity preflight:** each shard's input starts with the first 80 typed-final gold-free prompts; their
  predictions must equal A20r's stored formal predictions (`typed-final.predictions.jsonl` `ea573b2d…`: same answer, max
  probability drift ≤ 1e-4), or the shard fails and is not relabeled. Those 80 outputs are never targets.
- **Conversion:** `teacher_probs` = the scored probabilities (Choice / Score: the answer's `probabilities`; Noul:
  true = `noul`, false = 1 − `noul`), keyed by the TRAIN row's option keys and bound to its `input_sha256`.
- **Provenance and coverage (reported in the lock):** teacher identity, package manifests, runtime, cache digests,
  per-shard prediction hashes, rows labeled / slice rows, argmax agreement with gold by type and pool class, mean
  max probability, and argmax agreement with the own-Lux targets on the rows that have them. Own model; no third-party
  caveat. Three shards on node A GPU0 / GPU1 (shared leases, as allocated) and GPU5.

## Data

- **Base:** N4XF's mixture `m4-xl-full-29m` (`c7d51219…`, 58,742 rows) minus the quarantined 4B near-match group
  (3 HotpotQA rows, 17:15 note; `ops/m7/specs/m7-quarantine-groups.json`) = M7's 4B base (58,739 rows, 29.40M tokens).
- **Top-up slice:** whole groups of the base, `build_template_s.select_groups` (strata source × task type ×
  language, seed-keyed hash order), seed `dec-m8-topup-v1`, budget 6,200,000 native tokens (Nox tokenizer
  `363c4a5e`).
- **Human-rated rows S** (for D2): the 9B M6 frozen soft-target rule copied verbatim
  (`lux9b/specs/m6-ka-x60-ajS.json` → `soft_target_rule`; `ops/m8/specs/m8-human-rated-rule.json`), applied to each
  row's recipe pool. The only change is the pool alias `A0s` → `A0s-strict` (the same 6,547 rows under the 9B name).
  S = rows whose gold is a human rating or judgment of the text (A7q / A7k / A7s, V1:A1, V1:A6h, the listed V1:A5 /
  H1 / H5 / H6 / H8 sources, A0s GoEmotions / SNLI / human-labelled stage-3 replays). **Typed rows = every other
  row** (A7 Stage curricula, programmatic / verifiable rows, and human-annotated QA / evidence rows whose gold is an
  answer fact rather than a rating). Pools come from M7's composed 4B C file (`train.ids.jsonl` of `m7-4b-C`, whose
  base rows are this base); the build refuses a row without a pool.
- **Teacher files (one per arm, `v2.dec.compose_teacher`-compatible records):**
  - **C:** N4XF's own-Lux targets (`e2ff27ce…`) on the slice rows, H7 / H8 rows gold-only — the released recipe exactly.
  - **D1:** A20r targets on every slice row (H7 / H8 included).
  - **D2:** A20r targets on the slice's S rows only; every typed row is gold-only.
- **Guards** (build and lock): `v2.common.eval_only` on every input; isolation from SELECT700 / CAL698, `hs1-dev`,
  HT-DEV v2 and Score5-typed-DEV prompts (ids and input hashes); C1 registry source guard; no typed-FINAL-family
  generator draws and no panel items (16:40 rule); an exposure receipt (`v2.eval.overlap_effects exposure`, r2 payload
  `2194716a…`) for the slice file; teacher coverage exactly as specified per arm; unique ids.

## Arms (each 3 members, then the uniform FP32 soup)

Common to every arm: `v2.dec.train_dec --init decision2` from the member's checkpoint (keeps the root
`full_training_source` = Nox 1.0, so soups with N4XF stay defined), full fine-tuning, the N4XF recipe's
hyper-parameters (backbone / head peak LR 5e-6 / 5e-5, AdamW wd .01, clip 1.0, warmup .05 then cosine to 10%, fresh
optimizer state, one epoch, token batching ≤ 32,768 tokens / ≤ 64 rows, 64 rows per update, max length 8,192,
`--brier-weight 0.5`), `--teacher-partial`, **λ = 1.0** (`--teacher-kl-weight 1.0`: CE + 0.5·Brier on gold for every
row + 1.0·KL(teacher ‖ student) on each row with a teacher record; the recipe's value, and M4 found KL 2.0 no better),
and **one checkpoint at the final update** (SELECT700 read at the start and the end and recorded; it selects nothing).

| Arm | Targets on the slice | Contrast |
| --- | --- | --- |
| **D1** | A20r KL on all rows + gold | D1 − C: the teacher on every row |
| **D2** | A20r KL on human-rated rows S + gold; typed rows gold only | D2 − C: the teacher on human rows, no teacher on typed rows; D2 − D1: typed-row KD |
| **C** (matched control) | own-Lux KL (N4XF's targets) + gold, H7 / H8 gold only | the N4XF recipe continued at equal tokens |

Member 1 of each arm runs the `drive_arm.sh` preflights (zero-step, one-step, `preflight_dec` load parity + one-step
reload); members 2–3 run the full continuation directly (same code path, different start and seed).

## Early stop after member 1 (`ops/m8/m8_rules.py early`)

As in 9B M7, the early rule is read in-process on node B so the chains decide without waiting for node-A scoring:
arm X (D1 or D2) continues to members 2–3 only if **X-m1's final SELECT700 family-macro accuracy ≥ C-m1's − 0.02**
(typed protection; both read by the trainer at the final update). If X-m1 or C-m1 does not finish, X stops. The
human-transfer and typed screens act at the line level below, where every point is read. A stopped arm runs no
further member and no line, and is reported. C always runs (control and a candidate line).

## Lines and the development rule (`ops/m8/m8_rules.py alpha`; never v3, never C1, never mlx-diag, never public 231)

- θ(α) = α·S_arm + (1 − α)·I for α ∈ {1, ⅔, ⅓} (`v2.dec.soup` with repeated members: [A], [I, A, A], [I, I, A]),
  I = the N4XF soup (the released weights, node B `m4/soup/N4XF/build/N4XF-soup`, list `df602ca9…`).
- Every point and `I` are read the same way: `v2.dec.infer_dec`, node B GPU3 / GPU4, image `dbe5f32b`, 16,384 tokens,
  T = 1: typed DEV (1,600), CSS pilot (1,430; report only), HT-DEV v2 (1,944), Score5-typed-DEV (800). `I` reuses M7's
  node-B readouts of the same weights (typed DEV, CSS pilot, HT-DEV v2; files hash-checked); its Score5-typed-DEV is
  read fresh. HT-DEV v2 and Score5-typed-DEV are scored on node A (gold stays there; gold-free predictions relayed).
- **A point is eligible** if, against `I` (typed-DEV counts: Choice 501 / 800, Noul = `rule_precedence` 264 / 400,
  Score = `set_reconciliation` 362 / 400; families `attribute_gate` 341, `transition_table` 160):
  1. **typed floors:** every type c_t ≥ c_t,I − 0.03·n_t (Choice ≥ 477, Noul ≥ 252, Score ≥ 350) and every typed-DEV
     family F_f ≥ F_f,I − 0.10;
  2. **Noul floor:** `rule_precedence` c ≥ c_I − 0.01·400 (≥ 260);
  3. **Score floor:** Score5-typed-DEV check half has no COLLAPSE, and no WARN unless `I`'s check half has WARN (typed
     DEV's Score family has three levels, so the 5-level screen comes from Score5-typed-DEV);
  4. **HT-DEV v2 not FLAG:** ΔH_dev2 vs `I` > −0.02 (`v2.dec.ops.m7.m7_htdev2` paired bootstrap, verdict at ±0.02).
- **Pick per line:** among eligible points, those with an HT-DEV v2 GAIN (ΔH_dev2 ≥ +0.02) first; then the largest α.
  No typed-gain requirement (M6–M7: development typed gains did not carry to typed FINAL). A pick whose proxy
  P = 100·√(T_dev·H_pilot) is ≥ 8 below the best pick's is dropped.
- **Finalists (≤ 3):** slot 1 = the D1 pick, 2 = the D2 pick, 3 = the C pick; a line without a pick, or a stopped arm,
  leaves its slot empty. The newest COORDINATION notes are re-read before the finalists are fixed.

## Diagnostics (read and reported, never selected on)

- `hs1-dev` (quote adoption, ideal .50; false yes on unmet conditions, ideal 0) for `I`, every arm soup and every
  finalist, scored against `I` with `v2.data.hs1.validity` (I on the M7 path: adopt .775, false yes .262).
- Matched-token contrasts at α 1: D1 − C, D2 − C, D2 − D1 on every readout; the CSS pilot and P.
- Distillation in family: the trainer's per-update teacher-KL (first vs last updates) per member, and the teacher
  files' agreement statistics (A20r vs gold, A20r vs own-Lux) from the lock; no extra GPU readout.

## Formal (finalists only)

The M7 procedure with `M6_PREFIX=m8`: CAL698 16K fit and the 23:15 calibration rule, an 8-item smoke, then typed
FINAL, CSS15 and public 231 at 16,384 tokens on node B (`dbe5f32b`, a copy of `formal/m5/cache-frozen` `f6d0f920…`),
relayed gold-free and scored on node A; `mlx-diag` on node B against the node-B N4XF reference; paired CIs (5,000
draws) vs DEV2.0-4B's T = 1 run, adopted Nox 1.0, Nox 1.0 16K, Decider 4B and Jet v6.2; human transfer per task.

## Successor rule, choice, item 8 and card

- Items 1–8 exactly as in M7 (bars above): 1 v3 lower bound > 0 vs `dev2-4b-t1-derived`; 2 H not significantly
  below; 3 no type collapsed (`gates types`); 4 card-eligible mlx-diag not significantly below; 5 tier gates (vs the
  adopted Nox 1.0 lower bound > 0, v3 ≥ 55.7, H not significantly below Decider 4B / Jet v6.2); 6 no overlap exposure
  (the slice's receipt; points containing `I` inherit its disclosed exposure: none); 7 `gates public231` vs the bar not
  REGRESSION; 8 the C1 post-key guard through the eval custodian.
- **Choice among passers of items 1–7:** the highest v3 lower bound vs the bar; within 0.25 of it, an HT-DEV v2 GAIN,
  then the higher formal ΔH (09:50 hint). Only that candidate goes to the custodian: frozen package on node A, a
  `dev2-c1-postkey-spec/1` spec and the C1 spec notes (the slice is a subset of N4XF's own TRAIN, already under the
  4B C1 baseline's independence check; A20r targets add no text).
- **Card for any successor:** names DEV2.0-27B (A20r, `5323310`) as the distillation teacher (D arms), keeps own
  Lux 1.0 as the recipe teacher of the continued members, and discloses every per-type / task / language regression.

## HR2

If the HR2 human-rated data lands during the milestone, it may be added only by an amendment committed before its
first readout (COORDINATION 17:00); otherwise it is not used.

## Budget, GPUs, chains and stop rules

- **Cap 24 GPU-h** (wall-clock × GPUs, including preflights, labels, readouts, CAL fits, smokes and formal).

  | Attempt | Cap (GPU-h) |
  | --- | ---: |
  | A20r labels (3 shards incl. load and smoke) | 1.5 |
  | Each arm (3 members; member-1 preflights) | 1.4 |
  | Lines, references, diagnostics | 2.5 |
  | Formal (≤ 3 finalists incl. CAL fits, smokes, mlx-diag) | 1.5 |
  | Contingency (never for reruns) | 2.0 |

- At 20 GPU-h cumulative no new training member starts; formal work finishes within the cap.
- A failed preflight stops that arm (no rerun, no replacement member); an arm stops at its cap; an arm with fewer than
  two finished members has no soup (a two-member soup is disclosed); a failed labeling shard stops D1 and D2 (C
  continues); a failed smoke, calibration or collection stops that finalist. Failed arms are never rerun.
- **GPUs:** node A GPU0 / GPU1 (shared leases, labels only) and GPU5 (label shard 3); node B GPU3 / GPU4 (members,
  readouts, formal collections); node A CPU scores HT-DEV v2, Score5-typed-DEV and formal runs.
- **Chain rule:** every script runs from an exact mirror (`mirror_to_node.sh`); chains are uploaded, their size and
  SHA-256 verified, then launched in a separate step; liveness by PID and container name, never `pgrep -f`; the first
  log line is confirmed. Poll every ≤ 30 min; `records/m8-state.md` is kept current and the worker hands off near 5 h.

## Code (committed before use)

`v2/dec/ops/m8/`: `m8_data.py` (slice, S split, prompts, C teacher, conversion check, guards) with `m8-prep.sh`
(node B build + exposure receipt), `m8_teacher.py` (A20r predictions → teacher files, parity smoke, provenance),
`m8-label.sh` (node A shards), `m8-chains.sh` (members, early rule, soups), `m8_gpuh.py`, `m8-lines.sh` (points,
readouts, hs1-dev), `m8-score.sh` + `m8_rules.py` (HT-DEV v2 / Score5-typed-DEV scoring on node A; early, alpha,
finalists), `m8-relay.sh`, `specs/m8-human-rated-rule.json`; tests `v2/dec/tests/test_m8.py`. The M6 / M7 formal and
successor scripts are reused with `M6_PREFIX=m8` through an M8 copy of the M7 formal wrapper (committed before use).
No shared module changes are planned; any becomes a separate commit with tests.
