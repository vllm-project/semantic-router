# Decoder Milestone 8-small — preregistration (DEV2.0-27B → DEV2.0-2B / DEV2.0-0.8B distillation; 2026-09-30)

Written 2026-09-30 ≈18:05 UTC+8, before any M8-small GPU job. Assignment: COORDINATION 2026-09-30 17:05 (user
directive: push 27B, 9B, 4B and also 2B and 0.8B; decoder M8-small = "the same distillation design as 4B M8").
Worktree `vllm-sr-dev2-dec-small`, branch `xunzhuo/decision-2-training-dec-small`, gist `04b-decision-2-dec-small.md`.
Development readouts are never release scores; v3 / public 231 / C1 are post-key same-panel comparisons;
`hs1-dev` and `mlx-diag` are diagnostics. Nothing goes to Hugging Face. This track never opens C1.

**4B M8 design.** The 4B M8 preregistration had not been pushed when this record was frozen (checked 17:05–18:00 on
`origin/xunzhuo/decision-2-training-dec` and in its worktree). The design below follows the assignment's arm
definitions and the 9B M7 top-up pattern (`lux9b-m7-prereg-2026-09-30.md`). **λ rule, fixed now:** if the 4B M8
prereg lands before this milestone's first *training* job and sets a different A20r KL weight for its D arms, that
weight replaces λ = 1.0 below by an amendment committed before the first training job. The teacher targets do not
depend on λ, so labeling may start first.

## Goal, incumbents and bars

Per tier, a successor that passes successor items 1–8 against the current revision, with its gain carried by human
transfer (the program diagnosis: typed gains do not carry to fresh human-labeled data; M5–M7 found no successor at
2B / 0.8B, and development typed gains did not carry).

| Tier | Current revision (incumbent I) | Weights identity | Scored T = 1 run (bar, node A) | v3 (T / H) | C1 v1.2 baseline |
| --- | --- | --- | --- | ---: | ---: |
| 2B | `DEV2.0-2B@a53cf66a0d9d492a84b6617b61e7ce35fcd03af0` | BF16 copy `32872f29…` of the S2T soup `073bd1f2…` (0 answer changes) | `runs/release/dev2-2b-t1-derived` | 53.437 (.5434 / .5255) | 45.70 |
| 0.8B | `DEV2.0-0.8B@bede7938a8c209c09f27400b79eed57948d6b75e` | BF16 copy `3f02f0e5…` of the E8F soup `60356482…` (0 answer changes) | `runs/release/dev2-0p8b-t1-derived` | 50.236 (.5734 / .4401) | 40.17 |

Teacher: **DEV2.0-27B = A20r** (`llm-semantic-router/DEV2.0-27B@5323310327e52d4eadd119cd10accac9b106c97d`; node B
package `/data/dev2/runs/27b/M4-A20r-soup/package`, checkpoint `soup/checkpoint` model sha `2e074511…`, rank-64
adapter on the pinned Qwen3.8-27B, T = 1, 32K). Post-key v3 72.36 (T .896 / H .584), C1 57.56. Peers for the paired
report: 2B Decider 2B 49.499, This-That 1.2 46.112; 0.8B Intern-0.8B 43.535.

## Top-up rows (CPU build; `ops/m8s/m8s_compose.py`)

- **Source = the tier's released recipe, r2-clean:** 2B `m4-v2m-ret-r2` (`1527b38b…`, S2T's recipe minus the 57
  r2-exposed rows); 0.8B `m6-e8f-r2clean` (`f9f3c022…`, E8F's minus the 81 r2 / A7-v3 rows). Every row has already
  been trained on by the incumbent once, so the arms differ only in their targets, and the new TRAIN file's exposure
  receipt (r2 payload `2194716a…`) must be empty. The quarantined 4B near-match group
  (`m2:multihop:41be17b19b01045b01fd0dfc`) is dropped if present.
- **Partition (frozen):** a row is **typed** if its `source` is one of `dec10:generated_stage4_v2`,
  `dec10:generated_stage1_3`, `decision2_verifiable_v2_a2`, `decision2_verifiable_v2_a4v2h`,
  `decision2_verifiable_v2_a6`, `legacy:stage4-general-composition-v2`, `decision2_targeted_programmatic_v1`,
  `decision2_programmatic_original_v1`, any other `decision2_*programmatic*` / `decision2_verifiable*` /
  `dec10:generated*` source, or `legacy:stage3_replay` without an `upstream_label`; every other row is **human**
  (public human-labeled sources; `legacy:stage3_replay` rows count only with their upstream human label, the 9B M6
  rule). A group is typed or human by its first row; mixed groups are reported.
- **Budget:** 3.0M human + 3.0M typed native tokens (shared 1.0 tokenizer `06b95093…`, `build_mixture.token_lengths`)
  = ≈6.0M per tier (9B M7's top-up size; 2B ≈ 21% of its recipe, 0.8B ≈ 4%). Whole groups, sampled per partition by
  `build_template_s.select_groups` (strata source × task type × language, `sha256(seed \0 group)` order, each
  stratum's share), seed `dec-m8s-topup-<tier>-v1`. The 1:1 split gives D2 a real human dose at 0.8B, whose recipe is
  only ≈21% human by characters.
- Guards: `eval_only.guard` / `check_rows`, the C1 registry source keys (0 hits allowed), SELECT700 / CAL698 /
  `hs1-dev` / HT-DEV v2 isolation by id and input hash.

## Teacher targets (GPU; `ops/m8s/m8s_label.py`)

A20r's decision distributions at **T = 1 on every top-up row of both tiers** (one union label pass, deduplicated by
id + input hash), on the scored runtime: image `dbe5f32b…` with the FLA / causal-conv1d overlay and
`HIP_FORCE_DEV_KERNARG=1`, a fresh `cp -a` copy of A20r's frozen autotune cache (`27b/m3-f2/f1-scored-cache`, tree
`03b172f1…`), `DecisionModel.from_checkpoint(checkpoint, source_path=<pinned base>)`, FP32 parameters, BF16 autocast
backbone, FP32 head, batch size 1, max length 32,768 (no truncation) — the code path of A20r's kernel-path CAL fit.

- **Parity gate first:** the tool re-collects CAL698 (`19cc1a8c…`) and must reproduce A20r's stored CAL logits
  (`M4-A20r-soup/cal698/cal.logits.jsonl`): 0 argmax changes and max |Δ logit| ≤ 1e-4. FAIL stops labeling and both
  D arms (C still runs and is reported).
- The checkpoint identity must equal the package's `model_sha256` and the calibration must be T = 1.
- **4B overlap:** if the 4B worker's A20r target files exist at label time, rows shared by (id, input hash) are
  compared with mine (report only). This track writes its own complete files.

## Arms (per tier; all start from I; `v2.dec.train_dec --init decision2`)

Every arm trains one epoch on the tier's identical top-up file (same rows, tokens and order per seed), so the matched
control is exact; only the targets differ.

| Arm | Objective |
| --- | --- |
| **D1** | CE + 0.5·Brier + λ·KL(A20r ‖ student) on every row |
| **D2** | CE + 0.5·Brier + λ·KL(A20r) on human rows; typed rows gold only (`--teacher-partial`) |
| **C** (matched-token control) | the released recipe's objective: 2B CE + 0.5·Brier + 0.5·KL(own Sol 1.0; S2T's targets `947bc65b…`) on every row; 0.8B CE + 0.5·Brier (E8F had no teacher) |

λ = 1.0 (the program's cross-tier teacher weight at 4B / 9B), subject to the λ rule above. Common: full fine-tuning,
backbone / head LR 5e-6 / 5e-5 (the decoder recipe; half E8F's 1e-5 / 1e-4, as 9B M7 halved for continuations), fresh
AdamW, wd 0.01, warmup 5% then cosine to 10%, token micro-batches ≤ 32,768 tokens / ≤ 64 rows, ≥ 64 rows per
update, max length 8,192, **one checkpoint at the final update** (`--checkpoint-schedule every --save-every
1000000`; SELECT700 is recorded at the end, never used to choose). Seeds 20260930 / 31 / 32, the same triple in every
arm. Seed 1 of every arm runs the preflights (zero-step, one update, `preflight_dec` load parity against I + one-step
reload); a failure stops that arm. Arm artifact = the uniform FP32 soup of its finished seeds (≥ 2 needed; a
two-seed soup is disclosed).

## Early stop (after seed 1; `m8s_rules.py early`)

D-s1 and C-s1 (final checkpoints) are collected on HT-DEV v2 at 16K on the readout path. A D arm trains seeds 2–3
only if (i) HT-DEV v2 D-s1 vs C-s1 is not FLAG (Δ > −0.02) and (ii) its final SELECT700 family-macro accuracy is at
least C-s1's − 0.03. Otherwise the arm stops (no line, no finalist). C always continues.

## Candidates and development readouts

Lines toward the released weights: θ(α) = α·A + (1 − α)·I for α ∈ {1, ½, ⅓} (members `[A]`, `[I, A]`, `[I, I, A]`,
`v2.dec.soup`, CPU), for A ∈ {D1, D2, C} soups. Every point and I itself are read on node B (image `dbe5f32b`, the
decoder's persisted Triton cache, `v2.dec.infer_dec`, **16,384 tokens**, T = 1): typed DEV 1,600, CSS pilot 1,430 and
HT-DEV v2 1,944, scored on node B (gold `659c92b4…`) with the eval scorer. `I` is also compared with the eval track's
reference collection of the same weights (`dec-m3-S2T-soup`, `dec-m2-E8F-soup`; path check, expected Δ = 0); if it
differs, the M8s collection of I is the reference and the difference is disclosed.

## Development gates and finalists (fixed now; `m8s_rules.py finalists`)

A point passes when, against I read the same way:

1. **Typed / Score / Noul floors:** every type keeps c_t ≥ c_t,I − 0.03·n_t on typed DEV (Choice, Noul, Score), and
   no typed-DEV family falls below F_f,I − 0.10;
2. **HT-DEV v2 not FLAG:** ΔH_dev2 > −0.02 against the tier reference (COORDINATION 04:10).

No typed-gain requirement. Per line, the pick is the passing point with an HT-DEV v2 **GAIN** (Δ ≥ +0.02) if any,
else a TIE; within a class, the largest α. A pick with proxy P ≥ 8 below the tier's best pick is dropped. Slots in
order D1, D2, C; a line without a pick leaves its slot empty; **at most three finalists per tier**. The CSS-pilot
H3 is reported only.

## Diagnostics (never selected on)

`hs1-dev` (quote adoption, ideal .50; false yes on unmet conditions, ideal 0) for I, every arm soup and every
finalist; the matched-token effects D1 − C, D2 − C and D1 − D2 at equal α on every readout; A20r's gold agreement
on the top-up rows by type and partition; `mlx-diag` (formal, item 4).

## Formal (development passers only)

Post-key same-panel v3 (typed FINAL + CSS15) and public 231 at 16,384 tokens, T = 1 (the incumbents ship T = 1; a
CAL698 16K fit and the 23:15 rule decide the shipped calibration for any hand-off), an 8-item smoke first, then
seal, report and paired CIs; `mlx-diag` paired on the same path.

- **2B:** node B, image `dbe5f32b`, a `cp -a` copy of `formal/m6/cache-frozen-2b` (the node-B S2T reference with 0
  differing answers against the node-A scored run, M6).
- **0.8B:** a node-B reference collection of I first (fresh cache, then frozen as `cache-frozen-08b`). If its
  answers equal the node-A T = 1 binding on typed FINAL, CSS15 and public 231, finalists run on that frozen cache and
  the node-A bar stands; otherwise the node-B collection of I is the paired bar (same weights, image and cache) and
  the substitution is disclosed.
- Scoring on node B (formal gold present, the same scorer); the bars and peer runs are relayed from node A
  gold-free where needed.

## Successor rule, choice and item 8

Items 1–8 as in M7: (1) v3 `ci95.low` > 0 vs the bar; (2) `axis_ci95.H.delta.high` ≥ 0; (3) `gates types` OK;
(4) card-eligible `mlx-diag` (Choice + Noul) `mlx_paired` `ci95.high` ≥ 0; (5) tier gates: 2B vs Sol 1.0 16K
`ci95.low` > 0, v3 ≥ 44.5, H not significantly below Decider 2B; 0.8B vs the adopted Eos 1.0 `ci95.low` > 0; (6) no
overlap exposure (empty receipt; points containing I inherit its disclosed exposure: 2B 24 groups / 57 rows, 0.8B 33
groups); (7) `gates public231` vs the bar not REGRESSION; (8) C1 post-key guard through the eval custodian. Paired
CIs are also reported vs Decider 2B and This-That 1.2 (2B) and Intern-0.8B (0.8B).

Among finalists passing items 1–7: the highest v3 `ci95.low` vs the bar; within 0.25 of it, an HT-DEV v2 GAIN, then
the higher formal ΔH. That one candidate per tier goes to the eval custodian with a frozen package on node A (direct
node link, temporary key removed afterwards) and a `dev2-c1-postkey-spec/1` spec. A candidate passing items 1–8 is
handed to the coordinator; its card must name **DEV2.0-27B as the distillation teacher**.

## Budget, GPUs, stop rules

- **Cap 24 GPU-h for both tiers** (wall-clock × GPUs, including preflights, labels, readouts, co-tenant jobs, CAL
  fits, smokes, references and formal collections).

  | Attempt | Cap (GPU-h) |
  | --- | ---: |
  | A20r labels incl. the parity gate (both tiers) | 3.0 |
  | 2B arms D1 / D2 / C (3 seeds each) | 1.2 each |
  | 0.8B arms D1 / D2 / C (3 seeds each) | 0.9 each |
  | Early-stop collections | 0.6 |
  | Lines, references, diagnostics (both tiers) | 4.0 |
  | Formal (≤ 6 finalists, the 0.8B node-B reference, `mlx-diag`, CAL fits, smokes) | 4.0 |
  | Contingency (never for reruns) | 3.1 |

- **At 21 GPU-h cumulative** no new training seed starts; formal work finishes within the cap.
- Stop rules: a failed preflight stops that arm (no rerun, no replacement seed); an arm stops at its cap; fewer than
  two finished seeds drops the arm; a failed smoke, calibration or collection stops that finalist; failed arms are
  never rerun.
- **GPUs:** node B GPU5–7 (27B-owned, lent since M5's lane B finished) and GPU2 (spare), under shared lease entries
  `owner.dec-m8s` after checking that no 27B job runs there. Nothing runs on any other GPU; the 4B worker's runs,
  worktree and GPUs are not touched.
- **Chain rule:** every script runs from an exact mirror uploaded and verified by `mirror_to_node.sh`, launched in a
  separate step; liveness by PID and container name, never `pgrep -f`; the first log line is confirmed; before a
  turn ends with an automatic chain, the chain is verified live and the check is recorded in `m8s-state.md`.

## Code (committed before use)

`v2/dec/ops/m8s/`: `m8s_compose.py`, `m8s_label.py`, `m8s_lock.py`, `m8s_rules.py`, `m8s-prep.sh`, `m8s-label.sh`,
`m8s-run.sh`, `m8s-chains.sh`, `m8s-lines.sh`; tests `v2/dec/tests/test_m8s.py`. Formal wrappers are committed before
their first use. No shared-module change is planned; one would land as a separate small commit with tests.
