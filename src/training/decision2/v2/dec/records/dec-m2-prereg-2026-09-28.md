# Decoder track Milestone 2 preregistration (0.8B / 2B / 4B), 2026-09-28

Status: **frozen before any Milestone 2 arm is launched.** Code: `v2/dec` at
`15ef238ef` (trainer options, mixture builder, arm driver) on top of the
correctness checks `aac6959ad`/`d81745176`/`56525ff81`; the commit adding this
record also adds the runtime check to `infer_1p0` (same-runtime 1.0 readouts)
and its formal adapter spec (`adapter-spec-infer-1p0.json`, 8,192 tokens,
package temperatures) for the same-limit 1.0 controls. Development readouts
(SELECT, CAL, typed DEV, CSS pilot) are never release scores; JevArena v3 and
public JevBench 231 numbers are post-key same-panel comparisons.

## Coordinator decisions implemented

1. E1-X2 stays HOLD (no rescoring). 2. Correctness checks first (below).
3. Data: A7 (`39a120ca`, `v2/a7/`) as the retention/replay base with the two
A7h families excluded, A0s (renumbered here until the data track publishes
its version), v1 arms, own-Lux soft targets (the canonical Lux file is not
announced yet; the decoder file `752b7c8f…` is used). 4. Arms in priority
order (a) 4B, (b) 2B, (c) 0.8B after the Score check, (d) only if GPUs allow.
5. Promotion reads typed-DEV Choice/Noul/Score and CSS pilot separately, uses
the eval track's calibrated proxy rule (|ΔP| < 4 is a tie), and formal runs
only for finalists on node A at 8,192 tokens with a same-limit 1.0 control.

## Correctness checks (results before launch)

- **(a) Padding equivalence** (`check_padding`, receipts under
  `/data/dev2/runs/dec/m2/checks/pad-*`): Eos 1.0, official Qwen3.5-0.8B-Base,
  Sol 1.0, Nox 1.0 with the image kernels; 18 TRAIN rows (6 per type,
  75–6,596 tokens), LoRA B perturbed, dropout 0. Padded (single row rounded to
  8, four rows padded together with 29,918 pad tokens, +61 extra pads) vs
  exactly unpadded rows: gradient cosine ≥ 0.99999 (FP32) / ≥ 0.9985 (BF16),
  relative gradient error ≤ 0.3% (FP32) / ≤ 5.5% (BF16), **zero decision
  flips**, BF16 repeats bit-identical, inference batch 2/8 vs 1 drift ≤ .023.
  The BF16 padded-vs-unpadded gaps equal the BF16-vs-FP32 gap of the same
  unpadded rows (gradient error ratio 1.0–1.1; loss ratio ≤ 2.6) and do not
  grow with the amount of padding (62 → 1,098 → 29,918 pad tokens). The
  absolute tolerances written into the probe before the first run were too
  tight for these kernels (FP32 logit bound 1e-3 vs observed 1.4–5.3e-3, FLA
  uses TF32-class dots for FP32 inputs; BF16 loss bound .05 vs .066–.131 at
  2B/4B, which even the unpadded BF16-vs-FP32 gap reaches), so the probe
  prints FAIL; the regression test on a real tiny Qwen3.5 hybrid backbone
  (reference kernels, FP32) matches padded and unpadded loss and gradients to
  1e-5. **Verdict: no padding bug; Milestone 1 arms are not invalidated by (a).**
- **(b) Score-only overfit** (`build_score_overfit`, 288 rows: 32 per level
  count L = 2..10, every gold level 3–16 times; 134 A7g + 154 A6g rows ≤ 1,536
  tokens; Eos 1.0, 16 epochs, LoRA lr 2e-4 / head 1e-3, plus a diagnostic at the
  Milestone 1 rates 5e-5 / 2.5e-5): PASS rule = fitted-row accuracy ≥ 0.95
  overall, ≥ 0.90 per level count, every gold level predicted. **PASS: 288/288
  fitted rows correct at the final step (checkpoint 288), every level of every
  level count 2..10 predicted** (training loss 2.3 → 0.003; the Milestone 1-rate
  diagnostic reaches loss 1e-4 by update 180). The Milestone 1 "level 0 on all
  400 DEV Score items" is therefore not a trainer or readout defect; a DEV
  diagnostic of this checkpoint is recorded with the results. Arm (c) is cleared.
- **(c) Runtime.** Every GPU entry point now fails unless the Qwen3.5
  gated-delta functions are bound to FLA (`/opt/decision-fla`, 0.5.2) and
  causal-conv1d to its kernel, and a shared persisted Triton autotune cache is
  set (`launch.sh` mounts one per node + image); the preflight requires exact
  cross-process reload (≤ 1e-4, all SELECT argmax equal). Node A image
  `f83b1d10…` passes. **Node B's Milestone 1 image `ce895822…` lacks
  causal-conv1d** (Transformers' reference convolution was used; E1 logs
  carry the fallback warning; FLA gated-delta was bound); node B now uses
  `dbe5f32b…` (`decision20-train-fast:latest`, the same torch/Transformers/
  FLA/causal-conv1d/Triton/PEFT versions as node A). Milestone 1 C1 (node A)
  and E1 (node B) ran without a persisted autotune cache (cross-process
  readout drift ≤ .027, ≤ 5/700 SELECT flips, recorded in their preflights);
  C2 used one. The reference convolution computes the same operation and the
  drift is below σ_seed, so these arms are flagged as runtime-noncompliant,
  not invalidated.

## Data (hash-verified; mixtures identical on both nodes)

Built by `v2.dec.build_mixture` from the registries of HF private
`llm-semantic-router/decision-2.0-training-data` revisions `39a120ca…`
(`v2/a7/registry.json`) and `5c0255ed…` (`v2/registry.json`); tokens under the
shared 1.0 tokenizer (Eos/Sol/Nox `tokenizer.json` `06b95093…`).

| Mixture (spec) | Rows (C/N/S) | Tokens | TRAIN SHA-256 |
| --- | --- | ---: | --- |
| control A0s-r (`m2-control-a0sr.json`) | 6,547 (3,337/2,694/516) | 3,977,352 | `156d8dab…90f39` |
| retention (`m2-retention.json`) = A0s-r + A2 (5,453) + A7 `dec10-stage4v2` replay at 2× A0s-r tokens (6,606 rows, 7,964,049 tokens, whole groups) | 18,606 (10,349/7,026/1,231) | 14,189,303 | `37fb8856…d0ca` |
| full (`m2-full-a7-v1.json`) = A0s-r + A1 + A2 + A3 + A4v2h + A5 + A6g + A6h + all A7 | recorded at build | | recorded before (c) launches |

A0s-r = A0s minus the two excluded families (752 A0s rows are Cosmos QA /
SQuAD 2.0 answerability rows from the same natural pool; excluded everywhere
for consistency) with rule 7d applied to 117 rows (keys `result_<n>`
renumbered by position; input hashes recomputed; those rows get no teacher
term because the teacher saw the leaking keys). Dedup by `input_sha256` in
component order (2,545 stage4v2 rows equal A0 rows). Teacher: own Lux 1.0
labels `752b7c8f…` (T = 2.00544), KL weight 0.5 on covered rows only
(`--teacher-partial`, ≈ 6,430 rows).

## Arms (one epoch each; seeds s1 = 20260926, s2 = 20260927)

Common: SELECT700 matrix-v1 selection over 8 evenly spaced checkpoints
(earliest tie-break), CAL700 per-type temperatures fit once on BEST, one typed
DEV1600 + CSS pilot1430 readout (`postrun.sh`), CE + 0.5 Brier, AdamW wd 0.01,
warmup 5%, cosine to 10%, max length 8,192 (no truncation), token-budget
micro-batches (`--batching tokens`).

| Arm | Tier / start | Mixture | Recipe | Cap (GPU-h) |
| --- | --- | --- | --- | ---: |
| X4C-s1, X4C-s2 | 4B Nox 1.0 `@cde2a68d` | control | LoRA r16/α32/dropout .05, lora lr 5e-5, head lr 2.5e-5 (the X2 recipe), Lux KL; 16,384 tokens and ≤ 16 rows per micro-batch, ≥ 16 rows per update | 0.6 each |
| X4R-s1, X4R-s2 | 4B Nox 1.0 | retention | same | 1.5 each |
| S4C-s1, S4C-s2 | 2B Sol 1.0 `@ce0c018a` | control | same | 0.4 each |
| S4R-s1, S4R-s2 | 2B Sol 1.0 | retention | same | 1.0 each |
| E8F-s1 | 0.8B Eos 1.0 `@363c4a5e` | full | full fine-tuning, backbone lr 1e-5, head lr 1e-4, no teacher; 32,768 tokens and ≤ 64 rows per micro-batch, ≥ 64 rows per update | 3.0 |
| B8F-s1 | 0.8B `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2` (fresh head) | full | same | 3.0 |
| (d) B4F | 4B `Qwen/Qwen3.5-4B-Base@1001bb4d` | full | same as E8F | deferred unless GPUs allow (≈ 5 GPU-h) |

Second seeds for E8F/B8F only for an arm that clears the 0.8B development
rule. Every arm: zero-step + one-step preflight and gate (`drive_arm.sh`); a
failed preflight, nonfinite loss, OOM or cap breach stops the arm (recorded,
not rerun). Same-runtime 1.0 development controls: each tier's zero-step
checkpoint (the 1.0 weights through the same adapter) read on the same node.

## Development screen and formal rule (fixed now)

Proxy P = 100·√(T·H), T = typed-DEV family macro, H = CSS-pilot task-median
macro-F1; Choice / Noul / Score counts read separately.

- **Screen (per arm):** P ≥ P(same-runtime 1.0); no typed-DEV type below the
  1.0 count by more than 3.0 points; H ≥ H(1.0) − .015; all answers valid.
- **4B / 2B formal:** both seeds of the retention recipe (R) if both pass the
  screen, plus the tier's same-limit 1.0 control (zero-step checkpoint at
  8,192 tokens) — all on node A with the eval track's frozen runner, persisted
  autotune cache, compared with the adopted 1.0 run (Nox1 56.470, Sol1 45.580).
  The control recipe (C) is read on development data only; its formal
  reference is E1-X2 (v3 55.993). If R fails the screen but C passes both
  seeds, C's two seeds take the formal slots instead.
- **0.8B:** an arm needs P ≥ P(Eos 1.0 same runtime) + 3.0 with paired lower
  bound > 0 and DEV Score answers spread over ≥ 2 levels; then a second seed;
  both seeds then take formal runs with the same-limit Eos control.
- **Release reading (coordinator threshold):** seed s1 is the candidate; it
  qualifies if its v3 paired 95% lower bound vs the adopted 1.0 run > 0 on
  node A, seed s2's v3 point estimate is also above the 1.0, and no decision
  type collapses; per-type regressions are disclosed. A qualifying candidate
  goes to a private HF staging model repo and is reported immediately with its
  packaging profile (`qwen-adapter` for LoRA, `qwen-full` for full) and scored
  run directory. Anything else is HOLD with the readouts recorded.
