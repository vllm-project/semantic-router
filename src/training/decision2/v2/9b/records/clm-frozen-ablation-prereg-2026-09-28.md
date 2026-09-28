# 9B CLM architecture experiment: frozen-feature ablation grid (preregistration)

Status: frozen 2026-09-28 before any head was trained, any SELECT readout was
seen, or any typed DEV / CSS pilot feature was computed. Development readouts
here are never release scores; JevArena v3 is post-key and is run only through
the eval track's frozen runners for a candidate that passes the gate below.

## Question

On the same frozen 9B features, same TRAIN rows and same optimizer budget, does
a CLM-style readout (separately encoded state and candidates, state/action
dual projection heads, bidirectional InfoNCE, explicit hard negatives, soft
replay) match or beat an ordinary Decision head on transfer, calibration,
latency, throughput and candidate-cache reuse? Which 9B release route does
the evidence favor? No outcome is presupposed.

## Fixed inputs

- Features: [extraction protocol](frozen-feature-extraction-protocol-2026-09-28.md).
  Primary official start `Qwen/Qwen3.5-9B@c2022362…`; sensitivity sources
  `Qwen/Qwen3.5-9B-Base@68c46c4b…` and own `Decision-1.0-Lux-9B@bd45a30a…`
  (frozen "Lux continuation" features).
- Data: TRAIN 7,324 (`fe9c419a…`; 3,824 Choice, 2,993 Noul, 507 Score of which
  99 three-level), SELECT 700 (`32a4352d…`), CAL 700 (`3e34f6cb…`), identical to
  the completed 9B LoRA shared-head and Score-cardinality arms.
- Runtime: trainer image `sha256:f83b1d10…` (PyTorch 2.12.0+git6bbd260, HIP
  7.2.53211, Transformers 5.17.0), node A GPU2–4 only.

## Arms (one attributable change per step)

| Arm | Representation | Relative readout (Choice, Noul, diagnostic Score) | Training objective |
| --- | --- | --- | --- |
| A0J ordinary Decision head | joint native prompt: option endpoints + query | shared bilinear/MLP `CandidateHead` (4.2M) | CE + 0.5·Brier over own options |
| A0D ordinary head, disaggregated | state text vector + separate candidate vectors | same `CandidateHead` | CE + 0.5·Brier |
| A1 raw embedding | disaggregated | `100·cos` in raw encoder space (CLM `clm-raw`) | none |
| A2 dual projection + InfoNCE | disaggregated | state/action MLPs 4096→1536→1536→512 (LayerNorm, GELU), learned scale | bidirectional in-batch InfoNCE; pool = distinct gold texts of the batch; each state's own non-gold option texts are masked in both directions |
| A3 + hard negatives | disaggregated | as A2 | A2 + each state's own non-gold options as explicit negatives in the state→action denominator |
| A4 + replay | disaggregated | as A2 | A3 + 0.5·KL(Lux1 ‖ student) over own options on TRAIN rows (weight from the prior Eos soft-replay protocol) |

A4 requires the own-Lux teacher gate below; on failure A4 is recorded as stopped
and not replaced. A4 is not run on the Lux source (self-distillation).

## Native readouts

- Choice: softmax over the offered options of the arm's relative logits.
- Noul: `P(true)` from the two-option relative softmax.
- **Score (primary): absolute ordinal head** in every arm. The state vector maps
  to one scalar position `z`; adjacent rubric-level pairs define strictly
  increasing thresholds `t_1 < … < t_{K-1}` (first threshold plus softplus
  gaps, conditioned on level vectors and relative position); `P(level ≥ k) =
  σ(z − t_k)`. Level probabilities are differences of these, so the
  distribution is proper and ordinal and not a softmax of state–level
  similarities. Loss: NLL + 0.5·Brier on TRAIN Score rows (+0.5·KL to the Lux
  level distribution in A4). Its inputs are the arm's representation with
  gradients stopped, so it never shapes the arm's representation. Joint arms
  use the query and level-endpoint vectors; A0D/A1 use raw disaggregated
  vectors; A2–A4 use their projected vectors.
- Score (diagnostic): the arm's candidate-relative softmax over levels.

## Same budget

Batch 128 TRAIN questions sampled without replacement per epoch-permutation
(seeded), 1,200 AdamW updates (≈21 epochs), learning rate 5e-4 one-cycle cosine
with 10% warmup, weight decay 0.01, gradient-norm clip 1.0 per parameter group.
SELECT every 100 updates plus step 0. No early stopping. Parameter counts
differ by architecture and are disclosed.

## Selection, layers, seeds, calibration

- Checkpoint: SELECT family-macro accuracy, then family-macro Brier, then
  earliest update, using primary readouts at temperature 1 (the existing
  `metric_summary`).
- Layer ∈ {16, 24, 32}: chosen per (source, arm) by the same rule on the
  primary seed 20260928's selected checkpoints; ties go to the later layer.
- Seeds 20260928 (primary), 1, 2 at the selected layer.
- CAL: one temperature per readout (Choice, Noul, relative Score, absolute
  Score) minimizing CAL NLL over [0.05, 20]; CAL never selects checkpoints.
- Technical gates per run: finite losses/gradients; step-0 outputs valid;
  reload of the saved head reproduces SELECT logits exactly (zero drift). Each
  arm first runs a one-update preflight.

## Own-Lux teacher gate (A4 only)

Apply Lux's unchanged published head and CAL temperature (2.00544) to
final-norm joint features from the frozen Lux backbone. It must reproduce the
published package example's five answer slots with identical categories and
maximum probability drift ≤ 0.05 (this runtime is the pinned reference Gated
DeltaNet path, not Lux's qualified FLA image, so it is a TRAIN-only teacher,
not a Lux comparator), and give finite normalized distributions for all 7,324
TRAIN rows. Lux's own training sources (programmatic tasks, MultiNLI,
BANKING77, CLINC150, Cosmos QA, SQuAD 2.0, SNLI) overlap TRAIN sources; exact
row overlap is not established. None of the three CSS pilot tasks is a Lux
training source.

## One sealed development readout

1. Freeze every selected run (head, calibration and BEST hashes) in a signed
   heads seal pushed to the branch **before** extracting typed DEV 1,600
   (`a17ec4b6…`) and CSS pilot 1,430 (`598319a4…`) features.
2. Extract readout features per source, predict every sealed run, and write a
   joint prediction seal before opening either key.
3. Score with the unchanged `benchmark/score.py` (`d02a3b2b…`) and
   `transfer/score.py` (`cfe199a1…`). Proxy `100·sqrt(T·H)`, `T` = typed
   four-family macro accuracy, `H` = CSS pilot task-median macro-F1.
   Over-length inputs are invalid and count as failures.

No checkpoint, layer, temperature or seed is changed after the readout.

## Decision rules (fixed now)

- **Absolute Score validation**: on typed DEV Score, the absolute head passes
  if its ECE-10 ≤ 0.10 and its Brier ≤ the same model's relative Score Brier;
  CAL ECE-10 after temperature is also reported. Also reported: middle-level
  recall, ranked probability score and expected-score calibration.
- **CLM versus ordinary head** (seed means, primary source): the CLM route
  (best of A2–A4 by SELECT) is *quality-competitive* if its DEV proxy is within
  1.0 point of A0J, *better* if ≥ 1.0 above, *worse* otherwise. The serving
  claim additionally requires a measured candidate-cache latency or token
  saving ≥ 2× at ≥ 10 options.
- **Release-route gate**: any 9B candidate (frozen head, fine-tuned or Lux
  continuation) needs DEV proxy ≥ 72.33 (Lux1 historical same-input DEV
  70.326 plus 2.0, the gate used by every prior 9B arm) before a formal post-key
  same-panel run against the current-package Lux1 comparator (v3 66.268,
  public231 183/231). Failing arms stay development HOLD.
- Comparators, not re-run: official LoRA shared-head DEV proxy 61.255 (Score
  174/400); Score-cardinality 63.655 (Score 237/400); Lux1 historical DEV
  70.326 (T .8675, H .57011).

## Resource caps

Extraction ≤ 1.5 GPU-hours per source; readout extraction ≤ 0.5 per source;
head grid ≤ 4; latency/throughput ≤ 1; total Milestone 1 ≤ 12 GPU-hours on
node A GPU2–4. GPU-hours are container wall time × one GPU, including
preflights and failures.

## Not tested in Milestone 1

Agent trajectories and CLM's large-scale three-stage schedule (see the
[source study](clm-source-study-2026-09-28.md)); backbone fine-tuning; mined
cross-question negatives (risk of false negatives).
