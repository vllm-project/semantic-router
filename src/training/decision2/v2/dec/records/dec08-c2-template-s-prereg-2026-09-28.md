# DEV2.0-0.8B contrast set C2: matrix wave-1 data factors (template S)

Status: **pre-registered before any C2 build or launch** (2026-09-28). Testbed
T1b (0.8B decoder) of the research & data track's experiment matrix v1.1
(`experiment-matrix-v1-2026-09-28.md`, integrated `9041156e3`). Development
evidence only; JevArena v3 / public231 only through the formal rule below.

## Factors and arms

Arms come from the private HF dataset `llm-semantic-router/decision-2.0-training-data`
at revision `f8c50b1339b99cac3c602d508f1483b09eba7d63`; all 35 downloaded files match
`v2/registry.json`. Budget template S: control `A0 ∪ A0-resample(ρ)`, treatment
`A0 ∪ X(ρ)`, whole groups stratified by source × type × language in a fixed hash
order (`v2/dec/build_template_s.py`, seed `dec-template-s-v1`), native tokens
under the Eos 1.0 tokenizer (A0 = 4,194,465).

| Arm | Matrix factor | Mixture | ρ (tokens) | Primary dev metric |
| --- | --- | --- | ---: | --- |
| C2-C1 | control for ρ1 | A0 ∪ A0-resample | 2,097,232 | — |
| C2-D6g | D6 Score all levels (generated) | A0 ∪ A6g (`ac94b7f4…`, 4,905 rows) | 2,097,232 | typed-DEV Score, RPS, AHO-A6g per level |
| C2-D2 | D2 verifiable counterfactual families | A0 ∪ A2 (`2cd61397…`, 5,453 rows) | 2,097,232 | typed-DEV T (AHO-A2 apart) |
| C2-C2 | control for ρ2 | A0 ∪ A0-resample | 1,436,130 | — |
| C2-D6h | D6 Score all levels (human ordinal) | A0 ∪ A6h (`23440f0c…`, 5,622 rows) | 1,436,130 | typed-DEV Score, AHO-A6h |
| C2-D1 | D1 cross-domain human labels | A0 ∪ A1 (`b50b43af…`, 4,434 rows) | 1,436,130 | CSS-pilot H, AHO-A1 |

ρ1 = floor(0.5 × A0). ρ2 is A6h's full size; A1 (1,488,842 tokens) is
subsampled to ρ2 so that D1 and D6h share one control — a disclosed 3.5%
deviation from the per-arm ρ = min(0.5 × A0, arm). D3 (A3, long evidence),
D9 (A0p) and D5 (A5) are next in line on the same controls.

## Frozen configuration

Everything else is C1-A0 as amended (`1a5241981`, `d8b3f9137`): own
`Decision-1.0-Eos-0.8B@363c4a5e`, LoRA r16/α32/dropout .05, LoRA LR 5e-5, head LR
2.5e-5, CE + 0.5 Brier, one epoch over the mixture (update count follows the
row count: token-matched, not update-matched; reported per arm), microbatch 1 ×
accumulation 16, seed 20260926, max length 8,192, 8 evenly spaced SELECT700
checkpoints with earliest tie-break, CAL700 per-type temperatures, one
calibrated typed DEV1600 + CSS pilot1430 readout, and the arm's AHO slice
(raw probabilities) for each treatment **and** its control. Runtime: node B
GPU0–2, image `ce895822…`, one persistent Triton autotune cache shared by every
C2 run (`DEC_TRITON_CACHE`), code at the commit adding this record.

## Preflight, stop and decision rules

- Builder receipts must show the stated ρ within +1 group, whole groups only and
  a clean `load_partition` + TRAIN/SELECT/CAL isolation check.
- Each arm: zero-step, one-step, `dec-arm-preflight/2` gate; a failure stops it.
- Stop on nonfinite loss/gradient or a projected exclusive time above 3.0 GPU-hours.
- Effect (matrix v1.1): paired bootstrap (10,000 draws) on the arm's primary
  metric versus its control; counts only if the 95% lower bound > 0 and
  |Δ| > 2σ_seed, with σ_seed from the C1 N0 pair (node A; borrowed, disclosed).
  Retention floors: typed-DEV types −3.0 points, H −1.5, CAL Brier +0.010,
  invalid answers not increased. Report the dev proxy for every arm.
- Two seeds before any finalist claim (coordinator note 12:30): an arm that
  passes gets a seed-20260927 replicate before it is combined or scaled.
- Formal rule (unchanged from C1): only an arm with dev proxy ≥ 33.8286 and a
  positive paired lower bound versus same-runtime Eos 1.0 can be frozen for one
  post-key v3 + public231 run, collected on node A next to the Eos 1.0 comparator.
