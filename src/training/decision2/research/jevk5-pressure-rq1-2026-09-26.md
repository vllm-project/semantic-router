# JevK5 native extended-choice capacity (RQ1)

This is a **public development diagnostic**, not a release result. The same
frozen, gold-free RQ1 prompts were sent to JevK5 4B, JevK5 9B, and the
Decision 1.0 Lux 9B reference. Each source text appears with nested candidate
sets of size 2, 4, 8, 16, 32, 64, and 128: 800 source UIDs and 5,600 requests
per model. Neither CSS15 nor the family-disjoint final gold was read.

## Receipt and scoring contract

- Panel: [pressure protocol](pressure-panel-protocol-2026-09-26.md), manifest
  SHA-256 `b7ccbafd9c579431cde85103f809f48bab816d91d5e5efd703412d47c2885e9c`;
  gold-free prompts `f1584742583dacab334e0f5ca7702e7248991d7f58f41176e87e14b14446aa19`;
  scoring gold `53787d25aa72427a7af64600cee9d4622b0340ef6d170473d1dadda4bf4711ab`.
  All three reports attest the same exact files. The upstream public data are
  [CC BY-SA 4.0](https://github.com/gazelle93/decision-models-under-pressure/tree/21820dc8b455e37aebaf5a83122233b6e564c109/dataset/v3).
- Native JevK5: [`allebee/jevk5`](https://github.com/allebee/jevk5) runtime
  commit `1e5ae1b533b9eb80c0cbe3fbd010607d0b4e26ae`, clean source,
  `JevK5(local_path, graphs=False).decide(...)`, BF16 eager execution, native
  released temperatures and letter-logit readout. Collector version
  `jevk5-native-eager-v2-admission` records size/context refusals as invalid
  answers. It does not cut a candidate list to fit a model. K>16 takes the
  native runtime's multiple option passes. JevK5 4B revision
  `c4f7fdb3aeab5582336406e78d3bef11bf98833d`, weight SHA-256
  `13824e47f2e40fe052f06943976cf742cb366ba305741a111e75a8ebae907a9c`;
  JevK5 9B revision `d6521a18a86999190e9d775c915af3d6d6772fc4`,
  weight SHA-256
  `7060dee98993bb70865817007870d557e3dd42700d36b0e7c367c847bcfd2e8a`.
- Reference: Decision 1.0 Lux 9B revision
  `bd45a30aee8c84032791c245c70f86dee5389cc8`, collector
  `native-published-v2`. Its RQ1 report uses the same scorer and panel.
- Scorer: `transfer.pressure_score`, signed source commit `927fca7`, version
  `decision2-pressure-score/2`, source SHA-256
  `4195553c0a4c619e35e94e77531346f4fd0d68ef6dc89c44f664489eeff065d1`.
  It checks every present prediction's exact `model_id`, `model_revision`,
  `adapter_version`, and input hash before scoring. Invalid or missing answers
  remain misses. Accuracy and validity 95% percentile intervals use 1,000
  bootstrap draws of complete seven-K source-UID vectors, seed `20260926`;
  repeated K variants are never treated as independent items.

## RQ1 curve

Entries are all-item accuracy percent with UID-clustered 95% intervals. All
three models returned **5,600/5,600 valid answers**, with complete option
probability maps at every K. The denominator at each K is the same 800 source
UIDs. K>16 is an extended capacity diagnostic and does not enter the common
K2–16 comparison with models whose native interface stops at 16 candidates.

| K | Lux 1.0 9B | JevK5 4B | JevK5 9B |
|---:|---:|---:|---:|
| 2 | 85.13 [82.75, 87.63] | 85.13 [82.88, 87.38] | 88.00 [85.88, 90.50] |
| 4 | 76.75 [73.75, 79.50] | 76.25 [73.25, 79.13] | 78.13 [75.13, 81.00] |
| 8 | 76.00 [73.12, 78.75] | 72.50 [69.25, 75.50] | 73.25 [70.00, 76.25] |
| 16 | 70.75 [67.50, 73.88] | 67.63 [64.25, 70.88] | 67.88 [64.38, 71.13] |
| 32 | 68.38 [65.00, 71.50] | 63.25 [60.00, 66.75] | 66.13 [62.75, 69.25] |
| 64 | 61.75 [58.37, 65.25] | 56.75 [53.38, 60.13] | 60.00 [56.50, 63.25] |
| 128 | 54.88 [51.50, 58.38] | 50.63 [47.25, 53.88] | 56.00 [52.50, 59.38] |

The paired within-model K128-minus-K2 drop was -30.25 points for Lux
[-33.62, -26.62], -34.50 for JevK5 4B [-37.88, -31.00], and -32.00 for
JevK5 9B [-35.25, -28.63]. JevK5 9B has a 5.38-point *point* advantage
over its 4B sibling at K128 and 1.13 points over Lux, while Lux leads it by
2.88 points at K16. To compare models on the same text, we evaluated each
receipt with the frozen gold, formed a 0/1 correctness difference for each
source UID and K, and resampled the same 800 UIDs 1,000 times with seed
`20260926`. The following 95% percentile intervals are paired by source UID.

| K | JevK5 9B − Lux (points) | JevK5 9B − JevK5 4B (points) |
|---:|---:|---:|
| 2 | +2.88 [+0.88, +4.75] | +2.88 [+1.00, +4.75] |
| 4 | +1.38 [-0.75, +3.25] | +1.88 [-0.50, +4.00] |
| 8 | -2.75 [-5.13, -0.63] | +0.75 [-1.50, +2.88] |
| 16 | -2.88 [-5.25, -0.62] | +0.25 [-2.00, +2.50] |
| 32 | -2.25 [-4.38, +0.13] | +2.88 [+0.38, +5.13] |
| 64 | -1.75 [-4.13, +0.63] | +3.25 [+0.50, +6.13] |
| 128 | +1.13 [-1.63, +4.00] | +5.38 [+2.88, +8.00] |

Thus the small K128 9B-over-Lux point lead is unresolved, whereas 9B's
K128 gain over 4B is clearer on this panel. These exploratory intervals are
not adjusted for inspecting seven K values and do not establish a release
ranking.

| Domain | Lux K16 → K128 | JevK5 4B K16 → K128 | JevK5 9B K16 → K128 |
|---|---:|---:|---:|
| CLINC | 92.0% → 79.0% | 88.0% → 73.0% | 88.0% → 79.0% |
| DBpedia | 87.5% → 68.0% | 83.5% → 63.0% | 85.5% → 71.5% |
| GoEmotions | 35.5% → 13.5% | 29.0% → 8.0% | 26.0% → 12.5% |
| MTOP | 68.0% → 59.0% | 70.0% → 58.5% | 72.0% → 61.0% |

The GoEmotions weakness is already present at K16 and worsens as candidates
grow. Increasing JevK5 size from 4B to 9B improves extended capacity on
CLINC, DBpedia, and MTOP, yet does not repair K16 GoEmotions. This supports
testing domain transfer and candidate scaling separately when selecting
Decision 2.0 checkpoints. Overall accuracy averaged over all seven K values
is 70.52% Lux, 67.45% JevK5 4B, and 69.91% JevK5 9B; that mixed-K mean is
only a panel summary, not a release ranking. Model latencies come from
unmatched runtime deployments and are not compared.

The source manifest records a failed G3 text-blind gold-picker gate in six of
ten domain/tier cells at K16, and public source items may overlap training or
pretraining. Option-label shortcuts therefore affect absolute accuracy. The
curve is useful for diagnosing capacity and domain behavior under a fixed
prompt, not for claiming public-benchmark SOTA.

## Immutable experiment artifact SHA-256

| Model | Native predictions | Scorer v2 report |
|---|---|---|
| Lux 1.0 9B | `d7cbf7f1ca99f649172bc743ac845634877a02fbd0fbf6a8300d34c4de6c7ba2` | `1809fcff290e713fc761c72705d90931b725142613534e3d699642d6eb48f6bd` |
| JevK5 4B | `53721589ed39166de7a0edc5110f7c684c45700cf9bfb64e9d2e19b7683bc88e` | `5010e96793cb124f3176d5ed7a8a12025ca97c47e84347abf1561632bb9e0051` |
| JevK5 9B | `4a8b1d92b5a3cd2e012d632ea8238584e5ac53ad70b4c47c91b2959ce97846a0` | `a0e0046fa8b8f87b89ea607e5bb578860edd285aabab349041251e2567c18ffb` |

Raw receipts and scorer reports remain in the experiment workspace. The
hashes let the unified research gist identify exact results without exposing
private infrastructure locations.
