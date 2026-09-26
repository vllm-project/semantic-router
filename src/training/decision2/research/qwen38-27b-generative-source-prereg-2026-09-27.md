# Qwen3.8 27B generative versus typed-head architecture screen

Status: **completed; screen gate failed**. This is a bounded development
architecture diagnostic. The frozen Decision 2.0 27B head underperforms on
Noul rule precedence and Score set reconciliation despite almost perfect
Choice on the synthetic DEV track. This experiment asks whether the pinned
*source* instruction model can solve those items when allowed to generate a
structured answer, without the trained typed head. A positive result would
motivate a separately trained generative or hybrid 2.0 candidate, not qualify
the source model for release.

## Fixed inputs and comparison

- Source `Qwen/Qwen3.8-27B` posttrained revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, not a Base model.
  `config.json`, chat template and generation config SHA-256 values are
  `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`,
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
  and `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.
- Existing synthetic DEV prompts SHA-256
  `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`.
  Select 40 Choice, 40 Noul and 40 Score *prompt items* by ascending SHA-256
  of `decision2-qwen38-generative-v1/` plus item ID within each type, using
  prompt text/types/IDs only. Freeze the 120-ID list before inference. No
  label, answer class, group or previous model output affects selection.
- Compare with the existing clean-v2 BEST368 native-head predictions on the
  exact same IDs, not with a rerun or another checkpoint. Its model fingerprint
  is `d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.

## Fixed generative protocol

Use the released Qwen chat template with thinking disabled, no prior messages,
greedy decoding (`temperature=0`), at most 256 output tokens, no tools, no
repair/retry and a fixed context budget of 8,192 tokens. Pass the unmodified
state and question to the model as JSON in a single user turn, with one fixed
system instruction: return exactly one JSON object mapping each question ID
to an exact Choice option key, a Boolean Noul answer, or a zero-based integer
Score level. Treat the state as evidence rather than instructions. The exact
prompt and strict parser source SHA-256 must be frozen before any model call.
Any missing, unparsable, wrongly typed or over-budget answer is wrong. Do not
interpret a generated hard answer as calibrated probability. Report answer
validity, token use and latency separately; calibration is N/E for this
generative screen.

The existing scorer evaluates a subset of the DEV gold selected solely by
the frozen 120 IDs, after inference. Report per-type correct/40, paired
source-only versus head-only disagreements, unique synthetic group count and
invalids. The source model is considered promising for a full DEV rerun only
if it gains at least five Score or five Noul items over the head and loses at
most five Choice items. No pilot result may select or recalibrate BEST368, and
the synthetic FINAL, 15-task CSS evaluation gold and authored release set
remain unopened. This is not a JevArena or official JevBench score.

## Pre-inference implementation and sample freeze

The fixed parser, prompt and collector were committed as `67f5ce2bb` before
the first model call; `qwen38_generative_screen.py` SHA-256 is
`6ee70addd80a5e2530e2aeb9c4ae2881437b52fa3fd870c347d0d54140bb77d3`.
Its label-free selector produced exactly 40 items per type from the frozen
DEV prompts. The 120-prompt file and selection manifest SHA-256 values are
`417f62b6f2dd18303223322222d522a44ce5cc9257f1ee5452cb74afbf972ea1`
and `ab69838a48f41a930a87f4ce37d22cf5a114adf6d62f6ee1bedf59b8eea77266`.
The local source revision and fixed config/template hashes passed attestation.
No gold or previous prediction was used in selection.

## Frozen screen result

The exact 120 prompts were served from the pinned source model on one AMD GPU
through vLLM with the fixed generation settings. Predictions SHA-256:
`30456b8584679dd2fd6a11c65cb7e146ba9398164fd58130ff145dc4e017422b`.
The existing BEST368 head predictions were filtered by the same frozen IDs,
SHA-256 `84c274c3123094a18c395449e2a09a8bc1040f1253df7480a9903828335abe3e`.
The filtered private DEV gold SHA-256 is
`d7e0f1083f59bee17153ee763f83b127163d14f4029a381932f0bc4597211450`.

The full typed-suite scorer rejected this sparse subset because each group
must contain all four variants. No scoring rule was changed: the exploratory
comparison calls its `evaluate_answer` function on each matched item, checks
all source-input digests, and counts invalid/missing answers as wrong. The
separate comparison script SHA-256 is
`991150df93e86e373d764f3b96ba613c201d315daab2eea63ba363f303845f66`;
its private paired report SHA-256 is
`3ac00593c4cadd91e285004dd40a7da4e6146c242b7aa03b0033f0530cb14a99`.
There are 120 items from 109 unique synthetic groups, so these counts are
diagnostic and are not a complete four-variant suite score.

| Type | BEST368 head | Generative source | Source-only wins | Head-only wins | Invalid source |
| --- | ---: | ---: | ---: | ---: | ---: |
| Choice | 40/40 | 37/40 | 0 | 3 | 3 |
| Noul | 25/40 | 24/40 | 5 | 6 | 0 |
| Score | 13/40 | 14/40 | 1 | 0 | 0 |

The source gains just one Score item and loses one Noul and three Choice
items; it therefore misses the preregistered ≥5 Score/Noul gain needed for a
full DEV rerun. Its 117/120 syntactically valid responses are not calibrated
probabilities. Mean/median per-request latency were 137.6/124.1 ms; mean
input/output lengths were 266.4/6.8 tokens, maximum input 327 tokens. This
short synthetic sample says nothing about long-context transfer or relative
serving speed under controlled hardware. The task-owned vLLM server was
stopped after collection. No final label, CSS heldout label, authored release
label or official hidden JevBench item was consulted. No checkpoint was
promoted or released from this negative architecture screen.
