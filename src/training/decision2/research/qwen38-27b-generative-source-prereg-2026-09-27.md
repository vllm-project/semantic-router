# Qwen3.8 27B generative versus typed-head architecture screen

Status: **preregistered, no predictions yet**. This is a bounded development
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
