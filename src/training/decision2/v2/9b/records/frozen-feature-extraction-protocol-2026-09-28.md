# 9B frozen-backbone feature extraction: technical protocol

Status: frozen before any GPU use (2026-09-28). This step trains nothing, reads
no evaluation key and selects nothing. It produces cached, hashed features for
the separately preregistered head ablations. Typed DEV and CSS pilot features
are **not** extracted here; they are computed only at readout time, after every
head, selector and calibration is sealed.

## Sources (verified before load; fail closed on any digest mismatch)

| Source key | Repository @ revision | License | Loaded text parameters |
| --- | --- | --- | ---: |
| `qwen35-9b-posttrained` (primary official start) | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Apache-2.0 | 7,936,684,544 |
| `qwen35-9b-base` (backbone-stage sensitivity) | `Qwen/Qwen3.5-9B-Base@68c46c4b3498877f3ef123c856ecfde50c39f404` | Apache-2.0 | 7,936,684,544 |
| `lux1-9b` (own 1.0 lineage; frozen "Lux continuation" features and TRAIN-only teacher) | `llm-semantic-router/Decision-1.0-Lux-9B@bd45a30aee8c84032791c245c70f86dee5389cc8`, bundle `985ade73…` | Apache-2.0 | 7,936,684,544 |

Both official revisions equal the Hub head on 2026-09-28. The checkpoints hold
9,653,104,368 tensors in total: the 7,936,684,544-parameter text backbone that is
loaded, plus a separate untied 1,017,118,720-parameter LM head, a 456,010,480
vision tower and a 243,290,624 multi-token-prediction module, none of which is
loaded. Shard, config and tokenizer SHA-256 values are pinned in
`clm9b/pins.py`. No third-party decision model is loaded; CLM's Qwen3-8B
projection heads are never downloaded.

## Inputs

Rights-clean v2 partitions, identical to the completed 9B shared-head and
Score-cardinality arms: group-filtered TRAIN 7,324 rows (`fe9c419a…`), SELECT
700 (`32a4352d…`), CAL 700 (`3e34f6cb…`). Partition validation and cross-role
ID/group/input isolation use the unchanged `training.model.data` contract.
For `lux1-9b` only, the two published package example requests (five answer
slots, derived from the request fields with Lux's own Noul defaults) are
extracted as an unlabelled parity input.

## Representations

- **Joint (native)**: the unchanged Decision 2.0 segmented prompt (identical
  bytes to Lux's `structured-segmented-candidate-endpoints-global-query-v2`),
  one causal pass; vectors at every option endpoint and at the final query
  token. This is the ordinary Decision head's input.
- **Disaggregated (CLM-style)**: (superseded by the r1 amendment below) the state text (context, blank line, question
  instructions, structured values as prose) and each candidate text encoded
  alone, without special tokens. Candidates: Choice description (key if
  empty); Noul `key: description`; Score level description. Pooling: last
  token (CLM-faithful) and masked mean (stored for diagnostics only). Each
  distinct text is encoded once, so candidate vectors are reusable across
  requests.
- **Layers**: residual outputs of decoder layers 16 and 24 (both follow a full
  attention layer in the 3:1 Gated DeltaNet / attention stack) and the
  final-norm output (32).

FP32 parameters with BF16 autocast (the existing native inference numerics),
right-padded length-sorted batches under a 16,384-token budget, FP32 stored
vectors, 4,096-token cap per encoder pass with no truncation (over-length
inputs are marked invalid).

## Technical gates

1. Source and partition digests; loaded parameter count.
2. A 16-row preflight per source before the full pass.
3. Batched-versus-single parity on a fixed 16-row sample per input: minimum
   cosine ≥ 0.999 at every stored layer for joint vectors and state texts
   (absolute differences are reported, not gated, because mid-layer residual
   norms are large under BF16).
4. All gathered vectors finite.

A failed gate stops that source's extraction; the failure is recorded and not
retried with changed thresholds. GPU-hours are recorded from container
start/end, including preflights.

## Preflight r1 (official posttrained, 16 rows per partition): FAILED, method amended

Code `59b277e70`, GPU2, 2026-09-28 03:10:53–03:11:23 UTC (30 s, 0.0083
GPU-hour), exit 0, 7,936,684,544 parameters loaded, all digests matched.

- **Joint parity failed the 0.999 gate.** Batched (right-padded,
  length-sorted) joint prompts versus the same prompts run alone had minimum
  cosine 0.9979 (TRAIN), 0.9957 (SELECT) and 0.9869 (CAL) across stored
  layers. Disaggregated texts passed (minimum cosine ≥ 0.99991). The joint
  features therefore depended on batch shape, which would make training and
  serving features differ.
- **Prose rendering inflated structured states.** One of 16 TRAIN rows (a dense
  table) was 3,888 tokens as the native JSON prompt but 4,421 tokens as a
  CLM-style prose state text, exceeding the 4,096 cap in the disaggregated
  arms only.

Prospective amendment (before any full extraction, head training or readout;
threshold unchanged):

1. Joint prompts run one per forward, the native per-question shape; the
   gated joint check becomes repeat determinism of that shape, and the
   batched-shape comparison is kept as a reported diagnostic.
2. Disaggregated texts reuse the native prompt's own segments: the state text
   is the native prefix (context, task type, question) and each candidate text
   is that option's native `{"description","key"}` serialization. Joint and
   disaggregated arms then see identical content and serialization; only the
   encoding differs. CLM's prose rendering and description-only candidates are
   serving-schema details, not hypotheses under test.

Preflight r2 repeats the 16-row check with the amended code.
