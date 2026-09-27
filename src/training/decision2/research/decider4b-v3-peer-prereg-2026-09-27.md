# Decider 4B: same-panel external peer preregistration

**Status:** prospective post-key comparator run, before this peer's JevArena v3
and public-231 predictions. Project-level answer keys have been accessed in
earlier arms; this is a fixed same-panel rerun, not a never-opened blind test.

The [Decision Index 0.2.1 peer roster](decision-index-021-peer-roster-2026-09-27.md)
selected Decider 4B as a near-size, strong open comparator before the
official-Qwen 4B candidate's formal score was known. The Index number is only
for model selection and must never be combined with JevArena or JevBench
scores. Run all three panels, seal predictions before scoring, then show the
peer regardless of whether it ranks above or below our model.

## Fixed inputs and native route

- Model: `Mapika/decider-4b` at revision
  `eb5fbdfc9448473ec25e399882912863afbdb70e`, previously downloaded with
  HF CLI on the experiment node. Its local HF metadata attests that revision.
  The model safetensors SHA-256 is
  `ee8ce585b3cedd93206dd149b09b4bdd683174874211f85c77a36090b90c9fdd`,
  the native `decider/infer.py` SHA-256 is
  `6359e5989fe922054c99446115a25d22acf1a943f0dea097409eaa49e2ef84f1`,
  and `decider_config.json` SHA-256 is
  `fc83293b6f707172e20176d603ea88dd1a3be2abd3596fdb1ccbec9404626baf`.
  The weight file stores 4,205,751,296 tensor elements; measure the native
  loaded parameter count before a public size table.
- Adapter: local `inference.run` at SHA-256
  `b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce`,
  `native-published-v2`, using the model's released `system_one` API in eager
  mode, released per-type temperatures, native four-decimal probabilities,
  isolated Score levels, and no change to questions or options. The earlier
  `native-published-v1` development predictions are not reused.
- Common gold-free panel files: typed FINAL 1,600 items / 2,000 answer slots,
  SHA-256 `e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd`;
  CSS15 6,547 items, SHA-256
  `7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6`;
  public JevBench 231 items, SHA-256
  `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`.
  These are the exact same question files as the official-Qwen 4B roster;
  adapters may render them in their own native format.
- Scoring and invalid policy: exactly the [official-Qwen 4B lock](qwen35-4b-official-best466-v3-postkey-lock-2026-09-27.md):
  `100 × sqrt(T × H)` with typed four-family macro accuracy and CSS15 task
  median macro-F1. Missing, invalid, or over-budget answers fail on the full
  denominator. The public-231 score is separate. Record every panel and
  difficulty slice, without selecting favorable subsets.

## Execution and stopping rule

Use an isolated exact source mirror and a fresh output directory on the second
authorized GPU node. Verify the local model revision, three prompt hashes,
adapter SHA, container runtime and free GPU before launch. Run native inference
on the complete three gold-free panels using one available GPU. No label file
may be mounted in the inference container. A runtime failure is a failed run,
not permission to omit items or change adapter/prompt. Any fix needs a new
versioned record, and old partial files remain intact. Allow at most two GPU
hours for this peer. Hash and row-audit all completed predictions, then record
the seal's time and hashes **before** loading any panel labels. The peer can
enter the rank chart only with complete same-version predictions and a native
parameter count; otherwise mark it N/E and report the failure.

No training, distillation, release package or index evaluation is authorized
by this comparator preregistration.
