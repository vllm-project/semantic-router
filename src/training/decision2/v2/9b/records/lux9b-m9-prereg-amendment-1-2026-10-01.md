# 9B Milestone 9, amendment 1: the retention ceiling B0 is read through the base's untied LM head

Written 2026-10-01 ≈15:25 UTC+8, after B0's zero-step stopped and before any B0 readout. Everything else in the
[preregistration](lux9b-m9-prereg-2026-10-01.md) (`8570a5896`) is unchanged; no arm, gate or rule depends on B0.

## What happened

B0 (preregistered as the zero-step checkpoint of a label-token run on Qwen3.5-9B-Base, read on the retention probes as
the knowledge ceiling) stopped at model build on node C GPU5 (07:04Z): `v2.dec.label_token` raises "The label-token
readout needs a tied LM head". Qwen3.5-9B-Base has `tie_word_embeddings: false` (its `lm_head.weight` is a separate
tensor); the 4B base that M10 read this way ties them. Recorded, not rerun.

## Change

- B0 is read by a 9B-local diagnostic reader, `lux9b/m9_base_probes.py`: the same label-token prompt
  (`decision2-label-token-v1`, `encode_label`) with option logit i = h_last · W[label_i] in FP32, where W is the base's
  own `lm_head.weight` (the LM head the base was pre-trained with). Nothing is trained or saved; the output has the
  shared prediction format, so `m10_probes.py score` reads it unchanged.
- Panel: the M9 retention probes only (`m9-probes`, 2,839 items: MMLU 1,265, ARC-Challenge 254, ARC-Easy 570, GSM8K
  750 after the x60 exclusion dropped 250 GSM8K items). Node C GPU5 (idle, `track=9b-m9`), image `f83b1d10…`, 16,384
  tokens, T = 1; predictions are pulled gold-free to node A and scored there against C0.
- B0 stays report-only (the ceiling in the H1 reading); it is not a gate input. The shared `label_token` module is
  not changed.
