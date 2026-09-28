# 0.6B Milestone 2 amendment (M2c): query at the attention-sink position

Frozen 2026-09-28 before any QBQ run.

- QB finished collapsed (BEST SELECT 262 and 281/700); QBS (span readout) is
  collapsing the same way, so the candidate readout is not the cause.
- A gold-free-input gradient probe (`probe_gradients.py`) shows padded-batch
  backbone gradients match unpadded one-row passes (cosine .991 Qwen, .999
  EuroBERT, .998 mmBERT), for the model's own mask path and for contiguous bool
  and float masks. Padding and mask handling are not the cause.
- Remaining structural difference from the working causal Qwen control: this
  head's query is the first-token state, an attention sink in RMSNorm/SwiGLU
  backbones (Qwen, EuroBERT) but not in ModernBERT (mmBERT trains). The control
  queried the final token.

Arms (marker readout, everything else identical to QB):

| Arm | Change vs QB | Gate |
| --- | --- | --- |
| QBQ s1 | query = mean of every real token except the first | collapse stop: SELECT < 350 at update 175 ends the run |
| QBQ s2 | same, seed 2 | only if QBQ s1 BEST SELECT ≥ 450 |
| QBQL s1, s2 | QBQ + 1.0·KL(Lux1 ∥ student), teacher `2d90bc5b…` | only if QBQ s1 BEST SELECT ≥ 450 |

QBS finishes and is read out once; QBOS/QBLS stay gated on QBS as in M2b. LXL
is unchanged. Finalist, formal and stop rules are unchanged.
