# CLM source study for the Decision 2.0 9B architecture experiment

Reviewed 2026-09-28. Primary sources: the
[CLM-v0.1-8B model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
(Hub revision `e939398d4556fcd9400c76fa8c5a513202f42b0a`, Apache-2.0) and the
[Contrastive-LM/CLM repository](https://github.com/Contrastive-LM/CLM) at
commit `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` (Apache-2.0). Author claims
below are claims, not our measurements.

## What the public release contains

| Component | Public evidence | Our reading |
| --- | --- | --- |
| Backbone | Frozen Qwen3-8B, vLLM pooling runner, last-token pooling, L2-normalized 4,096-d embeddings (`config.json`: `encoder_pooling: last-token`). | Heads are encoder-locked; they cannot be reused on Qwen3.5-9B even if we wanted to. We do not download or load them. |
| State / action heads | Two separate MLPs (`heads.py`): 4,096 → width → … → 512, GELU, optional LayerNorm; fine-tune default width 1,536, depth 3, LayerNorm on. Score = `exp(logit_scale)·cos(state, action)`, scale clamped at 100, initialized at 1/0.07. About 20M parameters per head pair. | Reimplemented from this description with new initialization. |
| Typed-decision interface | `schema.py`: state head sees `context + blank line + instructions`; Choice candidates are option descriptions (key if empty); Noul candidates are `true: …`/`false: …` texts; Score candidates are the level descriptions; answer = softmax over the question's own candidates; Score = expected level index. The card states the probabilities are relative to the offered set. | This is exactly the candidate-relative readout that must not be presented as an absolute Score probability. |
| Objective | README: bidirectional InfoNCE over a B×B batch; mid-training adds explicit hard negatives in the state→action denominator only. `finetune.py --task choice`: in-batch InfoNCE over the batch's distinct option texts (which includes each question's own options) or a per-question softmax (`softce`), soft or hard targets. | We separate in-batch negatives from own-question distractors so that "+ hard negatives" is one attributable change. |
| Candidate cache | `cache.py`: a fixed device arena keyed by head generation and text; README reports hits skipping encoder, copy and head. | Tested as a serving property, measured on our hardware. |
| Fine-tuning data | Public embedding datasets for DeepSWE (`Contrastive-LM/deepswe-clm-train-embeddings-8k`) and a card-less `Contrastive-LM/CLM-v0.1-Pretrain-Nemotron` dump of precomputed Qwen3-8B question/answer vectors (1,876 files, no README or license at revision `05ca7d05…`). | Neither is usable for a Qwen3.5-9B encoder; the pretraining dump has no stated terms. |

## What is **not** public

The README says the scaling experiments, data pipelines and figures live in a
research repository's main branch that is not published. We found no public
builder for the ~60M Nemotron DQA pre-training pairs, the ~30M Gemini-generated
hard negatives, or the ~1M agent-trajectory post-training mixture with 40%
replay, and no public three-stage training script. We therefore do **not**
claim the complete three-stage data or training recipe is public, and we do not
call any arm a reproduction of CLM.

## Hypotheses carried into the ablation grid

1. Frozen-backbone representations suffice for native System One decisions
   (tested against the ordinary Decision head on the same frozen features).
2. Separate state/action encoding loses little relative to joint encoding
   while enabling candidate reuse (joint vs disaggregated with the same head).
3. Learned dual projections beat raw encoder cosine.
4. Explicit hard negatives are needed for fine discrimination (author claim:
   52.1% → 69.2% top-1 on held-out 1-of-11 questions after mid-training).
5. Replay protects earlier behavior (author claim: 69% → 68.5% with 40% replay
   versus 56.2% without). Our analog is soft-distribution replay of our own
   Decision 1.0 Lux 9B on TRAIN inputs; it is a translation of the idea, not
   the authors' data replay.
6. Agent trajectories improve decision transfer — **not testable in Milestone
   1**: the pinned rights-clean TRAIN has no agent trajectories, and no
   trajectory source has passed rights and overlap review.
