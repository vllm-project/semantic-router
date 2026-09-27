# Official Gemma 4 ~27B source feasibility, before training

This is a source and runtime admission check, **not** a Decision 2.0 result.
The candidate source is the official general model
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
at immutable revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.
The upstream model card lists Apache-2.0, 25.2B total / 3.8B active
parameters, a separate vision encoder, 30 MoE decoder layers, 128 experts
with eight chosen per token, and an advertised 256K context. These are source
claims, not measurements of the proposed Decision adapter.

On 2026-09-27, HF CLI on an authorized experiment node downloaded the pinned
configuration, tokenizer and two model shards into a private snapshot. A
header-only audit found 1,013 stored tensors and **25,805,936,206 stored
parameters**: 25,233,141,790 in `model.language_model`, 569,550,384 in the
vision tower, and 3,244,032 in its vision projection. The Hub's safetensors
summary agrees with the stored total. The repository's index separately says
`total_parameters=26,544,131,376`; that metadata is not used as a loaded text
parameter count. The official model-card 25.2B headline and the measured
25.233B text count are consistent to the card's displayed precision; active
3.8B remains an upstream claim until measured for the exact runtime. Full
shard SHA-256s and the immutable revision live in a private audit receipt;
none of these observations imply a Decision benchmark score.

The pinned configuration declares `Gemma4ForConditionalGeneration` with a
`Gemma4TextModel` language submodule, hidden size 2,816 and 262,144 maximum
positions. The qualified training image can instantiate this structure on
meta tensors and load `GemmaTokenizer`. The Qwen Decision 2.0 segmented-option
encoder admits three synthetic, label-free Choice/Noul/Score schema examples
at 102/100/123 tokens after explicit Gemma BOS insertion, with no truncation
under the **tested 8,192-token admission cap**. The existing Qwen-specific
LoRA selector is not valid for Gemma MoE; no Gemma optimizer arm is authorized
from this structural check alone.

The next bounded check uses GPU ordinal 3 on the selected authorized node,
remapped as the only `cuda:0` visible device, and the unchanged official source:
two independent BF16 zero-step loads, a synthetic three-type forward through
the text model and an explicitly **random, untrained** dynamic-option head.
It requires finite logits, wrapper-vs-text hidden-state maximum drift at most
`1e-4`, unchanged categories across processes and logit drift at most `1e-3`.
The combined cap is 15 minutes / 0.25 GPU-hour. Missing weights, OOM, a
native runtime fault, nonfinite output or timeout stops this path for diagnosis;
no training or formal labels are involved. Any later architecture comparison
must add a qualified LoRA/full-weight training path, exact reload and a
matched-data/token-budget development experiment against the already measured
official Qwen3.8 27B candidate. Gemma cannot inherit Qwen or external scores.
