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

A no-GPU meta instantiation reproduces the index's larger declared parameter
count, while the checkpoint omits precisely the `lm_head.weight` key and
the configuration declares tied word embeddings. The stored tensors also
include persistent scalar buffers: their element counts cannot be equated to
`sum(model.parameters())`. The GPU check requires all pinned shard keys and
shapes to match the loaded state, permits only the tied head alias beyond the
index, checks that the output head actually shares embedding storage, and
reconciles loaded parameters plus stored buffers to the stored tensor count.
Both parameter and stored-tensor counts will be reported separately.

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

## Zero-step launch incident and one-time setup amendment

The first authorized zero-step cell stopped after 31 seconds **before any
weights finished loading or forward pass began**. PyTorch reported `No CUDA
GPUs are available` during Transformers' caching-allocator warmup. No optimizer,
formal label, model prediction, or benchmark score was produced. The
conservative reserved use was 31/3600 = 0.00861 GPU-hour; the task containers
were removed by the preregistered cleanup trap. The private failure-log SHA-256
is `905b05bbed3bca7d453137b4b41d85329ce70b90685cb9b9261a71fbf96d2086`
and the timing-receipt SHA-256 is
`245e36ea19ae14b940de219c9607aaa6c92a7dacadb862865cb855bf2be69f1d`.
This is an infrastructure/setup failure, **not** a Gemma capability result.

The successful concurrent 4B training container uses the same exact image
digest, `/dev/kfd` and `/dev/dri` device mounts, `video` group, IPC mode, and
security settings. Its launch sets only `ROCR_VISIBLE_DEVICES` for GPU
isolation. The failed Gemma command set both `ROCR_VISIBLE_DEVICES=3` and
`HIP_VISIBLE_DEVICES=3`; the latter is the only isolation-mask difference.
Because the failed container was removed, its original HostConfig cannot be
inspected independently; the original immutable orchestration script is
preserved under SHA-256
`a6af3945a80ddc54165332febdb9059f040481f18223c95a07d7b1acdd4a85bd`.
This mask interaction is a **hypothesis**. AMD's [HIP environment variable
reference](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.0/reference/env_variables.html)
documents both variables, but a CPU-only smoke cannot prove GPU visibility.

The single proposed correction removes only `HIP_VISIBLE_DEVICES=3` and
retains `ROCR_VISIBLE_DEVICES=3`, matching the qualified isolation form.
The amended orchestration script SHA-256 is
`12c2c3e0dd8a6cf816bd83579b11ecb6d2ec6b9e82007cf03d40cd0fa5a31993`.
It uses new container and receipt names, so it cannot overwrite the failure.
`bash -n` passes. A no-device, no-network image smoke with the corrected
environment imported PyTorch 2.12.0+git6bbd260 (HIP 7.2.53211) and
Transformers 5.17.0, observed `ROCR_VISIBLE_DEVICES=3`, no HIP mask, and no
`/dev/kfd`; this checks the setup without initializing a GPU. The Gemma source
script and pinned checkpoint are unchanged.

The proposed **one-time** retry keeps the original two independent BF16
zero-step loads, three synthetic schema inputs, strict source-key/shape and
tied-head checks, finite-logit and two-load agreement thresholds, and a combined
15-minute / 0.25 GPU-hour cap. Before any retry, recheck the requested GPU's
occupancy, official source revision, both weight-shard hashes, and the mirrored
source-script hash. If GPU enumeration or any model check still fails, stop,
preserve new receipts, and do not search further setup variants or launch
training. No retry is authorized by this amendment itself.

## Corrected zero-step result

After separate approval, the one corrected GPU-3 cell completed successfully
with exit status 0 in 112 seconds, or **0.03111 conservative GPU-hour**. Both
independent BF16 loads produced byte-identical synthetic-probe receipts
(SHA-256 `a730f0e21250f3035a2fe2413ec95968cf3a1804b2262ee5151d9d78eb86c313`).
The strict cross-load comparison passed (receipt SHA-256
`b72269a9e2ea2c8e0e029c1b72dd241baf8a70cad7b83b39ea45e5b9f163b1ef`),
with unchanged selected categories and maximum logit drift 0 against the
preregistered `1e-3` cap. The timing receipt SHA-256 is
`060c2d9248b3e8dd044cbdb90bec2c9a4a89aaf915aaa79c026b778eb2de2ef3`.
Both task containers exited and the selected GPU returned to 0% allocated
memory. There was **no training, formal-label access, or Decision score**.

Actual loaded parameters are **25,805,933,872 total** and **25,233,141,760
text**, excluding the tied `lm_head` alias. A further 2,334 persisted buffer
elements account exactly for the 25,805,936,206 stored tensor elements; 30 of
those are in the text model. The loaded output head shares embedding storage.
Peak allocated GPU memory was 51,744,454,656 bytes. All official shard keys
and shapes matched the pinned index, and no unexpected or missing language
weights were accepted.

Choice, Noul, and Score synthetic inputs admitted 102, 100, and 123 tokens
respectively with zero truncation at the tested 8,192-token cap. All logits
were finite, and wrapper-versus-text hidden-state drift was 0 against the
`1e-4` cap. These logits come from an **untrained random decision head**;
their selected categories carry no quality evidence. Advertised 256K context,
active-parameter count, calibration, transfer, and task quality remain
unverified.

The next discriminative experiment is a separate preregistered, matched-data
development comparison with the existing official Qwen3.8-27B arm. It first
needs a Gemma-aware MoE adapter target selection and source-reload test, then
the same training examples, token/step budget, decision-head protocol, and
SELECT/DEV scoring. A source-only result is insufficient to select Gemma as a
Decision 2.0 training base or to publish a 27B model.
