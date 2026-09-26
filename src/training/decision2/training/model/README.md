# Decision 2.0 single-GPU pilot

This is a reproducible research trainer for the Qwen3.5-4B/9B text backbone and a
shared dynamic-candidate head. It accepts Choice, Noul, and Score as native
candidate sets (2–255 options). It is not a replacement for the frozen unified
benchmark or its held-out final set.

## Environment and preflight

Use Python 3.10+ and GPU-compatible PyTorch with BF16 support. Install
`transformers`, `safetensors`, and `accelerate`; add `peft` for LoRA. The installed Transformers must
export both `Qwen3_5ForConditionalGeneration` and `Qwen3_5TextModel`. Use the
matching PyTorch CUDA or ROCm build for the accelerator. The source directory
must already contain local weights and tokenizer files; the trainer never
downloads a model or logs in to a service.

```bash
python3 -c 'import torch, transformers, safetensors, accelerate, peft; from transformers import Qwen3_5ForConditionalGeneration; from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel; print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.is_bf16_supported())'
python3 -m unittest discover -s training/model/tests -v
```

The local workstation has no PyTorch installation, so only the standard-library
contract tests ran here. Run the two CPU PyTorch loss tests in the accelerator
environment before launching the pilot.

## Frozen row contract

Each JSONL row has `id`, `state`, `instructions`, `options` (ordered
`[{"key": "...", "description": ...}, ...]`; descriptions may be text or
structured finite JSON, preserving published 1.0 rendering), zero-based integer `label`,
`task_type`, `family`, `group_id`, `language`, `split`, `source`,
`evaluation_role`, `render_template`, `audit_metadata`, and `input_sha256`.
`input_sha256` is SHA-256 of UTF-8 canonical JSON over exactly
`{state,instructions,options,task_type}` with sorted keys, no spaces, and Unicode
unescaped. The supported partitions are train (`split=train, role=train`),
selection (`split=select, role=select`), and calibration (`split=cal,
role=calibrate` or `cal`). IDs, group IDs, and canonical input hashes cannot
cross partitions. No partition can be empty. Benchmark `questions/gold` rows
are rejected by design.

For Noul, candidate keys are exactly `false` and `true`. For Score, keys are
integer levels `0..K-1`; candidate order may be permuted by the data producer.
The data producer owns all option-order augmentation. The trainer never
truncates prompts or silently drops examples. An optional replay JSONL uses the
same train split and adds `teacher_probs` as a complete option-key probability
map. Legacy `target_probs` is rejected.

## First full-tuning pilot on one 256 GB GPU

Use immutable local model revisions and absolute paths. The example chooses
the CE+Brier arm and exposes the batch size rather than fixing it in source.
Start a matched CE arm with `--objective ce` and a distinct output directory.

```bash
CUDA_VISIBLE_DEVICES=7 python3 -m training.model.train \
  --model-path /ABS/Qwen3.5-4B-Base --init-kind base \
  --base-revision IMMUTABLE_MODEL_COMMIT \
  --train /ABS/train.jsonl --select /ABS/select.jsonl --cal /ABS/cal.jsonl \
  --output /ABS/decision2-ce-brier \
  --objective ce_brier --brier-weight 0.5 \
  --epochs 2 --microbatch 1 --accumulation 32 \
  --max-length 4096 --save-every 100
```

This gives an effective optimizer batch of 32 samples (the last window uses its
actual smaller size), FP32 model/head/Adam parameters, BF16 backbone compute,
FP32 head and loss, gradient checkpointing, AdamW and warmup/cosine learning
rates. Initial learning rates are `1e-6` for the backbone and `1e-4` for the
new head. These are pilot values, not validated optima. A 4B FP32 model plus
gradients and Adam states is roughly 64 GB before activations and loading
overhead, so one 256 GB card has plausible capacity; verify the real peak with
the first short run, then adjust microbatch/context length. Use `--max-steps 5`
for a bounded smoke run in a **different** output directory. Do not compare
that shortened run as a full training arm.

To initialize from a published Decision 1.0 checkpoint whose `backbone/` is a
Qwen3.5 text model, use `--init-kind decision1 --model-path /ABS/decision1`.
The 1.0 segmented candidate prompt and head architecture match this pilot's
renderer and head parameters, so this path strictly loads both the trained
text backbone and head. It rejects a mismatched 1.0 prompt or head dimension.
The 1.0 per-type temperatures are not carried over; fit 2.0 calibration after
checkpoint selection. To continue
from a Decision 2.0 checkpoint with a fresh optimizer/schedule, use
`--init-kind decision2` and its directory. For exact interrupted-run resume,
keep every optimization argument and the same train/select/cal files, omit
`--model-path`, and add `--resume /ABS/run/checkpoint-0000100`. Resume checks
data SHA-256, source identity, loss/optimizer contract, code hashes, step,
cursor and RNG states.

Replay is optional: add `--replay /ABS/replay.jsonl --replay-fraction 0.2
--replay-kl-weight 0.3`. It samples replay rows deterministically without
replacing primary train examples; `KL(teacher || student)` is applied only to
sampled replay rows. Its weight is per mixed-batch example, so replay fraction
and KL weight jointly determine the effect. Keep a matched no-replay arm.

`provenance.json` records data/model/code hashes, full optimization contract,
sample and token counts, versions and precision. Every checkpoint saves model,
tokenizer, optimizer, RNG states, data cursor, and select metrics in a pending
directory before an atomic rename. `LATEST.json` and `BEST.json` identify the
latest and best select checkpoint; best selection uses family-macro accuracy,
then normalized Brier, then earliest step. The calibration file is audited for
lineage overlap and hash, but its labels are never evaluated or optimized by
the trainer. Temperature fitting must happen only after checkpoint selection
is frozen.

Risk: the Qwen3.5 model class and local checkpoint structure must match the
installed Transformers version. FP32 full tuning may be limited by checkpoint
disk usage and long-context activation peaks. The first GPU preflight should
exercise one batch, one backward pass, and a save/resume before a full arm.

## Decision 1.0 warm-start LoRA arm

LoRA trains the dynamic head and selected Qwen3.5 text projections while
freezing the original backbone. The target selector checks every hybrid layer:
full-attention Q/K/V/O projections, Gated DeltaNet input QKV/Z/B/A and output
projections, and MLP gate/up/down projections. It rejects a missing or
nonlinear target rather than quietly adapting only part of the model. These
module names follow the [Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py);
the adapter follows [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora).

Use a locally pinned Decision 1.0 Nox-4B or Lux-9B directory with the published
`backbone/`, head, tokenizer, and decision config. The following is a first
4B or 9B arm, not a tuned optimum. Run CE and CE+Brier in separate output
directories with the same train/select/cal data and settings.

```bash
CUDA_VISIBLE_DEVICES=7 python3 -m training.model.train \
  --model-path /ABS/Decision-1.0-Nox-4B --init-kind decision1 \
  --train-mode lora --lora-rank 16 --lora-alpha 32 \
  --lora-dropout 0.05 --lora-lr 1e-4 --head-lr 3e-5 \
  --train /ABS/train.jsonl --select /ABS/select.jsonl --cal /ABS/cal.jsonl \
  --output /ABS/decision2-nox-lora-ce-brier \
  --objective ce_brier --brier-weight 0.5 \
  --epochs 2 --microbatch 1 --accumulation 32 \
  --max-length 4096 --save-every 100
```

For Lux-9B, substitute its local source and a distinct output directory.
The source may be relocated, but its content hashes must remain identical.
`provenance.json` records total and per-component trainable parameter counts,
the exact target list, source file hashes, and optimizer settings. FP32 frozen
base weights still occupy GPU memory; LoRA saves gradient and optimizer state
memory and makes each checkpoint smaller. Profile peak memory and step time
before choosing a longer context or larger microbatch.

LoRA checkpoints contain `adapter/`, an FP32 decision head, tokenizer, and
`decision_config.json`; they deliberately do not duplicate the frozen source
weights. Resume with all original optimization/data arguments, `--resume`,
and `--source-path` pointing to that same content-identical 1.0 directory;
omit `--model-path`. The loader verifies every source model/config/tokenizer
hash and PEFT adapter parameters before restoring the optimizer and RNG state.
The LoRA source directory itself is never modified.

For frozen benchmark inference, add the source path to the existing adapter
command:

```bash
CUDA_VISIBLE_DEVICES=7 python3 -m training.model.infer \
  --checkpoint /ABS/lora-run/checkpoint-0000100 \
  --source-path /ABS/Decision-1.0-Nox-4B \
  --input /ABS/final.prompts.jsonl --output /ABS/lora.final.predictions.jsonl \
  --model-id decision2-nox-lora --model-revision checkpoint-0000100
```

The prediction manifest's model digest includes **source + adapter + head +
tokenizer**, so two adapters over different bases cannot share an identity.
For a self-contained publishable full checkpoint, merge only the selected
adapter after evaluation:

```bash
CUDA_VISIBLE_DEVICES=7 python3 -m training.model.materialize \
  --checkpoint /ABS/lora-run/checkpoint-0000100 \
  --source-path /ABS/Decision-1.0-Nox-4B \
  --output /ABS/decision2-nox-merged --device cuda:0
```

The merged `backbone/` checkpoint loads through the regular inference path
without `--source-path`. Its portable `materialization_receipt.json`, adjacent
receipt, and embedded `lora_origin` retain the exact source and adapter file hashes. Merging is a
separate, potentially high-peak-memory operation; verify merged versus
unmerged predictions on a small gold-free prompt sample before publication.

## Post-hoc calibration on independent CAL

Only after training finishes and `BEST.json` freezes the select-chosen
checkpoint, fit one positive temperature for each native Choice, Noul, and
Score type. This command requires `COMPLETE.json`, verifies that the supplied
CAL JSONL is byte-identical to the file audited in the run's provenance, and
rejects rows outside `split=cal` with role `calibrate` or `cal`. Every CAL row
must be scored without truncation or dropping. No train, select, or benchmark
final labels are used for fitting.

```bash
CUDA_VISIBLE_DEVICES=7 python3 -m training.model.calibrate \
  --run-dir /ABS/decision2-nox-lora-ce-brier \
  --source-path /ABS/Decision-1.0-Nox-4B \
  --cal /ABS/cal.jsonl \
  --output /ABS/decision2-nox-lora-calibration.json \
  --batch-size 2 --device cuda:0
```

Omit `--source-path` for a full-backbone checkpoint. The calibration JSON
stores exact source, checkpoint, CAL, `BEST.json`, `COMPLETE.json`, and run
provenance hashes; three fitted temperatures; CAL NLL, normalized Brier, ECE
with ten bins, and accuracy before/after. Temperatures minimize per-type CAL
NLL in the bounded range 0.05–20. Brier and ECE are measured outcomes and may
move in either direction.

Apply the resulting file explicitly at inference:

```bash
CUDA_VISIBLE_DEVICES=7 python3 -m training.model.infer \
  --checkpoint /ABS/decision2-nox-lora-ce-brier/checkpoint-0000100 \
  --source-path /ABS/Decision-1.0-Nox-4B \
  --calibration /ABS/decision2-nox-lora-calibration.json \
  --input /ABS/final.prompts.jsonl --output /ABS/final.calibrated.predictions.jsonl \
  --model-id decision2-nox-lora --model-revision checkpoint-0000100
```

Inference rejects a calibration file for different source/adapter/head/tokenizer
bytes. The prediction manifest records the calibration file hash and three
temperatures; each prediction row also carries the calibration hash. Without
`--calibration`, the existing scalar `--temperature` path remains available
and defaults to 1.0. A calibration file cannot be combined with a nondefault
scalar temperature.

The selected LoRA checkpoint can later be materialized as a full model. Its
model hash changes, but the same CAL file can be applied **without editing it**
when the merged model's embedded `lora_origin` and its portable
`materialization_receipt.json` (or adjacent `<merged-dir>.materialization.json`)
both prove the selected LoRA source hash and merged checkpoint hash. If a bundle
relocates the receipt, pass its path with `--materialization-receipt`. The inference manifest marks this
binding as `materialized_from_lora`; validate a small gold-free prompt sample
for merged/unmerged numerical equivalence before release.

## Frozen benchmark inference

After a pilot checkpoint is selected, run its gold-blind adapter against the
benchmark **prompts** JSONL. The adapter refuses records containing gold or
provenance. It processes one benchmark item at a time and batches that item's
questions in one forward pass. Use the same maximum prompt length and
temperature when comparing checkpoints. The model revision is a human-readable
immutable ID; `model_sha256` in the output is computed from actual checkpoint
weights, tokenizer, and config files.

```bash
CUDA_VISIBLE_DEVICES=7 python3 -m training.model.infer \
  --checkpoint /ABS/decision2-run/checkpoint-0000100 \
  --input /ABS/dev.prompts.jsonl \
  --output /ABS/decision2.dev.predictions.jsonl \
  --model-id llm-semantic-router/decision-2-pilot \
  --model-revision checkpoint-0000100 \
  --max-length 4096 --temperature 1.0

python3 -m benchmark.score \
  --gold /ABS/dev.gold.jsonl \
  --predictions /ABS/decision2.dev.predictions.jsonl \
  --model-id llm-semantic-router/decision-2-pilot \
  --model-revision checkpoint-0000100 --backend local-dynamic-candidate \
  --output /ABS/decision2.dev.report.json
```

Each prediction has the standardized `id/answers/latency_ms/usage` fields plus
per-item input, adapter, and model hashes. Its adjacent `.manifest.json`
records whole-input and output hashes, exact model/adapter file hashes,
inference settings, and valid/invalid/over-budget question counts. Input text
is never truncated. Questions over `--max-length` produce an explicit invalid
answer; `truncated_questions` remains zero. Runtime exceptions abort the run
instead of fabricating benchmark predictions. The adapter never reads the gold
file; only the separate scorer does.
