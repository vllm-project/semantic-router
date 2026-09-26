# Kev-derived 4B research arm

This arm continues the [Apache-2.0 Kev-4B release](https://huggingface.co/jaredpalmer/kev-4b)
with its native question-branch encoder, pointer head, and LoRA adapter. It is
attributed as **Kev-derived**. The published Kev checkpoint remains an unchanged
benchmark baseline. Its architecture differs from `training.model.CandidateHead`,
so the Kev head cannot be loaded into that trainer while preserving its learned
decision behavior.

Pin the published model to Hugging Face revision
`139fdd94f1b6a6ad80cc15e08fcb99cac885a101`, the Qwen3.5-4B-Base to
`1001bb4d826a52d1f399e183466143f4da7b741b`, and the
[native Kev source](https://github.com/jaredpalmer/kev/tree/6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63)
to `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`. Use the model's original
Apache-2.0 notices, Qwen attribution, and each training source's license and
attribution if a derived model is published. The parent model's fitted
temperature changes its published `head.pt` file hash after the training
checkpoint recorded in `provenance.json`; conversion records the exact
HF-revision-attested published head bytes used for the warm start.

## Convert and audit TRAIN

Use Python 3.10+, PyTorch, Transformers with Qwen3.5, PEFT, safetensors,
Pydantic, and the pinned Kev source checkout. The base weights and tokenizer
must already be downloaded in the local HF cache. All commands are offline.
`--audit-prompts` accepts only gold-free prompt JSONL; pass every frozen
unified benchmark and transfer prompt set to check exact overlap without
opening evaluation labels.

```bash
cd /ABS/decision-2-research
PYTHONPATH=/ABS/decision-2-research HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m training.kev.prepare \
  --train /ABS/train-arm.train.jsonl \
  --select /ABS/select.jsonl --cal /ABS/cal.jsonl \
  --audit-prompts /ABS/unified.dev.prompts.jsonl \
  --audit-prompts /ABS/unified.final.prompts.jsonl \
  --model-path /ABS/kev-4b --source-path /ABS/kev-source \
  --output /ABS/kev-derived.train.jsonl --max-state 7552
```

The converter validates all flattened rows and exact train/select/cal role,
ID, lineage-group, canonical-input, state, and order-independent question
overlap. Choice retains candidate order and key label. Noul retains false/true
descriptions and maps the selected key to a boolean. Score is put into numeric
level order and maps the flat selected option to its level number. It then runs
**every** converted request through the pinned Kev `load_records`,
`materialize`, `fits`, and strict tokenizer encoder with the same
`training_context(max_state)` that `kev.train` uses. Any potential overlong
record aborts conversion, because native `kev.train --data` would otherwise
filter it silently. The output manifest hashes the original partitions,
gold-free audits, parent adapter/head, source pin, converted bytes, code, row
counts, and native token limits. The training JSONL is never built from SELECT,
CAL, benchmark development, or benchmark final labels.

## One-GPU native delta trial

This command is a first CE trial; hyperparameters are a hypothesis, not a
qualified result. Use a fresh `--out`. The deliberately zero native option
augmentation and zero `--replay` make every source record traceable to the
audited converter output. Option order augmentation already resides in the
training arm. Keep SELECT for choosing a candidate, CAL for calibration only,
and the frozen final set for a single report after selection.

```bash
cd /ABS/kev-source
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=/ABS/kev-source python3 -m kev.train \
  --base Qwen/Qwen3.5-4B-Base \
  --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --init_from /ABS/kev-4b --data /ABS/kev-derived.train.jsonl \
  --lora 16 --lora_targets all --lora_placement full --head_dim 256 \
  --weights_dtype fp32 --dtype bf16 --checkpointing 1 --device cuda \
  --epochs 1 --batch 2 --accum 4 --lr 2e-5 --head_lr 2e-5 \
  --max_state 7552 --p_none 0 --p_none_distract 0 \
  --p_distract 0 --p_none_pair 0 --replay 0 \
  --out /ABS/kev-derived-run
```

The native trainer creates `adapter_model.safetensors`, `head.pt`,
`adapter_config.json`, `training_config.json`, and `training_metrics.json` in
the run directory. Its warm-start loader checks compatible base, revision,
adapter and pointer-head topology before training. The derived adapter's
inference identity also binds the converted data manifest and the exact
published parent weights. `training.kev.infer --verify-only` checks that the
native trainer requested every converted row in every epoch and reported zero
truncated or rejected rows. Do not score a checkpoint that fails this check.

## Frozen benchmark inference

```bash
cd /ABS/decision-2-research
PYTHONPATH=/ABS/decision-2-research HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m training.kev.infer \
  --checkpoint /ABS/kev-derived-run \
  --parent-model /ABS/kev-4b --source-path /ABS/kev-source \
  --train-data /ABS/kev-derived.train.jsonl \
  --train-manifest /ABS/kev-derived.train.jsonl.manifest.json \
  --verify-only

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/ABS/decision-2-research \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m training.kev.infer \
  --checkpoint /ABS/kev-derived-run \
  --parent-model /ABS/kev-4b --source-path /ABS/kev-source \
  --train-data /ABS/kev-derived.train.jsonl \
  --train-manifest /ABS/kev-derived.train.jsonl.manifest.json \
  --input /ABS/unified.dev.prompts.jsonl \
  --output /ABS/kev-derived.dev.predictions.jsonl \
  --model-id decision2-kev-derived-4b --device cuda:0
```

Inference uses native `Checkpoint.load(LoadOptions())`, Kev's strict encoder,
FP32 eager probabilities, and TypeSafe answer formatter. An overlong question
is explicitly invalid. Each prediction records the exact model, source
prompt, adapter, and input hashes. `--max-items 1` followed by `--resume`
supports a bounded smoke without mixing identities. The native delta head is
uncalibrated (temperature 1.0); fit on the independent CAL partition after
candidate selection before comparing Brier/ECE against a calibrated baseline.

CPU contract tests:

```bash
python3 -m unittest discover -s training/kev/tests -v
```
