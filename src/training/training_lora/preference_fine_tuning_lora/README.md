# Qwen3 Preference LoRA (Bootstrap)

Train a small Qwen3 causal LM with LoRA to emit a single preference label (route name) from a user request/short dialogue. The script is self-contained and defaults to a synthetic toy dataset; drop in your own JSONL to train on real preference signals.

## Quickstart

```bash
cd src/training/training_lora/preference_fine_tuning_lora
python ft_qwen3_preference_lora.py --mode train \
  --model-name Qwen/Qwen3-0.6B-Instruct \
  --output-dir ./qwen3_preference_lora \
  --epochs 3 --batch-size 4 --lora-rank 16
```

Test a trained adapter:

```bash
python ft_qwen3_preference_lora.py --mode test \
  --model-path ./qwen3_preference_lora \
  --prompt "I need a fast lightweight model with low latency for mobile"
```

## Dataset

- Default: synthetic ~80 examples with four preference labels: `fast_small`, `high_quality`, `lowest_cost`, `guardrail_first`.
- Custom: pass `--dataset-path your.jsonl` where each line has `{ "text": "...", "preference": "label" }`.

## Key Flags

- `--model-name`: base Qwen3 model (keep to small variants for CPU/GPU memory).
- `--lora-rank`, `--lora-alpha`, `--lora-dropout`: LoRA knobs.
- `--max-length`: truncation length (tokens).
- `--batch-size`, `--grad-accum`, `--learning-rate`, `--epochs`: training knobs.
- `--labels`: optional comma list to constrain allowed labels in prompts.

Outputs are written to `--output-dir` (adapter weights + tokenizer). Use `PeftModel.from_pretrained` to merge at inference time.
