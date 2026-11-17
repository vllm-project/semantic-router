# MMLU-Pro solver LoRA experiment

Fine-tune specialized Qwen3-0.6B solvers for MMLU-Pro category groups. This
directory is an experiment, not a release-owning trainer.

## Setup

Create the environment from this folder. `pyproject.toml` bounds the
dependencies and `uv.lock` pins them:

```bash
uv sync --locked
```

`requirements.txt` stays for anyone who installs with `pip`.

Run the Python commands below with `uv run`. From the repository root:

```bash
uv run --project src/training/model_experiment/mmlu_pro_solver_lora \
  python src/training/model_experiment/mmlu_pro_solver_lora/ft_qwen3_mmlu_solver_lora.py --help

uv run --project src/training/model_experiment/mmlu_pro_solver_lora \
  python src/training/model_experiment/mmlu_pro_solver_lora/ft_qwen3_mmlu_solver_lora_no_leakage.py --help
```

The shared helpers under `src/training/model_classifier/` resolve through each
script's path bootstrap. The training-contract import check for this area is:

```bash
uv run --project src/training/model_experiment/mmlu_pro_solver_lora \
  python -m unittest src.training.tests.test_training_entrypoint_imports
```

## Train

Start with a limited sample from the repository root:

```bash
uv run --project src/training/model_experiment/mmlu_pro_solver_lora \
  python src/training/model_experiment/mmlu_pro_solver_lora/ft_qwen3_mmlu_solver_lora.py \
  --mode train \
  --model-type math-reasoner \
  --epochs 1 \
  --batch-size 1 \
  --max-samples-per-category 20 \
  --output-dir /tmp/mmlu-pro-solver-smoke
```

The no-leakage trainer trains on external datasets and evaluates on MMLU-Pro.
See `ft_qwen3_mmlu_solver_lora_no_leakage.py --help` for options.

## Batch scripts

Both scripts run through the locked uv project and can be started from the
repository root:

```bash
./src/training/model_experiment/mmlu_pro_solver_lora/train_all_specialists.sh [GPU_ID]
./src/training/model_experiment/mmlu_pro_solver_lora/train_all_specialists_no_leakage.sh [GPU_ID] [SAMPLES_PER_DATASET] [EPOCHS]
```

Generated models, logs, and `.dataset_cache` stay outside Git.
