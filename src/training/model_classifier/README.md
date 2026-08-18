# Classifier Training

This directory contains research and maintenance scripts for training the text
classifiers used by semantic-router. Each task has its own dataset preparation,
training, export, and verification path.

## Choose a Task

| Directory | Output | Use |
|---|---|---|
| `classifier_model_fine_tuning_lora/` | sequence classifier LoRA | domain or intent classification |
| `pii_model_fine_tuning_lora/` | token classifier LoRA | PII span detection |
| `prompt_guard_fine_tuning_lora/` | binary classifier LoRA | jailbreak and prompt-attack detection |
| `fact_check_fine_tuning_lora/` | binary classifier LoRA | decide whether a prompt needs fact checking |
| `hallucination_detection_classifier/` | token classifier | mark unsupported answer spans |
| `modality_routing_classifier/` | three-class classifier LoRA | choose text, image, or mixed response |
| `user_feedback_classifier/` | four-class classifier | classify satisfaction signals in follow-up text |
| `safety_classifier/` | binary and legacy nine-class LoRA/merged models | hierarchical content-safety classification |

These scripts download models or datasets when needed. Review the task README
before running a full job; dataset licenses, compute needs, and output formats
differ.

## Typical Workflow

1. Create an isolated Python environment and install the dependencies required
   by the selected script.
2. Prepare or download the dataset. Keep generated data and model artifacts out
   of Git.
3. Run a small sample first and inspect label distribution and evaluation
   output.
4. Train the full adapter or model.
5. Run the task's Python test path and, where provided, the Go verifier to check
   compatibility with the native runtime.
6. Publish a model only with a model card that records the base model, dataset,
   split, label mapping, metrics, and limitations.

For example, from this directory:

```bash
cd classifier_model_fine_tuning_lora
python ft_linear_lora.py --help
python ft_linear_lora.py --mode train --model bert-base-uncased
```

Use `--help` as the source of truth for script options; do not copy output from
an old training run into this README.

## Dataset Label Review

`verify_text_classification_datasets.py` audits text-classification labels with
an OpenAI-compatible judge endpoint. A fast first-stage model can filter the
dataset before ambiguous samples are sent to multiple judges.

```bash
python src/training/model_classifier/verify_text_classification_datasets.py \
  --task feedback intent jailbreak fact-check modality \
  --stage1-model STAGE1_MODEL \
  --judge-model JUDGE_A JUDGE_B JUDGE_C \
  --api-url http://localhost:8000/v1/chat/completions \
  --sample 500
```

Add `--correct --confidence high` only when you intend to write a reviewed
corrected dataset. Use `--split all-splits` to audit every available split and
`--dataset-id-override task=PATH_OR_REPO` for a local or alternate dataset.

Reports are written under `verified_datasets_vote/`; they are review artifacts,
not proof that a dataset is correct.

## Runtime Compatibility

The router does not automatically accept every Transformers checkpoint. Native
loading depends on the exported architecture, tokenizer, adapter layout, and
label mapping. Use the verifier supplied with the task where one exists, and
test the resulting artifact through the router before publishing it as
supported.

Public usage and configuration belong in the
[website documentation](../../../website/docs/overview/semantic-router-overview.md). This
directory documents how to produce and inspect training artifacts.
