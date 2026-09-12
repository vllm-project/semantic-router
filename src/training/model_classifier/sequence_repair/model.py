"""HF standard sequence checkpoints with an unchanged label contract."""

# Heavy imports stay optional for data-contract tests.
# ruff: noqa: PLC0415

import json
from pathlib import Path

from .data import load_contract
from .task_head import verify_saved_task_head


def configuration_overrides(contract):
    """Validate optional standard HF task settings without guessing a pooling mode."""
    content = json.loads(Path(contract).read_text())
    problem_type = content.get("problem_type", "single_label_classification")
    if problem_type not in {
        "single_label_classification",
        "multi_label_classification",
    }:
        raise ValueError("Unsupported classification problem type")
    overrides = {"problem_type": problem_type}
    if "classifier_pooling" in content:
        pooling = content["classifier_pooling"]
        if pooling not in {"mean", "cls"}:
            raise ValueError("Sequence pooling must be explicitly mean or cls")
        overrides["classifier_pooling"] = pooling
    return overrides


def load_model(base, contract, adapter=None, trainable=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    label_to_id, id_to_label = load_contract(contract)
    overrides = configuration_overrides(contract)
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        num_labels=len(label_to_id),
        label2id=label_to_id,
        id2label=id_to_label,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        reference_compile=False,
        **overrides,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        raise ValueError("Continuation requires an explicit initial adapter")
    if model.config.label2id != label_to_id:
        raise ValueError("The loaded classifier changed the label contract")
    return model, tokenizer, label_to_id, id_to_label


def save_adapter(model, tokenizer, destination, base_id, base_revision):
    import json

    destination.mkdir(parents=True, exist_ok=True)
    model.peft_config["default"].base_model_name_or_path = base_id
    model.peft_config["default"].revision = base_revision
    model.save_pretrained(destination)
    scope = verify_saved_task_head(model, destination / "adapter_model.safetensors")
    (destination / "task-head.json").write_text(json.dumps(scope, indent=2) + "\n")
    tokenizer.save_pretrained(destination)
    (destination / "label_mapping.json").write_text(
        json.dumps(
            {"label2id": model.config.label2id, "id2label": model.config.id2label},
            indent=2,
        )
        + "\n"
    )
