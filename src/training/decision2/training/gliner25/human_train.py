"""One frozen head-only GLiNER2.5 human-TRAIN continuation screen.

Run on an authorized GPU experiment host. Preflight performs no optimizer
step; the 32-step trainer reads only the approved native TRAIN export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inference import gliner25

from training.gliner25.human_pilot import CONTRACT, EXPECTED_HASHES
from training.model.data import file_sha256

UPDATES = 32
BATCH_SIZE = 2
ACCUMULATION = 8
TASK_LR = 5e-6
ENCODER_LR = 1e-6  # frozen encoder; this optimizer group is empty
WARMUP_STEPS = 4
WEIGHT_DECAY = 0.01
ADAM_EPS = 1e-8
SEED = 20260928
TRAINABLE_PARAMETERS = 2_101_249
TOTAL_PARAMETERS = 486_444_053


def _inputs(source: Path, native_train: Path, manifest: Path) -> dict[str, Any]:
    if not source.is_dir() or not native_train.is_file() or not manifest.is_file():
        raise FileNotFoundError("Frozen human pilot source or TRAIN file is absent")
    gliner25.verify_release(source, gliner25.REVISION)
    detail = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        detail.get("contract") != CONTRACT
        or detail.get("source_data_sha256") != EXPECTED_HASHES
        or detail.get("source_revision") != gliner25.REVISION
        or detail.get("source_weights_sha256")
        != gliner25.MODEL_FILES["model.safetensors"]
        or detail.get("library_commit") != gliner25.LIBRARY_COMMIT
        or detail.get("selected_rows") != 512
        or detail.get("selected_groups") != 384
        or detail.get("selected_by_type") != {"choice": 336, "noul": 176}
        or detail.get("native_train_sha256") != file_sha256(native_train)
    ):
        raise ValueError("Frozen human pilot manifest mismatch")
    return detail


def _trainer(source: Path, output: Path):
    from gliner2 import GLiNER2
    from gliner2.processor import SchemaTransformer
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    class ExactSchemaProcessor(SchemaTransformer):
        """Disable training-only label removal and synthetic label insertion."""

        def _process_classifications(self, schema, schemas, labels, types, sampling):
            was_training = self.is_training
            self.is_training = False
            try:
                return super()._process_classifications(
                    schema, schemas, labels, types, sampling=None
                )
            finally:
                self.is_training = was_training

    model = GLiNER2.from_pretrained(str(source))
    if sum(p.numel() for p in model.parameters()) != TOTAL_PARAMETERS:
        raise ValueError("GLiNER source parameter count changed")
    processor = ExactSchemaProcessor(
        tokenizer=model.processor.tokenizer,
        token_pooling=model.processor.token_pooling,
        word_splitter=model.processor.word_splitter,
    )
    config = TrainingConfig(
        output_dir=str(output),
        experiment_name="decision2-gliner25-human-head512-pilot",
        num_epochs=1,
        max_steps=UPDATES,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION,
        encoder_lr=ENCODER_LR,
        task_lr=TASK_LR,
        weight_decay=WEIGHT_DECAY,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=ADAM_EPS,
        scheduler_type="linear",
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        max_len=512,
        eval_strategy="no",
        save_best=False,
        save_total_limit=0,
        logging_steps=4,
        report_to_wandb=False,
        num_workers=0,
        validate_data=False,
        strict_training=True,
        skip_step_errors=False,
        allow_invalid_samples=False,
        fused_optimizer=False,
        seed=SEED,
    )
    return model, processor, config, GLiNER2Trainer


def _freeze_head(model: Any) -> int:
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("classifier."))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if trainable != [
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.2.weight",
        "classifier.2.bias",
    ]:
        raise ValueError("Head-only parameter names changed")
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != TRAINABLE_PARAMETERS:
        raise ValueError("Head-only parameter count changed")
    return count


def run(
    *,
    source: Path,
    native_train: Path,
    manifest: Path,
    output: Path,
    preflight: bool,
    preflight_receipt: Path | None,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU is required for the pinned native pilot")
    if output.exists():
        raise FileExistsError(output)
    detail = _inputs(source, native_train, manifest)
    model, processor, config, trainer_class = _trainer(source, output)
    trainable = _freeze_head(model)
    trainer = trainer_class(model=model, config=config, processor=processor)
    if trainer.device.type != "cuda":
        raise RuntimeError("Trainer silently fell back to CPU")
    common = {
        "contract": CONTRACT,
        "source_revision": gliner25.REVISION,
        "source_weights_sha256": gliner25.MODEL_FILES["model.safetensors"],
        "library_commit": gliner25.LIBRARY_COMMIT,
        "native_train_sha256": detail["native_train_sha256"],
        "manifest_sha256": file_sha256(manifest),
        "trainable_parameters": trainable,
        "total_parameters": TOTAL_PARAMETERS,
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    if preflight:
        from gliner2 import GLiNER2

        dataset = trainer._prepare_data(native_train, is_train=True)
        loader = trainer._create_dataloader(
            dataset, BATCH_SIZE, shuffle=False, is_training=True
        )
        batch = next(iter(loader))
        reference = GLiNER2.from_pretrained(str(source)).to(trainer.device).eval()
        trainer.model.eval()
        with torch.inference_mode():
            before = reference(batch)["total_loss"].float()
            after = trainer.model(batch)["total_loss"].float()
        parity = float((before - after).abs())
        if not torch.isfinite(before) or not torch.isfinite(after) or parity > 1e-6:
            raise RuntimeError("Frozen-head native loss differs from source")
        del reference
        trainer.model.train()
        trainer.processor.change_mode(is_training=True)
        loss = trainer.model(batch)["total_loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite source head preflight loss")
        loss.backward()
        gradients = [
            p.grad for p in trainer.model.classifier.parameters() if p.grad is not None
        ]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("No finite classifier gradient")
        if not any(g.abs().sum() > 0 for g in gradients):
            raise RuntimeError("Classifier gradient is identically zero")
        if any(p.grad is not None for p in trainer.model.encoder.parameters()):
            raise RuntimeError("Frozen encoder acquired a gradient")
        optimizer = trainer._create_optimizer()
        actual = [len(group["params"]) for group in optimizer.param_groups]
        if actual != [0, 4]:
            raise RuntimeError("Optimizer includes parameters outside the head")
        return {
            **common,
            "source_native_loss_parity_max_abs": parity,
            "preflight_loss": float(loss.detach()),
            "classifier_gradient_finite_nonzero": True,
            "frozen_encoder_gradient_count": 0,
            "optimizer_group_tensor_counts": actual,
            "optimizer_steps": 0,
        }
    if preflight_receipt is None:
        raise ValueError("A frozen preflight receipt is required")
    checked = json.loads(preflight_receipt.read_text(encoding="utf-8"))
    if any(checked.get(key) != value for key, value in common.items()) or (
        checked.get("optimizer_steps") != 0
        or checked.get("source_native_loss_parity_max_abs", float("inf")) > 1e-6
        or checked.get("classifier_gradient_finite_nonzero") is not True
        or checked.get("optimizer_group_tensor_counts") != [0, 4]
    ):
        raise ValueError("Preflight receipt does not match this source and data")
    result = trainer.train(train_data=native_train)
    if result["total_steps"] != UPDATES:
        raise RuntimeError("Human head pilot did not complete 32 updates")
    weights = output / "final" / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError("Final native GLiNER checkpoint is absent")
    return {
        **common,
        "preflight_receipt_sha256": file_sha256(preflight_receipt),
        "optimizer_steps": result["total_steps"],
        "task_lr": TASK_LR,
        "encoder_lr": ENCODER_LR,
        "warmup_steps": WARMUP_STEPS,
        "effective_batch": BATCH_SIZE * ACCUMULATION,
        "final_weights_sha256": file_sha256(weights),
        "time_seconds": result["total_time_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "native-train", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
