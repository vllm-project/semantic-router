"""Fixed-budget GLiNER2.5 span continuation with no native-schema augmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference import gliner25

from training.gliner25.pilot import CONTRACT, TRAIN_QUOTAS
from training.model.data import file_sha256

UPDATES = 64
EFFECTIVE_BATCH = 16
ENCODER_LR = 1e-5
TASK_LR = 5e-5


def run(source: Path, native_train: Path, manifest: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source = source.resolve(strict=True)
    gliner25.verify_release(source, gliner25.REVISION)
    detail = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        detail.get("contract") != CONTRACT
        or detail.get("native_train_rows") != sum(TRAIN_QUOTAS.values())
        or detail.get("native_train_sha256") != file_sha256(native_train)
        or detail.get("source_weights_sha256")
        != gliner25.MODEL_FILES["model.safetensors"]
    ):
        raise ValueError("Native training manifest mismatch")

    from gliner2 import GLiNER2
    from gliner2.processor import SchemaTransformer
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    class ExactSchemaProcessor(SchemaTransformer):
        """Preserve the compiled inference labels, instruction and descriptions."""

        def _process_classifications(self, schema, schemas, labels, types, sampling):
            previous = self.is_training
            self.is_training = False
            try:
                return super()._process_classifications(
                    schema, schemas, labels, types, sampling=None
                )
            finally:
                self.is_training = previous

    model = GLiNER2.from_pretrained(str(source))
    processor = ExactSchemaProcessor(
        tokenizer=model.processor.tokenizer,
        token_pooling=model.processor.token_pooling,
        word_splitter=model.processor.word_splitter,
    )
    config = TrainingConfig(
        output_dir=str(output),
        experiment_name="decision2-gliner25-clean-v2-pilot",
        num_epochs=1,
        max_steps=UPDATES,
        batch_size=2,
        gradient_accumulation_steps=EFFECTIVE_BATCH // 2,
        encoder_lr=ENCODER_LR,
        task_lr=TASK_LR,
        bf16=True,
        fp16=False,
        max_len=512,
        eval_strategy="no",
        save_best=False,
        save_total_limit=0,
        logging_steps=8,
        report_to_wandb=False,
        num_workers=0,
        validate_data=False,
        strict_training=True,
        skip_step_errors=False,
        allow_invalid_samples=False,
        fused_optimizer=False,
        seed=20260927,
    )
    trainer = GLiNER2Trainer(model=model, config=config, processor=processor)
    result = trainer.train(train_data=native_train)
    if result["total_steps"] != UPDATES:
        raise ValueError("Pilot did not complete its fixed optimizer budget")
    checkpoint = output / "final"
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise ValueError("Final checkpoint weights missing")
    return {
        "contract": CONTRACT,
        "steps": result["total_steps"],
        "native_train_sha256": file_sha256(native_train),
        "source_weights_sha256": gliner25.MODEL_FILES["model.safetensors"],
        "final_weights_sha256": file_sha256(weights),
        "time_seconds": result["total_time_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native-train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.source, args.native_train, args.manifest, args.output),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
