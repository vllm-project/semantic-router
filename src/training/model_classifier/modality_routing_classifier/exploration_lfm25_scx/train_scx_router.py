"""Curiosity exploration: fine-tune SCX Router v0.1 (GLiClass, decoder-KV Qwen3-0.6B
backbone) on the modality-routing task, using gliclass's own native training pipeline
(full fine-tune, not LoRA -- PEFT compatibility with this custom nested architecture
wasn't verified, and this is exploratory, not decision-record-quality work).

Note: this checkpoint's decoder-KV architecture only implements a working get_loss()
path for its native problem_type ("multi_label_classification": per-label independent
BCE/focal loss, labels as a multi-hot vector). Its "single_label_classification" path
is broken as shipped (labels[:, :num_labels] assumes a 2D tensor that a scalar class
index never produces, and the collator drops 0-dim label tensors entirely), verified by
direct reproduction. See exploration_common.make_gliclass_dataset for how the
mutually exclusive 3-class task is expressed as multi-hot.

Usage:
    python train_scx_router.py --output-dir runs/scx_router_finetuned
"""

import argparse
import sys

from exploration_common import (
    DATA_DIR,
    RUNS_DIR,
    SCX_MODEL_ID,
    SCX_REVISION,
    load_jsonl,
    make_gliclass_dataset,
    pick_device,
)


def build_training_args(args: argparse.Namespace):
    """Build the gliclass TrainingArguments for the run.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The TrainingArguments.
    """
    import torch  # noqa: PLC0415  (lazy: keeps the parser importable without gliclass)
    from gliclass.training import TrainingArguments  # noqa: PLC0415

    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        logging_steps=20,
        eval_strategy="epoch",
        # Two consecutive WSL crashes both hit right after the first epoch's 3.4GB
        # checkpoint write (2.5GB weights + 1.2GB optimizer state) -- no intermediate
        # checkpoints, and no optimizer state saved, to avoid large write bursts.
        save_strategy="no",
        save_optimizer_state=False,
        report_to="none",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )


def main(args: argparse.Namespace) -> None:
    """Fine-tune SCX Router and save it in bf16.

    Args:
        args: Parsed command-line arguments.
    """
    import torch  # noqa: PLC0415
    from gliclass import GLiClassModel  # noqa: PLC0415
    from gliclass.data_processing import DataCollatorWithPadding  # noqa: PLC0415
    from gliclass.training import Trainer  # noqa: PLC0415
    from transformers import AutoTokenizer, set_seed  # noqa: PLC0415

    set_seed(args.seed)
    print("Loading model + tokenizer...", file=sys.stderr)
    model = GLiClassModel.from_pretrained(SCX_MODEL_ID, revision=SCX_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(SCX_MODEL_ID, revision=SCX_REVISION)
    print(f"Native problem_type: {model.config.problem_type}", file=sys.stderr)

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    print(f"Train: {len(train_rows)}, Val: {len(val_rows)}", file=sys.stderr)

    trainer = Trainer(
        model=model,
        args=build_training_args(args),
        train_dataset=make_gliclass_dataset(train_rows, tokenizer),
        eval_dataset=make_gliclass_dataset(val_rows, tokenizer),
        data_collator=DataCollatorWithPadding(
            device=pick_device(args.device), config=model.config
        ),
    )
    print("Starting training...", file=sys.stderr)
    trainer.train()

    model.to(torch.bfloat16).save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned SCX Router to: {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-file", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--val-file", default=str(DATA_DIR / "validation.jsonl"))
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "scx_router_finetuned"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
