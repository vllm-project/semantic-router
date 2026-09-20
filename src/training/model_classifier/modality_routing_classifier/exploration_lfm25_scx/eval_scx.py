"""Evaluate an SCX Router (GLiClass decoder-KV) checkpoint on the modality-routing test
split, using the SAME GLiClassDataset/collator path as training so the label formatting
matches what the model saw.

Usage:
    python eval_scx.py <model_path_or_repo> <out_json>
"""

import argparse
from pathlib import Path

from exploration_common import (
    DATA_DIR,
    SCX_REVISION,
    format_summary,
    load_jsonl,
    make_gliclass_dataset,
    pick_device,
    save_predictions,
    summarize_predictions,
)


def predict_scx(model, dataset, collator, *, batch_size: int, device: str) -> list[str]:
    """Predict a label name for every example of a GLiClass dataset.

    Args:
        model: The SCX model, on device and in eval mode.
        dataset: A GLiClassDataset built with shuffle_labels=False.
        collator: The GLiClass collator for the model's config.
        batch_size: Number of examples per forward pass.
        device: Device the model is on.

    Returns:
        Predicted label names, in dataset order.
    """
    import torch  # noqa: PLC0415  (lazy: keeps the module importable without torch)

    preds: list[str] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = collator(
                [
                    dataset[i]
                    for i in range(start, min(start + batch_size, len(dataset)))
                ]
            )
            labels_text = batch.pop("labels_text")
            batch.pop("input_texts", None)
            batch.pop("labels", None)
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(**batch).logits.float()
            for i in range(logits.shape[0]):
                n_labels = len(labels_text[i])
                preds.append(labels_text[i][int(logits[i, :n_labels].argmax())])
    return preds


def main(args: argparse.Namespace) -> None:
    """Load a checkpoint, score it on the test split, print and save the result.

    Args:
        args: Parsed command-line arguments.
    """
    from gliclass import GLiClassModel  # noqa: PLC0415
    from gliclass.data_processing import DataCollatorWithPadding  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    device = pick_device(args.device)
    rows = load_jsonl(args.test_file)
    is_local = Path(args.model_path).exists()
    revision = "main" if is_local else SCX_REVISION  # a local directory ignores it
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=revision)
    model = GLiClassModel.from_pretrained(args.model_path, revision=revision)
    model = model.to(device).eval()

    preds = predict_scx(
        model,
        make_gliclass_dataset(rows, tokenizer, shuffle_labels=False),
        DataCollatorWithPadding(device=device, config=model.config),
        batch_size=args.batch_size,
        device=device,
    )
    metrics = summarize_predictions([r["label_name"] for r in rows], preds)
    print("\n".join(format_summary(args.model_path, metrics)))
    save_predictions(
        Path(args.out_json), args.model_path, metrics["accuracy"], preds, rows
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_path", help="local checkpoint directory or Hub repo id")
    parser.add_argument("out_json", help="where to write the predictions")
    parser.add_argument("--test-file", default=str(DATA_DIR / "test.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
