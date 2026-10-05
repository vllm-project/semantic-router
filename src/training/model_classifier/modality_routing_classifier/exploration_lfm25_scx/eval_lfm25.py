"""Evaluate the fine-tuned LFM2.5-Encoder (LoRA + classifier head) on the modality
routing test split.

Rebuilds the exact training-time wrapper: body loaded via
AutoModelForMaskedLM(...).lfm2 (NOT the broken documented AutoModel path), the LoRA
adapter applied on top, masked-mean pooling, then the saved linear head.

Usage:
    python eval_lfm25.py --run-dir runs/lfm25_encoder_finetuned
"""

import argparse
from pathlib import Path

import torch
from exploration_common import (
    DATA_DIR,
    MODALITY_LABELS,
    RUNS_DIR,
    format_summary,
    load_jsonl,
    pick_device,
    save_predictions,
    summarize_predictions,
)
from lfm25_classifier import (
    hidden_size_of,
    load_lfm25_body,
    load_lfm25_tokenizer,
    predict_labels,
)
from peft import PeftModel
from torch import nn


def load_trained_model(run_dir: Path, device: str):
    """Rebuild the trained encoder and head from a run directory.

    Args:
        run_dir: Directory written by train_lfm25_encoder.py.
        device: Device to put the model on.

    Returns:
        (tokenizer, body, head), the model parts in eval mode.
    """
    tokenizer = load_lfm25_tokenizer(str(run_dir))
    body = PeftModel.from_pretrained(load_lfm25_body(), run_dir / "lora_adapter")
    body = body.to(device).eval()
    head = nn.Linear(hidden_size_of(body), len(MODALITY_LABELS))
    head.load_state_dict(
        torch.load(
            run_dir / "classifier_head.pt", map_location="cpu", weights_only=True
        )
    )
    return tokenizer, body, head.to(device).eval()


def main(args: argparse.Namespace) -> None:
    """Score the trained model on the test split, print and save the result.

    Args:
        args: Parsed command-line arguments.
    """
    device = pick_device(args.device)
    rows = load_jsonl(args.test_file)
    tokenizer, body, head = load_trained_model(Path(args.run_dir), device)
    preds = predict_labels(
        body,
        head,
        tokenizer,
        [r["text"] for r in rows],
        batch_size=args.batch_size,
        max_length=256,
        device=device,
    )
    metrics = summarize_predictions([r["label_name"] for r in rows], preds)
    print("\n".join(format_summary("LFM2.5-Encoder-350M + LoRA (fine-tuned)", metrics)))
    save_predictions(Path(args.out_json), "lfm25", metrics["accuracy"], preds, rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default=str(RUNS_DIR / "lfm25_encoder_finetuned"))
    parser.add_argument(
        "--out-json", default=str(RUNS_DIR / "lfm25_finetuned_preds.json")
    )
    parser.add_argument("--test-file", default=str(DATA_DIR / "test.jsonl"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
