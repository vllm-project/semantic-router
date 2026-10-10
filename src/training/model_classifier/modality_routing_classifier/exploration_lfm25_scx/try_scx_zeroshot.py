"""Quick curiosity probe: zero-shot SCX Router v0.1 on our modality-routing test set.

Not part of the formal decision-record pipeline, exploratory only.

Usage:
    python try_scx_zeroshot.py --out-json runs/scx_zeroshot_results.json
"""

import argparse
import json
import sys
from pathlib import Path

from exploration_common import (
    DATA_DIR,
    LABEL_DESCRIPTIONS,
    MODALITY_LABELS,
    RUNS_DIR,
    SCX_MODEL_ID,
    SCX_REVISION,
    format_summary,
    load_jsonl,
    pick_device,
    summarize_predictions,
)


def classify_rows(pipe, rows: list[dict]) -> list[str]:
    """Classify every row zero-shot with the highest-scoring label description.

    Args:
        pipe: A ZeroShotClassificationPipeline.
        rows: Rows with "text".

    Returns:
        Predicted label names, in row order.
    """
    label_texts = [f"{name}: {desc}" for name, desc in LABEL_DESCRIPTIONS.items()]
    name_by_text = dict(zip(label_texts, MODALITY_LABELS, strict=True))
    preds = []
    for row in rows:
        scores = pipe(row["text"], label_texts, threshold=0.0)[0]
        preds.append(name_by_text[max(scores, key=lambda x: x["score"])["label"]])
    return preds


def main(args: argparse.Namespace) -> None:
    """Run the zero-shot probe, print the summary and save the per-row results.

    Args:
        args: Parsed command-line arguments.
    """
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    device = pick_device(args.device)
    print("Loading SCX Router v0.1...", file=sys.stderr)
    model = GLiClassModel.from_pretrained(SCX_MODEL_ID, revision=SCX_REVISION).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(SCX_MODEL_ID, revision=SCX_REVISION)
    pipe = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type="multi-class",
        device=device,
        progress_bar=False,
    )

    rows = load_jsonl(args.test_file)
    print(f"Loaded {len(rows)} test rows", file=sys.stderr)
    preds = classify_rows(pipe, rows)
    metrics = summarize_predictions([r["label_name"] for r in rows], preds)
    print("\n".join(format_summary("SCX Router v0.1 (zero-shot)", metrics)))

    results = [
        {
            "text": r["text"][:80],
            "true": r["label_name"],
            "pred": p,
            "correct": r["label_name"] == p,
        }
        for r, p in zip(rows, preds, strict=True)
    ]
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": metrics["accuracy"],
                "confusion_matrix": metrics["confusion_matrix"],
                "per_class": metrics["per_class"],
                "results": results,
            },
            f,
            indent=2,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test-file", default=str(DATA_DIR / "test.jsonl"))
    parser.add_argument(
        "--out-json", default=str(RUNS_DIR / "scx_zeroshot_results.json")
    )
    parser.add_argument(
        "--device", default=None, help="default: cuda if available, else cpu"
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
