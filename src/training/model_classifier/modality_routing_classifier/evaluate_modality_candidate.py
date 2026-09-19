"""
Three-Way Evaluation for the Modality Routing Candidate Experiment (Issue #3198)
=================================================================================

Evaluates the published production baseline, a leakage-free "clean baseline"
reproduction, and the distilled candidate against export_modality_dataset.py's
deterministic test.jsonl, and reports:

  1. A test-set contamination check: whether a test row's text (exact or
     whitespace/case-normalized) also appears in train.jsonl/validation.jsonl.
     Motivated by a measured finding, not speculation -- one of the real DIFFUSION-
     class source datasets (Gustavosta/Stable-Diffusion-Prompts) was found to have
     a 32.44% exact-duplicate row rate, and the exporter's stratified split does not
     deduplicate before assigning train/val/test membership. See DECISION_RECORD.md.
  2. Standard classification metrics (accuracy, weighted F1, per-class P/R/F1,
     confusion matrix) for each model, computed on both the full test set and the
     contamination-filtered subset.
  3. Three pairwise routing-agreement metrics (does the pair predict the same label;
     on disagreement, which side matched ground truth), also full and filtered:
       - candidate vs clean_baseline   (primary, leakage-free graduation-gate pair)
       - candidate vs published_baseline (secondary/reference; what #3856's harness
         will eventually re-run with real latency numbers attached)
       - clean_baseline vs published_baseline (quantifies how much the cross-run
         leakage risk actually mattered, empirically)

Same-run p99 latency (#3856) and fixed-policy control comparison (#3857) are
explicitly out of scope here -- see the `pending_dependencies` block in the report.

Usage:
    python evaluate_modality_candidate.py \\
        --test-file exported_modality_routing_dataset/test.jsonl \\
        --train-file exported_modality_routing_dataset/train.jsonl \\
        --val-file exported_modality_routing_dataset/validation.jsonl \\
        --published-baseline-model-path llm-semantic-router/mmbert32k-modality-router-merged \\
        --clean-baseline-model-path models/mmbert32k-modality-router-clean-merged \\
        --candidate-model-path models/distilbert-modality-router-candidate-merged \\
        --output-report modality_candidate_eval_report.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)
sys.path.append(os.path.dirname(_THIS_DIR))

from common_lora_utils import (  # noqa: E402
    load_sequence_classifier_for_inference,
    setup_logging,
)
from modality_data import load_jsonl  # noqa: E402
from modality_eval_metrics import (  # noqa: E402
    AGREEMENT_PAIRS,
    MODEL_KEYS,
    build_report,
    flag_test_contamination,
)
from modality_label_mapping import (  # noqa: E402
    MODALITY_LABELS,
    build_label_remap,
    check_output_size,
)

logger = setup_logging()


@dataclass(frozen=True)
class EvalConfig:
    """Inputs of one three-way evaluation.

    Attributes:
        test_file: Path to test.jsonl.
        train_file: Path to train.jsonl, used for the contamination check.
        val_file: Path to validation.jsonl, used for the contamination check.
        published_baseline_model_path: Checkpoint of the published baseline.
        clean_baseline_model_path: Checkpoint of the clean baseline.
        candidate_model_path: Checkpoint of the candidate.
        output_report: Where to write the JSON report.
        batch_size: Number of texts per forward pass.
        max_length: Token limit; longer texts are truncated.
        include_full_text: Whether to store each prompt's text in the records.
    """

    test_file: str
    train_file: str
    val_file: str
    published_baseline_model_path: str
    clean_baseline_model_path: str
    candidate_model_path: str
    output_report: str = "modality_candidate_eval_report.json"
    batch_size: int = 32
    max_length: int = 256
    include_full_text: bool = False

    @property
    def model_paths(self) -> dict[str, str]:
        """Checkpoint path or repo id per model key."""
        return {
            "published_baseline": self.published_baseline_model_path,
            "clean_baseline": self.clean_baseline_model_path,
            "candidate": self.candidate_model_path,
        }


def load_checkpoint(model_path: str):
    """Load a checkpoint and check that its labels map onto the canonical ones.

    Accepts a LoRA adapter directory or a merged checkpoint, and logs the label
    mapping it resolved so the eval log shows which one was used.

    Args:
        model_path: Local directory or Hub repo id of the checkpoint.

    Returns:
        (model, tokenizer, remap) where remap maps checkpoint class ids to
        canonical class ids.

    Raises:
        ValueError: If the checkpoint's labels do not map onto AR/DIFFUSION/BOTH.
    """
    model, tokenizer, id2label = load_sequence_classifier_for_inference(
        model_path, num_labels=len(MODALITY_LABELS)
    )
    remap = build_label_remap(id2label, model_path)
    check_output_size(remap, model.config.num_labels, model_path)
    logger.info(
        "Label mapping for %s: %s",
        model_path,
        ", ".join(
            f"{ckpt_id}={id2label[ckpt_id]}->{canonical_id}"
            for ckpt_id, canonical_id in sorted(remap.items())
        )
        + " (checkpoint id=name->canonical id)",
    )
    return model, tokenizer, remap


def predict_canonical(
    model,
    tokenizer,
    remap: dict[int, int],
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    """Predict the canonical label id of each text.

    Args:
        model: A sequence classifier already on device and in eval mode.
        tokenizer: The model's tokenizer.
        remap: Mapping from the model's class ids to canonical class ids.
        texts: Prompt texts to classify.
        batch_size: Number of texts per forward pass.
        max_length: Token limit; longer texts are truncated.
        device: Device the model is on.

    Returns:
        Array of shape [N] with canonical label ids, in the order of texts.
    """
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            batch_preds = torch.argmax(model(**enc).logits, dim=-1).cpu().tolist()
            preds.extend(remap[p] for p in batch_preds)
    return np.array(preds, dtype=np.int64)


def run_inference(
    model_path: str, texts: list[str], batch_size: int, max_length: int
) -> np.ndarray:
    """Load one checkpoint, predict every text, and free the model.

    Args:
        model_path: Local directory or Hub repo id of the checkpoint.
        texts: Prompt texts to classify.
        batch_size: Number of texts per forward pass.
        max_length: Token limit; longer texts are truncated.

    Returns:
        Array of shape [N] with canonical label ids, in the order of texts.
    """
    model, tokenizer, remap = load_checkpoint(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    try:
        return predict_canonical(
            model,
            tokenizer,
            remap,
            texts,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def log_summary(report: dict) -> None:
    """Log accuracy per model and agreement per pair on the full test set.

    Args:
        report: Report from build_report.
    """
    full = report["full_test_set"]
    logger.info("=" * 70)
    logger.info("Summary (full test set):")
    for key in MODEL_KEYS:
        logger.info(f"  {key}: accuracy={full['metrics'][key]['accuracy']:.4f}")
    for a, b in AGREEMENT_PAIRS:
        rate = full["routing_agreement"][f"{a}_vs_{b}"]["agreement_rate"]
        logger.info(f"  agreement {a} vs {b}: {rate:.4f}")
    logger.info("=" * 70)


def main(config: EvalConfig) -> dict:
    """Run the three-way evaluation and write the JSON report.

    Args:
        config: The evaluation inputs.

    Returns:
        The report that was written.
    """
    test_rows = load_jsonl(config.test_file)
    train_rows = load_jsonl(config.train_file)
    val_rows = load_jsonl(config.val_file)
    logger.info(
        f"Loaded {len(test_rows)} test rows, {len(train_rows)} train, {len(val_rows)} val"
    )

    contamination = flag_test_contamination(
        [r["text"] for r in train_rows], [r["text"] for r in val_rows], test_rows
    )
    logger.info(
        f"Contamination check: {len(contamination[0])}/{len(test_rows)} test rows "
        f"({len(contamination[0]) / len(test_rows) * 100:.2f}%) have an exact/normalized "
        f"duplicate in train+val. By class: {contamination[1]}"
    )

    texts = [r["text"] for r in test_rows]
    preds: dict[str, np.ndarray] = {}
    for key, path in config.model_paths.items():
        logger.info(f"Running inference for {key}: {path}")
        preds[key] = run_inference(path, texts, config.batch_size, config.max_length)

    report = build_report(
        test_rows,
        preds,
        contamination,
        paths={
            "test_file": config.test_file,
            "train_file": config.train_file,
            "val_file": config.val_file,
        },
        model_paths=config.model_paths,
        generated_at=datetime.now(timezone.utc),
        include_full_text=config.include_full_text,
    )
    if report["contamination_filtered_test_set"] is None:
        logger.warning(
            "Contamination filter removed all test rows; filtered report skipped."
        )
    with open(config.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report written to: {config.output_report}")
    log_summary(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser for the evaluation's options.
    """
    parser = argparse.ArgumentParser(
        description="Three-way modality routing candidate evaluation"
    )
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--published-baseline-model-path", required=True)
    parser.add_argument("--clean-baseline-model-path", required=True)
    parser.add_argument("--candidate-model-path", required=True)
    parser.add_argument(
        "--output-report", default="modality_candidate_eval_report.json"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--include-full-text", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> EvalConfig:
    """Turn parsed command-line arguments into an EvalConfig.

    Args:
        args: Result of build_parser().parse_args().

    Returns:
        The evaluation inputs.
    """
    return EvalConfig(
        test_file=args.test_file,
        train_file=args.train_file,
        val_file=args.val_file,
        published_baseline_model_path=args.published_baseline_model_path,
        clean_baseline_model_path=args.clean_baseline_model_path,
        candidate_model_path=args.candidate_model_path,
        output_report=args.output_report,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_full_text=args.include_full_text,
    )


if __name__ == "__main__":
    main(config_from_args(build_parser().parse_args()))
