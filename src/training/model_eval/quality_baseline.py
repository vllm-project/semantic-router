"""Versioned quality baseline for the built-in Router Models (#3197).

What this adds over ``mom_collection_eval.py``:

* It measures the artifact a maintained configuration actually loads, resolved
  from ``config/config.yaml`` rather than from a hand-maintained registry.
* It takes the label order from the artifact's own mapping and reports when a
  copy kept elsewhere disagrees, instead of silently scoring the wrong class.
* It reports calibration and threshold behaviour, because a router consumes a
  score and a threshold rather than an argmax.
* It emits dataset, artifact, and evaluation provenance manifests, so a number
  in the report can be traced to the bytes that produced it.

Example:

    python src/training/model_eval/quality_baseline.py \
        --task jailbreak --device cuda --output-dir baseline/jailbreak
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact_inventory import (
    ServedArtifact,
    load_config,
    registry_drift,
    served_artifacts,
    uncovered_artifacts,
)
from baseline_manifests import emit_baseline_manifests
from baseline_metrics import peak_memory_mb, predict, summarise
from baseline_tasks import (
    TASK_SPECS,
    BaselineError,
    artifact_config,
    check_registry_label_order,
    load_artifact,
    load_rows,
    referenced_artifact,
    resolve_label_mapping,
    tokenizer_class,
)
from calibration import DEFAULT_BIN_COUNT, DEFAULT_THRESHOLDS
from constants import MODEL_REGISTRY
from provenance.emit import resolve_hf_revision

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAX_LENGTH = 512
DEFAULT_SEED = 42
ARTIFACT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]
SHORT_REVISION = 12

logger = logging.getLogger("QualityBaseline")


@dataclass(frozen=True)
class MeasuredArtifact:
    """The artifact this run will score, and how its identity was established."""

    model_dir: Path
    repo: str
    revision: str
    referenced: dict[str, Any] | None

    def is_served_by(self, served: ServedArtifact) -> bool:
        return self.repo == served.hf_repo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a served Router Model and emit provenance manifests"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_SPECS),
        help="Task to measure; the artifact is resolved from config/config.yaml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "config.yaml",
        help="Router configuration that declares which artifact is served",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows; recorded in the manifest as a limit",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bin-count", type=int, default=DEFAULT_BIN_COUNT, help="Calibration bins"
    )
    parser.add_argument(
        "--artifact-repo",
        default=None,
        help=(
            "Measure this Hugging Face repo instead of the configured artifact. "
            "Recorded in the result so a candidate is never mistaken for the "
            "served baseline."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help=(
            "Measure a local artifact directory instead of a published repo, so a "
            "run can be evaluated before anything is uploaded"
        ),
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=None,
        help=(
            "Reference this existing artifact manifest rather than rebuilding one; "
            "required when the artifact was emitted by a training run"
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Where manifests are written (default: <output-dir>/manifests)",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="Batches run before timing starts, to keep p99 free of warmup cost",
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Compute metrics without writing provenance manifests",
    )
    return parser.parse_args(argv)


def resolve_measured_artifact(
    args: argparse.Namespace, served: ServedArtifact
) -> MeasuredArtifact:
    """Decide which bytes this run scores, and warn when they are not the served ones."""
    referenced = referenced_artifact(args.artifact_manifest)

    if args.artifact_dir is not None:
        if referenced is None:
            raise BaselineError(
                "--artifact-dir requires --artifact-manifest so the evaluation can "
                "reference the identity the run already published"
            )
        measured = MeasuredArtifact(
            model_dir=args.artifact_dir,
            repo=referenced["identity"]["repo"],
            revision=referenced["identity"]["revision"],
            referenced=referenced,
        )
        logger.warning(
            "measuring local artifact %s, which is NOT the artifact %s serves (%s)",
            measured.model_dir,
            args.config,
            served.hf_repo,
        )
        return measured

    repo = args.artifact_repo or served.hf_repo
    if repo != served.hf_repo:
        logger.warning(
            "measuring %s, which is NOT the artifact %s serves (%s)",
            repo,
            args.config,
            served.hf_repo,
        )
    revision = resolve_hf_revision(repo)
    model_dir = Path(
        snapshot_download(repo, revision=revision, allow_patterns=ARTIFACT_PATTERNS)
    )
    return MeasuredArtifact(
        model_dir=model_dir, repo=repo, revision=revision, referenced=referenced
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    inventory = served_artifacts(config)
    served = inventory.get(args.task)
    if served is None:
        raise BaselineError(
            f"{args.config} does not configure any artifact for task {args.task!r}"
        )

    spec = TASK_SPECS[args.task]
    dataset_revision = resolve_hf_revision(spec.dataset_repo, repo_type="dataset")
    measured = resolve_measured_artifact(args, served)

    mapping = (
        dict(measured.referenced["label_mapping"])
        if measured.referenced is not None
        else resolve_label_mapping(measured.model_dir, served)
    )
    logger.info(
        "artifact %s@%s label order: %s",
        measured.repo,
        measured.revision[:SHORT_REVISION],
        mapping,
    )

    findings = (
        registry_drift(inventory, MODEL_REGISTRY)
        + uncovered_artifacts(config)
        + check_registry_label_order(args.task, mapping)
    )
    for finding in findings:
        logger.warning("gap: %s", finding)

    texts, labels, split_rows = load_rows(spec, mapping, args.limit)
    summary, model_config = _measure(args, measured, served, texts, labels, mapping)
    result = _build_result(
        args=args,
        served=served,
        measured=measured,
        model_config=model_config,
        spec=spec,
        dataset_revision=dataset_revision,
        split_rows=split_rows,
        mapping=mapping,
        findings=findings,
        summary=summary,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.task}_baseline.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not args.skip_manifests:
        result["manifests"] = emit_baseline_manifests(
            args=args,
            artifact=served,
            revision=measured.revision,
            model_dir=measured.model_dir,
            measured_repo=measured.repo,
            referenced=measured.referenced,
            model_config=model_config,
            spec=spec,
            dataset_revision=dataset_revision,
            texts=texts,
            labels=labels,
            split_rows=split_rows,
            mapping=mapping,
            summary=summary,
        )

    return result


def _measure(
    args: argparse.Namespace,
    measured: MeasuredArtifact,
    served: ServedArtifact,
    texts: list[str],
    labels: np.ndarray,
    mapping: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score every row and summarise, reporting the thresholds the config sets."""
    tokenizer, model = load_artifact(measured.model_dir, mapping)
    model.to(args.device).eval()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    probabilities, latencies = predict(
        model,
        tokenizer,
        texts,
        args.device,
        args.batch_size,
        args.max_length,
        args.warmup_batches,
    )
    summary = summarise(
        probabilities,
        labels,
        texts,
        mapping,
        latencies,
        args.bin_count,
        tuple(sorted(set(DEFAULT_THRESHOLDS) | set(served.thresholds))),
        peak_memory_mb(args.device),
    )
    return summary, artifact_config(measured.model_dir, model)


def _build_result(
    *,
    args: argparse.Namespace,
    served: ServedArtifact,
    measured: MeasuredArtifact,
    model_config: dict[str, Any],
    spec: Any,
    dataset_revision: str,
    split_rows: int,
    mapping: dict[str, int],
    findings: list[str],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": args.task,
        "artifact": {
            "repo": measured.repo,
            "revision": measured.revision,
            "is_served_artifact": measured.is_served_by(served),
            "served_repo": served.hf_repo,
            "config_locations": [site.location for site in served.sites],
            "configured_thresholds": list(served.thresholds),
            "tokenizer_class": tokenizer_class(measured.model_dir),
            "max_position_embeddings": int(model_config["max_position_embeddings"]),
        },
        "dataset": {
            "repo": spec.dataset_repo,
            "split": spec.split,
            "revision": dataset_revision,
            "rows_available": split_rows,
            "rows_scored": summary["metrics"]["rows"],
        },
        "label_mapping": mapping,
        "gaps": findings,
        **summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    metrics = result["metrics"]
    calibration = result["calibration"]
    print(
        f"{result['task']}: rows={metrics['rows']} acc={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} ece={calibration['ece']:.4f} "
        f"brier={calibration['brier']:.4f}"
    )
    for finding in result["gaps"]:
        print(f"gap: {finding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
