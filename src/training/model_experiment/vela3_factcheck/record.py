"""Write a Router Model provenance bundle for a local FactCheck candidate."""

import argparse
import json
from collections import Counter
from pathlib import Path

from ...model_eval.provenance.emit import (
    build_artifact_manifest,
    build_dataset_manifest,
    build_evaluation_manifest,
    build_run_manifest,
    split_digest,
    write_manifest,
)
from ...model_eval.provenance.metrics import (
    abstention_curve,
    calibration_metrics,
    classification_metrics,
    discrimination,
    operating_points,
)
from .data import NEGATIVE, POSITIVE, read_rows, suite_rows
from .package import ADAPTER, sha256

TASK = "fact-check"
LABELS = {NEGATIVE: 0, POSITIVE: 1}
SUITE = "vllm-sr/router-signal-suite"
SUITE_REVISION = "fa08b2a642df30955ad2ad2206d74050c7f12b5c"
BASE = "vllm-sr/vela3-fast"
BASE_REVISION = "7cbefd4b92b57234bd0ffb9872451e81abcdac79"
PREFIX = "src/training/model_experiment/vela3_factcheck"
ROOT = Path(__file__).resolve().parents[4]


def split(name, rows):
    counts = Counter(row["label"] for row in rows)
    return {
        "name": name,
        "rows": len(rows),
        "digest": split_digest((row["text"], LABELS[row["label"]]) for row in rows),
        "label_counts": {label: counts[label] for label in LABELS},
    }


def make_bundle(args):
    training = read_rows(args.train)
    augment = read_rows(args.augmentation)
    test_rows = suite_rows(args.suite, "test")
    report = json.loads(args.report.read_text())
    run_stats = json.loads(args.run.read_text())
    http = json.loads(args.serving.read_text())
    scores = json.loads(args.scores.read_text())["test"]
    if (
        sha256(args.model / ADAPTER) != run_stats["adapter_sha256"]
        or report["adapter_sha256"] != run_stats["adapter_sha256"]
    ):
        raise ValueError(
            "run, evaluation and artifact describe different adapter bytes"
        )
    if (
        sha256(args.train) != run_stats["train_sha256"]
        or sha256(args.augmentation) != run_stats["augmentation_sha256"]
    ):
        raise ValueError("the training input files differ from the run receipt")
    if [(row["id"], row["positive"]) for row in scores] != [
        (row["id"], row["label"] == POSITIVE) for row in test_rows
    ]:
        raise ValueError("prediction IDs or labels differ from the pinned test split")

    dataset = build_dataset_manifest(
        manifest_id="vela3-factcheck-data",
        task=TASK,
        source_type="composite",
        locator="local/factcheck-inputs",
        revision=args.input_revision,
        license_id="Per source; see the suite's sources.json and the authored augmentation receipt",
        splits=[
            split("train", training),
            split("augmentation", [r for r in augment if r.get("kind") != "pair"]),
            split("test", test_rows),
        ],
        text_field="text",
        label_field="label",
        label_mapping=LABELS,
        preprocessing_steps=[
            "Apply the reviewed training labels and exclude training rows matching held-out IDs, groups or text.",
            "Append the authored examples and apply the separately identified evidence pairs in the ranking loss.",
        ],
    )
    dataset["source"]["components"] = [
        {"type": "huggingface", "locator": SUITE, "revision": SUITE_REVISION},
        {
            "type": "in-repo",
            "locator": "local/factcheck-inputs",
            "revision": args.input_revision,
        },
    ]
    dataset_ref = {"id": dataset["id"], "revision": args.input_revision}
    run = build_run_manifest(
        manifest_id="vela3-factcheck-run",
        task=TASK,
        base_model_repo=BASE,
        base_model_revision=BASE_REVISION,
        entrypoint=f"{PREFIX}/train.py",
        repo_root=ROOT,
        dataset_refs=[{**dataset_ref, "splits": ["train", "augmentation"]}],
        seed=run_stats["seed"],
        hyperparameters={
            **run_stats,
            "lora_lr": 3e-5,
            "head_lr": 1e-4,
            "grad_accum": 4,
            "label_smoothing": 0.05,
            "logit_l2": 0.005,
            "anchor": 0.02,
            "pair_margin": 2.5,
        },
        label_mapping=LABELS,
    )
    run["code"]["repo"] = args.code_repo
    artifact = build_artifact_manifest(
        manifest_id="vela3-factcheck-candidate",
        task=TASK,
        repo="local/vela3-fast-factcheck",
        revision=args.artifact_revision,
        artifact_dir=args.model,
        label_mapping=LABELS,
        architecture="UltraBERT with Vela3 gated adapters",
        run_id=run["id"],
        max_position_embeddings=report["max_input_tokens"],
        served_paths=("/v1/decisions",),
        include_globs=("*.safetensors", "*.json", "SHA256SUMS"),
    )
    y_true = [int(row["positive"]) for row in scores]
    y_pred = [int(row["score"] >= 0.5) for row in scores]
    probs = [[1 - row["score"], row["score"]] for row in scores]
    confidence = [prob[pred] for prob, pred in zip(probs, y_pred, strict=True)]
    abstention = abstention_curve(y_true, y_pred, confidence)
    abstention["operating_points"] = operating_points(
        y_true, probs, LABELS, (POSITIVE,), (0.5, report["dev_threshold"])
    )
    evaluation = build_evaluation_manifest(
        manifest_id="vela3-factcheck-test",
        task=TASK,
        artifact_ref={
            "id": artifact["id"],
            "revision": args.artifact_revision,
            "digest": artifact["identity"]["digest"],
        },
        dataset_ref={**dataset_ref, "splits": ["test"]},
        split_rule="predefined",
        entrypoint=f"{PREFIX}/evaluate.py",
        repo_root=ROOT,
        device="cuda",
        device_name=args.device_name,
        batch_size=report["batch_size"],
        max_length=report["max_input_tokens"],
        sample_limit=None,
        seed=run_stats["seed"],
        label_mapping=LABELS,
        metrics=classification_metrics(y_true, y_pred, LABELS),
        calibration=calibration_metrics(y_true, y_pred, confidence),
        abstention=abstention,
        discrimination=discrimination(
            y_true, probs, LABELS, (POSITIVE,), fpr_budget=0.20
        ),
        performance={
            "latency_ms": http["candidate_latency"],
            "peak_memory_mb": report["peak_cuda_memory_mib"],
        },
        description="Latency comes from the separate public HTTP replay of the same artifact; quality comes from the predefined test split.",
    )
    evaluation["harness"]["code"]["repo"] = args.code_repo
    for record in (dataset, run, artifact, evaluation):
        write_manifest(record, args.out / f"{record['kind']}.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train",
        "augmentation",
        "suite",
        "run",
        "model",
        "report",
        "scores",
        "serving",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("input-revision", "artifact-revision", "device-name"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--code-repo", required=True)
    make_bundle(parser.parse_args())


if __name__ == "__main__":
    main()
