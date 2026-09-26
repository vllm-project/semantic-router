"""Rerank every durable LoRA checkpoint with the released Eikos serving path.

This uses SELECT only, before hard CAL or external DEV/CSS diagnostics. The
source Eikos weights are measured by the same released `serve.Decider` path.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from inference.eikos import REVISION, verify_release

from training.eikos.calibrate import probabilities
from training.eikos.data import native_question
from training.eikos.io import atomic_json, atomic_jsonl
from training.eikos.native import load_decider, selected_checkpoint
from training.model.data import file_sha256, load_partition


def evaluate(native, rows: list[dict], *, tag: str, output: Path) -> dict:
    import decision_core

    records = []
    by_family = defaultdict(list)
    for index, row in enumerate(rows):
        question, gold = native_question(row)
        keys = [key for key, _ in decision_core.options_of(question)]
        if row["task_type"] == "noul":
            gold = {"true": "yes", "false": "no"}[gold]
        answer, tokens = native.decide(row["state"], question)
        p = probabilities(answer, keys)
        label = keys.index(gold)
        prediction = max(range(len(p)), key=p.__getitem__)
        result = {
            "id": row["id"],
            "family": row["family"],
            "input_sha256": row["input_sha256"],
            "gold_key": gold,
            "prediction_key": keys[prediction],
            "probabilities": dict(zip(keys, p)),
            "correct": prediction == label,
            "brier": sum((value - float(i == label)) ** 2 for i, value in enumerate(p))
            / 2,
            "input_tokens": tokens,
        }
        records.append(result)
        by_family[row["family"]].append(result)
        if (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {"tag": tag, "select_scored": index + 1, "total": len(rows)}
                ),
                flush=True,
            )
    families = {
        key: {
            "n": len(subset),
            "accuracy": sum(item["correct"] for item in subset) / len(subset),
            "brier": sum(item["brier"] for item in subset) / len(subset),
        }
        for key, subset in sorted(by_family.items())
    }
    summary = {
        "tag": tag,
        "n": len(rows),
        "correct": sum(row["correct"] for row in records),
        "micro_accuracy": sum(row["correct"] for row in records) / len(rows),
        "family_macro_accuracy": sum(v["accuracy"] for v in families.values())
        / len(families),
        "family_macro_brier": sum(v["brier"] for v in families.values())
        / len(families),
        "by_family": families,
        "readout": "released Eikos serve.Decider; prompt letter-v1-semif; T=1",
    }
    atomic_jsonl(output / f"{tag}-predictions.jsonl", records)
    atomic_json(output / f"{tag}-metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    if (
        provenance["model_revision"] != REVISION
        or verify_release(args.model_path, REVISION) != provenance["source_release"]
    ):
        raise ValueError("Native SELECT source release differs from training source")
    if file_sha256(args.select) != provenance["data_sha256"]["select"]:
        raise ValueError("Native SELECT partition differs from training SELECT")
    if (run / "NATIVE_BEST.json").exists():
        raise FileExistsError(run / "NATIVE_BEST.json")
    if json.loads((run / "COMPLETE.json").read_text())["status"] != "complete":
        raise ValueError("LoRA run is incomplete")
    rows = load_partition(args.select, "select")
    if len(rows) != provenance["select_examples"]:
        raise ValueError("Native SELECT count differs from training SELECT")
    if max(len(row["options"]) for row in rows) > 100:
        raise ValueError("Native SELECT one-pass limit exceeded")
    checkpoint_names = sorted(
        path.name
        for path in run.glob("checkpoint-*")
        if path.is_dir() and not path.name.endswith(".pending")
    )
    if not checkpoint_names:
        raise ValueError("No durable LoRA checkpoints to compare")
    calibration = args.model_path / "calib.json"
    metrics = {}
    adapter_sha256_by_checkpoint = {}
    native = load_decider(args.model_path, None, calibration, device=args.device)
    metrics["source"] = evaluate(native, rows, tag="native-source", output=run)
    del native
    for name in checkpoint_names:
        selected = selected_checkpoint(run, args.model_path, checkpoint=name)
        adapter_sha256_by_checkpoint[name] = selected["adapter_weights_sha256"]
        native = load_decider(
            args.model_path, selected["adapter"], calibration, device=args.device
        )
        metrics[name] = evaluate(native, rows, tag=f"native-{name}", output=run)
        del native
    winner = max(
        metrics,
        key=lambda name: (
            metrics[name]["family_macro_accuracy"],
            -metrics[name]["family_macro_brier"],
            0 if name == "source" else -int(name.split("-")[-1]),
        ),
    )
    atomic_json(
        run / "NATIVE_BEST.json",
        {
            "checkpoint": winner,
            "source_model_revision": REVISION,
            "select_sha256": file_sha256(args.select),
            "selection_policy": "released serve.Decider family-macro accuracy, then Brier, then source/earliest",
            "metrics": metrics,
            "adapter_sha256_by_checkpoint": adapter_sha256_by_checkpoint,
            "trainer_best": json.loads((run / "BEST.json").read_text())["checkpoint"],
        },
    )
    print(
        json.dumps(
            {
                "native_best": winner,
                "source_correct": metrics["source"]["correct"],
                "winner_correct": metrics[winner]["correct"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
