"""One sealed development readout for frozen, selected and calibrated heads.

``predict`` turns readout-time features into native System One predictions for
every sealed run and writes a joint seal before any gold file is opened.
``score`` then verifies the seal and runs the unchanged typed and CSS scorers.
Score answers use each arm's absolute ordinal head; a separate typed file
holds the candidate-relative Score readout as a diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import pins

READOUT_VERSION = "decision2-9b-frozen-head-readout-v1"
PANELS = ("dev", "css_pilot")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def answer(
    record: dict[str, Any], temperatures: dict[str, float], score_readout: str
) -> dict[str, Any]:
    from .train_heads import option_probabilities

    kind, keys = record["task_type"], record["keys"]
    probabilities = option_probabilities(record, temperatures, score_readout)
    if kind == "noul":
        return {"type": "noul", "noul": probabilities[keys.index("true")]}
    mapping = dict(zip(keys, probabilities))
    if kind == "score":
        return {
            "type": "score",
            "score": sum(int(key) * p for key, p in mapping.items()),
            "probabilities": mapping,
        }
    top = max(probabilities)
    return {
        "type": "choice",
        "choice": keys[probabilities.index(top)],
        "probabilities": mapping,
    }


def predictions_for(
    records, rows, items, temperatures, score_readout, identity
) -> list[dict[str, Any]]:
    from training.model.infer import prompt_input_sha256

    by_item: dict[str, dict[str, Any]] = {}
    for record, row in zip(records, rows):
        if record["id"] != row["id"]:
            raise ValueError("Readout records and feature rows are misaligned")
        entry = by_item.setdefault(
            row["item_id"], {"answers": {}, "errors": {}, "tokens": 0}
        )
        if record["valid"]:
            entry["answers"][row["question_id"]] = answer(
                record, temperatures, score_readout
            )
            entry["tokens"] += identity["tokens"](row)
        else:
            entry["answers"][row["question_id"]] = {
                "type": row["task_type"],
                "error": "max_length_exceeded",
            }
            entry["errors"][row["question_id"]] = "max_length_exceeded"
    output = []
    for item in items:
        entry = by_item[item["id"]]
        if set(entry["answers"]) != set(item["questions"]):
            raise ValueError(f"{item['id']}: question coverage differs")
        errors = entry["errors"]
        output.append(
            {
                "id": item["id"],
                "answers": {qid: entry["answers"][qid] for qid in item["questions"]},
                "usage": {"input_tokens": entry["tokens"], "output_tokens": 0},
                "adapter_status": (
                    "ok"
                    if not errors
                    else (
                        "invalid"
                        if len(errors) == len(item["questions"])
                        else "partial"
                    )
                ),
                "adapter_errors": errors,
                "truncated_questions": 0,
                "input_sha256": prompt_input_sha256(item),
                "source_input_sha256": prompt_input_sha256(item),
                "model_sha256": identity["model_sha256"],
                "adapter_sha256": identity["adapter_sha256"],
                "calibration_sha256": identity["calibration_sha256"],
            }
        )
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    return pins.file_sha256(path)


def predict(args) -> None:
    import torch

    from training.model.infer import load_prompts

    from .features import FeatureSet, read_rows
    from .train_heads import collect, load_head

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    items = {}
    for spec in args.prompts:
        name, path = spec.split("=", 1)
        pins.verify_data(name, path)
        items[name] = load_prompts(Path(path))
    if set(items) != set(PANELS):
        raise SystemExit("Readout needs exactly the dev and css_pilot prompt files")
    code = {
        name: pins.file_sha256(Path(__file__).with_name(name))
        for name in (
            "readout.py",
            "train_heads.py",
            "arms.py",
            "heads.py",
            "features.py",
            "render.py",
            "extract.py",
        )
    }
    adapter_sha = canonical_sha(code)
    seal: dict[str, Any] = {
        "version": READOUT_VERSION,
        "code_commit": args.code_commit,
        "adapter_files_sha256": code,
        "runs": [],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    features_by_source = {}
    for folder in args.features:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        features_by_source[manifest["source"]] = (folder, manifest)
    for run in args.run:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        best = json.loads((run / "BEST.json").read_text(encoding="utf-8"))
        calibration = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
        if (
            pins.file_sha256(run / "head.safetensors") != best["head_sha256"]
            or calibration["head_sha256"] != best["head_sha256"]
        ):
            raise SystemExit(f"{run}: head or calibration identity mismatch")
        folder, manifest = features_by_source[config["source"]]
        model = load_head(run / "head.safetensors", config["arm"], device)
        representation = config["arm_spec"]["representation"]
        tag = f"{config['source']}__{config['arm']}__L{config['layer']}__s{config['seed']}"
        model_sha = canonical_sha(
            {
                "source_files": manifest["source_files_sha256"],
                "head": best["head_sha256"],
                "arm": config["arm"],
                "layer": config["layer"],
            }
        )
        identity = {
            "model_sha256": model_sha,
            "adapter_sha256": adapter_sha,
            "calibration_sha256": best["calibration_sha256"],
            "tokens": (
                (lambda row: row["j_tokens"])
                if representation == "joint"
                else (
                    lambda row: row["d_state_tokens"] + sum(row["d_candidate_tokens"])
                )
            ),
        }
        out = args.output / tag
        out.mkdir()
        entry = {
            "tag": tag,
            "run": str(run.name),
            "run_path": str(run),
            "config": config,
            "head_sha256": best["head_sha256"],
            "calibration_sha256": best["calibration_sha256"],
            "model_sha256": model_sha,
            "files": {},
        }
        for panel in PANELS:
            features = FeatureSet(
                folder / panel, config["layer"], representation, device
            )
            rows = read_rows(folder / panel)
            records = collect(model, features)
            temperatures = calibration["temperature_by_readout"]
            readouts = [("absolute", f"{panel}.predictions.jsonl")]
            if panel == "dev":
                readouts.append(("relative", "dev.score-relative.predictions.jsonl"))
            for score_readout, filename in readouts:
                predictions = predictions_for(
                    records, rows, items[panel], temperatures, score_readout, identity
                )
                entry["files"][filename] = write_jsonl(out / filename, predictions)
            with (out / f"{panel}.logits.jsonl").open("x", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
            entry["files"][f"{panel}.logits.jsonl"] = pins.file_sha256(
                out / f"{panel}.logits.jsonl"
            )
        seal["runs"].append(entry)
    seal["features_manifests"] = {
        source: pins.file_sha256(folder / "manifest.json")
        for source, (folder, _) in features_by_source.items()
    }
    seal["status"] = "gold-free predictions sealed before any development key was read"
    path = args.output / "SEAL.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(seal, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "seal": str(path),
                "sha256": pins.file_sha256(path),
                "runs": len(seal["runs"]),
            }
        ),
        flush=True,
    )


def proxy(typed: dict[str, Any], css: dict[str, Any]) -> dict[str, float]:
    t = typed["macro_family_accuracy"]
    h = css["roles"]["pilot"]["median_task_macro_f1_all"]
    return {"T": t, "H": h, "proxy": 100 * math.sqrt(t * h)}


def score(args) -> None:
    seal_path = args.readout / "SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for scorer, digest in pins.SCORERS.items():
        if pins.file_sha256(Path(scorer)) != digest:
            raise SystemExit(f"{scorer} differs from the unchanged development scorer")
    for name, path in (("dev", args.dev_gold), ("css_pilot", args.css_gold)):
        pins.verify_data(name, path, pins.GOLD)
    summary = {"seal_sha256": pins.file_sha256(seal_path), "runs": {}}
    for entry in seal["runs"]:
        out = args.readout / entry["tag"]
        for filename, digest in entry["files"].items():
            if pins.file_sha256(out / filename) != digest:
                raise SystemExit(
                    f"{entry['tag']}/{filename}: sealed prediction changed"
                )
        reports = {}
        for filename, gold, module, report in (
            (
                "dev.predictions.jsonl",
                args.dev_gold,
                "benchmark.score",
                "dev.score.json",
            ),
            (
                "dev.score-relative.predictions.jsonl",
                args.dev_gold,
                "benchmark.score",
                "dev.score-relative.score.json",
            ),
            (
                "css_pilot.predictions.jsonl",
                args.css_gold,
                "transfer.score",
                "css_pilot.score.json",
            ),
        ):
            command = [
                sys.executable,
                "-m",
                module,
                "--gold",
                str(gold),
                "--predictions",
                str(out / filename),
                "--output",
                str(out / report),
            ]
            if module == "benchmark.score":
                command += [
                    "--model-id",
                    entry["tag"],
                    "--model-revision",
                    entry["head_sha256"][:12],
                    "--backend",
                    "decision2-9b-frozen-head",
                ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            reports[report] = json.loads((out / report).read_text(encoding="utf-8"))
        typed, typed_relative, css = (
            reports["dev.score.json"],
            reports["dev.score-relative.score.json"],
            reports["css_pilot.score.json"],
        )
        summary["runs"][entry["tag"]] = {
            **proxy(typed, css),
            "typed_by_type": {
                k: typed["by_type"][k]["correct_n"] for k in ("choice", "noul", "score")
            },
            "typed_by_family": {
                k: v.get("accuracy_all") for k, v in typed["by_family"].items()
            },
            "typed_brier": typed["overall"]["brier"],
            "typed_ece_10": typed["overall"]["ece_10"],
            "score_absolute": {
                k: typed["by_type"]["score"][k]
                for k in ("correct_n", "brier", "ece_10", "score_mae")
            },
            "score_relative": {
                k: typed_relative["by_type"]["score"][k]
                for k in ("correct_n", "brier", "ece_10", "score_mae")
            },
            "css_tasks": {k: v["macro_f1_all"] for k, v in css["tasks"].items()},
            "css_invalid": sum(
                v["invalid_or_missing_n"] for v in css["tasks"].values()
            ),
            "typed_invalid": typed["overall"]["invalid_or_missing_n"],
            "reports_sha256": {name: pins.file_sha256(out / name) for name in reports},
        }
    path = args.readout / "SCORES.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"scores": str(path), "sha256": pins.file_sha256(path)}), flush=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("predict")
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument(
        "--features",
        type=Path,
        action="append",
        required=True,
        help="readout-time extraction per source",
    )
    p.add_argument(
        "--prompts", action="append", required=True, help="dev=PATH and css_pilot=PATH"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--code-commit", required=True)
    s = commands.add_parser("score")
    s.add_argument("--readout", type=Path, required=True)
    s.add_argument("--dev-gold", type=Path, required=True)
    s.add_argument("--css-gold", type=Path, required=True)
    args = parser.parse_args()
    predict(args) if args.command == "predict" else score(args)


if __name__ == "__main__":
    main()
