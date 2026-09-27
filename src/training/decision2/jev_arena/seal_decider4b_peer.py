"""Seal the pinned Decider 4B external peer without reading benchmark labels."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from inference.run import completed_rows, file_digest, load_prompts

SCHEMA = "decision2-decider4b-v3-peer-prediction-seal/1"
REVISION = "eb5fbdfc9448473ec25e399882912863afbdb70e"
ADAPTER_SHA256 = "b49054f1aef7a35c0a65b88dd5c1f1e5e252bb96dc210c7ef9e1d83942bb45ce"
MODEL_SHA256 = "ee8ce585b3cedd93206dd149b09b4bdd683174874211f85c77a36090b90c9fdd"
MODEL_CONFIG_SHA256 = "fc83293b6f707172e20176d603ea88dd1a3be2abd3596fdb1ccbec9404626baf"
NATIVE_SHA256 = "6359e5989fe922054c99446115a25d22acf1a943f0dea097409eaa49e2ef84f1"
PANELS = {
    "typed-final": (
        1600,
        "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
    ),
    "css15": (
        6547,
        "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    ),
    "public231": (
        231,
        "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
    ),
}


def seal(*, run_root: Path, model_path: Path, source_root: Path, output: Path) -> dict:
    """Bind complete native answers to exact public model and question bytes."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    model_path = model_path.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    run_root = run_root.resolve(strict=True)
    fixed_files = {
        "adapter": (source_root / "inference/run.py", ADAPTER_SHA256),
        "model": (model_path / "model.safetensors", MODEL_SHA256),
        "config": (model_path / "decider_config.json", MODEL_CONFIG_SHA256),
        "native": (model_path / "decider/infer.py", NATIVE_SHA256),
    }
    for name, (path, expected) in fixed_files.items():
        if file_digest(path) != expected:
            raise ValueError(f"{name} changed from the preregistered peer")
    results = {}
    for panel, (expected_n, prompt_sha) in PANELS.items():
        prompt_path = run_root / "panels" / f"{panel}.prompts.jsonl"
        prediction_path = run_root / "output" / f"{panel}.predictions.jsonl"
        if file_digest(prompt_path) != prompt_sha:
            raise ValueError(f"{panel} prompt hash changed")
        prompts = load_prompts(prompt_path)
        if len(prompts) != expected_n:
            raise ValueError(f"{panel} prompt count changed")
        completed = completed_rows(
            prediction_path,
            prompts,
            "decider",
            REVISION,
            MODEL_CONFIG_SHA256,
            model_id=None,
            revision_attested=True,
        )
        if len(completed) != expected_n:
            raise ValueError(f"{panel} has missing or incomplete predictions")
        invalid = 0
        with prediction_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                invalid += sum(
                    bool(answer.get("error")) or not answer.get("type")
                    for answer in row["answers"].values()
                )
        results[panel] = {
            "items": expected_n,
            "prompts_sha256": prompt_sha,
            "predictions_sha256": file_digest(prediction_path),
            "invalid_answer_slots_observed": invalid,
        }
    result = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "post_key_same_panel": True,
        "source_sha256": file_digest(Path(__file__)),
        "model_revision": REVISION,
        "model_files_sha256": {
            name: expected for name, (_, expected) in fixed_files.items()
        },
        "panels": results,
        "gold_or_scores_read": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_root", "model_path", "source_root", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = seal(
        run_root=args.run_root,
        model_path=args.model_path,
        source_root=args.source_root,
        output=args.output,
    )
    print(json.dumps({"schema": result["schema"], "sha256": file_digest(args.output)}))


if __name__ == "__main__":
    main()
