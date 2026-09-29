"""Trainer-path probabilities of a frozen causal 0.6B state for training-format rows.

    python -m v2.06b.m7_probs --run-dir RUN [--recipe SPEC] --pair IN.jsonl OUT.probs.jsonl ...
        [--cal-parity] --manifest PROBS.json

Runs inside the training container (image Python, like soup.py). RUN is a soup output
(`SOUP.json`, `best.safetensors`, `best-export`, `cal.probs.jsonl`; `--recipe` must be
the soup's recipe spec, checked against `recipe_spec_sha256`) or a training run (`BEST.json`,
`RUN.json`, `best.safetensors`; the recipe is RUN.json's frozen spec). The state is
restored into `CausalQwenFamily` exactly as soup.py does before it writes
`select/cal.probs.jsonl`, and `train.evaluate` (FP32, no autocast, chunks of 8) writes one
`{id, probabilities}` line per row in option order. `--cal-parity` recomputes CAL700 and
compares it with RUN/cal.probs.jsonl (max |dp| and argmax changes), which proves the path is
the one that wrote that file. M7 prereg section 1.3.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from training.model.data import load_partition

from .common import (
    file_sha256,
    load_rights_clean,
    native_records,
    read_jsonl,
    select_record,
    write_json,
    write_jsonl,
)

PARITY_TOLERANCE = 1e-5


def frozen_state(run_dir: Path, recipe_path: Path | None) -> dict[str, Any]:
    from .train import load_spec

    soup_path, best_path = run_dir / "SOUP.json", run_dir / "BEST.json"
    if soup_path.is_file():
        soup = json.loads(soup_path.read_text())
        if (
            recipe_path is None
            or file_sha256(recipe_path) != soup["recipe_spec_sha256"]
        ):
            raise ValueError("--recipe must be the soup's recipe spec")
        info = {
            "kind": "soup",
            "state_sha256": soup["state_sha256"],
            "record": {"path": str(soup_path), "sha256": file_sha256(soup_path)},
            "recipe_spec_sha256": soup["recipe_spec_sha256"],
        }
        recipe = load_spec(recipe_path)
    elif best_path.is_file():
        best = json.loads(best_path.read_text())
        recipe = json.loads((run_dir / "RUN.json").read_text())["spec"]
        info = {
            "kind": "run",
            "state_sha256": best["state_sha256"],
            "record": {"path": str(best_path), "sha256": file_sha256(best_path)},
            "run_json_sha256": file_sha256(run_dir / "RUN.json"),
        }
    else:
        raise FileNotFoundError(f"{run_dir}: neither SOUP.json nor BEST.json")
    if recipe["family"] != "qwen-causal":
        raise ValueError("m7_probs is for the causal Qwen family")
    complete = run_dir / "COMPLETE.json"
    if complete.is_file():
        info["best_export_manifest_sha256"] = json.loads(complete.read_text()).get(
            "best_export_manifest_sha256"
        )
    return {"recipe": recipe, **info}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    from .train import CausalQwenFamily, evaluate

    started = time.monotonic()
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    for _, output in args.pair:
        if Path(output).exists():
            raise FileExistsError(output)
    frozen = frozen_state(args.run_dir, args.recipe)
    recipe = frozen.pop("recipe")
    state_path = args.run_dir / "best.safetensors"
    if file_sha256(state_path) != frozen["state_sha256"]:
        raise ValueError("best.safetensors differs from its recorded state hash")
    inputs = {source: load_partition(source, "select") for source, _ in args.pair}
    rows = {row["id"]: row for part in inputs.values() for row in part}
    cal = None
    if args.cal_parity:
        cal = load_rights_clean(recipe["data"]["parent"])["cal"]
        for row in cal:
            if row["id"] in rows:
                raise ValueError(f"{row['id']}: input row is a CAL row")
            rows[row["id"]] = row

    torch.use_deterministic_algorithms(True, warn_only=True)
    family = CausalQwenFamily(recipe, args.device, rows)
    family.restore(load_file(str(state_path)))
    bundle = recipe["data"]["converter_bundle"]
    result: dict[str, Any] = {
        "schema": "dev2-06b-m7a-probs/1",
        "path": "train.evaluate: FP32, no autocast, chunks of 8 (as soup.py)",
        "run_dir": str(args.run_dir),
        **frozen,
        "outputs": [],
    }
    for source, output in args.pair:
        part = inputs[source]
        metrics, probs = evaluate(family, part, native_records(part, bundle))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        result["outputs"].append(
            {
                "input": source,
                "input_sha256": file_sha256(source),
                "rows": len(part),
                "output": output,
                "output_sha256": write_jsonl(
                    output,
                    [{"id": r["id"], "probabilities": p} for r, p in zip(part, probs)],
                ),
                "metrics": metrics,
            }
        )
    if cal is not None:
        _, probs = evaluate(family, cal, native_records(cal, bundle))
        reference_path = args.run_dir / "cal.probs.jsonl"
        reference = {r["id"]: r["probabilities"] for r in read_jsonl(reference_path)}
        drift = max(
            abs(a - b)
            for row, p in zip(cal, probs)
            for a, b in zip(reference[row["id"]], p)
        )
        changes = sum(
            select_record(row, reference[row["id"]])["chosen"]
            != select_record(row, p)["chosen"]
            for row, p in zip(cal, probs)
        )
        result["cal_parity"] = {
            "reference": str(reference_path),
            "reference_sha256": file_sha256(reference_path),
            "rows": len(cal),
            "max_abs_drift": drift,
            "argmax_changes": changes,
            "tolerance": PARITY_TOLERANCE,
            "pass": drift <= PARITY_TOLERANCE and changes == 0,
        }
    result["torch_version"] = torch.__version__
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(args.manifest, result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument(
        "--pair", nargs=2, action="append", required=True, metavar=("IN", "OUT")
    )
    parser.add_argument("--cal-parity", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = run(args)
    print(
        json.dumps(
            {
                "outputs": [(o["output"], o["rows"]) for o in result["outputs"]],
                "cal_parity": result.get("cal_parity"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
