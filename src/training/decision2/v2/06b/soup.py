"""Uniform weight soup of same-recipe causal 0.6B checkpoints.

A soup spec names one frozen recipe spec (model skeleton, data, converter) and
two or more hash-pinned BEST states from runs whose frozen specs equal the
recipe except for the declared fields (arm, seed and, for older runs, the head
seed). The states are averaged uniformly in float64 and cast back, SELECT and
CAL probabilities are written, and the soup is exported in the causal
checkpoint format and reloaded with the trainer's parity check, so the
development readout (`m1_arm.sh <gpu> <sha> <soup-arm> readout`) treats it
like any BEST export. A single ingredient with `"export": false` writes the
SELECT and CAL probabilities of that checkpoint alone.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .common import (
    file_sha256,
    load_rights_clean,
    native_keys,
    native_records,
    original_probabilities,
    read_jsonl,
    select_record,
    write_json,
    write_jsonl,
)

SOUP_KEYS = {"arm", "family", "recipe", "ingredients", "allowed_differences", "export"}


def load_soup_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    if set(spec) != SOUP_KEYS:
        raise ValueError(f"Soup spec keys differ: {sorted(set(spec) ^ SOUP_KEYS)}")
    if spec["family"] != "qwen-causal":
        raise ValueError("Soups are implemented for the causal Qwen family")
    if not spec["ingredients"] or (spec["export"] and len(spec["ingredients"]) < 2):
        raise ValueError("An exported soup needs at least two ingredients")
    return spec


def strip(spec: dict[str, Any], allowed: list[str]) -> dict[str, Any]:
    """The spec without the declared per-seed fields (dotted paths)."""
    out = json.loads(json.dumps(spec))
    for dotted in allowed:
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node.pop(leaf)
    return out


def uniform_average(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Average floating tensors key by key; other tensors must be identical."""
    import torch

    keys = list(states[0])
    for state in states[1:]:
        if list(state) != keys:
            raise ValueError("Ingredient state keys differ")
    out = {}
    for key in keys:
        first = states[0][key]
        for state in states[1:]:
            if state[key].shape != first.shape or state[key].dtype != first.dtype:
                raise ValueError(f"{key}: ingredient shape or dtype differs")
        if first.is_floating_point():
            total = torch.zeros(first.shape, dtype=torch.float64)
            for state in states:
                total += state[key].to(torch.float64)
            out[key] = (total / len(states)).to(first.dtype).contiguous()
        else:
            if any(not torch.equal(state[key], first) for state in states[1:]):
                raise ValueError(
                    f"{key}: non-floating tensor differs across ingredients"
                )
            out[key] = first.clone()
    return out


def probabilities(
    family: Any, rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[list[float]]]:
    from .train import evaluate

    return evaluate(family, rows, records)


def run(spec_path: Path, output: Path, device: str = "cuda:0") -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    from .train import CausalQwenFamily, load_spec

    started = time.monotonic()
    spec = load_soup_spec(spec_path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    recipe_path = Path(spec["recipe"])
    recipe = load_spec(recipe_path)
    if recipe["family"] != "qwen-causal":
        raise ValueError("Recipe is not a causal Qwen arm")
    reference = strip(recipe, spec["allowed_differences"])
    ingredients = []
    states = []
    for entry in spec["ingredients"]:
        run_dir = Path(entry["run"])
        state_path = run_dir / "best.safetensors"
        best = json.loads((run_dir / "BEST.json").read_text())
        actual = file_sha256(state_path)
        if actual != entry["state_sha256"] or actual != best["state_sha256"]:
            raise ValueError(f"{entry['arm']}: BEST state differs from its frozen hash")
        run_spec = json.loads((run_dir / "RUN.json").read_text())["spec"]
        if run_spec["arm"] != entry["arm"]:
            raise ValueError(f"{entry['arm']}: run directory holds {run_spec['arm']}")
        if strip(run_spec, spec["allowed_differences"]) != reference:
            raise ValueError(f"{entry['arm']}: not the soup's recipe")
        states.append(load_file(str(state_path)))
        ingredients.append(
            {
                "arm": entry["arm"],
                "state_sha256": actual,
                "best_step": best["step"],
                "best_select": best["metrics"]["correct"],
            }
        )
    soup = uniform_average(states) if len(states) > 1 else states[0]
    del states

    torch.use_deterministic_algorithms(True, warn_only=True)
    splits = load_rights_clean(recipe["data"]["parent"])
    records = {
        role: native_records(splits[role], recipe["data"]["converter_bundle"])
        for role in ("select", "cal")
    }
    family = CausalQwenFamily(
        recipe,
        device,
        {row["id"]: row for role in ("select", "cal") for row in splits[role]},
    )
    family.restore(soup)
    result: dict[str, Any] = {
        "soup_spec_sha256": file_sha256(spec_path),
        "recipe_spec_sha256": file_sha256(recipe_path),
        "ingredients": ingredients,
        "weights": "uniform",
    }
    for role in ("select", "cal"):
        metrics, probs = probabilities(family, splits[role], records[role])
        result[role] = {
            "metrics": metrics,
            "probabilities_sha256": write_jsonl(
                output / f"{role}.probs.jsonl",
                [
                    {"id": r["id"], "probabilities": p}
                    for r, p in zip(splits[role], probs)
                ],
            ),
        }
    if spec["export"]:
        save_file(soup, str(output / "best.safetensors"))
        result["state_sha256"] = file_sha256(output / "best.safetensors")
        manifest = family.export(
            output / "best-export",
            {"arm": spec["arm"], "soup": ingredients, "weights": "uniform"},
        )
        reference_probs = read_jsonl(output / "select.probs.jsonl")
        reloaded_native = family.reload_probabilities(
            output / "best-export", manifest, records["select"], device
        )
        reloaded = [
            original_probabilities(r, native_keys(n), p)
            for r, n, p in zip(splits["select"], records["select"], reloaded_native)
        ]
        drift = max(
            abs(a - b)
            for ref, y in zip(reference_probs, reloaded)
            for a, b in zip(ref["probabilities"], y)
        )
        changed = sum(
            select_record(r, ref["probabilities"])["chosen"]
            != select_record(r, y)["chosen"]
            for r, ref, y in zip(splits["select"], reference_probs, reloaded)
        )
        result.update(
            {
                "status": (
                    "COMPLETE"
                    if drift <= 1e-5 and changed == 0
                    else "COMPLETE_RELOAD_PARITY_FAIL"
                ),
                "best_export_manifest_sha256": manifest,
                "reload_max_abs_drift": drift,
                "reload_category_changes": changed,
            }
        )
    else:
        result["status"] = "PROBABILITIES_ONLY"
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "SOUP.json", result, exclusive=True)
    if spec["export"]:
        write_json(output / "COMPLETE.json", result, exclusive=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = run(args.spec, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("select", "cal")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
