"""Compare an exported full model to every sealed gold-free native answer.

This is a functional package test, not a benchmark scorer. It never opens
answer keys, keeps prompts and per-item outputs private, and counts invalids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any

PANEL_COUNTS = {"typed": (1600, 2000), "css": (6547, 6547), "public": (231, 231)}
MAX_NUMERIC_DRIFT = 1e-4


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            yield value


def _compare(expected: Any, actual: Any) -> tuple[int, float]:
    """Return category/missing changes and maximum numeric drift recursively."""
    if type(expected) in (int, float) and type(actual) in (int, float):
        if not math.isfinite(expected) or not math.isfinite(actual):
            return 1, math.inf
        return 0, abs(float(expected) - float(actual))
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = set(expected) | set(actual)
        changes = len(set(expected) ^ set(actual))
        drift = 0.0
        for key in set(expected) & set(actual):
            c, d = _compare(expected[key], actual[key])
            changes += c
            drift = max(drift, d)
        return changes, drift
    if isinstance(expected, list) and isinstance(actual, list):
        changes = abs(len(expected) - len(actual))
        drift = 0.0
        for old, new in zip(expected, actual):
            c, d = _compare(old, new)
            changes += c
            drift = max(drift, d)
        return changes, drift
    return (0 if expected == actual and type(expected) is type(actual) else 1), 0.0


def run(
    package: Path, paths: dict[str, tuple[Path, Path, Path]], output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(package.resolve()))
    from decision2 import Decision2, verify_bundle

    manifest = verify_bundle(package)
    panels = {}
    for name, (prompts, predictions, scored_manifest) in paths.items():
        scored = json.loads(scored_manifest.read_text(encoding="utf-8"))
        binding = manifest["scored_predictions"][name]
        expected_items, expected_slots = PANEL_COUNTS[name]
        if (
            _sha(prompts) != scored.get("input_sha256")
            or _sha(predictions) != scored.get("predictions_sha256")
            or _sha(scored_manifest) != binding["manifest_sha256"]
            or _sha(predictions) != binding["predictions_sha256"]
            or scored.get("model_sha256") != manifest["model_sha256"]
            or scored.get("input_items") != expected_items
            or scored.get("counts", {}).get("questions") != expected_slots
        ):
            raise ValueError(f"{name}: source panel differs from sealed native scoring")
        panels[name] = (prompts, predictions, scored_manifest)

    model = Decision2.from_pretrained(package, device="cuda:0")
    start = time.time()
    summary = {}
    for name, (prompts, predictions, _) in panels.items():
        items = slots = changes = invalid = 0
        drift = 0.0
        for source, scored in zip_longest(_jsonl(prompts), _jsonl(predictions)):
            if source is None or scored is None or source.get("id") != scored.get("id"):
                raise ValueError(f"{name}: item ordering or item count differs")
            if set(source) - {"id", "state", "questions"}:
                raise ValueError(
                    f"{name}: prompt contains non-inference top-level fields"
                )
            answer = model.system_one(
                state=source["state"], questions=source["questions"]
            )
            if set(answer["answers"]) != set(scored["answers"]):
                raise ValueError(f"{name}: answer slot identity differs")
            c, d = _compare(scored["answers"], answer["answers"])
            changes += c
            drift = max(drift, d)
            items += 1
            slots += len(answer["answers"])
            invalid += sum("error" in value for value in answer["answers"].values())
        if (items, slots) != PANEL_COUNTS[name]:
            raise ValueError(f"{name}: incomplete panel replay")
        summary[name] = {
            "items": items,
            "answer_slots": slots,
            "invalid": invalid,
            "category_or_structure_changes": changes,
            "max_numeric_drift": drift,
            "prompts_sha256": _sha(prompts),
            "scored_predictions_sha256": _sha(predictions),
            "scored_manifest_sha256": _sha(panels[name][2]),
        }
    status = (
        "pass"
        if all(
            panel["category_or_structure_changes"] == 0
            and panel["max_numeric_drift"] <= MAX_NUMERIC_DRIFT
            for panel in summary.values()
        )
        else "fail"
    )
    receipt = {
        "schema_version": "decision2-full-product-native-parity/1",
        "status": status,
        "package_manifest_sha256": _sha(package / "MODEL_MANIFEST.json"),
        "model_sha256": manifest["model_sha256"],
        "max_numeric_drift_allowed": MAX_NUMERIC_DRIFT,
        "wall_seconds": time.time() - start,
        "panels": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status != "pass":
        raise ValueError("Full package differs from the sealed native answers")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in PANEL_COUNTS:
        for field in ("prompts", "predictions", "manifest"):
            parser.add_argument(f"--{name}-{field}", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: tuple(
            getattr(args, f"{name}_{field}")
            for field in ("prompts", "predictions", "manifest")
        )
        for name in PANEL_COUNTS
    }
    receipt = run(args.package, paths, args.output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "panels": receipt["panels"],
                "wall_seconds": receipt["wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
