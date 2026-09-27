"""Prepare and audit a small, gold-free cross-process baseline smoke panel.

The fixed source digests refer to the existing 32-item length-stratified
Choice preflight and the open typed DEV prompts. No answer files are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from inference.run import digest

CSS32_SHA = "06b5b13b8a51452ca7f8bae588bb43c7faa6b21d9498d262440041f32c541312"
TYPED_DEV_SHA = "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a"
SCHEMA = "decision2-v3-baseline-repeat-smoke/1"
PROMPTS_SHA = "3376ed4093c7efb591519912b88605960923cf33fd19d8adfa5018eb02270a55"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    result = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if any(
        not isinstance(row, dict)
        or set(row) != {"id", "state", "questions"}
        or not isinstance(row["id"], str)
        or not isinstance(row["questions"], dict)
        or len(row["questions"]) != 1
        for row in result
    ):
        raise ValueError("Expected one-question gold-free prompt rows")
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("Duplicate prompt ID")
    return result


def _quantiles(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(values) < count:
        raise ValueError("Insufficient source prompts")
    return [values[(i * (len(values) - 1)) // (count - 1)] for i in range(count)]


def _exclusive(path: Path, content: bytes) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("Output must be an absent absolute private file")
    if path.parent.stat().st_mode & 0o077:
        raise ValueError("Output directory is not private")
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def prepared_panel(css32: Path, typed_dev: Path) -> tuple[bytes, dict[str, Any]]:
    """Rebuild the fixed gold-free panel without writing or reading labels."""
    if sha(css32) != CSS32_SHA or sha(typed_dev) != TYPED_DEV_SHA:
        raise ValueError("Gold-free source digest changed")
    css, typed = rows(css32), rows(typed_dev)
    if len(css) != 32 or len(typed) != 1600:
        raise ValueError("Source panel length changed")
    by_type: dict[str, list[dict[str, Any]]] = {"noul": [], "score": []}
    for row in typed:
        kind = next(iter(row["questions"].values()))["type"]
        if kind in by_type:
            by_type[kind].append(row)
    # Reuse the prior token-stratified Choice set, then cover both native typed
    # families from the open DEV panel without reading its labels.
    selected = _quantiles(
        sorted(css, key=lambda row: (len(str(row["state"])), row["id"])), 12
    )
    for kind in ("noul", "score"):
        selected += _quantiles(
            sorted(
                by_type[kind],
                key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
            ),
            10,
        )
    if len(selected) != 32 or len({row["id"] for row in selected}) != 32:
        raise ValueError("Smoke panel has duplicate or missing prompts")
    payload = b"".join(
        (
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        for row in selected
    )
    identity = {
        "schema_version": SCHEMA,
        "source_sha256": {"choice32": CSS32_SHA, "typed_dev": TYPED_DEV_SHA},
        "prompts_sha256": hashlib.sha256(payload).hexdigest(),
        "ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in selected).encode()
        ).hexdigest(),
        "items": 32,
        "types": {
            kind: sum(
                next(iter(row["questions"].values()))["type"] == kind
                for row in selected
            )
            for kind in ("choice", "noul", "score")
        },
        "scope": "gold-free runtime repeatability only; no score or selection",
    }
    return payload, identity


def prepare(
    css32: Path, typed_dev: Path, output: Path, manifest: Path
) -> dict[str, Any]:
    if output == manifest or output.exists() or manifest.exists():
        raise ValueError("Outputs must be distinct and absent")
    payload, identity = prepared_panel(css32, typed_dev)
    _exclusive(output, payload)
    try:
        _exclusive(
            manifest, (json.dumps(identity, sort_keys=True, indent=2) + "\n").encode()
        )
    except BaseException:
        output.unlink()
        raise
    return identity


def _category(answer: dict[str, Any]) -> Any:
    kind = answer["type"]
    if kind == "choice":
        return answer["choice"]
    if kind in {"noul", "boolean"}:
        return bool(answer.get("value", float(answer["noul"]) >= 0.5))
    if kind == "score":
        probabilities = answer["probabilities"]
        return max(probabilities, key=probabilities.get)
    raise ValueError("Unsupported answer type")


def _probabilities(answer: dict[str, Any]) -> dict[str, float]:
    if answer["type"] in {"noul", "boolean"}:
        return {"yes": float(answer.get("probability", answer["noul"]))}
    return {str(key): float(value) for key, value in answer["probabilities"].items()}


def compare(
    prompts: Path,
    first: Path,
    second: Path,
    *,
    model_id: str,
    revision: str,
    backend: str,
    adapter_version: str,
    max_probability_drift: float = 1e-6,
) -> dict[str, Any]:
    inputs = rows(prompts)
    if len(inputs) != 32:
        raise ValueError("Expected exactly 32 smoke prompts")
    outputs = [rows_predictions(path) for path in (first, second)]
    if any(len(records) != 32 for records in outputs):
        raise ValueError("Incomplete smoke predictions")
    drifts: list[float] = []
    changed: list[str] = []
    for index, prompt in enumerate(inputs):
        pair = [records[index] for records in outputs]
        expected_sha = digest(
            {"state": prompt["state"], "questions": prompt["questions"]}
        )
        if any(
            row.get("id") != prompt["id"]
            or row.get("source_input_sha256") != expected_sha
            or row.get("model_id") != model_id
            or row.get("model_revision") != revision
            or row.get("backend") != backend
            or row.get("adapter_version") != adapter_version
            or row.get("revision_attested") is not True
            or set(row.get("answers", {})) != set(prompt["questions"])
            for row in pair
        ):
            raise ValueError("Prediction identity or prompt binding differs")
        if pair[0].get("usage", {}).get("input_tokens") != pair[1].get("usage", {}).get(
            "input_tokens"
        ):
            raise ValueError("Native token counts differ")
        if backend == "nox" and any(
            row.get("runtime_matches_validated") is not True for row in pair
        ):
            raise ValueError("Nox runtime differs from qualified release")
        for question in prompt["questions"]:
            left, right = pair[0]["answers"][question], pair[1]["answers"][question]
            expected_type = prompt["questions"][question]["type"]
            if left.get("type") != expected_type or right.get("type") != expected_type:
                raise ValueError("Native answer type differs")
            a, b = _probabilities(left), _probabilities(right)
            if (
                not a
                or set(a) != set(b)
                or any(not math.isfinite(value) for value in (*a.values(), *b.values()))
            ):
                raise ValueError("Invalid or mismatched probabilities")
            drifts.append(max(abs(a[key] - b[key]) for key in a))
            if _category(left) != _category(right):
                changed.append(prompt["id"])
    result = {
        "schema_version": SCHEMA,
        "scope": "gold-free runtime repeatability only; no score or selection",
        "model_id": model_id,
        "revision": revision,
        "backend": backend,
        "adapter_version": adapter_version,
        "prompts_sha256": sha(prompts),
        "predictions_sha256": {"first": sha(first), "second": sha(second)},
        "items": 32,
        "categorical_mismatch_n": len(changed),
        "categorical_mismatch_ids": changed,
        "max_option_probability_drift": max(drifts),
        "p99_option_probability_drift": sorted(drifts)[
            math.ceil(len(drifts) * 0.99) - 1
        ],
        "gate": {
            "categorical_mismatch_n": 0,
            "max_option_probability_drift_lte": max_probability_drift,
        },
        "gate_pass": not changed and max(drifts) <= max_probability_drift,
    }
    return result


def rows_predictions(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if any(not isinstance(row, dict) for row in records):
        raise ValueError("Malformed prediction row")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--choice32", type=Path, required=True)
    prep.add_argument("--typed-dev", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    audit = commands.add_parser("compare")
    audit.add_argument("--prompts", type=Path, required=True)
    audit.add_argument("--first", type=Path, required=True)
    audit.add_argument("--second", type=Path, required=True)
    audit.add_argument("--model-id", required=True)
    audit.add_argument("--revision", required=True)
    audit.add_argument("--backend", required=True)
    audit.add_argument("--adapter-version", required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.choice32, args.typed_dev, args.output, args.manifest)
    else:
        result = compare(
            args.prompts,
            args.first,
            args.second,
            model_id=args.model_id,
            revision=args.revision,
            backend=args.backend,
            adapter_version=args.adapter_version,
        )
        _exclusive(
            args.output, (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
        )
    print(
        json.dumps(
            {key: result[key] for key in ("prompts_sha256", "items")}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
