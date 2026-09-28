"""Collect a release run's step receipts into one RELEASE-RECEIPT.json (stdlib only)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from v2.release.layout import sha_file, write_json

STEPS = (
    "build",
    "pre-a",
    "pre-b",
    "repeat-pre",
    "card-pre",
    "parity-pre",
    "ensure",
    "upload",
    "download",
    "tree",
    "post",
    "repeat-post",
    "card-post",
    "parity-post",
    "readback",
    "collect",
    "readback-collected",
)


def summarize(
    work: Path, wall_seconds: float, device: str, image: str, source: str
) -> dict[str, Any]:
    receipts = work / "receipts"
    steps: dict[str, Any] = {}
    for step in STEPS:
        path = receipts / f"{step}.json"
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        entry = {"sha256": sha_file(path), "passed": value.get("passed", True)}
        for key in (
            "revision",
            "answers_sha256",
            "manifest_sha256",
            "loaded_parameters",
            "bit_identical_answers",
            "private",
            "files",
            "seconds",
            "load_seconds",
            "device",
        ):
            if key in value:
                entry[key] = value[key]
        if step in ("repeat-pre", "repeat-post"):
            entry["totals"] = value["totals"]
        if step.startswith("parity"):
            entry["panels"] = {
                k: {
                    x: v[x]
                    for x in (
                        "prompts",
                        "slots",
                        "category_changes",
                        "max_abs_drift",
                        "input_mismatch",
                    )
                }
                for k, v in value["panels"].items()
            }
        if step.startswith("readback"):
            entry["collection"] = value["collection"]
            entry["card_problems"] = value["card_problems"]
        steps[step] = entry
    build = json.loads((receipts / "build.json").read_text(encoding="utf-8"))
    return {
        "schema": "dev2-release-receipt/1",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": source,
        "device": device,
        "image": image,
        "wall_seconds": wall_seconds,
        "gpu_hours": 0.0 if device == "cpu" else wall_seconds / 3600,
        "package_manifest_sha256": build["manifest_sha256"],
        "parameters": build["parameters"],
        "steps": steps,
        "passed": all(entry["passed"] for entry in steps.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    receipt = summarize(
        args.work, args.wall_seconds, args.device, args.image, args.source
    )
    write_json(args.work / "RELEASE-RECEIPT.json", receipt)
    print(
        json.dumps(
            {
                "passed": receipt["passed"],
                "steps": {k: v["passed"] for k, v in receipt["steps"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
