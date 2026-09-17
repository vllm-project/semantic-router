#!/usr/bin/env python3
"""Run the E2E framework's Go units once, independently of cluster profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ci_results import actual_platform, collection_errors
from workflow_evidence import go_cases

ROOT = Path(__file__).resolve().parents[2]
MODULE = "github.com/vllm-project/semantic-router/e2e"
# These packages already have a dedicated required executor and receipt.
DEDICATED_OWNERS = {
    f"{MODULE}/pkg/soak": "soak-tools",
    f"{MODULE}/profiles/multimodal-routing": "native.image-calibration-cpu",
}


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    discovered = subprocess.check_output(
        [
            "go",
            "list",
            "-f",
            "{{if or .TestGoFiles .XTestGoFiles}}{{.ImportPath}}{{end}}",
            "./...",
        ],
        cwd=ROOT / "e2e",
        text=True,
    ).splitlines()
    packages = sorted(
        {name for name in discovered if name and name not in DEDICATED_OWNERS}
    )
    if not packages:
        raise ValueError("no E2E framework unit packages discovered")
    (output / "packages.json").write_text(
        json.dumps(
            {
                "selected": packages,
                "dedicated_owners": {
                    name: owner
                    for name, owner in DEDICATED_OWNERS.items()
                    if name in discovered
                },
            },
            indent=2,
        )
        + "\n"
    )
    for name, options in (
        ("inventory.jsonl", ["-list", "^(Test|Fuzz|Example)"]),
        ("tests.jsonl", ["-count=1"]),
    ):
        path = output / name
        with path.open("w") as stream:
            result = subprocess.run(
                ["go", "test", "-json", *options, *packages],
                cwd=ROOT / "e2e",
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            print(path.read_text())
            result.check_returncode()
    cases, expected = go_cases(
        output / "tests.jsonl", output / "inventory.jsonl", set()
    )
    evidence = {
        "runtime": "none",
        "device": "none",
        "platform": actual_platform(),
        "cases": cases,
        "expected_cases": expected,
        "artifacts": [],
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    errors = collection_errors(evidence, "test")
    if errors:
        raise ValueError("; ".join(errors))
    print(
        f"E2E framework units: {len(cases)}/{len(expected)} passed across {len(packages)} packages"
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
