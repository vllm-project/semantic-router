"""Pin cross-module trainer imports outside the two primary Eikos code hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.eikos.io import atomic_json
from training.model.data import file_sha256

DEPENDENCIES = (
    "inference/eikos.py",
    "inference/run.py",
    "training/eikos/data.py",
    "training/eikos/train.py",
    "training/model/data.py",
    "training/model/lora.py",
    "training/model/loss.py",
    "training/model/plan.py",
)


def hashes(source: Path) -> dict[str, str]:
    return {name: file_sha256(source / name) for name in DEPENDENCIES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    manifest_path = run / "DEPENDENCY_SHA256.json"
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    current = hashes(args.source)
    for name in ("data.py", "train.py"):
        if current[f"training/eikos/{name}"] != provenance["code_sha256"][name]:
            raise ValueError(
                f"Running Eikos source differs from trainer provenance: {name}"
            )
    if args.verify:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current != expected["source_sha256"]:
            raise ValueError("A trainer dependency changed during the Eikos pilot")
        print(json.dumps({"dependency_audit": "unchanged", "files": len(current)}))
    else:
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        atomic_json(
            manifest_path,
            {
                "source_sha256": current,
                "training_provenance_sha256": file_sha256(run / "provenance.json"),
            },
        )
        print(
            json.dumps(
                {
                    "dependency_audit": "pinned",
                    "files": len(current),
                    "manifest_sha256": file_sha256(manifest_path),
                }
            )
        )


if __name__ == "__main__":
    main()
