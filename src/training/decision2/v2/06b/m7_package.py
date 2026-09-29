"""Build the M7 (a) package: a frozen causal export plus its score_bias.json (stdlib only).

    python3 -m v2.06b.m7_package --source /data/dev2/runs/06b/m1/arms/m6-mxcx-soup/full \
        --score-bias SCORE_BIAS.json --dest /data/dev2/runs/06b/m1/arms/m7a-mxcx-sb/full

SOURCE is a finished soup/run `full/` directory: `best-export/`, `best-export.MANIFEST.json`
and `COMPLETE.json` whose `best_export_manifest_sha256` is that manifest's sha256. Every
export file must match the manifest (bytes and sha256) and no unlisted file may exist. The
score bias must bind the export's `model_sha256` (training.model.infer's fingerprint).
DEST gets `best-export/` = byte-identical copies (re-hashed after copying) plus
`score_bias.json`, `best-export.MANIFEST.json` in the trainer's export format (all files,
bias included), and `COMPLETE.json` whose `best_export_manifest_sha256` is the new
manifest's sha256, so m6_formal/m7_formal identity checks and revision=auto work. The copy's
model fingerprint must equal the source's (the bias file is not part of it).
M7 prereg section 1.6.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from training.model.infer import checkpoint_fingerprint
from training.model.score_bias import validate_score_bias

from .common import file_sha256, write_json

EXPORT_SCHEMA = "dev2-06b-causal-files/1"
BIAS_NAME = "score_bias.json"


def verified_files(export: Path, manifest: dict[str, Any]) -> dict[str, dict]:
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("source export manifest has an unknown schema")
    listed = manifest["files"]
    present = {
        item.relative_to(export).as_posix()
        for item in export.rglob("*")
        if item.is_file()
    }
    if present != set(listed):
        raise ValueError(
            f"source export files differ from its manifest: {sorted(present ^ set(listed))}"
        )
    if BIAS_NAME in listed:
        raise ValueError("source export already has a score bias")
    for name, entry in listed.items():
        path = export / name
        if (
            path.stat().st_size != entry["bytes"]
            or file_sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"{name}: source bytes differ from the export manifest")
    return listed


def build(source: Path, score_bias: Path, dest: Path) -> dict[str, Any]:
    if dest.exists():
        raise FileExistsError(dest)
    export = source / "best-export"
    manifest_path = source / "best-export.MANIFEST.json"
    complete = json.loads((source / "COMPLETE.json").read_text(encoding="utf-8"))
    manifest_sha = file_sha256(manifest_path)
    if complete.get("best_export_manifest_sha256") != manifest_sha:
        raise ValueError("source COMPLETE.json does not bind its export manifest")
    listed = verified_files(export, json.loads(manifest_path.read_text()))
    model = checkpoint_fingerprint(export)
    bias = json.loads(score_bias.read_text(encoding="utf-8"))
    validate_score_bias(bias, model["model_sha256"])

    pending = dest.with_name(dest.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"interrupted package build exists: {pending}")
    target = pending / "best-export"
    target.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    for name, entry in sorted(listed.items()):
        copy = target / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(export / name, copy)
        if (
            file_sha256(copy) != entry["sha256"]
            or copy.stat().st_size != entry["bytes"]
        ):
            raise ValueError(f"{name}: copy differs from the source")
        files[name] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
    shutil.copyfile(score_bias, target / BIAS_NAME)
    bias_sha = file_sha256(target / BIAS_NAME)
    if bias_sha != file_sha256(score_bias):
        raise ValueError("score bias copy differs")
    files[BIAS_NAME] = {
        "bytes": (target / BIAS_NAME).stat().st_size,
        "sha256": bias_sha,
    }
    if checkpoint_fingerprint(target) != model:
        raise ValueError("package model fingerprint differs from the source export")
    package_manifest = write_json(
        pending / "best-export.MANIFEST.json",
        {"schema": EXPORT_SCHEMA, "files": files},
        exclusive=True,
    )
    record = {
        "status": "COMPLETE",
        "kind": "m7a-score-bias-package",
        "prereg": "v2/06b/records/m7-prereg-2026-09-29.md section 1.6",
        "best_export_manifest_sha256": package_manifest,
        "model_sha256": model["model_sha256"],
        "score_bias_sha256": bias_sha,
        "score_bias_offsets": bias["offsets"],
        "source": {
            "dir": str(source),
            "complete_sha256": file_sha256(source / "COMPLETE.json"),
            "best_export_manifest_sha256": manifest_sha,
            "state_sha256": complete.get("state_sha256"),
        },
        "files_verified": len(listed),
    }
    write_json(pending / "COMPLETE.json", record, exclusive=True)
    os.replace(pending, dest)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--score-bias", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build(args.source, args.score_bias, args.dest)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "best_export_manifest_sha256",
                    "model_sha256",
                    "score_bias_sha256",
                    "files_verified",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
