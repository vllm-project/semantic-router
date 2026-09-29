"""Build an M8 finalist package: the frozen m6-mxcx-soup export plus its score_bias.json (stdlib only).

    python3 -m v2.06b.m8_package --candidate s5-b05 \
        [--source /data/dev2/runs/06b/m1/arms/m6-mxcx-soup/full] \
        [--score-bias /data/dev2/runs/06b/m8/dev/s5-b05/score_bias.json] \
        [--dest /data/dev2/runs/06b/m1/arms/m8-s5-b05/full] --expect-score-bias-sha256 SHA

`m7_package` generalized to M8 (prereg section 6): SOURCE must be the frozen soup (export
manifest `3ae009ad…`, `model_sha256` `01fae750…`), the bias must be an M8 finalist's file with
the given sha256 and offsets for level count 5 only. DEST gets `best-export/` = byte-identical
copies (re-hashed after copying) plus `score_bias.json`, `best-export.MANIFEST.json` in the
trainer's export format (all files, bias included; revision = its sha256) and `COMPLETE.json`
whose `best_export_manifest_sha256` binds it, so m8_formal.sh's identity checks and
revision=auto work. The copy's model fingerprint must equal the source's.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from training.model.infer import checkpoint_fingerprint
from training.model.score_bias import load_score_bias

from .common import file_sha256, write_json

m7p = importlib.import_module("v2.06b.m7_package")
m8 = importlib.import_module("v2.06b.m8_scorebias")

ARMS = Path("/data/dev2/runs/06b/m1/arms")
DEV = Path("/data/dev2/runs/06b/m8/dev")
FINALISTS = ("s5-b05", "s5h-b05")
KIND = "m8-score-bias-package"


def build(
    candidate: str,
    source: Path,
    score_bias: Path,
    dest: Path,
    expect_bias_sha256: str,
    *,
    model_sha256: str = m8.MODEL_SHA256,
    source_manifest_sha256: str = m8.EXPORT_MANIFEST_SHA256,
) -> dict[str, Any]:
    if dest.exists():
        raise FileExistsError(dest)
    export = source / "best-export"
    manifest_path = source / "best-export.MANIFEST.json"
    complete = json.loads((source / "COMPLETE.json").read_text(encoding="utf-8"))
    manifest_sha = file_sha256(manifest_path)
    if complete.get("best_export_manifest_sha256") != manifest_sha:
        raise ValueError("source COMPLETE.json does not bind its export manifest")
    if manifest_sha != source_manifest_sha256:
        raise ValueError(
            f"source export manifest {manifest_sha} is not the frozen soup's"
        )
    listed = m7p.verified_files(export, json.loads(manifest_path.read_text()))
    model = checkpoint_fingerprint(export)
    if model["model_sha256"] != model_sha256:
        raise ValueError("source export is not the frozen model")
    bias_sha = file_sha256(score_bias)
    if bias_sha != expect_bias_sha256:
        raise ValueError(f"{score_bias}: sha256 {bias_sha} != {expect_bias_sha256}")
    offsets, bias = load_score_bias(score_bias, model_sha256)
    if set(offsets) != {m8.LEVELS}:
        raise ValueError("M8 offsets hold level count 5 only")
    if (bias.get("fit") or {}).get("candidate") != candidate:
        raise ValueError(f"{score_bias} was fitted for another candidate")

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
    shutil.copyfile(score_bias, target / m7p.BIAS_NAME)
    if file_sha256(target / m7p.BIAS_NAME) != bias_sha:
        raise ValueError("score bias copy differs")
    files[m7p.BIAS_NAME] = {
        "bytes": (target / m7p.BIAS_NAME).stat().st_size,
        "sha256": bias_sha,
    }
    if checkpoint_fingerprint(target) != model:
        raise ValueError("package model fingerprint differs from the source export")
    package_manifest = write_json(
        pending / "best-export.MANIFEST.json",
        {"schema": m7p.EXPORT_SCHEMA, "files": files},
        exclusive=True,
    )
    record = {
        "status": "COMPLETE",
        "kind": KIND,
        "candidate": candidate,
        "prereg": f"{m8.PREREG} section 6",
        "prereg_commit": m8.PREREG_COMMIT,
        "best_export_manifest_sha256": package_manifest,
        "model_sha256": model["model_sha256"],
        "model_sha256_unchanged": True,
        "score_bias_path": str(score_bias),
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
    parser.add_argument("--candidate", required=True, choices=FINALISTS)
    parser.add_argument("--source", type=Path, default=m8.SOUP_DIR)
    parser.add_argument("--score-bias", type=Path)
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--expect-score-bias-sha256", required=True)
    args = parser.parse_args(argv)
    bias = args.score_bias or DEV / args.candidate / "score_bias.json"
    dest = args.dest or ARMS / f"m8-{args.candidate}" / "full"
    record = build(
        args.candidate, args.source, bias, dest, args.expect_score_bias_sha256
    )
    print(
        json.dumps(
            {
                "dest": str(dest),
                **{
                    k: record[k]
                    for k in (
                        "best_export_manifest_sha256",
                        "model_sha256",
                        "score_bias_sha256",
                        "files_verified",
                    )
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
