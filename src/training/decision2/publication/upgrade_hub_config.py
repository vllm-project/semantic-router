"""Add a root Hub download query file to a verified private native package.

Only metadata changes. Source model files are hardlinked into a new directory
on the same filesystem and are never rewritten. The new manifest covers the
root ``config.json``; callers must still verify an exact Hub readback.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .bundle import sha_file
from .download_config import build_download_config
from .full_runtime_api import verify_bundle


def upgrade(source: Path, output: Path) -> dict[str, Any]:
    if source.is_symlink():
        raise ValueError("Source package must be a regular directory")
    source = source.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not source.is_dir():
        raise ValueError("Source package must be a regular directory")
    original = verify_bundle(source)
    if "config.json" in original["files_sha256"] or (source / "config.json").exists():
        raise ValueError("Source already has a root Hub config")
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError("Metadata upgrade requires an output on the same filesystem")
    old_manifest_hash = sha_file(source / "MODEL_MANIFEST.json")
    with tempfile.TemporaryDirectory(
        prefix=".decision2-hub-", dir=output.parent
    ) as tmp:
        stage = Path(tmp) / "package"
        shutil.copytree(
            source,
            stage,
            copy_function=os.link,
            ignore=shutil.ignore_patterns(".cache", ".gitattributes", "__pycache__"),
        )
        config = build_download_config(stage, original["model_id"])
        (stage / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        updated = json.loads(
            (stage / "MODEL_MANIFEST.json").read_text(encoding="utf-8")
        )
        updated["files_sha256"]["config.json"] = sha_file(stage / "config.json")
        (stage / "MODEL_MANIFEST.json").unlink()
        (stage / "MODEL_MANIFEST.json").write_text(
            json.dumps(updated, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        verified = verify_bundle(stage)
        if verified["model_sha256"] != original["model_sha256"]:
            raise ValueError("Metadata upgrade changed model identity")
        stage.rename(output)
    if sha_file(source / "MODEL_MANIFEST.json") != old_manifest_hash:
        raise ValueError("Source package manifest changed")
    return {
        "model_id": original["model_id"],
        "model_sha256": original["model_sha256"],
        "old_manifest_sha256": old_manifest_hash,
        "new_manifest_sha256": sha_file(output / "MODEL_MANIFEST.json"),
        "root_config_sha256": sha_file(output / "config.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
