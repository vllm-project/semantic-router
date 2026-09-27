"""Add a root Hub download query file to a verified private native package.

Only metadata changes. Source model files are hardlinked into a new directory
on the same filesystem and are never rewritten. The new manifest covers the
root ``config.json``; callers must still verify an exact Hub readback.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .bundle import sha_file
from .download_config import build_download_config
from .full_runtime_api import _inventory


def _verify_native(root: Path) -> dict[str, Any]:
    """Check file bytes before importing the exact packaged native verifier."""
    manifest_path = root / "MODEL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _inventory(root) != {
        **manifest["files_sha256"],
        "MODEL_MANIFEST.json": sha_file(manifest_path),
    }:
        raise ValueError("Package differs from its file inventory")
    package = root / "decision2/__init__.py"
    name = "_decision2_hub_upgrade_runtime"
    spec = importlib.util.spec_from_file_location(
        name, package, submodule_search_locations=[str(package.parent)]
    )
    if spec is None or spec.loader is None:
        raise ValueError("Native verifier is missing")
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module.verify_bundle(root)
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def upgrade(source: Path, output: Path) -> dict[str, Any]:
    if source.is_symlink():
        raise ValueError("Source package must be a regular directory")
    source = source.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not source.is_dir():
        raise ValueError("Source package must be a regular directory")
    original = _verify_native(source)
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
        verified = _verify_native(stage)
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
