#!/usr/bin/env python3
"""Package frozen development prototypes from repository files, without network I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(repository: Path, manifest_path: Path, output: Path) -> int:
    repository = repository.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or not manifest.get("assets"):
        raise ValueError("unsupported or empty image prototype manifest")
    checked = []
    for entry in manifest["assets"]:
        source = Path(entry["source"])
        asset = entry["asset"]
        if source.is_absolute() or ".." in source.parts:
            raise ValueError("prototype source must be repository relative")
        resolved = (repository / source).resolve(strict=True)
        if not resolved.is_relative_to(repository) or not resolved.is_file():
            raise ValueError("prototype source escapes the repository")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"prototype source checksum differs: {source}")
        if asset != digest + source.suffix.lower() or Path(asset).name != asset:
            raise ValueError(
                "prototype destination must be its content-addressed filename"
            )
        if entry["role"] not in ("positive", "negative"):
            raise ValueError("invalid prototype role")
        checked.append((resolved, asset))
    # Validate every input before publishing any of the prepared assets.
    output.mkdir(parents=True, exist_ok=True)
    prior = output / "manifest.json"
    if prior.is_file():
        retained = {asset for _, asset in checked}
        for entry in json.loads(prior.read_text()).get("assets", []):
            name = entry.get("asset", "")
            if name and Path(name).name == name and name not in retained:
                obsolete = output / name
                if obsolete.is_file() and not obsolete.is_symlink():
                    obsolete.unlink()
    for source, asset in checked:
        target = output / asset
        if target.is_symlink():
            raise ValueError("prototype destination must not be a symlink")
        shutil.copyfile(source, target)
        target.chmod(0o644)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return len(checked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = (
        args.manifest
        or args.repository_root / "config/assets/image-routing/manifest.json"
    )
    print(
        f"Prepared {prepare(args.repository_root, manifest, args.output)} image prototypes"
    )


if __name__ == "__main__":
    main()
