"""Refresh only the product README in a verified private full-checkpoint bundle.

The replacement must come from the scored product renderer. Every other
product file and every model, calibration and prediction identity is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .full_bundle import CONTENT_FILES
from .refresh_full_contract import _manifest, sha_file


def refresh(*, bundle: Path, content: Path, output: Path) -> dict[str, str]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not bundle.is_dir() or bundle.is_symlink() or not content.is_dir():
        raise ValueError("Expected regular bundle and product content directories")
    old_manifest = _manifest(bundle / "MODEL_MANIFEST.json")
    old_files = old_manifest["files_sha256"]
    actual = {
        path.relative_to(bundle).as_posix(): sha_file(path)
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "MODEL_MANIFEST.json"
    }
    if actual != old_files or any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("Source bundle differs from its manifest")
    product = {
        path.relative_to(content).as_posix(): path
        for path in content.rglob("*")
        if path.is_file()
    }
    if set(product) != CONTENT_FILES or any(
        path.is_symlink() for path in content.rglob("*")
    ):
        raise ValueError("Replacement product content differs from its whitelist")
    if any(
        sha_file(path) != old_files.get(name)
        for name, path in product.items()
        if name != "README.md"
    ):
        raise ValueError("A non-card product file changed")
    original = (bundle / "README.md").read_text(encoding="utf-8")
    replacement = product["README.md"].read_text(encoding="utf-8")
    if (
        original == replacement
        or original.split("---", 2)[:2] != replacement.split("---", 2)[:2]
    ):
        raise ValueError("Card is unchanged or its license/source metadata changed")
    if any(
        required not in replacement
        for required in (
            "DEV2.0-0.6B",
            "Choice",
            "Noul",
            "Score",
            "8,147",
            "231 public JevBench",
            "assets/jevarena-rank.svg",
            "assets/jevarena-task-matrix.svg",
            "assets/jevbench-public-rank.svg",
        )
    ):
        raise ValueError("Replacement card is missing product or score elements")

    shutil.copytree(bundle, output, copy_function=os.link)
    try:
        (output / "README.md").unlink()  # Do not mutate a hard-linked source.
        shutil.copy2(product["README.md"], output / "README.md")
        (output / "MODEL_MANIFEST.json").unlink()
        manifest = dict(old_manifest)
        files = dict(old_files)
        files["README.md"] = sha_file(output / "README.md")
        manifest["files_sha256"] = files
        (output / "MODEL_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if any(
            manifest[key] != old_manifest[key]
            for key in (
                "model_sha256",
                "model_files_sha256",
                "calibration_sha256",
                "scored_predictions",
            )
        ):
            raise ValueError("A model, calibration or score identity changed")
        return {
            "old_manifest_sha256": sha_file(bundle / "MODEL_MANIFEST.json"),
            "new_manifest_sha256": sha_file(output / "MODEL_MANIFEST.json"),
            "old_card_sha256": old_files["README.md"],
            "new_card_sha256": files["README.md"],
        }
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--content", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            refresh(bundle=args.bundle, content=args.content, output=args.output),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
