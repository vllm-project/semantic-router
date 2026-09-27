"""Refresh a verified private full-checkpoint bundle's System One loader.

The model, calibration, card and scored prediction identity stay byte-for-byte
unchanged. The new loader is checked separately against its old benchmark
scoring core and must be validated on the exact downloaded revision before use.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

SCORED_INFER_FUNCTIONS = (
    "load_prompts",
    "normalized_answer",
    "prompt_input_sha256",
    "checkpoint_fingerprint",
    "run_prompts",
)
PRESERVED_API_FUNCTIONS = ("verify_bundle",)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(functions) != 1:
        raise ValueError(f"Expected exactly one {name} in {path}")
    return ast.dump(functions[0], include_attributes=False)


def verify_unchanged_core(old: Path, new: Path, names: tuple[str, ...]) -> None:
    for name in names:
        if function_ast(old, name) != function_ast(new, name):
            raise ValueError(f"Scored inference function changed: {name}")


def _manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("files_sha256"), dict):
        raise ValueError("Invalid bundle manifest")
    return value


def refresh(
    *, bundle: Path, source_infer: Path, source_api: Path, output: Path
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("Expected a regular source bundle")
    old_manifest = _manifest(bundle / "MODEL_MANIFEST.json")
    old_files = old_manifest["files_sha256"]
    actual_files = {
        path.relative_to(bundle).as_posix(): sha_file(path)
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "MODEL_MANIFEST.json"
    }
    if actual_files != old_files or any(
        path.is_symlink() for path in bundle.rglob("*")
    ):
        raise ValueError("Source bundle differs from its manifest")

    changes = {
        "decision2/infer.py": source_infer,
        "decision2/api.py": source_api,
    }
    if not set(changes) <= set(old_files):
        raise ValueError("Source bundle is missing loader files")
    verify_unchanged_core(
        bundle / "decision2/infer.py", source_infer, SCORED_INFER_FUNCTIONS
    )
    verify_unchanged_core(
        bundle / "decision2/api.py", source_api, PRESERVED_API_FUNCTIONS
    )

    shutil.copytree(bundle, output, copy_function=os.link)
    try:
        for name, source in changes.items():
            target = output / name
            target.unlink()  # Never modify the old bundle through a hard link.
            shutil.copy2(source, target)
        (output / "MODEL_MANIFEST.json").unlink()
        files = dict(old_files)
        files.update({name: sha_file(output / name) for name in changes})
        manifest = dict(old_manifest)
        manifest["files_sha256"] = files
        manifest["loader_files_sha256"] = {
            name.removeprefix("decision2/"): digest
            for name, digest in files.items()
            if name.startswith("decision2/")
        }
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
            raise ValueError("Model, calibration or scored prediction identity changed")
        return {
            "old_manifest_sha256": sha_file(bundle / "MODEL_MANIFEST.json"),
            "new_manifest_sha256": sha_file(output / "MODEL_MANIFEST.json"),
            "old_model_sha256": old_manifest["model_sha256"],
            "new_model_sha256": manifest["model_sha256"],
            "changed_files": sorted([*changes, "MODEL_MANIFEST.json"]),
        }
    except BaseException:
        shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-infer", type=Path, required=True)
    parser.add_argument("--source-api", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            refresh(
                bundle=args.bundle,
                source_infer=args.source_infer,
                source_api=args.source_api,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
