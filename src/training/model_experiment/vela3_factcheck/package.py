"""Reassemble a local Vela package with one replacement adapter."""

import argparse
import hashlib
import shutil
from pathlib import Path

ADAPTER = "factcheck_adapter.safetensors"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def export_variant(base, files, adapter, out):
    out.mkdir(parents=True)
    for name in files:
        shutil.copyfile(adapter if name == ADAPTER else base / name, out / name)
    sums = {name: sha256(out / name) for name in sorted(files)}
    (out / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sums.items()),
        encoding="ascii",
    )
    return sums


def release_files(base, model):
    """Check the complete official package against the runtime's built-in pins."""
    from vllm_srun.registry.tables.vela3 import MODELS

    release = next((entry for entry in MODELS if entry.repo_id == model), None)
    if release is None or ADAPTER not in release.files:
        raise ValueError(f"{model} has no pinned Vela3 FactCheck adapter")
    for name, digest in release.files.items():
        if sha256(base / name) != digest:
            raise ValueError(
                f"{name} differs from {release.repo_id}@{release.revision}"
            )
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--model", default="vllm-sr/vela3-fast")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    release = release_files(args.base, args.model)
    print(export_variant(args.base, release.files, args.adapter, args.out)[ADAPTER])


if __name__ == "__main__":
    main()
