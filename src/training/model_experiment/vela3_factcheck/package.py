"""Reassemble a local Vela package with one replacement adapter."""

import argparse
import hashlib
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

ADAPTER = "factcheck_adapter.safetensors"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def export_variant(base, files, adapter, out):
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        staged = Path(tmp) / "package"
        staged.mkdir()
        for name in files:
            shutil.copyfile(adapter if name == ADAPTER else base / name, staged / name)
        sums = {name: sha256(staged / name) for name in sorted(files)}
        changed = [
            name
            for name, digest in files.items()
            if name != ADAPTER and digest != sums[name]
        ]
        if changed:
            raise ValueError(f"copied release files differ from their pins: {changed}")
        (staged / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sums.items()),
            encoding="ascii",
        )
        staged.rename(out)
    return sums


def adapter_layout(path):
    from safetensors import safe_open  # noqa: PLC0415 - model-backed checks only

    with safe_open(path, framework="pt") as reader:
        metadata = reader.metadata() or {}
        names = reader.keys()
        return {name: tuple(reader.get_slice(name).get_shape()) for name in names}, (
            metadata.get("lora_rank"),
            metadata.get("lora_alpha"),
            metadata.get("readout"),
        )


def release_files(base, model):
    """Check the complete official package against the runtime's built-in pins."""
    from vllm_srun.registry.tables.vela3 import MODELS  # noqa: PLC0415

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
    if adapter_layout(args.adapter) != adapter_layout(args.base / ADAPTER):
        raise ValueError(
            "candidate adapter layout differs from the released FactCheck branch"
        )
    print(export_variant(args.base, release.files, args.adapter, args.out)[ADAPTER])


if __name__ == "__main__":
    main()
