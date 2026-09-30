"""Prepare and optionally upload a compact multimodal checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_weights(checkpoint: Path) -> Path:
    for name in ("model.safetensors", "model.pt", "pytorch_model.bin"):
        path = checkpoint / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No supported weights found under {checkpoint}")


def _write_safetensors(source: Path, destination: Path) -> None:
    import torch  # noqa: PLC0415 - optional packaging dependency
    from safetensors.torch import save_file  # noqa: PLC0415 - optional dependency

    if source.suffix == ".safetensors":
        shutil.copy2(source, destination)
        return
    state_dict = torch.load(source, map_location="cpu", weights_only=True)
    contiguous = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in state_dict.items()
    }
    save_file(contiguous, str(destination))


def prepare_release(checkpoint: str, output_dir: str) -> Path:
    """Create a self-contained, checksummed release directory."""
    source = Path(checkpoint).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Release directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    config = source / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Checkpoint has no config.json: {source}")
    shutil.copy2(config, destination / "config.json")
    shutil.copy2(ROOT / "model_card_template.md", destination / "README.md")
    shutil.copytree(ROOT / "models", destination / "models")
    _write_safetensors(_find_weights(source), destination / "model.safetensors")

    files = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    (destination / "release_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def upload_release(
    release_dir: Path,
    repo_id: str,
    token: str | None,
    allow_existing: bool,
) -> None:
    from huggingface_hub import HfApi  # noqa: PLC0415 - optional upload dependency

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        if exc.__class__.__name__ != "RepositoryNotFoundError":
            raise
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=False)
    else:
        if not allow_existing:
            raise FileExistsError(
                f"Refusing to upload into existing Hugging Face model: {repo_id}"
            )
    api.upload_folder(
        folder_path=str(release_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish multi-modal-embed-small",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--repo-id",
        default="llm-semantic-router/multi-modal-embed-small",
    )
    parser.add_argument("--token")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    release_dir = prepare_release(args.checkpoint, args.output_dir)
    if args.upload:
        upload_release(
            release_dir,
            args.repo_id,
            args.token,
            args.allow_existing,
        )
    print(release_dir)


if __name__ == "__main__":
    main()
