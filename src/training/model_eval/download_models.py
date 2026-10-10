"""Download the native artifacts used by this evaluator at its pinned revisions.

Production downloads continue to use the Router's --download-only contract.
This helper deliberately does not download or rearrange ONNX or LoRA variants.
"""

import argparse
from pathlib import Path

from .constants import COLLECTIONS, model_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", choices=list(COLLECTIONS), default="served")
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    for entry in model_registry(args.collection).values():
        snapshot_download(
            repo_id=entry["id"],
            revision=entry.get("revision"),
            local_dir=args.output / entry["id"].split("/")[-1],
            allow_patterns=["*.json", "*.safetensors", "*.model"],
            ignore_patterns=["onnx/*", "lora/*", "reproducibility/*"],
        )


if __name__ == "__main__":
    main()
