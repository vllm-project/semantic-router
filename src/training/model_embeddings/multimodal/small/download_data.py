"""Materialize the public image-caption metadata into an explicit directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def download_llava_cc3m(output_dir: str, revision: str) -> Path:
    """Download a revision-pinned LLaVA-CC3M snapshot."""
    from huggingface_hub import (  # noqa: PLC0415 - optional download dependency
        snapshot_download,
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="liuhaotian/LLaVA-CC3M-Pretrain-595K",
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable dataset commit recorded for this training run",
    )
    args = parser.parse_args()
    print(download_llava_cc3m(args.output_dir, args.revision))


if __name__ == "__main__":
    main()
