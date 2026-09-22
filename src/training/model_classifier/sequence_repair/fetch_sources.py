"""Fetch only the pinned public dataset files used by the repair recipes."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "data/source-files.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = []
    for item in json.loads(args.manifest.read_text()):
        directory = args.output / item["repo"].replace("/", "--")
        path = Path(
            hf_hub_download(
                repo_id=item["repo"],
                repo_type="dataset",
                filename=item["file"],
                revision=item["revision"],
                local_dir=directory,
            )
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(
                f"Pinned dataset file hash differs: {item['repo']}/{item['file']}"
            )
        receipts.append({**item, "verified_sha256": digest})
        print(
            json.dumps({"repo": item["repo"], "file": item["file"], "verified": True}),
            flush=True,
        )
    (args.output / "verified-source-files.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
