"""Official public dataset downloads, pinned Git objects and content receipts."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from .artifacts import file_digest, read_json, write_json

SOURCES = {
    "banking77": {
        "url": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv",
        "revision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
        "filename": "banking77.csv",
        "license": "CC-BY-4.0",
        "attribution": "Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence Encoders",
        "license_url": "https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE",
    },
    "boolq": {
        "url": "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip",
        "revision": "official training release; byte identity bound by download SHA256",
        "filename": "boolq.zip",
        "sha256": "853fbe7922f70c59629f06a39e8d9ca440c3d740e760fd3b87a5ddf3dcba2436",
        "license": "CC-BY-SA-3.0",
        "attribution": "Clark et al. (2019), BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions",
        "license_url": "https://github.com/google-research-datasets/boolean-questions/blob/90af34107399cc7a446b373dc4ee35b8001da7c2/README.md#license",
    },
    "dynasent": {
        "url": "https://raw.githubusercontent.com/cgpotts/dynasent/707b6208978bfe7ff251a0c3263c9f0552c27b48/dynasent-v1.1.zip",
        "revision": "707b6208978bfe7ff251a0c3263c9f0552c27b48",
        "filename": "dynasent.zip",
        "license": "CC-BY-4.0",
        "attribution": "Potts et al. (2021), DynaSent: A Dynamic Benchmark for Sentiment Analysis",
        "license_url": "https://github.com/cgpotts/dynasent/blob/707b6208978bfe7ff251a0c3263c9f0552c27b48/README.md#license",
    },
}


def download(directory: Path, lock: Path | None = None) -> dict:
    """Reusing a download requires its receipt; --lock verifies a previous run."""
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "sources.json"
    previous = (
        read_json(lock or receipt_path) if (lock or receipt_path).exists() else None
    )
    if lock is not None and previous is None:
        raise ValueError("source lock does not exist")
    result = {"schema_version": "systemone-sources/v1", "sources": {}}
    for name, source in SOURCES.items():
        path = directory / source["filename"]
        expected = previous["sources"].get(name, {}).get("sha256") if previous else None
        expected = expected or source.get("sha256")
        if lock is not None and expected is None:
            raise ValueError(f"source lock has no checksum for {name}")
        if path.exists() and expected is None:
            raise ValueError("existing download has no integrity receipt")
        if not path.exists():
            with urllib.request.urlopen(source["url"], timeout=120) as response:
                data = response.read(128 * 1024 * 1024 + 1)
            if len(data) > 128 * 1024 * 1024:
                raise ValueError("dataset exceeds the bounded download size")
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"{name} differs from the supplied source lock")
            path.write_bytes(data)
        actual = file_digest(path)
        if expected and actual != expected:
            raise ValueError(f"{name} download checksum mismatch")
        result["sources"][name] = {**source, "sha256": actual}
        # A receipt after each successful source makes interrupted downloads resumable.
        if previous:
            write_json(
                receipt_path,
                {**previous, "sources": {**previous["sources"], **result["sources"]}},
            )
        else:
            write_json(receipt_path, result)
    return result
