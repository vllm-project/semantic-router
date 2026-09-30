"""Read-only Hub audit of candidate MoE bases and MoE opponents (27B MoE milestone).

Run on a node with the HF CLI venv (token from its default file, never printed):
``/data/dev2/tools/hf-cli/bin/python3 - < hub_audit.py > receipt.json``. It records, per
repository: the immutable revision, gating, card licence fields, licence-like files, every
file's size and LFS SHA-256, the safetensors parameter summary, and the card lines that
mention licence terms. Nothing is downloaded except the card and licence-like text files.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPOS = {
    "google/gemma-4-26B-A4B": "candidate base (pretrained)",
    "google/gemma-4-26B-A4B-it": "candidate base (instruction-tuned)",
    "Qwen/Qwen3.5-35B-A3B-Base": "candidate base (pretrained)",
    "Qwen/Qwen3.5-35B-A3B": "candidate base (post-trained)",
    "surogate/rune-26b-a4b-GGUF": "opponent reference only (third-party decider)",
    "Mapika/decider-35b-a3b": "opponent reference only (third-party decider)",
}
LICENCE_FILE = re.compile(
    r"(^|/)(LICEN[CS]E|COPYING|NOTICE|USE_POLICY|TERMS)[^/]*$", re.I
)
TERMS = re.compile(
    r"licen[cs]e|apache|terms of use|prohibited|use policy|gemma terms|restrict", re.I
)


def audit(api: HfApi, repo: str, role: str) -> dict:
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 - record the Hub's answer verbatim
        return {
            "repo": repo,
            "role": role,
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }
    card = info.card_data.to_dict() if info.card_data else {}
    files = []
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        files.append(
            {
                "path": sibling.rfilename,
                "size": sibling.size,
                "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
                "blob_id": getattr(sibling, "blob_id", None),
            }
        )
    texts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for entry in files:
            path = entry["path"]
            if path == "README.md" or LICENCE_FILE.search(path):
                local = hf_hub_download(
                    repo, path, revision=info.sha, cache_dir=tmp, local_dir=None
                )
                data = Path(local).read_bytes()
                text = data.decode("utf-8", "replace")
                texts[path] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "first_line": (
                        text.strip().splitlines()[0][:200] if text.strip() else ""
                    ),
                    "terms_lines": [
                        line.strip()[:240]
                        for line in text.splitlines()
                        if TERMS.search(line)
                    ][:40],
                }
    safetensors = getattr(info, "safetensors", None)
    return {
        "repo": repo,
        "role": role,
        "revision": info.sha,
        "last_modified": str(getattr(info, "last_modified", None)),
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "disabled": getattr(info, "disabled", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library_name": getattr(info, "library_name", None),
        "license_tags": [
            tag for tag in (info.tags or []) if tag.startswith("license:")
        ],
        "base_model_tags": [
            tag for tag in (info.tags or []) if tag.startswith("base_model:")
        ],
        "card_license": card.get("license"),
        "card_license_name": card.get("license_name"),
        "card_license_link": card.get("license_link"),
        "card_base_model": card.get("base_model"),
        "extra_gated_prompt": (card.get("extra_gated_prompt") or "")[:400] or None,
        "safetensors": (
            {"total": safetensors.total, "parameters": dict(safetensors.parameters)}
            if safetensors
            else None
        ),
        "files": files,
        "text_files": texts,
        "weight_bytes": sum(
            entry["size"] or 0
            for entry in files
            if entry["path"].endswith(".safetensors")
        ),
    }


def main() -> None:
    api = HfApi()
    out = {
        "schema": "decision2-27b-moe-hub-audit/1",
        "whoami_orgs": [org.get("name") for org in api.whoami().get("orgs", [])],
        "repos": [audit(api, repo, role) for repo, role in REPOS.items()],
    }
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
