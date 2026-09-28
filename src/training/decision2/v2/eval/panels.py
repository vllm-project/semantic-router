"""Frozen panel registry: exact files, hashes and sizes for every eval panel.

Formal panels (post-key same-panel): JevArena v3 typed FINAL + CSS15 and the
JevBench public 231 subset. Development panels (never release scores): typed
DEV, the three-task CSS pilot and the private-dataset SELECT/CAL partitions.

Layout under a panel root (default ``/data/dev2/private/panels``)::

    goldfree/<panel>.prompts.jsonl        mounted read-only into inference
    gold/<panel>.gold.jsonl               read only by sealed scoring
    gold/public231/{manifest,prompts,targets}   JevBench scorer panel directory
    gold/{select,cal}.jsonl               labelled training-format rows
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/data/dev2/private/panels")

FORMAL: dict[str, dict[str, Any]] = {
    "typed-final": {
        "prompts": "goldfree/typed-final.prompts.jsonl",
        "prompts_sha256": "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
        "gold": "gold/typed-final.gold.jsonl",
        "gold_sha256": "707dd28dfbab10d124d437434023729f319501542e999e536fce9b7ff7f2361e",
        "originals": 1600,
        "answer_slots": 2000,
    },
    "css15": {
        "prompts": "goldfree/css15.prompts.jsonl",
        "prompts_sha256": "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
        "gold": "gold/css15.gold.jsonl",
        "gold_sha256": "1cda9623032138bb7b124be0c1b0a4239c06bed7be3169264e6eb31805c19ba4",
        "originals": 6547,
        "answer_slots": 6547,
    },
    "public231": {
        "prompts": "goldfree/public231.prompts.jsonl",
        "prompts_sha256": "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
        "gold": "gold/public231/targets.jsonl",
        "gold_sha256": "abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f",
        "panel_dir": "gold/public231",
        "panel_prompts": "gold/public231/prompts.jsonl",
        "manifest": "gold/public231/manifest.json",
        "manifest_sha256": "e0e7c67701cf05f996d3b4eac09abf1bb3e6bb363cfe007089acd3644a38ed35",
        "originals": 231,
        "answer_slots": 231,
    },
}

DEVELOPMENT: dict[str, dict[str, Any]] = {
    "typed-dev": {
        "prompts": "goldfree/typed-dev.prompts.jsonl",
        "prompts_sha256": "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
        "gold": "gold/typed-dev.gold.jsonl",
        "gold_sha256": "c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc",
        "originals": 1600,
    },
    "css-pilot": {
        "prompts": "goldfree/css-pilot.prompts.jsonl",
        "prompts_sha256": "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
        "gold": "gold/css-pilot.gold.jsonl",
        "gold_sha256": "9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391",
        "originals": 1430,
    },
    "select": {
        "gold": "gold/select.jsonl",
        "gold_sha256": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
        "originals": 700,
    },
    "cal": {
        "gold": "gold/cal.jsonl",
        "gold_sha256": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
        "originals": 700,
    },
}

ALL = {**FORMAL, **DEVELOPMENT}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_files(names: list[str] | None = None) -> dict[str, str]:
    """Relative path -> SHA-256 for the selected panels (all when None)."""
    files: dict[str, str] = {}
    for name, spec in ALL.items():
        if names is not None and name not in names:
            continue
        for kind in ("prompts", "gold", "manifest"):
            if kind in spec:
                files[spec[kind]] = spec[f"{kind}_sha256"]
        if "panel_prompts" in spec:
            files[spec["panel_prompts"]] = spec["prompts_sha256"]
    return files


def verify(root: Path, names: list[str] | None = None) -> dict[str, str]:
    observed = {}
    for relative, expected in sorted(expected_files(names).items()):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing panel file {relative}")
        digest = sha_file(path)
        if digest != expected:
            raise ValueError(
                f"{relative}: SHA-256 {digest} differs from frozen {expected}"
            )
        observed[relative] = digest
    return observed


def path(root: Path, panel: str, kind: str) -> Path:
    return root / ALL[panel][kind]


def install(root: Path, sources: dict[str, Path]) -> dict[str, str]:
    """Copy hash-verified source files into the frozen layout (never overwrite)."""
    wanted = expected_files()
    by_digest: dict[str, list[str]] = {}
    for relative, digest in wanted.items():
        by_digest.setdefault(digest, []).append(relative)
    installed = {}
    for label, source in sources.items():
        digest = sha_file(source)
        targets = by_digest.get(digest)
        if not targets:
            raise ValueError(f"{label}: {source.name} matches no frozen panel file")
        for relative in targets:
            target = root / relative
            if target.exists():
                if sha_file(target) != digest:
                    raise ValueError(f"{relative} exists with different content")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(target.parent, 0o700 if relative.startswith("gold") else 0o755)
            shutil.copyfile(source, target)
            os.chmod(target, 0o600 if relative.startswith("gold") else 0o644)
            installed[relative] = digest
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install_parser = commands.add_parser("install")
    install_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    install_parser.add_argument(
        "--source", action="append", default=[], help="LABEL=PATH of a candidate file"
    )
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    verify_parser.add_argument("--panel", action="append")
    args = parser.parse_args()
    if args.command == "install":
        sources = {}
        for entry in args.source:
            label, _, value = entry.partition("=")
            sources[label] = Path(value)
        result = install(args.root, sources)
        print(json.dumps({"installed": result}, indent=2, sort_keys=True))
    else:
        print(json.dumps(verify(args.root, args.panel), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
