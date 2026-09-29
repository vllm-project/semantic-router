"""HT-DEV v2 training-scan manifest (prereg amendment 1 item 6).

    python3 -m v2.eval.htdev2.scans manifest --base <c1-corpora manifest.json> \
        --exclude <path prefix> ... --extra <path> ... --output <manifest.json>

Keeps every file of the base manifest except those under an excluded prefix, relabels each
file by its first four path components (so scan hits can be attributed to a tree) and adds
extra files with their sha256 and size. The output is a `c1-corpora/1` manifest for
`v2.eval.sealed.overlap scan --manifest`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from v2.eval.same_panel import sha_file, write_json

DEPTH = 4


def label_for(path: str) -> str:
    return "train:" + "/".join(Path(path).parts[1 : 1 + DEPTH])


def manifest(base: Path, exclude: list[str], extra: list[Path]) -> dict:
    data = json.loads(base.read_text(encoding="utf-8"))
    labels: dict[str, dict] = defaultdict(lambda: {"kind": "training", "files": []})
    dropped: Counter = Counter()
    seen: set[str] = set()
    for entry in data["labels"].values():
        for item in entry["files"]:
            path = item["path"]
            prefix = next((p for p in exclude if path.startswith(p)), None)
            if prefix:
                dropped[prefix] += 1
                continue
            if path in seen:
                continue
            seen.add(path)
            labels[label_for(path)]["files"].append(item)
    for path in extra:
        labels[label_for(str(path))]["files"].append(
            {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}
        )
    return {
        "schema": data["schema"],
        "base_sha256": sha_file(base),
        "excluded_prefixes": {p: dropped[p] for p in exclude},
        "extra": [str(p) for p in extra],
        "labels": dict(sorted(labels.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("manifest")
    one.add_argument("--base", type=Path, required=True)
    one.add_argument("--exclude", action="append", default=[])
    one.add_argument("--extra", type=Path, action="append", default=[])
    one.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = manifest(args.base, args.exclude, args.extra)
    write_json(args.output, result)
    files = sum(len(v["files"]) for v in result["labels"].values())
    print(
        json.dumps(
            {
                "labels": len(result["labels"]),
                "files": files,
                "excluded": result["excluded_prefixes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
