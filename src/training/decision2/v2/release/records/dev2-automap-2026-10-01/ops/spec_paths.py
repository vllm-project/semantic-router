"""Every absolute node path a release spec names (and the mirrors it pins), for relaying builder inputs.

python3 spec_paths.py SPEC.json...    -> one path per line; mirrors as "mirror <full sha>"
"""

from __future__ import annotations

import json
import re
import sys

MIRROR = re.compile(r"^/data/dev2/src/([0-9a-f]{40})-src_training_decision2/")


def walk(value, out):
    if isinstance(value, dict):
        for item in value.values():
            walk(item, out)
    elif isinstance(value, list):
        for item in value:
            walk(item, out)
    elif isinstance(value, str) and value.startswith("/data/"):
        out.add(value)


def main() -> None:
    paths: set[str] = set()
    for name in sys.argv[1:]:
        walk(json.load(open(name, encoding="utf-8")), paths)
    mirrors = sorted({m.group(1) for p in paths if (m := MIRROR.match(p))})
    for path in sorted(p for p in paths if not MIRROR.match(p)):
        print(path)
    for sha in mirrors:
        print(f"mirror {sha}")


if __name__ == "__main__":
    main()
