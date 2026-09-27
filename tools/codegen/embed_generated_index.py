#!/usr/bin/env python3
"""Insert a generated endpoint index block into a Docusaurus markdown page.

The block is delimited by deterministic HTML comment markers so humans can
keep editing the surrounding narrative while `make api-docs-generate` owns
the table in between (issue: vllm-project/semantic-router#2774).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", required=True, help="markdown file to patch")
    parser.add_argument("--index", required=True, help="generated index body to embed")
    parser.add_argument("--begin", required=True, help="begin marker comment")
    parser.add_argument("--end", required=True, help="end marker comment")
    args = parser.parse_args()

    md_path = pathlib.Path(args.markdown)
    index = (
        pathlib.Path(args.index).read_text(encoding="utf-8").lstrip("\n").rstrip()
        + "\n"
    )
    text = md_path.read_text(encoding="utf-8")

    begin = _as_comment(args.begin)
    end = _as_comment(args.end)

    block = begin + "\n" + index + end
    pattern = re.compile(re.escape(begin) + ".*?" + re.escape(end), re.S)
    if pattern.search(text):
        text = pattern.sub(lambda _m: block, text, count=1)
        md_path.write_text(text, encoding="utf-8")
        print(f"updated {md_path}")
        return 0

    sys.exit(
        f"{md_path}: generated markers not found. Add '{begin}' ... '{end}' "
        "to the document before regenerating."
    )


def _as_comment(marker: str) -> str:
    marker = marker.strip()
    if marker.startswith("<!--") and marker.endswith("-->"):
        return marker
    return f"<!-- {marker.strip('-').strip()} -->"


if __name__ == "__main__":
    raise SystemExit(main())
