"""Check the reference's /v1/decisions examples against a running runtime.

    vllm-srun serve vllm-sr/Vela-2.0-0.3B --device cpu --port 8100
    python3 tools/reference_examples.py --url http://127.0.0.1:8100

Each ``json title="POST /v1/decisions"`` block of
``website/docs/model-runtime/reference.md`` is sent as written, and the
answer must match the ``json title="Response"`` block that follows it: every
documented field, with numbers within ``--tolerance`` of the printed value
(the page rounds them). Fields the page leaves out, such as ``usage``, are
not compared. Exits 1 and lists the differences when an example drifts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

REFERENCE = (
    Path(__file__).resolve().parents[3] / "website/docs/model-runtime/reference.md"
)
BLOCK = re.compile(r'```json title="(?P<title>[^"]+)"\n(?P<body>.*?)\n```', re.DOTALL)
REQUEST_TITLE = "POST /v1/decisions"
RESPONSE_TITLE = "Response"


def examples(text: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The page's (request, response) pairs, in page order."""
    pairs = []
    for block, following in itertools.pairwise(BLOCK.finditer(text)):
        if block["title"] == REQUEST_TITLE and following["title"] == RESPONSE_TITLE:
            pairs.append((json.loads(block["body"]), json.loads(following["body"])))
    return pairs


def differences(
    documented: Any, actual: Any, tolerance: float, path: str = "$"
) -> list[str]:
    """Where actual departs from what the page documents."""
    if isinstance(documented, bool) or isinstance(actual, bool):
        same = type(documented) is type(actual) and documented == actual
        return [] if same else [f"{path}: {actual!r} != {documented!r}"]
    if isinstance(documented, (int, float)) and isinstance(actual, (int, float)):
        if abs(documented - actual) <= tolerance:
            return []
        return [f"{path}: {actual} is not within {tolerance} of {documented}"]
    if isinstance(documented, dict) and isinstance(actual, dict):
        found = []
        for key, value in documented.items():
            if key not in actual:
                found.append(f"{path}.{key}: missing")
                continue
            found.extend(differences(value, actual[key], tolerance, f"{path}.{key}"))
        return found
    if isinstance(documented, list) and isinstance(actual, list):
        if len(documented) != len(actual):
            return [f"{path}: {len(actual)} items, the page shows {len(documented)}"]
        found = []
        for index, (left, right) in enumerate(zip(documented, actual, strict=True)):
            found.extend(differences(left, right, tolerance, f"{path}[{index}]"))
        return found
    return [] if documented == actual else [f"{path}: {actual!r} != {documented!r}"]


def post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/decisions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        answer: dict[str, Any] = json.loads(response.read())
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="the runtime's base URL")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()

    pairs = examples(args.reference.read_text(encoding="utf-8"))
    if not pairs:
        print(f"no {REQUEST_TITLE} example in {args.reference}", file=sys.stderr)
        return 1
    failed = 0
    for index, (request, documented) in enumerate(pairs, 1):
        drift = differences(documented, post(args.url, request), args.tolerance)
        print(f"example {index}: {'matches' if not drift else 'differs'}")
        for line in drift:
            print(f"  {line}")
        failed += bool(drift)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
