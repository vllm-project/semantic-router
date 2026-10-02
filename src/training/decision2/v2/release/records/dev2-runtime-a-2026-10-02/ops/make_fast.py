"""Specs of the speed-up phase A runtime (``runtime/fast.py``) for one Decision 2.0 tier.

    python3 make_fast.py preview --base SPEC --runtime-source DIR --key KEY --output OUT

``preview``: a staging spec that builds the tier's released package with the new
runtime: the released spec (Hub IDs made current), kind ``staging`` under
``vllm-sr/dev2-release-staging-ra<key>``, no release decision or gate profile,
and ``runtime_source`` set to the new runtime's mirror. Weights, tokenizer,
vendored sources and remote code stay those of the released spec; only the
package runtime (and the staging name in config / manifest / card) differs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.release import layout


def preview(base: dict, runtime_source: str, key: str) -> dict:
    spec = layout.current_ids(base)
    for name in ("_release", "gate_receipt", "gate_profile"):
        spec.pop(name, None)
    spec["kind"] = "staging"
    spec["repo_id"] = f"{layout.ORG}/dev2-release-staging-ra{key}"
    spec["runtime_source"] = runtime_source
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preview")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--runtime-source", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    source = Path(args.runtime_source)
    if not (source / "v2/release/runtime/fast.py").is_file():
        raise SystemExit(f"{source} has no fast-path runtime")
    spec = preview(base, str(source), args.key)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(spec, sink, indent=2, ensure_ascii=False)
        sink.write("\n")
    print(json.dumps({"spec": str(args.output), "repo_id": spec["repo_id"]}))


if __name__ == "__main__":
    main()
