"""Decoder M10: M6-format finalist entries for the formal library (m6-formal.sh stage / smoke / finalist).

For every point: the checkpoint's file list in ``tree_manifest`` form (``<sha256>  ./<path>``, C-sorted; the list
lives outside the checkpoint), a ``weights.json`` (soup members, or the single checkpoint) and the point's 16K typed
DEV / CSS pilot readouts for the 23:15 calibration rule. Slots follow the argument order (the M10 pick order; C0, the
formal-path parity reference, is slot 0).

usage: m10_formal_select.py [--tier 4b|2b|08b] --point NAME=CHECKPOINT,TYPED_DEV_PREDS,CSS_PILOT_PREDS [...] --output DIR
       (writes DIR/<tier>-finalists.json, DIR/<NAME>.files.sha256, DIR/<NAME>.weights.json; tier default 4b)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> str:
    files = sorted(
        ("./" + p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()),
        key=lambda s: s.encode("utf-8"),
    )
    return "".join(f"{sha256(root / name[2:])}  {name}\n" for name in files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", choices=("4b", "2b", "08b"), default="4b")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"{args.tier}-finalists.json"
    if target.exists():
        raise FileExistsError(target)
    entries = []
    for slot, spec in enumerate(args.point):
        name, paths = spec.split("=", 1)
        checkpoint, dev, css = (Path(p) for p in paths.split(","))
        for path in (checkpoint / "decision_config.json", dev, css):
            if not path.is_file():
                raise FileNotFoundError(path)
        listing = args.output / f"{name}.files.sha256"
        listing.write_text(tree_manifest(checkpoint))
        config = json.loads((checkpoint / "decision_config.json").read_text())
        weights = args.output / f"{name}.weights.json"
        weights.write_text(
            json.dumps(
                {
                    "point": name,
                    "checkpoint": str(checkpoint),
                    "soup": config.get("soup"),
                    "readout": config.get("readout", "head"),
                },
                indent=2,
            )
            + "\n"
        )
        entries.append(
            {
                "slot": slot,
                "line": name,
                "point": name,
                "checkpoint": str(checkpoint),
                "weights_json": str(weights),
                "files_sha256_list": str(listing),
                "files_sha256_list_sha256": sha256(listing),
                "typed_dev_predictions": str(dev),
                "css_pilot_predictions": str(css),
            }
        )
    target.write_text(
        json.dumps(
            {"schema": "dec-m10-formal-finalists/1", "finalists": entries}, indent=2
        )
        + "\n"
    )
    print(
        json.dumps(
            [(e["slot"], e["point"], e["files_sha256_list_sha256"]) for e in entries]
        )
    )


if __name__ == "__main__":
    main()
