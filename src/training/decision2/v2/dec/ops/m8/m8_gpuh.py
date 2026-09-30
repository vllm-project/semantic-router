"""Decoder M8 GPU-hours (wall-clock x GPUs) from the node's job receipts (prereg dec-m8-prereg-2026-09-30.md, "Budget").

Sources under /data/dev2/runs/dec/m8: every GPU `*.launch.json` of launch.sh under arms/ (member jobs), lines/
GPU-SECONDS.jsonl (readouts), label/run-*/GPU-SECONDS.json (node A label shards) and gpuh-formal.json (formal
receipts). `write <node>` stores this node's parts in gpuh-node-<node>.json; `total <node>` adds the other node's file
(relayed by hand) to this node's parts.

usage: m8_gpuh.py arm <ARM> | total <a|b> | write <a|b>
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

M = Path("/data/dev2/runs/dec/m8")


def seconds(receipt: dict) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (
        dt.datetime.strptime(receipt["end_utc"], fmt)
        - dt.datetime.strptime(receipt["start_utc"], fmt)
    ).total_seconds()


def member_jobs(root: Path = M) -> dict[str, float]:
    out: dict[str, float] = {}
    for path in sorted((root / "arms").rglob("*.launch.json")):
        receipt = json.loads(path.read_text())
        if receipt.get("gpu") is None:
            continue
        out[receipt["job"]] = seconds(receipt)
    return out


def arm_hours(arm: str, root: Path = M) -> float:
    return (
        sum(s for job, s in member_jobs(root).items() if job.startswith(f"m8-{arm}-m"))
        / 3600
    )


def node_hours(root: Path = M) -> dict[str, float]:
    parts = {"members": sum(member_jobs(root).values()) / 3600}
    lines = (
        sorted((root / "lines").rglob("GPU-SECONDS.jsonl"))
        if (root / "lines").is_dir()
        else []
    )
    if lines:
        parts["lines"] = (
            sum(
                json.loads(x)["gpu_seconds"]
                for f in lines
                for x in f.read_text().splitlines()
                if x.strip()
            )
            / 3600
        )
    labels = [
        json.loads(p.read_text())
        for p in (root / "label").glob("run-*/GPU-SECONDS.json")
    ]
    if labels:
        parts["labels"] = sum(r["gpu_seconds"] for r in labels) / 3600
    formal = root / "gpuh-formal.json"
    if formal.is_file():
        parts["formal"] = json.loads(formal.read_text())["gpu_hours"]
    return parts


def total_hours(node: str, root: Path = M) -> float:
    total = sum(node_hours(root).values())
    other = root / f"gpuh-node-{'a' if node == 'b' else 'b'}.json"
    if other.is_file():
        total += json.loads(other.read_text())["node_gpu_hours"]
    return total


def main(argv: list[str]) -> int:
    if argv[:1] == ["arm"] and len(argv) == 2:
        print(f"{arm_hours(argv[1]):.4f}")
    elif len(argv) == 2 and argv[0] == "total" and argv[1] in ("a", "b"):
        print(f"{total_hours(argv[1]):.4f}")
    elif len(argv) == 2 and argv[0] == "write" and argv[1] in ("a", "b"):
        parts = node_hours()
        doc = {
            "schema": "dec-m8-gpuh/1",
            "node": argv[1],
            "parts": parts,
            "node_gpu_hours": sum(parts.values()),
            "updated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        (M / f"gpuh-node-{argv[1]}.json").write_text(
            json.dumps(doc, indent=1, sort_keys=True) + "\n"
        )
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
