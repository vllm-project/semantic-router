"""Milestone 6 data and soup helpers for the ~27B track (host CPU, stdlib only).

``drop-families``: the IBX ablation block (preregistration "Stage 1"): the IB TRAIN file without the rows whose
``family`` is listed (the in-distribution families ``w2c`` and ``isarc``), every other line copied byte for byte and in
order; refuses unknown families and an output that exists; prints the counts and SHA-256 values for the data lock.

``soup-check``: M5's ``lsoup`` check of a ``v2.27b.lora_soup`` output: members, rank and alpha are the members' sums,
the verification passed within its tolerance, the manifest lists the members in order, and a member relayed from
another node (``--relay-sums "CKPT=SHA256SUMS ..."``) matches its relay list file by file.

    python3 -m v2.27b.m6.m6_data drop-families --input IB.train.jsonl --drop w2c --drop isarc --output IBX.train.jsonl
    python3 -m v2.27b.m6.m6_data soup-check --manifest SOUP/soup_manifest.json [--relay-sums ...] CKPT CKPT...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def drop_families(source: Path, drop: list[str], output: Path) -> dict:
    data = source.read_bytes()
    kept, counts = [], Counter()
    for line in data.splitlines(keepends=True):
        if not line.strip():
            raise ValueError(f"{source}: blank line")
        family = json.loads(line)["family"]
        counts[family] += 1
        if family not in drop:
            kept.append(line)
    unknown = sorted(set(drop) - set(counts))
    if unknown:
        raise ValueError(f"{source}: no rows of families {unknown}")
    out = b"".join(kept)
    fd = os.open(output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(out)
    return {
        "input": str(source),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "dropped_families": sorted(drop),
        "rows_in": sum(counts.values()),
        "rows_dropped": sum(counts[f] for f in drop),
        "rows_out": len(kept),
        "families_out": {f: n for f, n in sorted(counts.items()) if f not in drop},
        "output": str(output),
        "output_sha256": hashlib.sha256(out).hexdigest(),
    }


def soup_problems(
    manifest: dict, relays: dict[str, str], members: list[Path]
) -> list[str]:
    configs = [
        json.loads((m / "decision_config.json").read_text(encoding="utf-8"))["lora"]
        for m in members
    ]
    lora, check = manifest["lora"], manifest["verification"]
    expect = (
        len(members),
        sum(c["rank"] for c in configs),
        sum(c["alpha"] for c in configs),
    )
    problems = []
    if (lora["members"], lora["rank"], lora["alpha"]) != expect:
        problems.append(f"soup lora {lora}, expected members / rank / alpha {expect}")
    if (
        not check["verify_adapter_config"]
        or check["max_relative_diff"] > check["tolerance_relative"]
    ):
        problems.append(
            f"soup verification {check['max_relative_diff']} / {check['verify_adapter_config']}"
        )
    if [Path(m["path"]) for m in manifest["members"]] != members:
        problems.append("soup_manifest.json lists other members")
    for member in manifest["members"]:
        sums = relays.get(member["path"])
        if not sums:
            continue
        listed = {}
        for line in Path(sums).read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(None, 1)
                listed[name.strip().removeprefix("./")] = digest
        for name, digest in member["files_sha256"].items():
            if listed.get(name) != digest:
                problems.append(f"{member['path']}/{name} is not the relayed file")
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("drop-families")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--drop", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("soup-check")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--relay-sums", default="", help='"CKPT=SHA256SUMS ..." (may be empty)'
    )
    p.add_argument("members", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.mode == "drop-families":
        print(
            json.dumps(
                drop_families(args.input, args.drop, args.output), sort_keys=True
            )
        )
        return
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    relays = dict(spec.split("=", 1) for spec in args.relay_sums.split())
    problems = soup_problems(manifest, relays, list(args.members))
    if problems:
        raise SystemExit("; ".join(problems))
    lora, check = manifest["lora"], manifest["verification"]
    print(
        json.dumps(
            {
                "soup": str(args.manifest),
                "model_sha256": manifest["output"]["model_sha256"],
                "members": lora["members"],
                "rank": lora["rank"],
                "alpha": lora["alpha"],
                "max_relative_diff": check["max_relative_diff"],
                "relay_checked": sorted(relays),
            }
        )
    )


if __name__ == "__main__":
    main()
