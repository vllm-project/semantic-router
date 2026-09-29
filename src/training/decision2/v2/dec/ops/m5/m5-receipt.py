"""Receipt for one decoder M5 formal collection on node B (prereg dec-m5-prereg-2026-09-29.md, "Formal runs").

Writes <run-dir>/M5-RECEIPT.json (exclusive): package identity (per-file SHA-256 list and its hash = revision),
calibration used and the 23:15 decision, adapter spec, collect arguments, image, GPU and wall seconds from the
runner's GPU-TIME.json (GPU-hour accounting), per-panel wall seconds and prediction hashes, the seal hash, and the
autotune cache: manifest of the cache after the run and its difference from the frozen master it was copied from
(new / removed / changed entries; any new entry is flagged).

usage: python3 m5-receipt.py --run-dir R --kind ref|finalist|mlx|smoke --name N --label L --package-dir P
    --package-list LIST --revision REV --model DIR --spec SPEC --calibration-used PATH|none
    --calibration-decision PATH|none --cache-dir C --cache-before MANIFEST|none --cache-after-manifest OUT
    --mirror SRC --image IMAGE [--collect-arg A ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

LISTED = 50


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> str:
    """Same bytes as `find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum` run inside root."""
    files = sorted(
        (
            "./" + str(p.relative_to(root))
            for p in Path(root).rglob("*")
            if p.is_file() and not p.is_symlink()
        ),
        key=lambda s: s.encode(),
    )
    return "".join(f"{sha_file(root / f[2:])}  {f}\n" for f in files)


def parse_manifest(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            out[name] = digest
    return out


def cache_diff(before: dict[str, str] | None, after: dict[str, str]) -> dict:
    if before is None:
        return {"master": None, "entries_after": len(after)}
    new = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(after) & set(before) if after[k] != before[k])
    return {
        "entries_master": len(before),
        "entries_after": len(after),
        "new_entries": len(new),
        "removed_entries": len(removed),
        "changed_entries": len(changed),
        "new_entry_names": new[:LISTED],
        "changed_entry_names": changed[:LISTED],
        "flag_new_cache_entries": bool(new or changed),
    }


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def receipt(args: argparse.Namespace) -> dict:
    run = args.run_dir
    gpu_time = load(run / "GPU-TIME.json") or {}
    collect = load(run / "COLLECT.json") or load(run / "SMOKE.json") or {}
    after_text = tree_manifest(args.cache_dir)
    args.cache_after_manifest.write_text(after_text)
    before = None
    if args.cache_before != "none":
        before = parse_manifest(Path(args.cache_before).read_text())
    cache = {
        "dir": str(args.cache_dir),
        "master_manifest": None if before is None else args.cache_before,
        "master_manifest_sha256": (
            None if before is None else sha_file(Path(args.cache_before))
        ),
        "after_manifest": str(args.cache_after_manifest),
        "after_manifest_sha256": hashlib.sha256(after_text.encode()).hexdigest(),
        **cache_diff(before, parse_manifest(after_text)),
    }
    decision = None
    if (
        args.calibration_decision != "none"
        and Path(args.calibration_decision).is_file()
    ):
        d = json.loads(Path(args.calibration_decision).read_text())
        decision = {
            "file": args.calibration_decision,
            "sha256": sha_file(Path(args.calibration_decision)),
            "adopt": d.get("adopt"),
            "worsened": d.get("worsened"),
            "candidate_sha256": d.get("candidate_sha256"),
            "candidate_temperatures": d.get("candidate_temperatures"),
        }
    params = load(run / "PARAMS.json") or {}
    seal = run / "SEAL.json"
    return {
        "schema": "dec-m5-formal-receipt/1",
        "kind": args.kind,
        "name": args.name,
        "label": args.label,
        "node": "B",
        "mirror": args.mirror,
        "package_dir": str(args.package_dir),
        "package_list": str(args.package_list),
        "package_list_sha256": sha_file(args.package_list),
        "revision": args.revision,
        "revision_matches_list": args.revision == sha_file(args.package_list),
        "model": str(args.model),
        "spec": str(args.spec),
        "calibration_used": args.calibration_used,
        "calibration_used_sha256": (
            sha_file(Path(args.calibration_used))
            if args.calibration_used != "none"
            else None
        ),
        "calibration_decision": args.calibration_decision,
        "calibration_rule": decision,
        "collect_args": args.collect_arg,
        "image": args.image,
        "image_id": gpu_time.get("image_id") or collect.get("image_id"),
        "gpu": gpu_time.get("gpu"),
        "start_utc": gpu_time.get("start_utc"),
        "end_utc": gpu_time.get("end_utc"),
        "wall_seconds": gpu_time.get("wall_seconds"),
        "gpu_hours": gpu_time.get("gpu_hours"),
        "exit_code": gpu_time.get("exit_code"),
        "shared_gpu": gpu_time.get("shared", False),
        "runtime_env": collect.get("runtime_env"),
        "panels": [
            {
                k: p.get(k)
                for k in (
                    "panel",
                    "wall_seconds",
                    "output_rows",
                    "expected_rows",
                    "output_sha256",
                    "exit_code",
                )
            }
            for p in collect.get("panels", [])
        ],
        "seal_sha256": sha_file(seal) if seal.is_file() else None,
        "parameters": params.get("parameters"),
        "parameter_stubs_sha256": params.get("stubs_manifest_sha256"),
        "cache": cache,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in (
        "run-dir",
        "package-dir",
        "package-list",
        "model",
        "spec",
        "cache-dir",
        "cache-after-manifest",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "kind",
        "name",
        "label",
        "revision",
        "calibration-used",
        "calibration-decision",
        "cache-before",
        "mirror",
        "image",
    ):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--collect-arg", action="append", default=[])
    args = p.parse_args()
    out = receipt(args)
    target = args.run_dir / "M5-RECEIPT.json"
    with target.open("x") as stream:
        json.dump(out, stream, indent=1, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    c = out["cache"]
    print(
        json.dumps(
            {
                "wall_seconds": out["wall_seconds"],
                "exit_code": out["exit_code"],
                "seal": out["seal_sha256"],
                "new_cache_entries": c.get("new_entries"),
                "changed_cache_entries": c.get("changed_entries"),
                "flag": c.get("flag_new_cache_entries"),
            }
        )
    )


if __name__ == "__main__":
    main()
