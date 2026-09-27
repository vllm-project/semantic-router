"""Repair only the line order of a frozen v3 raw prediction hash manifest.

The saved planner serializes ``paths`` with sorted JSON keys, while its emitted
hash command used insertion order. Preserve that command's exact output and
verify every file digest before making the auditor's expected ordering. This
pre-key execution addendum does not read gold labels or alter predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def reorder(
    plan_path: Path,
    plan_sha256: str,
    raw: Path,
    planned_copy: Path,
    receipt: Path,
) -> dict:
    if sha(plan_path) != plan_sha256:
        raise ValueError("frozen plan digest changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_version") != "decision2-first-release-v3-plan/2":
        raise ValueError("unexpected plan version")
    if raw != Path(plan["evaluation_root"]) / "RAW_PREDICTIONS.sha256":
        raise ValueError("raw manifest is not the planned output")
    if any(path.exists() for path in (planned_copy, receipt, raw.with_suffix(".tmp"))):
        raise FileExistsError("repair evidence already exists")
    original_sha256 = sha(raw)
    original = raw.read_text(encoding="utf-8").splitlines()
    ordered = []
    for model in plan["inference"]:
        for prediction in model["paths"].values():
            path = Path(prediction)
            ordered.append(f"{sha(path)}  {path}")
            if model["group"] == "decision2":
                manifest = Path(prediction + ".manifest.json")
                ordered.append(f"{sha(manifest)}  {manifest}")
    if len(ordered) != len(original) or sorted(ordered) != sorted(original):
        raise ValueError("raw manifest differs beyond line ordering")
    if ordered == original:
        raise ValueError("raw manifest already has auditor order")
    shutil.copyfile(raw, planned_copy)
    os.chmod(planned_copy, 0o600)
    if sha(planned_copy) != original_sha256:
        raise ValueError("original manifest copy differs")
    temporary = raw.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        os.chmod(temporary, 0o600)
        output.write("\n".join(ordered) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, raw)
    result = {
        "schema_version": "decision2-v3-raw-manifest-order-repair/1",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_sha256,
        "script_sha256": sha(Path(__file__)),
        "original_sha256": original_sha256,
        "original_copy_sha256": sha(planned_copy),
        "auditor_order_sha256": sha(raw),
        "line_count": len(ordered),
        "same_path_digest_multiset": True,
        "prediction_bytes_modified": False,
    }
    with receipt.open("x", encoding="utf-8") as output:
        os.chmod(receipt, 0o600)
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--planned-copy", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = reorder(
        args.plan,
        args.plan_sha256,
        args.raw,
        args.planned_copy,
        args.receipt,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
