"""Make an opaque, gold-free v8.2 near-pair review packet from a frozen audit."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from training.model.data import file_sha256

VERSION = "decision2-score-v8.2-near-pair-review/1"
FIELDS = ("state", "instructions", "options")


def _rows(path: Path, *, blind: bool) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if blind:
            items = entry["items"]
            ids = [item["review_id"] for item in items]
        else:
            items = [entry]
            ids = [entry["id"]]
        for item_id, item in zip(ids, items):
            if item_id in result:
                raise ValueError("Duplicate row ID in near-pair source")
            result[item_id] = {field: item[field] for field in FIELDS}
    return result


def write(
    *,
    audit_path: Path,
    audit_sha256: str,
    seed_path: Path,
    sources: dict[str, Path],
    output: Path,
    mapping_output: Path,
) -> dict[str, Any]:
    if file_sha256(audit_path) != audit_sha256:
        raise ValueError("Frozen audit bytes changed")
    if seed_path.stat().st_mode & 0o077:
        raise PermissionError("Private seed must be mode 0600")
    secret = seed_path.read_bytes()
    if len(secret) != 32 or output.exists() or mapping_output.exists():
        raise ValueError("Need a 32-byte seed and fresh output paths")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("schema_version") != "decision2-score-v8.2-pilot-audit/1"
        or audit.get("status") != "HOLD_OVERLAP"
    ):
        raise ValueError("Near-pair packet requires the frozen v8.2 overlap hold")
    required = {
        "v82_train",
        "v82_select",
        "v8_train",
        "v8_select_blind",
        "v81_train",
        "v81_select_blind",
    }
    if set(sources) != required:
        raise ValueError("Near-pair source roster differs")
    rows = {
        name: _rows(path, blind=name.endswith("_blind"))
        for name, path in sources.items()
    }
    roster = {
        "train_vs_candidate_select": ("v82_train", "v82_select"),
        "train_vs_prior_v8_train": ("v82_train", "v8_train"),
        "train_vs_prior_v81_train": ("v82_train", "v81_train"),
        "select_vs_prior_v8_select_goldfree": ("v82_select", "v8_select_blind"),
    }
    if set(audit["flagged_comparisons"]) != set(roster):
        raise ValueError("Frozen flagged comparison roster differs")
    packet, mapping = [], {}
    for comparison in sorted(roster):
        left_source, right_source = roster[comparison]
        value = audit["overlap"][comparison]
        for category in ("near_context", "near_full"):
            for hit in value[category]["examples"]:
                left_id, right_id = hit["left_id"], hit["right_id"]
                if (
                    left_id not in rows[left_source]
                    or right_id not in rows[right_source]
                ):
                    raise ValueError("Audited near hit is absent from frozen source")
                raw_id = f"{comparison}\0{category}\0{left_id}\0{right_id}".encode()
                alias = hmac.new(
                    secret, b"near-pair\0" + raw_id, hashlib.sha256
                ).hexdigest()[:16]
                if alias in mapping:
                    raise ValueError("Near-pair alias collision")
                pair = [rows[left_source][left_id], rows[right_source][right_id]]
                if hmac.new(secret, b"swap\0" + raw_id, hashlib.sha256).digest()[0] & 1:
                    pair.reverse()
                packet.append(
                    {"review_pair": alias, "item_a": pair[0], "item_b": pair[1]}
                )
                mapping[alias] = {
                    "comparison": comparison,
                    "category": category,
                    "left_id": left_id,
                    "right_id": right_id,
                    "similarity": hit["similarity"],
                }
    if len(packet) != 8:
        raise ValueError(f"Expected eight near-text pairs, found {len(packet)}")
    for path, content in (
        (
            output,
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                for row in packet
            ),
        ),
        (mapping_output, json.dumps(mapping, sort_keys=True, indent=2) + "\n"),
    ):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "schema_version": VERSION,
        "pairs": len(packet),
        "packet_sha256": file_sha256(output),
        "mapping_sha256": file_sha256(mapping_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--seed", required=True, type=Path)
    for name in (
        "v82-train",
        "v82-select",
        "v8-train",
        "v8-select-blind",
        "v81-train",
        "v81-select-blind",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping-output", required=True, type=Path)
    args = parser.parse_args()
    sources = {
        name: getattr(args, name)
        for name in (
            "v82_train",
            "v82_select",
            "v8_train",
            "v8_select_blind",
            "v81_train",
            "v81_select_blind",
        )
    }
    print(
        json.dumps(
            write(
                audit_path=args.audit,
                audit_sha256=args.audit_sha256,
                seed_path=args.seed,
                sources=sources,
                output=args.output,
                mapping_output=args.mapping_output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
