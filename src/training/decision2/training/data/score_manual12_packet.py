"""Seal a private, hand-authored Score quality probe for independent review.

This tool does not admit training data. The authoring JSON and outputs must
remain in a private ignored directory; only a digest-only receipt is public.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MECHANISMS = frozenset(
    {
        "exception_precedence",
        "state_transition",
        "multi_source_evidence",
        "numeric_boundary",
        "temporal_precedence",
        "evidence_insufficiency",
    }
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def _hmac(secret: bytes, value: str) -> str:
    return hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _oracle(variant: dict[str, Any]) -> int:
    if not isinstance(variant["fatal"], bool) or not isinstance(
        variant["pending"], bool
    ):
        raise ValueError("Oracle facts must be Boolean")
    if variant["fatal"] and variant["pending"]:
        raise ValueError("Contradictory oracle facts")
    return 0 if variant["fatal"] else 1 if variant["pending"] else 2


def render(case: dict[str, Any], evidence: str | None) -> str:
    middle = evidence if evidence is not None else "[source excerpt withheld]"
    return "\n\n".join(
        (
            str(case["question"]).strip(),
            str(case["before"]).strip(),
            middle.strip(),
            str(case["after"]).strip(),
        )
    )


def _inspect(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if len(cases) != 12:
        raise ValueError("Pilot requires exactly 12 independently authored groups")
    identities = [case["id"] for case in cases]
    if len(set(identities)) != 12:
        raise ValueError("Group IDs must be unique")
    pair_counts = Counter((case["mechanism"], case["length"]) for case in cases)
    if set(pair_counts) != {
        (name, length) for name in MECHANISMS for length in ("short", "long")
    }:
        raise ValueError("Require one short and one long case per mechanism")
    if any(count != 1 for count in pair_counts.values()):
        raise ValueError("Case mechanism/length pair repeated")
    language_counts = Counter((case["length"], case["language"]) for case in cases)
    if language_counts != {
        ("short", "en"): 3,
        ("short", "zh"): 3,
        ("long", "en"): 3,
        ("long", "zh"): 3,
    }:
        raise ValueError("Short/long English/Chinese allocation changed")

    full_texts: set[str] = set()
    metrics: list[dict[str, Any]] = []
    for case in cases:
        variants = case["variants"]
        if len(variants) != 3 or sorted(_oracle(row) for row in variants) != [0, 1, 2]:
            raise ValueError(f"Three distinct oracle levels required: {case['id']}")
        if len({row["evidence"] for row in variants}) != 3:
            raise ValueError(f"Variants must change the decisive source: {case['id']}")
        ablated = {render(case, None) for _ in variants}
        if len(ablated) != 1:
            raise ValueError(f"Source ablation failed: {case['id']}")
        for index, row in enumerate(variants):
            document = render(case, row["evidence"])
            if document in full_texts:
                raise ValueError(f"Duplicate rendered question: {case['id']}/{index}")
            full_texts.add(document)
            metrics.append(
                {
                    "case": case["id"],
                    "variant": index,
                    "language": case["language"],
                    "length": case["length"],
                    "characters": len(document),
                    "words": (
                        len(re.findall(r"\b\w+\b", document))
                        if case["language"] == "en"
                        else None
                    ),
                    "cjk_characters": (
                        len(re.findall(r"[\u3400-\u9fff]", document))
                        if case["language"] == "zh"
                        else None
                    ),
                    "source_ablation_identical_within_group": True,
                }
            )
    return {
        "metrics": metrics,
        "mechanism_counts": dict(Counter(case["mechanism"] for case in cases)),
    }


def seal(source_file: Path, secret_file: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if secret_file.stat().st_mode & 0o077:
        raise PermissionError("HMAC seed must have mode 0600")
    secret = secret_file.read_bytes()
    if len(secret) != 32:
        raise ValueError("HMAC seed must contain exactly 32 bytes")
    cases = json.loads(source_file.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Source must be a list of cases")
    audit = _inspect(cases)

    packets: list[dict[str, str]] = []
    keys: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for case in cases:
        members = []
        for index, variant in enumerate(case["variants"]):
            review_id = _hmac(secret, f"score12-review\0{case['id']}\0{index}")[:20]
            members.append(review_id)
            packets.append(
                {"review_id": review_id, "question": render(case, variant["evidence"])}
            )
            keys.append(
                {
                    "review_id": review_id,
                    "group_id": case["id"],
                    "mechanism": case["mechanism"],
                    "language": case["language"],
                    "length": case["length"],
                    "label": _oracle(variant),
                    "author_reason": variant["reason"],
                }
            )
        pairs.append({"group_id": case["id"], "review_ids": members})
    packets.sort(key=lambda row: _hmac(secret, "shuffle\0" + row["review_id"]))
    if len({row["review_id"] for row in packets}) != 36:
        raise ValueError("Opaque review IDs collided")

    output_dir.mkdir(parents=True, mode=0o700)
    blind_path = output_dir / "blind-questions.jsonl"
    key_path = output_dir / "sealed-key.jsonl"
    pair_path = output_dir / "sealed-pairs.jsonl"
    audit_path = output_dir / "author-audit.json"
    manifest_path = output_dir / "blind-manifest.json"
    _write_jsonl(blind_path, packets)
    _write_jsonl(key_path, keys)
    _write_jsonl(pair_path, pairs)
    audit.update(
        {
            "status": "HOLD_PENDING_INDEPENDENT_BLIND_REVIEW_AND_PROTECTED_OVERLAP",
            "source_sha256": sha_file(source_file),
            "blind_sha256": sha_file(blind_path),
            "key_sha256": sha_file(key_path),
            "pairs_sha256": sha_file(pair_path),
            "seed_sha256": sha_file(secret_file),
            "answer_balance": dict(Counter(str(row["label"]) for row in keys)),
            "native_token_count_status": "NOT_CHECKED",
            "protected_overlap_status": "NOT_CHECKED",
            "human_language_review_status": "NOT_CHECKED",
        }
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit_path.chmod(0o600)
    manifest = {
        "schema": "decision2-score12-blind/1",
        "items": len(packets),
        "questions_sha256": sha_file(blind_path),
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path.chmod(0o600)
    return {
        "manifest": manifest,
        "audit_sha256": sha_file(audit_path),
        "manifest_sha256": sha_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            seal(args.source_file, args.seed_file, args.output_dir), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
