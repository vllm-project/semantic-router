"""Aggregate-only CPU screen for the publisher VitaminC TRAIN as native Score.

No source text, source IDs, answer keys, or training rows are written. Publisher
development/test members are never opened. A missing protected-role inventory
or native tokenizer produces HOLD, not an implicit admission.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

PUBLISHER_ZIP_SHA256 = (
    "49d82dc1690cbee420d18e2c26f687a7937710bb211845d2571430dfd4dc0337"
)
PROTECTED_INVENTORY_SHA256 = (
    "c6d3f497b4385ff48817e9a6f98f63baf529b99b27a132a190164d5d729c18c2"
)
TRAIN_MEMBER = "vitaminc/train.jsonl"
SCHEMA_REAL = {
    "unique_id",
    "case_id",
    "wiki_revision_id",
    "label",
    "claim",
    "evidence",
    "page",
    "revision_type",
}
SCHEMA_SYNTHETIC = (SCHEMA_REAL - {"wiki_revision_id"}) | {"FEVER_id"}
LABELS = ("REFUTES", "NOT ENOUGH INFO", "SUPPORTS")
PROTECTED_FIELDS = {
    "id",
    "group_id",
    "review_id",
    "family",
    "operation",
    "language",
    "state",
    "questions",
    "instructions",
    "options",
}
REQUIRED_ROLES = {
    "typed_dev",
    "css_pilot",
    "typed_final_goldfree",
    "css15_goldfree",
    "jevbench_public231",
    "rights_clean_train",
    "rights_clean_select",
    "rights_clean_cal",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def compact(value: str) -> str:
    return "".join(char for char in normalize(value) if char.isalnum())


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def load_train(archive: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    if sha_file(archive) != PUBLISHER_ZIP_SHA256:
        raise ValueError("Publisher archive SHA-256 changed")
    cases: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    labels: collections.Counter[str] = collections.Counter()
    revision_types: collections.Counter[str] = collections.Counter()
    unique_ids: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    with zipfile.ZipFile(archive) as source:
        if TRAIN_MEMBER not in source.namelist():
            raise ValueError("Publisher TRAIN member missing")
        with source.open(TRAIN_MEMBER) as stream:
            for line in stream:
                row = json.loads(line)
                expected = (
                    SCHEMA_REAL
                    if row.get("revision_type") == "real"
                    else SCHEMA_SYNTHETIC
                )
                if set(row) != expected or not all(
                    isinstance(value, str) and value for value in row.values()
                ):
                    raise ValueError("Publisher TRAIN schema or text changed")
                if row["label"] not in LABELS or row["revision_type"] not in {
                    "real",
                    "synthetic",
                }:
                    raise ValueError("Unexpected label or revision type")
                if row["unique_id"] in unique_ids:
                    raise ValueError("Duplicate publisher unique_id")
                unique_ids.add(row["unique_id"])
                key = (normalize(row["claim"]), normalize(row["evidence"]))
                duplicate_pairs += key in pair_keys
                pair_keys.add(key)
                labels[row["label"]] += 1
                revision_types[row["revision_type"]] += 1
                cases[row["case_id"]].append(row)
    pages = {row["page"] for rows in cases.values() for row in rows}
    return cases, {
        "rows": len(unique_ids),
        "cases": len(cases),
        "pages": len(pages),
        "labels": {label: labels[label] for label in LABELS},
        "revision_types": dict(sorted(revision_types.items())),
        "exact_duplicate_claim_evidence_rows": duplicate_pairs,
        "source_language": "en (publisher collection language; row-level language not verified)",
    }


def eligible_case(rows: list[dict[str, str]]) -> str | None:
    if not rows or {row["revision_type"] for row in rows} != {"real"}:
        return None
    if len({row["page"] for row in rows}) != 1:
        return None
    claims: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        claims[normalize(row["claim"])].append(row)
    if not claims or any(
        len(items) != 2
        or len({normalize(item["evidence"]) for item in items}) != 2
        or "SUPPORTS" not in {item["label"] for item in items}
        or len({item["label"] for item in items}) != 2
        for items in claims.values()
    ):
        return None
    other = {item["label"] for items in claims.values() for item in items} - {
        "SUPPORTS"
    }
    return next(iter(other)) if len(other) == 1 else None


def select_cases(
    cases: dict[str, list[dict[str, str]]], per_stratum: int = 128
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    ranked: dict[str, list[tuple[bytes, str, str]]] = {
        "REFUTES": [],
        "NOT ENOUGH INFO": [],
    }
    for case_id, rows in cases.items():
        other = eligible_case(rows)
        if other in ranked:
            page = rows[0]["page"]
            rank = hashlib.sha256((page + "\0" + case_id).encode()).digest()
            ranked[other].append((rank, case_id, page))
    selected_cases: list[str] = []
    used_pages: set[str] = set()
    for other in ranked:
        stratum_count = 0
        for _, case_id, page in sorted(ranked[other]):
            if page in used_pages:
                continue
            selected_cases.append(case_id)
            used_pages.add(page)
            stratum_count += 1
            if stratum_count == per_stratum:
                break
    counts = collections.Counter(eligible_case(cases[case]) for case in selected_cases)
    if any(counts[label] != per_stratum for label in ranked):
        raise ValueError("Insufficient independent real, contrastive page groups")
    selected = [row for case in selected_cases for row in cases[case]]
    label_counts = collections.Counter(row["label"] for row in selected)
    pair_keys = {
        (normalize(row["claim"]), normalize(row["evidence"])) for row in selected
    }
    # Every retained claim appears under opposing evidence. This is a necessary,
    # not sufficient, negative control against claim-only shortcuts.
    claim_labels: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for row in selected:
        claim_labels[normalize(row["claim"])][row["label"]] += 1
    claim_only_upper = sum(
        max(counts.values()) for counts in claim_labels.values()
    ) / len(selected)
    return selected, {
        "candidate_cases_by_stratum": {
            key: len(value) for key, value in ranked.items()
        },
        "selected_cases": len(selected_cases),
        "selected_pages": len(used_pages),
        "selected_rows": len(selected),
        "selected_duplicate_claim_evidence_rows": len(selected) - len(pair_keys),
        "selected_labels": {label: label_counts[label] for label in LABELS},
        "claim_only_memorization_upper_bound": round(claim_only_upper, 4),
        "sampling": "lowest SHA256(page + NUL + case_id), one case per page, 128 per contrastive stratum",
        "note": "This is a bounded source screen, not an admitted training sample or transfer estimate.",
    }


def text_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in text_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in text_leaves(child)]
    return []


def protected_texts(manifest: Path) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    if sha_file(manifest) != PROTECTED_INVENTORY_SHA256:
        raise ValueError("Protected inventory SHA-256 changed")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Protected inventory missing")
    roles = {entry["role"] for entry in entries}
    if len(roles) != len(entries) or not roles >= REQUIRED_ROLES:
        raise ValueError("Missing or duplicate required protected roles")
    texts: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_file() or sha_file(path) != entry["sha256"]:
            raise ValueError("Protected role missing or changed")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not set(row) <= PROTECTED_FIELDS:
                    raise ValueError("Protected role contains answer-bearing field")
                count += 1
                for field in ("state", "instructions", "options"):
                    for leaf in text_leaves(row.get(field)):
                        normalized = normalize(leaf)
                        if len(compact(normalized)) >= 20:
                            texts.append((entry["role"], normalized))
        counts[entry["role"]] = count
    return texts, {
        "roles": len(roles),
        "row_comparisons": sum(counts.values()),
        "text_leaves": len(texts),
    }


def sixgrams(value: str) -> set[str]:
    value = compact(value)
    return (
        {value[i : i + 6] for i in range(len(value) - 5)} if len(value) >= 20 else set()
    )


def overlap_screen(
    rows: list[dict[str, str]], protected: list[tuple[str, str]]
) -> dict[str, Any]:
    exact: dict[str, set[str]] = collections.defaultdict(set)
    short: dict[str, set[str]] = collections.defaultdict(set)
    long_count = 0
    for role, value in protected:
        exact[value].add(role)
        if len(value) <= 800:
            short[value].add(role)
        else:
            long_count += 1
    refs = sorted(short)
    ref_grams = [sixgrams(value) for value in refs]
    index: dict[str, list[int]] = collections.defaultdict(list)
    for ref_id, grams in enumerate(ref_grams):
        for gram in sorted(grams)[::2]:
            index[gram].append(ref_id)
    hits: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: {"exact": set(), "near": set()}
    )
    source: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        for field in ("claim", "evidence"):
            source[normalize(row[field])].add(row["page"])
    for value, pages in source.items():
        for role in exact.get(value, set()):
            hits[role]["exact"].update(pages)
        grams = sixgrams(value)
        if not grams:
            continue
        rare = sorted(
            (gram for gram in grams if gram in index),
            key=lambda gram: (len(index[gram]), gram),
        )[:6]
        candidates = {
            ref_id for gram in rare if len(index[gram]) <= 128 for ref_id in index[gram]
        }
        for ref_id in candidates:
            other = ref_grams[ref_id]
            if not other:
                continue
            common = len(grams & other)
            if (
                common / len(grams | other) < 0.72
                and common / min(len(grams), len(other)) < 0.9
            ):
                continue
            for role in short[refs[ref_id]] - exact.get(value, set()):
                hits[role]["near"].update(pages)
    return {
        "suspected_page_groups_by_role": {
            role: {kind: len(pages) for kind, pages in kinds.items()}
            for role, kinds in sorted(hits.items())
        },
        "suspected_page_groups_total": (
            len(
                set().union(
                    *(pages for kinds in hits.values() for pages in kinds.values())
                )
            )
            if hits
            else 0
        ),
        "protected_leaves_over_800_chars_not_near_scanned": long_count,
        "near_method": "rare compact six-grams, <=800-character protected leaves; Jaccard>=.72 or containment>=.90; heuristic only",
    }


def native_lengths(
    rows: list[dict[str, str]], tokenizer_path: Path, cap: int
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from training.model.decision_model import encode

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    lengths = []
    overflow = 0
    for index, source in enumerate(rows):
        row = {
            "id": f"screen-{index}",
            "state": f"Evidence:\n{source['evidence']}\n\nClaim:\n{source['claim']}",
            "instructions": "Rate whether the shown evidence refutes, leaves undetermined, or supports the claim. Use only this evidence.",
            "task_type": "score",
            "family": "evidence_revision",
            "options": [
                {"key": "0", "description": "The evidence refutes the claim."},
                {
                    "key": "1",
                    "description": "The evidence does not determine the claim.",
                },
                {"key": "2", "description": "The evidence supports the claim."},
            ],
            "label": LABELS.index(source["label"]),
        }
        try:
            lengths.append(len(encode(row, tokenizer, cap)["ids"]))
        except ValueError as exc:
            if "exceeds max_length" not in str(exc):
                raise
            overflow += 1
    return {
        "tokenizer_json_sha256": sha_file(tokenizer_path / "tokenizer.json"),
        "native_rows": len(rows),
        "over_cap": overflow,
        "cap": cap,
        "tokens": (
            {
                "median": percentile(lengths, 0.5),
                "p90": percentile(lengths, 0.9),
                "p99": percentile(lengths, 0.99),
                "max": max(lengths),
            }
            if lengths
            else None
        ),
    }


def audit(
    archive: Path, protected_manifest: Path | None, tokenizer: Path | None, cap: int
) -> dict[str, Any]:
    cases, corpus = load_train(archive)
    selected, selection = select_cases(cases)
    result: dict[str, Any] = {
        "source": "publisher VitaminC TRAIN only",
        "publisher_archive_sha256": PUBLISHER_ZIP_SHA256,
        "corpus": corpus,
        "selection": selection,
        "rights": "Publisher DATA_LICENSE: Wikipedia article terms, fallback CC BY-SA 3.0; synthetic FEVER-derived rows excluded from pilot",
    }
    if protected_manifest is not None:
        protected, inventory = protected_texts(protected_manifest)
        result["protected_inventory"] = inventory
        result["overlap"] = overlap_screen(selected, protected)
    if tokenizer is not None:
        result["native_length"] = native_lengths(selected, tokenizer, cap)
    blockers = []
    if protected_manifest is None:
        blockers.append("NO_PINNED_PROTECTED_INVENTORY")
    elif result["overlap"]["suspected_page_groups_total"]:
        blockers.append("PROTECTED_NEAR_OR_EXACT_MATCHES_REQUIRE_ADJUDICATION")
    if tokenizer is None:
        blockers.append("NO_PINNED_NATIVE_TOKENIZER")
    elif result["native_length"]["over_cap"]:
        blockers.append("NATIVE_REQUEST_OVER_CAP")
    blockers += [
        "INDEPENDENT_LABEL_TO_RUBRIC_REVIEW_PENDING",
        "SOURCE_DISJOINT_RULE_STATE_SCORE_DIAGNOSTIC_PENDING",
        "SIZE_SPECIFIC_EQUAL_TOKEN_CONTROL_PENDING",
    ]
    result["decision"] = "HOLD"
    result["blockers"] = blockers
    result["gpu_hours"] = 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--protected-manifest", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.archive, args.protected_manifest, args.tokenizer, args.max_length
            ),
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
