"""Aggregate-only ContractNLI TRAIN screen for a source-disjoint 4B probe.

Read the publisher TRAIN member and the bundled terms from the original ZIP.
The publisher DEV and TEST labels are intentionally never opened. This tool
does not generate training rows or an evaluation set. Run it only in a private
experiment directory; its JSON output contains no source text or row IDs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

LABELS = ("Contradiction", "NotMentioned", "Entailment")
TRAIN_MEMBER = "contract-nli/train.json"
TERMS_MEMBER = "contract-nli/TERMS"
DOC_TYPES = ("search-pdf", "sec-text", "sec-html")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("Cannot summarize empty values")
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(values: list[int]) -> dict[str, int]:
    return {
        "min": min(values),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def _majority(counts: collections.Counter[str]) -> str:
    # Fixed tie order independent of publisher row order.
    return max(LABELS, key=lambda label: (counts[label], -LABELS.index(label)))


def _macro_f1(truth: list[str], predicted: list[str]) -> float:
    scores = []
    for label in LABELS:
        tp = sum(
            a == label and b == label for a, b in zip(truth, predicted, strict=True)
        )
        fp = sum(
            a != label and b == label for a, b in zip(truth, predicted, strict=True)
        )
        fn = sum(
            a == label and b != label for a, b in zip(truth, predicted, strict=True)
        )
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def document_token_summary(
    documents: list[dict[str, Any]],
    encode: Callable[[str], list[int]],
    *,
    cap: int,
) -> dict[str, Any]:
    """Count document tokens only, a lower bound on native prompt length."""
    lengths = [len(encode(document["text"])) for document in documents]
    return {
        "document_token_lengths": summarize(lengths),
        "documents_over_cap_before_native_wrapper": sum(
            length > cap for length in lengths
        ),
        "cap": cap,
        "interpretation": "document-only lower bound; full native prompts are longer",
    }


def audit_train(payload: dict[str, Any], *, seed: str) -> dict[str, Any]:
    """Screen official TRAIN with a deterministic, document-disjoint holdout."""
    documents = payload["documents"]
    hypotheses = payload["labels"]
    if not isinstance(documents, list) or not documents:
        raise ValueError("TRAIN documents missing")
    if not isinstance(hypotheses, dict) or not hypotheses:
        raise ValueError("TRAIN hypothesis definitions missing")
    if any(
        not isinstance(value.get("hypothesis"), str) for value in hypotheses.values()
    ):
        raise ValueError("TRAIN hypothesis text missing")

    doc_ids: set[str] = set()
    rows: list[tuple[str, str, str, str]] = []
    doc_chars: list[int] = []
    evidence_position: list[int] = []
    type_counts: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    label_counts: collections.Counter[str] = collections.Counter()
    by_hypothesis: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )

    for document in documents:
        doc_id = str(document["id"])
        if doc_id in doc_ids:
            raise ValueError("Duplicate publisher document ID")
        doc_ids.add(doc_id)
        text = document["text"]
        if not isinstance(text, str) or not text:
            raise ValueError("Empty publisher document")
        kind = document["document_type"]
        if kind not in DOC_TYPES:
            raise ValueError("Unexpected publisher document type")
        type_counts[kind] += 1
        doc_chars.append(len(text))
        spans = document["spans"]
        if len(document["annotation_sets"]) != 1:
            raise ValueError("Unexpected number of annotation sets")
        annotations = document["annotation_sets"][0]["annotations"]
        if set(annotations) != set(hypotheses):
            raise ValueError("Incomplete document annotation set")
        for hypothesis_id, annotation in annotations.items():
            label = annotation["choice"]
            if label not in LABELS:
                raise ValueError("Unexpected publisher NLI label")
            evidence = annotation["spans"]
            if label == "NotMentioned" and evidence:
                errors["not_mentioned_has_evidence"] += 1
            if label != "NotMentioned" and not evidence:
                errors["entailed_or_contradicted_lacks_evidence"] += 1
            for span_index in evidence:
                if not isinstance(span_index, int) or not 0 <= span_index < len(spans):
                    errors["invalid_span_index"] += 1
                    continue
                start, end = spans[span_index]
                if not (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(text)
                ):
                    errors["invalid_span_bounds"] += 1
                    continue
                evidence_position.append(int(start * 100 / len(text)))
            rows.append((doc_id, kind, hypothesis_id, label))
            label_counts[label] += 1
            by_hypothesis[hypothesis_id][label] += 1

    training: list[tuple[str, str, str, str]] = []
    holdout: list[tuple[str, str, str, str]] = []
    for row in rows:
        value = int.from_bytes(
            hashlib.sha256(f"{seed}\0{row[0]}".encode()).digest()[:8], "big"
        )
        (holdout if value % 5 == 0 else training).append(row)
    if not training or not holdout:
        raise ValueError("Deterministic document split is empty")
    global_counts = collections.Counter(label for _, _, _, label in training)
    train_by_hypothesis: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    train_by_doc_type: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for _, kind, hypothesis_id, label in training:
        train_by_hypothesis[hypothesis_id][label] += 1
        train_by_doc_type[kind][label] += 1
    global_guess = _majority(global_counts)
    shortcut_predictions: dict[str, list[str]] = collections.defaultdict(list)
    holdout_counts = collections.Counter()
    holdout_truth: list[str] = []
    for _, kind, hypothesis_id, label in holdout:
        holdout_counts[label] += 1
        holdout_truth.append(label)
        shortcut_predictions["global_majority"].append(global_guess)
        shortcut_predictions["hypothesis_only"].append(
            _majority(train_by_hypothesis.get(hypothesis_id) or global_counts)
        )
        shortcut_predictions["document_type_only"].append(
            _majority(train_by_doc_type.get(kind) or global_counts)
        )

    # IDs and text are never emitted. The fixed hypothesis count is meaningful:
    # memorizing 17 clauses can look like document comprehension.
    per_hypothesis_majority = [
        max(counts.values()) / sum(counts.values()) for counts in by_hypothesis.values()
    ]
    class_support_per_hypothesis = collections.Counter(
        sum(counts[label] > 0 for label in LABELS) for counts in by_hypothesis.values()
    )
    return {
        "status": "source_screen_only",
        "source_role": "publisher_train_only",
        "developer_test_labels_opened": False,
        "documents": len(documents),
        "hypotheses": len(hypotheses),
        "labeled_pairs": len(rows),
        "document_types": dict(sorted(type_counts.items())),
        "class_counts": {label: label_counts[label] for label in LABELS},
        "document_chars": summarize(doc_chars),
        "evidence_start_percent_of_document": (
            summarize(evidence_position) if evidence_position else None
        ),
        "evidence_integrity_errors": dict(sorted(errors.items())),
        "document_disjoint_split": {
            "train_documents": len({row[0] for row in training}),
            "holdout_documents": len({row[0] for row in holdout}),
            "holdout_pairs": len(holdout),
            "holdout_class_counts": {label: holdout_counts[label] for label in LABELS},
            "shortcuts": {
                name: {
                    "correct": sum(
                        a == b
                        for a, b in zip(
                            holdout_truth, shortcut_predictions[name], strict=True
                        )
                    ),
                    "total": len(holdout),
                    "macro_f1": round(
                        _macro_f1(holdout_truth, shortcut_predictions[name]), 6
                    ),
                }
                for name in (
                    "global_majority",
                    "hypothesis_only",
                    "document_type_only",
                )
            },
        },
        "per_hypothesis_majority_fraction": {
            "median_milli": percentile(
                [round(1000 * number) for number in per_hypothesis_majority], 0.5
            ),
            "max_milli": round(1000 * max(per_hypothesis_majority)),
        },
        "class_support_per_hypothesis": {
            str(class_count): class_support_per_hypothesis[class_count]
            for class_count in (1, 2, 3)
        },
    }


def audit_archive(
    path: Path,
    *,
    expected_sha256: str,
    seed: str,
    tokenizer_path: Path | None = None,
    cap: int = 8192,
) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("Publisher archive SHA-256 differs")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if not {TRAIN_MEMBER, TERMS_MEMBER}.issubset(names):
            raise ValueError("Publisher archive members missing")
        terms = archive.read(TERMS_MEMBER).decode("utf-8")
        if "Creative Commons Attribution 4.0" not in terms:
            raise ValueError("Publisher usage terms differ")
        train = archive.read(TRAIN_MEMBER)
        payload = json.loads(train)
        result = audit_train(payload, seed=seed)
        if tokenizer_path is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, local_files_only=True, trust_remote_code=False
            )
            result["tokenizer_provenance"] = "file SHA-256 only; private path omitted"
            result["tokenizer_files"] = {
                filename: sha256_file(tokenizer_path / filename)
                for filename in ("tokenizer.json", "tokenizer_config.json")
            }
            result["token_screen"] = document_token_summary(
                payload["documents"],
                lambda text: tokenizer.encode(text, add_special_tokens=False),
                cap=cap,
            )
        result.update(
            {
                "archive_sha256": actual_sha256,
                "publisher_train_member_sha256": hashlib.sha256(train).hexdigest(),
                "publisher_terms_member_sha256": hashlib.sha256(
                    terms.encode()
                ).hexdigest(),
                "publisher_terms": "CC BY 4.0; preserve attribution and source conditions",
            }
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", default="decision20-4b-contractnli-train-screen-v1")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--context-cap", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_archive(
        args.archive,
        expected_sha256=args.expected_sha256,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        cap=args.context_cap,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
