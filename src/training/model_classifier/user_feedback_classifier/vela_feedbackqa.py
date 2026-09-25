"""Reserved external SAT-only diagnostic from human FeedbackQA judgments.

The source is not a four-class feedback benchmark. This conservative projection
uses only excellent-rated comments explicitly evaluating an answer positively.
It supports external SAT recall, not four-way accuracy or precision. No part of
FeedbackQA is used by the Vela training recipe.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .data_contract import fingerprint

SOURCE_REVISION = "41ea0e39f062f9ca791fd5ec95c364a22150b56e"
SOURCE_SHA256 = "50c4a21dc778cf064f731161e2213f21d2951cabd9331a1c524f791055040d02"
MIN_CHARACTERS = 20
MAX_CHARACTERS = 600
POSITIVE = re.compile(
    r"\b(answer|response)\b.{0,80}\b(excellent|clear|correct|helpful|relevant|useful|directly|comprehensive)\b",
    re.I,
)
QUALIFIED = re.compile(
    r"\b(not|but|however|could|missing|lack|lacks|incomplete|wrong|incorrect|unhelpful|poor)\b",
    re.I,
)


def project(rows, excluded):
    selected, seen, dropped = [], set(), Counter()
    for index, row in enumerate(rows):
        if len(row["feedback"]) != len(row["rating"]):
            raise ValueError("FeedbackQA rating/comment alignment changed")
        group = hashlib.sha256(fingerprint(row["question"]).encode()).hexdigest()
        for variant, (raw_text, rating) in enumerate(
            zip(row["feedback"], row["rating"], strict=True)
        ):
            text = raw_text.strip()
            if (
                rating != "Excellent"
                or not MIN_CHARACTERS <= len(text) <= MAX_CHARACTERS
            ):
                dropped["rating_or_length"] += 1
                continue
            if not POSITIVE.search(text) or QUALIFIED.search(text):
                dropped["not_an_explicit_unqualified_positive_answer_judgment"] += 1
                continue
            key = fingerprint(text)
            if key in excluded or key in seen:
                dropped["duplicate_or_train_dev_overlap"] += 1
                continue
            seen.add(key)
            selected.append(
                {
                    "id": f"feedbackqa:sat-external:{index}:{variant}",
                    "group_id": f"feedbackqa:question:{group}",
                    "text": text,
                    "label": "SAT",
                    "language": "en",
                    "source": "feedbackqa_human_excellent_conservative_sat_projection",
                    "length_bucket": "natural_short",
                    "label_provenance": "human Excellent rating plus explicit positive-answer lexical filter",
                }
            )
    return selected, dict(dropped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-test", type=Path, required=True)
    parser.add_argument("--exclude", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite reserved evaluation data")
    payload = args.source_test.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("FeedbackQA source differs from the pinned official test")
    excluded = {
        fingerprint(json.loads(line)["text"])
        for path in args.exclude
        for line in path.read_text().split("\n")
        if line.strip()
    }
    rows, dropped = project(json.loads(payload), excluded)
    if not rows:
        raise ValueError("No eligible external SAT feedback")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
    )
    receipt = {
        "dataset": "McGill-NLP/feedbackQA",
        "revision": SOURCE_REVISION,
        "license": "apache-2.0",
        "source_sha256": SOURCE_SHA256,
        "rows": len(rows),
        "question_groups": len({row["group_id"] for row in rows}),
        "dropped": dropped,
        "scope": "Reserved external SAT-only diagnostic; not a four-way gold benchmark",
        "training_use": False,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "excluded_files": [
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in args.exclude
        ],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
