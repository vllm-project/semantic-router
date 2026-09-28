#!/usr/bin/env python3
"""Aggregate-only TRAIN review inventory for a pinned PeerRead checkout.

The output contains no review text, paper IDs, or individual ratings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


SECTIONS = ("acl_2017", "conll_2016", "iclr_2017")
RATING = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)\s*")
SCORE_LEAK = re.compile(
    r"\b(?:recommendation|overall\s+(?:rating|score))\s*[:=]\s*[0-9]+\b",
    re.IGNORECASE,
)


def rating_value(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        match = RATING.fullmatch(value)
        if match:
            return match.group(1)
    return None


def inventory_section(repo: Path, name: str) -> dict[str, object]:
    review_dir = repo / "data" / name / "train" / "reviews"
    files = sorted(review_dir.glob("*.json"))
    digest = hashlib.sha256()
    keys: Counter[str] = Counter()
    ratings: Counter[str] = Counter()
    text_lens: list[int] = []
    papers_with_review: set[str] = set()
    papers_with_rating: set[str] = set()
    review_count = 0
    nonempty_count = 0
    leak_count = 0
    ambiguous_rating_count = 0
    for path in files:
        raw = path.read_bytes()
        digest.update(path.relative_to(repo).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
        paper = json.loads(raw)
        paper_id = str(paper.get("id", path.stem))
        reviews = paper.get("reviews", [])
        if not isinstance(reviews, list):
            raise ValueError(f"invalid reviews list in {path}")
        for review in reviews:
            if not isinstance(review, dict):
                raise ValueError(f"invalid review in {path}")
            review_count += 1
            papers_with_review.add(paper_id)
            keys.update(review.keys())
            comments = review.get("comments")
            if not isinstance(comments, str) or not comments.strip():
                continue
            nonempty_count += 1
            words = len(comments.split())
            text_lens.append(words)
            leak_count += bool(SCORE_LEAK.search(comments))
            if "RECOMMENDATION" in review:
                parsed = rating_value(review["RECOMMENDATION"])
                if parsed is None:
                    ambiguous_rating_count += 1
                else:
                    ratings[parsed] += 1
                    papers_with_rating.add(paper_id)
    text_lens.sort()
    return {
        "section": name,
        "train_review_files": len(files),
        "train_review_bytes_sha256": digest.hexdigest(),
        "section_license_file_present": (
            repo / "data" / name / "LICENSE.txt"
        ).is_file(),
        "review_count": review_count,
        "nonempty_review_count": nonempty_count,
        "papers_with_reviews": len(papers_with_review),
        "strict_numeric_recommendations_with_text": sum(ratings.values()),
        "papers_with_strict_numeric_recommendations": len(papers_with_rating),
        "ambiguous_recommendation_values_with_text": ambiguous_rating_count,
        "recommendation_distribution": dict(sorted(ratings.items())),
        "review_key_frequencies": dict(sorted(keys.items())),
        "text_word_length_p10_p50_p90": [
            text_lens[(len(text_lens) - 1) * numerator // 10] if text_lens else None
            for numerator in (1, 5, 9)
        ],
        "explicit_rating_string_leak_count": leak_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != args.expected_revision:
        raise ValueError("publisher revision mismatch")
    result = {
        "publisher_revision": revision,
        "scope": "publisher TRAIN reviews only; aggregate metadata; no data admitted",
        "sections": [inventory_section(args.repo, name) for name in SECTIONS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
