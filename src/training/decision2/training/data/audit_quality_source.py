"""Aggregate-only QuALITY v1.0.1 source screen for decision training.

This audit never emits articles, questions, labels, IDs, or derived examples.
It is a source feasibility screen, not an admission into TRAIN or evaluation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import subprocess
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REVISION = "f84977c40dbfef70c9cab48037b7becfc8e45f73"
SPLITS = ("train", "dev", "test")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def license_kind(value: str) -> str:
    if "Project Gutenberg License" in value:
        return "gutenberg_terms_review"
    if value == "https://www.anc.org/OANC/license.txt":
        return "anc_terms_review"
    if "Creative Commons Attribution 4.0 International License (CC BY 4.0)" in value:
        return "cc_by_4_review"
    return "unrecognized_hold"


def percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def url_key(value: str) -> str:
    """Normalize only URL identity; never emit the source URL."""
    if not value:
        return ""
    parsed = urlsplit(value.strip())
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    path = parsed.path.rstrip("/").casefold()
    return f"{host}{path}" if host else ""


def native_choice_tokens(article: str, question: dict[str, Any], tokenizer: Any) -> int:
    """Count the production segmented Choice prompt without loading a label."""
    from training.model.decision_model import segments

    row = {
        "state": article,
        "instructions": question["question"],
        "task_type": "choice",
        "options": [
            {"key": chr(ord("A") + index), "description": option}
            for index, option in enumerate(question["options"])
        ],
    }
    prefix, options, suffix = segments(row)
    return sum(
        len(tokenizer.encode(part, add_special_tokens=False))
        for part in (prefix, *options, suffix)
    )


def summarize(root: Path, tokenizer_dir: Path | None = None) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Official QuALITY source revision mismatch")
    files = {
        split: root / "data" / "v1.0.1" / f"QuALITY.v1.0.1.htmlstripped.{split}"
        for split in SPLITS
    }
    by_split: dict[str, dict[str, Any]] = {}
    article_ids: dict[str, set[str]] = {}
    article_hashes: dict[str, set[str]] = {}
    title_author_hashes: dict[str, set[str]] = {}
    title_hashes: dict[str, set[str]] = {}
    title_to_articles: dict[str, dict[str, set[str]]] = {}
    url_hashes: dict[str, set[str]] = {}
    question_hashes: dict[str, set[str]] = {}
    article_questions_by_split: dict[str, collections.Counter[str]] = {}
    fully_fitting_by_split: dict[str, dict[str, bool]] = {}
    all_question_ids: set[str] = set()
    tokenizer = None
    if tokenizer_dir is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, local_files_only=True, trust_remote_code=False
        )

    for split, path in files.items():
        sets = 0
        lengths: list[int] = []
        question_total = 0
        answer_positions: collections.Counter[int] = collections.Counter()
        difficulty: collections.Counter[int] = collections.Counter()
        license_articles: collections.Counter[str] = collections.Counter()
        license_questions: collections.Counter[str] = collections.Counter()
        source_articles: collections.Counter[str] = collections.Counter()
        high_context_votes = 0
        context_votes = 0
        article_counts: collections.Counter[str] = collections.Counter()
        article_questions: collections.Counter[str] = collections.Counter()
        token_proxy_lengths: list[int] = []
        native_token_lengths: list[int] = []
        fully_native_fitting: dict[str, bool] = {}
        title_author_to_ids: dict[str, set[str]] = collections.defaultdict(set)
        article_ids[split] = set()
        article_hashes[split] = set()
        title_author_hashes[split] = set()
        title_hashes[split] = set()
        title_to_articles[split] = collections.defaultdict(set)
        url_hashes[split] = set()
        question_hashes[split] = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                sets += 1
                article_id = str(row["article_id"])
                article = row["article"]
                if not isinstance(article, str) or not article.strip():
                    raise ValueError("Empty article")
                article_ids[split].add(article_id)
                article_hashes[split].add(digest(norm(article)))
                title = norm(row["title"])
                author = norm(row["author"])
                if title:
                    title_hash = digest(title)
                    title_hashes[split].add(title_hash)
                    title_to_articles[split][title_hash].add(article_id)
                    if author:
                        title_author = digest((title, author))
                        title_author_hashes[split].add(title_author)
                        title_author_to_ids[title_author].add(article_id)
                normalized_url = url_key(row["url"])
                if normalized_url:
                    url_hashes[split].add(digest(normalized_url))
                article_counts[article_id] += 1
                fully_native_fitting.setdefault(article_id, True)
                lengths.append(len(article.split()))
                article_tokens = (
                    len(tokenizer.encode(article, add_special_tokens=False))
                    if tokenizer is not None
                    else None
                )
                kind = license_kind(row["license"])
                license_articles[kind] += 1
                source_articles[row["source"]] += 1
                for question in row["questions"]:
                    options = question["options"]
                    gold = question.get("gold_label")
                    if len(options) != 4 or (
                        split != "test"
                        and (type(gold) is not int or gold not in range(1, 5))
                    ):
                        raise ValueError("Invalid four-option source record")
                    if split == "test" and gold is not None:
                        raise ValueError("Expected official test labels to be hidden")
                    qid = str(question["question_unique_id"])
                    if qid in all_question_ids:
                        raise ValueError("Duplicate question_unique_id across release")
                    all_question_ids.add(qid)
                    question_hashes[split].add(
                        digest(
                            (
                                norm(article),
                                norm(question["question"]),
                                tuple(norm(x) for x in options),
                            )
                        )
                    )
                    question_total += 1
                    article_questions[article_id] += 1
                    if tokenizer is not None and article_tokens is not None:
                        # Raw content only; native System One adds framing.
                        token_proxy_lengths.append(
                            article_tokens
                            + len(
                                tokenizer.encode(
                                    question["question"], add_special_tokens=False
                                )
                            )
                            + sum(
                                len(tokenizer.encode(x, add_special_tokens=False))
                                for x in options
                            )
                        )
                        native_length = native_choice_tokens(
                            article, question, tokenizer
                        )
                        native_token_lengths.append(native_length)
                        if native_length > 8192:
                            fully_native_fitting[article_id] = False
                    if gold is not None:
                        answer_positions[gold] += 1
                    difficulty[int(question["difficult"])] += 1
                    license_questions[kind] += 1
                    for review in question.get("validation", []):
                        rating = review.get("untimed_eval2_context")
                        if rating in (1, 2, 3, 4):
                            context_votes += 1
                            high_context_votes += rating in (3, 4)
        if any(count != 2 for count in article_counts.values()):
            raise ValueError("Expected exactly two writer sets per article")
        article_questions_by_split[split] = article_questions
        fully_fitting_by_split[split] = fully_native_fitting
        by_split[split] = {
            "source_file_sha256": sha(path),
            "writer_sets": sets,
            "articles": len(article_ids[split]),
            "questions": question_total,
            "article_words": {
                "median": statistics.median(lengths),
                "p90": percentile(lengths, 0.9),
                "p99": percentile(lengths, 0.99),
                "max": max(lengths),
            },
            "answer_position_counts": dict(sorted(answer_positions.items())),
            "difficulty_counts": dict(sorted(difficulty.items())),
            "license_writer_sets": dict(sorted(license_articles.items())),
            "license_questions": dict(sorted(license_questions.items())),
            "source_writer_sets": dict(sorted(source_articles.items())),
            "distinct_title_author_cross_article_groups": sum(
                len(ids) > 1 for ids in title_author_to_ids.values()
            ),
            "context_votes_3_or_4": high_context_votes,
            "context_votes_total": context_votes,
            "raw_content_token_proxy": (
                {
                    "median": percentile(token_proxy_lengths, 0.5),
                    "p90": percentile(token_proxy_lengths, 0.9),
                    "p99": percentile(token_proxy_lengths, 0.99),
                    "at_most_8192": sum(n <= 8192 for n in token_proxy_lengths),
                    "at_most_4096": sum(n <= 4096 for n in token_proxy_lengths),
                }
                if tokenizer is not None
                else None
            ),
            "native_system_one_choice_tokens": (
                {
                    "median": percentile(native_token_lengths, 0.5),
                    "p90": percentile(native_token_lengths, 0.9),
                    "p99": percentile(native_token_lengths, 0.99),
                    "max": max(native_token_lengths),
                    "at_most_8192": sum(n <= 8192 for n in native_token_lengths),
                    "at_most_4096": sum(n <= 4096 for n in native_token_lengths),
                    "fully_fitting_article_groups_at_8192": sum(
                        fully_native_fitting.values()
                    ),
                    "questions_in_fully_fitting_article_groups_at_8192": sum(
                        article_questions[article_id]
                        for article_id, fits in fully_native_fitting.items()
                        if fits
                    ),
                }
                if tokenizer is not None
                else None
            ),
        }

    overlaps: dict[str, dict[str, int]] = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            overlaps[f"{left}_{right}"] = {
                "article_ids": len(article_ids[left] & article_ids[right]),
                "normalized_articles": len(
                    article_hashes[left] & article_hashes[right]
                ),
                "normalized_title_author": len(
                    title_author_hashes[left] & title_author_hashes[right]
                ),
                "normalized_title": len(title_hashes[left] & title_hashes[right]),
                "normalized_url": len(url_hashes[left] & url_hashes[right]),
                "normalized_question_article_options": len(
                    question_hashes[left] & question_hashes[right]
                ),
            }
    reused_titles = title_hashes["train"] & (title_hashes["dev"] | title_hashes["test"])
    held_train_articles = {
        article_id
        for title in reused_titles
        for article_id in title_to_articles["train"][title]
    }
    clean_train_articles = article_ids["train"] - held_train_articles
    return {
        "source": "https://github.com/nyu-mll/quality",
        "revision": revision,
        "version": "v1.0.1.htmlstripped",
        "splits": by_split,
        "cross_split_overlaps": overlaps,
        "train_title_collision_hold": {
            "title_keys": len(reused_titles),
            "article_groups": len(held_train_articles),
            "questions": sum(
                article_questions_by_split["train"][article_id]
                for article_id in held_train_articles
            ),
            "remaining_8192_fit_article_groups": sum(
                fully_fitting_by_split["train"][article_id]
                for article_id in clean_train_articles
            ),
            "remaining_questions_in_whole_8192_fit_groups": sum(
                article_questions_by_split["train"][article_id]
                for article_id in clean_train_articles
                if fully_fitting_by_split["train"][article_id]
            ),
        },
        "tokenizer_sha256": (
            {
                name: sha(tokenizer_dir / name)
                for name in ("tokenizer.json", "tokenizer_config.json")
            }
            if tokenizer_dir is not None
            else None
        ),
        "admission": "HOLD: upstream rights, protected-panel overlap, native token lengths and external-transfer value unverified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(args.root, args.tokenizer_dir),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
