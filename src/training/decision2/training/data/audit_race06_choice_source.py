"""Bounded, CPU-only official RACE TRAIN screen for native 0.6B Choice.

The protected reference is the pinned eight-role input-only projection. This
tool never opens protected answer keys or runs a model. Source text and source
answer keys remain in a private directory; stdout and the aggregate receipt
contain no individual example, prediction, passage, question, or option.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.data.audit_27b_full_input_overlap import full_input_overlap_rows
from training.data.plan_goldfree_inventory import (
    CORE_ROLES,
    NATIVE_ROLE_COUNTS,
    PARTITION_ROLE_COUNTS,
    validate_core_rows,
)
from training.model.data import canonical, file_sha256

SOURCE_URL = "https://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"
SOURCE_ARCHIVE_SHA256 = (
    "b2769cc9fdc5c546a693300eb9a966cec6870bd349fbc44ed5225f8ad33006e5"
)
PROTECTED_MANIFEST_SHA256 = (
    "26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1"
)
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
SOURCE_SPLITS = {"middle": 6409, "high": 18728}
SAMPLE_GROUPS_PER_STRATUM = 64
REVIEW_GROUPS_PER_STRATUM = 12
MAX_NATIVE_TOKENS = 8192
SAMPLE_SEED = "decision2-race06-choice-source-screen-v1"
WORD = re.compile(r"\b[a-z][a-z']+\b")
PROMPT_VERSION = "decision2-segmented-options-global-query-v1"
PRODUCTION_SEGMENTS_SHA256 = (
    "464efa93fa25f5a7e9dda6b9011d38507930a282d25972848d88ca2bb446bb10"
)


@dataclass(frozen=True)
class Passage:
    source_id: str
    level: str
    article: str
    questions: tuple[str, ...]
    options: tuple[tuple[str, ...], ...]
    answers: tuple[str, ...]
    group: str


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _pinned(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError("Pinned source bytes are missing or changed")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Projected inventory has duplicate JSON keys")
        result[key] = value
    return result


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def load_protected(manifest: Path) -> dict[str, list[dict[str, Any]]]:
    """Verify the complete eight-role input-only manifest and every role file."""
    _pinned(manifest, PROTECTED_MANIFEST_SHA256)
    document = json.loads(manifest.read_text(), object_pairs_hook=_unique_keys)
    if (
        not isinstance(document, dict)
        or set(document)
        != {"schema", "roles", "excluded_optional_role_count", "source_manifest_sha256"}
        or document["schema"] != "decision2-projected-core-inputs-v1"
        or not isinstance(document["roles"], dict)
        or set(document["roles"]) != CORE_ROLES
        or not _is_sha(document["source_manifest_sha256"])
    ):
        raise ValueError("Projected inventory manifest schema differs")
    roles = {}
    for role, entry in sorted(document["roles"].items()):
        expected = (
            NATIVE_ROLE_COUNTS[role]
            if role in NATIVE_ROLE_COUNTS
            else PARTITION_ROLE_COUNTS[role][1]
        )
        required = {"path", "rows", "sha256", "source_sha256"}
        if role in NATIVE_ROLE_COUNTS:
            required.add("native_input_digest_list_sha256")
        if (
            not isinstance(entry, dict)
            or set(entry) != required
            or type(entry["rows"]) is not int
            or entry["rows"] != expected
            or not all(
                _is_sha(entry[key]) for key in required if key.endswith("sha256")
            )
            or not isinstance(entry["path"], str)
        ):
            raise ValueError("Projected role manifest differs")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError("Projected role path escapes manifest")
        path = manifest.parent / relative
        _pinned(path, entry["sha256"])
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line, object_pairs_hook=_unique_keys) for line in stream]
        roles[role] = rows
    validate_core_rows(roles)
    return roles


def native_segments(row: dict[str, Any]) -> tuple[str, list[str], str]:
    """Torch-free byte-identical rendition of the pinned production renderer."""

    def payload(value: Any) -> str:
        return value if isinstance(value, str) else canonical(value)

    prefix = (
        f"Context:\n{payload(row['state'])}\n\n"
        f"Task type: {row['task_type']}\nQuestion:\n{payload(row['instructions'])}\nOptions:"
    )
    options = [
        "\n<option>\n"
        + canonical({"key": option["key"], "description": option["description"]})
        + "\n</option>"
        for option in row["options"]
    ]
    suffix = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"
    return prefix, options, suffix


def verify_renderer_identity() -> None:
    source_path = Path(__file__).resolve().parents[1] / "model" / "decision_model.py"
    source = source_path.read_text(encoding="utf-8")
    segments = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "segments"
    )
    actual = _sha(ast.get_source_segment(source, segments))
    if actual != PRODUCTION_SEGMENTS_SHA256:
        raise ValueError("Production native renderer changed")
    if f'PROMPT_VERSION = "{PROMPT_VERSION}"' not in source:
        raise ValueError("Production native prompt version changed")


def read_train(source_dir: Path) -> tuple[list[Passage], dict[str, Any]]:
    """Read official TRAIN only; retain source labels solely for private review."""
    all_rows: list[Passage] = []
    ids: set[str] = set()
    answer_counts: collections.Counter[str] = collections.Counter()
    by_level: collections.Counter[str] = collections.Counter()
    questions_by_level: collections.Counter[str] = collections.Counter()
    empty_question_passages: collections.Counter[str] = collections.Counter()
    for level, expected in SOURCE_SPLITS.items():
        files = sorted((source_dir / "train" / level).glob("*.txt"))
        if len(files) != expected:
            raise ValueError("Official TRAIN passage count changed")
        for path in files:
            if path.is_symlink():
                raise ValueError("RACE source contains a symlink")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if set(payload) != {"article", "questions", "options", "answers", "id"}:
                raise ValueError("RACE TRAIN schema changed")
            source_id = payload["id"]
            article = payload["article"]
            questions = payload["questions"]
            options = payload["options"]
            answers = payload["answers"]
            if (
                not isinstance(source_id, str)
                or not source_id
                or source_id in ids
                or not isinstance(article, str)
                or not article.strip()
                or not isinstance(questions, list)
                or not isinstance(options, list)
                or not isinstance(answers, list)
            ):
                raise ValueError("Malformed RACE TRAIN passage")
            if not (len(questions) == len(options) == len(answers)):
                raise ValueError("RACE question/option/answer count mismatch")
            if not questions:
                # The pinned publisher archive contains two article files
                # without questions. They have no Choice target to admit.
                empty_question_passages[level] += 1
                ids.add(source_id)
                continue
            for question, slate, answer in zip(
                questions, options, answers, strict=True
            ):
                if (
                    not isinstance(question, str)
                    or not question.strip()
                    or not isinstance(slate, list)
                    or len(slate) != 4
                    or any(
                        not isinstance(option, str) or not option.strip()
                        for option in slate
                    )
                    or answer not in ("A", "B", "C", "D")
                ):
                    raise ValueError("Malformed RACE TRAIN question")
                answer_counts[answer] += 1
            ids.add(source_id)
            by_level[level] += 1
            questions_by_level[level] += len(questions)
            all_rows.append(
                Passage(
                    source_id=source_id,
                    level=level,
                    article=article,
                    questions=tuple(questions),
                    options=tuple(tuple(slate) for slate in options),
                    answers=tuple(answers),
                    group=_sha(_norm(article)),
                )
            )
    group_sizes = collections.Counter(row.group for row in all_rows)
    return all_rows, {
        "publisher_train_files": sum(SOURCE_SPLITS.values()),
        "passages": len(all_rows),
        "excluded_empty_question_passages": dict(empty_question_passages),
        "independent_normalized_passages": len(group_sizes),
        "repeated_article_groups": sum(count > 1 for count in group_sizes.values()),
        "passages_by_level": dict(by_level),
        "questions_by_level": dict(questions_by_level),
        "answer_positions": dict(sorted(answer_counts.items())),
    }


def choose_passages(rows: list[Passage], count_per_level: int) -> list[Passage]:
    """Select whole normalized article groups independent of answer labels."""
    chosen: list[Passage] = []
    for level in SOURCE_SPLITS:
        groups: dict[str, list[Passage]] = collections.defaultdict(list)
        for row in rows:
            if row.level == level:
                groups[row.group].append(row)
        selected_ids = sorted(
            groups,
            key=lambda group: _sha(f"{SAMPLE_SEED}\0{level}\0{group}"),
        )[:count_per_level]
        chosen.extend(row for group in selected_ids for row in groups[group])
    return chosen


def choice_row(
    passage: Passage, question_index: int, *, with_passage: bool
) -> dict[str, Any]:
    slate = passage.options[question_index]
    return {
        "id": f"race-train:{passage.level}:{passage.source_id}:{question_index}",
        "state": (
            passage.article
            if with_passage
            else "[Passage withheld for evidence-necessity review]"
        ),
        "instructions": passage.questions[question_index],
        "task_type": "choice",
        "options": [
            {"key": chr(ord("A") + index), "description": option}
            for index, option in enumerate(slate)
        ],
    }


def percentile(values: list[int], percent: int) -> int:
    values = sorted(values)
    return values[(percent * (len(values) - 1)) // 100]


def prompt_profile(rows: list[Passage], tokenizer: Any) -> dict[str, Any]:
    lengths: dict[str, list[int]] = collections.defaultdict(list)
    passage_lengths: dict[str, list[int]] = collections.defaultdict(list)
    question_counts: dict[str, int] = collections.Counter()
    fitting_groups: dict[str, int] = collections.Counter()
    question_shortcut_correct: dict[str, int] = collections.Counter()
    question_shortcut_determinate: dict[str, int] = collections.Counter()
    for passage in rows:
        passage_lengths[passage.level].append(
            len(tokenizer.encode(passage.article, add_special_tokens=False))
        )
        complete = True
        for index, answer in enumerate(passage.answers):
            row = choice_row(passage, index, with_passage=True)
            prefix, options, suffix = native_segments(row)
            length = sum(
                len(tokenizer.encode(text, add_special_tokens=False))
                for text in (prefix, *options, suffix)
            )
            lengths[passage.level].append(length)
            question_counts[passage.level] += 1
            complete &= length <= MAX_NATIVE_TOKENS
            # A deliberately weak lexical-only shortcut. It cannot establish
            # evidence necessity, but high accuracy would be an early HOLD.
            question_words = set(WORD.findall(_norm(passage.questions[index])))
            overlap = [
                len(question_words & set(WORD.findall(_norm(option))))
                for option in passage.options[index]
            ]
            if overlap.count(max(overlap)) == 1:
                question_shortcut_determinate[passage.level] += 1
                if overlap.index(max(overlap)) == ord(answer) - ord("A"):
                    question_shortcut_correct[passage.level] += 1
        fitting_groups[passage.level] += complete
    return {
        "prompt_version": PROMPT_VERSION,
        "cap": MAX_NATIVE_TOKENS,
        "truncation": "none",
        "by_level": {
            level: {
                "passages": len(passage_lengths[level]),
                "questions": question_counts[level],
                "article_tokens": {
                    "median": percentile(passage_lengths[level], 50),
                    "p90": percentile(passage_lengths[level], 90),
                    "max": max(passage_lengths[level]),
                },
                "full_native_tokens": {
                    "median": percentile(lengths[level], 50),
                    "p90": percentile(lengths[level], 90),
                    "p99": percentile(lengths[level], 99),
                    "max": max(lengths[level]),
                    "over_cap": sum(
                        value > MAX_NATIVE_TOKENS for value in lengths[level]
                    ),
                },
                "whole_passages_with_all_questions_in_cap": fitting_groups[level],
                "question_only_word_overlap_shortcut": {
                    "determinate": question_shortcut_determinate[level],
                    "correct": question_shortcut_correct[level],
                },
            }
            for level in SOURCE_SPLITS
        },
    }


def overlap_profile(
    rows: list[Passage], protected: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    source = [
        choice_row(passage, index, with_passage=True)
        for passage in rows
        for index in range(len(passage.questions))
    ]
    by_role = {
        role: full_input_overlap_rows(source, reference)
        for role, reference in sorted(protected.items())
    }
    blocked = any(any(value["counts"].values()) for value in by_role.values())
    return {
        "status": (
            "HOLD_OBSERVABLE_OVERLAP" if blocked else "PASS_BOUNDED_LEXICAL_SCREEN"
        ),
        "source_questions": len(source),
        "source_independent_passage_groups": len({row.group for row in rows}),
        "role_counts": {
            role: len(reference) for role, reference in sorted(protected.items())
        },
        "by_role": by_role,
        "limit": "Only deterministic sampled whole passage groups; no full RACE admission",
    }


def _write_private(path: Path, payload: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to overwrite private audit output")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def review_packet(
    rows: list[Passage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return matched passage-present/removed prompts and separate source key."""
    packet = []
    key = []
    for passage in rows:
        index = int(
            _sha(f"{SAMPLE_SEED}\0review-question\0{passage.group}")[:8], 16
        ) % len(passage.questions)
        review_id = _sha(f"{SAMPLE_SEED}\0review\0{passage.group}")[:16]
        present = choice_row(passage, index, with_passage=True)
        absent = choice_row(passage, index, with_passage=False)
        for label, row in (("present", present), ("removed", absent)):
            packet.append(
                {
                    "review_id": review_id,
                    "condition": label,
                    "state": row["state"],
                    "question": row["instructions"],
                    "options": row["options"],
                }
            )
        key.append({"review_id": review_id, "source_answer": passage.answers[index]})
    return packet, key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--protected-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError("Refusing to overwrite private audit directory")
    verify_renderer_identity()
    _pinned(args.archive, SOURCE_ARCHIVE_SHA256)
    _pinned(args.tokenizer_directory / "tokenizer.json", TOKENIZER_SHA256)
    source, source_profile = read_train(args.source_directory)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_directory, local_files_only=True, trust_remote_code=False
    )
    selected = choose_passages(source, SAMPLE_GROUPS_PER_STRATUM)
    review = choose_passages(selected, REVIEW_GROUPS_PER_STRATUM)
    protected = load_protected(args.protected_manifest)
    prompt = prompt_profile(selected, tokenizer)
    overlap = overlap_profile(selected, protected)
    packet, key = review_packet(review)
    args.output_directory.mkdir(parents=True, mode=0o700)
    os.chmod(args.output_directory, 0o700)
    _write_private(args.output_directory / "blind_review.json", packet)
    _write_private(args.output_directory / "review_key.json", key)
    aggregate = {
        "schema": "decision2-race06-choice-source-screen/1",
        "source_url": SOURCE_URL,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_role": "official TRAIN only",
        "terms": "noncommercial research; passage and derived-data commercial reuse restricted",
        "source_profile": source_profile,
        "sample_method": SAMPLE_SEED,
        "sample_passages": len(selected),
        "sample_independent_groups": len({row.group for row in selected}),
        "native_choice_prompt": prompt,
        "overlap": overlap,
        "protected_manifest_sha256": PROTECTED_MANIFEST_SHA256,
        "tokenizer_json_sha256": TOKENIZER_SHA256,
        "blind_review_pairs": len(key),
        "blind_review_packet_sha256": file_sha256(
            args.output_directory / "blind_review.json"
        ),
        "separate_review_key_sha256": file_sha256(
            args.output_directory / "review_key.json"
        ),
        "decision": "HOLD_PENDING_BLIND_EVIDENCE_NECESSITY_AND_RIGHTS; NO_TRAIN_ADMISSION",
        "gpu_hours": 0,
    }
    _write_private(args.output_directory / "aggregate.json", aggregate)
    print(
        json.dumps(
            {
                "status": aggregate["decision"],
                "sample_groups": aggregate["sample_independent_groups"],
                "protected_roles": len(protected),
                "aggregate_sha256": file_sha256(
                    args.output_directory / "aggregate.json"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
