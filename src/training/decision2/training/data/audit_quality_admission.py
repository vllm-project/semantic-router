"""Aggregate-only, gold-free admission audit of official QuALITY TRAIN.

Only source TRAIN labels are read for a fixed shortcut diagnostic. Protected
roles are verified and projected to input text before comparison. No source or
protected row, article ID, answer, URL or individual match is emitted.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from training.data.audit_quality_source import REVISION, license_kind, norm
from training.model.data import file_sha256

SOURCE_SHA256 = {
    "train": "4011e9952d5395beb8ff7637b963481a400630c1bbe2f40dc0d83f5d59f926ed",
    "dev": "99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef",
    "test": "ca103a953741c56888124a14958460b07941ee40a844914d39e851f1c3099897",
}
EXPECTED_INVENTORY_SHA256 = (
    "26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1"
)
CORE_ROLE_COUNTS = {
    "typed_dev": 1600,
    "css_pilot": 1430,
    "typed_final_goldfree": 1600,
    "css15_goldfree": 6547,
    "jevbench_public231": 231,
    "rights_clean_train": 7455,
    "rights_clean_select": 700,
    "rights_clean_cal": 700,
}
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokens(value: str) -> list[str]:
    return WORD_RE.findall(norm(value))


def _grams(value: str, size: int = 5) -> set[str]:
    words = _tokens(value)
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Input must be an ordinary file")
    with path.open(encoding="utf-8") as stream:
        result = [json.loads(line) for line in stream if line.strip()]
    if not result or any(not isinstance(row, dict) for row in result):
        raise ValueError("Input JSONL is empty or malformed")
    return result


def _source_articles(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        != REVISION
    ):
        raise ValueError("Official source revision changed")
    articles: dict[str, dict[str, dict[str, Any]]] = {}
    for split, expected in SOURCE_SHA256.items():
        path = root / "data" / "v1.0.1" / f"QuALITY.v1.0.1.htmlstripped.{split}"
        if file_sha256(path) != expected:
            raise ValueError("Official source bytes changed")
        groups: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(path):
            article_id = row["article_id"]
            if not isinstance(article_id, str) or not article_id:
                raise ValueError("Missing article ID")
            metadata = {
                key: row[key]
                for key in (
                    "article",
                    "title",
                    "author",
                    "year",
                    "url",
                    "license",
                    "source",
                )
            }
            if article_id not in groups:
                groups[article_id] = {"metadata": metadata, "questions": []}
            elif groups[article_id]["metadata"] != metadata:
                raise ValueError("Writer records disagree about source article")
            groups[article_id]["questions"].extend(row["questions"])
        if any(not group["questions"] for group in groups.values()):
            raise ValueError("Article group lacks questions")
        articles[split] = groups
    return articles


def _near_matches(
    left: dict[str, str],
    right: dict[str, str],
    *,
    threshold: float = 0.80,
    skip_identical_keys: bool = False,
) -> set[str]:
    """Find source groups with near-identical full texts via 5-gram overlap.

    Shared-gram retrieval is exact for Jaccard >= threshold when both sides
    have >=5 words. It does not establish semantic non-overlap.
    """
    right_grams = {key: _grams(text) for key, text in right.items()}
    index: dict[str, set[str]] = collections.defaultdict(set)
    for key, grams in right_grams.items():
        for gram in grams:
            index[gram].add(key)
    matched: set[str] = set()
    for key, text in left.items():
        grams = _grams(text)
        if not grams:
            continue
        candidates: collections.Counter[str] = collections.Counter()
        for gram in grams:
            candidates.update(index.get(gram, ()))
        for other, intersection in candidates.items():
            if skip_identical_keys and key == other:
                continue
            union = len(grams) + len(right_grams[other]) - intersection
            if union and intersection / union >= threshold:
                matched.add(key)
    return matched


def _protected_roles(
    manifest: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if file_sha256(manifest) != EXPECTED_INVENTORY_SHA256:
        raise ValueError("Strict core inventory manifest changed")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(entries, dict)
        or not isinstance(entries.get("roles"), dict)
        or set(entries["roles"]) != set(CORE_ROLE_COUNTS)
        or entries.get("excluded_optional_role_count") != 27
    ):
        raise ValueError("Expected eight pinned projected core roles")
    roles: dict[str, list[dict[str, Any]]] = {}
    hashes = {}
    for role, entry in entries["roles"].items():
        relative = Path(entry["path"])
        if relative.is_absolute() or relative.name != relative.as_posix():
            raise ValueError("Projected role path must be a local filename")
        path, expected = manifest.parent / relative, entry["sha256"]
        if file_sha256(path) != expected:
            raise ValueError("Projected role bytes changed")
        rows = _read_jsonl(path)
        if len(rows) != CORE_ROLE_COUNTS[role] or entry["rows"] != len(rows):
            raise ValueError("Projected role count changed")
        if any(
            not set(row) <= {"id", "state", "instructions", "options"}
            or not isinstance(row.get("id"), str)
            or not isinstance(row.get("state"), str)
            or not isinstance(row.get("instructions"), str)
            or any(not isinstance(value, str) for value in row.values())
            for row in rows
        ):
            raise ValueError("Projected role has non-input or non-text field")
        roles[role] = rows
        hashes[role] = expected
    return roles, hashes


def _containment(
    articles: dict[str, str], roles: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, dict[str, int]], set[str]]:
    """Conservatively flag long-article excerpts in any protected input.

    A 5-word-shingle index finds exact shared phrases; complete-state Jaccard
    >=.60 or shorter-state containment >=.80 with >=12 shared shingles flags a
    group. This is a lexical gate, not proof of semantic disjointness.
    """
    source_grams = {group: _grams(text) for group, text in articles.items()}
    source_exact = {norm(text): group for group, text in articles.items()}
    index: dict[str, set[str]] = collections.defaultdict(set)
    for group, grams in source_grams.items():
        for gram in grams:
            index[gram].add(group)
    report: dict[str, dict[str, int]] = {}
    flagged: set[str] = set()

    def near_groups(text: str) -> set[str]:
        grams = _grams(text)
        if len(grams) < 12:
            return set()
        candidates: collections.Counter[str] = collections.Counter()
        for gram in grams:
            candidates.update(index.get(gram, ()))
        matches = set()
        for group, intersection in candidates.items():
            if intersection < 12:
                continue
            denominator = min(len(grams), len(source_grams[group]))
            union = len(grams) + len(source_grams[group]) - intersection
            if intersection / denominator >= 0.80 or intersection / union >= 0.60:
                matches.add(group)
        return matches

    for role, rows in sorted(roles.items()):
        exact_states = 0
        near_states = 0
        near_full = 0
        hit_groups: set[str] = set()
        for row in rows:
            state = row["state"]
            normalized = norm(state)
            if normalized in source_exact:
                exact_states += 1
                hit_groups.add(source_exact[normalized])
            state_hits = near_groups(state)
            near_states += bool(state_hits)
            hit_groups.update(state_hits)
            if not state_hits:
                full_text = "\n".join(
                    (state, row.get("instructions", ""), row.get("options", ""))
                )
                full_hits = near_groups(full_text)
                near_full += bool(full_hits)
                hit_groups.update(full_hits)
        flagged |= hit_groups
        report[role] = {
            "rows": len(rows),
            "exact_article_state_rows": exact_states,
            "near_or_excerpt_state_rows": near_states,
            "near_or_excerpt_full_input_only_rows": near_full,
            "quarantined_article_groups": len(hit_groups),
        }
    return report, flagged


def _question_surface_overlap(
    groups: dict[str, dict[str, Any]], roles: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, dict[str, int]], set[str]]:
    """Match question and option input leaves, independent of article text."""
    exact_questions: dict[str, set[str]] = collections.defaultdict(set)
    question_grams: dict[str, set[str]] = {}
    near_index: dict[str, set[str]] = collections.defaultdict(set)
    for group_id, item in groups.items():
        for question in item["questions"]:
            payload = norm(question["question"])
            exact_questions[payload].add(group_id)
            if payload not in question_grams:
                grams = _grams(payload, 3)
                question_grams[payload] = grams
                for gram in grams:
                    near_index[gram].add(payload)
    report = {}
    held: set[str] = set()
    for role, rows in sorted(roles.items()):
        role_groups: set[str] = set()
        exact_rows = 0
        near_rows = 0
        for row in rows:
            instruction = row.get("instructions", "")
            try:
                parsed = json.loads(instruction)
            except (ValueError, TypeError):
                parsed = instruction
            leaves = [norm(value) for value in _string_leaves(parsed)]
            exact_hit = False
            near_hit = False
            for leaf in leaves:
                exact_groups = exact_questions.get(leaf, ())
                if exact_groups:
                    exact_hit = True
                    role_groups.update(exact_groups)
                    continue
                grams = _grams(leaf, 3)
                if len(grams) < 4:
                    continue
                candidates: collections.Counter[str] = collections.Counter()
                for gram in grams:
                    candidates.update(near_index.get(gram, ()))
                for candidate, intersection in candidates.items():
                    other = question_grams[candidate]
                    union = len(grams) + len(other) - intersection
                    if union and intersection / union >= 0.90:
                        near_hit = True
                        role_groups.update(exact_questions[candidate])
            exact_rows += exact_hit
            near_rows += near_hit
        held |= role_groups
        report[role] = {
            "exact_question_text_rows": exact_rows,
            "near_question_text_rows": near_rows,
            "quarantined_article_groups": len(role_groups),
        }
    return report, held


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in _string_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _string_leaves(child)]
    return []


def _fixed_shortcut_sample(groups: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Frozen 24-article diagnostic; it cannot certify evidence necessity."""
    ordered = sorted(groups, key=_hash)[:24]
    if len(ordered) != 24:
        raise ValueError("Insufficient independent article groups")
    answer_in_question = 0
    longest_option_hits = 0
    sample_questions = 0
    for group_id in ordered:
        questions = groups[group_id]["questions"]
        question = min(questions, key=lambda row: _hash(row["question_unique_id"]))
        gold = question["gold_label"]
        if type(gold) is not int or gold not in range(1, 5):
            raise ValueError("TRAIN shortcut sample lacks source key")
        option = question["options"][gold - 1]
        answer_in_question += bool(
            norm(option) and norm(option) in norm(question["question"])
        )
        lengths = [len(_tokens(choice)) for choice in question["options"]]
        longest_option_hits += (
            lengths[gold - 1] == max(lengths) and lengths.count(max(lengths)) == 1
        )
        sample_questions += 1
    return {
        "independent_articles": len(ordered),
        "questions": sample_questions,
        "gold_option_verbatim_in_question": answer_in_question,
        "longest_unique_option_correct": longest_option_hits,
        "limits": "These are shortcut indicators only; independent answer-blind article ablation and evidence review remain required.",
    }


def _write_private_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Private packet is empty")
    content = (
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
        + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
    return hashlib.sha256(content).hexdigest()


def _private_review_artifacts(
    groups: dict[str, dict[str, Any]],
    rights_ledger: Path,
    blind_packet: Path,
    blind_key: Path,
) -> dict[str, Any]:
    paths = (rights_ledger, blind_packet, blind_key)
    if len(set(paths)) != 3 or any(path.exists() for path in paths):
        raise ValueError("Private review output paths must be distinct and new")
    ledger = []
    for group_id, item in sorted(groups.items(), key=lambda pair: _hash(pair[0])):
        metadata = item["metadata"]
        ledger.append(
            {
                "article_id_sha256": _hash(group_id),
                "source": metadata["source"],
                "license_bucket": license_kind(metadata["license"]),
                "license_statement": metadata["license"],
                "title": metadata["title"],
                "author": metadata["author"],
                "year": metadata["year"],
                "url": metadata["url"],
                "rights_status": "PENDING_PER_ARTICLE_REVIEW",
            }
        )
    blind = []
    key = []
    for group_id in sorted(groups, key=_hash)[:24]:
        item = groups[group_id]
        question = min(
            item["questions"], key=lambda row: _hash(row["question_unique_id"])
        )
        for condition in ("article_available", "article_removed"):
            review_id = _hash(
                group_id + "\0" + question["question_unique_id"] + "\0" + condition
            )[:24]
            blind.append(
                {
                    "review_id": review_id,
                    "condition": condition,
                    "state": (
                        item["metadata"]["article"]
                        if condition == "article_available"
                        else ""
                    ),
                    "question": question["question"],
                    "options": question["options"],
                    "review_request": "Choose A/B/C/D or mark unanswerable; cite decisive article evidence when available.",
                }
            )
            key.append(
                {
                    "review_id": review_id,
                    "article_id_sha256": _hash(group_id),
                    "source_gold_position": question["gold_label"],
                }
            )
    blind.sort(key=lambda row: _hash(row["review_id"]))
    key.sort(key=lambda row: _hash(row["review_id"]))
    return {
        "rights_ledger_articles": len(ledger),
        "rights_ledger_sha256": _write_private_jsonl(rights_ledger, ledger),
        "blind_review_rows": len(blind),
        "blind_packet_sha256": _write_private_jsonl(blind_packet, blind),
        "separate_key_sha256": _write_private_jsonl(blind_key, key),
    }


def audit(
    root: Path,
    manifest: Path,
    private_outputs: tuple[Path, Path, Path] | None = None,
) -> dict[str, Any]:
    source = _source_articles(root)
    train = source["train"]
    other = {**source["dev"], **source["test"]}
    if len(other) != len(source["dev"]) + len(source["test"]):
        raise ValueError("DEV/TEST article ID collision")
    title_hashes = {norm(group["metadata"]["title"]) for group in other.values()}
    title_held = {
        group_id
        for group_id, group in train.items()
        if norm(group["metadata"]["title"]) in title_hashes
    }
    near_held = _near_matches(
        {key: value["metadata"]["article"] for key, value in train.items()},
        {key: value["metadata"]["article"] for key, value in other.items()},
    )
    train_title_to_ids: dict[str, set[str]] = collections.defaultdict(set)
    for group_id, group in train.items():
        train_title_to_ids[norm(group["metadata"]["title"])].add(group_id)
    same_split_title_held = {
        group_id
        for ids in train_title_to_ids.values()
        if len(ids) > 1
        for group_id in ids
    }
    same_split_near_held = _near_matches(
        {key: value["metadata"]["article"] for key, value in train.items()},
        {key: value["metadata"]["article"] for key, value in train.items()},
        skip_identical_keys=True,
    )
    source_hold = title_held | near_held | same_split_title_held | same_split_near_held
    candidate = {key: value for key, value in train.items() if key not in source_hold}
    rights = collections.Counter()
    missing_credit = collections.Counter()
    for group in candidate.values():
        metadata = group["metadata"]
        kind = license_kind(metadata["license"])
        rights[kind] += 1
        for field in ("title", "author", "year", "url"):
            missing_credit[field] += not bool(str(metadata[field]).strip())
    roles, role_hashes = _protected_roles(manifest)
    articles = {key: value["metadata"]["article"] for key, value in candidate.items()}
    context, context_held = _containment(articles, roles)
    question, question_held = _question_surface_overlap(candidate, roles)
    overlap_held = context_held | question_held
    after_overlap = set(candidate) - overlap_held
    result = {
        "schema": "decision2-quality-admission-cpu/1",
        "official_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "manifest_sha256": file_sha256(manifest),
        "protected_role_sha256": role_hashes,
        "source_independence": {
            "train_article_groups": len(train),
            "title_collision_hold_groups": len(title_held),
            "near_article_cross_split_hold_groups": len(near_held),
            "same_split_title_hold_groups": len(same_split_title_held),
            "same_split_near_article_hold_groups": len(same_split_near_held),
            "union_hold_groups": len(source_hold),
            "candidate_before_rights_or_protected": len(candidate),
        },
        "article_rights_screen": {
            "candidate_groups_by_source_license_bucket": dict(sorted(rights.items())),
            "missing_attribution_fields": dict(sorted(missing_credit.items())),
            "article_level_clearance": "PENDING_INDIVIDUAL_WORK_REVIEW",
        },
        "protected_overlap": {
            "roles": len(roles),
            "row_comparisons": sum(len(rows) for rows in roles.values()),
            "by_role_context": context,
            "by_role_question_options": question,
            "union_quarantined_article_groups": len(overlap_held),
            "remaining_article_groups_before_rights_and_evidence_review": len(
                after_overlap
            ),
            "excluded_optional_role_count": 27,
            "limits": "Five-gram lexical near/excerpt search and exact question-option hashes do not prove semantic non-overlap.",
        },
        "fixed_shortcut_sample": _fixed_shortcut_sample(candidate),
        "admitted_train_rows": 0,
        "decision": "HOLD: article-level rights and answer-blind evidence necessity review remain incomplete; no GPU arm authorized",
    }
    if private_outputs is not None:
        result["private_review_artifacts"] = _private_review_artifacts(
            candidate, *private_outputs
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--protected-inventory", required=True, type=Path)
    parser.add_argument("--private-rights-ledger", type=Path)
    parser.add_argument("--private-blind-packet", type=Path)
    parser.add_argument("--private-blind-key", type=Path)
    args = parser.parse_args()
    outputs = (
        args.private_rights_ledger,
        args.private_blind_packet,
        args.private_blind_key,
    )
    if any(path is None for path in outputs) and any(
        path is not None for path in outputs
    ):
        raise ValueError("All three private review outputs are required together")
    result = audit(
        args.source_root,
        args.protected_inventory,
        outputs if all(path is not None for path in outputs) else None,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
