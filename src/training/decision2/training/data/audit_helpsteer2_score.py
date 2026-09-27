"""Private, aggregate-only audit of pinned HelpSteer2 TRAIN for ordinal Score.

The source archive, protected prompts, blind packet and key remain in a private
remote scratch directory. Only aggregate findings belong in tracked research.
This module never reads the upstream validation split or evaluation gold.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

ATTRIBUTES = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
PILOT_PAIRS = ((0, 4), (1, 3), (2, 4), (0, 2), (1, 4), (2, 3))
CRITERIA = (
    "0: incorrect or does not answer the user's request",
    "1: mostly incorrect or materially incomplete",
    "2: partially correct, with significant omissions or errors",
    "3: mostly correct, with minor omissions or errors",
    "4: correct and sufficiently complete",
)
ALLOWED_PROTECTED_FIELDS = {
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


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def shingle(value: str) -> set[str]:
    words = re.findall(r"[^\W_]+|[^\w\s]", normalize(value), re.UNICODE)
    if len(words) < 5:
        return set(words)
    return {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}


def read_train(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"prompt", "response", *ATTRIBUTES}:
                raise ValueError("HelpSteer2 TRAIN schema changed")
            if not isinstance(row["prompt"], str) or not isinstance(
                row["response"], str
            ):
                raise ValueError("HelpSteer2 prompt/response must be strings")
            if any(
                type(row[key]) is not int or row[key] not in range(5)
                for key in ATTRIBUTES
            ):
                raise ValueError("HelpSteer2 ordinal label outside 0..4")
            rows.append(row)
    if not rows:
        raise ValueError("Empty HelpSteer2 TRAIN")
    return rows


def percentile(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[int(p * (len(ordered) - 1))]


def summarize(
    rows: list[dict[str, Any]], tokenizer: Any | None = None
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[normalize(row["prompt"])].append(row)
    histograms = {
        attr: {
            str(level): sum(row[attr] == level for row in rows) for level in range(5)
        }
        for attr in ATTRIBUTES
    }
    response_lengths = {}
    for level in range(5):
        values = [len(row["response"]) for row in rows if row["correctness"] == level]
        response_lengths[str(level)] = {
            "n": len(values),
            "median_chars": percentile(values, 0.5),
            "p90_chars": percentile(values, 0.9),
        }
    pair_hist = collections.Counter()
    distinct = 0
    longer_correct = 0
    length_ties = 0
    for group in groups.values():
        if len(group) == 2:
            pair_hist[tuple(sorted(row["correctness"] for row in group))] += 1
            a, b = group
            if a["correctness"] != b["correctness"]:
                distinct += 1
                lengths = (len(a["response"]), len(b["response"]))
                if lengths[0] == lengths[1]:
                    length_ties += 1
                elif (lengths[0] > lengths[1]) == (a["correctness"] > b["correctness"]):
                    longer_correct += 1
    result: dict[str, Any] = {
        "rows": len(rows),
        "normalized_prompt_groups": len(groups),
        "group_multiplicity": dict(
            sorted(collections.Counter(map(len, groups.values())).items())
        ),
        "duplicate_prompt_response_rows": len(rows)
        - len({(normalize(row["prompt"]), normalize(row["response"])) for row in rows}),
        "attribute_histograms": histograms,
        "correctness_response_chars": response_lengths,
        "correctness_pair_histogram": {
            f"{a}-{b}": count for (a, b), count in sorted(pair_hist.items())
        },
        "different_label_pairs": distinct,
        "longer_response_higher_correctness": longer_correct,
        "response_length_ties": length_ties,
        "prompt_chars": {
            "median": percentile([len(row["prompt"]) for row in rows], 0.5),
            "p90": percentile([len(row["prompt"]) for row in rows], 0.9),
            "p99": percentile([len(row["prompt"]) for row in rows], 0.99),
            "max": max(len(row["prompt"]) for row in rows),
        },
        "response_chars": {
            "median": percentile([len(row["response"]) for row in rows], 0.5),
            "p90": percentile([len(row["response"]) for row in rows], 0.9),
            "p99": percentile([len(row["response"]) for row in rows], 0.99),
            "max": max(len(row["response"]) for row in rows),
        },
    }
    if tokenizer is not None:
        token_lengths = [
            len(
                tokenizer.encode(
                    row["prompt"] + "\n" + row["response"], add_special_tokens=False
                )
            )
            for row in rows
        ]
        result["raw_prompt_response_tokens"] = {
            "median": percentile(token_lengths, 0.5),
            "p90": percentile(token_lengths, 0.9),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths),
            "over_8192": sum(length > 8192 for length in token_lengths),
            "total": sum(token_lengths),
        }
    return result


def choose_pilot(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[normalize(row["prompt"])].append(row)
    selected = []
    for pair in PILOT_PAIRS:
        for high_longer in (False, True):
            eligible = []
            for prompt, group in groups.items():
                if (
                    len(group) != 2
                    or tuple(sorted(r["correctness"] for r in group)) != pair
                ):
                    continue
                lengths = [len(r["response"]) for r in group]
                if not (
                    40 <= len(group[0]["prompt"]) <= 1200
                    and min(lengths) >= 100
                    and max(lengths) <= 2400
                    and max(lengths) / min(lengths) <= 1.5
                ):
                    continue
                low, high = sorted(group, key=lambda r: r["correctness"])
                if (len(high["response"]) > len(low["response"])) != high_longer:
                    continue
                if len(high["response"]) == len(low["response"]):
                    continue
                eligible.append((sha(prompt.encode()), prompt, group))
            if not eligible:
                raise ValueError(
                    f"No qualified pilot group for pair {pair}/{high_longer}"
                )
            _, prompt, group = min(eligible)
            selected.append(
                (prompt, sorted(group, key=lambda r: sha(r["response"].encode())))
            )
    if len({prompt for prompt, _ in selected}) != 12:
        raise ValueError("Pilot groups are not independent")
    return selected


def protected_text(row: dict[str, Any]) -> str:
    state = row.get("state")
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    raise ValueError("Protected row lacks textual state")


def reference_audit(
    rows: list[dict[str, Any]],
    pilot: list[tuple[str, list[dict[str, Any]]]],
    inventory: Path,
    extra_roles: list[tuple[str, Path]],
) -> dict[str, Any]:
    entries = json.loads(inventory.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Protected inventory is missing")
    candidates_exact = {
        normalize(row[field]) for row in rows for field in ("prompt", "response")
    }
    prompt_values = sorted({normalize(row["prompt"]) for row in rows})
    prompt_lookup = {value: index for index, value in enumerate(prompt_values)}
    source_group_sizes = collections.Counter(normalize(row["prompt"]) for row in rows)
    prompt_shingles = [shingle(value) for value in prompt_values]
    prompt_index: dict[str, list[int]] = collections.defaultdict(list)
    for index, shingles in enumerate(prompt_shingles):
        for gram in shingles:
            prompt_index[gram].append(index)
    pilot_values = [
        value
        for prompt, group in pilot
        for value in (prompt, *(row["response"] for row in group))
    ]
    pilot_shingles = [shingle(value) for value in pilot_values]
    report = {}
    for item in entries + [
        {"role": role, "path": str(path), "sha256": file_sha(path), "extra": True}
        for role, path in extra_roles
    ]:
        role, path = item["role"], Path(item["path"])
        if role in report or not path.is_file() or file_sha(path) != item["sha256"]:
            raise ValueError(f"Missing or changed protected role: {role}")
        exact, near, full_prompt_near, count = 0, 0, 0, 0
        matched_prompt_groups: set[int] = set()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if (
                    not item.get("extra")
                    and not set(record) <= ALLOWED_PROTECTED_FIELDS
                ):
                    raise ValueError(
                        f"Protected role contains non-prompt fields: {role}"
                    )
                text = protected_text(record)
                count += 1
                normalized = normalize(text)
                exact += normalized in candidates_exact
                if normalized in prompt_lookup:
                    matched_prompt_groups.add(prompt_lookup[normalized])
                shingles = shingle(text)
                if shingles:
                    # The rare-gram index is a retrieval heuristic. Every
                    # candidate is checked with exact five-gram Jaccard;
                    # source-wide near-overlap recall is not guaranteed.
                    grams = sorted(
                        (gram for gram in shingles if gram in prompt_index),
                        key=lambda gram: len(prompt_index[gram]),
                    )
                    candidate_indices = {
                        index
                        for gram in grams[:16]
                        for index in prompt_index.get(gram, ())
                    }
                    row_near = False
                    for index in candidate_indices:
                        other = prompt_shingles[index]
                        if (
                            other
                            and min(len(other), len(shingles))
                            / max(len(other), len(shingles))
                            >= 0.6
                            and len(other & shingles) / len(other | shingles) >= 0.6
                        ):
                            row_near = True
                            matched_prompt_groups.add(index)
                    full_prompt_near += row_near
                    for other in pilot_shingles:
                        if (
                            not other
                            or min(len(other), len(shingles))
                            / max(len(other), len(shingles))
                            < 0.6
                        ):
                            continue
                        if len(other & shingles) / len(other | shingles) >= 0.6:
                            near += 1
                            break
        report[role] = {
            "rows": count,
            "source_sha256": item["sha256"],
            "full_train_exact_state_matches": exact,
            "full_train_near_prompt_matches_heuristic": full_prompt_near,
            "full_train_near_prompt_groups_heuristic": len(matched_prompt_groups),
            "full_train_near_prompt_source_rows_heuristic": sum(
                source_group_sizes[prompt_values[index]]
                for index in matched_prompt_groups
            ),
            "pilot_near_state_matches": near,
        }
    required = {
        "typed_dev",
        "css_pilot",
        "typed_final_goldfree",
        "css15_goldfree",
        "jevbench_public231",
    }
    if not required <= set(report):
        raise ValueError("Missing core protected roles")
    return report


def private_pilot_packets(
    pilot: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[bytes, bytes]:
    blind, key = [], []
    for prompt, group in pilot:
        group_id = sha(prompt.encode())[:24]
        for row in group:
            record_id = sha((prompt + "\0" + row["response"]).encode())[:24]
            blind.append(
                {
                    "id": record_id,
                    "group_id": group_id,
                    "state": {
                        "user_request": prompt,
                        "candidate_response": row["response"],
                    },
                    "questions": {
                        "correctness": {
                            "type": "score",
                            "instructions": "Rate the response's correctness for the user request.",
                            "criteria": list(CRITERIA),
                        }
                    },
                }
            )
            key.append(
                {
                    "id": record_id,
                    "group_id": group_id,
                    "correctness": row["correctness"],
                }
            )
    return (
        (
            "\n".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) for row in blind
            )
            + "\n"
        ).encode(),
        (
            "\n".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) for row in key
            )
            + "\n"
        ).encode(),
    )


def write_private(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--reference", action="append", default=[], help="role=path")
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() or file_sha(args.train) != args.expected_train_sha256:
        raise ValueError("Output exists or pinned source changed")
    tokenizer = None
    if args.tokenizer_dir:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False
        )
    rows = read_train(args.train)
    pilot = choose_pilot(rows)
    extras = []
    for value in args.reference:
        role, sep, path = value.partition("=")
        if not sep or not role:
            raise ValueError("Reference must be role=path")
        extras.append((role, Path(path)))
    summary = summarize(rows, tokenizer)
    summary["source_sha256"] = args.expected_train_sha256
    summary["protected"] = reference_audit(
        rows, pilot, args.protected_inventory, extras
    )
    blind, key = private_pilot_packets(pilot)
    summary["pilot"] = {
        "groups": len(pilot),
        "rows": 24,
        "blind_sha256": sha(blind),
        "key_sha256": sha(key),
        "label_histogram": dict(
            sorted(
                collections.Counter(
                    r["correctness"] for _, group in pilot for r in group
                ).items()
            )
        ),
        "status": "HOLD_INDEPENDENT_REVIEW_AND_SOURCE_SEMANTICS",
    }
    args.out_dir.mkdir(mode=0o700, parents=True)
    write_private(args.out_dir / "blind.jsonl", blind)
    write_private(args.out_dir / "key.jsonl", key)
    write_private(
        args.out_dir / "aggregate.json",
        (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k
                not in {
                    "protected",
                    "attribute_histograms",
                    "correctness_pair_histogram",
                }
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
