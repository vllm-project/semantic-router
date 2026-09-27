"""CPU-only, aggregate audit of publisher-pinned NLI TRAIN as a Score hypothesis.

Input text, source record identities and protected prompts must remain in a
private experiment directory. This module never opens publisher dev/test or
protected answer keys and does not create training examples.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

LABELS = ("contradiction", "neutral", "entailment")
SNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
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


@dataclass(frozen=True)
class Pair:
    premise: str
    hypothesis: str
    label: str
    group: str
    genre: str
    position: int


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def compact(value: str) -> str:
    return "".join(char for char in normalize(value) if char.isalnum())


def grams(value: str) -> set[str]:
    value = compact(value)
    if len(value) < 20:
        return set()
    return {value[i : i + 6] for i in range(len(value) - 5)}


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def read_snli(path: Path) -> tuple[list[Pair], dict[str, Any]]:
    table = pq.read_table(path)
    if table.column_names != ["premise", "hypothesis", "label"]:
        raise ValueError("SNLI TRAIN schema changed")
    pairs = []
    invalid = 0
    for index, (premise, hypothesis, raw_label) in enumerate(
        zip(*(table[name].to_pylist() for name in table.column_names), strict=True)
    ):
        if raw_label == -1:
            invalid += 1
            continue
        if raw_label not in SNLI_LABELS or not premise or not hypothesis:
            raise ValueError("SNLI TRAIN has an unexpected label or missing text")
        pairs.append(
            Pair(
                premise,
                hypothesis,
                SNLI_LABELS[raw_label],
                normalize(premise),
                "caption",
                index,
            )
        )
    return pairs, {"raw_rows": table.num_rows, "excluded_no_consensus": invalid}


def read_ocnli(path: Path) -> tuple[list[Pair], dict[str, Any]]:
    pairs = []
    invalid = 0
    missing_provenance = 0
    duplicate_ids = 0
    seen_ids: set[int] = set()
    group_premises: dict[str, set[str]] = collections.defaultdict(set)
    premise_groups: dict[str, set[str]] = collections.defaultdict(set)
    parents: dict[str, str] = {}

    def root(group: str) -> str:
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    def union(left: str, right: str) -> None:
        left, right = root(left), root(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    with path.open(encoding="utf-8") as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            needed = {"sentence1", "sentence2", "label", "genre", "prem_id", "id"}
            if not needed <= set(row):
                raise ValueError("OCNLI publisher TRAIN schema changed")
            if row["id"] in seen_ids:
                duplicate_ids += 1
            seen_ids.add(row["id"])
            label = row["label"]
            if label == "-":
                invalid += 1
                continue
            if not isinstance(row["genre"], str) or not isinstance(row["prem_id"], str):
                missing_provenance += 1
                continue
            if label not in LABELS or not row["sentence1"] or not row["sentence2"]:
                raise ValueError("OCNLI TRAIN has an unexpected label or missing text")
            group = f'{row["genre"]}:{row["prem_id"]}'
            prem = normalize(row["sentence1"])
            parents.setdefault(group, group)
            group_premises[group].add(prem)
            for other_group in premise_groups[prem]:
                union(group, other_group)
            premise_groups[prem].add(group)
            pairs.append(
                Pair(
                    row["sentence1"],
                    row["sentence2"],
                    label,
                    group,
                    row["genre"],
                    position,
                )
            )
    pairs = [
        Pair(
            row.premise,
            row.hypothesis,
            row.label,
            root(row.group),
            row.genre,
            row.position,
        )
        for row in pairs
    ]
    return pairs, {
        "raw_rows": len(pairs) + invalid + missing_provenance,
        "excluded_no_consensus": invalid,
        "excluded_missing_genre_or_premise_id": missing_provenance,
        "duplicate_publisher_ids": duplicate_ids,
        "premise_ids_with_multiple_texts": sum(
            len(texts) > 1 for texts in group_premises.values()
        ),
        "source_ids_sharing_premise_text": len(
            {
                group
                for groups in premise_groups.values()
                if len(groups) > 1
                for group in groups
            }
        ),
        "premise_id_groups_before_components": len(parents),
    }


def source_summary(pairs: list[Pair], tokenizer: Any | None = None) -> dict[str, Any]:
    group_sizes = collections.Counter(pair.group for pair in pairs)
    pair_keys = {
        (normalize(pair.premise), normalize(pair.hypothesis)) for pair in pairs
    }
    class_counts = collections.Counter(pair.label for pair in pairs)
    genres = collections.Counter(pair.genre for pair in pairs)
    genre_class = collections.defaultdict(collections.Counter)
    for pair in pairs:
        genre_class[pair.genre][pair.label] += 1
    lengths = [len(pair.premise) + len(pair.hypothesis) for pair in pairs]
    result = {
        "eligible_rows": len(pairs),
        "independent_premise_groups": len(group_sizes),
        "group_multiplicity": {
            "median": percentile(list(group_sizes.values()), 0.5),
            "p90": percentile(list(group_sizes.values()), 0.9),
            "p99": percentile(list(group_sizes.values()), 0.99),
            "max": max(group_sizes.values()),
        },
        "exact_duplicate_pair_rows": len(pairs) - len(pair_keys),
        "class_counts": {label: class_counts[label] for label in LABELS},
        "genre_counts": dict(sorted(genres.items())),
        "genre_class_counts": {
            genre: {label: genre_class[genre][label] for label in LABELS}
            for genre in sorted(genres)
        },
        "pair_chars": {
            "median": percentile(lengths, 0.5),
            "p90": percentile(lengths, 0.9),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths),
        },
    }
    if tokenizer is not None:
        sample = sorted(
            pairs,
            key=lambda row: hashlib.sha256(
                (row.group + "\0" + row.hypothesis).encode()
            ).digest(),
        )[:10_000]
        token_lengths = [
            len(
                tokenizer.encode(
                    row.premise + "\n" + row.hypothesis, add_special_tokens=False
                )
            )
            for row in sample
        ]
        result["raw_pair_tokens_sample"] = {
            "n": len(sample),
            "median": percentile(token_lengths, 0.5),
            "p90": percentile(token_lengths, 0.9),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths),
            "sampling": "lowest SHA256(group + NUL + hypothesis)",
            "scope": "raw pair, before System One prompt wrapping",
        }
    return result


def text_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for child in value.values() for leaf in text_leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in text_leaves(child)]
    return []


def read_protected(
    inventory: Path, extra_roles: list[tuple[str, Path]]
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    entries = json.loads(inventory.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Protected inventory missing")
    all_entries = entries + [
        {"role": role, "path": str(path), "sha256": sha_file(path), "extra": True}
        for role, path in extra_roles
    ]
    seen_roles: set[str] = set()
    texts: list[tuple[str, str]] = []
    role_counts = {}
    for entry in all_entries:
        role = entry["role"]
        path = Path(entry["path"])
        if (
            role in seen_roles
            or not path.is_file()
            or sha_file(path) != entry["sha256"]
        ):
            raise ValueError(f"Missing, changed or duplicate role: {role}")
        seen_roles.add(role)
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not entry.get("extra") and not set(row) <= PROTECTED_FIELDS:
                    raise ValueError(
                        f"Protected role includes non-prompt field: {role}"
                    )
                count += 1
                for field in ("state", "instructions", "options"):
                    for leaf in text_leaves(row.get(field)):
                        normalized = normalize(leaf)
                        if len(compact(normalized)) >= 20:
                            texts.append((role, normalized))
        role_counts[role] = {"rows": count, "sha256": entry["sha256"]}
    required = {
        "typed_dev",
        "css_pilot",
        "typed_final_goldfree",
        "css15_goldfree",
        "jevbench_public231",
        "rights_clean_train",
        "rights_clean_select",
        "rights_clean_cal",
    }
    if not required <= seen_roles:
        raise ValueError("Core gold-free and rights-clean roles missing")
    return texts, {"roles": role_counts, "inventory_sha256": sha_file(inventory)}


def overlap_screen(
    pairs: list[Pair], protected: list[tuple[str, str]]
) -> dict[str, Any]:
    # The lexical near scan is intentionally a recall-limited screen, not a
    # semantic deduper. Exact matches inspect every normalized source string.
    exact_roles: dict[str, set[str]] = collections.defaultdict(set)
    snippets: dict[str, set[str]] = collections.defaultdict(set)
    too_long = 0
    for role, text in protected:
        text = normalize(text)
        exact_roles[text].add(role)
        if len(text) <= 800:
            snippets[text].add(role)
        else:
            too_long += 1
    references = list(snippets)
    ref_grams = [grams(ref) for ref in references]
    index: dict[str, list[int]] = collections.defaultdict(list)
    for index_id, gram_set in enumerate(ref_grams):
        for gram in sorted(gram_set)[::2]:
            index[gram].append(index_id)
    source_text_groups: dict[str, set[str]] = collections.defaultdict(set)
    for pair in pairs:
        for field in (pair.premise, pair.hypothesis):
            source_text_groups[normalize(field)].add(pair.group)
    hits = collections.defaultdict(lambda: {"exact": set(), "near": set()})
    hashed_matches = []
    for source_text, groups in source_text_groups.items():
        exact = exact_roles.get(source_text, set())
        for role in exact:
            hits[role]["exact"].update(groups)
            hashed_matches.append(
                (
                    role,
                    "exact",
                    hashlib.sha256(source_text.encode()).hexdigest(),
                    len(groups),
                )
            )
        source_grams = grams(source_text)
        if not source_grams:
            continue
        rare = sorted(
            (gram for gram in source_grams if gram in index),
            key=lambda gram: (len(index[gram]), gram),
        )[:6]
        candidate_ids = {
            candidate
            for gram in rare
            if len(index[gram]) <= 128
            for candidate in index[gram]
        }
        for candidate in candidate_ids:
            other = ref_grams[candidate]
            if not other:
                continue
            common = len(source_grams & other)
            # Jaccard protects against generic phrases; containment catches a
            # short source sentence quoted inside a longer protected state.
            jaccard = common / len(source_grams | other)
            contained = common / min(len(source_grams), len(other))
            if jaccard < 0.72 and contained < 0.90:
                continue
            for role in snippets[references[candidate]]:
                if role in exact:
                    continue
                hits[role]["near"].update(groups)
                hashed_matches.append(
                    (
                        role,
                        "near",
                        hashlib.sha256(source_text.encode()).hexdigest(),
                        len(groups),
                    )
                )
    return {
        "role_group_counts": {
            role: {kind: len(group_set) for kind, group_set in role_hits.items()}
            for role, role_hits in sorted(hits.items())
        },
        "matched_source_group_count": (
            len(
                set().union(
                    *[
                        group_set
                        for role_hits in hits.values()
                        for group_set in role_hits.values()
                    ]
                )
            )
            if hits
            else 0
        ),
        "protected_snippets": len(protected),
        "protected_snippets_over_800_chars_not_near_scanned": too_long,
        "distinct_source_texts": len(source_text_groups),
        "near_method": "rare compact six-character grams, <=800-character protected leaves; Jaccard>=.72 or containment>=.90; heuristic",
        "hashed_matches": sorted(set(hashed_matches)),
    }


def shortcut_screen(pairs: list[Pair], cap_train: int, cap_test: int) -> dict[str, Any]:
    train = sorted(
        (
            row
            for row in pairs
            if int(hashlib.sha256(row.group.encode()).hexdigest(), 16) % 5
        ),
        key=lambda row: hashlib.sha256(
            (row.group + "\0" + row.hypothesis).encode()
        ).digest(),
    )[:cap_train]
    test = sorted(
        (
            row
            for row in pairs
            if not int(hashlib.sha256(row.group.encode()).hexdigest(), 16) % 5
        ),
        key=lambda row: hashlib.sha256(
            (row.group + "\0" + row.hypothesis).encode()
        ).digest(),
    )[:cap_test]
    if (
        not train
        or not test
        or {row.group for row in train} & {row.group for row in test}
    ):
        raise ValueError("Shortcut train/test source groups overlap or are empty")
    y_train = [LABELS.index(row.label) for row in train]
    y_test = [LABELS.index(row.label) for row in test]
    majority = collections.Counter(y_train).most_common(1)[0][0]
    result: dict[str, Any] = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_premise_groups": len({row.group for row in train}),
        "test_premise_groups": len({row.group for row in test}),
        "majority_accuracy": round(sum(y == majority for y in y_test) / len(y_test), 4),
        "feature": "hypothesis only, binary char 2-4 gram multinomial NB (top 30k); source-group-disjoint TRAIN partition",
    }

    def char_ngrams(value: str) -> set[str]:
        value = compact(value)
        return {
            value[index : index + n]
            for n in (2, 3, 4)
            for index in range(max(0, len(value) - n + 1))
        }

    label_counts = collections.Counter(y_train)
    class_grams = [collections.Counter() for _ in LABELS]
    vocabulary_counts = collections.Counter()
    for row, label in zip(train, y_train, strict=True):
        features = char_ngrams(row.hypothesis)
        class_grams[label].update(features)
        vocabulary_counts.update(features)
    vocabulary = {
        gram for gram, count in vocabulary_counts.most_common(30_000) if count >= 3
    }
    denominators = [
        sum(count for gram, count in counts.items() if gram in vocabulary)
        + len(vocabulary)
        for counts in class_grams
    ]
    log_likelihood = [
        {gram: math.log((counts[gram] + 1) / denominator) for gram in vocabulary}
        for counts, denominator in zip(class_grams, denominators, strict=True)
    ]
    predicted = []
    for row in test:
        features = char_ngrams(row.hypothesis) & vocabulary
        score = [
            math.log(label_counts[label] / len(train))
            + sum(log_likelihood[label][gram] for gram in features)
            for label in range(len(LABELS))
        ]
        predicted.append(max(range(len(LABELS)), key=lambda label: score[label]))
    result["hypothesis_only_accuracy"] = round(
        sum(want == got for want, got in zip(y_test, predicted, strict=True))
        / len(y_test),
        4,
    )
    result["hypothesis_only_balanced_accuracy"] = round(
        sum(
            sum(
                want == got == label
                for want, got in zip(y_test, predicted, strict=True)
            )
            / sum(want == label for want in y_test)
            for label in range(len(LABELS))
        )
        / len(LABELS),
        4,
    )
    for modulus in (3, 9):
        label_by_slot = {}
        for slot in range(modulus):
            labels = [row.label for row in train if row.position % modulus == slot]
            label_by_slot[slot] = (
                collections.Counter(labels).most_common(1)[0][0]
                if labels
                else LABELS[majority]
            )
        result[f"position_mod_{modulus}_accuracy"] = round(
            sum(row.label == label_by_slot[row.position % modulus] for row in test)
            / len(test),
            4,
        )
    return result


def write_once(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snli-train", type=Path, required=True)
    parser.add_argument("--snli-sha256", required=True)
    parser.add_argument("--ocnli-train", type=Path, required=True)
    parser.add_argument("--ocnli-sha256", required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--reference", action="append", default=[], help="role=path")
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("Refusing to overwrite an existing private receipt")
    if (
        sha_file(args.snli_train) != args.snli_sha256
        or sha_file(args.ocnli_train) != args.ocnli_sha256
    ):
        raise ValueError("Publisher TRAIN bytes differ from the pinned SHA-256")
    refs = []
    for value in args.reference:
        role, separator, path = value.partition("=")
        if not separator or not role:
            raise ValueError("Reference must be role=path")
        refs.append((role, Path(path)))
    protected, inventory = read_protected(args.protected_inventory, refs)
    tokenizer = None
    if args.tokenizer_dir:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False
        )
    summary = {
        "sources": {},
        "protected_inventory": inventory,
        "status": "CPU_SOURCE_SCREEN_ONLY_NO_DATA_ADMISSION",
    }
    for name, reader, source_path, expected_sha, cap_train, cap_test in (
        ("snli", read_snli, args.snli_train, args.snli_sha256, 60_000, 15_000),
        ("ocnli", read_ocnli, args.ocnli_train, args.ocnli_sha256, 30_000, 10_000),
    ):
        pairs, provenance = reader(source_path)
        summary["sources"][name] = {
            "source_sha256": expected_sha,
            **provenance,
            **source_summary(pairs, tokenizer),
            "overlap": overlap_screen(pairs, protected),
            "shortcut": shortcut_screen(pairs, cap_train, cap_test),
        }
        if name == "ocnli":
            # The publisher separately identifies the news premises as LCMC
            # content. Screen the remaining genres without admitting them.
            non_news = [row for row in pairs if row.genre != "news"]
            summary["sources"][name]["non_news_screen"] = {
                "rows": len(non_news),
                "premise_groups": len({row.group for row in non_news}),
                "class_counts": dict(
                    collections.Counter(row.label for row in non_news)
                ),
                "shortcut": shortcut_screen(non_news, 24_000, 8_000),
                "status": "SOURCE_SCREEN_ONLY_NOT_ADMITTED",
            }
    args.out_dir.mkdir(mode=0o700, parents=True)
    write_once(
        args.out_dir / "aggregate.json",
        (
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    )
    print(
        json.dumps(
            {
                name: {
                    key: value
                    for key, value in source.items()
                    if key
                    in (
                        "source_sha256",
                        "raw_rows",
                        "eligible_rows",
                        "excluded_no_consensus",
                        "independent_premise_groups",
                        "class_counts",
                        "shortcut",
                    )
                }
                for name, source in summary["sources"].items()
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
