"""Prepare multilingual domains with historical-source exposure screening."""

# ruff: noqa: PLC0415
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ..sequence_repair.corpus import group_split, write_corpus
from ..sequence_repair.data import load_contract, normalized_text

SUBJECT_GROUPS = {
    "biology": [
        "anatomy",
        "college_biology",
        "high_school_biology",
        "medical_genetics",
        "virology",
    ],
    "business": [
        "business_ethics",
        "management",
        "marketing",
        "professional_accounting",
        "public_relations",
    ],
    "chemistry": ["college_chemistry", "high_school_chemistry"],
    "computer science": [
        "college_computer_science",
        "high_school_computer_science",
        "computer_security",
        "machine_learning",
    ],
    "economics": [
        "econometrics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
    ],
    "engineering": ["electrical_engineering"],
    "health": [
        "clinical_knowledge",
        "college_medicine",
        "professional_medicine",
        "human_aging",
        "human_sexuality",
        "nutrition",
    ],
    "history": [
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "prehistory",
    ],
    "law": ["international_law", "jurisprudence", "professional_law"],
    "math": [
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
    ],
    "philosophy": [
        "philosophy",
        "moral_disputes",
        "moral_scenarios",
        "formal_logic",
        "logical_fallacies",
    ],
    "physics": [
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "high_school_physics",
    ],
    "psychology": ["high_school_psychology", "professional_psychology"],
}
LANGUAGES = ["en", "zh", "es", "fr", "de", "ja"]
SIMILARITY_LIMIT = 0.75
GLOBAL_REVISION = "0e619dbeb34206cd48705a1a0ea7fb21cae09993"


def question_with_choices(row):
    return (
        row["question"]
        + "\n"
        + "\n".join(f"{letter.upper()}. {row[f'option_{letter}']}" for letter in "abcd")
    )


def canonical_groups(rows):
    """Union translated siblings when any normalized request repeats."""
    parents = {row["group_id"]: row["group_id"] for row in rows}

    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    texts = {}
    for row in rows:
        key = normalized_text(row["text"])
        previous = texts.setdefault(key, row["group_id"])
        first, second = sorted([root(previous), root(row["group_id"])])
        parents[second] = first
    labels = {}
    for row in rows:
        row["group_id"] = root(row["group_id"])
        labels.setdefault(row["group_id"], set()).add(row["label"])
    retained = [row for row in rows if len(labels[row["group_id"]]) == 1]
    return retained, len(rows) - len(retained)


def main():
    from pyarrow import parquet
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--legacy-datasets", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels, ids = load_contract(args.contract)
    contract = {
        "label2id": labels,
        "id2label": ids,
        "problem_type": "single_label_classification",
    }
    mapping = {
        subject: label
        for label, subjects in SUBJECT_GROUPS.items()
        for subject in subjects
    }
    known = []
    for folder in [
        "TIGER-Lab--MMLU-Pro",
        "LLM-Semantic-Router--category-classifier-supplement",
    ]:
        for path in (args.legacy_datasets / folder).rglob("*.parquet"):
            for row in parquet.read_table(path).to_pylist():
                text = row.get("question", row.get("text"))
                if text:
                    known.append(normalized_text(text))
    known = sorted(set(known))
    known_set = set(known)
    if not known:
        raise ValueError("Historical-source screening requires actual source rows")
    english = parquet.read_table(
        args.sources / "CohereLabs--Global-MMLU/en/test-00000-of-00001.parquet"
    ).to_pylist()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=60000, min_df=2, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(known)
    search = NearestNeighbors(
        n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=8
    ).fit(matrix)
    exposure = {}
    for start in range(0, len(english), 128):
        batch = english[start : start + 128]
        distance, _ = search.kneighbors(
            vectorizer.transform([normalized_text(row["question"]) for row in batch])
        )
        for index, row in enumerate(batch):
            exposure[row["sample_id"]] = (
                normalized_text(row["question"]) in known_set
                or 1 - float(distance[index, 0]) >= SIMILARITY_LIMIT
            )
    groups = {
        row["sample_id"]: "global-mmlu:"
        + hashlib.sha256(
            normalized_text(question_with_choices(row)).encode()
        ).hexdigest()
        for row in english
    }
    group_exposed = {}
    for identifier, group in groups.items():
        group_exposed[group] = group_exposed.get(group, False) or exposure[identifier]
    raw_rows = []
    for language in LANGUAGES:
        rows = parquet.read_table(
            args.sources
            / f"CohereLabs--Global-MMLU/{language}/test-00000-of-00001.parquet"
        ).to_pylist()
        for row in rows:
            raw_rows.append(
                {
                    "id": f"global-{language}-{row['sample_id']}",
                    "text": question_with_choices(row),
                    "label": mapping.get(row["subject"], "other"),
                    "group_id": groups[row["sample_id"]],
                    "language": language,
                    "source": "global-mmlu-domain",
                    "subject": row["subject"],
                    "historical_source_similarity": bool(exposure[row["sample_id"]]),
                    "length_bucket": "short",
                }
            )
    raw_rows, quarantine = canonical_groups(raw_rows)
    group_exposed = {}
    for row in raw_rows:
        group = row["group_id"]
        group_exposed[group] = (
            group_exposed.get(group, False) or row["historical_source_similarity"]
        )
    splits = {split: [] for split in ["train", "dev", "test"]}
    counts = Counter()
    for row in sorted(
        raw_rows, key=lambda item: hashlib.sha256(item["id"].encode()).hexdigest()
    ):
        split = (
            "train" if group_exposed[row["group_id"]] else group_split(row["group_id"])
        )
        cap = {"train": 180, "dev": 25, "test": 35}[split]
        key = (split, row["language"], row["label"])
        if counts[key] >= cap:
            continue
        counts[key] += 1
        splits[split].append({**row, "split": split})
    for line in args.annotations.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        label = row["task_labels"]["domain"]
        if label is None:
            continue
        split = group_split(row["group_id"])
        splits[split].append(
            {**row, "label": label, "split": split, "length_bucket": "short"}
        )
    provenance = {
        "global_mmlu_revision": GLOBAL_REVISION,
        "license": "Apache-2.0",
        "historical_sources_screened": len(known),
        "ambiguous_label_rows_quarantined": quarantine,
        "screening": "Exact normalized text or word unigram/bigram TF-IDF cosine >=0.75 moves entire translated question group into train",
        "historical_exposure_not_guaranteed": True,
        "subject_mapping": SUBJECT_GROUPS,
        "unmapped_subject_policy": "other",
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "source_group_rows": dict(
            Counter(row["source"] for rows in splits.values() for row in rows)
        ),
    }
    print(
        json.dumps(write_corpus(args.output, splits, contract, provenance)), flush=True
    )


if __name__ == "__main__":
    main()
