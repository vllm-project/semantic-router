"""Licensed Vela PromptGuard data with submitter/question group isolation.

LLMail labels are source weak annotations of attacks in an untrusted email
setting, not general malicious-content labels or agent-resilience scores.
SALAD attack/base-question pairs distinguish instruction attacks from harm.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .jailbreak_data_v2 import fingerprint, salad_examples, split_examples

TRAIN_PARTITION_BUCKETS = 8
MIN_TEXT_CHARS = 20
MAX_TEXT_CHARS = 14000

LABEL2ID = {"benign": 0, "jailbreak": 1}
LLMAIL_REVISION = "1063bdf01ec8762b812d5e06ee768a06faa5a6f7"
LLMAIL_SHA256 = {
    "labelled_unique_submissions_phase1.json": "691dfa1595d2bd0e731069f233bd5448f7af1c0ebd9732bc6f12dd6cee446586",
    "labelled_unique_submissions_phase2.json": "f89af984e345430c3b357903890e30867bf4676f4ef10c138cc7bad218e890b8",
    "raw_submissions_phase1.jsonl": "a9c62eca699dd270fdfbbfbfcc1253f5e5017f6d5a34ff7cb0f2cbb80b7f7c0a",
    "raw_submissions_phase2.jsonl": "a9207e1d893ccb74ca6f9cc5eecea433bc49c23a26bed88088afd385c7ab18b6",
}


def annotation_conflicts(annotations, known=None):
    """Keep conflicts even when one phase has already discarded that text."""
    known = {} if known is None else known
    conflicts = set()
    for text, value in annotations.items():
        label = value["attack_attempt"]
        if isinstance(label, bool):
            label = str(label)
        if label not in ("True", "False"):
            continue
        key = fingerprint(text)
        if key in known and known[key] != label:
            conflicts.add(key)
        known.setdefault(key, label)
    return known, conflicts


def normalize_annotations(annotations):
    labels, conflicts = {}, set()
    valid = 0
    for text, value in annotations.items():
        label = value["attack_attempt"]
        if isinstance(label, bool):
            label = str(label)
        if label not in ("True", "False"):
            continue
        valid += 1
        key = fingerprint(text)
        if key in labels and labels[key][1] != label:
            conflicts.add(key)
        labels.setdefault(key, (text, label))
    return {key: value for key, value in labels.items() if key not in conflicts}, {
        "source_boolean_annotations": valid,
        "source_unclear_or_invalid": len(annotations) - valid,
        "normalized_duplicates": valid - len(labels),
        "conflicting_normalized_annotations": len(conflicts),
    }


def team_partition(teams):
    buckets = {
        int(hashlib.sha256(f"vela-promptguard:{team}".encode()).hexdigest()[:8], 16)
        % 10
        for team in teams
    }
    partitions = {
        (
            "train"
            if value < TRAIN_PARTITION_BUCKETS
            else "validation" if value == TRAIN_PARTITION_BUCKETS else "test"
        )
        for value in buckets
    }
    return next(iter(partitions)) if len(partitions) == 1 else None


def llmail_rows(directory, per_team=160):
    directory = Path(directory)
    result = {"train": [], "validation": [], "test": []}
    audit = Counter()
    seen = set()
    normalized, cross_phase_conflicts, all_labels = {}, set(), {}
    for filename, expected in LLMAIL_SHA256.items():
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"LLMail source does not match pinned revision: {filename}"
            )
    for phase in ["phase1", "phase2"]:
        annotations = json.loads(
            (directory / f"labelled_unique_submissions_{phase}.json").read_text()
        )
        all_labels, conflicts = annotation_conflicts(annotations, all_labels)
        cross_phase_conflicts.update(conflicts)
        labels, normalization_audit = normalize_annotations(annotations)
        normalized[phase] = labels
        audit.update(
            {f"{phase}:{key}": value for key, value in normalization_audit.items()}
        )
    audit["any_phase_conflicting_normalized_annotations"] = len(cross_phase_conflicts)
    for phase in ["phase1", "phase2"]:
        labels = {
            key: value
            for key, value in normalized[phase].items()
            if key not in cross_phase_conflicts
        }
        teams = defaultdict(set)
        with (directory / f"raw_submissions_{phase}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                text = f"Subject of the email: {row['subject']}.   Body: {row['body']}"
                key = fingerprint(text)
                if key in labels and row.get("team_id"):
                    teams[key].add(row["team_id"])
        by_team = Counter()
        for key, (text, raw_label) in sorted(labels.items()):
            if key in seen:
                continue
            if not teams[key]:
                audit[f"{phase}:missing_team_mapping"] += 1
                continue
            split = team_partition(teams[key])
            if split is None:
                audit[f"{phase}:text_shared_across_split_teams"] += 1
                continue
            if phase == "phase2" and split != "test":
                continue
            if not MIN_TEXT_CHARS <= len(text) <= MAX_TEXT_CHARS:
                audit[f"{phase}:outside_character_budget"] += 1
                continue
            group = (
                "llmail:"
                + hashlib.sha256("|".join(sorted(teams[key])).encode()).hexdigest()
            )
            label = int(raw_label == "True")
            # Cap prolific submitters within each label to keep broad coverage.
            cap = per_team if split == "train" else max(20, per_team // 4)
            if by_team[(group, label)] >= cap:
                continue
            by_team[(group, label)] += 1
            seen.add(key)
            result[split].append(
                {
                    "id": f"llmail:{key}",
                    "text": text,
                    "label": label,
                    "group_id": group,
                    "source": f"llmail_{phase}_weak",
                    "language": "source_unspecified",
                    "length_bucket": "natural_short",
                    "annotation_scope": "untrusted_email",
                }
            )
        audit[f"{phase}:source_annotated_boolean"] = len(labels)
    return result, dict(audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llmail", required=True)
    parser.add_argument("--salad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite")
    splits, audit = llmail_rows(args.llmail)
    payload = args.salad.read_bytes()
    if (
        hashlib.sha256(payload).hexdigest()
        != "389da12f029a7f0c4a562f753e1dd5e53aed46d873ef1f568919d9f4b72e7062"
    ):
        raise ValueError("SALAD revision SHA mismatch")
    salad, report = split_examples(salad_examples(json.loads(payload)))
    for split in splits:
        splits[split].extend(
            {
                **row,
                "source": "salad_paired",
                "language": "en",
                "length_bucket": "natural_short",
            }
            for row in salad[split]
        )
    seen = set()
    for split in ["test", "validation", "train"]:
        unique = []
        for row in splits[split]:
            key = fingerprint(row["text"])
            if key in seen:
                continue
            seen.add(key)
            unique.append({**row, "label": ["benign", "jailbreak"][row["label"]]})
        splits[split] = unique
    args.output.mkdir(parents=True)
    manifest = {
        "llmail_revision": LLMAIL_REVISION,
        "llmail_license": "mit",
        "salad_revision": "d21a325e276a99bd69b1fbb8aa51a9f249486b72",
        "salad_license": "apache-2.0",
        "llmail_audit": audit,
        "salad_group_audit": report,
        "limitations": "LLMail labels are weak and trust-context dependent. Team/question isolation does not guarantee all attack-method families are novel.",
        "excluded_training_sources": [
            "toxic-chat: cc-by-nc-4.0",
            "wildguardmix/wildjailbreak: access not granted",
        ],
        "files": {},
    }
    for split, rows in splits.items():
        random.Random(20260913).shuffle(rows)
        path = args.output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows)
        )
        manifest["files"][split] = {
            "rows": len(rows),
            "labels": dict(Counter(row["label"] for row in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "contract.json").write_text(
        json.dumps(
            {
                "id2label": {i: label for label, i in LABEL2ID.items()},
                "label2id": LABEL2ID,
                "problem_type": "single_label_classification",
            },
            indent=2,
        )
        + "\n"
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
