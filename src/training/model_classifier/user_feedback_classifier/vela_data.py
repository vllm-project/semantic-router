"""Conservative weak supervision from licensed, real user feedback.

The four-way labels below are heuristic projections, not human gold. Dialogue
roles, conversation boundaries, group splits and exact normalization are
preserved. Independent author-labelled/final tests must remain separate.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .data_contract import ID2LABEL, LABEL2ID, fingerprint

MIN_TEXT_CHARS = 3
MAX_TEXT_CHARS = 4000
TRAIN_PERCENT = 80
TRAIN_AND_VALIDATION_PERCENT = 90
SOURCE_SHA256 = "2caea8361756b35a512b2ebef93245df1e81d982a3602460f75a4445fc16d305"

PATTERNS = {
    "WRONG_ANSWER": re.compile(
        r"\b(wrong|incorrect|inaccurate|false|mistake|doesn['\u2019]?t work|does not work|didn['\u2019]?t work|not correct|not working|still fails|you made up|you missed|you forgot|not what i asked)\b|错误|不正确|算错|答错|不对|编造|运行不了",
        re.I,
    ),
    "NEED_CLARIFICATION": re.compile(
        r"\b(clarify|clarification|elaborate|what do you mean|i don['\u2019]?t understand|i do not understand|i['\u2019]?m confused|explain (that|this|it|why|how)|more detail|explain.{0,20}(step|again|further))\b|没看懂|不理解|解释一下|解释清楚|详细说明|什么意思",
        re.I,
    ),
    "WANT_DIFFERENT": re.compile(
        r"\b(rewrite|rephrase|shorter|longer|another (option|approach|version|example)|different (approach|style|way|option)|try again|instead|bullet points|more concise|less formal|simpler (language|words)|make it (short|long|more|less))\b|换一种|换个|重写|重新写|简短|简洁|换成|改成|另一种",
        re.I,
    ),
    "SAT": re.compile(
        r"\b(thanks?|thank you|perfect|excellent|great|helpful|exactly|awesome|brilliant|nice|works|worked|much better|appreciate|that helps|good answer|well done)\b|谢谢|感谢|很好|太好了|明白了|有帮助|解决了|不错|满意",
        re.I,
    ),
}
SENSITIVE = re.compile(
    r"-----BEGIN.{0,20}PRIVATE KEY|\b(?:sk-|ghp_|hf_)[A-Za-z0-9_-]{20,}|(?:password|api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S+",
    re.I,
)


def project_label(text, state, satisfied, dissatisfied):
    if satisfied and dissatisfied:
        return None
    if satisfied and not dissatisfied:
        return (
            "SAT"
            if PATTERNS["SAT"].search(text)
            and not any(
                PATTERNS[k].search(text)
                for k in ["WRONG_ANSWER", "NEED_CLARIFICATION", "WANT_DIFFERENT"]
            )
            else None
        )
    if state not in {"FEEDBACK", "REFINEMENT", "REVISION", "CONTINUATION"}:
        return None
    for label in ["WRONG_ANSWER", "NEED_CLARIFICATION", "WANT_DIFFERENT"]:
        if PATTERNS[label].search(text):
            return label
    return None


def wild_records(rows):
    conversation = -1
    previous = -1
    for index, row in enumerate(rows):
        utterance = row["UtterranceId"]
        if utterance == 0:
            conversation += 1
            previous = -1
        if utterance <= previous:
            raise ValueError("WildFeedback conversation ordering changed")
        previous = utterance
        if row["Role"] != "User":
            continue
        if not isinstance(row.get("Content"), str):
            continue
        text = row["Content"].strip()
        if not MIN_TEXT_CHARS <= len(text) <= MAX_TEXT_CHARS or SENSITIVE.search(text):
            continue
        label = project_label(
            text, row["State"], row["Satisfaction"], row["Disatisfaction"]
        )
        if label:
            yield {
                "id": f"wildfeedback:{index}",
                "group_id": f"wildfeedback:conversation:{conversation}",
                "text": text,
                "label": label,
                "source": "wildfeedback_weak_projection",
                "language": (
                    "zh"
                    if re.search(r"[\u4e00-\u9fff]", text)
                    else "source_unspecified"
                ),
                "length_bucket": "natural_short",
                "label_provenance": "source_flags_and_versioned_lexical_projection",
            }


def split_groups(rows):
    result = {"train": [], "validation": [], "test": []}
    by_text, conflicts = {}, set()
    for row in rows:
        key = fingerprint(row["text"])
        if key in by_text and by_text[key]["label"] != row["label"]:
            conflicts.add(key)
        by_text.setdefault(key, row)
    for key, row in by_text.items():
        if key in conflicts:
            continue
        value = int(hashlib.sha256(row["group_id"].encode()).hexdigest()[:8], 16) % 100
        split = (
            "train"
            if value < TRAIN_PERCENT
            else "validation" if value < TRAIN_AND_VALIDATION_PERCENT else "test"
        )
        result[split].append(row)
    return result, len(conflicts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wildfeedback", type=Path, required=True)
    parser.add_argument("--authored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an immutable corpus")
    payload = args.wildfeedback.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("WildFeedback source does not match the pinned revision")
    rows = list(wild_records(json.loads(payload)))
    splits, conflicts = split_groups(rows)
    # Authored bilingual families are already partitioned by semantic family.
    # Held-out authored data takes precedence over any natural-text duplicates.
    for split in ["train", "validation"]:
        for line in (args.authored / f"{split}.jsonl").read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label_name", row["label"])
            if isinstance(label, int):
                label = ID2LABEL[label]
            splits[split].append(
                {
                    **row,
                    "label": label,
                    "id": row.get("id") or row["sample_id"],
                    "source": "authored_feedback_families_v2",
                    "language": row.get("language", "unspecified"),
                    "length_bucket": "authored_short",
                }
            )
    seen = set()
    for split in ["test", "validation", "train"]:
        unique = []
        for row in splits[split]:
            key = fingerprint(row["text"])
            if key not in seen:
                seen.add(key)
                unique.append(row)
        splits[split] = unique
    args.output.mkdir(parents=True)
    manifest = {
        "source": "microsoft/WildFeedback",
        "revision": "8b1a3e530b949d6aacfad6ba8912e209a05bc846",
        "license": "odc-by",
        "source_sha256": hashlib.sha256(args.wildfeedback.read_bytes()).hexdigest(),
        "label_type": "weak four-way projection plus authored bilingual families",
        "label_conflicts_removed": conflicts,
        "test_used_for_selection": False,
        "files": {},
    }
    for split, items in splits.items():
        path = args.output / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items)
        )
        manifest["files"][split] = {
            "rows": len(items),
            "labels": dict(Counter(row["label"] for row in items)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (args.output / "contract.json").write_text(
        json.dumps(
            {
                "id2label": ID2LABEL,
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
