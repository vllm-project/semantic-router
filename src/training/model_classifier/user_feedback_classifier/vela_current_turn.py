"""Refine weak feedback supervision using the visible current user turn only.

This is a new training projection. It preserves all previous artifacts and never
relabels development or final data. Exclusion rules are conservative weak
supervision; only separately reviewed utterances receive corrected labels.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ..sequence_repair.data import assert_disjoint, normalized_text
from .vela_data import SOURCE_SHA256, TRAIN_PERCENT
from .vela_weak_supervision_v2 import TERSE_ERROR, visible_user_prose

# Curly apostrophes and Chinese punctuation are intentional input characters.
# ruff: noqa: RUF001

MAX_UNATTRIBUTED_CHARACTERS = 160

PRIOR_ANSWER = re.compile(
    r"\b(?:your (?:answer|response|code|solution|explanation)|you (?:said|mentioned|made|missed|forgot)|previous (?:answer|response)|earlier (?:answer|response)|still (?:fails|gives|returns|crashes))\b|你的(?:回答|回复|代码|解释)|你(?:刚才|上次|说的|给的|写的)|还是|仍然",
    re.I,
)
CLARIFICATION = re.compile(
    r"\b(?:what (?:do|did) you mean|what does (?:this|that|it) mean|explain (?:this|that|it)|clarify (?:this|that|it)|(?:do not|don['’]?t) understand (?:it|this|that|your))\b|(?:没看懂|没有看懂|没理解)(?:这|那|你|第)|解释一下(?:这|那|第)|解释清楚(?:这|那)|再详细",
    re.I,
)
REVISION = re.compile(
    r"\b(?:make (?:it|this|that)|(?:rewrite|rephrase) (?:it|this|that|your)|try again|(?:another|different) (?:version|approach|answer)|shorter|longer|more concise|less formal)\b|改成|换成|换一种|换个|简短点|简洁点|重写|重新写|我想要(?:更|简短)",
    re.I,
)
REQUEST_ACTION = re.compile(
    r"\b(?:translate|proofread|summarize|continue|next chapter|reply to|respond to)\b|翻译|回复以下|回复这封|续写|接下来|下一章",
    re.I,
)


def weak_exclusion_reason(row):
    text = visible_user_prose(row["text"], "structured-v3")
    label = row["label"]
    prior = bool(PRIOR_ANSWER.search(text))
    if text != row["text"].strip() and not prior:
        return "Quoted or supplied material without explicit prior-answer attribution"
    if label == "SAT":
        if str(row.get("source_state", "")).upper() not in {
            "FEEDBACK",
            "POSITIVE_CLOSURE",
        }:
            return "Continuation satisfaction flag is insufficient visible feedback"
        if REQUEST_ACTION.search(text):
            return "Satisfaction words mixed with a new content request"
        return None
    if not prior and (
        len(text) > MAX_UNATTRIBUTED_CHARACTERS
        or "\n" in text
        or ":" in text
        or "：" in text
    ):
        return "Supplied content without an explicit prior-answer attribution"
    if REQUEST_ACTION.search(text) and not prior:
        return "A content transformation request does not establish feedback"
    if label == "NEED_CLARIFICATION":
        if (
            prior
            or CLARIFICATION.search(text)
            or re.fullmatch(
                r"(?:i (?:do not|don['’]?t) understand|i['’]?m confused|什么意思|没看懂|没有看懂|没理解)",
                text.strip("?？.!。 "),
                re.I,
            )
        ):
            return None
        return (
            "New explanation task or named-term meaning is not prior-answer confusion"
        )
    if label == "WANT_DIFFERENT":
        return (
            None if prior or REVISION.search(text) else "No visible revision reference"
        )
    if label == "WRONG_ANSWER":
        if TERSE_ERROR.fullmatch(text) or prior:
            return None
        if re.search(
            r"(?:有.*错误吗|是否.*错误|what.*error|is .*wrong|is .*incorrect)",
            text,
            re.I,
        ):
            return "Asking to check supplied content is not an asserted answer error"
        if re.search(
            r"\b(?:this|that|it|your|you)\b|这个|这段|不对|运行不了", text, re.I
        ):
            return None
        return "Error wording lacks a visible feedback reference"
    raise ValueError("Only the original four weak labels are eligible")


def reviewed_source_rows(source, review):
    wanted = {item["id"]: item for item in review["items"]}
    if len(wanted) != len(review["items"]):
        raise ValueError("Duplicate review IDs")
    groups = {item["group_id"] for item in review["items"]}
    resolved, reviewed = set(), []
    conversation = -1
    for index, source_row in enumerate(source):
        if source_row["UtterranceId"] == 0:
            conversation += 1
        identifier = f"wildfeedback:{index}"
        if identifier not in wanted:
            continue
        item = wanted[identifier]
        group = f"wildfeedback:conversation:{conversation}"
        text = source_row["Content"].strip()
        if (
            source_row["Role"] != "User"
            or group != item["group_id"]
            or hashlib.sha256(text.encode()).hexdigest() != item["text_sha256"]
            or int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
            >= TRAIN_PERCENT
        ):
            raise ValueError("Reviewed row changed or belongs outside training")
        resolved.add(identifier)
        if item["reviewed_label"] is None:
            continue
        if item["reviewed_label"] not in {
            "SAT",
            "WRONG_ANSWER",
            "NEED_CLARIFICATION",
            "WANT_DIFFERENT",
            "NO_FEEDBACK",
        }:
            raise ValueError("Unknown reviewed feedback label")
        reviewed.append(
            {
                "id": identifier,
                "group_id": group,
                "text": text,
                "label": item["reviewed_label"],
                "source": "reviewed_feedback_current_turn_v4",
                "source_split": "train",
                "source_state": source_row["State"],
                "language": (
                    "zh"
                    if re.search(r"[\u4e00-\u9fff]", text)
                    else "source_unspecified"
                ),
                "length_bucket": "natural_short",
                "label_provenance": review["reviewer"],
                "review_reason": item["reason"],
            }
        )
    if resolved != set(wanted):
        raise ValueError("Reviewed source row is missing")
    return reviewed, groups


def refine_rows(previous, development, reviewed, groups):
    retained, excluded = [], []
    for source_row in previous:
        row = source_row
        if row["group_id"] in groups:
            excluded.append(
                {
                    "id": row["id"],
                    "reason": "Reviewed group override; no label propagation",
                }
            )
            continue
        if row["source"] == "wildfeedback_conservative_v3":
            reason = weak_exclusion_reason(row)
            if reason:
                excluded.append({"id": row["id"], "reason": reason})
                continue
            row = {**row, "source": "wildfeedback_current_turn_v4"}
        retained.append(row)
    seen = {normalized_text(row["text"]): row for row in retained}
    for row in reviewed:
        key = normalized_text(row["text"])
        if key in seen:
            if seen[key]["label"] != row["label"]:
                raise ValueError("Reviewed label conflicts with another retained group")
            excluded.append(
                {"id": row["id"], "reason": "Same-label text already retained"}
            )
            continue
        retained.append(row)
        seen[key] = row
    assert_disjoint(retained, development)
    return retained, excluded


def read_rows(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--wildfeedback", type=Path, required=True)
    parser.add_argument(
        "--review",
        type=Path,
        default=Path(__file__).parent / "configs/vela-current-turn-review-v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a versioned corpus")
    source = args.wildfeedback.read_bytes()
    review = json.loads(args.review.read_text())
    if (
        hashlib.sha256(source).hexdigest() != SOURCE_SHA256
        or review["source_sha256"] != SOURCE_SHA256
    ):
        raise ValueError("Source differs from the pinned reviewed revision")
    reviewed, groups = reviewed_source_rows(json.loads(source), review)
    rows, excluded = refine_rows(
        read_rows(args.previous / "train.jsonl"),
        read_rows(args.previous / "validation.jsonl"),
        reviewed,
        groups,
    )
    args.output.mkdir(parents=True)
    path = args.output / "train.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    for name in ["validation.jsonl", "contract.json"]:
        (args.output / name).write_bytes((args.previous / name).read_bytes())
    manifest = {
        "protocol": "feedback-current-turn-v4",
        "source_sha256": SOURCE_SHA256,
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "previous_train_sha256": hashlib.sha256(
            (args.previous / "train.jsonl").read_bytes()
        ).hexdigest(),
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "development_sha256": hashlib.sha256(
            (args.output / "validation.jsonl").read_bytes()
        ).hexdigest(),
        "rows": len(rows),
        "source_label_counts": dict(
            Counter(row["source"] + ":" + row["label"] for row in rows)
        ),
        "reviewed_exact_rows": len(reviewed),
        "reviewed_groups": len(groups),
        "excluded": excluded,
        "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "labels": "Remaining source projections are weak; reviewed rows are assistant judgments",
        "final_texts_or_predictions_used": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "excluded"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
