"""Training-only conservative refinement of WildFeedback's weak four-way labels.

An occurrence of 'false', 'instead', or 'explain' inside requested content does
not establish the user's feedback intent. This recipe retains source state and
filters ambiguity; it neither relabels held-out data nor claims human gold.
"""

# Curly apostrophes and Chinese punctuation are intentional input characters.
# ruff: noqa: RUF001

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .vela_data import SOURCE_SHA256, TRAIN_PERCENT

PROTOCOL = "feedback-weak-training-v2"
FEEDBACK_STATES = {"FEEDBACK", "REFINEMENT", "REVISION"}
SAT_STATES = {"FEEDBACK", "CONTINUATION", "POSITIVE_CLOSURE"}
MAX_SAT_CHARACTERS = 500
VISIBLE_FEEDBACK = re.compile(
    r"\b(you|your|this|that|it|answer|response|result|output|code|calculation|summary|claim)\b|你|回复|回答|结果|输出|代码|这里|这个|那个|还是|仍然",
    re.I,
)
EXPLICIT_ERROR = re.compile(
    r"\b(incorrect|inaccurate|mistake|doesn['’]?t work|does not work|didn['’]?t work|not correct|not working|still fails|you made up|you missed|you forgot|not what i asked|you.{0,30}wrong|your.{0,40}wrong|(?:that|this|it).{0,20}(?:wrong|false))\b|错误|不正确|算错|答错|不对|编造|运行不了",
    re.I,
)
TERSE_ERROR = re.compile(
    r"\s*(?:no[,;!]?\s*)?(?:wrong|incorrect|not correct|not right|错误|不对|错了)[.!。！ ]*",
    re.I,
)
CONFUSION = re.compile(
    r"\b(i (?:do not|don['’]?t) understand|i['’]?m confused|what (?:do|did) you mean|what does (?:that|this|it) mean|clarify (?:that|this|it)|explain (?:that|this|it|why|how))\b|没看懂|不理解|解释一下|解释清楚|什么意思",
    re.I,
)
FORMAT_CHANGE = re.compile(
    r"\b(as a poem|in (?:verse|prose)|(?:bullet|point) form|(?:bullet|numbered) points|as a (?:table|list|checklist)|rewrite|rephrase|shorter|longer|more concise|less formal)\b|改成|换成|换个|换一种|重写|简短|简洁|表格|诗歌",
    re.I,
)
REVISION_REQUEST = re.compile(
    r"(?:^|[.!?]\s*)(?:please\s+)?(?:rewrite|rephrase|shorter|longer|try again|make (?:it|this|that)|use (?:a |the )?(?:different|another)|give (?:me )?(?:another|a different))\b|\b(?:can|could|would) you (?:rewrite|rephrase|make (?:it|this|that)|try|use|give)\b|换一种|换个|重写|重新写|简短|简洁|换成|改成|另一种",
    re.I,
)
SAT_CONTINUATION = re.compile(
    r"\b(?:but|however|though|instead|actually|could you|can you|please (?:write|make|add|explain|give|do))\b|不过|但是|可是|能不能|请你|再帮|另外",
    re.I,
)


def visible_user_prose(text, quote_policy="ascii-v2"):
    if quote_policy == "structured-v3":
        text = re.sub(
            r"“[^”]*”|‘[^’]*’|「[^」]*」|『[^』]*』|^\s*>.*$", " ", text, flags=re.M
        )
    elif quote_policy != "ascii-v2":
        raise ValueError("Unknown quoted-text policy")
    text = re.sub(r"```[\s\S]*?```|`[^`]*`", " ", text)
    return re.sub(r'"[^"\n]*"', " ", text).strip()


def exclusion_reason(row, annotation, quote_policy="ascii-v2"):
    state = str(annotation["State"]).upper()
    text = visible_user_prose(row["text"], quote_policy)
    label = row["label"]
    if label == "SAT":
        if state not in SAT_STATES:
            return "positive words during a new topic or revision are not standalone satisfaction"
        if (
            len(text) > MAX_SAT_CHARACTERS
            or "?" in text
            or "？" in text
            or SAT_CONTINUATION.search(text)
        ):
            return "mixed satisfaction and a further request are ambiguous"
        return None
    if state not in FEEDBACK_STATES:
        return "continuation without explicit feedback state"
    if label == "WRONG_ANSWER":
        return (
            None
            if TERSE_ERROR.fullmatch(text)
            or (VISIBLE_FEEDBACK.search(text) and EXPLICIT_ERROR.search(text))
            else "error keyword does not identify a problem with the prior answer"
        )
    if label == "NEED_CLARIFICATION":
        if FORMAT_CHANGE.search(text):
            return "clarification keyword mixed with an explicit format change"
        return (
            None
            if CONFUSION.search(text)
            else "elaboration alone does not establish confusion"
        )
    if label == "WANT_DIFFERENT":
        if EXPLICIT_ERROR.search(text) or CONFUSION.search(text):
            return "revision keyword mixed with an error or misunderstanding"
        return (
            None
            if REVISION_REQUEST.search(text) or FORMAT_CHANGE.search(text)
            else "replacement keyword alone does not establish a revision request"
        )
    raise ValueError("Unexpected feedback label")


def training_rows(rows, annotations, quote_policy="ascii-v2", quote_review=None):
    if quote_policy not in {"ascii-v2", "structured-v3"}:
        raise ValueError("Unknown quoted-text policy")
    quote_review = quote_review or {}
    kept, excluded = [], []
    for row in rows:
        if row.get("source") != "wildfeedback_weak_projection":
            continue
        group = row.get("group_id", "")
        if (
            not group.startswith("wildfeedback:conversation:")
            or int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
            >= TRAIN_PERCENT
        ):
            raise ValueError(
                "Only the original WildFeedback training partition is eligible"
            )
        index = int(row["id"].removeprefix("wildfeedback:"))
        source = annotations[index]
        if source["Role"] != "User" or source["Content"].strip() != row["text"]:
            raise ValueError("Training row differs from the pinned user utterance")
        reason = exclusion_reason(row, source)
        # V3 only refines the V2 subset; it does not admit new weak positives.
        if reason is None and quote_policy == "structured-v3":
            reason = exclusion_reason(row, source, quote_policy)
            if row["id"] in quote_review:
                item = quote_review[row["id"]]
                if (
                    item["text_sha256"]
                    != hashlib.sha256(row["text"].encode()).hexdigest()
                    or item["label"] != row["label"]
                ):
                    raise ValueError(
                        "Quoted-text review no longer matches the source row"
                    )
                if item["action"] not in {"retain", "exclude"}:
                    raise ValueError("Unknown quoted-text review action")
                reason = None if item["action"] == "retain" else item["reason"]
        if reason:
            excluded.append({"id": row["id"], "reason": reason})
        else:
            kept.append(
                {
                    **row,
                    "source": (
                        "wildfeedback_conservative_v3"
                        if quote_policy == "structured-v3"
                        else "wildfeedback_conservative_v2"
                    ),
                    "source_state": source["State"],
                    "source_split": "train",
                    "label_provenance": "original weak label retained after state/prose ambiguity filtering",
                }
            )
    return kept, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--wildfeedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quote-policy", choices=["ascii-v2", "structured-v3"], default="ascii-v2"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing overwrite of a versioned training corpus")
    source = args.wildfeedback.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("WildFeedback source does not match the fixed revision")
    original = [
        json.loads(line) for line in args.input.read_text().split("\n") if line.strip()
    ]
    quote_review = {}
    quote_review_sha256 = None
    if args.quote_policy == "structured-v3":
        review_path = Path(__file__).parent / "configs/vela-quote-review-v1.json"
        reviewed = json.loads(review_path.read_text())["items"]
        quote_review = {row["id"]: row for row in reviewed}
        if len(quote_review) != len(reviewed):
            raise ValueError("Quoted-text review has duplicate IDs")
        quote_review_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
    rows, excluded = training_rows(
        original, json.loads(source), args.quote_policy, quote_review
    )
    args.output.mkdir(parents=True)
    path = args.output / "train.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    manifest = {
        "protocol": (
            "feedback-weak-training-v3"
            if args.quote_policy == "structured-v3"
            else PROTOCOL
        ),
        "quote_policy": args.quote_policy,
        "quote_review_sha256": quote_review_sha256,
        "source_sha256": SOURCE_SHA256,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rows": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "groups": len({row["group_id"] for row in rows}),
        "excluded": excluded,
        "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "heldout_modified": False,
        "label_type": "conservative weak supervision, not human gold",
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
