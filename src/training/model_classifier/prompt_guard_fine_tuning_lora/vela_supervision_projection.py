"""Conservative TRAIN projection for text-only PromptGuard supervision.

Source attack labels can depend on authority or execution context that a text
classifier does not receive. These checks quarantine uncertain weak supervision;
they do not assign attack labels from a regex or replace source evaluation gold.
"""

import hashlib
import re
from collections import Counter

_CONTROL_PATTERNS = (
    r"<\|(?:im_start|im_end|system|assistant|user|endoftext)\|>",
    r"\[/?INST\]|<<\s*/?SYS\s*>>",
    r"[\"']role[\"']\s*:\s*[\"'](?:system|assistant|user)[\"']",
    r"</message>\s*<message\s+role\s*=\s*[\"'](?:user|system)[\"']",
    r"</(?:main|email)>.{0,80}<(?:prompt|main)>",
    r"\b(?:ignore|disregard|forget|override|clear|replace|bypass)\b.{0,100}"
    r"\b(?:instructions?|system|developer|policy|rules)\b",
    r"\b(?:print|provide|output|issue|make|use|using|call)\b.{0,80}"
    r"\b(?:tool[ _-]?call|api[ _-]?call|send_email)\b",
    r"\b(?:you are chatgpt|pretend you are an? assistant)\b",
)
_CONTROL = re.compile("|".join(_CONTROL_PATTERNS), re.IGNORECASE | re.DOTALL)
_BROWSER = re.compile(
    r"<script\b|\bon(?:error|click|load|mouseover)\s*=|javascript:", re.IGNORECASE
)
_LABELS = {"benign", "jailbreak"}
_REVIEW_LABELS = {"attack", "benign", "unknown", "out_of_scope"}


def control_evidence(text):
    """Return a quarantine/retention feature, never a semantic gold label."""
    return bool(_CONTROL.search(text))


def _validated_reviews(reviews):
    indexed = {}
    for row in reviews:
        if row.get("split") != "train":
            raise ValueError("Only explicitly reviewed TRAIN rows may enter projection")
        label = row.get("reviewed_label")
        if (
            label not in _REVIEW_LABELS
            or not row.get("id")
            or not row.get("group_id")
            or not row.get("text")
            or not row.get("rationale")
        ):
            raise ValueError("Reviewed rows need identity, full text and rationale")
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if row.get("text_sha256") != digest:
            raise ValueError("Reviewed text does not match its frozen hash")
        if row["id"] in indexed:
            raise ValueError("Duplicate reviewed row")
        indexed[row["id"]] = row
    return indexed


def project_training(rows, reviews):
    """Preserve source records, with separate retained/excluded decisions.

    Individually reviewed labels apply only to their exact text. Other members
    of a group with a review disagreement are quarantined, not relabelled. Weak
    negative control patterns are quarantined, while weak positive patterns
    remain explicitly weak. Unqualified correspondence is unknown, not benign.
    """
    reviewed = _validated_reviews(reviews)
    conflicted_groups = {
        row["group_id"]
        for row in reviewed.values()
        if (
            {"attack": "jailbreak", "benign": "benign"}.get(row["reviewed_label"])
            != row["source_label"]
        )
    }
    retained, excluded, seen, used_reviews = [], [], set(), set()
    counts = Counter()
    for source_row in rows:
        row = dict(source_row)
        if (
            not row.get("id")
            or row["id"] in seen
            or not row.get("group_id")
            or not row.get("text")
            or row.get("label") not in _LABELS
            or not row.get("source", "").startswith("llmail_")
        ):
            raise ValueError("Projection requires unique labelled LLMail TRAIN rows")
        seen.add(row["id"])
        annotation = reviewed.get(row["id"])
        reason, evidence = None, control_evidence(row["text"])
        if annotation is not None:
            used_reviews.add(row["id"])
            if (
                annotation["group_id"] != row["group_id"]
                or annotation["text"] != row["text"]
                or annotation["source_label"] != row["label"]
            ):
                raise ValueError("Review does not match the original training row")
            if annotation["reviewed_label"] in {"unknown", "out_of_scope"}:
                reason = "reviewed_" + annotation["reviewed_label"]
            else:
                row.update(
                    raw_label=row["label"],
                    raw_source=row["source"],
                    label=(
                        "jailbreak"
                        if annotation["reviewed_label"] == "attack"
                        else "benign"
                    ),
                    source="reviewed_promptguard_training_v1",
                    supervision="individual_full_text_review",
                    review_rationale=annotation["rationale"],
                )
        elif row["group_id"] in conflicted_groups:
            reason = "unreviewed_member_of_conflicted_group"
        elif _BROWSER.search(row["text"]) and not evidence:
            reason = "browser_only_without_model_control_evidence"
        elif row["label"] == "benign" and evidence:
            reason = "weak_negative_with_control_evidence"
        elif row["label"] == "jailbreak" and not evidence:
            reason = "weak_positive_without_visible_control_evidence"
        else:
            row.update(raw_label=row["label"], supervision="source_weak_retained")
        if reason is None:
            retained.append(row)
            counts["retained:" + row["supervision"] + ":" + row["label"]] += 1
        else:
            excluded.append({**source_row, "exclusion_reason": reason})
            counts["excluded:" + reason] += 1
    if used_reviews != set(reviewed):
        raise ValueError("Some reviewed TRAIN records are missing from source rows")
    return (
        retained,
        excluded,
        {
            "counts": dict(counts),
            "input_rows": len(seen),
            "retained_rows": len(retained),
            "excluded_rows": len(excluded),
            "conflicted_groups": len(conflicted_groups),
            "source_evaluation_gold_modified": False,
            "regex_outputs_are_gold": False,
        },
    )
