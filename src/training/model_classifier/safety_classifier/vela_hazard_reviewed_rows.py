"""Exact TRAIN reviews with partial negatives and group-wide supersession."""

import hashlib

from .vela_data import LABELS

REVIEWED_SOURCE = "reviewed_hazard_training_v12"
WEAK_SOURCES = frozenset(
    {"aegis2", "cultureguard_generic_partial_positive", "cultureguard_safe"}
)


def project_reviews(rows):
    """Retain only observed labels; unknown binary status is never a safe label."""
    output, excluded, groups, seen = [], [], set(), set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate reviewed ID")
        seen.add(row["id"])
        if (
            row.get("source_split") != "train"
            or not row.get("group_id")
            or hashlib.sha256(row["text"].encode()).hexdigest()
            != row["reviewed_text_sha256"]
        ):
            raise ValueError("Review text, group or training split changed")
        targets = list(row["reviewed_targets"])
        mask = list(row["reviewed_label_mask"])
        if (
            len(targets) != len(LABELS)
            or len(mask) != len(LABELS)
            or any(value not in {0, 1} for value in targets + mask)
            or any(
                value and not observed
                for value, observed in zip(targets, mask, strict=True)
            )
        ):
            raise ValueError("Invalid reviewed label observations")
        expected = "unsafe" if any(targets) else "safe" if all(mask) else "unknown"
        if row["reviewed_binary_label"] != expected:
            raise ValueError("Binary review conflicts with observed hazard labels")
        groups.add(row["group_id"])
        if not any(mask):
            excluded.append({"id": row["id"], "reason": "all labels unknown"})
            continue
        output.append(
            {
                **row,
                "source": REVIEWED_SOURCE,
                "source_before_exact_review": row["source"],
                "binary_before_exact_review": row["label"],
                "targets_before_exact_review": list(row["targets"]),
                "mask_before_exact_review": list(row["label_mask"]),
                "label": expected,
                "targets": targets,
                "label_mask": mask,
                "supervision_protocol": "vela-hazard-exact-review-v12",
            }
        )
    return output, groups, excluded


def filter_replay(rows, reviewed_groups, *, strong_only=False):
    """Drop every reviewed family variant, including wrapped weak payloads."""
    output, excluded = [], []
    for row in rows:
        if row.get("source_split") != "train":
            raise ValueError("Replay filtering only accepts training rows")
        reason = None
        if row["group_id"] in reviewed_groups:
            reason = "reviewed family supersedes all inherited variants"
        elif strong_only and (
            row["source"] in WEAK_SOURCES
            or row.get("source_payload_source") in WEAK_SOURCES
        ):
            reason = "weak source or wrapped weak payload excluded by experiment"
        if reason:
            excluded.append(
                {"id": row["id"], "group_id": row["group_id"], "reason": reason}
            )
        else:
            output.append(dict(row))
    return output, excluded
