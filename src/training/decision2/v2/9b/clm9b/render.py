"""Disaggregated state/candidate texts for separately encoded CLM-style arms.

The joint arms use the unchanged native prompt from
``training.model.decision_model``. The disaggregated arms encode the native
prompt's own segments separately: the state text is the native prefix
(context, task type and question) and each candidate text is that option's
native serialization. Content and serialization are therefore identical to
the joint prompt; only the encoding differs. A candidate vector depends only
on its text, so it can be cached across requests.
"""

from __future__ import annotations

import hashlib
from typing import Any

DISAGGREGATED_RENDER_VERSION = "decision2-9b-disaggregated-native-segments-v2"


def _native():
    from training.model.data import canonical
    from training.model.decision_model import _payload

    return canonical, _payload


def state_text(row: dict[str, Any]) -> str:
    _, payload = _native()
    return (
        f"Context:\n{payload(row['state'])}\n\n"
        f"Task type: {row['task_type']}\nQuestion:\n{payload(row['instructions'])}"
    )


def candidate_texts(row: dict[str, Any]) -> list[str]:
    """One text per offered option, in the row's option order."""
    canonical, _ = _native()
    return [
        canonical({"key": option["key"], "description": option.get("description")})
        for option in row["options"]
    ]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_level_order(row: dict[str, Any]) -> list[int]:
    """Option positions sorted by ordinal level for a Score row."""
    if row["task_type"] != "score":
        raise ValueError("Only Score rows have ordinal levels")
    levels = [int(option["key"]) for option in row["options"]]
    if sorted(levels) != list(range(len(levels))):
        raise ValueError(f"{row.get('id')}: Score keys must enumerate 0..K-1")
    return sorted(range(len(levels)), key=lambda index: levels[index])
