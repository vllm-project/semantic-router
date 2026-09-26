"""Map isolated Decision training rows to the released Eikos letter readout."""

from __future__ import annotations

import json
import random
from typing import Any

from training.model.data import digest

PROMPT_VERSION = "letter-v1-semif"
MAX_ONE_PASS = 100


def _description(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def native_question(
    row: dict[str, Any], *, shuffle_seed: int | None = None
) -> tuple[dict[str, Any], str]:
    """Return an API question and its gold key, keeping native boolean semantics.

    Choice permutation is deterministic per epoch. Ordinal Score retains its
    increasing level order; boolean always follows native yes/no order.
    """
    options = list(row["options"])
    if shuffle_seed is not None and row["task_type"] == "choice":
        random.Random(digest([shuffle_seed, row["id"]])).shuffle(options)
    gold = row["options"][row["label"]]["key"]
    criteria = {item["key"]: _description(item["description"]) for item in options}
    if row["task_type"] == "noul":
        criteria = {"true": criteria["true"], "false": criteria["false"]}
    return {
        "type": row["task_type"],
        "instructions": row["instructions"],
        "criteria": criteria,
    }, gold


def encode(
    row: dict[str, Any],
    tokenizer: Any,
    core: Any,
    *,
    max_length: int,
    shuffle_seed: int | None = None,
) -> dict[str, Any]:
    question, gold = native_question(row, shuffle_seed=shuffle_seed)
    options = core.options_of(question)
    keys = [key for key, _ in options]
    if len(keys) > len(core.LABELS):
        raise ValueError(
            f"{row['id']}: option count exceeds single-token label inventory"
        )
    text = tokenizer.apply_chat_template(
        core.messages(row["state"], question, options),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids or len(ids) > max_length:
        raise ValueError(
            f"{row['id']}: native prompt has {len(ids)} tokens, max {max_length}"
        )
    letters = [
        tokenizer.encode(core.LABELS[index], add_special_tokens=False)
        for index in range(len(keys))
    ]
    if any(len(item) != 1 for item in letters) or len(
        {item[0] for item in letters}
    ) != len(letters):
        raise ValueError("Eikos letter label tokenization is not one-to-one")
    native_gold = (
        {"true": "yes", "false": "no"}[gold] if row["task_type"] == "noul" else gold
    )
    return {
        "id": row["id"],
        "family": row["family"],
        "task_type": row["task_type"],
        "ids": ids,
        "letter_ids": [item[0] for item in letters],
        "keys": keys,
        "label": keys.index(native_gold),
        "source_input_sha256": row["input_sha256"],
    }


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """Right-pad like released letter_adapter; gather each actual last token."""
    import torch

    length = max(len(item["ids"]) for item in items)
    width = max(len(item["letter_ids"]) for item in items)
    ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(items), length), dtype=torch.long)
    letters = torch.zeros((len(items), width), dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.bool)
    for i, item in enumerate(items):
        n = len(item["ids"])
        k = len(item["letter_ids"])
        ids[i, :n] = torch.tensor(item["ids"])
        attention[i, :n] = 1
        letters[i, :k] = torch.tensor(item["letter_ids"])
        mask[i, :k] = True
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "last_positions": attention.sum(1) - 1,
        "letter_ids": letters,
        "candidate_mask": mask,
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }


def one_pass_quarantine(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only rows whose candidate set matches the native one-pass policy."""
    effective: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        n = len(row["options"])
        if n <= MAX_ONE_PASS:
            effective.append(row)
            continue
        original = row["audit_metadata"].get("original_source", {})
        rights_source = (
            original.get("original_source")
            or original.get("generator")
            or row["source"]
        )
        quarantined.append(
            {
                "id": row["id"],
                "family": row["family"],
                "task_type": row["task_type"],
                "option_count": n,
                "source": row["source"],
                "rights_source": rights_source,
                "original_source": original,
                "input_sha256": row["input_sha256"],
                "reason": "candidate_count_exceeds_Eikos4B_native_max_one_pass_100",
            }
        )
    if not effective:
        raise ValueError("No TRAIN rows fit Eikos native one-pass policy")
    return effective, quarantined
