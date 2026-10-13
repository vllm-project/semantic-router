"""The ``vela-encoder`` runtime of Decision 1.0: Kai, Lex and Route.

One embedding pass feeds three ModernBERT layer stacks, one per question type:
Noul reads the shared encoder, Choice and Score their own copies. A question
reads ``[bos] "<type> question: <instructions>" [sep]``, then ``[mask]
<candidate> [sep]`` per candidate, then the state and ``[sep]``; its type's
transformer head layers and scorer read the candidate markers. The released
numerics: FP32 on every device; a request's questions stably sorted by type
into physical batches of eight, each padded to its longest question; every
type present runs its stack over the whole physical batch.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ...errors import INVALID_QUESTION, MAX_LENGTH_EXCEEDED, PackageError, QuestionError
from ...heads.typed import TypeReadout
from ...plugins.base import EncoderBatch
from ...plugins.decisions import RenderedItem
from ...systemone import canonical
from .questions import KINDS, NoulDefaults, Row

PHYSICAL_BATCH = 8
MAX_INPUT_TOKENS = 1024
READOUT_PADDING = 0.25
# Below this many padded tokens one read costs less than several.
READOUT_ONE_GRID = 1024
# Larger coalesced batches run slower per token on CPUs (max_speed at 16 concurrent requests, 16 cores).
CPU_BATCH_TOKENS = 1024
NOUL_DEFAULTS = NoulDefaults(
    false="No. The statement or question is not satisfied.",
    true="Yes. The statement or question is satisfied.",
    null_is_default=True,
)
ARM = {"arm": "all22", "training_arm": "S22", "type_order": list(KINDS)}
# The layer stack each question type reads: Noul the shared encoder, the others their branch.
BRANCH = {"choice": "choice", "noul": None, "score": "score"}
SPECIAL_TOKENS = {"bos": ("cls_token", "bos_token"), "sep": ("sep_token", "eos_token"),
                  "pad": ("pad_token",), "marker": ("mask_token",)}  # fmt: skip


def check_config(config: dict[str, Any]) -> dict[str, Any]:
    """The head geometry of the three-path ``all22`` encoder the packages ship; raises on anything else."""
    if any(config.get(key) != value for key, value in ARM.items()):
        raise PackageError(
            "only the three-path all22 / S22 Decision 1.0 encoder is supported"
        )
    if (config.get("packing") or {}).get("state_truncation") != "error":
        raise PackageError("Decision 1.0 encoders never truncate the state")
    head = config.get("head") or {}
    if (
        type(head.get("head_layers")) is not int
        or type(head.get("head_heads")) is not int
    ):
        raise PackageError("decision_config.json head needs head_layers and head_heads")
    return head


def special_ids(backend: Any, root: Path, files: dict[str, str]) -> dict[str, int]:
    """Token IDs of the BOS (or CLS), SEP (or EOS), PAD and MASK tokens the layout uses."""
    names: dict[str, Any] = {}
    for key in ("config", "special_tokens_map"):
        if key in files:
            document = json.loads((root / files[key]).read_text(encoding="utf-8"))
            names.update({k: v for k, v in document.items() if v is not None})
    found = {}
    for role, fields in SPECIAL_TOKENS.items():
        for field in fields:
            token = names.get(field)
            token = token.get("content") if isinstance(token, dict) else token
            identifier = backend.token_to_id(token) if isinstance(token, str) else None
            if identifier is not None:
                found[role] = identifier
                break
        else:
            raise PackageError(
                "the tokenizer must define CLS/BOS, SEP/EOS, PAD and MASK tokens"
            )
    return found


def candidate_text(kind: str, index: int, key: str, description: Any) -> str:
    """A candidate's text: its key without a description, ``key: text`` for Choice, ``level i:`` for Score."""
    if description is None:
        text = key
    else:
        text = description if isinstance(description, str) else canonical(description)
        if kind == "choice":
            text = f"{key}: {text}"
    return f"level {index}: {text}" if kind == "score" else text


def render(
    row: Row,
    state: str,
    tokens: Callable[[str], list[int]],
    special: dict[str, int],
    max_length: int,
) -> RenderedItem:
    """The question's token layout; the complete input must fit ``max_length`` (nothing is truncated)."""
    instructions = (
        row.instructions
        if isinstance(row.instructions, str)
        else canonical(row.instructions)
    )
    ids = [
        special["bos"],
        *tokens(f"{row.kind} question: {instructions}"),
        special["sep"],
    ]
    markers = []
    for index, candidate in enumerate(row.candidates):
        markers.append(len(ids))
        text = candidate_text(row.kind, index, candidate.key, candidate.description)
        ids += [special["marker"], *tokens(text), special["sep"]]
    state_ids = tokens(state)
    room = max_length - len(ids) - 1
    if room < 1 or len(state_ids) > room:
        raise QuestionError(
            MAX_LENGTH_EXCEEDED,
            f"{row.question_id}: the complete input exceeds {max_length} tokens; no truncation",
        )
    ids += [*state_ids, special["sep"]]
    return RenderedItem(
        question_id=row.question_id,
        task_type=row.kind,
        ids=ids,
        gather=markers,
        query=0,
        keys=row.keys,
        descriptions=[candidate.description for candidate in row.candidates],
    )


def token_cache(encode: Callable[[str], list[int]]) -> Callable[[str], list[int]]:
    """Tokenize each distinct text once per request (the state is shared by every question)."""
    cache: dict[str, list[int]] = {}

    def tokens(text: str) -> list[int]:
        if text not in cache:
            ids = encode(text)
            if not ids:
                raise QuestionError(
                    INVALID_QUESTION,
                    "a candidate, question or state renders to no tokens",
                )
            cache[text] = ids
        return cache[text]

    return tokens


def physical_batches(items: list[RenderedItem]) -> list[list[int]]:
    """Item indices sorted by type (choice, noul, score; stable), in batches of eight."""
    order = sorted(
        range(len(items)), key=lambda index: KINDS.index(items[index].task_type)
    )
    return [
        order[start : start + PHYSICAL_BATCH]
        for start in range(0, len(order), PHYSICAL_BATCH)
    ]


def collate(items: list[RenderedItem], pad_id: int) -> dict[str, torch.Tensor]:
    """Rows padded to the longest (no rounding), markers to the widest row, as the released runtime pads."""
    length = max(len(item.ids) for item in items)
    width = max(len(item.gather) for item in items)
    input_ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(items), length), dtype=torch.bool)
    markers = torch.zeros((len(items), width), dtype=torch.long)
    valid = torch.zeros((len(items), width), dtype=torch.bool)
    for slot, item in enumerate(items):
        input_ids[slot, : len(item.ids)] = torch.tensor(item.ids, dtype=torch.long)
        attention_mask[slot, : len(item.ids)] = True
        markers[slot, : len(item.gather)] = torch.tensor(item.gather, dtype=torch.long)
        valid[slot, : len(item.gather)] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "markers": markers,
        "valid": valid,
    }


def readout_groups(lengths: list[int]) -> list[list[int]]:
    """Row indices for the type heads' padded reads of packed rows.

    One group while the padded grid is small (``READOUT_ONE_GRID`` tokens) or
    wastes at most ``READOUT_PADDING`` of the real tokens. Past that, rows in
    length order form groups in which no row pads by more than
    ``READOUT_PADDING``, so a short row never pads to a long one.
    """
    grid = len(lengths) * max(lengths)
    if grid <= READOUT_ONE_GRID or grid <= (1 + READOUT_PADDING) * sum(lengths):
        return [list(range(len(lengths)))]
    groups: list[list[int]] = []
    for row in sorted(range(len(lengths)), key=lengths.__getitem__):
        if groups and lengths[row] <= (1 + READOUT_PADDING) * lengths[groups[-1][0]]:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def padded_marker_logits(
    readout: TypeReadout,
    kind: str,
    items: list[RenderedItem],
    hidden: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Masked marker logits ``[rows, width]`` of ``items`` whose hidden states lie packed back to back."""
    lengths = [len(item.ids) for item in items]
    width = max(lengths)
    starts = torch.arange(len(items), device=device)[:, None] * width
    positions = (starts + torch.arange(width, device=device)[None, :])[
        torch.arange(width, device=device)[None, :]
        < torch.tensor(lengths, device=device)[:, None]
    ]
    rows = hidden.new_zeros((len(items) * width, hidden.shape[-1])).index_copy_(
        0, positions, hidden
    )
    batch = {name: value.to(device) for name, value in collate(items, 0).items()}
    scores: torch.Tensor = readout(
        kind,
        rows.view(len(items), width, hidden.shape[-1]),
        ~batch["attention_mask"],
        batch["markers"],
    )
    return scores.masked_fill(~batch["valid"], torch.finfo(torch.float32).min)


def packed_marker_logits(
    encode: Callable[[EncoderBatch], Any],
    readout: TypeReadout,
    kind: str,
    items: list[RenderedItem],
    device: torch.device,
    exit_layer: int,
) -> torch.Tensor:
    """Masked marker logits ``[rows, width]`` of rows of one type, their layer stack run packed.

    The rows run back to back without padding (the engine's packed layout);
    the type's head layers then read them padded with a key mask: by length
    group on CPUs (``readout_groups``), where padding costs compute, and as one
    grid on GPUs, where several reads' launches cost more than the padding.
    Packing changes the attention and GEMM shapes, so this serves approximate
    profiles, which also run the engine's reduced-precision copy when it loaded
    one.
    """
    lengths = [len(item.ids) for item in items]
    ids = torch.tensor(
        [token for item in items for token in item.ids], dtype=torch.long
    )
    packed = encode(
        EncoderBatch(
            input_ids=ids,
            attention_mask=None,
            lengths=lengths,
            branch=BRANCH[kind],
            reduced=True,
        )
    ).hidden[exit_layer]
    groups = (
        readout_groups(lengths) if device.type == "cpu" else [list(range(len(items)))]
    )
    if len(groups) == 1:
        return padded_marker_logits(readout, kind, items, packed, device)
    offsets = [0, *itertools.accumulate(lengths)]
    width = max(len(item.gather) for item in items)
    output = torch.full(
        (len(items), width), torch.finfo(torch.float32).min, device=device
    )
    for group in groups:
        tokens = torch.cat(
            [torch.arange(offsets[row], offsets[row + 1]) for row in group]
        ).to(device)
        scores = padded_marker_logits(
            readout,
            kind,
            [items[row] for row in group],
            packed.index_select(0, tokens),
            device,
        )
        output[torch.tensor(group, device=device), : scores.shape[1]] = scores
    return output


def marker_logits(
    encode: Callable[[EncoderBatch], Any],
    readout: TypeReadout,
    items: list[RenderedItem],
    pad_id: int,
    device: torch.device,
    exit_layer: int,
) -> torch.Tensor:
    """Masked marker logits ``[rows, width]`` of one physical batch.

    Each type present runs its layer stack over the whole batch, as the
    released runtime does, so every row's numerics equal the reference's.
    """
    batch = {name: value.to(device) for name, value in collate(items, pad_id).items()}
    padding = ~batch["attention_mask"]
    output = torch.empty(batch["markers"].shape, device=device, dtype=torch.float32)
    kinds = [item.task_type for item in items]
    for kind in KINDS:
        rows = [slot for slot, item_kind in enumerate(kinds) if item_kind == kind]
        if not rows:
            continue
        hidden = encode(
            EncoderBatch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                branch=BRANCH[kind],
            )
        ).hidden[exit_layer]
        index = torch.tensor(rows, device=device)
        scores = readout(
            kind,
            hidden.index_select(0, index),
            padding.index_select(0, index),
            batch["markers"].index_select(0, index),
        )
        output = output.index_copy(0, index, scores)
    return output.masked_fill(~batch["valid"], torch.finfo(torch.float32).min)
