"""Vela 2.0 layouts: truncation, 0.3B marker sequences and windows, decoder trees and span windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from vllm_srun.families.vela2.calibration import Calibration
from vllm_srun.families.vela2.decoder_layout import DecoderLayout
from vllm_srun.families.vela2.dispatch import Dispatcher
from vllm_srun.families.vela2.encoder_layout import (
    MARKERS,
    EncoderLayout,
    batch_indices,
    split_outputs,
)
from vllm_srun.families.vela2.layout import (
    SchemaTooLongError,
    Tokens,
    fit,
    rows_of,
    word_windows,
)
from vllm_srun.families.vela2.request import QuestionReader
from vllm_srun.testing.vela2 import calibration


@dataclass
class _Encoding:
    ids: list[int]
    offsets: list[tuple[int, int]]


def _encode(text: str) -> _Encoding:
    """One token per character (spaces included), so token positions equal code points."""
    return _Encoding(
        [10 + ord(c) % 1000 for c in text], [(i, i + 1) for i in range(len(text))]
    )


TOKENS = Tokens(encode=_encode, ids=lambda text: _encode(text).ids)
MARKER_IDS = {name: 2000 + index for index, name in enumerate(MARKERS)}


def _plan(state, questions, broad: bool = False):
    reader = QuestionReader(Calibration(calibration(decoder=broad)), broad_head=broad)
    plan = reader.read(state, questions)
    assert not plan.errors, plan.errors
    return plan


def encoder(max_length: int = 400, overlap: int = 16) -> EncoderLayout:
    config = {
        "marker_ids": MARKER_IDS,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    return EncoderLayout(
        {**config, "max_length": max_length, "window_overlap": overlap}
    )


def test_fit_shrinks_the_unread_part_first_then_the_read_parts() -> None:
    budget = fit([10], [], [("user", 100), ("context", 300)], 200, {"user"}, "context")
    assert budget.lengths == {"user": 100, "context": 85} and budget.protected_cut == 0
    budget = fit([10], [], [("user", 200), ("context", 300)], 200, {"user"}, "context")
    assert budget.lengths == {"user": 184, "context": 1} and budget.protected_cut == 16
    with pytest.raises(SchemaTooLongError):
        fit([500], [], [("user", 10)], 200, {"user"}, None)


def test_encoder_sequence_places_markers_pools_and_words() -> None:
    plan = _plan(
        {"request": "hi there", "answer": "ok"},
        {
            "c": {
                "type": "choice",
                "instructions": "Q",
                "criteria": {"a": "x", "b": None},
                "over": "request",
            },
            "p": {
                "type": "span",
                "instructions": "ignored",
                "criteria": {"L": "d"},
                "over": "request",
            },
        },
    )
    rows = rows_of(plan, TOKENS)
    (sequence,) = encoder().sequences(rows[0], TOKENS)
    ids = sequence.ids
    (entry,) = sequence.questions
    assert ids[0] == 2 and ids[-1] == 1
    assert ids[entry.query] == MARKER_IDS["[Q]"]
    assert [ids[p] for p in entry.options] == [MARKER_IDS["[O]"]] * 2 + [
        MARKER_IDS["[ABS]"]
    ]
    assert [ids[p] for p in sequence.labels] == [MARKER_IDS["[E]"]]
    start = ids.index(MARKER_IDS["[SEG_user]"])
    assert entry.pool == (start, start + 1 + len("hi there"))
    assert sequence.words.tolist() == [start + 1, start + 4]
    assert sequence.word_offsets.tolist() == [[0, 2], [3, 8]]


def test_encoder_windows_a_cut_labelled_part_and_combines_logits() -> None:
    text = " ".join(f"w{i}" for i in range(120))
    plan = _plan(
        {"request": text},
        {
            "c": {
                "type": "choice",
                "instructions": "Q",
                "criteria": {"a": "x", "b": "y"},
            },
            "s": {"type": "set", "instructions": "S", "criteria": {"x": "y"}},
            "p": {"type": "span", "instructions": "Spans?", "criteria": {"L": "d"}},
        },
    )
    rows = rows_of(plan, TOKENS)
    layout = encoder(max_length=300, overlap=64)
    sequences = layout.sequences(rows[0], TOKENS)
    assert len(sequences) > 1 and all(len(s.ids) <= 300 for s in sequences)
    covered = np.concatenate([s.word_index for s in sequences])
    assert set(covered.tolist()) == set(range(len(rows[0].part("user").words)))
    outputs = [
        (
            [
                np.array([float(i), 1.0, 0.0], np.float32),
                np.array([float(i)], np.float32),
            ],
            np.full((len(s.words), 1), float(i), np.float32),
        )
        for i, s in enumerate(sequences)
    ]
    raw = layout.combine(rows[0], sequences, outputs)
    assert raw.windows == len(sequences)
    assert raw.logits["c"][0] == pytest.approx(np.mean(range(len(sequences))))
    assert raw.logits["s"][0] == len(sequences) - 1
    assert raw.span.logits.shape == (len(rows[0].part("user").words), 1)


def test_batch_indices_and_split_outputs_round_trip() -> None:
    plan = _plan(
        "abc def",
        {
            "c": {
                "type": "choice",
                "instructions": "Q",
                "criteria": {"a": "x", "b": "y"},
            },
            "p": {
                "type": "span",
                "instructions": "Spans?",
                "criteria": {"L": "d", "M": "e"},
            },
        },
    )
    sequences = encoder().sequences(rows_of(plan, TOKENS)[0], TOKENS)
    indices = batch_indices(sequences * 2)
    assert indices["opt_index"].shape == (6, 3) and indices["ent_index"].shape == (4, 2)
    assert indices["opt_index"][3:, 2].tolist() == [1, 1, 1]
    span = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    split = split_outputs(sequences * 2, np.arange(6, dtype=np.float32), span)
    assert split[1][0][0].tolist() == [3, 4, 5]
    assert split[1][1].tolist() == [[10, 11], [14, 15]]
    empty = batch_indices([])
    assert empty["q_index"].tolist() == [[0, 0, 0, 1]]


def decoder(
    repeat: int = 2048, max_length: int = 16384, broad: bool = True
) -> DecoderLayout:
    config = {
        "max_length": max_length,
        "span_layout": {
            "hybrid_rmax": repeat,
            "window": repeat - 8,
            "stride": repeat - 16,
            "window_above": repeat,
        },
    }
    return DecoderLayout(
        config, Dispatcher(Calibration(calibration(decoder=True)), broad)
    )


def test_word_windows_start_at_words_and_cover_every_word() -> None:
    first = np.array([0, 3, 9, 14, 20, 26, 31], np.int32)
    windows = word_windows(first, 35, window=12, stride=8)
    assert windows[0][0] == 0 and windows[-1][1] == 35
    assert all(start in first for start, _ in windows)
    covered = {int(f) for start, end in windows for f in first if start <= f < end}
    assert covered == set(first.tolist())


def test_decoder_blocks_share_the_parts_and_read_endpoints() -> None:
    plan = _plan(
        {"request": "hello world", "answer": "done"},
        {
            "c": {
                "type": "choice",
                "instructions": "Q",
                "criteria": {"a": "x", "b": "y"},
                "over": "request",
            },
            "p": {"preset": "pii", "over": "request"},
            "h": {"preset": "halu"},
        },
        broad=True,
    )
    rows = rows_of(plan, TOKENS)
    trees, plans = decoder().trees(rows, TOKENS)
    assert len(rows) == 2 and len(trees) == 2
    assert trees[0].prefix == trees[1].prefix
    tree = trees[0]
    text = "".join(chr(t - 10) for t in tree.prefix)
    assert (
        text
        == 'Context:\n<segment role="user">\nhello world\n</segment>\n<segment role="answer">\ndone\n</segment>\n'
    )
    choice = tree.blocks[0]
    rendered = "".join(chr(t - 10) for t in choice.ids)
    assert rendered.endswith("Decision:") and choice.query == len(choice.ids) - 1
    assert [rendered[e] for e in choice.ends] == [">"] * 3
    pii = next(b for b in tree.blocks if b.question.id == "p")
    block = "".join(chr(t - 10) for t in pii.ids)
    assert (
        block.startswith("\n\nTask type: span\nTarget: user\n")
        and "\n\nText:\n" in block
    )
    assert [block[w] for w in pii.words] == ["h", "w"]
    assert len(pii.starts) == 17 and pii.head == "router"
    assert plans[0].span is pii and plans[1].span is trees[1].blocks[0]
    assert plans[0].tokens == len(tree.prefix) + len(choice.ids) + len(pii.ids)


def test_decoder_windows_long_span_targets() -> None:
    plan = _plan(
        {"request": " ".join(f"lorem{i}" for i in range(40))},
        {
            "p": {"type": "span", "instructions": "x", "criteria": {"L": "d"}},
            "c": {"type": "noul", "instructions": "y"},
        },
    )
    rows = rows_of(plan, TOKENS)
    trees, plans = decoder(repeat=64).trees(rows, TOKENS)
    (row,) = plans
    assert row.windows and row.span is not None and not row.span.read
    assert [b.question.id for b in row.blocks] == ["c"]
    assert len(trees) == 1 + len(row.windows)
    assert trees[0].blocks == [row.blocks[0], row.span] and not trees[0].window
    assert all(tree.window for tree in trees[1:])
    covered = np.concatenate([w.word_index for w in row.windows])
    assert set(covered.tolist()) == set(range(len(rows[0].part("user").words)))
    outputs = {id(row.blocks[0]): np.zeros(3)} | {
        id(w): np.ones((len(w.words), 1)) for w in row.windows
    }
    raw = DecoderLayout.combine(row, outputs)
    assert raw.windows == len(row.windows) and np.all(raw.span.logits == 1.0)


def test_decoder_routes_open_labels_to_the_broad_head_and_renames_halu_aliases() -> (
    None
):
    plan = _plan(
        {"request": "r", "answer": "a"},
        {
            "open": {
                "type": "span",
                "instructions": "x",
                "criteria": {"city": "a city"},
                "over": "request",
            },
            "alias": {
                "type": "span",
                "instructions": "x",
                "criteria": {"Hallucinated": "bad"},
            },
        },
        broad=True,
    )
    _, plans = decoder().trees(rows_of(plan, TOKENS), TOKENS)
    heads = {
        p.span.question.id: (p.span.head, p.span.question.names, p.span.alias)
        for p in plans
    }
    assert heads["open"] == ("broad", ["city"], None)
    assert heads["alias"] == (
        "router",
        ["unsupported"],
        {"unsupported": "Hallucinated"},
    )
    _, plans = decoder(broad=False).trees(rows_of(plan, TOKENS), TOKENS)
    assert {p.span.head for p in plans} == {"router"}


def test_rows_whose_questions_do_not_fit_map_to_none() -> None:
    plan = _plan(
        "text",
        {
            "c": {
                "type": "choice",
                "instructions": "Q" * 400,
                "criteria": {"a": "x", "b": "y"},
            }
        },
    )
    trees, plans = decoder(max_length=100).trees(rows_of(plan, TOKENS), TOKENS)
    assert trees == [] and plans == [None]
