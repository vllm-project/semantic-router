"""The released span decoder's rules (span_decode_v2.decode_v2 with extend=False), word by word."""

from __future__ import annotations

import re

import numpy as np
from vllm_srun.families.vela3_d3.decode import decode, evidence_spans, units


def words(text: str) -> np.ndarray:
    return np.asarray(
        [
            (m.start(), m.end())
            for m in re.finditer(r"\S+?(?=[\s,.;:!?]|$)|[,.;:!?]", text)
        ]
    )


def spans_of(text, probs, names, threshold):
    return [
        (s["label"], text[s["start"] : s["end"]])
        for s in decode(probs, words(text), names, text, threshold)
    ]


def test_a_word_takes_its_best_label_only_strictly_above_the_threshold() -> None:
    text = "call Tom now"
    probs = [[0.1, 0.2], [0.5, 0.7], [0.5, 0.5]]
    assert spans_of(text, probs, ["A", "B"], 0.5) == [("B", "Tom")]
    assert spans_of(text, probs, ["A", "B"], 0.7) == []


def test_contiguous_words_of_one_label_join_and_different_labels_never_do() -> None:
    text = "Tom Baker Paris France"
    probs = [[0.9, 0.0], [0.8, 0.0], [0.0, 0.9], [0.0, 0.8]]
    assert spans_of(text, probs, ["PERSON", "GPE"], 0.5) == [
        ("PERSON", "Tom Baker"),
        ("GPE", "Paris France"),
    ]


def test_one_unit_votes_for_the_label_with_the_largest_summed_probability() -> None:
    text = "tom.baker@example.com"
    offsets = np.asarray([(0, 3), (3, 4), (4, 9), (9, 10), (10, 21)])
    probs = [[0.9, 0.0], [0.0, 0.6], [0.0, 0.6], [0.0, 0.0], [0.0, 0.6]]
    got = decode(probs, offsets, ["PERSON", "EMAIL"], text, 0.5)
    assert got == [{"start": 0, "end": 21, "label": "EMAIL"}]


def test_unlabelled_words_between_two_of_one_label_inside_a_unit_fill() -> None:
    text = "pavel_200015@summit15.example.org"
    offsets = np.asarray([(0, 5), (5, 6), (6, 12), (12, 13), (13, 33)])
    probs = [[0.9], [0.1], [0.2], [0.1], [0.9]]
    assert decode(probs, offsets, ["EMAIL"], text, 0.5) == [
        {"start": 0, "end": 33, "label": "EMAIL"}
    ]
    # Across units nothing fills: "a" and "c" are separate units.
    assert spans_of("a b c", [[0.9], [0.1], [0.9]], ["X"], 0.5) == [
        ("X", "a"),
        ("X", "c"),
    ]


def test_spans_are_trimmed_of_whitespace_brackets_and_quotes() -> None:
    text = "see [ref] here"
    offsets = np.asarray([(0, 3), (3, 5), (5, 8), (8, 9), (9, 14)])
    probs = [[0.0], [0.9], [0.9], [0.9], [0.0]]
    assert decode(probs, offsets, ["X"], text, 0.5) == [
        {"start": 5, "end": 8, "label": "X"}
    ]
    only_edge = np.asarray([(0, 1)])
    assert decode([[0.9]], only_edge, ["X"], "[", 0.5) == []


def test_every_cjk_character_is_its_own_unit() -> None:
    text = "王小明"
    assert list(units(text)[:3]) == [0, 1, 2]
    offsets = np.asarray([(0, 1), (1, 2), (2, 3)])
    probs = [[0.9, 0.0], [0.0, 0.8], [0.0, 0.9]]
    assert decode(probs, offsets, ["A", "B"], text, 0.5) == [
        {"start": 0, "end": 1, "label": "A"},
        {"start": 1, "end": 3, "label": "B"},
    ]


def test_the_released_decoders_recorded_outputs() -> None:
    import json
    from importlib.resources import files

    recorded = json.loads(
        files("vllm_srun.testing")
        .joinpath("data/vela3_decode_cases.json")
        .read_text("utf-8")
    )["decode"]
    assert len(recorded) == 42
    for case in recorded:
        got = decode(
            np.asarray(case["probabilities"]),
            np.asarray(case["offsets"]),
            case["names"],
            case["text"],
            case["threshold"],
        )
        assert got == case["spans"], case["text"]


def test_the_unit_class_starts_its_compatibility_block_at_u_f900() -> None:
    # U+E000 lies between U+8C48 and U+F900: not CJK to the released decoder, so a URL unit runs through it.
    text = "https://a.example/xy"
    assert units(text)[len(text) - 1] == 0
    from vllm_srun.families.vela3_d3.decode import CJK

    assert "豈-﫿" in CJK and "豈" not in CJK


def test_zero_width_words_and_no_words_decode_nothing() -> None:
    assert decode(np.zeros((0, 1)), np.zeros((0, 2)), ["X"], "", 0.5) == []
    offsets = np.asarray([(0, 3), (3, 3), (4, 7)])
    probs = [[0.9], [0.9], [0.9]]
    assert decode(probs, offsets, ["X"], "abc def", 0.5) == [
        {"start": 0, "end": 7, "label": "X"}
    ]


EVIDENCE = "Adults take 4 grams. Never more!\nKids take less.\n\nUnrelated text here. More of it."


def evidence(kind, scores, threshold=0.5, gap=0, min_chars=0):
    offsets = words(EVIDENCE)
    probs = np.asarray(scores, dtype=np.float64)[:, None]
    assert len(probs) == len(offsets)
    return [
        EVIDENCE[s["start"] : s["end"]]
        for s in evidence_spans(
            probs,
            offsets,
            ["evidence"],
            EVIDENCE,
            threshold,
            {"name": kind, "gap": gap, "min_chars": min_chars},
        )
    ]


def scores(high: set[int]) -> list[float]:
    return [0.9 if i in high else 0.1 for i in range(len(words(EVIDENCE)))]


def test_evidence_keeps_sentences_whose_mean_word_probability_passes() -> None:
    first_sentence = set(range(5))
    assert evidence("sent", scores(first_sentence)) == ["Adults take 4 grams."]
    assert evidence("sent", scores(first_sentence | {5, 6, 7}), gap=2) == [
        "Adults take 4 grams. Never more!"
    ]


def test_evidence_fills_a_paragraph_when_half_its_sentences_pass() -> None:
    first_two = set(range(8))
    assert evidence("sentfill", scores(first_two)) == [
        "Adults take 4 grams. Never more!"
    ]
    assert evidence("para", scores(set(range(8)))) == [
        "Adults take 4 grams. Never more!"
    ]


def test_evidence_word_decoding_drops_short_spans() -> None:
    assert evidence("plain", scores({0, 1})) == ["Adults take"]
    assert evidence("plain", scores({0, 1}), min_chars=20) == []


def test_an_answer_leaves_out_the_comma_the_decoder_keeps_before_a_value() -> None:
    from vllm_srun.families.vela3_d3.span_read import SpanItem, answer

    # The tokenizer joins the comma to the letters after it, so the comma's
    # word reads the e-mail's label and the released decoder opens the span on it.
    text = "Park,mina@example.com"
    offsets = np.array([[0, 4], [4, 5], [5, 21]])
    logits = np.array([[-6.0], [6.0], [6.0]])
    assert decode(1 / (1 + np.exp(-logits)), offsets, ["EMAIL"], text, 0.5) == [
        {"start": 4, "end": 21, "label": "EMAIL"}
    ]
    item = SpanItem(
        question_id="q",
        task_type="span",
        ids=[],
        keys=[],
        descriptions=[],
        names=("EMAIL",),
        text=text,
    )
    spans = answer(item, offsets, logits)["spans"]
    assert [(s["start"], s["end"], s["text"]) for s in spans] == [
        (5, 21, "mina@example.com")
    ]
