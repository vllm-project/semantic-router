"""Vela 2.0 word units and span decoder."""

from __future__ import annotations

import numpy as np
from vllm_sr_runtime.families.vela2.words import (
    decode_spans,
    split_words,
    trim,
    words_of,
)


def test_words_keep_urls_emails_handles_and_split_cjk() -> None:
    text = "Mail tom.baker@example.com or see https://example.org/a?b=1. @alice 你好"
    words = [text[a:b] for a, b in split_words(text)]
    assert "tom.baker@example.com" in words
    assert "https://example.org/a?b=1" in words
    assert "@alice" in words
    assert words[-2:] == ["你", "好"]


def test_words_hyphenated_and_punctuation() -> None:
    text = "state-of-the-art, well_known (yes)!"
    words = [text[a:b] for a, b in split_words(text)]
    assert words == ["state-of-the-art", ",", "well_known", "(", "yes", ")", "!"]


def test_words_take_the_first_overlapping_token_and_drop_uncovered_words() -> None:
    text = "ab cd ef"
    offsets = np.array([[0, 2], [2, 2], [3, 4], [4, 5], [6, 8]], np.int32)
    words = words_of(text, offsets)
    assert words.offsets.tolist() == [[0, 2], [3, 5], [6, 8]]
    assert words.first.tolist() == [0, 2, 4]
    assert len(words_of(text, offsets[:1])) == 1
    assert len(words_of("", offsets)) == 0


def test_trim_edges_and_url_tails() -> None:
    text = '("hello")'
    assert trim(text, 0, len(text)) == (2, 7)
    url = "see www.example.com/x)."
    start = url.index("www")
    assert url[slice(*trim(url, start, len(url)))] == "www.example.com/x"


def _decode(text: str, rows: list[list[float]], threshold: float = 0.5):
    offsets = np.array([[a, b] for a, b in split_words(text)], np.int32)
    return decode_spans(
        np.array(rows, np.float64), offsets, ["A", "B"], text, threshold
    )


def test_decode_joins_neighbours_and_averages_probabilities() -> None:
    spans = _decode(
        "Tom Baker said hi", [[0.9, 0.1], [0.7, 0.1], [0.1, 0.2], [0.1, 0.1]]
    )
    assert spans == [
        {"start": 0, "end": 9, "label": "A", "probability": spans[0]["probability"]}
    ]
    assert abs(spans[0]["probability"] - 0.8) < 1e-6


def test_decode_compares_in_float32() -> None:
    # 0.5000000001 rounds to 0.5 in float32 and is not above a 0.5 threshold.
    assert _decode("x", [[0.5000000001, 0.0]]) == []


def test_decode_votes_inside_a_unit_and_fills_gaps() -> None:
    # One unit (an e-mail address) split into words with conflicting labels: the summed probability wins.
    text = "a.b@c.de"
    offsets = np.array([[0, 3], [3, 4], [4, 8]], np.int32)
    probabilities = np.array([[0.9, 0.0], [0.0, 0.0], [0.0, 0.6]])
    spans = decode_spans(probabilities, offsets, ["A", "B"], text, 0.5)
    assert [(s["label"], s["start"], s["end"]) for s in spans] == [("A", 0, 8)]


def test_decode_without_words_is_empty() -> None:
    assert (
        decode_spans(np.zeros((0, 2)), np.zeros((0, 2), np.int32), ["A", "B"], "", 0.5)
        == []
    )
