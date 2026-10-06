"""Vela 2.0 word units and span decoder."""

from __future__ import annotations

import numpy as np
from vllm_srun.families.vela2.words import (
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


# The packages' loops, kept here as the oracle the vectorized versions must equal.


def _oracle_words(text, offsets):
    starts, ends = offsets[:, 0], offsets[:, 1]
    valid = ends > starts
    reach = np.maximum.accumulate(np.where(valid, ends, -1))
    out, first = [], []
    for a, b in split_words(text):
        t = int(np.searchsorted(reach, a, side="right"))
        while t < len(offsets) and not valid[t]:
            t += 1
        if t < len(offsets) and starts[t] < b:
            out.append([a, b])
            first.append(t)
    return out, first


def _oracle_decode(probs, offs, names, text, thr):
    from vllm_srun.families.vela2.words import _UNIT

    probs = np.asarray(probs, dtype=np.float32)
    count = len(offs)
    k = probs.argmax(1)
    pk = probs[np.arange(count), k]
    lab = np.where(pk > thr, k, -1)
    u = np.full(len(text) + 1, -1, dtype=np.int64)
    for index, match in enumerate(_UNIT.finditer(text)):
        u[match.start() : match.end()] = index
    tu = np.array([u[a] if a < len(text) else -1 for a in offs[:, 0]])
    t = 0
    while t < count:
        if tu[t] < 0:
            t += 1
            continue
        t2 = t
        while t2 + 1 < count and tu[t2 + 1] == tu[t]:
            t2 += 1
        members = list(range(t, t2 + 1))
        labelled = [i for i in members if lab[i] >= 0]
        if labelled:
            if len({int(lab[i]) for i in labelled}) > 1:
                votes = {}
                for i in labelled:
                    votes[int(lab[i])] = votes.get(int(lab[i]), 0.0) + float(pk[i])
                winner = max(votes, key=votes.get)
                for i in labelled:
                    lab[i] = winner
            for c in {int(lab[i]) for i in labelled}:
                pos = [i for i in labelled if lab[i] == c]
                for i in members:
                    if pos[0] < i < pos[-1] and lab[i] < 0:
                        lab[i] = c
        t = t2 + 1
    spans, cur = [], None
    for t in range(count):
        c = int(lab[t])
        if cur is not None and c == cur[0]:
            cur[2] = int(offs[t, 1])
            cur[3].append(t)
            continue
        if cur is not None:
            spans.append(cur)
        cur = [c, int(offs[t, 0]), int(offs[t, 1]), [t]] if c >= 0 else None
    if cur is not None:
        spans.append(cur)
    out = []
    for c, first, last, ts in spans:
        s, e = trim(text, first, last)
        if e > s:
            out.append(
                {
                    "start": s,
                    "end": e,
                    "label": names[c],
                    "probability": float(np.mean(probs[ts, c])),
                }
            )
    return out


PIECES = [
    "Tom",
    "o'Neil",
    "tom.b@ex.com",
    "(hi)",
    "https://a.b/c).",
    "你好",
    "art-x",
    "42",
    "!",
    "x_y",
]


def test_words_and_decoder_equal_the_packages_loops() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        text = " ".join(rng.choice(PIECES, size=rng.integers(1, 40)))
        size = min(len(text) - 1, int(rng.integers(1, 30)))
        cuts = np.sort(rng.choice(np.arange(1, len(text)), size=size, replace=False))
        bounds = np.concatenate(([0], cuts, [len(text)]))
        offsets = np.stack([bounds[:-1], bounds[1:]], 1).astype(np.int32)
        empty = rng.random(len(offsets)) < 0.1
        offsets[empty, 1] = offsets[empty, 0]
        words = words_of(text, offsets)
        expected, first = _oracle_words(text, offsets)
        assert words.offsets.tolist() == expected and words.first.tolist() == first
        if not len(words):
            continue
        probs = rng.random((len(words), 3)) ** 3
        names = ["A", "B", "C"]
        for threshold in (0.05, 0.3, 0.7):
            assert decode_spans(probs, words.offsets, names, text, threshold) == (
                _oracle_decode(probs, words.offsets, names, text, threshold)
            )
