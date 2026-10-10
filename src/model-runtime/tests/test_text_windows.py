"""Reject, truncate and window planning, against the legacy binding's window contract."""

import numpy as np
import pytest
from tokenizers import Tokenizer
from vllm_srun.testing.task_heads import BOS, EOS, encoder_tokenizer
from vllm_srun.text.windows import (
    Envelope,
    InputTooLongError,
    Window,
    encode,
    fit_prefix,
    merge_token_windows,
    plan_windows,
    reduce_max,
    truncate,
)

LONG = "My name is Tom Baker and my email is tom.baker@example.com. " * 6


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    root = tmp_path_factory.mktemp("tokenizer")
    encoder_tokenizer(root)
    return Tokenizer.from_file(str(root / "tokenizer.json"))


def test_envelope_is_the_tokenizers_fixed_framing(tokenizer):
    envelope = Envelope.of(tokenizer)
    assert envelope == Envelope((BOS,), (EOS,))
    encoded = encode(tokenizer, envelope, LONG)
    assert encoded.framed() == tokenizer.encode(LONG, add_special_tokens=True).ids
    assert encoded.tokens == len(encoded.content) + 2
    assert all(LONG[start:end] for start, end in encoded.offsets)


def test_a_tokenizer_without_special_tokens_has_an_empty_envelope():
    from tokenizers import models

    plain = Tokenizer(models.WordLevel({"a": 0, "[UNK]": 1}, unk_token="[UNK]"))
    assert Envelope.of(plain) == Envelope((), ())


def windows_of(length, size=512, overlap=255):
    return plan_windows(length, Envelope((2,), (1,)), size, overlap)


@pytest.mark.parametrize("length", [1, 509, 510, 511, 512, 765, 766, 32766])
def test_windows_cover_every_token_like_the_legacy_planner(length):
    windows = windows_of(length)
    covered = np.zeros(length, dtype=bool)
    for index, window in enumerate(windows):
        assert window.start == index * 255
        assert window.end - window.start <= 510
        covered[window.start : window.end] = True
        if index + 1 < len(windows):
            assert window.end < length
    assert covered.all() and windows[-1].end == length


def test_window_boundaries_and_budgets_match_the_legacy_planner():
    assert [(w.start, w.end) for w in windows_of(766)] == [
        (0, 510),
        (255, 765),
        (510, 766),
    ]
    assert len(plan_windows(7, Envelope((2,), (1,)), 5, 0)) == 3
    for length, size, overlap in [(0, 512, 255), (1, 2, 0), (1, 512, 510)]:
        with pytest.raises(ValueError):
            windows_of(length, size, overlap)


def test_token_windows_keep_the_observation_with_most_context():
    windows = [Window(0, 5), Window(2, 7)]
    rows = [np.tile([1.0, 0.0], (7, 1)), np.tile([0.0, 100.0], (7, 1))]
    merged = merge_token_windows(windows, rows, prefix=1, content_tokens=7)
    assert merged.tolist() == [[1.0, 0.0]] * 4 + [[0.0, 100.0]] * 3
    with pytest.raises(ValueError, match="every content token"):
        merge_token_windows([Window(0, 2)], [np.zeros((4, 2))], 1, 3)


def test_reduction_is_the_per_label_maximum():
    assert reduce_max([[0.1, 0.9], [0.4, 0.2]]) == [0.4, 0.9]


def test_truncation_matches_the_tokenizers_own(tokenizer):
    envelope = Envelope.of(tokenizer)
    encoded = encode(tokenizer, envelope, LONG)
    clone = Tokenizer.from_str(tokenizer.to_str())
    clone.enable_truncation(max_length=12)
    assert truncate(encoded, 12) == clone.encode(LONG, add_special_tokens=True).ids
    with pytest.raises(InputTooLongError):
        truncate(encoded, 1)


def test_token_heads_read_the_longest_prefix_that_fits(tokenizer):
    envelope = Envelope.of(tokenizer)
    encoded = encode(tokenizer, envelope, LONG)
    same, cut = fit_prefix(tokenizer, encoded, encoded.tokens)
    assert same is encoded and not cut
    prefix, cut = fit_prefix(tokenizer, encoded, 20)
    assert cut and prefix.tokens <= 20 and LONG.startswith(prefix.text)
    assert prefix.text == LONG[: encoded.offsets[17][1]]
