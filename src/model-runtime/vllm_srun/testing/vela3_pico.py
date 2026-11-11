"""Test helpers for the Vela 3.1 pico packages (``vela3-pico/1``).

``byte_tokenizer`` is the deterministic tokenizer of the recorded byte-level
cases (``data/render31_cases.json``): one token per UTF-8 byte, built from
code, so a recorded token ID never depends on a trained vocabulary.
"""

from __future__ import annotations

from typing import Any

from .vela3 import MARKERS


def byte_tokenizer() -> Any:
    """One token per UTF-8 byte (IDs 1 to 256; 0 is ``<|endoftext|>``), offsets in code points."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<|endoftext|>": 0, **{char: i + 1 for i, char in enumerate(alphabet)}}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


# The byte tokenizer's marker IDs and special tokens, as a package's config names them.
BYTE_MARKERS = {marker: 257 + index for index, marker in enumerate(MARKERS)}
BYTE_SPECIAL = {"bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0}
