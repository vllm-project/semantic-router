"""A package's tokenizer through the ``tokenizers`` library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import MAX_LENGTH_EXCEEDED, QuestionError
from . import bounds


class Tokenizer:
    """The package tokenizer through the ``tokenizers`` library (no special tokens added).

    ``limit`` is the model's input limit: a text with more tokens fails its
    question with ``max_length_exceeded``, after only that many are read.
    """

    def __init__(self, backend: Any, pad_id: int, limit: int | None = None):
        self.backend = backend
        self.pad_id = pad_id
        self.limit = limit

    @classmethod
    def from_package(cls, root: Path, limit: int | None = None) -> Tokenizer:
        from tokenizers import Tokenizer as Backend

        backend = Backend.from_file(str(root / "tokenizer.json"))
        config_path = root / "tokenizer_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        pad_id = None
        for name in ("pad_token", "eos_token"):
            token = config.get(name)
            if isinstance(token, dict):
                token = token.get("content")
            if isinstance(token, str):
                pad_id = backend.token_to_id(token)
                if pad_id is not None:
                    break
        if pad_id is None:
            raise ValueError("the tokenizer needs a pad or EOS token")
        return cls(backend, pad_id, limit)

    def encode(self, text: str) -> list[int]:
        if self.limit is None:
            return list(self.backend.encode(text, add_special_tokens=False).ids)
        over = bounds.surely_over(self.backend, text, self.limit)
        read = None if over else bounds.read(self.backend, text, self.limit + 1)
        if read is None or read.tokens > self.limit:
            raise QuestionError(
                MAX_LENGTH_EXCEEDED,
                f"a text of the question exceeds max_length={self.limit}; no truncation",
            )
        return list(read.encoding.ids)
