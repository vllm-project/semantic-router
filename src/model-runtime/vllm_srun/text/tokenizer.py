"""A package's tokenizer through the ``tokenizers`` library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Tokenizer:
    """The package tokenizer through the ``tokenizers`` library (no special tokens added)."""

    def __init__(self, backend: Any, pad_id: int):
        self.backend = backend
        self.pad_id = pad_id

    @classmethod
    def from_package(cls, root: Path) -> Tokenizer:
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
        return cls(backend, pad_id)

    def encode(self, text: str) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=False).ids)
