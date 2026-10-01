"""calibrate_frozen renders CAL rows with the checkpoint's prompt version (Gemma 4: BOS)."""

from __future__ import annotations

import unittest

from training.model.decision_model import (
    BOS_PROMPT_VERSION,
    PROMPT_VERSION,
    encode,
)
from v2.release.calibrate_frozen import encode_rows


class _Tokenizer:
    bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [10 + ord(c) % 50 for c in text]


ROW = {
    "id": "cal-1",
    "state": "A short context.",
    "task_type": "choice",
    "instructions": "Pick one.",
    "options": [
        {"key": "a", "description": "first"},
        {"key": "b", "description": "second"},
    ],
    "label": 0,
    "family": "f",
}


class PromptVersionTest(unittest.TestCase):
    def test_plain_prompt_is_unchanged(self):
        tokenizer = _Tokenizer()
        (row,) = encode_rows([ROW], tokenizer, 4096, {"prompt_version": PROMPT_VERSION})
        self.assertEqual(row, encode(ROW, tokenizer, 4096))

    def test_bos_prompt_shifts_every_readout_position(self):
        tokenizer = _Tokenizer()
        plain = encode(ROW, tokenizer, 4096)
        (row,) = encode_rows(
            [ROW], tokenizer, 4096, {"prompt_version": BOS_PROMPT_VERSION}
        )
        self.assertEqual(row["ids"], [2, *plain["ids"]])
        self.assertEqual(
            row["candidate_positions"], [p + 1 for p in plain["candidate_positions"]]
        )
        self.assertEqual(row["query_position"], plain["query_position"] + 1)

    def test_unknown_prompt_version_is_refused(self):
        with self.assertRaises(ValueError):
            encode_rows([ROW], _Tokenizer(), 4096, {"prompt_version": "other"})


if __name__ == "__main__":
    unittest.main()
