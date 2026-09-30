from __future__ import annotations

import math
import unittest
from pathlib import Path

from v2.serving.session import label_ids, parse_panel, read_labels


class ParsePanelTest(unittest.TestCase):
    def test_default_and_explicit_concurrency(self) -> None:
        self.assertEqual(
            parse_panel("typed-final:/p/a.jsonl:/p/b.jsonl:1600", 1),
            ("typed-final", Path("/p/a.jsonl"), Path("/p/b.jsonl"), 1600, 1),
        )
        self.assertEqual(
            parse_panel("public231-c32:/p/a.jsonl:/p/b.jsonl:231:32", 1)[-1], 32
        )


class LabelTokenTest(unittest.TestCase):
    def test_read_labels_uses_token_id_keys(self) -> None:
        value = {
            "choices": [
                {
                    "logprobs": {
                        "tokens": ["token_id:9"],
                        "top_logprobs": [
                            {"token_id:9": math.log(0.75), "token_id:4": math.log(0.25)}
                        ],
                    }
                }
            ]
        }
        probs = read_labels(value, [9, 4, 7])
        self.assertAlmostEqual(probs[0], 0.75)
        self.assertAlmostEqual(probs[1], 0.25)
        self.assertEqual(probs[2], 0.0)

    def test_labels_must_be_single_tokens(self) -> None:
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [len(text)] * (1 if len(text) < 4 else 2)

        self.assertEqual(label_ids(Tokenizer(), ("yes", "no")), [3, 2])
        with self.assertRaises(ValueError):
            label_ids(Tokenizer(), ("maybe", "no"))


if __name__ == "__main__":
    unittest.main()
