from __future__ import annotations

import unittest

from v2.data.embed_scan import CJK_WINDOW_CHARS, WINDOW_CHARS, state_text, windows


class EmbedScanWindowTest(unittest.TestCase):
    def test_short_text_is_one_window_and_tiny_text_is_skipped(self) -> None:
        self.assertEqual(windows("x" * 10), [])
        self.assertEqual(len(windows("word " * 50)), 1)

    def test_long_latin_text_windows_cover_the_end(self) -> None:
        text = " ".join(f"w{i}" for i in range(2000))
        parts = windows(text)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) == WINDOW_CHARS for part in parts))
        self.assertTrue(parts[-1].endswith("w1999"))

    def test_cjk_text_uses_short_windows(self) -> None:
        parts = windows("决策模型需要证据。" * 200)
        self.assertTrue(all(len(part) == CJK_WINDOW_CHARS for part in parts))

    def test_structured_state_is_serialized_stably(self) -> None:
        self.assertEqual(state_text({"b": 1, "a": "x"}), '{"a": "x", "b": 1}')


if __name__ == "__main__":
    unittest.main()
