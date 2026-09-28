"""Offline checks for the published reference decision/offset projection."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "halu_export", Path(__file__).with_name("export.py")
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class HaluReferenceTests(unittest.TestCase):
    def test_strict_threshold_and_unicode_bytes(self):
        answer = "é 猫 hello"
        spans = exporter.answer_spans(
            answer,
            [None, 1, 1, 1],
            [(0, 0), (0, 1), (2, 3), (4, 9)],
            [1.0, 0.5, 0.51, 0.49],
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["text"], "猫")
        self.assertEqual((spans[0]["start"], spans[0]["end"]), (3, 6))
        self.assertEqual(answer.encode()[3:6].decode(), spans[0]["text"])


if __name__ == "__main__":
    unittest.main()
