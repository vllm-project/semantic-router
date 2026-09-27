"""The post-key diagnostic corrects only the typed item/answer distinction."""

from __future__ import annotations

import copy
import unittest

from jev_arena import arena_v3
from jev_arena.tests.test_arena_v3 import panel
from scripts.rank_postkey_answer_count_v3 import corrected_typed


class PostkeyAnswerCountTest(unittest.TestCase):
    def test_matches_corrected_ranker_on_real_final_shape(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            typed, _ = panel(Path(temporary), "candidate", 0.6)
        corrected = corrected_typed(
            arena_v3, typed, "example/candidate", "revision-candidate"
        )
        self.assertEqual(
            corrected,
            arena_v3._typed(typed, "example/candidate", "revision-candidate"),
        )
        self.assertEqual(typed["items"], 1600)
        self.assertEqual(typed["overall"]["n"], 2000)

    def test_rejects_wrong_answer_shape(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            typed, _ = panel(Path(temporary), "candidate", 0.6)
        malformed = copy.deepcopy(typed)
        malformed["by_family"]["evidence_join"].update(
            n=400, valid_n=400, correct_n=240
        )
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            corrected_typed(
                arena_v3, malformed, "example/candidate", "revision-candidate"
            )


if __name__ == "__main__":
    unittest.main()
