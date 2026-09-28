"""Optional comparator note on the card and its evaluation page (stdlib, fixture reports)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from v2.release import card
from v2.release.tests.test_release import REPORTS, ROSTER, facts

NOTE = "GLiNER2.5-Decide declares Apache-2.0 in its model card metadata; its repository has no LICENSE file."


class ComparatorNoteTest(unittest.TestCase):
    def render(self, text: dict) -> tuple[str, str, list[str]]:
        entries = [
            {
                "key": "cand",
                "role": "candidate",
                "report": str(REPORTS / "bosun.json"),
                "label": "DEV2.0-0.6B",
            },
            {
                "key": "kai1",
                "role": "own-1.0",
                "report": str(REPORTS / "kai1.json"),
                "repo_id": "llm-semantic-router/Decision-1.0-Kai-0.6B",
                "label": "Decision 1.0 Kai",
            },
            {
                "key": "gliner25",
                "role": "peer",
                "report": str(REPORTS / "gliner25.json"),
                "repo_id": "fastino/GLiNER2.5-Decide",
            },
        ]
        with tempfile.TemporaryDirectory() as scratch:
            out, banner = Path(scratch) / "pkg", Path(scratch) / "banner.png"
            banner.write_bytes(b"\x89PNG\r\n\x1a\n")
            card.build_card(
                entries=entries,
                roster=ROSTER,
                paired=None,
                facts=facts(),
                text={"tagline": "A decision model.", "limitations": [], **text},
                banner=banner,
                work=Path(scratch) / "work",
                output=out,
            )
            readme = (out / "README.md").read_text()
            evaluation = (out / "evaluation/EVALUATION.md").read_text()
            files = {
                p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
            }
            files |= {"LICENSE", "NOTICE", "ATTRIBUTIONS.md"}
            return readme, evaluation, card.check_rendered(readme, files)

    def test_note_follows_the_rank_scope_on_card_and_evaluation_page(self):
        readme, evaluation, problems = self.render({"comparator_note": NOTE})
        self.assertEqual(problems, [])
        self.assertIn(f"Ranks include only the models shown. {NOTE} [Methods", readme)
        self.assertIn(f"are not shown on this card. {NOTE}", evaluation)

    def test_without_a_note_the_text_is_unchanged(self):
        readme, evaluation, _ = self.render({})
        self.assertIn("Ranks include only the models shown. [Methods", readme)
        self.assertIn("are not shown on this card.\n", evaluation)


if __name__ == "__main__":
    unittest.main()
