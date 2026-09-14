"""The public install skill is generated from one complete repository source."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import sync_public_skill as skill  # noqa: E402


class PublicSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.destination = Path(self.temp.name) / "public"
        (self.source / "references").mkdir(parents=True)
        (self.source / "SKILL.md").write_text(
            "---\nname: vllm-sr-agent-operations\ndescription: Test\n---\n"
            "[Details](references/details.md#configure)\n"
            "[External](https://example.com/api) and [Anchor](#here)\n"
        )
        (self.source / "references/details.md").write_text("[Back](../SKILL.md)\n")

    def test_renders_name_and_reference_links_without_other_body_changes(self):
        documents = skill.published_files(self.source)
        entry = documents[Path("SKILL.md")]
        self.assertIn("name: vllm-sr\n", entry)
        self.assertIn(
            f"[Details]({skill.PUBLIC_ORIGIN}references/details.md#configure)", entry
        )
        self.assertIn("[External](https://example.com/api) and [Anchor](#here)", entry)
        self.assertEqual(
            documents[Path("references/details.md")],
            f"[Back]({skill.PUBLIC_ORIGIN}SKILL.md)\n",
        )

    def test_check_detects_stale_and_missing_files_without_mutating(self):
        self.assertTrue(skill.sync(self.source, self.destination, check=False))
        self.assertEqual(skill.sync(self.source, self.destination, check=True), [])
        original = (self.destination / "SKILL.md").read_bytes()
        with (self.source / "SKILL.md").open("a") as stream:
            stream.write("New instruction.\n")
        (self.destination / "references/details.md").unlink()
        self.assertEqual(len(skill.sync(self.source, self.destination, check=True)), 2)
        self.assertEqual((self.destination / "SKILL.md").read_bytes(), original)
        self.assertFalse((self.destination / "references/details.md").exists())

    def test_missing_source_reference_is_rejected_before_writing(self):
        (self.source / "references/details.md").unlink()
        with self.assertRaisesRegex(ValueError, "not a published skill document"):
            skill.sync(self.source, self.destination, check=False)
        self.assertFalse(self.destination.exists())

    def test_reference_cannot_escape_the_authored_skill(self):
        (self.source / "references/details.md").write_text(
            "[Private](../../other.md)\n"
        )
        with self.assertRaisesRegex(ValueError, "escapes skill source"):
            skill.published_files(self.source)

    def test_stale_generated_reference_is_reported_then_removed(self):
        skill.sync(self.source, self.destination, check=False)
        stale = self.destination / "references/stale.md"
        stale.write_text("Old workflow")
        self.assertIn(
            "unexpected generated document: references/stale.md",
            skill.sync(self.source, self.destination, check=True),
        )
        self.assertTrue(stale.exists())
        skill.sync(self.source, self.destination, check=False)
        self.assertFalse(stale.exists())

    def test_repository_publication_is_complete_and_current(self):
        self.assertEqual(skill.sync(skill.SOURCE, skill.DESTINATION, check=True), [])
        source_refs = list((skill.SOURCE / "references").glob("*.md"))
        self.assertEqual(
            {path.name for path in source_refs},
            {"configuration-loop.md", "deployment-loop.md", "evaluation-loop.md"},
        )
        source_entry = (skill.SOURCE / "SKILL.md").read_text()
        for path in source_refs:
            self.assertIn(f"(references/{path.name})", source_entry)
        self.assertTrue((skill.SOURCE / "agents/openai.yaml").is_file())
        public_entry = (skill.DESTINATION / "SKILL.md").read_text()
        self.assertNotIn("](references/", public_entry)

    def test_harness_checks_the_explicit_publication_source(self):
        makefile = (skill.ROOT / "tools/make/agent.mk").read_text()
        self.assertIn("tools/agent/scripts/sync_public_skill.py --check", makefile)
        hook = (skill.ROOT / ".pre-commit-config.yaml").read_text()
        self.assertIn("id: public-agent-skill", hook)
        self.assertIn("tools/agent/scripts/sync_public_skill.py --check", hook)


if __name__ == "__main__":
    unittest.main()
