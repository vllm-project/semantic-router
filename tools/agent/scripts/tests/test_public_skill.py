"""The public install skill is generated from one complete repository source."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

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
        self.assertIn("[External](https://example.com/api)", entry)
        self.assertIn(f"[Anchor]({skill.PUBLIC_ORIGIN}SKILL.md#here)", entry)
        self.assertEqual(
            documents[Path("references/details.md")],
            f"[Back]({skill.PUBLIC_ORIGIN}SKILL.md)\n",
        )

    def test_downloaded_references_use_absolute_urls_for_each_link_form(self):
        (self.source / "references/details.md").write_text(
            '[Home](../SKILL.md "Start")\n'
            "[Local section](#configure)\n"
            "[Site page](/docs/installation)\n"
            "[External](//example.com/guide)\n"
            "[More][entry]\n"
            '[entry]: <../SKILL.md#next> "Next step"\n'
        )
        skill.sync(self.source, self.destination, check=False)
        # Install a reference by itself, outside both the repository and the
        # published directory; every destination must remain usable remotely.
        installed = Path(self.temp.name) / "installed-reference.md"
        installed.write_bytes((self.destination / "references/details.md").read_bytes())
        content = installed.read_text()
        self.assertIn(f'[Home]({skill.PUBLIC_ORIGIN}SKILL.md "Start")', content)
        self.assertIn(
            f"[Local section]({skill.PUBLIC_ORIGIN}references/details.md#configure)",
            content,
        )
        self.assertIn("[Site page](https://vllm-sr.ai/docs/installation)", content)
        self.assertIn("[External](https://example.com/guide)", content)
        self.assertIn(
            f'[entry]: <{skill.PUBLIC_ORIGIN}SKILL.md#next> "Next step"', content
        )

    def test_absolute_same_site_reference_must_also_be_published(self):
        (self.source / "references/details.md").write_text(
            f"[Missing]({skill.PUBLIC_ORIGIN}references/missing.md)\n"
        )
        with self.assertRaisesRegex(ValueError, "not a published skill document"):
            skill.published_files(self.source)

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
            {
                "configuration-loop.md",
                "deployment-loop.md",
                "evaluation-loop.md",
                "recipe-tuning.md",
            },
        )
        # Specialized references may be discovered through another reference;
        # every published document must still be reachable from the entrypoint.
        documents = skill.published_files(skill.SOURCE)
        pending = [Path("SKILL.md")]
        reached = set()
        while pending:
            path = pending.pop()
            if path in reached:
                continue
            reached.add(path)
            for pattern in (skill.MARKDOWN_LINK, skill.REFERENCE_LINK):
                for match in pattern.finditer(documents[path]):
                    target = match[2].strip("<>")
                    self.assertIn(urlsplit(target).scheme, {"http", "https"})
                    target = target.split("#", 1)[0]
                    if target.startswith(skill.PUBLIC_ORIGIN):
                        linked = Path(target[len(skill.PUBLIC_ORIGIN) :])
                        self.assertIn(linked, documents)
                        pending.append(linked)
        self.assertEqual(reached, set(documents))
        self.assertTrue((skill.SOURCE / "agents/openai.yaml").is_file())
        public_entry = (skill.DESTINATION / "SKILL.md").read_text()
        self.assertNotIn("](references/", public_entry)

    def test_authored_skill_also_works_when_installed_without_its_repository(self):
        for path in [
            skill.SOURCE / "SKILL.md",
            *(skill.SOURCE / "references").glob("*.md"),
        ]:
            content = path.read_text()
            for pattern in (skill.MARKDOWN_LINK, skill.REFERENCE_LINK):
                for match in pattern.finditer(content):
                    target = match[2].strip("<>")
                    self.assertIn(urlsplit(target).scheme, {"http", "https"}, str(path))

    def test_deployment_guidance_uses_host_capabilities_without_vendor_branch(self):
        documents = skill.published_files(skill.SOURCE)
        entry = documents[Path("SKILL.md")]
        deployment = " ".join(documents[Path("references/deployment-loop.md")].split())
        self.assertIn("installed `vllm-sr serve --help`", entry)
        self.assertIn("explicitly select the `--platform` value", deployment)
        self.assertIn(
            "The CLI does not select a GPU platform automatically.", deployment
        )
        self.assertIn(
            "Check inherited platform, image, and runtime overrides", deployment
        )
        self.assertIn("actual devices in the live inventory", deployment)
        for path, content in documents.items():
            self.assertNotIn("rocm", content.lower(), str(path))
            self.assertNotIn("--platform amd", content, str(path))

    def test_harness_checks_the_explicit_publication_source(self):
        makefile = (skill.ROOT / "tools/make/agent.mk").read_text()
        self.assertIn("tools/agent/scripts/sync_public_skill.py --check", makefile)
        hook = (skill.ROOT / ".pre-commit-config.yaml").read_text()
        self.assertIn("id: public-agent-skill", hook)
        self.assertIn("tools/agent/scripts/sync_public_skill.py --check", hook)


if __name__ == "__main__":
    unittest.main()
