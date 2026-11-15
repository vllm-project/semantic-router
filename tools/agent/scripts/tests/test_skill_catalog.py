"""The skill catalog covers shipped sources and usable local links."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import check_skill_catalog as catalog  # noqa: E402


class SkillCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in (
            catalog.CATALOG,
            catalog.README,
            catalog.PUBLIC_COPY,
            Path("tools/agent/docs/README.md"),
        ):
            target = self.root / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(catalog.ROOT / source, target)
        for source in catalog.SKILL_ROOTS:
            for skill in (catalog.ROOT / source).rglob("SKILL.md"):
                target = self.root / skill.relative_to(catalog.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(skill, target)
        for reference in (
            catalog.ROOT / "tools/agent/skills/vllm-sr-agent-operations/references"
        ).glob("*.md"):
            target = self.root / reference.relative_to(catalog.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference, target)

    def test_repository_catalog_and_links_are_complete(self):
        self.assertEqual(catalog.check(self.root), [])

    def test_unlisted_skill_is_rejected(self):
        extra = self.root / "tools/agent/skills/extra/SKILL.md"
        extra.parent.mkdir(parents=True)
        extra.write_text("---\nname: extra\n---\n")
        self.assertTrue(
            any("unlisted skill" in error for error in catalog.check(self.root))
        )

    def test_stale_public_mapping_is_rejected(self):
        data = yaml.safe_load((self.root / catalog.CATALOG).read_text())
        data["skills"][0]["public_url"] = "https://example.com/stale/SKILL.md"
        (self.root / catalog.CATALOG).write_text(yaml.safe_dump(data))
        self.assertTrue(
            any("sync contract" in error for error in catalog.check(self.root))
        )

    def test_a_new_public_skill_can_have_its_own_path(self):
        source = Path("tools/agent/skills/new/SKILL.md")
        copy = Path("website/static/install/agent/new/SKILL.md")
        for path in (source, copy):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("---\nname: new\ndescription: Test\n---\n")
        data = yaml.safe_load((self.root / catalog.CATALOG).read_text())
        entry = data["skills"][0].copy()
        entry.update(
            id="new",
            name="new",
            source=source.as_posix(),
            published=copy.as_posix(),
            public_url="https://vllm-sr.ai/install/agent/new/SKILL.md",
        )
        data["skills"].append(entry)
        (self.root / catalog.CATALOG).write_text(yaml.safe_dump(data))
        readme = self.root / catalog.README
        readme.write_text(readme.read_text() + "\n[New](new/SKILL.md)\n")
        self.assertEqual(catalog.check(self.root), [])

    def test_broken_discovery_link_is_rejected(self):
        readme = self.root / catalog.README
        readme.write_text(
            readme.read_text().replace(
                "routing-calibration/SKILL.md", "missing/SKILL.md"
            )
        )
        errors = catalog.check(self.root)
        self.assertTrue(any("broken local link" in error for error in errors))
        self.assertTrue(any("missing discoverable entry" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
