"""name_basis base: a release is named after its base model's size (stdlib, fixture reports, fake API)."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from v2.release import build, card, hub, layout
from v2.release.tests.test_release import REPORTS, ROSTER, TEXT, facts

NINE_B = 7_940_895_744
TWENTY_SEVEN_B = 25_746_591_744


def lineage_spec(**extra):
    return {
        "origin": {"repo_id": "llm-semantic-router/Decision-1.0-Lux-9B"},
        "licence": {
            "components": [
                {"source": "llm-semantic-router/DEV2.0-9B"},
                {"source": "llm-semantic-router/Decision-1.0-Lux-9B@bd45a30a"},
                {"source": "Qwen/Qwen3.5-9B@c2022362"},
            ]
        },
        **extra,
    }


class BaseNameTest(unittest.TestCase):
    def test_size_label_of_the_base_repository(self):
        for repo, label in (
            ("Qwen/Qwen3.5-9B", "9B"),
            ("Qwen/Qwen3.8-27B", "27B"),
            ("Qwen/Qwen3-0.6B", "0.6B"),
            ("Qwen/Qwen3-30B-A3B", "30B"),
        ):
            self.assertEqual(layout.base_size_label(repo), label)
        for repo in ("Qwen/Qwen3.5", "someone/model-7B-13B"):
            with self.assertRaises(ValueError):
                layout.base_size_label(repo)

    def test_the_two_renamed_releases(self):
        self.assertEqual(
            layout.release_name(NINE_B, "base", "Qwen/Qwen3.5-9B"), "DEV2.0-9B"
        )
        self.assertEqual(
            layout.release_name(TWENTY_SEVEN_B, "base", "Qwen/Qwen3.8-27B"),
            "DEV2.0-27B",
        )
        self.assertEqual(layout.release_name(NINE_B, "loaded-parameters"), "DEV2.0-8B")
        self.assertEqual(
            layout.release_name(TWENTY_SEVEN_B, "loaded-parameters"), "DEV2.0-26B"
        )

    def test_the_base_size_must_share_the_loaded_tier(self):
        with self.assertRaises(ValueError):
            layout.release_name(NINE_B, "base", "Qwen/Qwen3.8-27B")
        with self.assertRaises(ValueError):
            layout.release_name(NINE_B, "base")
        with self.assertRaises(ValueError):
            layout.release_name(1_300_000_000, "base", "Qwen/Qwen3.5-2B")

    def test_the_naming_base_is_part_of_the_declared_lineage(self):
        self.assertEqual(
            build.name_base_model(lineage_spec(name_base_model="Qwen/Qwen3.5-9B")),
            "Qwen/Qwen3.5-9B",
        )
        self.assertEqual(
            build.name_base_model(lineage_spec(base={"repo_id": "Qwen/Qwen3.8-27B"})),
            "Qwen/Qwen3.8-27B",
        )
        for spec in (
            lineage_spec(),
            lineage_spec(name_base_model="Qwen/Qwen3.8-27B"),
            lineage_spec(
                name_base_model="Qwen/Qwen3.5-9B", base={"repo_id": "Qwen/Qwen3.8-27B"}
            ),
        ):
            with self.assertRaises(ValueError):
                build.name_base_model(spec)


class BaseNameCardTest(unittest.TestCase):
    def render(self, extra: dict) -> tuple[str, list[str]]:
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
        ]
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch) / "pkg"
            card.build_card(
                entries=entries,
                roster=ROSTER,
                paired=None,
                facts={**facts(), **extra},
                text=TEXT,
                work=Path(scratch) / "work",
                output=out,
            )
            readme = (out / "README.md").read_text()
            files = {
                p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
            }
            files |= {"LICENSE", "NOTICE", "ATTRIBUTIONS.md"}
            return readme, card.check_rendered(readme, files)

    def test_card_names_the_base_and_states_the_loaded_count(self):
        readme, problems = self.render(
            {"name_basis": "base", "name_base_model": "Qwen/Qwen3-0.6B"}
        )
        self.assertEqual(problems, [])
        self.assertIn(
            "| **Parameters** | 0.57B (571,909,635); named after its base model, "
            "[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |",
            readme,
        )

    def test_other_bases_add_no_name_line(self):
        readme, _ = self.render({})
        self.assertNotIn("named after its base model", readme)


class RenamedRepositoryTest(unittest.TestCase):
    def api(self, resolved: str | None):
        return SimpleNamespace(
            model_info=lambda repo, **_: SimpleNamespace(
                id=resolved or repo, private=True, sha="a" * 40
            )
        )

    def test_an_old_id_that_redirects_is_refused(self):
        args = argparse.Namespace(
            repo="llm-semantic-router/DEV2.0-8B", kind="release", model_name="DEV2.0-8B"
        )
        with mock.patch.object(
            hub, "_api", return_value=self.api("llm-semantic-router/DEV2.0-9B")
        ):
            with self.assertRaisesRegex(
                RuntimeError, "resolves to llm-semantic-router/DEV2.0-9B"
            ):
                hub.ensure(args)

    def test_the_repository_itself_is_accepted(self):
        args = argparse.Namespace(
            repo="llm-semantic-router/DEV2.0-9B", kind="release", model_name="DEV2.0-9B"
        )
        with mock.patch.object(hub, "_api", return_value=self.api(None)):
            self.assertFalse(hub.ensure(args)["created"])


if __name__ == "__main__":
    unittest.main()
