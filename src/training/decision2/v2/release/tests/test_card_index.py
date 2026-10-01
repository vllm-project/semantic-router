"""Index input of the product cards and the asset renderer's import surface (stdlib)."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

from v2.release import card_assets, card_index
from v2.release.tests import card_fixture


class CardIndexTest(unittest.TestCase):
    def write(self, scratch: Path, change=None) -> Path:
        path = card_fixture.index_file(scratch / "index.json", {"4B": "d" * 64})
        if change:
            data = json.loads(path.read_text())
            change(data)
            path.write_text(json.dumps(data))
        return path

    def test_view_checks_the_weights_and_picks_the_comparison(self):
        with tempfile.TemporaryDirectory() as scratch:
            index = card_index.load(self.write(Path(scratch)))
            view = card_index.view(index, "4B", "d" * 64)
            self.assertEqual(view["compare_name"], "Decision 1.0 Nox")
            self.assertEqual(view["compare_kind"], "decision1")
            self.assertAlmostEqual(view["delta"], 2.25)
            with self.assertRaisesRegex(ValueError, "other weights"):
                card_index.view(index, "4B", "e" * 64)
            vega = card_index.view(index, "27B", "a" * 64)
            self.assertEqual(
                (vega["compare_name"], vega["compare_kind"]),
                ("Decision-2.0-Lux-9B", "family"),
            )

    def test_load_refuses_incomplete_inputs(self):
        changes = (
            lambda d: d.update(schema="other"),
            lambda d: d.update(footnote=""),
            lambda d: d["family"].pop(),
            lambda d: d["family"][0].update(name="DEV2.0-0.6B"),
            lambda d: d["family"][0].update(model_sha256="short"),
            lambda d: d["family"][0]["areas"].pop("arts"),
            lambda d: d["decision1"].pop(0),
            lambda d: d["decision1"][0].update(balanced_skill="high"),
            lambda d: d.update(entrants=[]),
        )
        for change in changes:
            with tempfile.TemporaryDirectory() as scratch:
                with self.assertRaises(ValueError):
                    card_index.load(self.write(Path(scratch), change))


class CardAssetsTest(unittest.TestCase):
    def test_module_imports_only_stdlib_and_release_code_at_top_level(self):
        tree = ast.parse(Path(card_assets.__file__).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = (
                    node.module
                    if isinstance(node, ast.ImportFrom)
                    else node.names[0].name
                )
                top = name.split(".")[0]
                self.assertTrue(
                    top in sys.stdlib_module_names or top in ("__future__", "v2"), name
                )

    def test_banner_name(self):
        self.assertEqual(card_assets.short_name("Decision-2.0-Nox-4B"), "Nox 4B")
        self.assertEqual(card_assets.short_name("Decision-2.0-Kai-0.6B"), "Kai 0.6B")
        with self.assertRaises(ValueError):
            card_assets.short_name("DEV2.0-4B")


if __name__ == "__main__":
    unittest.main()
