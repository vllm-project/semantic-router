"""Index input of the product cards and the asset renderer's import surface (stdlib)."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

from v2.release import card_assets, card_index, layout
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
            lambda d: d.update(footnote=d["footnote"].split(" Training data")[0]),
            lambda d: d.pop("snapshot"),
            lambda d: d["family"][0].pop("parameters_basis"),
            lambda d: d["family"][0].update(parameters_basis="loaded"),
            lambda d: d["family"][0].pop("loaded_parameters"),
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

    def test_footnote_discloses_the_row_level_audit(self):
        footnote = card_index.FOOTNOTE.format(edition="0.2.1", snapshot="2026-09-28")
        self.assertEqual(
            footnote,
            "Decision 2.0: independent reproduction with the official 0.2.1 kit on the "
            "released weights; others: public board snapshot, 2026-09-28. Training data "
            "audited at row level against all Index test items.",
        )


def _model(name: str, base: str | None, served: int, skill: float = 0.1) -> dict:
    return {
        "name": name,
        "meta": {"base_model": base, "served_params": served},
        "scores": {"balanced_skill": skill},
        "categories": [{"id": a, "skill": skill} for a, _ in card_index.AREAS],
    }


def _board(extra=()) -> dict:
    models = [
        _model(f"peer {tier}", card_index.BOARD_BASES[tier][0], 1000 + i)
        for i, tier in enumerate(layout.TIERS)
    ]
    served = {tier: 1000 + i for i, tier in enumerate(layout.TIERS)}
    models += [
        (
            _model(name, card_index.BOARD_BASES[tier][-1], served[tier])
            if tier
            else _model(name, "org/encoder", 300)
        )
        for name, tier in card_index.DECISION1_TIERS.items()
    ]
    return {"generated_utc": "2000-01-01T00:00:00Z", "models": models + list(extra)}


class BoardConventionTest(unittest.TestCase):
    def test_family_takes_the_board_count_of_its_base(self):
        board = _board([_model("other", "org/unrelated", 5)])
        self.assertEqual(card_index.board_parameters(board, "2B"), 1002)
        with self.assertRaisesRegex(ValueError, "not one count"):
            card_index.board_parameters(
                _board([_model("odd", card_index.BOARD_BASES["2B"][1], 7)]), "2B"
            )
        with self.assertRaisesRegex(ValueError, "not one count"):
            card_index.board_parameters({"models": []}, "2B")

    def test_build_matches_runs_by_weights_and_loads(self):
        manifests = {
            tier: {
                "identity": {"model_sha256": f"{i:064x}"},
                "parameters": {"loaded": 900 + i},
            }
            for i, tier in enumerate(layout.TIERS)
        }
        runs = [
            {
                "model_sha256": f"{i:064x}",
                "edition": "0.0-test",
                "balanced_skill": 1.0 + i,
                "areas": {a: 1.0 for a, _ in card_index.AREAS},
                "index_sha256": "c" * 64,
            }
            for i in range(len(layout.TIERS))
        ]
        built = card_index.build(runs, _board(), "2000-01-01", manifests)
        for i, point in enumerate(built["family"]):
            self.assertEqual(
                (point["parameters"], point["loaded_parameters"]), (1000 + i, 900 + i)
            )
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "index.json"
            path.write_text(json.dumps(built))
            card_index.load(path)
        with self.assertRaisesRegex(ValueError, "2 kit runs"):
            card_index.build(runs + runs[:1], _board(), "2000-01-01", manifests)
        with self.assertRaisesRegex(ValueError, "unexpected own model"):
            card_index.build(
                runs, _board([_model("Decision 3.0", None, 1)]), "2000-01-01", manifests
            )

    def test_board_base_is_in_each_released_lineage(self):
        specs = Path(card_index.__file__).parent / "specs"
        for tier in layout.TIERS:
            key = tier.lower().replace(".", "p")
            spec = json.loads((specs / f"dev2-{key}-product.json").read_text())
            lineage = {spec["origin"]["repo_id"]} | {
                c["source"].split("@", 1)[0] for c in spec["licence"]["components"]
            }
            self.assertTrue(lineage & set(card_index.BOARD_BASES[tier]), tier)


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
