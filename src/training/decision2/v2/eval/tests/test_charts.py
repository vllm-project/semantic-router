from __future__ import annotations

import unittest

from v2.eval import charts


def report(label: str, model_id: str | None, family: str = "peer") -> dict:
    return {"model": {"label": label, "model_id": model_id, "family": family}}


class ChartOverridesTest(unittest.TestCase):
    def test_missing_model_id_is_filled_for_the_licence_lookup(self):
        rows = [
            report("Decider 2B", None),
            report("Lux (node A D1)", "x/lux", "decision1"),
        ]
        charts.apply_overrides(
            rows, {"Decider 2B": "Mapika/decider-2b"}, {"Lux (node A D1)": "Lux"}
        )
        self.assertEqual(charts.licence_class(rows[0]), "permissive")
        self.assertEqual(rows[1]["model"]["label"], "Lux")
        self.assertEqual(rows[1]["model"]["model_id"], "x/lux")

    def test_existing_model_id_is_never_replaced(self):
        rows = [report("Jet", "michaljach/jet")]
        with self.assertRaises(ValueError):
            charts.apply_overrides(rows, {"Jet": "other/jet"}, None)

    def test_unknown_peer_is_still_refused(self):
        with self.assertRaises(ValueError):
            charts.licence_class(report("Mystery", None))


if __name__ == "__main__":
    unittest.main()
