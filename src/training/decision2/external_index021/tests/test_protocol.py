from __future__ import annotations

import unittest

from external_index021.protocol import aggregate, replay_published, spec


class ProtocolTests(unittest.TestCase):
    def test_published_all_68_rows_are_display_compatible(self) -> None:
        report = replay_published()
        self.assertEqual(report["rows"], 68)
        self.assertLess(report["max_display_delta"], 0.01)
        names = {m["name"].lower() for m in report["models"]}
        self.assertTrue(any(name == "jev" for name in names))
        for family in ("kai", "lex", "eos", "sol", "nox", "lux"):
            self.assertTrue(any(family in name for name in names), family)

    def test_38_panel_13_gold_and_total_request_denominator(self) -> None:
        s = spec()
        self.assertEqual(
            len({n for area in s["areas"] for n in area["benchmarks"]}), 38
        )
        self.assertEqual(len(s["gold_ids"]), 13)
        self.assertAlmostEqual(sum(a["weight"] for a in s["areas"]), 1.0)
        self.assertEqual(
            s["suite"]["base_requests"] + s["suite"]["added_requests"], 150759
        )
        self.assertEqual(
            s["suite"]["base_scoreable"] + s["suite"]["added_requests"], 150317
        )
        self.assertEqual(sum(s["expected_drops"].values()), 717)

    def test_gold_weight_uses_one_point_two_within_area(self) -> None:
        s = spec()
        values = {
            n: {"raw": 0.0, "skill": 0.0, "coverage": 1.0}
            for area in s["areas"]
            for n in area["benchmarks"]
        }
        gold = s["gold_ids"][0]
        values[gold] = {"raw": 1.0, "skill": 1.0, "coverage": 1.0}
        result = aggregate(values)
        area = next(a for a in result["areas"] if gold in a["benchmarks"])
        denominator = sum(
            s["gold_weight"] if n in s["gold_ids"] else 1.0 for n in area["benchmarks"]
        )
        self.assertAlmostEqual(area["skill"], 1.2 / denominator)
        self.assertGreater(result["scores"]["balanced_skill"], 0)

    def test_missing_benchmark_cannot_get_full_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing headline"):
            aggregate({})


if __name__ == "__main__":
    unittest.main()
