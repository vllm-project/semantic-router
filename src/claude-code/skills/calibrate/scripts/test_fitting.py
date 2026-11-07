import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fitting
from cases import labeled, route
from decisions import TIERS


class FitBinary(unittest.TestCase):
    def test_picks_the_threshold_that_best_separates(self):
        cases = labeled([0.80, 0.82, 0.85, 0.92, 0.95, 0.97], [0.10, 0.30, 0.76, 0.78])
        fit = fitting.fit_binary(cases, (0.9,))
        self.assertEqual(fit["thresholds"], (0.8,), "the perfect split closest to 0.90")
        self.assertTrue(fit["changed"])
        self.assertEqual(fitting.confusion(cases, fit["thresholds"]).agree, 10)

    def test_f1_trades_errors_when_no_split_is_perfect(self):
        cases = labeled([0.60, 0.70, 0.86, 0.88, 0.95], [0.20, 0.65, 0.84, 0.97])
        fit = fitting.fit_binary(cases, (0.9,))
        self.assertEqual(
            fit["thresholds"],
            (0.85,),
            "F1 0.67 at 0.85 beats 0.29 at 0.90; the best, 0.77 at 0.60, is out of reach",
        )
        self.assertEqual(
            fitting.fit_binary(cases, (0.9,), beta=0.5)["thresholds"],
            (0.86,),
            "fewer-false-alarms best is reachable",
        )
        self.assertEqual(
            fitting.fit_binary(cases, (0.7,), beta=2.0)["thresholds"],
            (0.6,),
            "catch-more takes every yes",
        )

    def test_needs_eight_labels(self):
        cases = labeled([0.80, 0.82, 0.85, 0.92], [0.10, 0.30, 0.78])
        fit = fitting.fit_binary(cases, (0.9,))
        self.assertEqual((fit["thresholds"], fit["changed"]), ((0.9,), False))
        self.assertIn("needs 8", fit["why"])
        self.assertTrue(
            fitting.fit_binary(cases + labeled([0.95], []), (0.9,))["changed"]
        )

    def test_moves_at_most_015(self):
        cases = labeled([0.30, 0.35, 0.40, 0.45, 0.50], [0.0, 0.05, 0.10, 0.15, 0.20])
        self.assertEqual(fitting.fit_binary(cases, (0.9,))["thresholds"], (0.75,))
        self.assertEqual(
            fitting.fit_binary(cases, (0.1,))["thresholds"],
            (0.21,),
            "a move under the cap goes all the way",
        )

    def test_all_no_labels_raise_the_threshold_just_enough(self):
        cases = labeled([], [0.91, 0.92, 0.93, 0.94, 0.95, 0.5, 0.6, 0.7])
        self.assertEqual(fitting.fit_binary(cases, (0.9,))["thresholds"], (0.96,))

    def test_raise_only_never_lowers(self):
        cases = labeled([0.80, 0.82, 0.85, 0.92, 0.95, 0.97], [0.10, 0.30, 0.76, 0.78])
        self.assertEqual(
            fitting.fit_binary(cases, (0.9,), raise_only=True)["thresholds"], (0.9,)
        )
        self.assertEqual(
            fitting.fit_binary(
                labeled([0.95, 0.97], [0.91, 0.92, 0.93, 0.6, 0.7, 0.8]),
                (0.9,),
                raise_only=True,
            )["thresholds"],
            (0.94,),
        )

    def test_escalate_fits_both_thresholds(self):
        yes = [
            ((0.85, 0.0), True),
            ((0.9, 0.1), True),
            ((0.1, 0.8), True),
            ((0.2, 0.82), True),
        ]
        no = [
            ((0.6, 0.0), False),
            ((0.82, 0.0), False),
            ((0.1, 0.76), False),
            ((0.0, 0.2), False),
        ]
        fit = fitting.fit_binary(yes + no, (0.8, 0.75))
        self.assertEqual(fit["thresholds"], (0.83, 0.77))


class FitRouter(unittest.TestCase):
    def build(self, truth_easy, truth_hard, levels, confidence=0.8):
        decisions, labels = [], {}
        for i, level in enumerate(levels):
            used = fitting.route_tier(level, 0.5, 1.5)
            right = fitting.route_tier(level, truth_easy, truth_hard)
            decisions.append(route(i, level, TIERS[used], confidence))
            labels[f"r{i}"] = (
                "right"
                if right == used
                else ("too-low" if right > used else "too-high")
            )
        return decisions, labels

    def test_fits_both_cuts(self):
        decisions, labels = self.build(
            0.605, 1.325, [0.1, 0.3, 0.55, 0.6, 0.65, 0.9, 1.2, 1.3, 1.35, 1.45, 1.8]
        )
        cases = fitting.router_cases(decisions, labels)
        fit = fitting.fit_router(cases, 0.5, 1.5, 0.3)
        self.assertEqual(fit["thresholds"], (0.61, 1.35, 0.3))
        self.assertEqual(
            fitting.router_agreement(cases, *fit["thresholds"])[:2], (11, 11)
        )

    def test_caps_each_cut(self):
        decisions, labels = self.build(
            1.0, 1.9, [0.2, 0.6, 0.8, 0.95, 1.1, 1.4, 1.6, 1.85, 1.95]
        )
        fit = fitting.fit_router(fitting.router_cases(decisions, labels), 0.5, 1.5, 0.3)
        self.assertEqual(fit["thresholds"][:2], (0.65, 1.65))

    def test_raises_the_confidence_floor_over_wrong_unsure_routes(self):
        decisions, labels = self.build(
            0.5, 1.5, [0.1, 0.2, 0.3, 0.7, 0.9, 1.1, 1.7, 1.9]
        )
        decisions += [route(20, 0.2, "haiku", 0.35), route(21, 0.25, "haiku", 0.33)]
        labels.update({"r20": "too-low", "r21": "too-low"})
        fit = fitting.fit_router(fitting.router_cases(decisions, labels), 0.5, 1.5, 0.3)
        self.assertEqual(fit["thresholds"], (0.5, 1.5, 0.36))

    def test_kept_routes_take_the_named_model(self):
        kept = route(1, 0.29, "opus", 0.22, verdict="kept")
        self.assertEqual(kept.score, 0.22, "a keep logs its confidence as the score")
        self.assertEqual(
            fitting.router_cases([kept], {"r1": "haiku"}),
            [fitting.RouterCase(0.29, 0.22, 0)],
        )
        self.assertEqual(
            fitting.router_cases([kept], {"r1": "too-high"}),
            [fitting.RouterCase(0.29, 0.22, 1)],
        )
        routed = route(2, 0.7, "sonnet", 0.9)
        self.assertEqual(
            fitting.router_cases([routed], {"r2": "opus"}),
            [fitting.RouterCase(0.7, 0.9, 2)],
            "a routed task takes a tier name too",
        )

    def test_needs_eight_and_skips_escalated_routes(self):
        decisions, labels = self.build(0.7, 1.3, [0.1, 0.6, 0.65, 1.0, 1.4, 1.45, 1.8])
        decisions.append(route(9, 0.66, "opus", escalated={"alone": "sonnet"}))
        labels["r9"] = "too-high"
        self.assertEqual(len(fitting.router_cases(decisions, labels)), 7)
        self.assertFalse(
            fitting.fit_router(fitting.router_cases(decisions, labels), 0.5, 1.5, 0.3)[
                "changed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
