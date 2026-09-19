"""Pin the guard metric contract, without a model or a dataset.

Every candidate under #3194 is read against these numbers, so the properties
that make them comparable are the ones worth holding: a ranking that does not
reward ties, a recall quoted with the benign budget it spent, a macro over
length bands that stops length from carrying the score, and slices that are
never pooled away.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import guard_metrics


def linear(count: int, low: float, high: float) -> list[float]:
    if count == 1:
        return [low]
    step = (high - low) / (count - 1)
    return [low + step * index for index in range(count)]


class BandTest(unittest.TestCase):
    def test_a_row_lands_in_the_band_that_holds_its_word_count(self):
        self.assertEqual(guard_metrics.band_of(1), "1-2")
        self.assertEqual(guard_metrics.band_of(12), "11-25")
        self.assertEqual(guard_metrics.band_of(999), "151+")

    def test_the_bands_meet_without_a_gap(self):
        self.assertEqual(guard_metrics.band_of(2), "1-2")
        self.assertEqual(guard_metrics.band_of(3), "3-5")
        self.assertEqual(guard_metrics.band_of(150), "61-150")
        self.assertEqual(guard_metrics.band_of(151), "151+")


class PooledTest(unittest.TestCase):
    def test_recall_is_quoted_with_the_budget_it_spent(self):
        labels = [0] * 100 + [1] * 10
        scores = linear(100, 0.0, 0.5) + [0.9] * 10

        pooled = guard_metrics.guard_report(labels, scores)["pooled"]
        reached = pooled["recall_at_0.01_fpr"]

        self.assertEqual(reached["recall"], 1.0)
        self.assertLessEqual(reached["false_positive_rate"], 0.01)
        self.assertGreater(reached["threshold"], 0.5)

    def test_ties_do_not_buy_separation(self):
        # One score for everything is no ordering at all, and a tie-blind
        # ranking would call it perfect or hopeless depending on sort order.
        report = guard_metrics.guard_report([0, 0, 1, 1], [0.5] * 4)

        self.assertEqual(report["pooled"]["auc"], 0.5)

    def test_the_pooled_row_says_it_is_a_reference_line(self):
        report = guard_metrics.guard_report([0, 1], [0.1, 0.9])

        self.assertIn("reference line", report["pooled"]["note"])

    def test_an_evaluation_needs_rows_that_align(self):
        with self.assertRaises(ValueError):
            guard_metrics.guard_report([0, 1], [0.5])
        with self.assertRaises(ValueError):
            guard_metrics.guard_report([], [])


class BandMacroTest(unittest.TestCase):
    def crowded(self):
        """One band holds almost every row and is easy, the other is hard."""
        labels = [1] * 500 + [0] * 200 + [1] * 10 + [0] * 20
        scores = (
            [0.99] * 500 + linear(200, 0.0, 0.2) + [0.01] * 10 + linear(20, 0.0, 0.2)
        )
        words = [200] * 700 + [4] * 30
        return labels, scores, words

    def test_the_macro_does_not_follow_the_crowded_band(self):
        labels, scores, words = self.crowded()

        report = guard_metrics.guard_report(labels, scores, words=words)

        self.assertGreater(report["pooled"]["recall_at_0.01_fpr"]["recall"], 0.9)
        # The short band spends its whole budget on the first row it flags, so
        # it contributes a zero rather than dropping out of the average.
        self.assertEqual(report["band_macro_recall_at_0.01_fpr"], 0.5)
        self.assertIsNone(report["by_band"]["3-5"]["recall_at_0.01_fpr"])

    def test_a_band_without_both_classes_leaves_the_macro_alone(self):
        report = guard_metrics.guard_report(
            [0, 1, 1], [0.1, 0.9, 0.8], words=[4, 4, 200]
        )

        self.assertIsNone(report["by_band"]["151+"].get("auc"))
        self.assertEqual(report["band_macro_recall_at_0.01_fpr"], 1.0)


class SliceTest(unittest.TestCase):
    def test_a_slice_carries_its_own_rows_and_rates(self):
        labels = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.8, 0.2]
        report = guard_metrics.guard_report(
            labels, scores, slices={"source": ["a", "a", "b", "b"]}
        )

        first = report["by_slice"]["source"]["a"]
        second = report["by_slice"]["source"]["b"]
        self.assertEqual(first["rows"], 2)
        self.assertEqual(first["auc"], 1.0)
        self.assertEqual(second["auc"], 0.0)

    def test_a_benign_only_slice_still_owes_a_false_positive_rate(self):
        report = guard_metrics.guard_report(
            [0, 0, 1],
            [0.9, 0.1, 0.9],
            slices={"source": ["benign", "benign", "attack"]},
        )

        benign = report["by_slice"]["source"]["benign"]
        self.assertEqual(benign["false_positive_rate"], 0.5)
        self.assertIsNone(benign["recall"])
        self.assertNotIn("auc", benign)

    def test_bands_and_slices_are_reported_only_when_given(self):
        report = guard_metrics.guard_report([0, 1], [0.1, 0.9])

        self.assertNotIn("by_band", report)
        self.assertNotIn("by_slice", report)


class CalibrationTest(unittest.TestCase):
    def test_a_calibrated_score_has_almost_no_error(self):
        # Ten rows in each tenth, with the positive rate the score claims.
        labels = []
        scores = []
        for step in range(10):
            score = step / 10 + 0.05
            labels.extend([1] * step + [0] * (10 - step))
            scores.extend([score] * 10)

        calibration = guard_metrics.guard_report(labels, scores)["calibration"]

        self.assertLess(calibration["ece"], 0.06)

    def test_a_confident_wrong_score_is_reported_as_error(self):
        calibration = guard_metrics.guard_report([0] * 99 + [1], [0.95] * 100)[
            "calibration"
        ]

        self.assertGreater(calibration["ece"], 0.9)
        self.assertGreater(calibration["mce"], 0.9)


class RoutingAgreementTest(unittest.TestCase):
    def test_a_guard_agrees_with_itself(self):
        scores = [0.1, 0.2, 0.8, 0.9]

        agreement = guard_metrics.routing_agreement([0, 0, 1, 1], scores, scores)

        self.assertEqual(agreement["agreement_rate"], 1.0)
        self.assertIsNone(agreement["mcnemar_exact_p"])

    def test_equal_accuracy_with_different_errors_still_moves_traffic(self):
        agreement = guard_metrics.routing_agreement([0, 1], [0.9, 0.9], [0.1, 0.1])

        self.assertEqual(agreement["agreement_rate"], 0.0)
        self.assertEqual(agreement["on_disagreement_candidate_right"], 1)
        self.assertEqual(agreement["on_disagreement_baseline_right"], 1)
        self.assertEqual(agreement["candidate_blocks_only"], 0)
        self.assertEqual(agreement["baseline_blocks_only"], 2)


class SignTestTest(unittest.TestCase):
    def test_matches_the_exact_binomial_values(self):
        # Reference values from scipy.stats.binomtest(k, n, 0.5).pvalue, which
        # this replaces so the contract stays on the standard library. The last
        # row is the routing comparison reported on #3787.
        for successes, trials, expected in [
            (1, 10, 0.021484375),
            (0, 10, 0.001953125),
            (5, 10, 1.0),
            (3, 10, 0.34375),
            (2, 7, 0.453125),
            (1156, 1542, 3.0402468829258952e-89),
        ]:
            with self.subTest(successes=successes, trials=trials):
                got = guard_metrics.two_sided_sign_test(successes, trials)
                self.assertTrue(math.isclose(got, expected, rel_tol=1e-12))

    def test_no_disagreement_is_not_a_test(self):
        with self.assertRaises(ValueError):
            guard_metrics.two_sided_sign_test(0, 0)


class IntervalTest(unittest.TestCase):
    def scores(self):
        labels = [0] * 200 + [1] * 200
        scores = linear(200, 0.0, 0.6) + linear(200, 0.4, 1.0)
        return labels, scores

    def test_intervals_appear_only_when_resamples_are_asked_for(self):
        labels, scores = self.scores()

        self.assertNotIn("intervals", guard_metrics.guard_report(labels, scores))

        interval = guard_metrics.guard_report(labels, scores, bootstrap_resamples=50)[
            "intervals"
        ]["auc"]
        low, high = interval["interval"]
        self.assertLessEqual(low, interval["point"])
        self.assertLessEqual(interval["point"], high)

    def test_the_same_seed_gives_the_same_interval(self):
        labels, scores = self.scores()

        def auc(drawn_labels, drawn_scores):
            return guard_metrics.guard_report(drawn_labels, drawn_scores)["pooled"][
                "auc"
            ]

        first = guard_metrics.bootstrap_interval(labels, scores, auc, 20, seed=7)
        second = guard_metrics.bootstrap_interval(labels, scores, auc, 20, seed=7)

        self.assertEqual(first["interval"], second["interval"])

    def test_a_draw_that_spends_the_budget_counts_as_zero_recall(self):
        """A resample that cannot flag anything in budget is not a missing draw.

        Two negatives outrank every positive, so one false positive already
        exceeds a 1% budget on 100 negatives and no threshold reaches any
        positive. Dropping those resamples would leave only the ones that got
        past the two, and the interval could then not fall below a recall the
        sample never achieved.
        """
        labels = [0] * 100 + [1] * 10
        scores = [0.9, 0.9] + [0.1] * 98 + [0.8] * 10

        def recall(drawn_labels, drawn_scores):
            return guard_metrics._recall_or_none(drawn_labels, drawn_scores, 0.01)

        self.assertEqual(recall(labels, scores), 0.0)

        drawn = guard_metrics.bootstrap_interval(labels, scores, recall, resamples=2000)

        self.assertEqual(drawn["point"], 0.0)
        self.assertEqual(drawn["resamples"], 2000)
        self.assertEqual(drawn["interval"][0], 0.0)

    def test_a_single_class_draw_stays_undefined(self):
        """One class is the case a recall genuinely has no value for."""
        self.assertIsNone(guard_metrics._recall_or_none([0, 0], [0.1, 0.2], 0.01))
        self.assertIsNone(guard_metrics._recall_or_none([1, 1], [0.8, 0.9], 0.01))


if __name__ == "__main__":
    unittest.main()
