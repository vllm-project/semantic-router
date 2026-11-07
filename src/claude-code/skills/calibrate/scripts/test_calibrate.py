import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).parent))
import calibrate as c


def guard(i, score, verdict=None, **detail):
    return c.Decision(
        f"g{i}",
        None,
        "s",
        "guard",
        verdict or ("held" if score >= 0.9 else "allowed"),
        score,
        0.9,
        f"cmd {i}",
        detail,
    )


def labeled(scores_yes, scores_no):
    return [((s,), True) for s in scores_yes] + [((s,), False) for s in scores_no]


class FitBinary(unittest.TestCase):
    def test_picks_the_threshold_that_best_separates(self):
        cases = labeled([0.80, 0.82, 0.85, 0.92, 0.95, 0.97], [0.10, 0.30, 0.76, 0.78])
        fit = c.fit_binary(cases, (0.9,))
        self.assertEqual(fit["thresholds"], (0.8,), "the perfect split closest to 0.90")
        self.assertTrue(fit["changed"])
        self.assertEqual(c.confusion(cases, fit["thresholds"]).agree, 10)

    def test_f1_trades_errors_when_no_split_is_perfect(self):
        cases = labeled([0.60, 0.70, 0.86, 0.88, 0.95], [0.20, 0.65, 0.84, 0.97])
        fit = c.fit_binary(cases, (0.9,))
        self.assertEqual(
            fit["thresholds"],
            (0.85,),
            "F1 0.67 at 0.85 beats 0.29 at 0.90; the best, 0.77 at 0.60, is out of reach",
        )
        self.assertEqual(
            c.fit_binary(cases, (0.9,), beta=0.5)["thresholds"],
            (0.86,),
            "fewer-false-alarms best is reachable",
        )
        self.assertEqual(
            c.fit_binary(cases, (0.7,), beta=2.0)["thresholds"],
            (0.6,),
            "catch-more takes every yes",
        )

    def test_needs_eight_labels(self):
        cases = labeled([0.80, 0.82, 0.85, 0.92], [0.10, 0.30, 0.78])
        fit = c.fit_binary(cases, (0.9,))
        self.assertEqual((fit["thresholds"], fit["changed"]), ((0.9,), False))
        self.assertIn("needs 8", fit["why"])
        self.assertTrue(c.fit_binary(cases + labeled([0.95], []), (0.9,))["changed"])

    def test_moves_at_most_015(self):
        cases = labeled([0.30, 0.35, 0.40, 0.45, 0.50], [0.0, 0.05, 0.10, 0.15, 0.20])
        self.assertEqual(c.fit_binary(cases, (0.9,))["thresholds"], (0.75,))
        self.assertEqual(
            c.fit_binary(cases, (0.1,))["thresholds"],
            (0.21,),
            "a move under the cap goes all the way",
        )

    def test_all_no_labels_raise_the_threshold_just_enough(self):
        cases = labeled([], [0.91, 0.92, 0.93, 0.94, 0.95, 0.5, 0.6, 0.7])
        self.assertEqual(c.fit_binary(cases, (0.9,))["thresholds"], (0.96,))

    def test_raise_only_never_lowers(self):
        cases = labeled([0.80, 0.82, 0.85, 0.92, 0.95, 0.97], [0.10, 0.30, 0.76, 0.78])
        self.assertEqual(
            c.fit_binary(cases, (0.9,), raise_only=True)["thresholds"], (0.9,)
        )
        self.assertEqual(
            c.fit_binary(
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
        fit = c.fit_binary(yes + no, (0.8, 0.75))
        self.assertEqual(fit["thresholds"], (0.83, 0.77))


def route(i, level, model, confidence=0.8, verdict="routed", **extra):
    score = confidence if verdict == "kept" else level
    return c.Decision(
        f"r{i}",
        None,
        "s",
        "router",
        verdict,
        score,
        None,
        f"task {i}",
        {
            "level": level,
            "confidence": confidence,
            "model": model,
            "escalated": None,
            **extra,
        },
    )


def tier_model(level, easy=0.5, hard=1.5):
    return c.TIERS[c.route_tier(level, easy, hard)]


class FitRouter(unittest.TestCase):
    def build(self, truth_easy, truth_hard, levels, confidence=0.8):
        decisions, labels = [], {}
        for i, level in enumerate(levels):
            used = c.route_tier(level, 0.5, 1.5)
            right = c.route_tier(level, truth_easy, truth_hard)
            decisions.append(route(i, level, c.TIERS[used], confidence))
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
        cases = c.router_cases(decisions, labels)
        fit = c.fit_router(cases, 0.5, 1.5, 0.3)
        self.assertEqual(fit["thresholds"], (0.61, 1.35, 0.3))
        self.assertEqual(c.router_agreement(cases, *fit["thresholds"])[:2], (11, 11))

    def test_caps_each_cut(self):
        decisions, labels = self.build(
            1.0, 1.9, [0.2, 0.6, 0.8, 0.95, 1.1, 1.4, 1.6, 1.85, 1.95]
        )
        fit = c.fit_router(c.router_cases(decisions, labels), 0.5, 1.5, 0.3)
        self.assertEqual(fit["thresholds"][:2], (0.65, 1.65))

    def test_raises_the_confidence_floor_over_wrong_unsure_routes(self):
        decisions, labels = self.build(
            0.5, 1.5, [0.1, 0.2, 0.3, 0.7, 0.9, 1.1, 1.7, 1.9]
        )
        decisions += [route(20, 0.2, "haiku", 0.35), route(21, 0.25, "haiku", 0.33)]
        labels.update({"r20": "too-low", "r21": "too-low"})
        fit = c.fit_router(c.router_cases(decisions, labels), 0.5, 1.5, 0.3)
        self.assertEqual(fit["thresholds"], (0.5, 1.5, 0.36))

    def test_kept_routes_take_the_named_model(self):
        kept = route(1, 0.29, "opus", 0.22, verdict="kept")
        self.assertEqual(kept.score, 0.22, "a keep logs its confidence as the score")
        self.assertEqual(
            c.router_cases([kept], {"r1": "haiku"}), [c.RouterCase(0.29, 0.22, 0)]
        )
        self.assertEqual(
            c.router_cases([kept], {"r1": "too-high"}), [c.RouterCase(0.29, 0.22, 1)]
        )

    def test_needs_eight_and_skips_escalated_routes(self):
        decisions, labels = self.build(0.7, 1.3, [0.1, 0.6, 0.65, 1.0, 1.4, 1.45, 1.8])
        decisions.append(route(9, 0.66, "opus", escalated={"alone": "sonnet"}))
        labels["r9"] = "too-high"
        self.assertEqual(len(c.router_cases(decisions, labels)), 7)
        self.assertFalse(
            c.fit_router(c.router_cases(decisions, labels), 0.5, 1.5, 0.3)["changed"]
        )


class Sampling(unittest.TestCase):
    options: ClassVar[dict] = {
        "guardReview": 0.9,
        "routerEasyBelow": 0.5,
        "routerHardFrom": 1.5,
    }

    def test_closest_on_both_sides(self):
        decisions = [
            guard(i, s)
            for i, s in enumerate([0.5, 0.8, 0.88, 0.89, 0.91, 0.93, 0.99, 0.1])
        ]
        picked = [d.score for d in c.sample(decisions, {}, self.options, ["guard"], 2)]
        self.assertEqual(picked, [0.91, 0.93, 0.89, 0.88])

    def test_skips_labeled_and_scoreless(self):
        decisions = [
            guard(i, s) for i, s in enumerate([0.5, 0.8, 0.88, 0.89, 0.91, 0.93])
        ]
        decisions.append(
            c.Decision("off1", None, "s", "guard", "off", None, None, "x", {})
        )
        picked = [
            d.score
            for d in c.sample(decisions, {"g3": "no"}, self.options, ["guard"], 2)
        ]
        self.assertEqual(picked, [0.91, 0.93, 0.88, 0.8])

    def test_router_uses_the_nearest_cut(self):
        decisions = [
            route(i, s, tier_model(s))
            for i, s in enumerate([0.05, 0.45, 0.55, 1.0, 1.4, 1.6])
        ]
        picked = [d.score for d in c.sample(decisions, {}, self.options, ["router"], 1)]
        self.assertEqual(sorted(picked), [0.45, 0.55])


class Features(unittest.TestCase):
    def test_escalate_reads_both_signals(self):
        reask_only = c.Decision(
            "e1",
            None,
            "s",
            "escalate",
            "escalated",
            0.8,
            0.75,
            "again",
            {"feedback": None, "reask": 0.8},
        )
        self.assertEqual(
            c.features(reask_only),
            (0.0, 0.8),
            "the reask score is not taken for feedback",
        )
        both = c.Decision(
            "e2",
            None,
            "s",
            "escalate",
            "escalated",
            0.9,
            0.8,
            "no",
            {
                "feedback": {"WRONG_ANSWER": 0.9, "WANT_DIFFERENT": 0.05, "SAT": 0.01},
                "reask": None,
            },
        )
        self.assertEqual(c.features(both), (0.9, 0.0))

    def test_redact_hidden_only_by_secret_rules_is_not_a_threshold_case(self):
        secret = c.Decision(
            "x1",
            None,
            "s",
            "redact",
            "hidden",
            1.0,
            0.7,
            "env",
            {"counts": {"secret": 2}, "source": ".env"},
        )
        mixed = c.Decision(
            "x2",
            None,
            "s",
            "redact",
            "hidden",
            0.74,
            0.7,
            "notes",
            {"counts": {"secret": 1, "email": 1}, "source": "n"},
        )
        self.assertIsNone(c.features(secret))
        self.assertEqual(c.features(mixed), (0.74,))

    def test_guard_held_for_unchecked_parts_is_not_a_threshold_case(self):
        too_long = guard(1, 0.1, "held", unchecked=2, outcome="declined")
        self.assertIsNone(c.features(too_long))
        self.assertEqual(
            c.implicit_labels([too_long]),
            {},
            "a declined hold of a too-long command says nothing of its score",
        )

    def test_guard_held_by_a_fixed_rule_is_not_a_threshold_case(self):
        ruled = guard(
            2,
            1.0,
            "held",
            rule="deletes a whole top-level directory",
            outcome="declined",
        )
        self.assertIsNone(c.features(ruled))
        self.assertEqual(c.implicit_labels([ruled]), {})


class LogParsing(unittest.TestCase):
    def test_partial_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            full = [
                {
                    "id": "a",
                    "at": "2026-11-06T10:00:00Z",
                    "session": "s",
                    "app": "guard",
                    "verdict": "held",
                    "score": 0.95,
                    "threshold": 0.9,
                    "subject": "git push -f",
                    "detail": {"outcome": "declined"},
                },
                {
                    "id": "b",
                    "at": 1793700000000,
                    "session": "s",
                    "app": "shield",
                    "verdict": "shielded",
                    "score": 0.99,
                    "threshold": 0.9,
                    "subject": "INSTALL.md",
                    "detail": {"source": "vendor/INSTALL.md"},
                },
            ]
            path.write_text(
                "".join(json.dumps(x) + "\n" for x in full) + '{"id": "c", "app": "gua'
            )
            log = c.read_log(path)
            self.assertEqual([d.id for d in log.decisions], ["a", "b"])
            self.assertEqual(log.skipped, 1)
            self.assertAlmostEqual(log.decisions[1].at, 1793700000.0)
            self.assertIsNotNone(log.decisions[0].at)

    def test_missing_log_is_empty(self):
        self.assertEqual(c.read_log(Path("/nonexistent/decisions.jsonl")).decisions, [])


class ImplicitLabels(unittest.TestCase):
    def test_declined_hold_is_yes(self):
        found = c.implicit_labels(
            [
                guard(1, 0.95, "held", outcome="declined"),
                guard(2, 0.95, "held", outcome="approved"),
            ]
        )
        self.assertEqual(found, {"g1": "yes"})

    def test_escalation_soon_after_a_light_route(self):
        r_haiku = c.Decision(
            "r1", 1000.0, "s", "router", "routed", 0.2, None, "t", {"model": "haiku"}
        )
        r_opus = c.Decision(
            "r2", 1000.0, "s2", "router", "routed", 1.8, None, "t", {"model": "opus"}
        )
        e1 = c.Decision(
            "e1", 1200.0, "s", "escalate", "escalated", 0.92, 0.8, "no, wrong", {}
        )
        e2 = c.Decision(
            "e2", 1200.0, "s2", "escalate", "escalated", 0.92, 0.8, "no, wrong", {}
        )
        late = c.Decision(
            "r3", 1000.0, "s3", "router", "routed", 0.2, None, "t", {"model": "sonnet"}
        )
        e3 = c.Decision(
            "e3", 1000.0 + 3600, "s3", "escalate", "escalated", 0.92, 0.8, "no", {}
        )
        self.assertEqual(
            c.implicit_labels([r_haiku, r_opus, e1, e2, late, e3]), {"r1": "too-low"}
        )


class EndToEnd(unittest.TestCase):
    def test_suggest_then_apply(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ):
            os.environ["CLAUDE_CONFIG_DIR"] = tmp
            try:
                settings_path = Path(tmp) / "settings.json"
                log = Path(tmp) / "vsr" / "decisions.jsonl"
                settings_path.write_text(
                    json.dumps(
                        {
                            "theme": "dark",
                            "pluginConfigs": {
                                "vsr@inline": {"options": {"logPath": str(log)}}
                            },
                        }
                    )
                )
                log.parent.mkdir()
                scores = [
                    (0.80, "yes"),
                    (0.82, "yes"),
                    (0.85, "yes"),
                    (0.92, "yes"),
                    (0.95, "yes"),
                    (0.97, "yes"),
                    (0.10, "no"),
                    (0.30, "no"),
                    (0.76, "no"),
                    (0.78, "no"),
                ]
                log.write_text(
                    "".join(
                        json.dumps(
                            {
                                "id": f"g{i}",
                                "at": 0,
                                "session": "s",
                                "app": "guard",
                                "verdict": "held" if s >= 0.9 else "allowed",
                                "score": s,
                                "threshold": 0.9,
                                "subject": f"cmd {i}",
                                "detail": {},
                            }
                        )
                        + "\n"
                        for i, (s, _) in enumerate(scores)
                    )
                )
                self.assertEqual(
                    c.main(
                        [
                            "--id",
                            "vsr@inline",
                            "label",
                            *[f"g{i}={label}" for i, (_, label) in enumerate(scores)],
                        ]
                    ),
                    0,
                )
                plan_path = Path(tmp) / "plan.json"
                self.assertEqual(
                    c.main(["--id", "vsr@inline", "suggest", "--out", str(plan_path)]),
                    0,
                )
                plan = json.loads(plan_path.read_text())
                self.assertEqual(
                    plan["changes"], {"guardReview": {"from": 0.9, "to": 0.8}}
                )
                self.assertEqual(
                    json.loads(settings_path.read_text())["pluginConfigs"][
                        "vsr@inline"
                    ]["options"].get("guardReview"),
                    None,
                    "suggest writes nothing",
                )
                self.assertEqual(
                    c.main(["--id", "vsr@inline", "apply", str(plan_path)]), 0
                )
                saved = json.loads(settings_path.read_text())
                self.assertEqual(saved["theme"], "dark")
                self.assertEqual(
                    saved["pluginConfigs"]["vsr@inline"]["options"],
                    {"logPath": str(log), "guardReview": 0.8},
                )
                record = json.loads(
                    (log.parent / "calibrations.jsonl").read_text().splitlines()[-1]
                )
                self.assertEqual(
                    record["changes"], {"guardReview": {"from": 0.9, "to": 0.8}}
                )
            finally:
                del os.environ["CLAUDE_CONFIG_DIR"]


if __name__ == "__main__":
    unittest.main()
