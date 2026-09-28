import importlib
import json
import random
import tempfile
import unittest
from pathlib import Path

mixtures = importlib.import_module("v2.27b.build_mixtures")
contrast = importlib.import_module("v2.27b.contrast")
typed_collect = importlib.import_module("v2.27b.typed_collect")
launch = importlib.import_module("v2.27b.launch")
aho_eval = importlib.import_module("v2.27b.aho_eval")


def rows(prefix, groups, per_group=2, source="s", types=("choice", "noul")):
    out, lengths = [], {}
    rng = random.Random(prefix)
    for g in range(groups):
        for r in range(per_group):
            row_id = f"{prefix}-{g}-{r}"
            out.append(
                {
                    "id": row_id,
                    "group_id": f"{prefix}-g{g}",
                    "source": source,
                    "family": f"{prefix}-fam{g % 3}",
                    "task_type": types[(g + r) % len(types)],
                    "language": "en" if g % 4 else "zh",
                }
            )
            lengths[row_id] = rng.randint(50, 400)
    return out, lengths


class MixtureTest(unittest.TestCase):
    def test_admission_excludes_over_limit_rows(self):
        data = [{"id": "a", "task_type": "choice"}, {"id": "b", "task_type": "score"}]
        kept, lengths, report = mixtures.admit(
            data, lambda r: {"a": 10, "b": 11}[r["id"]], 10
        )
        self.assertEqual([r["id"] for r in kept], ["a"])
        self.assertEqual(lengths, {"a": 10})
        self.assertEqual(report["over_limit_by_type"], {"score": 1})

    def test_template_s_mixtures(self):
        base, base_len = rows("base", 60)
        arm, arm_len = rows("arm", 80, source="t")
        lengths = {**base_len, **arm_len}
        specs = ["control=base:full,resample:rho", "treat=base:full,arm:rho"]
        out, report = mixtures.build(base, {"arm": arm}, dict(lengths), specs, "seed")
        again, report_again = mixtures.build(
            base, {"arm": arm}, dict(lengths), specs, "seed"
        )
        self.assertEqual(out, again)
        self.assertEqual(report, report_again)
        rho = sum(base_len.values()) // 2
        group_tokens = {}
        for row in base + arm:
            group_tokens[row["group_id"]] = (
                group_tokens.get(row["group_id"], 0) + lengths[row["id"]]
            )
        largest = max(group_tokens.values())
        for name in ("control", "treat"):
            part = report[name]["parts"]["resample" if name == "control" else "arm"]
            self.assertEqual(part["target_tokens"], rho)
            self.assertLessEqual(abs(part["tokens"] - rho), largest)
            ids = [r["id"] for r in out[name]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(ids, sorted(ids))
        duplicates = [
            r for r in out["control"] if r["id"].endswith(mixtures.RESAMPLE_SUFFIX)
        ]
        self.assertTrue(duplicates)
        originals = {r["id"]: r for r in base}
        for row in duplicates:
            original = originals[row["id"][: -len(mixtures.RESAMPLE_SUFFIX)]]
            self.assertEqual({**row, "id": original["id"]}, original)
        by_group = {}
        for row in out["treat"]:
            if row["source"] == "t":
                by_group.setdefault(row["group_id"], 0)
                by_group[row["group_id"]] += 1
        self.assertTrue(all(count == 2 for count in by_group.values()))

    def test_order_is_stratified_and_seeded(self):
        data, lengths = rows("x", 90)
        groups = mixtures.group_rows(data)
        tokens = {g: sum(lengths[r["id"]] for r in m) for g, m in groups.items()}
        first = mixtures.stratified_order(groups, tokens, "a")
        self.assertEqual(first, mixtures.stratified_order(groups, tokens, "a"))
        self.assertNotEqual(first, mixtures.stratified_order(groups, tokens, "b"))
        half = mixtures.take_prefix(first, tokens, sum(tokens.values()) // 2)
        strata = {mixtures.stratum(groups[g]) for g in groups}
        for key in strata:
            members = [g for g in groups if mixtures.stratum(groups[g]) == key]
            share = sum(tokens[g] for g in half if g in members) / sum(
                tokens[g] for g in members
            )
            self.assertGreater(share, 0.2)
            self.assertLess(share, 0.8)

    def test_bad_specs_fail(self):
        with self.assertRaises(ValueError):
            mixtures.parse_mixture("x=arm:rho")
        base, lengths = rows("b", 4)
        with self.assertRaises(ValueError):
            mixtures.build(base, {}, dict(lengths), ["x=base:full,resample:full"], "s")


class ContrastTest(unittest.TestCase):
    def test_css_macro_matches_frozen_scorer(self):
        from transfer.score import macro_f1

        rng = random.Random(1)
        labels = ["a", "b", "c"]
        panels = contrast.Panels.__new__(contrast.Panels)
        panels.css = [
            {"task": "t", "gold": rng.choice(labels), "labels": labels}
            for _ in range(200)
        ]
        panels.task_items = {"t": list(range(200))}
        choices = [rng.choice(labels + [None]) for _ in range(200)]
        got = contrast.css_macro(panels, choices, panels.task_items)["css:t"]
        self.assertAlmostEqual(
            got, macro_f1([r["gold"] for r in panels.css], choices, labels), places=12
        )

    def test_dev_family_macro(self):
        panels = contrast.Panels.__new__(contrast.Panels)
        panels.families = ["f1", "f2"]
        outcomes = [
            [("f1", "choice", True, True)],
            [("f1", "score", False, True)],
            [("f2", "noul", True, True)],
        ]
        metrics = contrast.dev_metrics(panels, outcomes, [0, 1, 2])
        self.assertAlmostEqual(metrics["T_dev"], (0.5 + 1.0) / 2)
        self.assertEqual(metrics["score_accuracy"], 0.0)

    def test_decision_rule(self):
        base = {
            "T_dev": 0.70,
            "H_pilot": 0.60,
            "cal_brier": 0.10,
            "invalid": 2,
            "choice_accuracy": 0.9,
            "noul_accuracy": 0.6,
            "score_accuracy": 0.4,
        }
        better = {**base, "T_dev": 0.76, "score_accuracy": 0.39}
        boot = {"T_dev": {"ci95": [0.01, 0.1]}}
        self.assertTrue(
            contrast.decide([better], [base], boot, "T_dev", 0.0186)["passes"]
        )
        drop = {**better, "noul_accuracy": 0.55}
        self.assertFalse(
            contrast.decide([drop], [base], boot, "T_dev", 0.0186)["passes"]
        )
        self.assertFalse(
            contrast.decide(
                [better], [base], {"T_dev": {"ci95": [-0.01, 0.1]}}, "T_dev", 0.0186
            )["passes"]
        )

    def test_percentile(self):
        self.assertEqual(contrast.percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(contrast.percentile([0, 10], 0.25), 2.5)


class CollectAndLeaseTest(unittest.TestCase):
    def test_smoke_argv_feeds_first_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "p.jsonl"
            source.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(20)))
            output = Path(tmp) / "out" / "x.predictions.jsonl"
            argv = ["--input", str(source), "--output", str(output), "--max-items", "3"]
            new = typed_collect.smoke_argv(argv)
            self.assertNotIn("--max-items", new)
            subset = Path(new[new.index("--input") + 1])
            self.assertEqual(len(subset.read_text().splitlines()), 3)
            self.assertEqual(
                typed_collect.smoke_argv(["--input", "a"]), ["--input", "a"]
            )

    def test_fla_overlay_removed(self):
        import sys

        self.assertFalse(any(typed_collect.FLA_OVERLAY in p for p in sys.path))

    def test_wait_until_idle(self):
        readings = iter(["31", "12", "0"])
        original = launch.vram_percent
        try:
            launch.vram_percent = lambda gpu: next(readings)
            launch.wait_until_idle(5, timeout=5, poll=0)
            launch.vram_percent = lambda gpu: "31"
            with self.assertRaises(ValueError):
                launch.wait_until_idle(5, timeout=0, poll=0)
        finally:
            launch.vram_percent = original

    def test_read_lease_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner = Path(tmp) / "owner"
            owner.write_text(json.dumps({"track": "27b", "status": "running"}))
            self.assertEqual(launch.read_lease(owner)["status"], "running")
            owner.write_text("track=27b\npurpose=a=b\nlast_job_exit=0\n")
            self.assertEqual(
                launch.read_lease(owner),
                {"track": "27b", "purpose": "a=b", "last_job_exit": "0"},
            )


class AhoSummaryTest(unittest.TestCase):
    def test_over_limit_rows_count_as_failures(self):
        rows_by_id = {
            "a": {
                "task_type": "score",
                "family": "f",
                "language": "en",
                "options": [0, 1, 2],
            },
            "b": {
                "task_type": "score",
                "family": "f",
                "language": "en",
                "options": [0, 1, 2, 3],
            },
        }
        records = [{"id": "a", "task_type": "score", "family": "f", "correct": True}]
        summary = aho_eval.summarize(records, rows_by_id, ["b"])
        self.assertEqual(
            (summary["n"], summary["correct"], summary["over_limit"]), (2, 1, 1)
        )
        self.assertEqual(summary["micro_accuracy"], 0.5)
        self.assertEqual(summary["cells"]["levels:3"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
