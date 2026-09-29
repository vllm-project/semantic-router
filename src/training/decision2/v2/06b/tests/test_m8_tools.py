import importlib
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from training.model.infer import normalized_answer
from training.model.score_bias import validate_score_bias

m8 = importlib.import_module("v2.06b.m8_scorebias")
m7 = importlib.import_module("v2.06b.m7_scorebias")
common = importlib.import_module("v2.06b.common")

TRUE_BIAS = [0.0, -0.4, 0.3, 0.6, 1.8]


def quiet(fn, *args):
    with redirect_stdout(io.StringIO()):
        return fn(*args)


def synthetic_probs(rng, y, bias=TRUE_BIAS, signal=1.5):
    t = [signal if k == y else 0.0 for k in range(5)]
    return m7.softmax([a + rng.gauss(0, 0.8) + b for a, b in zip(t, bias)])


def synthetic_items(n, seed=3, source="ledger", prior=(0.1, 0.15, 0.2, 0.25, 0.3)):
    rng = random.Random(seed)
    items = []
    for i in range(n):
        y = rng.choices(range(5), weights=prior)[0]
        items.append(
            {
                "id": f"{source}{i}",
                "source": source,
                "p": synthetic_probs(rng, y),
                "y": y,
            }
        )
    return items


def rows_of(items):
    return [(m7.logits_of(i["p"]), i["y"]) for i in items]


def score_answer(p):
    return normalized_answer("score", m8.KEYS, m7.logits_of(p), 1.0)


def gold_row(key, half, y):
    return {
        "id": key,
        "half": half,
        "task": "score5t/resource_ledger",
        "group_id": f"g-{key}",
        "cluster_id": f"g-{key}",
        "source": "synthetic",
        "language": "en",
        "long": False,
        "questions": {
            "decision": {"type": "score", "instructions": "Rate", "criteria": m8.KEYS}
        },
        "gold": {"decision": {"type": "score", "value": y, "semantic_value": y}},
    }


class WeightsTest(unittest.TestCase):
    def test_weights_are_normalized_within_a_source(self):
        labels = [0, 0, 0, 1, 2, 2, 4, 4, 4, 4]
        for lam in (1.0, 0.5, 0.0):
            w = m8.source_weights(labels, lam)
            self.assertAlmostEqual(sum(w), 1.0, places=12)
        uniform = m8.source_weights(labels, 0.0)
        self.assertTrue(all(abs(v - 0.1) < 1e-15 for v in uniform))
        balanced = m8.source_weights(labels, 1.0)
        for level in set(labels):
            total = sum(w for w, y in zip(balanced, labels) if y == level)
            self.assertAlmostEqual(total, 1 / 4, places=12)
        half = m8.source_weights(labels, 0.5)
        self.assertAlmostEqual(half[0] / half[6], math.sqrt(4 / 3), places=12)

    def test_block_weights_are_one_and_w_h(self):
        ledger = synthetic_items(40, 1)
        human = synthetic_items(90, 2, "human")
        for w_h in (0.0, 1.0, 0.5):
            rows, weights = m8.objective_rows(
                {"ledger": ledger, "human": human}, 0.5, w_h
            )
            self.assertEqual(len(rows), 40 + (90 if w_h else 0))
            self.assertAlmostEqual(sum(weights[:40]), 1.0, places=12)
            self.assertAlmostEqual(sum(weights[40:]), w_h, places=12)


class FitTest(unittest.TestCase):
    def test_lam_1_without_human_rows_is_m7_fit_level(self):
        for seed in (5, 6, 7):
            items = synthetic_items(400, seed)
            rows, weights = m8.objective_rows({"ledger": items}, 1.0, 0.0)
            ours = m8.fit_weighted(rows, weights, 5)
            theirs = m7.fit_level(rows_of(items), 5)
            self.assertEqual(ours["offsets"], theirs["offsets"])
            # M7's weights are 5x ours, so the |g| < 1e-9 stop lands ~1e-9 apart.
            for a, b in zip(ours["raw_offsets"], theirs["raw_offsets"]):
                self.assertAlmostEqual(a, b, places=7)
            self.assertLess(ours["gradient_norm"], 1e-9)

    def test_lam_0_matches_the_empirical_level_prior(self):
        items = synthetic_items(400, 9)
        rows, weights = m8.objective_rows({"ledger": items}, 0.0, 0.0)
        result = m8.fit_weighted(rows, weights, 5)
        b = result["raw_offsets"]
        mean = [0.0] * 5
        for z, _ in rows:
            for k, v in enumerate(m7.softmax([a + c for a, c in zip(z, b)])):
                mean[k] += v / len(rows)
        counts = [sum(i["y"] == k for i in items) / len(items) for k in range(5)]
        for got, want in zip(mean, counts):
            self.assertLess(abs(got - want), 1e-6)
        self.assertAlmostEqual(sum(result["offsets"]), 0.0, places=5)

    def test_human_block_moves_the_fit_and_zero_weight_does_not(self):
        ledger = synthetic_items(300, 11)
        human = synthetic_items(900, 12, "human", prior=(0.4, 0.3, 0.15, 0.1, 0.05))
        alone = m8.fit_weighted(*m8.objective_rows({"ledger": ledger}, 0.0, 0.0), 5)
        zero = m8.fit_weighted(
            *m8.objective_rows({"ledger": ledger, "human": human}, 0.0, 0.0), 5
        )
        both = m8.fit_weighted(
            *m8.objective_rows({"ledger": ledger, "human": human}, 0.0, 1.0), 5
        )
        self.assertEqual(alone["offsets"], zero["offsets"])
        self.assertNotEqual(alone["offsets"], both["offsets"])

    def test_missing_level_stops_the_fit(self):
        items = [i for i in synthetic_items(200, 4) if i["y"] != 3]
        with self.assertRaisesRegex(m7.FitError, r"miss level\(s\) \[3\]"):
            m8.fit_weighted(*m8.objective_rows({"ledger": items}, 1.0, 0.0), 5)


class FitCommandTest(unittest.TestCase):
    def fixture(self, root: Path):
        rng = random.Random(21)
        gold, predictions = [], []
        for i in range(160):
            half = "fit" if i % 2 == 0 else "check"
            y = rng.choices(range(5), weights=(1, 2, 2, 3, 4))[0]
            gold.append(gold_row(f"i{i}", half, y))
            predictions.append(
                {
                    "id": f"i{i}",
                    "model_sha256": m8.MODEL_SHA256,
                    "answers": {"decision": score_answer(synthetic_probs(rng, y))},
                }
            )
        paths = {
            "fit": root / "fit.gold.jsonl",
            "check": root / "check.gold.jsonl",
            "pred": root / "pred.jsonl",
        }
        common.write_jsonl(paths["fit"], [g for g in gold if g["half"] == "fit"])
        common.write_jsonl(paths["check"], [g for g in gold if g["half"] == "check"])
        common.write_jsonl(paths["pred"], predictions)
        expected = {
            **m8.EXPECTED,
            "score5t_fit_gold": common.file_sha256(paths["fit"]),
            "score5t_check_gold": common.file_sha256(paths["check"]),
            "score5t_predictions": common.file_sha256(paths["pred"]),
        }
        return gold, paths, expected

    def argv(self, root, paths, name, lam, w_h):
        return [
            "fit", "--name", name, "--lam", lam, "--human-weight", w_h,
            "--fit-gold", str(paths["fit"]), "--check-gold", str(paths["check"]),
            "--predictions", str(paths["pred"]),
            "--output", str(root / name / "score_bias.json"),
            "--report", str(root / name / "FIT.json"),
        ]  # fmt: skip

    def test_fit_writes_a_valid_bound_file_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold, paths, expected = self.fixture(root)
            with (
                mock.patch.dict(m8.EXPECTED, expected),
                mock.patch.object(m8, "score5t_gold", return_value=(gold, {})),
                mock.patch.object(
                    m7, "soup_model_sha256", return_value=m8.MODEL_SHA256
                ),
            ):
                for name, lam in (("s5-b1", "1"), ("s5-b0", "0")):
                    argv = self.argv(root, paths, name, lam, "0")
                    self.assertEqual(quiet(m8.main, argv), 0)
                    bias = json.loads((root / name / "score_bias.json").read_text())
                    offsets = validate_score_bias(bias, m8.MODEL_SHA256)
                    self.assertEqual(set(offsets), {5})
                    self.assertEqual(set(bias["offsets"]), {"5"})
                    self.assertEqual(bias["fit"]["candidate"], name)
                    self.assertEqual(bias["fit"]["rows"]["ledger"]["rows"], 80)
                    self.assertEqual(bias["fit"]["prereg_commit"], m8.PREREG_COMMIT)
                    report = json.loads((root / name / "FIT.json").read_text())
                    self.assertEqual(
                        report["score_bias"]["sha256"],
                        common.file_sha256(root / name / "score_bias.json"),
                    )
                    with self.assertRaises(FileExistsError):
                        quiet(m8.main, argv)
                bias = json.loads((root / "s5-b1" / "score_bias.json").read_text())
                self.assertEqual(
                    bias["fit"]["fit"]["m7_fit_level_offsets"], bias["offsets"]["5"]
                )
                with self.assertRaisesRegex(ValueError, "differs from prereg"):
                    quiet(m8.main, self.argv(root, paths, "s5h-b1", "1", "0"))
                m8.EXPECTED["score5t_predictions"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "not the preregistered"):
                    quiet(m8.main, self.argv(root, paths, "s5-b05", "0.5", "0"))

    def test_human_rows_drop_score5_dev_panel_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel-rows.jsonl"
            common.write_jsonl(
                panel,
                [
                    {"panel_id": "score5-x", "source_row_id": "a1", "input_sha256": "s", "group_id": "g"},
                    {"panel_id": "c2", "source_row_id": "zz", "input_sha256": "t", "group_id": "h"},
                ],
            )  # fmt: skip
            items = [
                {"id": "a1", "source": "A7q", "levels": 5, "p": [0.2] * 5, "y": 1},
                {"id": "a2", "source": "A7q", "levels": 5, "p": [0.2] * 5, "y": 2},
                {"id": "a3", "source": "H6", "levels": 3, "p": [0.3] * 3, "y": 0},
                {"id": "c1", "source": "CAL698", "levels": 5, "p": [0.2] * 5, "y": 4},
                {"id": "c2", "source": "CAL698", "levels": 5, "p": [0.2] * 5, "y": 0},
            ]
            provenance = {
                "fit_aho_probs_sha256": m8.EXPECTED["fit_aho_probs"],
                "best_export_manifest_sha256": m8.EXPORT_MANIFEST_SHA256,
            }
            args = mock.Mock(
                aho_dir=root, soup_dir=root, cal_gold=root, probs=root,
                score5_panel_rows=panel,
            )  # fmt: skip
            with (
                mock.patch.object(m8, "verified"),
                mock.patch.object(m7, "fit_inputs", return_value=(items, provenance)),
            ):
                kept, report = m8.human_items(args)
        self.assertEqual([i["id"] for i in kept], ["a2", "c1"])
        self.assertTrue(all(i["source"] == "human" for i in kept))
        self.assertEqual(report["removed_by_source"], {"A7q": 1, "CAL698": 1})
        self.assertEqual(report["rows_l5_before_removal"], 4)
        self.assertEqual(
            report["rows_by_level"], {"0": 0, "1": 0, "2": 1, "3": 0, "4": 1}
        )


class CheckPiecesTest(unittest.TestCase):
    def test_corrected_answer_applies_offsets_on_log_p(self):
        answer = score_answer([0.1, 0.1, 0.1, 0.1, 0.6])
        p = [answer["probabilities"][k] for k in m8.KEYS]
        row = [0.5, 0.2, 0.0, -0.1, -0.6]
        fixed = m8.corrected_answer(answer, row)
        want = normalized_answer(
            "score", m8.KEYS, [math.log(a) + b for a, b in zip(p, row)], 1.0
        )
        self.assertEqual(fixed, want)
        for bad in (None, {"type": "score", "score": 2.0}, {"type": "choice"}):
            self.assertEqual(m8.corrected_answer(bad, row), bad)

    def test_d1_rejects_collapse_and_warn_only(self):
        block = {
            "top_share": 0.5, "top_share_wilson95": [0.45, 0.55],
            "accuracy": 0.3, "accuracy_wilson95": [0.25, 0.35],
            "always_majority_accuracy": 0.27, "acc_minus_majority_boot95": [-0.01, 0.07],
        }  # fmt: skip
        for flags, ok in (([], True), (["NO-GAIN"], True), (["WARN"], False),
                          (["COLLAPSE", "NO-GAIN"], False)):  # fmt: skip
            self.assertEqual(m8.d1_gate({**block, "flags": flags})["pass"], ok)

    def test_d3_gate(self):
        before = m7.summary([4, 4, 4, 1], [1, 2, 4, 1], intervals=True)
        after = m7.summary([1, 2, 4, 1], [1, 2, 4, 1], intervals=True)
        paired = m7.paired_delta([4, 4, 4, 1], [1, 2, 4, 1], [1, 2, 4, 1])
        gate = m8.d3_gate(before, after, paired)
        self.assertTrue(gate["checks"]["a_delta_upper_bound_ge_0"])
        self.assertTrue(gate["checks"]["b_modal_share_le_0.90"])
        self.assertEqual(gate["accuracy_after"], 1.0)
        worse = m7.paired_delta([1, 2, 4, 1], [4, 4, 4, 4], [1, 2, 4, 1])
        gate = m8.d3_gate(
            after, m7.summary([4, 4, 4, 4], [1, 2, 4, 1], intervals=True), worse
        )
        self.assertFalse(gate["pass"])
        self.assertFalse(gate["checks"]["b_modal_share_le_0.90"])


def check_record(name, acc, chk5, top, d1=True, d2=True, d3=True):
    lam, w_h = m8.CANDIDATES[name]
    return {
        "candidate": name,
        "lam": lam,
        "human_weight": w_h,
        "score_bias": {"sha256": name, "offsets": {"5": [0.0] * 5}},
        "gates": {
            "D1": {"pass": d1, "flags": [], "accuracy": acc, "top_share": top},
            "D2": {"pass": d2},
            "D3": {"pass": d3, "accuracy_after": chk5},
        },
    }


class SelectTest(unittest.TestCase):
    def test_ranking_and_tie_breaks(self):
        checks = [
            check_record("s5-b1", 0.30, 0.40, 0.50),
            check_record("s5h-b1", 0.30, 0.40, 0.50),
            check_record("s5-b05", 0.30, 0.40, 0.45),
            check_record("s5h-b05", 0.30, 0.41, 0.60),
            check_record("s5-b0", 0.35, 0.30, 0.80, d2=False),
            check_record("s5h-b0", 0.31, 0.20, 0.80),
        ]
        result = m8.ranking(checks)
        ranked = [(r["candidate"], r["rank"]) for r in result["table"]]
        self.assertEqual(
            ranked,
            [("s5h-b0", 1), ("s5h-b05", 2), ("s5-b05", 3), ("s5-b1", 4),
             ("s5h-b1", 5), ("s5-b0", None)],
        )  # fmt: skip
        self.assertEqual(result["finalists"], ["s5h-b0", "s5h-b05", "s5-b05"])
        self.assertEqual(result["candidates_missing"], [])

    def test_no_eligible_candidate_and_bad_names(self):
        result = m8.ranking([check_record("s5-b1", 0.3, 0.4, 0.9, d1=False)])
        self.assertEqual(result["finalists"], [])
        self.assertFalse(result["successor_possible"])
        self.assertEqual(len(result["candidates_missing"]), 5)
        with self.assertRaises(ValueError):
            m8.ranking([check_record("s5-b1", 0.3, 0.4, 0.5)] * 2)
        bad = check_record("s5-b1", 0.3, 0.4, 0.5)
        bad["candidate"] = "s5-extra"
        with self.assertRaises(ValueError):
            m8.ranking([bad])


class ReplayTest(unittest.TestCase):
    def setUp(self):
        rng = random.Random(8)
        self.row = [0.4, 0.3, 0.1, -0.2, -0.6]
        self.gold, self.logged = [], {}
        for i in range(40):
            y = rng.randrange(5)
            self.gold.append(gold_row(f"i{i}", "fit" if i < 20 else "check", y))
            self.logged[f"i{i}"] = {
                "id": f"i{i}",
                "model_sha256": m8.MODEL_SHA256,
                "answers": {"decision": score_answer(synthetic_probs(rng, y))},
            }
        self.sha = "b" * 64

    def online(self, jitter=0.0, bias_sha=None):
        out = {}
        for key, row in self.logged.items():
            answer = m8.corrected_answer(row["answers"]["decision"], self.row)
            if jitter:
                p = [answer["probabilities"][k] for k in m8.KEYS]
                p = [v + (jitter if k == 0 else -jitter / 4) for k, v in enumerate(p)]
                answer = score_answer(p)
            out[key] = {
                "id": key,
                "model_sha256": m8.MODEL_SHA256,
                "score_bias_sha256": bias_sha or self.sha,
                "answers": {"decision": answer},
            }
        return out

    def run_replay(self, online, manifest=None):
        return m8.replay_result(
            self.gold, self.logged, online, self.row, self.sha, manifest, replicates=200
        )

    def test_replay_passes_on_the_offline_correction(self):
        manifest = {"score_bias_sha256": self.sha, "model_sha256": m8.MODEL_SHA256}
        result = self.run_replay(self.online(), manifest)
        self.assertTrue(all(result["checks"].values()), result["checks"])
        self.assertEqual(result["max_abs_dp"], 0.0)
        self.assertEqual(result["check_flags_online"], result["check_flags_offline"])

    def test_replay_detects_drift_binding_and_missing_items(self):
        result = self.run_replay(self.online(jitter=5e-4))
        self.assertFalse(result["checks"]["max_abs_dp_le_1e-4"])
        result = self.run_replay(self.online(jitter=0.6))
        self.assertFalse(result["checks"]["same_argmax_all_items"])
        result = self.run_replay(self.online(bias_sha="c" * 64))
        self.assertFalse(result["checks"]["online_rows_score_bias_sha256"])
        online = self.online()
        online.pop("i3")
        result = self.run_replay(online)
        self.assertFalse(result["checks"]["same_items"])
        result = self.run_replay(self.online(), {"score_bias_sha256": "d" * 64})
        self.assertFalse(result["checks"]["online_manifest_score_bias_sha256"])


if __name__ == "__main__":
    unittest.main()
