"""CPU tests for the decoder Milestone 5 formal tooling (synthetic inputs)."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "ops" / "m5"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, OPS / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


results = _load("m5_results", "m5-results.py")
mlx_agg = _load("m5_mlx_agg", "m5-mlx-agg.py")
receipt = _load("m5_receipt", "m5-receipt.py")
params = _load("m5_params", "m5-params.py")

NOX1_TYPED = {"choice": 552, "noul": 653, "score": 178}


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def dev_arm(proxy, typed=(500, 260, 360)):
    return {
        "T": 0.7,
        "H": 0.55,
        "proxy": proxy,
        "H_mean": 0.56,
        "proxy_mean_H": proxy - 0.5,
        "by_type": {
            k: {"correct": v, "n": 600, "invalid": 0}
            for k, v in zip(results.TYPES, typed)
        },
    }


def paired_file(delta, lo, hi, h_delta=0.01, h_lo=-0.02, h_hi=0.04):
    return {
        "point": {"delta": {"score": delta, "T": 0.01, "H": h_delta}},
        "ci95": {"low": lo, "high": hi},
        "axis_ci95": {
            "T": {"delta": {"low": -0.01, "high": 0.03}},
            "H": {"delta": {"low": h_lo, "high": h_hi}},
        },
    }


def report(typed=(580, 730, 180), v3=64.0):
    return {
        "v3": {"score": v3, "T": 0.69, "H": 0.59},
        "panels": {
            "typed-final": {
                "by_type": {k: {"correct": v} for k, v in zip(results.TYPES, typed)}
            },
            "public231": {
                "correct": 170,
                "items": 231,
                "tiers": {
                    "easy": {"correct": 48},
                    "standard": {"correct": 66},
                    "hard": {"correct": 56},
                },
            },
        },
    }


def typed_predictions(path: Path, constant_score=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(20):
        rows.append(
            {"id": f"c{i}", "answers": {"a": {"type": "choice", "choice": "xy"[i % 2]}}}
        )
        rows.append(
            {
                "id": f"n{i}",
                "answers": {"a": {"type": "noul", "noul": 0.9 if i % 3 else 0.1}},
            }
        )
        probs = (
            {"0": 0.9, "1": 0.1} if constant_score or i % 2 else {"0": 0.1, "1": 0.9}
        )
        rows.append(
            {"id": f"s{i}", "answers": {"a": {"type": "score", "probabilities": probs}}}
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def mlx_agg_file(noul, choice=0.80, score=0.70):
    return {
        "overall": 0.78,
        "non_english_by_type": {"choice": choice, "noul": noul, "score": score},
        "per_language_mean_accuracy": {"ko": 0.7, "ja": 0.72, "en": 0.85},
        "non_english_noul": {"pred_yes_rate_mean": 0.7, "gold_no_recall_mean": 0.55},
    }


class Results(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dev, self.formal = root / "dev", root / "formal"
        self.nox = root / "nox1" / "REPORT.json"
        write(self.nox, report(typed=tuple(NOX1_TYPED.values()), v3=56.47))
        ref = self.formal / results.REF_RUN
        write(ref / "REPORT.json", report(v3=63.15))
        write(ref / "PAIRED-vs-n4xf-nodeA.json", paired_file(0.02, -0.3, 0.4))
        write(ref / "PAIRED-vs-adopted-1.0.json", paired_file(6.68, 0.99, 9.64))
        write(
            ref / "REPEAT-vs-n4xf-nodeA.json",
            {
                "panels": {
                    "typed-final": {"category_changes": 3},
                    "css15": {"category_changes": 5},
                }
            },
        )
        typed_predictions(ref / "output" / "typed-final.predictions.jsonl")
        write(
            self.formal / f"{results.REF_RUN}-mlx" / "mlx-agg.json", mlx_agg_file(0.727)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def arm(
        self,
        arm,
        soup_proxy=63.0,
        seeds=(62.0, 61.0, 63.5),
        paired_ref=None,
        typed=(580, 730, 180),
        mlx=None,
        constant_score=False,
    ):
        arms = {
            "nox1": dev_arm(55.3),
            "n4xf": dev_arm(62.9),
            "soup": dev_arm(soup_proxy),
        }
        arms.update({s: dev_arm(p) for s, p in zip(results.SEEDS, seeds)})
        write(
            self.dev / "soup" / arm / "readout.json",
            {
                "arms": arms,
                "comparisons": {
                    "soup-minus-n4xf": {
                        "delta_b_minus_a": {"proxy": {"lower95": -2.0, "upper95": 2.5}}
                    }
                },
            },
        )
        chosen = results.pick_artifact(arms)
        name = f"m5-{arm}-{chosen}"
        mdir = self.dev / "mlxdev" / "readouts" / name
        write(
            mdir / "score.json",
            {
                "noul_ml": 0.8,
                "choice_ml": 0.7,
                "score_ml": 0.6,
                "m_dev": 0.7,
                "noul_pred_yes_rate_macro": 0.6,
                "noul_gold_no_recall_macro": 0.7,
            },
        )
        write(
            mdir / "vs-n4xf-soup.json",
            {
                "metrics": {
                    k: {"diff": 0.02, "ci95": [0.01, 0.03]} for k in results.MLX_KEYS
                }
            },
        )
        if paired_ref is not None:
            run = self.formal / name
            write(run / "REPORT.json", report(typed=typed))
            write(run / "PAIRED-vs-n4xf-ref.json", paired_ref)
            write(run / "PAIRED-vs-adopted-1.0.json", paired_file(7.5, 2.0, 10.0))
            write(run / "PAIRED-vs-decider4b.json", paired_file(3.0, -1.0, 6.0))
            typed_predictions(
                run / "output" / "typed-final.predictions.jsonl", constant_score
            )
        if mlx is not None:
            write(self.formal / f"{name}-mlx" / "mlx-agg.json", mlx)
        return name

    def build(self, arms):
        return results.build(self.dev, self.formal, self.nox, arms)

    def test_artifact_rule(self):
        arms = {
            "soup": {"proxy": 61.0},
            "s1": {"proxy": 59.0},
            "s2": {"proxy": 62.0},
            "s3": {"proxy": 61.0},
        }
        self.assertEqual(results.pick_artifact(arms), "soup")
        arms["soup"]["proxy"] = 60.5
        arms["s1"]["proxy"] = 63.0
        self.assertEqual(results.pick_artifact(arms), "s2")
        self.assertEqual(results.pick_artifact(arms, "s1"), "s1")

    def test_successor(self):
        self.arm("N5N", paired_ref=paired_file(1.8, 0.2, 3.5, h_delta=0.004))
        row = self.build(["N5N"])["rows"][0]
        self.assertEqual(row["outcome"], "successor")
        self.assertEqual(row["development"]["artifact"], "soup")
        self.assertAlmostEqual(row["post_key"]["vs_bar"]["H_delta"], 0.004)
        self.assertEqual(row["post_key"]["collapsed"], [])

    def test_successor_needs_human_transfer(self):
        self.arm("N5N", paired_ref=paired_file(1.8, 0.2, 3.5, h_delta=-0.001))
        self.assertEqual(self.build(["N5N"])["rows"][0]["outcome"], "negative")

    def test_successor_needs_typed_floor_and_no_collapse(self):
        self.arm("N5N", paired_ref=paired_file(1.8, 0.2, 3.5), typed=(580, 480, 180))
        row = self.build(["N5N"])["rows"][0]
        self.assertFalse(row["post_key"]["typed_floor_ok"])
        self.assertNotEqual(row["outcome"], "successor")
        self.arm("N5B", paired_ref=paired_file(1.8, 0.2, 3.5), constant_score=True)
        row = self.build(["N5B"])["rows"][0]
        self.assertEqual(row["post_key"]["collapsed"], ["score"])
        self.assertNotEqual(row["outcome"], "successor")

    def test_card_only_and_pending_mlx(self):
        self.arm("N5B", paired_ref=paired_file(-0.5, -2.0, 1.0))
        self.assertEqual(
            self.build(["N5B"])["rows"][0]["outcome"], "pending (mlx-diag)"
        )
        self.arm(
            "N5BN",
            paired_ref=paired_file(-0.5, -2.0, 1.0),
            mlx=mlx_agg_file(0.770, 0.795, 0.695),
        )
        self.assertEqual(
            self.build(["N5BN"])["rows"][0]["outcome"], "card-only multilingual fix"
        )
        self.arm(
            "N5N",
            paired_ref=paired_file(-0.5, -2.0, 1.0),
            mlx=mlx_agg_file(0.770, 0.785, 0.70),
        )
        self.assertEqual(self.build(["N5N"])["rows"][0]["outcome"], "negative")

    def test_negative_and_pending_post_key(self):
        self.arm("N5N", paired_ref=paired_file(-1.5, -3.0, 0.5), mlx=mlx_agg_file(0.80))
        self.arm("N5B")
        rows = {r["arm"]: r for r in self.build(["N5N", "N5B"])["rows"]}
        self.assertEqual(rows["N5N"]["outcome"], "negative")
        self.assertEqual(rows["N5B"]["outcome"], "pending (post-key)")

    def test_reference_and_markdown(self):
        self.arm("N5N", soup_proxy=60.0, paired_ref=paired_file(1.8, 0.2, 3.5))
        result = self.build(["N5N", "N5BN"])
        self.assertEqual(result["reference"]["answer_changes_vs_nodeA_total"], 8)
        self.assertEqual(result["rows"][0]["development"]["artifact"], "s1")
        md = results.markdown(result)
        for label in (
            "## DEVELOPMENT",
            "## POST-KEY",
            "## DIAGNOSTIC",
            "human transfer ΔH",
            "never release scores",
        ):
            self.assertIn(label, md)
        self.assertIn("missing", md)


class MlxAgg(unittest.TestCase):
    def test_bias_means_over_non_english(self):
        gold, preds = [], {}
        for lang, answers in (
            ("en", [True, False, True, False]),
            ("ja", [True, True, True, False]),
            ("ko", [True, True, False, False]),
        ):
            for i, p in enumerate(answers):
                item = f"{lang}{i}"
                gold.append(
                    {"id": item, "type": "noul", "language": lang, "value": i % 2 == 0}
                )
                preds[item] = {
                    "answers": {"q": {"type": "noul", "noul": 0.8 if p else 0.2}}
                }
        gold.append({"id": "c", "type": "choice", "language": "ja", "value": "x"})
        bias = mlx_agg.noul_bias(gold, preds)
        self.assertEqual(bias["ja"]["pred_yes"], 0.75)
        self.assertEqual(bias["ja"]["recall_no"], 0.5)
        self.assertEqual(bias["ko"]["recall_no"], 0.5)
        by_type = {
            t: {
                "non_english_mean_accuracy": 0.7,
                "english_accuracy": 0.8,
                "languages": {"ja": {"accuracy": 0.7}},
            }
            for t in mlx_agg.TYPES
        }
        score = {
            "predictions_sha256": "p",
            "gold_sha256": "g",
            "items": 13,
            "invalid_or_missing": 0,
            "type_macro_accuracy": 0.75,
            "english_type_macro_accuracy": 0.8,
            "non_english_type_macro_accuracy": 0.7,
            "per_language_mean_accuracy": {"ja": 0.7},
            "cross_language_consistency": 0.5,
            "by_type": by_type,
        }
        agg = mlx_agg.aggregate(score, bias)
        self.assertAlmostEqual(agg["non_english_noul"]["pred_yes_rate_mean"], 0.625)
        self.assertAlmostEqual(agg["non_english_noul"]["gold_no_recall_mean"], 0.5)
        self.assertNotIn("ja0", json.dumps(agg))


class Receipt(unittest.TestCase):
    def test_manifest_matches_find_and_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            (root / "B" / "a").mkdir(parents=True)
            (root / "a.json").write_text("1")
            (root / "B" / "a" / "k.bin").write_text("2")
            (root / "_x").write_text("3")
            text = receipt.tree_manifest(root)
            shell = subprocess.run(
                "find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum",
                shell=True,
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertEqual(text, shell)
            before = receipt.parse_manifest(text)
            (root / "new.json").write_text("4")
            (root / "a.json").write_text("5")
            diff = receipt.cache_diff(
                before, receipt.parse_manifest(receipt.tree_manifest(root))
            )
            self.assertEqual(
                (diff["new_entries"], diff["changed_entries"], diff["removed_entries"]),
                (1, 1, 0),
            )
            self.assertTrue(diff["flag_new_cache_entries"])
            self.assertFalse(
                receipt.cache_diff(before, before)["flag_new_cache_entries"]
            )


class Params(unittest.TestCase):
    def test_header_stubs_count_like_the_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg, stubs = Path(tmp) / "pkg", Path(tmp) / "stubs"
            (pkg / "backbone").mkdir(parents=True)
            for name, shapes in (
                ("backbone/model-1.safetensors", [[4, 3], [5]]),
                ("head.safetensors", [[2, 2, 2]]),
            ):
                header = {
                    f"t{i}": {"dtype": "F32", "shape": s, "data_offsets": [0, 0]}
                    for i, s in enumerate(shapes)
                }
                header["__metadata__"] = {"format": "pt"}
                blob = json.dumps(header).encode()
                (pkg / name).write_bytes(
                    struct.pack("<Q", len(blob)) + blob + b"\0" * 64
                )
            params.write_stubs(pkg, stubs)
            real, stub = params.count_safetensors(pkg), params.count_safetensors(stubs)
            self.assertEqual(real["parameters"], 25)
            self.assertEqual(stub, real)
            self.assertLess(
                (stubs / "head.safetensors").stat().st_size,
                (pkg / "head.safetensors").stat().st_size,
            )


if __name__ == "__main__":
    unittest.main()
