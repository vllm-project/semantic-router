import importlib
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from training.model.infer import checkpoint_fingerprint, normalized_answer
from training.model.score_bias import SCORE_BIAS_FORMAT

m8 = importlib.import_module("v2.06b.m8_scorebias")
m7 = importlib.import_module("v2.06b.m7_scorebias")
package = importlib.import_module("v2.06b.m8_package")
common = importlib.import_module("v2.06b.common")
multilingual = importlib.import_module("v2.eval.multilingual_panel")

ROW = [0.039188, 0.203049, 0.079362, -0.15162, -0.169979]


def quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def score_answer(p, keys=m8.KEYS):
    return normalized_answer("score", keys, m7.logits_of(p), 1.0)


def random_p(rng, n):
    raw = [rng.random() + 0.01 for _ in range(n)]
    return [v / sum(raw) for v in raw]


def typed_item(key, gold_value, levels=5):
    return {
        "id": key,
        "questions": {
            "q": {
                "type": "score",
                "instructions": "Rate",
                "criteria": [str(i) for i in range(levels)],
            }
        },
        "gold": {
            "q": {"type": "score", "value": gold_value, "semantic_value": gold_value}
        },
    }


class ScoreReportTest(unittest.TestCase):
    def test_flags_and_agreement(self):
        rng = random.Random(5)
        gold, predictions = [], {}
        for i in range(200):
            y = rng.choice([0, 1, 2, 3, 4, 4, 4])
            gold.append(typed_item(f"t{i}", y))
            p = [0.05] * 5
            p[4 if i % 10 else y] = 0.8
            predictions[f"t{i}"] = {"answers": {"q": score_answer(p)}}
        result = m8.score_report_result(gold, predictions, replicates=200)
        score = result["score"]
        self.assertEqual(score["n"], 200)
        self.assertEqual(sum(score["histogram"].values()), 200)
        self.assertGreaterEqual(score["top_share"], 0.9)
        self.assertIn("COLLAPSE", score["flags"])
        self.assertFalse(result["R3"]["pass"])
        self.assertTrue(all(result["agreement"].values()))

    def test_balanced_answers_pass_r3(self):
        rng = random.Random(6)
        gold, predictions = [], {}
        for i in range(300):
            y = rng.randrange(5)
            gold.append(typed_item(f"t{i}", y))
            p = [0.1] * 5
            p[y if rng.random() < 0.6 else rng.randrange(5)] = 0.6
            predictions[f"t{i}"] = {"answers": {"q": score_answer(p)}}
        result = m8.score_report_result(gold, predictions, replicates=200)
        self.assertEqual(result["score"]["flags"], [])
        self.assertTrue(result["R3"]["pass"])
        self.assertFalse(result["no_gain"])

    def test_non_five_level_score_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not 5-level"):
            m8.typed_final_score_rows([typed_item("x", 1, levels=3)], {})


def mlx_panel(root: Path, rng: random.Random):
    gold, prompts, cand, rel = [], [], [], []
    for kind, langs, n in (
        ("choice", ("en", "de"), 30),
        ("noul", ("en", "fr"), 25),
        ("score", ("en", "zh"), 20),
    ):
        for lang in langs:
            for i in range(n):
                key = f"{kind}-{lang}-{i}"
                digest = f"sha-{key}"
                if kind == "choice":
                    question = {"type": "choice", "criteria": ["a", "b", "c"]}
                    value = rng.choice(["a", "b", "c"])
                elif kind == "noul":
                    question = {"type": "noul", "criteria": []}
                    value = rng.random() < 0.5
                else:
                    question = {"type": "score", "criteria": ["0", "1", "2"]}
                    value = rng.randrange(3)
                gold.append(
                    {
                        "id": key,
                        "type": kind,
                        "language": lang,
                        "value": value,
                        "input_sha256": digest,
                        "source": kind,
                        "source_id": str(i),
                    }
                )
                prompts.append({"id": key, "questions": {"q": question}})
                for rows in (cand, rel):
                    if kind == "choice":
                        answer = {"type": "choice", "choice": rng.choice("abc")}
                    elif kind == "noul":
                        answer = {"type": "noul", "noul": rng.random()}
                    else:
                        answer = score_answer(random_p(rng, 3), ["0", "1", "2"])
                    rows.append(
                        {
                            "id": key,
                            "source_input_sha256": digest,
                            "answers": {"q": answer},
                        }
                    )
    panel = root / "panel"
    panel.mkdir()
    common.write_jsonl(panel / "gold.jsonl", gold)
    common.write_jsonl(panel / "prompts.jsonl", prompts)
    runs = []
    for name, rows in (("cand", cand), ("rel", rel)):
        run = root / name
        (run / "output").mkdir(parents=True)
        path = run / "output" / "mlx-diag.predictions.jsonl"
        common.write_jsonl(path, rows)
        common.write_json(run / "mlx-diag.score.json", multilingual.score(panel, path))
        runs.append(run)
    return panel, runs


class MlxPairedTest(unittest.TestCase):
    def test_reproduces_multilingual_score_and_bootstraps(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel, (cand_run, rel_run) = mlx_panel(Path(tmp), random.Random(11))
            cand, acc, prov = m8.mlx_run(cand_run, panel)
            rel, _, rel_prov = m8.mlx_run(rel_run, panel)
            self.assertTrue(prov["reproduces_score_json"])
            self.assertTrue(rel_prov["reproduces_score_json"])
            stored = json.loads((cand_run / "mlx-diag.score.json").read_text())
            self.assertAlmostEqual(acc["type_macro"], stored["type_macro_accuracy"], 12)
            result = m8.mlx_paired_result(cand, rel, draws=300)
            lo, hi = result["bootstrap"]["card_macro_ci95"]
            self.assertLessEqual(lo, result["delta"]["card_macro"])
            self.assertGreaterEqual(hi, result["delta"]["card_macro"])
            self.assertEqual(result["R4"]["pass"], hi >= 0)
            same = m8.mlx_paired_result(cand, cand, draws=50)
            self.assertEqual(same["bootstrap"]["card_macro_ci95"], [0.0, 0.0])
            self.assertTrue(same["R4"]["pass"])

    def test_detects_a_changed_score_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel, (cand_run, _) = mlx_panel(Path(tmp), random.Random(12))
            stored = json.loads((cand_run / "mlx-diag.score.json").read_text())
            stored["by_type"]["noul"]["languages"]["en"]["correct"] += 1
            common.write_json(cand_run / "mlx-diag.score.json", stored)
            self.assertFalse(m8.mlx_run(cand_run, panel)[2]["reproduces_score_json"])

    def test_unbalanced_languages_are_refused(self):
        outcomes = {
            "a": ("choice", "en", 1),
            "b": ("choice", "de", 0),
            "c": ("choice", "de", 1),
        }
        outcomes.update({"n": ("noul", "en", 1), "s": ("score", "en", 0)})
        with self.assertRaisesRegex(ValueError, "not balanced"):
            m8.mlx_accuracies(outcomes)


def formal_rows(rng, row, corrected=True, drift=0.0):
    ref, new = {}, {}
    for i in range(40):
        p5 = random_p(rng, 5)
        p3 = random_p(rng, 3)
        pc = random_p(rng, 3)
        choice = normalized_answer("choice", ["A", "B", "C"], m7.logits_of(pc), 1.0)
        noul = {"type": "noul", "noul": rng.random()}
        answers = {"s5": score_answer(p5), "s3": score_answer(p3, ["0", "1", "2"])}
        answers.update(c=choice, n=noul)
        ref[f"i{i}"] = {
            "id": f"i{i}",
            "model_sha256": m8.MODEL_SHA256,
            "answers": answers,
        }
        fixed = dict(answers)
        if corrected:
            fixed["s5"] = m8.corrected_answer(answers["s5"], row)
        if drift:
            p = dict(fixed["s5"]["probabilities"])
            p["0"] += drift
            p["1"] -= drift
            fixed["s5"] = {**fixed["s5"], "probabilities": p}
        new[f"i{i}"] = {
            "id": f"i{i}",
            "model_sha256": m8.MODEL_SHA256,
            "score_bias_sha256": "b" * 64,
            "answers": fixed,
        }
    return ref, new


class XcheckTest(unittest.TestCase):
    def test_exact_correction_passes(self):
        ref, new = formal_rows(random.Random(1), ROW)
        result = m8.xcheck_rows(ref, new, ROW)
        self.assertTrue(all(result["checks"].values()), result["checks"])
        self.assertEqual(result["by_kind"]["score5"]["slots"], 40)
        self.assertEqual(result["by_kind"]["score_other"]["identical"], 40)
        self.assertEqual(result["by_kind"]["score5"]["max_abs_dp_vs_offline"], 0.0)
        self.assertTrue(all(m8.xcheck_bindings(ref, new, "b" * 64).values()))

    def test_uncorrected_or_drifting_score_fails(self):
        ref, new = formal_rows(random.Random(2), ROW, corrected=False)
        result = m8.xcheck_rows(ref, new, ROW)
        self.assertFalse(result["checks"]["score5_max_abs_dp_le_1e-4"])
        ref, new = formal_rows(random.Random(3), ROW, drift=1e-3)
        self.assertFalse(
            m8.xcheck_rows(ref, new, ROW)["checks"]["score5_max_abs_dp_le_1e-4"]
        )

    def test_changed_choice_fails(self):
        ref, new = formal_rows(random.Random(4), ROW)
        answer = dict(new["i0"]["answers"]["c"])
        answer["probabilities"] = {
            k: v + 1e-9 for k, v in answer["probabilities"].items()
        }
        new["i0"]["answers"] = {**new["i0"]["answers"], "c": answer}
        result = m8.xcheck_rows(ref, new, ROW)
        self.assertFalse(result["checks"]["choice_noul_identical"])
        self.assertEqual(result["by_kind"]["choice"]["same_answer"], 40)


def paired_doc(left, right, v3, low, high, h=(0.0, -0.01, 0.02)):
    return {
        "models": {"left": left, "right": right},
        "point": {"delta": {"score": v3, "H": h[0], "T": 0.0}},
        "ci95": {"low": low, "high": high},
        "axis_ci95": {
            "H": {"delta": {"low": h[1], "high": h[2]}},
            "T": {"delta": {"low": 0.0, "high": 0.0}},
        },
    }


def successor_docs(**over):
    docs = {
        "report": {"v3": {"score": 48.0}},
        "paired": {
            "released": paired_doc("c", "released", 4.9, 2.5, 9.0),
            "kai1": paired_doc("c", "kai1", 3.0, 0.5, 6.0),
            "gliner25": paired_doc("c", "gliner25", 5.0, 1.0, 9.0),
        },
        "types": {
            "types": {
                "choice": {"verdict": "OK"},
                "noul": {"verdict": "OK"},
                "score": {"verdict": "OK"},
            }
        },
        "mlx": {
            "R4": {"rule": "r"},
            "delta": {"card_macro": 0.01, "type_macro": 0.02},
            "bootstrap": {
                "card_macro_ci95": [-0.01, 0.03],
                "type_macro_ci95": [0.0, 0.04],
            },
        },
        "exposure": {
            "payload_sha256": m8.EXCLUDED_GROUPS_SHA256,
            "files": [
                {"path": p, "sha256": s, "rows": 10} for p, s in m8.M6_MIXTURES.items()
            ],
            "groups": [],
            "matched_rows": {},
            "methods_agree": True,
        },
        "public": {
            "delta": -3,
            "left_correct": 100,
            "right_correct": 103,
            "mcnemar_exact_p": 0.4,
            "verdict": "OK",
        },
        "d4": {"D4": {"pass": True}},
        "score": {"score": {"flags": ["NO-GAIN"]}},
    }
    docs.update(over)
    return docs


def run_successor(docs):
    return m8.successor_result(
        docs["report"],
        docs["paired"],
        docs["types"],
        docs["mlx"],
        docs["exposure"],
        docs["public"],
        docs["d4"],
        docs["score"],
    )


class SuccessorTest(unittest.TestCase):
    def test_all_rules_pass(self):
        result = run_successor(successor_docs())
        self.assertTrue(result["verdict"])
        self.assertTrue(result["rules"]["R3"]["score_no_gain_disclosed"])

    def test_each_rule_can_fail(self):
        docs = successor_docs()
        docs["paired"]["released"] = paired_doc("c", "released", 1.0, -0.5, 3.0)
        self.assertFalse(run_successor(docs)["rules"]["R1"]["pass"])
        docs = successor_docs()
        docs["paired"]["released"] = paired_doc(
            "c", "released", 4.9, 2.5, 9.0, h=(-0.05, -0.09, -0.01)
        )
        self.assertFalse(run_successor(docs)["rules"]["R2"]["pass"])
        docs = successor_docs(types={"types": {"score": {"verdict": "COLLAPSED: x"}}})
        result = run_successor(docs)
        self.assertFalse(result["rules"]["R3"]["pass"] or result["rules"]["R5"]["pass"])
        docs = successor_docs()
        docs["mlx"]["bootstrap"]["card_macro_ci95"] = [-0.05, -0.001]
        self.assertFalse(run_successor(docs)["rules"]["R4"]["pass"])
        docs = successor_docs(report={"v3": {"score": 38.2}})
        self.assertFalse(run_successor(docs)["rules"]["R5"]["pass"])
        docs = successor_docs()
        docs["exposure"]["groups"] = ["g1"]
        self.assertFalse(run_successor(docs)["rules"]["R6"]["pass"])
        docs = successor_docs()
        docs["exposure"]["files"] = docs["exposure"]["files"][:5]
        self.assertFalse(run_successor(docs)["rules"]["R6"]["pass"])
        docs = successor_docs()
        docs["public"] = {**docs["public"], "verdict": "REGRESSION"}
        self.assertFalse(run_successor(docs)["rules"]["R7"]["pass"])
        docs = successor_docs(d4={"D4": {"pass": False}})
        self.assertFalse(run_successor(docs)["verdict"])


class ChooseTest(unittest.TestCase):
    def test_first_passer_unless_later_significantly_better(self):
        a = {"label": "m8-a", "verdict": True}
        b = {"label": "m8-b", "verdict": True}
        tie = paired_doc("m8-b", "m8-a", 0.5, -1.0, 2.0)
        self.assertEqual(m8.choose_result([a, b], [tie])["successor"], "m8-a")
        better = paired_doc("m8-b", "m8-a", 2.0, 0.2, 4.0)
        self.assertEqual(m8.choose_result([a, b], [better])["successor"], "m8-b")
        self.assertEqual(
            m8.choose_result([{**a, "verdict": False}, b], [])["successor"], "m8-b"
        )
        self.assertIsNone(
            m8.choose_result([{**a, "verdict": False}, {**b, "verdict": False}], [])[
                "successor"
            ]
        )
        with self.assertRaisesRegex(ValueError, "no paired file"):
            m8.choose_result([a, b], [])


class PackageTest(unittest.TestCase):
    def make_source(self, root: Path, candidate="s5-b05"):
        source = root / "src" / "full"
        export = source / "best-export"
        (export / "backbone").mkdir(parents=True)
        for name, data in {
            "backbone/config.json": b"{}",
            "backbone/model.safetensors": b"weights",
            "decision_config.json": b"{}",
            "decision_head.safetensors": b"head",
            "tokenizer.json": b"{}",
        }.items():
            (export / name).write_bytes(data)
        files = {
            item.relative_to(export).as_posix(): {
                "bytes": item.stat().st_size,
                "sha256": common.file_sha256(item),
            }
            for item in sorted(export.rglob("*"))
            if item.is_file()
        }
        manifest = common.write_json(
            source / "best-export.MANIFEST.json",
            {"schema": "dev2-06b-causal-files/1", "files": files},
        )
        common.write_json(
            source / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "best_export_manifest_sha256": manifest,
                "state_sha256": "s",
            },
        )
        model = checkpoint_fingerprint(export)["model_sha256"]
        bias = root / "score_bias.json"
        bias.write_text(
            json.dumps(
                {
                    "format": SCORE_BIAS_FORMAT,
                    "model_sha256": model,
                    "offsets": {"5": ROW},
                    "fit": {"candidate": candidate},
                }
            )
        )
        return source, bias, manifest, model

    def test_package_layout_matches_the_formal_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bias, manifest, model = self.make_source(root)
            dest = root / "arms" / "m8-s5-b05" / "full"
            record = quiet(
                package.build,
                "s5-b05",
                source,
                bias,
                dest,
                common.file_sha256(bias),
                model_sha256=model,
                source_manifest_sha256=manifest,
            )
            complete = json.loads((dest / "COMPLETE.json").read_text())
            self.assertEqual(
                complete["best_export_manifest_sha256"],
                common.file_sha256(dest / "best-export.MANIFEST.json"),
            )
            self.assertEqual(complete["kind"], "m8-score-bias-package")
            self.assertEqual(record["model_sha256"], model)
            self.assertEqual(
                (dest / "best-export" / "score_bias.json").read_bytes(),
                bias.read_bytes(),
            )
            self.assertEqual(
                checkpoint_fingerprint(dest / "best-export")["model_sha256"], model
            )
            with self.assertRaises(FileExistsError):
                package.build(
                    "s5-b05",
                    source,
                    bias,
                    dest,
                    common.file_sha256(bias),
                    model_sha256=model,
                    source_manifest_sha256=manifest,
                )

    def test_package_refuses_wrong_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, bias, manifest, model = self.make_source(root)
            kw = {"model_sha256": model, "source_manifest_sha256": manifest}
            with self.assertRaisesRegex(ValueError, "sha256"):
                package.build("s5-b05", source, bias, root / "d1", "0" * 64, **kw)
            with self.assertRaisesRegex(ValueError, "another candidate"):
                package.build(
                    "s5h-b05", source, bias, root / "d2", common.file_sha256(bias), **kw
                )
            with self.assertRaisesRegex(ValueError, "frozen soup"):
                package.build(
                    "s5-b05",
                    source,
                    bias,
                    root / "d3",
                    common.file_sha256(bias),
                    model_sha256=model,
                    source_manifest_sha256="1" * 64,
                )
            self.assertFalse(any((root / d).exists() for d in ("d1", "d2", "d3")))


if __name__ == "__main__":
    unittest.main()
