from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lux9b import m8_data, m8_rules
from training.model.infer import prompt_input_sha256


def row(rid, kind, keys, label=0, state="s", instructions="q", descriptions=None):
    descriptions = descriptions or [f"d{k}" for k in keys]
    return {
        "id": rid,
        "task_type": kind,
        "state": state,
        "instructions": instructions,
        "options": [{"key": k, "description": d} for k, d in zip(keys, descriptions)],
        "label": label,
        "input_sha256": f"h-{rid}",
    }


class PromptTest(unittest.TestCase):
    def test_round_trip_keeps_option_order_and_structured_payloads(self):
        rows = [
            row(
                "c", "choice", ["b", "a", "c"], descriptions=["x", {"k": [1, 2]}, None]
            ),
            row("n", "noul", ["true", "false"], state={"z": 1, "a": [2]}),
            row("s", "score", ["0", "1", "2"], instructions={"task": "rate"}),
        ]
        for r in rows:
            line = m8_data.prompt_line(r)
            prompt = json.loads(line)
            self.assertEqual(set(prompt), {"id", "state", "questions"})
            self.assertEqual(list(prompt["questions"]), [m8_data.QID])
            if r["task_type"] != "score":
                self.assertEqual(
                    list(prompt["questions"][m8_data.QID]["criteria"]),
                    [o["key"] for o in r["options"]],
                )

    def test_refuses_score_keys_out_of_order(self):
        with self.assertRaises(ValueError):
            m8_data.prompt_of(row("s", "score", ["1", "0"]))

    def test_render_check_catches_a_changed_option(self):
        r = row("c", "choice", ["a", "b"])
        prompt = m8_data.prompt_of(r)
        prompt["questions"][m8_data.QID]["criteria"]["b"] = "other"
        with self.assertRaises(ValueError):
            m8_data.check_render(r, prompt)


class TeacherProbsTest(unittest.TestCase):
    def test_noul_from_p_true_in_row_key_order(self):
        r = row("n", "noul", ["true", "false"])
        probs = m8_data.teacher_probs(r, {"type": "noul", "noul": 0.25})
        self.assertEqual(list(probs), ["true", "false"])
        self.assertAlmostEqual(probs["false"], 0.75)

    def test_choice_and_score_probabilities(self):
        r = row("s", "score", ["0", "1", "2"])
        probs = m8_data.teacher_probs(
            r,
            {
                "type": "score",
                "score": 1.0,
                "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
            },
        )
        self.assertEqual(probs, {"0": 0.2, "1": 0.6, "2": 0.2})

    def test_refuses_errors_mismatched_keys_and_bad_sums(self):
        r = row("c", "choice", ["a", "b"])
        for answer in (
            {"type": "choice", "error": "invalid_question"},
            {"type": "choice", "probabilities": {"a": 0.5, "c": 0.5}},
            {"type": "choice", "probabilities": {"a": 0.5, "b": 0.6}},
            {"type": "noul", "noul": 0.5},
        ):
            with self.assertRaises(ValueError):
                m8_data.teacher_probs(r, answer)

    def test_manifest_binding(self):
        want = {"model_sha256": "m", "max_length": 32768, "calibration_sha256": "c"}
        ok = {
            "model_sha256": "m",
            "max_length": 32768,
            "calibration": {
                "file_sha256": "c",
                "temperature_by_type": {"choice": 1.0, "noul": 1.0, "score": 1.0},
            },
            "counts": {"invalid_questions": 0, "truncated_questions": 0},
        }
        m8_data.check_manifest(ok, want)
        for patch in (
            {"model_sha256": "x"},
            {"max_length": 16384},
            {
                "calibration": {
                    "file_sha256": "c",
                    "temperature_by_type": {"choice": 0.5, "noul": 1.0, "score": 1.0},
                }
            },
            {"counts": {"invalid_questions": 1, "truncated_questions": 0}},
        ):
            with self.assertRaises(ValueError):
                m8_data.check_manifest({**ok, **patch}, want)

    def test_prompt_hash_is_insertion_ordered(self):
        r = row("c", "choice", ["b", "a"])
        prompt = json.loads(m8_data.prompt_line(r))
        self.assertEqual(
            prompt_input_sha256(prompt),
            prompt_input_sha256(json.loads(json.dumps(prompt))),
        )


def readout_arm(rp, fam_correct=90, choice=10, noul=10, score=10):
    return {
        "by_type": {
            "choice": {"correct": choice, "n": 20},
            "noul": {"correct": noul, "n": 20},
            "score": {"correct": score, "n": 20},
        },
        "by_family": {
            "rule_precedence": {"correct": rp, "n": 400},
            "other": {"correct": fam_correct, "n": 100},
        },
        "H_mean": 0.55,
        "proxy": 70.0,
    }


def pn1(hop, ref="ref-ka13"):
    return {
        "reference": ref,
        "candidate": {
            "hop": {"yes": hop},
            "clean_no": {"yes": 0.3},
            "pawsx6": {"yes": 0.6},
        },
    }


def ht(h):
    return {"htdev2": {"H_dev2": h}}


class EarlyTest(unittest.TestCase):
    def run_early(self, rp=338, hop=0.99, h=0.56, sel=0.86):
        arms = {"arm": readout_arm(rp), "control": readout_arm(338)}
        return m8_rules.early(
            arms, "arm", "control", ht(h), ht(0.56), pn1(hop), pn1(0.99), sel, 0.86
        )

    def test_continue_at_the_boundaries(self):
        out = self.run_early(rp=334, hop=0.96, h=0.5401, sel=0.84)
        self.assertTrue(out["continue"], out["reasons"])

    def test_each_guard_stops(self):
        for kwargs in ({"rp": 333}, {"hop": 0.959}, {"h": 0.54}, {"sel": 0.839}):
            out = self.run_early(**kwargs)
            self.assertFalse(out["continue"], kwargs)
            self.assertEqual(len(out["reasons"]), 1, out["reasons"])

    def test_references_must_match(self):
        arms = {"arm": readout_arm(338), "control": readout_arm(338)}
        with self.assertRaises(ValueError):
            m8_rules.early(
                arms,
                "arm",
                "control",
                ht(0.5),
                ht(0.5),
                pn1(0.9),
                pn1(0.9, "other"),
                0.8,
                0.8,
            )


def provenance(arm="C-m1", teacher="t0", rows=12443, **contract):
    base = {
        "arm": arm,
        "seed": 20260931,
        "teacher_kl_weight": 1.0,
        "teacher_partial": True,
        "teacher_rows": rows,
        "data_sha256": {"train": "eb55", "select": "s", "cal": "c", "teacher": teacher},
        "backbone_lr": 5e-6,
    }
    base.update(contract)
    return {
        "contract": base,
        "code_sha256": "code",
        "source_commit": "7168f086",
        "source_tree": "tree",
        "image_id": "img",
        "precision": "bf16",
        "model_source": {"source_name": "checkpoint-0001624"},
        "train_examples": 12443,
        "train_tokens": 1,
        "train_max_tokens": 2,
        "train_type_counts": {},
        "select_examples": 700,
        "trainable_parameters": 3,
    }


class RecipeTest(unittest.TestCase):
    def test_only_arm_teacher_and_rows_may_differ(self):
        out = m8_rules.recipe(provenance(), provenance("D2-m1", "t1", 3000), 3000)
        self.assertEqual(out["status"], "PASS", out["differences"])

    def test_other_differences_fail(self):
        for arm in (
            provenance("D1-m1", "t1", seed=1),
            provenance("D1-m1", "t1", backbone_lr=1e-5),
            provenance("D1-m1", "t1", teacher_kl_weight=0.5),
            {**provenance("D1-m1", "t1"), "source_commit": "other"},
        ):
            self.assertEqual(
                m8_rules.recipe(provenance(), arm, 12443)["status"], "FAIL"
            )
        wrong_rows = m8_rules.recipe(
            provenance(), provenance("D1-m1", "t1", 100), 12443
        )
        self.assertEqual(wrong_rows["status"], "FAIL")

    def test_cli_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, a = Path(tmp) / "c.json", Path(tmp) / "a.json"
            c.write_text(json.dumps(provenance()))
            a.write_text(json.dumps(provenance("D1-m1", "t1", seed=5)))
            self.assertEqual(
                m8_rules.main(
                    [
                        "recipe",
                        "--control",
                        str(c),
                        "--arm",
                        str(a),
                        "--teacher-rows",
                        "12443",
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
