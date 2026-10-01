"""CPU-only checks for the IX1 parity gate, shard assignment, result merge and CAL fits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.ix1 import merge, panel, parity


def _ok(run_id: str, p: float, noul: float) -> dict:
    return {
        "run_id": run_id,
        "status": "ok",
        "response": {
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": p, "b": 1 - p},
                },
                "n": {"type": "noul", "noul": noul},
            }
        },
    }


def _ref(record: dict) -> dict:
    out = {k: v for k, v in record.items() if k != "response"}
    if record["status"] == "ok":
        out["answers"] = json.loads(json.dumps(record["response"]["answers"]))
    return out


class ParityTests(unittest.TestCase):
    def test_identical_answers_pass_and_drift_is_measured(self) -> None:
        kit = {
            "r1": _ok("r1", 0.7, 0.2),
            "r2": {"run_id": "r2", "status": "unsupported"},
        }
        ref = {key: _ref(value) for key, value in kit.items()}
        ref["r1"]["answers"]["n"]["noul"] = 0.20004
        report = parity.compare(kit, ref)
        self.assertTrue(report["pass"])
        self.assertAlmostEqual(report["max_abs_dp"], 4e-5)
        self.assertEqual(report["statuses"], {"ok": 1, "unsupported": 1})

    def test_choice_status_drift_and_errors_fail(self) -> None:
        kit = {"r1": _ok("r1", 0.7, 0.2)}
        for change in ("choice", "status", "drift", "error", "missing"):
            ref = {"r1": _ref(kit["r1"])}
            if change == "choice":
                ref["r1"]["answers"]["q"]["choice"] = "b"
            elif change == "status":
                ref["r1"] = {"run_id": "r1", "status": "unsupported"}
            elif change == "drift":
                ref["r1"]["answers"]["q"]["probabilities"] = {"a": 0.7002, "b": 0.2998}
            elif change == "error":
                kit_error = {"r1": {"run_id": "r1", "status": "error"}}
                ref = {"r1": {"run_id": "r1", "status": "error"}}
                self.assertFalse(parity.compare(kit_error, ref)["pass"])
                continue
            else:
                ref = {}
            with self.subTest(change=change):
                self.assertFalse(parity.compare(kit, ref)["pass"])


class ShardAndMergeTests(unittest.TestCase):
    def test_shard_assignment_is_stable_and_in_range(self) -> None:
        ids = [f"{n}:{i}" for n in (25, 57) for i in range(200)]
        first = [panel.shard_of(r, 8) for r in ids]
        self.assertEqual(first, [panel.shard_of(r, 8) for r in ids])
        self.assertEqual(set(first), set(range(8)))
        self.assertEqual(
            panel.gold_free(
                {"_evaluation": {}, "state": "s", "questions": {}, "expected": 1}
            ),
            {"_evaluation": {}, "state": "s", "questions": {}},
        )

    def test_retried_errors_are_superseded_but_double_answers_fail(self) -> None:
        lines = [
            json.dumps({"run_id": "a", "status": "error"}),
            json.dumps({"run_id": "a", "status": "ok"}),
            json.dumps({"run_id": "b", "status": "unsupported"}),
        ]
        final, superseded = merge.final_records(lines)
        self.assertEqual(final["a"]["status"], "ok")
        self.assertEqual(superseded, 1)
        with self.assertRaisesRegex(ValueError, "answered twice"):
            merge.final_records(lines + [json.dumps({"run_id": "b", "status": "ok"})])

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(merge._percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5), 3.0)
        self.assertAlmostEqual(merge._percentile([0.0, 10.0], 0.95), 9.5)


class CalibrationTests(unittest.TestCase):
    def test_noul_labels_follow_the_option_keys(self) -> None:
        from v2.eval.ix1 import calib

        orders = {"ft": ["false", "true"], "tf": ["true", "false"]}
        with tempfile.TemporaryDirectory() as tmp:
            fits = {}
            for name, keys in orders.items():
                labels, refs = {}, []
                for i, (p, gold) in enumerate([(0.9, "true"), (0.2, "false")] * 20):
                    run_id = f"{name}{i}"
                    labels[run_id] = {
                        "type": "noul",
                        "label": keys.index(gold),
                        "keys": keys,
                    }
                    answer = {"type": "noul", "noul": p}
                    refs.append(
                        {"run_id": run_id, "status": "ok", "answers": {"q": answer}}
                    )
                labels_path = Path(tmp, f"{name}-labels.json")
                ref_path = Path(tmp, f"{name}-ref.jsonl")
                labels_path.write_text(json.dumps(labels))
                ref_path.write_text("".join(json.dumps(r) + "\n" for r in refs))
                fits[name] = calib.fit(labels_path, ref_path)
        self.assertAlmostEqual(
            fits["ft"]["temperatures"]["noul"], fits["tf"]["temperatures"]["noul"]
        )
        self.assertLess(fits["tf"]["temperatures"]["noul"], 1.0)
        self.assertEqual(fits["ft"]["noul_tb"], fits["tf"]["noul_tb"])


if __name__ == "__main__":
    unittest.main()
