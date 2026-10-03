"""CPU-only checks for the IX1 complement rows and the board-submission merge and public files."""

from __future__ import annotations

import json
import unittest

from v2.eval.ix1 import complement, submission


def _row(run_id: str, catalog: int) -> dict:
    return {
        "_evaluation": {"run_id": run_id, "catalog_id": catalog},
        "state": "s",
        "questions": {},
    }


def _result(run_id: str, status: str = "ok", **extra) -> dict:
    record = {
        "run_id": run_id,
        "status": status,
        "engine": "e",
        "payload_sha256": "ab" * 32,
    }
    record.update(extra)
    return record


class ComplementTests(unittest.TestCase):
    def test_missing_rows_are_the_scoreable_rows_without_a_result(self) -> None:
        rows = [_row("a", 1), _row("b", 24), _row("c", 6)]
        missing, expected = complement.complement(rows, {"a"})
        self.assertEqual([r["_evaluation"]["run_id"] for r in missing], ["b", "c"])
        self.assertEqual(expected, {"a", "b", "c"})

    def test_stored_ids_outside_the_edition_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            complement.complement([_row("a", 1)], {"a", "z"})

    def test_stored_duplicates_and_errors_are_refused(self) -> None:
        line = json.dumps({"run_id": "a", "status": "ok"})
        with self.assertRaises(ValueError):
            complement.stored_run_ids([line, line])
        with self.assertRaises(ValueError):
            complement.stored_run_ids([json.dumps({"run_id": "a", "status": "error"})])
        self.assertEqual(complement.stored_run_ids([line, ""]), {"a"})


class CombineTests(unittest.TestCase):
    expected = [("a", 1), ("b", 24), ("c", 24)]

    def test_suite_order_and_accounting(self) -> None:
        rows, acc = submission.combine(
            self.expected,
            {
                "a": _result("a"),
                "c": _result("c", "unsupported", error="max_length_exceeded"),
            },
            {"b": _result("b")},
        )
        self.assertEqual([r["run_id"] for r in rows], ["a", "b", "c"])
        self.assertEqual(acc["statuses"], {"ok": 2, "unsupported": 1})
        self.assertEqual(
            acc["per_benchmark_statuses"],
            {"1": {"ok": 1}, "24": {"ok": 1, "unsupported": 1}},
        )
        self.assertEqual(acc["non_ok_reasons"], {"max_length_exceeded": 1})

    def test_overlap_gap_extra_errors_and_mixed_engines_fail(self) -> None:
        cases = [
            (
                {"a": _result("a"), "b": _result("b")},
                {"b": _result("b"), "c": _result("c")},
            ),
            ({"a": _result("a")}, {"b": _result("b")}),
            (
                {"a": _result("a"), "c": _result("c")},
                {"b": _result("b"), "z": _result("z")},
            ),
            ({"a": _result("a"), "c": _result("c")}, {"b": _result("b", "error")}),
            (
                {"a": _result("a"), "c": _result("c")},
                {"b": {**_result("b"), "engine": "other"}},
            ),
        ]
        for stored, added in cases:
            with self.assertRaises(ValueError):
                submission.combine(self.expected, stored, added)


class PublicTests(unittest.TestCase):
    def test_rows_lose_payload_and_raw_output_only(self) -> None:
        record = _result(
            "a", payload={"state": "x"}, raw_output={"y": 1}, response={"answers": {}}
        )
        out = submission.public_row(record)
        self.assertNotIn("payload", out)
        self.assertNotIn("raw_output", out)
        self.assertEqual(out["payload_sha256"], "ab" * 32)
        self.assertIn("response", out)
        with self.assertRaises(ValueError):
            submission.public_row({"run_id": "a", "status": "ok"})

    def test_environment_keeps_file_names_and_paths_are_detected(self) -> None:
        env = {
            "rows_path": "/data/x/panel-3/shard-0-of-3.jsonl.gz",
            "engine_options": {"device": "cuda:0"},
        }
        self.assertEqual(
            submission.public_environment(env)["rows_path"], "shard-0-of-3.jsonl.gz"
        )
        self.assertEqual(
            submission.public_environment({"rows_path": ["/data/a.gz"]})["rows_path"],
            ["a.gz"],
        )
        self.assertEqual(
            submission.private_strings({"a": ["ok", {"b": "/root/x"}]}), ["/root/x"]
        )
        self.assertEqual(
            submission.private_strings(submission.public_environment(env)), []
        )

    def test_index_view_ignores_benchmarks_outside_the_index(self) -> None:
        base = {
            "index": 1.0,
            "raw_index": 2.0,
            "scores": {},
            "areas": [],
            "coverage": 1.0,
        }
        a = {
            **base,
            "benchmarks": {
                "1": {"raw": 0.5, "in_index": True},
                "24": {"raw": 0.0, "in_index": False},
            },
        }
        b = {
            **base,
            "benchmarks": {
                "1": {"raw": 0.5, "in_index": True},
                "24": {"raw": 0.7, "in_index": False},
            },
        }
        self.assertEqual(submission.index_view(a), submission.index_view(b))
        b["benchmarks"]["1"]["raw"] = 0.6
        self.assertNotEqual(submission.index_view(a), submission.index_view(b))

    def test_spot_sample_is_stratified_and_deterministic(self) -> None:
        rows = [_row(f"a{i}", 1) for i in range(5)] + [_row("b0", 24)]
        first = submission.spot_sample(rows, 2)
        self.assertEqual(len(first), 3)
        self.assertEqual(
            sorted(r["_evaluation"]["catalog_id"] for r in first), [1, 1, 24]
        )
        self.assertEqual(first, submission.spot_sample(list(reversed(rows)), 2))

    def test_spot_compare_counts_choice_status_and_drift(self) -> None:
        def ok(p: float, choice: str = "a") -> dict:
            return {
                "run_id": "r",
                "catalog_id": 1,
                "status": "ok",
                "response": {
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": choice,
                            "probabilities": {"a": p, "b": 1 - p},
                        },
                        "n": {"type": "noul", "noul": 0.25},
                    }
                },
            }

        same = submission.spot_compare({"r": ok(0.7)}, {"r": ok(0.7000004)})
        self.assertEqual(same["status_or_choice_mismatches"], 0)
        self.assertAlmostEqual(same["max_abs_dp"], 4e-7)
        flip = submission.spot_compare({"r": ok(0.7)}, {"r": ok(0.3, "b")})
        self.assertEqual(flip["status_or_choice_mismatches"], 1)
        refused = {"run_id": "r", "catalog_id": 1, "status": "unsupported"}
        self.assertEqual(
            submission.spot_compare({"r": ok(0.7)}, {"r": refused})[
                "status_or_choice_mismatches"
            ],
            1,
        )
        self.assertEqual(
            submission.spot_compare({"r": refused}, {"r": refused})[
                "status_or_choice_mismatches"
            ],
            0,
        )
        with self.assertRaises(ValueError):
            submission.spot_compare({}, {"r": ok(0.7)})

    def test_clean_drops_volatile_keys(self) -> None:
        self.assertEqual(
            submission.clean({"generated_utc": 1, "x": [{"out": 2, "y": 3}]}),
            {"x": [{"y": 3}]},
        )


if __name__ == "__main__":
    unittest.main()
