"""Offline integration coverage for the PR2 executor."""

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path

from .executor import DeterministicProvider, LooperTTSExecutor, ProviderResponse
from .plan import build_plan
from .records import validate_records
from .validation import load_json


EXAMPLE = Path(__file__).parent / "testdata" / "synthetic.json"


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_json(EXAMPLE)
        self.plan = build_plan(self.config, "executor-fixture", ["execute"])

    def test_executes_all_four_algorithms_and_writes_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = DeterministicProvider()
            records = LooperTTSExecutor(
                self.plan, provider, Path(directory), max_workers=2
            ).run()
            validate_records(records, self.plan)
            self.assertEqual(len(records["results"]), 8)
            self.assertEqual(
                {result["status"] for result in records["results"]}, {"success"}
            )
            stages = {call["stage"] for call in records["calls"]}
            self.assertEqual(stages, {"generate", "verify", "synthesize", "judge"})
            self.assertTrue((Path(directory) / "records.json").exists())
            self.assertTrue((Path(directory) / "runtime_receipt.json").exists())

    def test_parallel_evidence_order_is_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_records = LooperTTSExecutor(
                self.plan, DeterministicProvider(), Path(first), max_workers=2
            ).run()
            second_records = LooperTTSExecutor(
                self.plan, DeterministicProvider(), Path(second), max_workers=2
            ).run()
            self.assertEqual(first_records, second_records)

    def test_budget_exhaustion_is_a_paired_result(self):
        config = copy.deepcopy(self.config)
        config["budgets"][0].update(max_calls=1, max_total_tokens=1)
        config["budgets"][1].update(max_calls=2, max_total_tokens=2)
        plan = build_plan(config, "executor-fixture", ["execute"])
        with tempfile.TemporaryDirectory() as directory:
            records = LooperTTSExecutor(
                plan, DeterministicProvider(), Path(directory)
            ).run()
            self.assertEqual(
                {result["status"] for result in records["results"]},
                {"budget_exhausted"},
            )
            self.assertTrue(
                all(not result["call_ids"] for result in records["results"])
            )

    def test_unknown_usage_is_visible_in_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            records = LooperTTSExecutor(
                self.plan,
                DeterministicProvider(omit_usage=True),
                Path(directory),
            ).run()
            self.assertTrue(records["calls"])
            self.assertTrue(
                any(
                    event["usage_source"] == "reservation"
                    for event in _receipt_events(
                        Path(directory) / "runtime_receipt.json"
                    )
                )
            )
            receipt = json.loads(
                (Path(directory) / "runtime_receipt.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(receipt["totals"]["cost_usd"])

    def test_retry_is_recorded_after_later_call_slots_are_allocated(self):
        class FlakyProvider(DeterministicProvider):
            def __init__(self):
                super().__init__()
                self.failed = set()
                self.failure_lock = threading.Lock()

            def chat(self, **kwargs):
                key = (kwargs["stage"], kwargs["model"])
                with self.failure_lock:
                    first = key not in self.failed
                    self.failed.add(key)
                if first:
                    return ProviderResponse(error="transient fixture failure")
                return super().chat(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            records = LooperTTSExecutor(
                self.plan,
                FlakyProvider(),
                Path(directory),
                retries=1,
            ).run()
            direct_calls = [
                call
                for call in records["calls"]
                if call["stage"] == "generate"
                and call["cell_id"]
                == next(
                    cell["id"]
                    for cell in self.plan["matrix"]
                    if cell["algorithm"] == "direct"
                )
            ]
            self.assertEqual([call["attempt"] for call in direct_calls[:2]], [1, 2])
            self.assertEqual(direct_calls[0]["status"], "error")
            self.assertEqual(direct_calls[1]["status"], "success")


def _receipt_events(path):
    receipt = json.loads(path.read_text(encoding="utf-8"))
    return [event for cell in receipt["cells"] for event in cell["calls"]]


if __name__ == "__main__":
    unittest.main()
