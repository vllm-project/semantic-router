"""Offline integration coverage for the PR2 executor."""

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from .executor import (
    DeterministicProvider,
    LooperTTSExecutor,
    ProviderResponse,
    execute_manifest,
)
from .plan import build_plan
from .records import validate_records
from .validation import ContractError, load_json, write_json

EXAMPLE = Path(__file__).parent / "testdata" / "synthetic.json"


class ExecutorTests(unittest.TestCase):
    def test_fake_rejects_benchmark_before_dispatch_or_output(self):
        config = load_json(EXAMPLE)
        config["dataset"]["evidence_kind"] = "benchmark"
        plan = build_plan(config, "review-regression", ["execute"])
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            output = Path(directory) / "run"
            write_json(manifest, plan)
            with patch.object(DeterministicProvider, "chat") as chat:
                with self.assertRaisesRegex(ContractError, "synthetic"):
                    execute_manifest(manifest, output, fake=True)
                with self.assertRaisesRegex(ContractError, "synthetic"):
                    LooperTTSExecutor(plan, DeterministicProvider(), output)
                chat.assert_not_called()
            self.assertFalse(output.exists())

    def test_blank_answers_preserve_paid_calls_and_terminal_records(self):
        class BlankProvider(DeterministicProvider):
            def chat(self, **kwargs):
                response = super().chat(**kwargs)
                response.content = " \t\n"
                response.reasoning = "\n  "
                return response

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            records = LooperTTSExecutor(self.plan, BlankProvider(), output).run()
            validate_records(records, self.plan)
            self.assertEqual({r["status"] for r in records["results"]}, {"error"})
            self.assertTrue(all(r["final_answer"] is None for r in records["results"]))
            self.assertTrue(all(r["call_ids"] for r in records["results"]))
            self.assertEqual(load_json(output / "records.json"), records)
            receipt = load_json(output / "runtime_receipt.json")
            self.assertEqual(receipt["totals"]["calls"], len(records["calls"]))
            self.assertGreater(receipt["totals"]["tokens"], 0)
            for call in records["calls"]:
                self.assertTrue((output / call["raw_output_path"]).is_file())

    def test_blank_content_can_fall_back_to_nonblank_reasoning(self):
        class ReasoningProvider(DeterministicProvider):
            def chat(self, **kwargs):
                response = super().chat(**kwargs)
                response.content = "  "
                response.reasoning = "usable reasoning answer"
                return response

        with tempfile.TemporaryDirectory() as directory:
            records = LooperTTSExecutor(
                self.plan, ReasoningProvider(), Path(directory)
            ).run()
            direct_cells = {
                c["id"] for c in self.plan["matrix"] if c["algorithm"] == "direct"
            }
            for result in records["results"]:
                if result["cell_id"] in direct_cells:
                    self.assertEqual(result["status"], "success")
                    self.assertEqual(result["final_answer"], "usable reasoning answer")

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
