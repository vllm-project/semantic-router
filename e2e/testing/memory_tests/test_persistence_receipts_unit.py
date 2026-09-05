"""Offline regressions for the request-correlated persistence E2E assertions."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from memory_tests import test_persistence_receipts as receipts


def outcome(status, phase="terminal", reason="persist_error"):
    return {
        "verdict": status,
        "reason": reason,
        "metadata": {
            "kind": "memory_persistence_receipt",
            "phase": phase,
            "fail_open": "true",
        },
    }


class PersistenceReceiptAssertionsTest(unittest.TestCase):
    def setUp(self):
        self.case = receipts.MemoryPersistenceReceiptTest()
        self.case.replay_url = "http://router/v1/router_replay"
        self.case.metrics_url = "http://router/metrics"

    def test_terminal_receipt_is_read_from_the_response_replay_id(self):
        terminal = outcome("timeout", reason="persist_timeout")
        response = Mock(status_code=receipts.HTTP_OK)
        response.json.return_value = {
            "id": "request-replay",
            "outcomes": [outcome("scheduled", "scheduled"), terminal],
        }
        with patch.object(receipts.requests, "get", return_value=response) as get:
            actual = self.case._wait_for_terminal_receipt(
                {"_replay_id": "request-replay"}
            )
        self.assertEqual(actual, terminal)
        get.assert_called_once_with(
            "http://router/v1/router_replay/request-replay", timeout=10
        )

    def test_wrong_replay_id_is_rejected(self):
        response = Mock(status_code=receipts.HTTP_OK)
        response.json.return_value = {"id": "another-request", "outcomes": []}
        with (
            patch.object(receipts.requests, "get", return_value=response),
            self.assertRaises(AssertionError),
        ):
            self.case._wait_for_terminal_receipt({"_replay_id": "request-replay"})

    def test_missing_or_duplicate_terminal_receipts_do_not_pass(self):
        with self.assertRaises(AssertionError):
            self.case._wait_for_terminal_receipt({})
        response = Mock(status_code=receipts.HTTP_OK)
        response.json.return_value = {
            "id": "request-replay",
            "outcomes": [outcome("store_failed"), outcome("timeout")],
        }
        with (
            patch.object(receipts.requests, "get", return_value=response),
            self.assertRaises(AssertionError),
        ):
            self.case._wait_for_terminal_receipt({"_replay_id": "request-replay"})

    def test_missing_terminal_receipt_times_out_with_request_diagnostics(self):
        response = Mock(status_code=receipts.HTTP_OK)
        response.json.return_value = {
            "id": "request-replay",
            "outcomes": [outcome("scheduled", "scheduled")],
        }
        with (
            patch.object(receipts.requests, "get", return_value=response),
            patch.object(receipts.time, "monotonic", side_effect=[0, 0, 61]),
            patch.object(receipts.time, "sleep"),
            self.assertRaisesRegex(AssertionError, "request-replay.*scheduled"),
        ):
            self.case._wait_for_terminal_receipt({"_replay_id": "request-replay"})

    def test_metrics_ignore_other_decisions(self):
        response = Mock(status_code=receipts.HTTP_OK)
        response.text = "\n".join(
            [
                'llm_plugin_execution_total{plugin_type="memory_persistence",decision_name="default_route",status="completed"} 42',
                'llm_plugin_execution_total{plugin_type="memory_persistence",decision_name="persistence_receipt_route",status="completed"} 1',
            ]
        )
        with patch.object(receipts.requests, "get", return_value=response):
            self.assertEqual(self.case._receipt_count("completed"), 1)

    def test_failed_scrape_cannot_be_interpreted_as_zero(self):
        response = Mock(status_code=503, text="unavailable")
        with (
            patch.object(receipts.requests, "get", return_value=response),
            self.assertRaises(AssertionError),
        ):
            self.case._receipt_count("store_failed")

    def test_success_cannot_pass_on_another_requests_completed_counter(self):
        case = receipts.MemoryPersistenceReceiptTest()
        case.print_test_header = Mock()
        case._receipt_count = Mock(return_value=41)
        case.send_memory_request = Mock(return_value={"_replay_id": "own-request"})
        case._wait_for_receipt = Mock(return_value=42)
        case._wait_for_terminal_receipt = Mock(
            return_value=outcome("timeout", reason="persist_timeout")
        )
        with self.assertRaises(AssertionError):
            case.test_01_successful_store_reports_completed_receipt()
        case._wait_for_receipt.assert_not_called()

    def test_outage_accepts_only_failure_receipts_and_restores_backend(self):
        for status in ["store_failed", "timeout", "completed", "skipped", "rejected"]:
            with self.subTest(status=status):
                case = receipts.MemoryPersistenceReceiptTest()
                case.test_user, case.storage_wait = "test-user", 0
                case.print_test_header = Mock()
                case.print_test_result = Mock()
                case._container_available = Mock(return_value=True)
                case._container_command = Mock()
                case._receipt_count = Mock(return_value=0)
                case._wait_for_receipt = Mock(return_value=1)
                case.send_memory_request = Mock(
                    return_value={"_output_text": "model output"}
                )
                case._wait_for_terminal_receipt = Mock(
                    return_value=outcome(
                        status,
                        reason=receipts.FAILURE_RECEIPTS.get(status, "unexpected"),
                    )
                )
                expected = (
                    nullcontext()
                    if status in receipts.FAILURE_RECEIPTS
                    else self.assertRaises(AssertionError)
                )
                with patch.object(receipts.time, "sleep"), expected:
                    case.test_02_store_failure_keeps_response_fail_open()
                self.assertEqual(
                    [call.args[0] for call in case._container_command.call_args_list],
                    ["stop", "start"],
                )


if __name__ == "__main__":
    unittest.main()
