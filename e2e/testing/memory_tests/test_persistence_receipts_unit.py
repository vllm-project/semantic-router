"""Offline regressions for the request-correlated persistence E2E assertions.

These run without a stack. They exist so the E2E assertions themselves are
tested: a scenario that would pass on the wrong receipt, or that would leave
the stack mutated, fails here instead of silently weakening coverage.
"""

import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

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


def metrics_response(**status_counts):
    lines = [
        'llm_plugin_execution_total{plugin_type="memory_persistence",'
        f'decision_name="{receipts.RECEIPT_DECISION}",status="{status}"}} {count}'
        for status, count in status_counts.items()
    ]
    return Mock(status_code=receipts.HTTP_OK, text="\n".join(lines))


def hash_response(status, runtime_hash="hash-2"):
    response = Mock(status_code=receipts.HTTP_OK)
    response.json.return_value = {
        "activation_status": status,
        "active_runtime_hash": runtime_hash if status == "active" else "hash-1",
        "generated_runtime_hash": runtime_hash,
    }
    return response


def config_response(etag='"config-1"'):
    return Mock(status_code=receipts.HTTP_OK, headers={"ETag": etag})


def receipt_log(request_id, status="cancelled", reason="shutdown"):
    """A router receipt log line. request_id is Envoy's, not the test's."""
    return json.dumps(
        {
            "component": "extproc",
            "event": receipts.RECEIPT_LOG_EVENT,
            "request_id": request_id,
            "status": status,
            "reason": reason,
            "fail_open": True,
        }
    )


class PersistenceReceiptAssertionsTest(unittest.TestCase):
    def setUp(self):
        self.case = receipts.MemoryPersistenceReceiptTest()
        self.case.replay_url = "http://router/api/v1/observability/replays"
        self.case.metrics_url = "http://router/metrics"
        self.case.config_url = "http://router/api/v1/config"

    # ---- endpoint resolution ------------------------------------------------

    def test_default_replay_endpoint_matches_management_api_contract(self):
        with (
            patch.object(receipts.MemoryFeaturesTest, "setUp"),
            patch.object(self.case, "_resolve_metrics_url"),
            patch.dict(receipts.os.environ, {}, clear=True),
        ):
            self.case.setUp()

        repository_root = Path(__file__).resolve().parents[3]
        contract_path = (
            repository_root / "website/static/openapi/apiserver/apiserver.openapi.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        record_path = urlsplit(self.case.replay_url).path + "/{id}"
        self.assertIn(record_path, contract["paths"])
        self.assertIn("get", contract["paths"][record_path])

    def test_config_endpoints_match_management_api_contract(self):
        with (
            patch.object(receipts.MemoryFeaturesTest, "setUp"),
            patch.object(self.case, "_resolve_metrics_url"),
            patch.dict(receipts.os.environ, {}, clear=True),
        ):
            self.case.setUp()

        repository_root = Path(__file__).resolve().parents[3]
        contract_path = (
            repository_root / "website/static/openapi/apiserver/apiserver.openapi.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        config_path = urlsplit(self.case.config_url).path
        self.assertIn("patch", contract["paths"][config_path])
        self.assertIn("get", contract["paths"][config_path + "/hash"])
        self.assertTrue(
            any(
                parameter.get("name") == "If-Match"
                and parameter.get("in") == "header"
                and parameter.get("required")
                for parameter in contract["paths"][config_path]["patch"]["parameters"]
            )
        )

    def test_receipt_endpoints_follow_stack_port_offset(self):
        for offset in (0, 200, 4200):
            with (
                self.subTest(offset=offset),
                patch.object(receipts.MemoryFeaturesTest, "setUp"),
                patch.dict(
                    receipts.os.environ,
                    {"VLLM_SR_PORT_OFFSET": str(offset)},
                    clear=True,
                ),
                patch.object(
                    receipts.requests, "get", return_value=Mock(status_code=200)
                ) as get,
            ):
                self.case.setUp()
                self.assertEqual(
                    self.case.metrics_url, f"http://localhost:{9190 + offset}/metrics"
                )
                self.assertEqual(
                    self.case.replay_url,
                    f"http://localhost:{8080 + offset}/api/v1/observability/replays",
                )
                self.assertEqual(
                    self.case.config_url,
                    f"http://localhost:{8080 + offset}/api/v1/config",
                )
                get.assert_called_once_with(self.case.metrics_url, timeout=5)

    def test_explicit_receipt_endpoints_override_stack_defaults(self):
        metrics_url = "http://custom-router:7000/metrics"
        replay_url = "http://custom-router:8000/api/v1/observability/replays"
        with (
            patch.object(receipts.MemoryFeaturesTest, "setUp"),
            patch.dict(
                receipts.os.environ,
                {
                    "VLLM_SR_PORT_OFFSET": "4200",
                    "ROUTER_METRICS_URL": metrics_url,
                    "ROUTER_REPLAY_URL": replay_url + "/",
                },
                clear=True,
            ),
            patch.object(
                receipts.requests, "get", return_value=Mock(status_code=200)
            ) as get,
        ):
            self.case.setUp()
        self.assertEqual(self.case.metrics_url, metrics_url)
        self.assertEqual(self.case.replay_url, replay_url)
        get.assert_called_once_with(metrics_url, timeout=5)

    def test_fault_and_router_endpoints_follow_runtime_stack_export(self):
        with (
            patch.object(receipts.MemoryFeaturesTest, "setUp"),
            patch.object(self.case, "_resolve_metrics_url"),
            patch.dict(
                receipts.os.environ,
                {
                    "MEMORY_FAULT_CONTROL_URL": "http://127.0.0.1:19000/",
                    "ROUTER_CONTAINER": "ci-memory-vllm-sr-router",
                },
                clear=True,
            ),
        ):
            self.case.setUp()

        self.assertEqual(self.case.fault_url, "http://127.0.0.1:19000")
        self.assertEqual(self.case.router_container, "ci-memory-vllm-sr-router")

    def test_unavailable_stack_metrics_do_not_probe_another_stack(self):
        for failure in (
            Mock(status_code=503),
            receipts.requests.exceptions.ConnectionError("unreachable"),
        ):
            with (
                self.subTest(failure=failure),
                patch.object(receipts.MemoryFeaturesTest, "setUp"),
                patch.dict(
                    receipts.os.environ, {"VLLM_SR_PORT_OFFSET": "4200"}, clear=True
                ),
                patch.object(
                    receipts.requests,
                    "get",
                    side_effect=[failure, Mock(status_code=200)],
                ) as get,
            ):
                with self.assertRaises(AssertionError):
                    self.case.setUp()
                get.assert_called_once_with("http://localhost:13390/metrics", timeout=5)

    # ---- replay receipts ----------------------------------------------------

    def test_terminal_receipt_is_read_from_the_response_replay_id(self):
        terminal = outcome("completed", reason="persisted")
        pending = Mock(status_code=receipts.HTTP_OK)
        pending.json.return_value = {"id": "request-replay", "outcomes": []}
        ready = Mock(status_code=receipts.HTTP_OK)
        ready.json.return_value = {"id": "request-replay", "outcomes": [terminal]}
        with (
            patch.object(receipts.requests, "get", side_effect=[pending, ready]),
            patch.object(receipts.time, "sleep"),
        ):
            actual = self.case._wait_for_terminal_receipt(
                {"_replay_id": "request-replay"}, scheduled=False
            )
        self.assertEqual(actual, terminal)

    def test_wrong_replay_id_is_rejected(self):
        response = Mock(status_code=receipts.HTTP_OK)
        response.json.return_value = {"id": "another-request", "outcomes": []}
        with (
            patch.object(receipts.requests, "get", return_value=response),
            self.assertRaises(AssertionError),
        ):
            self.case._wait_for_terminal_receipt({"_replay_id": "request-replay"})

    def test_disabled_receipt_can_arrive_after_response_without_scheduled(self):
        terminal = outcome("disabled", reason="auto_store_off")
        response = Mock(status_code=receipts.HTTP_OK)
        response.json.return_value = {"id": "request-replay", "outcomes": [terminal]}
        with patch.object(receipts.requests, "get", return_value=response):
            self.assertEqual(
                self.case._wait_for_terminal_receipt(
                    {"_replay_id": "request-replay"}, scheduled=False
                ),
                terminal,
            )
            with self.assertRaises(AssertionError):
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

    def test_scheduled_wait_requires_the_attempt_to_hold_capacity(self):
        scheduled = Mock(status_code=receipts.HTTP_OK)
        scheduled.json.return_value = {
            "id": "request-replay",
            "outcomes": [outcome("scheduled", "scheduled")],
        }
        with patch.object(receipts.requests, "get", return_value=scheduled):
            self.case._wait_for_scheduled_receipt({"_replay_id": "request-replay"})

        empty = Mock(status_code=receipts.HTTP_OK)
        empty.json.return_value = {"id": "request-replay", "outcomes": []}
        with (
            patch.object(receipts.requests, "get", return_value=empty),
            patch.object(receipts.time, "monotonic", side_effect=[0, 0, 31]),
            patch.object(receipts.time, "sleep"),
            self.assertRaises(AssertionError),
        ):
            self.case._wait_for_scheduled_receipt({"_replay_id": "request-replay"})

    def test_oversized_history_e2e_requires_only_the_skipped_terminal(self):
        self.case.responses_url = "http://router/v1/responses"
        self.case.test_user = "test-user"
        self.case.timeout = 5
        model_response = Mock(
            status_code=receipts.HTTP_OK,
            headers={"x-vsr-replay-id": "oversized-replay"},
        )
        model_response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Model output"}],
                }
            ]
        }
        terminal = outcome("skipped", reason="history_too_large")
        for scheduled in (False, True):
            with self.subTest(scheduled=scheduled):
                replay_response = Mock(status_code=receipts.HTTP_OK)
                replay_response.json.return_value = {
                    "id": "oversized-replay",
                    "outcomes": (
                        [outcome("scheduled", "scheduled")] if scheduled else []
                    )
                    + [terminal],
                }
                expected = (
                    self.assertRaises(AssertionError) if scheduled else nullcontext()
                )
                with (
                    patch.object(
                        receipts.requests, "post", return_value=model_response
                    ),
                    patch.object(
                        receipts.requests, "get", return_value=replay_response
                    ),
                    expected,
                ):
                    self.case.test_04_oversized_history_skips_persistence_and_delivers_response()

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

    # ---- receipt metrics ----------------------------------------------------

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
            self.assertEqual(self.case._receipt_totals(), {"completed": 1.0})

    def test_failed_scrape_cannot_be_interpreted_as_zero(self):
        response = Mock(status_code=503, text="unavailable")
        with (
            patch.object(receipts.requests, "get", return_value=response),
            self.assertRaises(AssertionError),
        ):
            self.case._receipt_count("store_failed")

    def test_terminal_deltas_report_only_what_this_scenario_caused(self):
        baseline = {"completed": 7.0, "timeout": 2.0}
        with patch.object(
            receipts.requests,
            "get",
            return_value=metrics_response(completed=7, timeout=3, cancelled=1),
        ):
            self.assertEqual(
                self.case._terminal_deltas(baseline), {"timeout": 1.0, "cancelled": 1.0}
            )

    def test_a_counter_reset_is_not_read_as_progress(self):
        with (
            patch.object(
                receipts.requests, "get", return_value=metrics_response(completed=1)
            ),
            self.assertRaises(AssertionError),
        ):
            self.case._terminal_deltas({"completed": 5.0})

    def test_extra_terminal_receipts_fail_the_single_receipt_contract(self):
        """Two terminals for one attempt must fail, not be waited past."""
        with (
            patch.object(
                receipts.requests,
                "get",
                return_value=metrics_response(completed=1, timeout=1),
            ),
            patch.object(receipts.time, "sleep"),
            self.assertRaisesRegex(AssertionError, "exactly 1 terminal"),
        ):
            self.case._wait_for_terminal_deltas({}, 1)

    def test_terminal_deltas_wait_for_a_full_burst(self):
        pending = metrics_response(cancelled=1)
        settled = metrics_response(cancelled=3, completed=1)
        with (
            patch.object(receipts.requests, "get", side_effect=[pending, settled]),
            patch.object(receipts.time, "sleep"),
        ):
            self.assertEqual(
                self.case._wait_for_terminal_deltas({}, 4),
                {"cancelled": 3.0, "completed": 1.0},
            )

    # ---- runtime configuration ---------------------------------------------

    def test_persistence_bounds_are_merged_and_awaited(self):
        patched = Mock(status_code=receipts.HTTP_OK)
        patched.json.return_value = {"generated_runtime_hash": "hash-2"}
        with (
            patch.object(receipts.requests, "patch", return_value=patched) as sent,
            patch.object(
                receipts.requests,
                "get",
                side_effect=[
                    config_response(),
                    hash_response("pending"),
                    hash_response("active"),
                ],
            ) as get,
            patch.object(receipts.time, "sleep"),
        ):
            self.case._apply_persistence_bounds(concurrency=1, timeout_seconds=60)

        self.assertEqual(get.call_args_list[0].args, (self.case.config_url,))
        self.assertEqual(sent.call_args.kwargs["headers"], {"If-Match": '"config-1"'})
        # Unnamed bounds are still declared, as 0 (the documented default), so a
        # scenario never inherits another scenario's tuning.
        self.assertEqual(
            sent.call_args.kwargs["json"]["yaml"],
            "global:\n"
            "  stores:\n"
            "    memory:\n"
            "      persistence:\n"
            "        concurrency: 1\n"
            "        queue: 0\n"
            "        shutdown_grace_seconds: 0\n"
            "        timeout_seconds: 60\n",
        )

    def test_a_rejected_or_failed_activation_is_not_silently_ignored(self):
        rejected = Mock(status_code=409, text="RESTART_REQUIRED")
        with (
            patch.object(receipts.requests, "get", return_value=config_response()),
            patch.object(receipts.requests, "patch", return_value=rejected),
            self.assertRaises(AssertionError),
        ):
            self.case._apply_persistence_bounds(concurrency=1)

        accepted = Mock(status_code=receipts.HTTP_OK)
        accepted.json.return_value = {"generated_runtime_hash": "hash-2"}
        with (
            patch.object(receipts.requests, "patch", return_value=accepted),
            patch.object(
                receipts.requests,
                "get",
                side_effect=[config_response(), hash_response("failed")],
            ),
            self.assertRaises(AssertionError),
        ):
            self.case._apply_persistence_bounds(concurrency=1)

    def test_an_activation_of_someone_elses_document_is_not_accepted(self):
        accepted = Mock(status_code=receipts.HTTP_OK)
        accepted.json.return_value = {"generated_runtime_hash": "hash-9"}
        with (
            patch.object(receipts.requests, "patch", return_value=accepted),
            patch.object(
                receipts.requests,
                "get",
                side_effect=[config_response(), hash_response("active")],
            ),
            patch.object(receipts.time, "monotonic", side_effect=[0, 0, 999]),
            patch.object(receipts.time, "sleep"),
            self.assertRaises(AssertionError),
        ):
            self.case._apply_persistence_bounds(concurrency=1)

    def test_accepted_config_update_waits_for_matching_active_hash(self):
        patched = Mock(status_code=202)
        patched.json.return_value = {"generated_runtime_hash": "hash-2"}
        with (
            patch.object(receipts.requests, "patch", return_value=patched),
            patch.object(
                receipts.requests,
                "get",
                side_effect=[
                    config_response(),
                    hash_response("pending"),
                    hash_response("active", "other-hash"),
                    hash_response("active"),
                ],
            ) as get,
            patch.object(receipts.time, "sleep"),
        ):
            self.case._apply_persistence_bounds(concurrency=1)
        self.assertEqual(get.call_count, 4)

    def test_config_update_without_hash_cannot_accept_old_generation(self):
        patched = Mock(status_code=202)
        patched.json.return_value = {"activation_status": "pending"}
        with (
            patch.object(receipts.requests, "get", return_value=config_response()),
            patch.object(receipts.requests, "patch", return_value=patched),
            patch.object(self.case, "_wait_for_activation") as wait,
            self.assertRaisesRegex(AssertionError, "omitted its runtime hash"),
        ):
            self.case._apply_persistence_bounds(concurrency=1)
        wait.assert_not_called()

    def test_config_conflict_refetches_etag_before_retrying(self):
        accepted = Mock(status_code=202)
        accepted.json.return_value = {"generated_runtime_hash": "hash-2"}
        with (
            patch.object(
                receipts.requests,
                "get",
                side_effect=[config_response(), config_response('"config-2"')],
            ) as get,
            patch.object(
                receipts.requests,
                "patch",
                side_effect=[Mock(status_code=412, text="CONFIG_CHANGED"), accepted],
            ) as sent,
            patch.object(self.case, "_wait_for_activation") as wait,
        ):
            self.case._apply_persistence_bounds(concurrency=1)
        self.assertEqual(
            [call.args for call in get.call_args_list], [(self.case.config_url,)] * 2
        )
        self.assertEqual(
            [call.kwargs["headers"] for call in sent.call_args_list],
            [{"If-Match": '"config-1"'}, {"If-Match": '"config-2"'}],
        )
        wait.assert_called_once_with("hash-2")

    def test_repeated_config_conflicts_fail_after_bounded_retries(self):
        with (
            patch.object(
                receipts.requests, "get", return_value=config_response()
            ) as get,
            patch.object(
                receipts.requests,
                "patch",
                return_value=Mock(status_code=412, text="CONFIG_CHANGED"),
            ) as sent,
            patch.object(self.case, "_wait_for_activation") as wait,
            self.assertRaisesRegex(AssertionError, "CONFIG_CHANGED"),
        ):
            self.case._apply_persistence_bounds(concurrency=1)
        self.assertEqual(get.call_count, receipts.CONFIG_UPDATE_ATTEMPTS)
        self.assertEqual(sent.call_count, receipts.CONFIG_UPDATE_ATTEMPTS)
        wait.assert_not_called()

    def test_config_read_failure_or_missing_etag_prevents_mutation(self):
        for response in (
            config_response(None),
            Mock(status_code=403, text="forbidden"),
        ):
            with (
                self.subTest(response=response),
                patch.object(receipts.requests, "get", return_value=response),
                patch.object(receipts.requests, "patch") as sent,
                self.assertRaises(AssertionError),
            ):
                self.case._apply_persistence_bounds(concurrency=1)
            sent.assert_not_called()

    def test_write_gate_is_released_even_when_an_assertion_fails(self):
        self.case.fault_url = "http://localhost:19000"
        with (
            patch.object(
                receipts.requests, "post", return_value=Mock(status_code=200)
            ) as post,
            self.assertRaisesRegex(AssertionError, "receipt failed"),
            self.case._write_fault("hold"),
        ):
            raise AssertionError("receipt failed")
        self.assertEqual(
            [c.kwargs["json"] for c in post.call_args_list],
            [{"mode": "hold"}, {"mode": "pass"}],
        )

    def test_missing_fault_fixture_fails_instead_of_skipping_required_coverage(self):
        self.case.fault_url = ""
        with (
            self.assertRaisesRegex(AssertionError, "MEMORY_FAULT_CONTROL_URL"),
            self.case._write_fault("hold"),
        ):
            self.fail("missing fixture was accepted")

    def test_every_response_is_observed_while_the_write_remains_blocked(self):
        self.case.test_user = "test-user"
        self.case.send_memory_request = Mock(return_value={"_output_text": "output"})
        self.case._wait_for_fault = Mock()
        results = self.case._saturate_persistence(4, "held")
        self.assertEqual(len(results), 4)
        self.assertEqual(self.case._wait_for_fault.call_count, 4)
        predicate = self.case._wait_for_fault.call_args.args[0]
        self.assertTrue(predicate({"active": 1}))
        self.assertFalse(predicate({"active": 0}))
        self.case.send_memory_request.return_value = {"_output_text": ""}
        with self.assertRaises(AssertionError):
            self.case._saturate_persistence(1, "held")

    def test_pending_attempts_resolve_real_request_ids_and_reject_early_completion(
        self,
    ):
        results = [{"_replay_id": f"replay-{i}"} for i in range(4)]
        records = [
            {
                "id": r["_replay_id"],
                "request_id": f"envoy-{i}",
                "outcomes": [outcome("scheduled", "scheduled")],
            }
            for i, r in enumerate(results)
        ]
        self.case._wait_for_scheduled_receipt = Mock()
        with patch.object(self.case, "_replay_record", side_effect=records):
            self.assertEqual(
                self.case._assert_attempts_pending(results),
                {f"envoy-{i}" for i in range(4)},
            )
        # An earlier request completing must fail even if the last is pending.
        records[0]["outcomes"].append(outcome("completed", reason="persisted"))
        with (
            patch.object(self.case, "_replay_record", side_effect=records),
            self.assertRaises(AssertionError),
        ):
            self.case._assert_attempts_pending(results)

    def test_missing_or_reused_request_identity_cannot_cover_multiple_attempts(self):
        self.case._wait_for_scheduled_receipt = Mock()
        for identity in (None, "same-id"):
            with self.subTest(identity=identity):
                record = {
                    "request_id": identity,
                    "outcomes": [outcome("scheduled", "scheduled")],
                }
                with (
                    patch.object(self.case, "_replay_record", return_value=record),
                    self.assertRaises(AssertionError),
                ):
                    self.case._assert_attempts_pending(
                        [{"_replay_id": "a"}, {"_replay_id": "b"}]
                    )

    def test_success_cannot_pass_on_another_requests_completed_counter(self):
        self.case.print_test_header = Mock()
        self.case._receipt_count = Mock(return_value=41)
        self.case.send_memory_request = Mock(return_value={"_replay_id": "own-request"})
        self.case._wait_for_receipt = Mock(return_value=42)
        self.case._wait_for_terminal_receipt = Mock(
            return_value=outcome("timeout", reason="persist_timeout")
        )
        with self.assertRaises(AssertionError):
            self.case.test_01_successful_store_reports_completed_receipt()
        self.case._wait_for_receipt.assert_not_called()

    def test_failure_scenarios_reject_the_wrong_terminal_status(self):
        for scenario, expected_status, reason in (
            (
                "test_05_store_failure_reports_persist_error_and_stays_fail_open",
                "store_failed",
                "persist_error",
            ),
            (
                "test_06_exhausted_budget_reports_persist_timeout",
                "timeout",
                "persist_timeout",
            ),
        ):
            for status in ("store_failed", "timeout", "completed"):
                with self.subTest(scenario=scenario, status=status):
                    case = receipts.MemoryPersistenceReceiptTest()
                    case.test_user = "test-user"
                    result = {"_replay_id": "r", "_output_text": "output"}
                    case._apply_persistence_bounds = Mock()
                    case._receipt_totals = Mock(return_value={})
                    case._write_fault = Mock(side_effect=lambda *_: nullcontext())
                    case._wait_for_fault = Mock()
                    case._saturate_persistence = Mock(return_value=[result])
                    case._assert_attempts_pending = Mock()
                    case.send_memory_request = Mock(return_value=result)
                    case._wait_for_terminal_receipt = Mock(
                        return_value=outcome(status, reason=reason)
                    )
                    case._wait_for_terminal_deltas = Mock(return_value={status: 1})
                    case._terminal_deltas = Mock(return_value={status: 1})
                    expected = (
                        nullcontext()
                        if status == expected_status
                        else self.assertRaises(AssertionError)
                    )
                    with (
                        expected,
                        patch.dict(
                            receipts.os.environ,
                            {"USE_DETERMINISTIC_MEMORY_EMBEDDINGS": "1"},
                        ),
                    ):
                        getattr(case, scenario)()
                    case._write_fault.assert_called_once_with(
                        "fail" if expected_status == "store_failed" else "hold"
                    )

    def test_reload_checks_every_request_id_and_only_cancelled_metrics(self):
        ids = {f"envoy-{i}" for i in range(4)}
        for logs in (
            [json.loads(receipt_log(rid)) for rid in ids],
            [json.loads(receipt_log("envoy-0"))],
        ):
            case = receipts.MemoryPersistenceReceiptTest()
            case._apply_persistence_bounds = Mock()
            case._receipt_totals = Mock(return_value={})
            case._write_fault = Mock(side_effect=lambda *_: nullcontext())
            case._saturate_persistence = Mock(return_value=[{}] * 4)
            case._assert_attempts_pending = Mock(return_value=ids)
            case._wait_for_fault = Mock()
            case._router_receipt_logs = Mock(return_value=logs)
            case._wait_for_terminal_deltas = Mock(return_value={"cancelled": 4})
            case._terminal_deltas = Mock(return_value={"cancelled": 4})
            with patch.object(receipts.time, "monotonic", side_effect=[0, 0, 31]):
                expected = (
                    nullcontext()
                    if len(logs) == len(ids)
                    else self.assertRaises(AssertionError)
                )
                with expected, patch.object(receipts.time, "sleep"):
                    case.test_07_reload_cancels_queued_writes_with_shutdown_reason()


class ShutdownReceiptAssertionsTest(unittest.TestCase):
    def setUp(self):
        self.case = receipts.MemoryPersistenceShutdownTest()
        self.ids = {f"envoy-{i}" for i in range(4)}
        self.logs = [json.loads(receipt_log(rid)) for rid in sorted(self.ids)]

    def test_requires_one_cancelled_receipt_for_every_request(self):
        self.case._assert_cancelled_logs(self.logs, self.ids)
        for invalid in ([], self.logs[:1], self.logs[:3], self.logs + self.logs[:1]):
            with self.subTest(count=len(invalid)), self.assertRaises(AssertionError):
                self.case._assert_cancelled_logs(invalid, self.ids)

    def test_cancellation_requires_exact_status_reason_and_fail_open(self):
        for field, value in (
            ("status", "completed"),
            ("reason", "persist_timeout"),
            ("fail_open", False),
            ("fail_open", "true"),
        ):
            invalid = [dict(self.logs[0], **{field: value}), *self.logs[1:]]
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(AssertionError),
            ):
                self.case._assert_cancelled_logs(invalid, self.ids)

    def test_foreign_receipt_cannot_replace_a_missing_request(self):
        foreign = json.loads(receipt_log("unrelated-request"))
        self.case._assert_cancelled_logs([foreign, *self.logs], self.ids)
        with self.assertRaises(AssertionError):
            self.case._assert_cancelled_logs([foreign, *self.logs[1:]], self.ids)

    def test_shutdown_scenario_uses_all_request_ids_even_with_fast_embeddings(self):
        case = self.case
        case._apply_persistence_bounds = Mock()
        case._write_fault = Mock(side_effect=lambda *_: nullcontext())
        case._saturate_persistence = Mock(return_value=[{}] * 4)
        case._assert_attempts_pending = Mock(return_value=self.ids)
        case._wait_for_fault = Mock()
        case._stop_router = Mock()
        case._router_receipt_logs = Mock(return_value=self.logs)
        with patch.dict(receipts.os.environ, {"VLLM_SR_DETERMINISTIC_EMBEDDINGS": "1"}):
            case.test_01_shutdown_cancels_queued_writes_and_logs_one_receipt_each()
        case._stop_router.assert_called_once()
        case._router_receipt_logs.return_value = self.logs[:1]
        with self.assertRaisesRegex(AssertionError, "missing terminal receipts"):
            case.test_01_shutdown_cancels_queued_writes_and_logs_one_receipt_each()

    def test_forced_kill_does_not_count_as_graceful_shutdown(self):
        self.case.container_runtime, self.case.router_container = (
            "docker",
            "test-router",
        )
        for code in (0, 143, 137):
            with self.subTest(code=code):
                inspect = Mock(stdout=json.dumps({"Running": False, "ExitCode": code}))
                with (
                    patch.object(
                        receipts.shutil, "which", return_value="/usr/bin/docker"
                    ),
                    patch.object(
                        receipts.subprocess, "run", side_effect=[Mock(), inspect]
                    ),
                ):
                    expected = (
                        nullcontext()
                        if code in (0, 143)
                        else self.assertRaises(AssertionError)
                    )
                    with expected:
                        self.case._stop_router()

    def test_stop_requires_owned_container_and_runtime_before_invoking_commands(self):
        self.case.container_runtime = "docker"
        for container, executable, message in (
            ("", "/usr/bin/docker", "ROUTER_CONTAINER"),
            ("test-router", None, "container runtime unavailable"),
        ):
            self.case.router_container = container
            with (
                self.subTest(container=container),
                patch.object(receipts.shutil, "which", return_value=executable),
                patch.object(receipts.subprocess, "run") as run,
                self.assertRaisesRegex(AssertionError, message),
            ):
                self.case._stop_router()
            run.assert_not_called()

    def test_non_json_and_foreign_events_are_skipped(self):
        self.case.container_runtime, self.case.router_container = (
            "docker",
            "test-router",
        )
        lines = [
            "startup banner",
            json.dumps({"event": "foreign"}),
            receipt_log("envoy-0"),
        ]
        with (
            patch.object(receipts.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(
                receipts.subprocess,
                "run",
                return_value=Mock(stdout="\n".join(lines), stderr=""),
            ),
        ):
            self.assertEqual(self.case._router_receipt_logs(), self.logs[:1])


if __name__ == "__main__":
    unittest.main()
