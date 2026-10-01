"""Response-side memory persistence receipt tests.

Every enabled attempt owes exactly one terminal receipt. A test-owned Milvus
proxy holds or rejects writes while reads continue, making the failure modes
independent of embedding speed. Replay IDs correlate live receipts; the same
records supply request IDs for checking every receipt after process shutdown.
"""

import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager

import requests

from memory_tests.base import HTTP_OK, MemoryFeaturesTest

RECEIPT_METRIC = "llm_plugin_execution_total"
RECEIPT_PLUGIN = "memory_persistence"
RECEIPT_DECISION = "persistence_receipt_route"
RECEIPT_MARKER = "PERSISTENCE_RECEIPT_MARKER"
RECEIPT_LOG_EVENT = "memory_persistence_outcome"
LABEL_PATTERN = re.compile(r'(\w+)="([^"]*)"')

# Every status memoryPersistenceReceipt.record treats as terminal. "scheduled"
# is deliberately absent: it is a phase, not an outcome, and is not counted in
# the plugin execution metric.
TERMINAL_STATUSES = (
    "completed",
    "skipped",
    "disabled",
    "policy_blocked",
    "extraction_failed",
    "store_failed",
    "timeout",
    "cancelled",
    "rejected",
)

STORE_FAILURE_TIMEOUT_SECONDS = 60
STORE_FAILURE_RECEIPT_BUDGET_SECONDS = 90
# Allow embedding to finish and then keep the write gate closed until expiry.
PERSIST_TIMEOUT_SECONDS = 30
PERSIST_TIMEOUT_RECEIPT_BUDGET_SECONDS = 60
# Deadlines must outlive model preparation during reload. The gate, rather
# than CPU cost, prevents all four attempts from completing during the grace.
RETIREMENT_TIMEOUT_SECONDS = 600
RETIREMENT_GRACE_SECONDS = 1
RETIREMENT_BURST = 4
# Reloads republish a whole generation; model preparation dominates the wait.
ACTIVATION_TIMEOUT_SECONDS = 180
CONFIG_UPDATE_ATTEMPTS = 3
HTTP_PRECONDITION_FAILED = 412


class PersistenceReceiptSupport(MemoryFeaturesTest):
    """Endpoints, config control, and receipt assertions shared by the scenarios."""

    def setUp(self):
        super().setUp()
        self.port_offset = int(os.environ.get("VLLM_SR_PORT_OFFSET", "0"))
        self.metrics_url = self._resolve_metrics_url()
        management_url = f"http://localhost:{8080 + self.port_offset}/api/v1"
        self.replay_url = os.environ.get(
            "ROUTER_REPLAY_URL", f"{management_url}/observability/replays"
        ).rstrip("/")
        self.config_url = f"{management_url}/config"
        self.container_runtime = os.environ.get("CONTAINER_RUNTIME", "docker")
        self.router_container = os.environ.get("ROUTER_CONTAINER", "")
        self.fault_url = os.environ.get("MEMORY_FAULT_CONTROL_URL", "").rstrip("/")

    def _resolve_metrics_url(self) -> str:
        url = os.environ.get(
            "ROUTER_METRICS_URL",
            f"http://localhost:{9190 + self.port_offset}/metrics",
        )
        try:
            response = requests.get(url, timeout=5)
        except requests.exceptions.RequestException as e:
            self.fail(f"router metrics endpoint {url} is not reachable: {e}")
        self.assertEqual(
            response.status_code, HTTP_OK, f"router metrics endpoint {url} failed"
        )
        return url

    # ---- receipt metrics ---------------------------------------------------

    def _receipt_count(self, status: str) -> float:
        return self._receipt_totals().get(status, 0.0)

    def _receipt_totals(self) -> dict:
        """Count this decision's persistence receipts, per terminal status."""
        try:
            response = requests.get(self.metrics_url, timeout=10)
        except requests.exceptions.RequestException as e:
            self.fail(f"metrics scrape failed: {e}")

        self.assertEqual(response.status_code, HTTP_OK, "metrics scrape failed")
        totals = {}
        prefix = RECEIPT_METRIC + "{"
        for line in response.text.splitlines():
            if not line.startswith(prefix):
                continue
            labels_part, _, value = line.rpartition(" ")
            labels = dict(LABEL_PATTERN.findall(labels_part))
            if (
                labels.get("plugin_type") == RECEIPT_PLUGIN
                and labels.get("decision_name") == RECEIPT_DECISION
            ):
                status = labels.get("status", "")
                totals[status] = totals.get(status, 0.0) + float(value)
        return totals

    def _terminal_deltas(self, baseline: dict) -> dict:
        """Terminal receipts recorded since baseline, per status, dropping zeros."""
        current = self._receipt_totals()
        deltas = {}
        for status in TERMINAL_STATUSES:
            delta = current.get(status, 0.0) - baseline.get(status, 0.0)
            self.assertGreaterEqual(delta, 0, f"{status} counter went backwards")
            if delta:
                deltas[status] = delta
        return deltas

    def _wait_for_terminal_deltas(self, baseline: dict, expected: float, timeout=60):
        """Wait until exactly `expected` terminal receipts land, then report them.

        Attempts owe one terminal receipt each, so the totals must settle on
        exactly `expected`. Polling past that point would also catch a second
        receipt for an attempt that already reported one.
        """
        deadline = time.monotonic() + timeout
        deltas = {}
        while time.monotonic() < deadline:
            deltas = self._terminal_deltas(baseline)
            if sum(deltas.values()) >= expected:
                break
            time.sleep(1)
        self.assertEqual(
            sum(deltas.values()),
            expected,
            f"expected exactly {expected} terminal receipts, observed {deltas}",
        )
        return deltas

    def _wait_for_receipt(
        self, status: str, baseline: float, timeout: int = 60
    ) -> float:
        deadline = time.time() + timeout
        latest = baseline
        while time.time() < deadline:
            latest = self._receipt_count(status)
            if latest > baseline:
                return latest
            time.sleep(2)
        return latest

    # ---- receipt replay records --------------------------------------------

    def _replay_record(self, replay_id: str) -> dict:
        response = requests.get(f"{self.replay_url}/{replay_id}", timeout=10)
        self.assertEqual(response.status_code, HTTP_OK, response.text)
        record = response.json()
        self.assertEqual(record.get("id"), replay_id)
        return record

    def _replay_outcomes(self, replay_id: str) -> list:
        record = self._replay_record(replay_id)
        return [
            outcome
            for outcome in record.get("outcomes", [])
            if outcome.get("metadata", {}).get("kind") == "memory_persistence_receipt"
        ]

    def _wait_for_terminal_receipt(
        self, result: dict, timeout: int = 60, scheduled: bool = True
    ) -> dict:
        replay_id = result.get("_replay_id")
        self.assertTrue(replay_id, "response is missing x-vsr-replay-id")
        deadline = time.monotonic() + timeout
        outcomes = []
        while time.monotonic() < deadline:
            outcomes = self._replay_outcomes(replay_id)
            terminal = [
                outcome
                for outcome in outcomes
                if outcome.get("metadata", {}).get("phase") == "terminal"
            ]
            if terminal:
                self.assertEqual(len(terminal), 1, outcomes)
                self.assertEqual(
                    [outcome.get("verdict") for outcome in outcomes],
                    (["scheduled"] if scheduled else []) + [terminal[0]["verdict"]],
                    outcomes,
                )
                return terminal[0]
            time.sleep(0.2)
        self.fail(f"No terminal persistence receipt for replay {replay_id}: {outcomes}")

    def _wait_for_scheduled_receipt(self, result: dict, timeout: int = 30) -> None:
        """Block until the attempt holds runner capacity, so it can be disrupted."""
        replay_id = result.get("_replay_id")
        self.assertTrue(replay_id, "response is missing x-vsr-replay-id")
        deadline = time.monotonic() + timeout
        outcomes = []
        while time.monotonic() < deadline:
            outcomes = self._replay_outcomes(replay_id)
            if any(outcome.get("verdict") == "scheduled" for outcome in outcomes):
                return
            time.sleep(0.2)
        self.fail(f"Attempt {replay_id} was never scheduled: {outcomes}")

    def _assert_terminal(self, receipt: dict, status: str, reason: str, fail_open: str):
        self.assertEqual(receipt["verdict"], status, receipt)
        self.assertEqual(receipt["reason"], reason, receipt)
        self.assertEqual(receipt["metadata"].get("fail_open"), fail_open, receipt)

    # ---- runtime configuration ---------------------------------------------

    def _apply_persistence_bounds(self, **overrides) -> None:
        """Declare every persistence bound, then await the new generation.

        PATCH /api/v1/config deep-merges into the persisted document, which the
        router watcher republishes as a whole generation. Retiring the previous
        generation drains its persistence runner, so a reload is also the only
        way to observe cancellation without ending the process.

        Each scenario states all four bounds, including the ones it does not
        care about (0 selects the documented default). No scenario then depends
        on another having cleaned up, and none needs a second reload to undo
        itself — reloads are the expensive part of these tests.
        """
        bounds = {
            "timeout_seconds": 0,
            "concurrency": 0,
            "queue": 0,
            "shutdown_grace_seconds": 0,
        } | overrides
        document = "global:\n  stores:\n    memory:\n      persistence:\n" + "".join(
            f"        {key}: {value}\n" for key, value in sorted(bounds.items())
        )
        for _ in range(CONFIG_UPDATE_ATTEMPTS):
            current = requests.get(self.config_url, timeout=10)
            self.assertEqual(current.status_code, HTTP_OK, current.text)
            etag = current.headers.get("ETag")
            self.assertTrue(etag, "config read omitted its ETag")
            response = requests.patch(
                self.config_url,
                json={"yaml": document},
                headers={"If-Match": etag},
                timeout=30,
            )
            if response.status_code != HTTP_PRECONDITION_FAILED:
                break
        self.assertIn(response.status_code, (HTTP_OK, 202), response.text)
        runtime_hash = response.json().get("generated_runtime_hash")
        self.assertTrue(runtime_hash, "config update omitted its runtime hash")
        self._wait_for_activation(runtime_hash)

    def _wait_for_activation(self, runtime_hash: str) -> None:
        """A persisted document is not proof of an activated generation."""
        deadline = time.monotonic() + ACTIVATION_TIMEOUT_SECONDS
        payload = {}
        while time.monotonic() < deadline:
            response = requests.get(f"{self.config_url}/hash", timeout=10)
            self.assertEqual(response.status_code, HTTP_OK, response.text)
            payload = response.json()
            self.assertNotEqual(payload.get("activation_status"), "failed", payload)
            if (
                payload.get("activation_status") == "active"
                and payload.get("active_runtime_hash") == runtime_hash
            ):
                return
            time.sleep(1)
        self.fail(f"config activation did not complete: {payload}")

    # ---- backend control ----------------------------------------------------

    @contextmanager
    def _write_fault(self, mode):
        self.assertTrue(
            self.fault_url, "harness did not export MEMORY_FAULT_CONTROL_URL"
        )
        response = requests.post(
            f"{self.fault_url}/mode", json={"mode": mode}, timeout=5
        )
        self.assertEqual(response.status_code, HTTP_OK, response.text)
        try:
            yield
        finally:
            response = requests.post(
                f"{self.fault_url}/mode", json={"mode": "pass"}, timeout=5
            )
            self.assertEqual(response.status_code, HTTP_OK, response.text)

    def _wait_for_fault(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        state = {}
        while time.monotonic() < deadline:
            response = requests.get(f"{self.fault_url}/state", timeout=5)
            self.assertEqual(response.status_code, HTTP_OK, response.text)
            state = response.json()
            if predicate(state):
                return state
            time.sleep(0.05)
        self.fail(f"write fault did not reach its expected state: {state}")

    def _saturate_persistence(self, count: int, user_suffix: str) -> list:
        """Collect responses while the first write remains blocked in the proxy."""
        results = []
        for index in range(count):
            result = self.send_memory_request(
                message=f"{RECEIPT_MARKER} Retire note {index} for {user_suffix}.",
                auto_store=True,
                user_id=f"{self.test_user}_{user_suffix}",
                verbose=False,
            )
            self.assertIsNotNone(result, f"attempt {index} returned no response")
            self.assertTrue(
                result.get("_output_text"),
                "a disrupted memory write must not withhold model output",
            )
            results.append(result)
            self._wait_for_fault(lambda state: state["active"] == 1)
        return results

    def _assert_attempts_pending(self, results: list) -> set:
        """All responses must arrive before any of their writes completes."""
        request_ids = set()
        replay_ids = set()
        for result in results:
            self._wait_for_scheduled_receipt(result)
            replay_id = result["_replay_id"]
            self.assertNotIn(replay_id, replay_ids)
            replay_ids.add(replay_id)
            record = self._replay_record(replay_id)
            request_id = record.get("request_id")
            self.assertTrue(request_id, record)
            self.assertNotIn(request_id, request_ids)
            request_ids.add(request_id)
            receipts = [
                entry
                for entry in record.get("outcomes", [])
                if entry.get("metadata", {}).get("kind") == "memory_persistence_receipt"
            ]
            self.assertEqual(
                [entry["verdict"] for entry in receipts], ["scheduled"], record
            )
        return request_ids

    def _router_receipt_logs(self) -> list:
        self.assertTrue(
            self.router_container, "harness did not export ROUTER_CONTAINER"
        )
        self.assertTrue(
            shutil.which(self.container_runtime), "container runtime unavailable"
        )
        result = subprocess.run(
            [self.container_runtime, "logs", self.router_container],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        records = []
        for line in (result.stdout + result.stderr).splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(record, dict)
                and record.get("event") == RECEIPT_LOG_EVENT
                and record.get("component") == "extproc"
            ):
                records.append(record)
        return records

    def _assert_cancelled_logs(self, records, request_ids):
        # All these requests were pending behind a closed gate. Every one must
        # therefore cancel; an absent record cannot stand for silent success.
        selected = [r for r in records if r.get("request_id") in request_ids]
        self.assertEqual(
            {r["request_id"] for r in selected},
            request_ids,
            f"missing terminal receipts: {selected}",
        )
        self.assertEqual(
            len(selected), len(request_ids), f"duplicate terminal receipt: {selected}"
        )
        for record in selected:
            self.assertEqual(record.get("status"), "cancelled", record)
            self.assertEqual(record.get("reason"), "shutdown", record)
            self.assertIs(record.get("fail_open"), True, record)

    def _wait_for_cancelled_logs(self, request_ids):
        deadline = time.monotonic() + 30
        records = []
        while time.monotonic() < deadline:
            records = self._router_receipt_logs()
            if request_ids <= {r.get("request_id") for r in records}:
                break
            time.sleep(0.1)
        self._assert_cancelled_logs(records, request_ids)

    def _stop_router(self):
        self.assertTrue(
            self.router_container, "harness did not export ROUTER_CONTAINER"
        )
        self.assertTrue(
            shutil.which(self.container_runtime), "container runtime unavailable"
        )
        subprocess.run(
            [self.container_runtime, "stop", "--time", "30", self.router_container],
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
        result = subprocess.run(
            [
                self.container_runtime,
                "inspect",
                "--format",
                "{{json .State}}",
                self.router_container,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        state = json.loads(result.stdout)
        self.assertFalse(state["Running"], state)
        self.assertIn(
            state["ExitCode"],
            (0, 143),
            f"router did not stop gracefully (137 means SIGKILL): {state}",
        )


class MemoryPersistenceReceiptTest(PersistenceReceiptSupport):
    """Verify each terminal persistence receipt and that writes stay fail-open."""

    def test_01_successful_store_reports_completed_receipt(self):
        """An auto-stored turn reaches a completed persistence receipt."""
        self.print_test_header(
            "Persistence Receipt: Completed",
            "Auto-stored turn increments the memory_persistence completed counter",
        )

        baseline = self._receipt_count("completed")
        result = self.send_memory_request(
            message=f"{RECEIPT_MARKER} My preferred deployment region is eu-central-1.",
            auto_store=True,
        )
        self.assertIsNotNone(result, "auto-store request did not return a response")

        receipt = self._wait_for_terminal_receipt(result)
        self._assert_terminal(receipt, "completed", "persisted", "false")
        observed = self._wait_for_receipt("completed", baseline)
        self.assertGreater(
            observed,
            baseline,
            "auto-stored turn produced no completed persistence receipt",
        )
        self.print_test_result(True, f"completed receipts {baseline} -> {observed}")

    def test_03_disabled_store_reports_terminal_receipt_after_response(self):
        """An explicit opt-out still delivers output and an asynchronous receipt."""
        result = self.send_memory_request(
            message=f"{RECEIPT_MARKER} Explain how to deploy a regional service.",
            auto_store=False,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.get("_output_text"), "response must carry model output")
        receipt = self._wait_for_terminal_receipt(result, scheduled=False)
        self._assert_terminal(receipt, "disabled", "auto_store_off", "false")

    def test_04_oversized_history_skips_persistence_and_delivers_response(self):
        """The 256-message snapshot bound sheds persistence without truncating history."""
        response = requests.post(
            self.responses_url,
            json={
                "model": "MoM",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": f"{RECEIPT_MARKER} preference {i}",
                    }
                    for i in range(257)
                ],
                "auto_store": True,
            },
            headers={"x-authz-user-id": self.test_user},
            timeout=self.timeout,
        )
        self.assertEqual(response.status_code, HTTP_OK, response.text)
        result = response.json()
        self.assertTrue(self._extract_output_text(result))
        result["_replay_id"] = response.headers.get("x-vsr-replay-id", "")
        receipt = self._wait_for_terminal_receipt(result, scheduled=False)
        self._assert_terminal(receipt, "skipped", "history_too_large", "true")

    def test_05_store_failure_reports_persist_error_and_stays_fail_open(self):
        """A rejected write reports persist_error while the model response succeeds."""
        self._apply_persistence_bounds(timeout_seconds=STORE_FAILURE_TIMEOUT_SECONDS)
        baseline = self._receipt_totals()
        with self._write_fault("fail"):
            result = self.send_memory_request(
                message=f"{RECEIPT_MARKER} My on-call rotation starts on Thursday.",
                auto_store=True,
                user_id=f"{self.test_user}_failopen",
            )
            self.assertIsNotNone(result)
            self.assertTrue(result.get("_output_text"))
            self._wait_for_fault(lambda state: state["rejected"] > 0)
            receipt = self._wait_for_terminal_receipt(
                result, timeout=STORE_FAILURE_RECEIPT_BUDGET_SECONDS
            )
            self._assert_terminal(receipt, "store_failed", "persist_error", "true")
            self.assertEqual(
                self._wait_for_terminal_deltas(baseline, 1), {"store_failed": 1}
            )
        self.assertEqual(self._wait_for_terminal_receipt(result), receipt)

    def test_06_exhausted_budget_reports_persist_timeout(self):
        """Hold a real write until its deadline, after the client receives output."""
        self._apply_persistence_bounds(
            timeout_seconds=PERSIST_TIMEOUT_SECONDS, concurrency=1
        )
        baseline = self._receipt_totals()
        with self._write_fault("hold"):
            result = self._saturate_persistence(1, "timeout")[0]
            self._assert_attempts_pending([result])
            receipt = self._wait_for_terminal_receipt(
                result, timeout=PERSIST_TIMEOUT_RECEIPT_BUDGET_SECONDS
            )
            self._assert_terminal(receipt, "timeout", "persist_timeout", "true")
            self._wait_for_fault(
                lambda state: state["cancelled"] >= 1 and state["active"] == 0
            )
            self.assertEqual(
                self._wait_for_terminal_deltas(baseline, 1), {"timeout": 1}
            )
        # Check again after the actual backend RPC has unwound and the gate opens.
        self.assertEqual(self._wait_for_terminal_receipt(result), receipt)
        self.assertEqual(self._terminal_deltas(baseline), {"timeout": 1})

    def test_07_reload_cancels_queued_writes_with_shutdown_reason(self):
        """Reload cancels every pending attempt while their writes remain held."""
        self._apply_persistence_bounds(
            timeout_seconds=RETIREMENT_TIMEOUT_SECONDS,
            concurrency=1,
            queue=RETIREMENT_BURST,
            shutdown_grace_seconds=RETIREMENT_GRACE_SECONDS,
        )
        baseline = self._receipt_totals()
        with self._write_fault("hold"):
            results = self._saturate_persistence(RETIREMENT_BURST, "cancelled")
            request_ids = self._assert_attempts_pending(results)
            self._apply_persistence_bounds(
                timeout_seconds=RETIREMENT_TIMEOUT_SECONDS,
                concurrency=1,
                queue=RETIREMENT_BURST,
                shutdown_grace_seconds=RETIREMENT_GRACE_SECONDS + 1,
            )
            # The fixture uses generation-local in-memory Replay storage. Read
            # request IDs before reload, then correlate its surviving logs.
            self._wait_for_cancelled_logs(request_ids)
            self._wait_for_fault(
                lambda state: state["cancelled"] >= 1 and state["active"] == 0
            )
            self.assertEqual(
                self._wait_for_terminal_deltas(baseline, RETIREMENT_BURST),
                {"cancelled": RETIREMENT_BURST},
            )
        self._assert_cancelled_logs(self._router_receipt_logs(), request_ids)
        self.assertEqual(
            self._terminal_deltas(baseline), {"cancelled": RETIREMENT_BURST}
        )


class MemoryPersistenceShutdownTest(PersistenceReceiptSupport):
    """Run last: process shutdown destroys the stack and its Replay endpoint."""

    def test_01_shutdown_cancels_queued_writes_and_logs_one_receipt_each(self):
        """Match exactly one cancelled/shutdown log to every accepted request."""
        self._apply_persistence_bounds(
            timeout_seconds=RETIREMENT_TIMEOUT_SECONDS,
            concurrency=1,
            queue=RETIREMENT_BURST,
            shutdown_grace_seconds=RETIREMENT_GRACE_SECONDS,
        )
        with self._write_fault("hold"):
            results = self._saturate_persistence(RETIREMENT_BURST, "shutdown")
            request_ids = self._assert_attempts_pending(results)
            self._stop_router()
            self._wait_for_fault(
                lambda state: state["cancelled"] >= 1 and state["active"] == 0
            )
            self._assert_cancelled_logs(self._router_receipt_logs(), request_ids)
