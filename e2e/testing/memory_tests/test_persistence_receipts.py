"""Response-side memory persistence receipt tests."""

import os
import re
import shutil
import subprocess
import time

import requests
from memory_tests.base import MemoryFeaturesTest

RECEIPT_METRIC = "llm_plugin_execution_total"
RECEIPT_PLUGIN = "memory_persistence"
MILVUS_CONTAINER = "milvus-semantic-cache"
METRICS_CANDIDATES = (
    "http://localhost:9190/metrics",
    "http://localhost:9390/metrics",
)
LABEL_PATTERN = re.compile(r'(\w+)="([^"]*)"')


class MemoryPersistenceReceiptTest(MemoryFeaturesTest):
    """Verify persistence receipts reach Prometheus and that a failed write stays fail-open."""

    def setUp(self):
        super().setUp()
        self.metrics_url = self._resolve_metrics_url()
        self.container_runtime = os.environ.get("CONTAINER_RUNTIME", "docker")

    def _resolve_metrics_url(self) -> str:
        configured = os.environ.get("ROUTER_METRICS_URL")
        candidates = (configured,) if configured else METRICS_CANDIDATES
        for url in candidates:
            try:
                response = requests.get(url, timeout=5)
            except requests.exceptions.RequestException:
                continue
            if response.status_code == 200 and RECEIPT_METRIC in response.text:
                return url
        self.skipTest("router metrics endpoint is not reachable")

    def _receipt_count(self, status: str) -> float:
        try:
            response = requests.get(self.metrics_url, timeout=10)
        except requests.exceptions.RequestException as e:
            self.fail(f"metrics scrape failed: {e}")

        total = 0.0
        prefix = RECEIPT_METRIC + "{"
        for line in response.text.splitlines():
            if not line.startswith(prefix):
                continue
            labels_part, _, value = line.rpartition(" ")
            labels = dict(LABEL_PATTERN.findall(labels_part))
            if (
                labels.get("plugin_type") == RECEIPT_PLUGIN
                and labels.get("status") == status
            ):
                total += float(value)
        return total

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

    def _container_available(self) -> bool:
        if not shutil.which(self.container_runtime):
            return False
        result = subprocess.run(
            [self.container_runtime, "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return MILVUS_CONTAINER in result.stdout.split()

    def _container_command(self, action: str) -> None:
        subprocess.run(
            [self.container_runtime, action, MILVUS_CONTAINER],
            capture_output=True,
            text=True,
            check=True,
        )

    def test_01_successful_store_reports_completed_receipt(self):
        """An auto-stored turn reaches a completed persistence receipt."""
        self.print_test_header(
            "Persistence Receipt: Completed",
            "Auto-stored turn increments the memory_persistence completed counter",
        )

        baseline = self._receipt_count("completed")
        result = self.send_memory_request(
            message="My preferred deployment region is eu-central-1.",
            auto_store=True,
        )
        self.assertIsNotNone(result, "auto-store request did not return a response")

        observed = self._wait_for_receipt("completed", baseline)
        self.assertGreater(
            observed,
            baseline,
            "auto-stored turn produced no completed persistence receipt",
        )
        self.print_test_result(True, f"completed receipts {baseline} -> {observed}")

    def test_02_store_failure_keeps_response_fail_open(self):
        """A dead memory backend still returns the model response and reports store_failed."""
        self.print_test_header(
            "Persistence Receipt: Fail-Open",
            "With Milvus stopped the response still succeeds and reports store_failed",
        )

        if not self._container_available():
            self.skipTest(
                f"{MILVUS_CONTAINER} is not a running container for this runtime"
            )

        baseline = self._receipt_count("store_failed")
        self._container_command("stop")
        try:
            result = self.send_memory_request(
                message="My on-call rotation starts on Thursday.",
                auto_store=True,
                user_id=f"{self.test_user}_failopen",
            )
            self.assertIsNotNone(
                result,
                "memory write failure must not prevent the model response",
            )
            self.assertNotEqual(
                result.get("_output_text", ""),
                "",
                "fail-open response must still carry model output",
            )

            observed = self._wait_for_receipt("store_failed", baseline)
            self.assertGreater(
                observed,
                baseline,
                "failed write produced no store_failed persistence receipt",
            )
        finally:
            self._container_command("start")
            time.sleep(self.storage_wait)

        self.print_test_result(True, f"store_failed receipts {baseline} -> {observed}")
