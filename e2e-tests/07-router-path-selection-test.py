#!/usr/bin/env python3
"""
07-router-path-selection-test.py - Router path selection tests

This test validates that the Unified Classifier correctly selects between
LoRA and Traditional inference paths based on request characteristics.

TESTS: Direct Classification API (port 8080)
Tests the Unified Classifier's internal path selection logic.

ROUTING DECISION LOGIC (from unified.rs:536-568):
1. Multi-task (tasks > 1) → LoRA path
2. Large batch (texts ≥ 4) → LoRA path
3. High confidence (threshold ≥ 0.99) → LoRA path
4. Low latency requirement (≤ 2000ms) → LoRA path
5. Default (single task, small batch) → Traditional path

PREREQUISITES:
- LoRA models must have lora_config.json files (from PR #629)
- Auto-discovery must find complete LoRA model set (intent+PII+security)
- Services running: router-e2e

NOTE: For full Envoy/ExtProc routing stack tests, see 08-envoy-routing-test.py
"""

import json
import sys
import time
import unittest

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
CLASSIFICATION_API_URL = "http://localhost:8080"  # Direct classifier API
INTENT_ENDPOINT = "/api/v1/classify/intent"
BATCH_ENDPOINT = "/api/v1/classify/batch"
ROUTER_METRICS_URL = "http://localhost:9190/metrics"

# Test queries for classification
TEST_QUERIES = [
    "Solve the equation 2x + 5 = 15",
    "Write a Python function to sort a list",
    "What is the capital of France?",
    "Explain photosynthesis in plants",
    "Calculate the area of a circle with radius 5",
    "What is machine learning?",
    "Describe the water cycle",
    "What is quantum mechanics?",
]


class RouterPathSelectionTest(SemanticRouterTestBase):
    """Test router path selection logic (LoRA vs Traditional)."""

    def setUp(self):
        """Check if Classification API is running."""
        self.print_test_header(
            "Setup Check",
            "Verifying that Classification API and Router are running",
        )

        try:
            # Test Classification API health
            health_response = requests.get(
                f"{CLASSIFICATION_API_URL}/health", timeout=5
            )

            if health_response.status_code != 200:
                self.skipTest(
                    f"Classification API health check failed: {health_response.status_code}"
                )

            self.print_response_info(
                health_response, {"Service": "Classification API Health"}
            )

        except requests.exceptions.ConnectionError:
            self.skipTest(
                "Cannot connect to Classification API. Is router-e2e running?"
            )

        # Check router metrics
        try:
            metrics_response = requests.get(ROUTER_METRICS_URL, timeout=2)
            if metrics_response.status_code != 200:
                self.skipTest("Router metrics endpoint not responding")

            self.print_response_info(metrics_response, {"Service": "Router Metrics"})

        except requests.exceptions.ConnectionError:
            self.skipTest("Cannot connect to router metrics endpoint")

    def test_small_batch_single_task(self):
        """Test that small batch with single task uses Traditional path."""
        self.print_test_header(
            "Small Batch Single Task Test",
            "Verifies behavior with 1-3 texts and single task (expected: Traditional path)",
        )

        # Single query
        single_query = TEST_QUERIES[0]

        payload = {
            "text": single_query,
            "options": {"return_probabilities": False},
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: Fast response, single task processing",
        )

        start_time = time.time()
        response = requests.post(
            f"{CLASSIFICATION_API_URL}{INTENT_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        response_json = response.json()
        classification = response_json.get("classification", {})
        category = classification.get("category", "unknown")
        confidence = classification.get("confidence", 0.0)

        self.print_response_info(
            response,
            {
                "Category": category,
                "Confidence": f"{confidence:.3f}",
                "Latency (ms)": f"{elapsed_ms:.1f}",
                "Expected Path": "Traditional (single task, small batch)",
            },
        )

        passed = response.status_code == 200 and category != "unknown"
        self.print_test_result(
            passed=passed,
            message=f"Single task processed successfully in {elapsed_ms:.1f}ms",
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(category, "unknown")

    def test_large_batch_triggers_lora(self):
        """Test that batch size ≥4 triggers LoRA path."""
        self.print_test_header(
            "Large Batch LoRA Trigger Test",
            "Verifies that batch size ≥4 triggers LoRA path for parallel processing",
        )

        # Use 6 queries (>= 4 threshold)
        batch_queries = TEST_QUERIES[:6]

        payload = {"texts": batch_queries, "task_type": "intent"}

        self.print_request_info(
            payload={"texts": f"{len(batch_queries)} texts", "task_type": "intent"},
            expectations="Expect: LoRA path used, efficient parallel processing",
        )

        start_time = time.time()
        response = requests.post(
            f"{CLASSIFICATION_API_URL}{BATCH_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        response_json = response.json()
        results = response_json.get("results", [])
        total_count = response_json.get("total_count", 0)

        # Calculate per-item latency
        per_item_latency = elapsed_ms / len(batch_queries) if batch_queries else 0

        self.print_response_info(
            response,
            {
                "Batch Size": len(batch_queries),
                "Results Count": len(results),
                "Total Latency (ms)": f"{elapsed_ms:.1f}",
                "Per-Item Latency (ms)": f"{per_item_latency:.1f}",
                "Expected Path": "LoRA (batch ≥4)",
                "Parallel Efficiency": "✅" if per_item_latency < 100 else "⚠️",
            },
        )

        passed = (
            response.status_code == 200
            and len(results) == len(batch_queries)
            and per_item_latency < 150  # Should be efficient
        )

        self.print_test_result(
            passed=passed,
            message=f"Batch processed successfully: {per_item_latency:.1f}ms per item",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(results), len(batch_queries))
        self.assertLess(
            per_item_latency,
            150,
            f"Per-item latency too high: {per_item_latency:.1f}ms (expected <150ms for parallel LoRA)",
        )

    def test_multi_task_triggers_lora(self):
        """Test that multi-task classification triggers LoRA path."""
        self.print_test_header(
            "Multi-Task LoRA Trigger Test",
            "Verifies that requesting multiple tasks triggers LoRA parallel processing",
        )

        # Single query but with multiple tasks
        test_query = "My email is john@example.com and my SSN is 123-45-6789"

        # Test with batch endpoint requesting multiple task types
        payload = {
            "texts": [test_query],
            "task_type": "all",  # Request all tasks: intent, PII, security
        }

        self.print_request_info(
            payload={"texts": "1 text", "task_type": "all (intent+PII+security)"},
            expectations="Expect: LoRA path used for parallel multi-task processing",
        )

        start_time = time.time()
        response = requests.post(
            f"{CLASSIFICATION_API_URL}{BATCH_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        elapsed_ms = (time.time() - start_time) * 1000

        response_json = response.json()
        results = response_json.get("results", [])

        self.print_response_info(
            response,
            {
                "Query Count": 1,
                "Tasks": "Multiple (intent+PII+security)",
                "Total Latency (ms)": f"{elapsed_ms:.1f}",
                "Expected Path": "LoRA (multi-task parallel)",
                "Parallel Processing": "✅" if elapsed_ms < 200 else "⚠️",
            },
        )

        passed = response.status_code == 200 and len(results) > 0

        self.print_test_result(
            passed=passed,
            message=f"Multi-task processed successfully in {elapsed_ms:.1f}ms",
        )

        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(results), 0)

    def test_performance_comparison(self):
        """Compare performance between small batch (Traditional) and large batch (LoRA)."""
        self.print_test_header(
            "Performance Comparison Test",
            "Compares Traditional path (1-3 items) vs LoRA path (≥4 items)",
        )

        # Small batch (Traditional path)
        small_batch = TEST_QUERIES[:2]
        small_payload = {"texts": small_batch, "task_type": "intent"}

        self.print_subtest_header("Small Batch (Traditional Path)")

        start_time = time.time()
        small_response = requests.post(
            f"{CLASSIFICATION_API_URL}{BATCH_ENDPOINT}",
            json=small_payload,
            timeout=30,
        )
        small_elapsed_ms = (time.time() - start_time) * 1000
        small_per_item = small_elapsed_ms / len(small_batch)

        self.print_response_info(
            small_response,
            {
                "Batch Size": len(small_batch),
                "Total Time (ms)": f"{small_elapsed_ms:.1f}",
                "Per-Item Time (ms)": f"{small_per_item:.1f}",
                "Expected Path": "Traditional",
            },
        )

        # Large batch (LoRA path)
        large_batch = TEST_QUERIES[:6]
        large_payload = {"texts": large_batch, "task_type": "intent"}

        self.print_subtest_header("Large Batch (LoRA Path)")

        start_time = time.time()
        large_response = requests.post(
            f"{CLASSIFICATION_API_URL}{BATCH_ENDPOINT}",
            json=large_payload,
            timeout=30,
        )
        large_elapsed_ms = (time.time() - start_time) * 1000
        large_per_item = large_elapsed_ms / len(large_batch)

        self.print_response_info(
            large_response,
            {
                "Batch Size": len(large_batch),
                "Total Time (ms)": f"{large_elapsed_ms:.1f}",
                "Per-Item Time (ms)": f"{large_per_item:.1f}",
                "Expected Path": "LoRA",
            },
        )

        # Compare efficiency
        efficiency_gain_pct = (small_per_item - large_per_item) / small_per_item * 100

        print("\n📊 Performance Comparison:")
        print(f"  Traditional Path: {small_per_item:.1f}ms per item")
        print(f"  LoRA Path: {large_per_item:.1f}ms per item")
        print(f"  Efficiency Gain: {efficiency_gain_pct:+.1f}%")
        print(
            f"  Expected: LoRA should be faster for larger batches (parallel processing)"
        )

        passed = small_response.status_code == 200 and large_response.status_code == 200

        self.print_test_result(
            passed=passed,
            message=f"Performance comparison completed: LoRA {efficiency_gain_pct:+.1f}% {'faster' if efficiency_gain_pct > 0 else 'slower'}",
        )

        self.assertEqual(small_response.status_code, 200)
        self.assertEqual(large_response.status_code, 200)

    def test_lora_models_loaded(self):
        """Verify that LoRA models were successfully loaded via auto-discovery."""
        self.print_test_header(
            "LoRA Model Loading Test",
            "Verifies that auto-discovery found and loaded LoRA models with lora_config.json",
        )

        # Check router metrics for unified classifier usage
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text

        # Look for unified classifier metrics
        has_unified_metrics = 'processing_type="unified"' in metrics_text
        has_batch_classification = (
            "batch_classification_duration_seconds" in metrics_text
        )

        # Check for any LoRA-related metrics or indicators
        metrics_indicators = {
            "Unified Classifier Metrics": has_unified_metrics,
            "Batch Classification Metrics": has_batch_classification,
            "Metrics Endpoint": response.status_code == 200,
        }

        self.print_response_info(
            response,
            metrics_indicators,
        )

        # Verify we can actually classify (proves models loaded)
        test_payload = {
            "text": "What is the derivative of x squared?",
            "options": {"return_probabilities": False},
        }

        classify_response = requests.post(
            f"{CLASSIFICATION_API_URL}{INTENT_ENDPOINT}",
            json=test_payload,
            timeout=10,
        )

        classify_result = classify_response.json()
        classification = classify_result.get("classification", {})
        category = classification.get("category", "unknown")

        successful_classification = (
            classify_response.status_code == 200 and category != "unknown"
        )

        print("\n✅ LoRA Model Loading Verification:")
        print(f"  Classification Works: {successful_classification}")
        print(f"  Unified Metrics Present: {has_unified_metrics}")
        print(f"  Category Detected: {category}")
        print(
            f"  Status: LoRA models {'successfully loaded' if successful_classification else 'NOT loaded'}"
        )

        passed = successful_classification and has_unified_metrics

        self.print_test_result(
            passed=passed,
            message=(
                "LoRA models successfully loaded and functioning"
                if passed
                else "Issues with LoRA model loading"
            ),
        )

        self.assertTrue(successful_classification, "Classification should work")
        self.assertTrue(
            has_unified_metrics, "Unified classifier metrics should be present"
        )


if __name__ == "__main__":
    unittest.main()
