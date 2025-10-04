#!/usr/bin/env python3
"""
02-router-classification-test.py - Router classification tests

This test validates the router's ability to classify different types of queries
and select the appropriate model based on the content.
"""

import json
import os
import sys
import time
import unittest
from collections import defaultdict

import requests

# Add parent directory to path to allow importing common test utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_base import SemanticRouterTestBase

# Constants
ENVOY_URL = "http://localhost:8801"
OPENAI_ENDPOINT = "/v1/chat/completions"
ROUTER_METRICS_URL = "http://localhost:9190/metrics"
DEFAULT_MODEL = "Model-A"  # Use configured model that matches router config

# Category test cases - each designed to trigger a specific classifier category
# Based on config.e2e.yaml: math→Model-B, computer science→Model-B, business→Model-A, history→Model-A
CATEGORY_TEST_CASES = [
    {
        "name": "Math Query",
        "expected_category": "math",
        "expected_model": "Model-B",  # math has Model-B with score 1.0
        "content": "Solve the quadratic equation x^2 + 5x + 6 = 0 and explain the steps.",
    },
    {
        "name": "Computer Science/Coding Query",
        "expected_category": "computer science",
        "expected_model": "Model-B",  # computer science has Model-B with score 0.6
        "content": "Write a Python function to implement a linked list with insert and delete operations.",
    },
    {
        "name": "Business Query",
        "expected_category": "business",
        "expected_model": "Model-A",  # business has Model-A with score 0.8
        "content": "What are the key principles of supply chain management in modern business?",
    },
    {
        "name": "History Query",
        "expected_category": "history",
        "expected_model": "Model-A",  # history has Model-A with score 0.8
        "content": "Describe the main causes and key events of World War I.",
    },
]


class RouterClassificationTest(SemanticRouterTestBase):
    """Test the router's classification functionality."""

    def setUp(self):
        """Check if the services are running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that required services (Envoy and Router) are running",
        )

        # Check Envoy
        try:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "assistant", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "test"},
                ],
            }

            self.print_request_info(
                payload=payload,
                expectations="Expect: Service health check to succeed with 2xx status code",
            )

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )

            if response.status_code >= 500:
                self.skipTest(
                    f"Envoy server returned server error: {response.status_code}"
                )

            self.print_response_info(response)

        except requests.exceptions.ConnectionError:
            self.skipTest("Cannot connect to Envoy server. Is it running?")

        # Check router metrics endpoint
        try:
            response = requests.get(ROUTER_METRICS_URL, timeout=2)
            if response.status_code != 200:
                self.skipTest(
                    "Router metrics server is not responding. Is the router running?"
                )

            self.print_response_info(response, {"Service": "Router Metrics"})

        except requests.exceptions.ConnectionError:
            self.skipTest(
                "Cannot connect to router metrics server. Is the router running?"
            )

    def test_classification_consistency(self):
        """Test that the same query consistently routes to the same model."""
        self.print_test_header(
            "Classification Consistency Test",
            "Verifies that the same query is consistently routed to the same model",
        )

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "assistant", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "What are the main renewable energy sources?",
                },
            ],
            "temperature": 0.7,
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: Same model selection across multiple identical requests",
        )

        models = []
        # Send the same request 3 times
        for i in range(3):
            self.print_subtest_header(f"Request {i+1}")

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )

            passed = response.status_code < 400
            response_json = response.json()
            models.append(response_json.get("model", "unknown"))

            self.print_response_info(response, {"Selected Model": models[-1]})

            # Small delay between requests
            time.sleep(1)

        # Check consistency
        is_consistent = len(set(models)) == 1
        self.print_test_result(
            passed=is_consistent,
            message=f"Model selection was {'consistent' if is_consistent else 'inconsistent'}: {models}",
        )

        self.assertEqual(
            len(set(models)), 1, f"Same query routed to different models: {models}"
        )

    def test_category_classification(self):
        """Test that different categories of queries route to appropriate models."""
        self.print_test_header(
            "Category Classification Test",
            "Verifies that different types of queries are routed to appropriate specialized models",
        )

        results = {}

        for test_case in CATEGORY_TEST_CASES:
            self.print_subtest_header(test_case["name"])

            payload = {
                "model": "auto",  # Use "auto" to trigger category-based classification routing
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"You are an expert in {test_case['expected_category']}.",
                    },
                    {"role": "user", "content": test_case["content"]},
                ],
                "temperature": 0.7,
            }

            self.print_request_info(
                payload=payload,
                expectations=f"Expect: Query classified as '{test_case['expected_category']}' → routed to {test_case.get('expected_model', 'appropriate model')}",
            )

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )

            response_json = response.json()
            actual_model = response_json.get("model", "unknown")
            expected_model = test_case.get("expected_model", "unknown")
            results[test_case["name"]] = actual_model

            model_match = actual_model == expected_model
            passed = response.status_code < 400 and model_match

            self.print_response_info(
                response,
                {
                    "Expected Category": test_case["expected_category"],
                    "Expected Model": expected_model,
                    "Actual Model": actual_model,
                    "Routing Correct": "✅" if model_match else "❌",
                },
            )

            self.print_test_result(
                passed=passed,
                message=(
                    f"Query correctly routed to {actual_model}"
                    if model_match
                    else f"Routing failed: expected {expected_model}, got {actual_model}"
                ),
            )

            self.assertLess(
                response.status_code,
                400,
                f"{test_case['name']} request failed with status {response.status_code}",
            )

            self.assertEqual(
                actual_model,
                expected_model,
                f"{test_case['name']}: Expected routing to {expected_model}, but got {actual_model}",
            )

    def test_classifier_metrics(self):
        """Test that router metrics are being recorded and exposed."""
        self.print_test_header(
            "Router Metrics Test",
            "Verifies that router metrics (classification, cache operations) are being properly recorded and exposed",
        )

        # First, let's get the current metrics as a baseline
        response = requests.get(ROUTER_METRICS_URL)
        baseline_metrics = response.text

        # Check if classification and routing metrics exist
        # These are the actual metrics exposed by the router
        classification_metrics = [
            "llm_entropy_classification_latency_seconds",  # Entropy-based classification timing
            "llm_cache_hits_total",  # Cache operations (related to classification)
            "llm_cache_misses_total",  # Cache misses
        ]

        metrics_found = 0
        found_metrics = {}

        for metric in classification_metrics:
            if metric in baseline_metrics:
                metrics_found += 1
                found_metrics[metric] = []

                # Extract the relevant metric lines
                for line in baseline_metrics.split("\n"):
                    if metric in line and not line.startswith("#"):
                        found_metrics[metric].append(line.strip())

        self.print_response_info(response, {"Metrics Found": metrics_found})

        # Print detailed metrics information
        for metric, lines in found_metrics.items():
            print(f"\nMetric: {metric}")
            for line in lines:
                print(f"  {line}")

        passed = metrics_found > 0
        self.print_test_result(
            passed=passed,
            message=(
                f"Found {metrics_found}/{len(classification_metrics)} router metrics"
                if passed
                else "No router metrics found"
            ),
        )

        self.assertGreater(
            metrics_found,
            0,
            f"No router metrics found. Expected at least one of: {', '.join(classification_metrics)}",
        )


if __name__ == "__main__":
    unittest.main()
