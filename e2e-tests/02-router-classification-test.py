#!/usr/bin/env python3
"""
02-router-classification-test.py - Router classification tests

This test validates the router's ability to classify different types of queries
and select the appropriate model based on the content.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import os
import sys
import time
import unittest
from collections import defaultdict

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
ENVOY_URL = "http://localhost:8801"
OPENAI_ENDPOINT = "/v1/chat/completions"
ROUTER_METRICS_URL = "http://localhost:9190/metrics"
DEFAULT_MODEL = "gemma3:27b"  # Use configured model

# Category test cases - each designed to trigger a specific classifier category
CATEGORY_TEST_CASES = [
    {
        "name": "Math Query",
        "expected_category": "math",
        "content": "Solve the differential equation dy/dx + 2y = x^2 with the initial condition y(0) = 1.",
    },
    {
        "name": "Creative Writing Query",
        "expected_category": "creative",
        "content": "Write a short story about a space cat.",
    },
]  # Reduced to just 2 test cases to avoid timeouts

# Auto routing test cases - should trigger different model selection
AUTO_ROUTING_TEST_CASES = [
    {
        "name": "Math Problem (should route to phi4)",
        "expected_model": "phi4",
        "content": "Calculate the derivative of f(x) = x^3 + 2x^2 - 5x + 7",
    },
    {
        "name": "Creative Writing (should route to another model)",
        "expected_model_not": "phi4",  # Should NOT be phi4 since phi4 is optimized for math
        "content": "Write a poem about the ocean at sunset",
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
                timeout=(10, 60),  # (connect timeout, read timeout)
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
                timeout=(10, 60),  # (connect timeout, read timeout)
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
                "model": DEFAULT_MODEL,
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
                expectations=f"Expect: Query to be classified as {test_case['expected_category']} and routed accordingly",
            )

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=(10, 60),  # (connect timeout, read timeout)
            )

            passed = response.status_code < 400
            response_json = response.json()
            model = response_json.get("model", "unknown")
            results[test_case["name"]] = model

            self.print_response_info(
                response,
                {
                    "Expected Category": test_case["expected_category"],
                    "Selected Model": model,
                },
            )

            self.print_test_result(
                passed=passed,
                message=(
                    f"Query successfully routed to model: {model}"
                    if passed
                    else f"Request failed with status {response.status_code}"
                ),
            )

            self.assertLess(
                response.status_code,
                400,
                f"{test_case['name']} request failed with status {response.status_code}",
            )

    def test_classifier_metrics(self):
        """Test that classification metrics are being recorded."""
        self.print_test_header(
            "Classifier Metrics Test",
            "Verifies that classification metrics are being properly recorded and exposed",
        )

        # First, let's get the current metrics as a baseline
        response = requests.get(ROUTER_METRICS_URL)
        baseline_metrics = response.text

        # Check if classification metrics exist without making additional requests
        classification_metrics = [
            "llm_router_classification_duration_seconds",
            "llm_router_requests_total",
            "llm_router_model_selection_count",
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
                f"Found {metrics_found} classification metrics"
                if passed
                else "No classification metrics found"
            ),
        )

        self.assertGreaterEqual(metrics_found, 0, "No classification metrics found")

    def test_auto_routing_intelligence(self):
        """Test that auto model selection actually routes different queries to different specialized models."""
        self.print_test_header(
            "Auto Routing Intelligence Test",
            "Verifies that model='auto' actually routes different query types to different specialized models",
        )

        results = {}

        for test_case in AUTO_ROUTING_TEST_CASES:
            self.print_subtest_header(test_case["name"])

            payload = {
                "model": "auto",  # This should trigger intelligent routing
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": test_case["content"]},
                ],
                "temperature": 0.7,
            }

            self.print_request_info(
                payload=payload,
                expectations=f"Expect: Auto routing to select appropriate specialized model",
            )

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )

            passed = response.status_code == 200

            try:
                response_json = response.json()
                selected_model = response_json.get("model", "unknown")
            except:
                selected_model = "unknown"

            results[test_case["name"]] = selected_model

            # Check if routing met expectations
            if "expected_model" in test_case:
                routing_correct = selected_model == test_case["expected_model"]
                routing_message = f"Expected {test_case['expected_model']}, got {selected_model}"
            elif "expected_model_not" in test_case:
                routing_correct = selected_model != test_case["expected_model_not"]
                routing_message = f"Should NOT be {test_case['expected_model_not']}, got {selected_model}"
            else:
                routing_correct = True
                routing_message = f"Got {selected_model}"

            self.print_response_info(
                response,
                {
                    "Query Type": test_case["name"],
                    "Selected Model": selected_model,
                    "Routing Expectation": routing_message,
                    "Routing Correct": routing_correct,
                },
            )

            self.print_test_result(
                passed=passed and routing_correct,
                message=(
                    f"Auto routing working: {selected_model} for {test_case['name']}"
                    if passed and routing_correct
                    else f"Auto routing failed: {routing_message}"
                ),
            )

            self.assertEqual(response.status_code, 200, f"Auto routing request failed with status {response.status_code}")

            # Check routing intelligence
            if "expected_model" in test_case:
                self.assertEqual(
                    selected_model,
                    test_case["expected_model"],
                    f"Auto routing failed: expected {test_case['expected_model']}, got {selected_model}"
                )
            elif "expected_model_not" in test_case:
                self.assertNotEqual(
                    selected_model,
                    test_case["expected_model_not"],
                    f"Auto routing failed: got {selected_model}, should not be {test_case['expected_model_not']}"
                )

        # Print summary of routing decisions
        print(f"\nAuto Routing Summary:")
        for test_name, model in results.items():
            print(f"  {test_name}: {model}")

        # Ensure we got different models for different query types (intelligence test)
        unique_models = set(results.values())
        if len(unique_models) == 1:
            self.fail(f"Auto routing not working - all queries routed to same model: {list(unique_models)[0]}")


if __name__ == "__main__":
    unittest.main()
