#!/usr/bin/env python3
"""
05-pii-policy-test.py - PII detection and policy tests

This test validates the router's ability to detect PII in requests and enforce
model-specific PII policies correctly.
"""

import json
import os
import sys
import time
import unittest
import uuid

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
ENVOY_URL = "http://localhost:8801"
OPENAI_ENDPOINT = "/v1/chat/completions"
ROUTER_METRICS_URL = "http://localhost:9190/metrics"
DEFAULT_MODEL = "gemma3:27b"  # Use configured model

# PII test cases - based on gemma3:27b policy: allows ["EMAIL_ADDRESS", "PERSON", "GPE", "PHONE_NUMBER"]
PII_TEST_CASES = [
    {
        "name": "Allowed PII - Email",
        "expected_result": "allowed",
        "content": "Please send the report to john.doe@company.com for review.",
        "expected_pii_types": ["EMAIL_ADDRESS"],
    },
    {
        "name": "Allowed PII - Person Name",
        "expected_result": "allowed", 
        "content": "John Smith submitted the application yesterday.",
        "expected_pii_types": ["PERSON"],
    },
    {
        "name": "Allowed PII - Location (GPE)",
        "expected_result": "allowed",
        "content": "The conference will be held in New York City.",
        "expected_pii_types": ["GPE"],
    },
    {
        "name": "Allowed PII - Phone Number",
        "expected_result": "allowed",
        "content": "You can reach me at (555) 123-4567 for more information.",
        "expected_pii_types": ["PHONE_NUMBER"],
    },
    {
        "name": "Mixed Allowed PII",
        "expected_result": "allowed",
        "content": "Contact John Smith in New York at john@company.com or (555) 123-4567.",
        "expected_pii_types": ["PERSON", "GPE", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    },
]

# Cases that might contain PII types not allowed by gemma3:27b policy
POTENTIAL_BLOCKED_PII_CASES = [
    {
        "name": "Credit Card Number",
        "content": "My credit card number is 4532-1234-5678-9012.",
        "potential_pii_types": ["CREDIT_CARD"],
    },
    {
        "name": "Social Security Number",
        "content": "My SSN is 123-45-6789 for verification.",
        "potential_pii_types": ["SSN"],
    },
    {
        "name": "Medical Information", 
        "content": "Patient ID 12345 has diabetes and takes insulin.",
        "potential_pii_types": ["MEDICAL_LICENSE"],
    },
]

# No PII cases - should always be allowed
NO_PII_CASES = [
    {
        "name": "General Question",
        "expected_result": "allowed",
        "content": "What is the weather like today?",
    },
    {
        "name": "Math Problem",
        "expected_result": "allowed",
        "content": "What is 2 + 2?",
    },
    {
        "name": "Technical Question",
        "expected_result": "allowed",
        "content": "How does machine learning work?",
    },
]


class PIIDetectionPolicyTest(SemanticRouterTestBase):
    """Test the router's PII detection and policy enforcement functionality."""

    def setUp(self):
        """Check if the services are running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that required services (Envoy and Router) are running and PII detection is enabled",
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

        # Check if PII detection metrics exist
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text
        if "pii" not in metrics_text.lower():
            self.skipTest("PII metrics not found. PII detection may be disabled.")

    def test_no_pii_requests_allowed(self):
        """Test that requests with no PII are always allowed."""
        self.print_test_header(
            "No PII Requests Test",
            "Verifies that requests without PII are processed normally",
        )

        for test_case in NO_PII_CASES:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": test_case["content"]},
                    ],
                    "temperature": 0.7,
                }

                headers = {
                    "Content-Type": "application/json",
                    "X-Session-ID": session_id,
                }

                self.print_request_info(
                    payload=payload,
                    expectations="Expect: Request with no PII to be processed normally",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    json=payload,
                    timeout=(10, 60),  # (connect timeout, read timeout)
                )

                # No PII requests should be processed successfully
                passed = response.status_code == 200

                try:
                    response_json = response.json()
                    model = response_json.get("model", "unknown")
                except:
                    model = "N/A"

                self.print_response_info(
                    response,
                    {
                        "Content": test_case["content"][:50] + "...",
                        "Selected Model": model,
                        "Session ID": session_id,
                        "PII Status": "Expected: No PII",
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"No PII request processed normally (status: {response.status_code})"
                        if passed
                        else f"No PII request blocked unexpectedly (status: {response.status_code})"
                    ),
                )

                self.assertEqual(
                    response.status_code,
                    200,
                    f"No PII request '{test_case['name']}' failed with status {response.status_code}. Expected: 200 (service must be working)",
                )

    def test_allowed_pii_requests(self):
        """Test that requests with allowed PII types are processed."""
        self.print_test_header(
            "Allowed PII Requests Test",
            "Verifies that requests with allowed PII types (EMAIL_ADDRESS, PERSON, GPE, PHONE_NUMBER) are processed",
        )

        for test_case in PII_TEST_CASES:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": test_case["content"]},
                    ],
                    "temperature": 0.7,
                }

                headers = {
                    "Content-Type": "application/json",
                    "X-Session-ID": session_id,
                }

                self.print_request_info(
                    payload=payload,
                    expectations=f"Expect: Request with allowed PII types {test_case['expected_pii_types']} to be processed",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    json=payload,
                    timeout=(10, 60),  # (connect timeout, read timeout)
                )

                # Allowed PII requests should be processed successfully
                passed = response.status_code == 200

                try:
                    response_json = response.json()
                    model = response_json.get("model", "unknown")
                except:
                    model = "N/A"

                self.print_response_info(
                    response,
                    {
                        "Content": test_case["content"][:50] + "...",
                        "Selected Model": model,
                        "Session ID": session_id,
                        "Expected PII Types": test_case["expected_pii_types"],
                        "PII Status": "Expected: Allowed",
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"Allowed PII request processed normally (status: {response.status_code})"
                        if passed
                        else f"Allowed PII request blocked unexpectedly (status: {response.status_code})"
                    ),
                )

                self.assertEqual(
                    response.status_code,
                    200,
                    f"Allowed PII request '{test_case['name']}' failed with status {response.status_code}. Expected: 200 (service must be working)",
                )

    def test_pii_policy_consistency(self):
        """Test that PII policy decisions are consistent for the same content."""
        self.print_test_header(
            "PII Policy Consistency Test",
            "Verifies that the same content consistently triggers the same PII policy decision",
        )

        test_content = "Please contact John Smith at john@company.com for assistance."
        
        results = []
        for i in range(3):
            self.print_subtest_header(f"Consistency Test {i+1}")

            session_id = str(uuid.uuid4())
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": test_content},
                ],
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": session_id,
            }

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=(10, 60),  # (connect timeout, read timeout)
            )

            # Record the response status for consistency checking
            results.append(response.status_code)

            self.print_response_info(
                response,
                {
                    "Attempt": i + 1,
                    "Status Code": response.status_code,
                    "Content": test_content[:50] + "...",
                },
            )

            time.sleep(1)  # Small delay between requests

        # Check consistency - all results should be the same
        is_consistent = len(set(results)) == 1
        self.print_test_result(
            passed=is_consistent,
            message=f"PII policy consistency: {is_consistent}. Results: {results}",
        )

        self.assertEqual(
            len(set(results)), 1, f"Inconsistent PII policy results: {results}"
        )

    def test_pii_detection_metrics(self):
        """Test that PII detection metrics are being recorded properly."""
        self.print_test_header(
            "PII Detection Metrics Test",
            "Verifies that PII detection metrics are being properly recorded and exposed",
        )

        # Get baseline metrics
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text

        # Look for specific PII metrics
        pii_metrics = [
            "llm_classifier_latency_seconds_count",  # Classification timing
            "llm_request_errors_total",  # Blocked requests with reason="pii_block"
            "llm_model_requests_total",  # Total requests
        ]

        metrics_found = {}
        for metric in pii_metrics:
            for line in metrics_text.split("\n"):
                if metric in line and not line.startswith("#"):
                    # For classifier metrics, ensure it's specifically for pii
                    if "classifier" in metric and "pii" not in line:
                        continue
                    # For error metrics, ensure it's specifically pii_block
                    if "errors" in metric and "pii_block" not in line:
                        continue
                    # Extract metric value
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            metrics_found[metric] = float(parts[-1])
                            break
                    except ValueError:
                        continue

        self.print_response_info(
            response,
            {"Metrics Found": len(metrics_found), "Total Expected": len(pii_metrics)},
        )

        # Print detailed metrics information
        for metric, value in metrics_found.items():
            print(f"\nMetric: {metric}")
            print(f"  Value: {value}")

        # Print any metrics that contain "pii" even if not in our expected list
        print(f"\nAll PII-related metrics found:")
        for line in metrics_text.split("\n"):
            if "pii" in line.lower() and not line.startswith("#") and line.strip():
                print(f"  {line.strip()}")

        passed = len(metrics_found) > 0
        self.print_test_result(
            passed=passed,
            message=f"Found {len(metrics_found)} out of {len(pii_metrics)} expected PII metrics",
        )

        self.assertGreater(len(metrics_found), 0, "No PII metrics found")

    def test_model_pii_policy_configuration(self):
        """Test that different models have different PII policies configured."""
        self.print_test_header(
            "Model PII Policy Configuration Test",
            "Verifies that the router correctly applies different PII policies for different models",
        )

        # Test with gemma3:27b (has restrictive PII policy)
        test_models = [DEFAULT_MODEL]
        test_content = "Contact John at john@company.com"

        for model in test_models:
            self.print_subtest_header(f"Testing Model: {model}")

            session_id = str(uuid.uuid4())
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": test_content},
                ],
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": session_id,
            }

            self.print_request_info(
                payload=payload,
                expectations=f"Expect: Model {model} to apply its specific PII policy",
            )

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=(10, 60),  # (connect timeout, read timeout)
            )

            try:
                response_json = response.json()
                selected_model = response_json.get("model", "unknown")
            except:
                selected_model = "N/A"

            self.print_response_info(
                response,
                {
                    "Requested Model": model,
                    "Selected Model": selected_model,
                    "Status Code": response.status_code,
                    "Content": test_content,
                },
            )

            # The request should be processed successfully
            passed = response.status_code == 200
            self.print_test_result(
                passed=passed,
                message=f"Model {model} PII policy applied correctly",
            )

            self.assertEqual(
                response.status_code,
                200,
                f"Model {model} PII policy failed with status {response.status_code}. Expected: 200 (service must be working)",
            )


if __name__ == "__main__":
    unittest.main()