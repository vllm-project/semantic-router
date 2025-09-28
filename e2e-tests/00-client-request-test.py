#!/usr/bin/env python3
"""
00-client-request-test.py - Basic client request tests

This test validates that the Envoy proxy is running and accepting requests,
and that basic request formatting works correctly.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import os
import sys
import time
import unittest

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
ENVOY_URL = "http://localhost:8801"
OPENAI_ENDPOINT = "/v1/chat/completions"
DEFAULT_MODEL = (
    "Qwen/Qwen2-0.5B-Instruct"  # Use configured model that matches router config
)
MAX_RETRIES = 3
RETRY_DELAY = 2


class ClientRequestTest(SemanticRouterTestBase):
    """Test basic client requests to the Envoy proxy."""

    def setUp(self):
        """Check if the Envoy server is running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that Envoy server is running and accepting requests",
        )

        try:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [{"role": "user", "content": "test"}],
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
        except requests.exceptions.Timeout:
            self.skipTest("Connection to Envoy server timed out. Is it responding?")
        except Exception as e:
            self.skipTest(f"Unexpected error during setup: {str(e)}")

    def _make_request(self, payload, timeout=60):
        """Helper method to make requests with retries."""
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
                return response
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"Request attempt {attempt + 1} failed, retrying in {RETRY_DELAY} seconds..."
                    )
                    time.sleep(RETRY_DELAY)
                    continue
        if last_exception:
            print(f"All retry attempts failed. Last error: {str(last_exception)}")
        return None

    def test_basic_request_format(self):
        """Test that a well-formed request is accepted by the server."""
        self.print_test_header(
            "Basic Request Format Test",
            "Validates that a well-formed request is properly accepted by the server",
        )

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "temperature": 0.7,
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: 2xx status code, JSON response with 'model' field",
        )

        response = self._make_request(payload)
        if not response:
            self.fail("Failed to get response after retries")

        passed = True
        failure_reason = []

        if response.status_code >= 400:
            passed = False
            failure_reason.append(
                f"Request was rejected with status code {response.status_code}"
            )

        try:
            response_json = response.json()
            if "model" not in response_json:
                passed = False
                failure_reason.append("Response is missing 'model' field")
        except json.JSONDecodeError:
            passed = False
            failure_reason.append("Response is not valid JSON")

        self.print_response_info(
            response,
            {
                "Response Status": response.status_code,
                "Response Model": (
                    response_json.get("model", "Not specified") if passed else "N/A"
                ),
            },
        )

        self.print_test_result(
            passed=passed,
            message=(
                ", ".join(failure_reason) if failure_reason else "All checks passed"
            ),
        )

        self.assertTrue(passed, ", ".join(failure_reason))

    def test_malformed_request(self):
        """Test that a malformed request is properly rejected."""
        self.print_test_header(
            "Malformed Request Test",
            "Validates that malformed requests are properly rejected with appropriate error codes",
        )

        payload = {
            "model": DEFAULT_MODEL,
            # Missing required 'messages' field
            "temperature": 0.7,
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: 4xx client error (not 5xx), indicating proper request validation",
        )

        try:
            response = self._make_request(
                payload, timeout=10
            )  # Reduced timeout for error cases
            if not response:
                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=10,
                )

            passed = 400 <= response.status_code < 500

            self.print_response_info(
                response,
                {"Status Code": response.status_code, "Expected Status": "4xx"},
            )

            self.print_test_result(
                passed=passed,
                message=(
                    f"Got expected 4xx status code: {response.status_code}"
                    if passed
                    else f"Expected 4xx status code, got {response.status_code}"
                ),
            )

            self.assertTrue(
                passed, f"Expected 4xx status code, got {response.status_code}"
            )

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            self.print_test_result(
                passed=True,
                message="Request was rejected as expected (connection refused/timeout)",
            )
            # For malformed requests, connection errors or timeouts are acceptable outcomes
            return

    def test_domain_specific_queries(self):
        """Test sending different domain-specific queries."""
        self.print_test_header(
            "Domain-Specific Queries Test",
            "Tests routing behavior for different types of queries (math, creative, general knowledge)",
        )

        test_cases = [
            {
                "name": "Math Query",
                "messages": [
                    {"role": "system", "content": "You are a math teacher."},
                    {
                        "role": "user",
                        "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?",
                    },
                ],
                "expected_type": "Mathematical computation",
            },
            {
                "name": "Creative Writing Query",
                "messages": [
                    {"role": "system", "content": "You are a creative writer."},
                    {
                        "role": "user",
                        "content": "Write a short story about a space cat.",
                    },
                ],
                "expected_type": "Creative writing",
            },
            {
                "name": "General Knowledge Query",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "What are the main features of the Great Barrier Reef?",
                    },
                ],
                "expected_type": "General knowledge",
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": test_case["messages"],
                    "temperature": 0.7,
                }

                self.print_request_info(
                    payload=payload,
                    expectations=f"Expect: 2xx status code, appropriate model selection for {test_case['expected_type']}",
                )

                response = self._make_request(
                    payload, timeout=60
                )  # Increased timeout for model responses
                if not response:
                    self.fail(
                        f"Failed to get response for {test_case['name']} after retries"
                    )

                try:
                    response_json = response.json()
                    model = response_json.get("model", "unknown")
                    passed = response.status_code < 400 and model != "unknown"
                except json.JSONDecodeError:
                    passed = False
                    model = "invalid JSON response"

                self.print_response_info(
                    response,
                    {"Query Type": test_case["expected_type"], "Selected Model": model},
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"Successfully routed to model: {model}"
                        if passed
                        else f"Request failed with status {response.status_code} or invalid response"
                    ),
                )

                self.assertLess(
                    response.status_code,
                    400,
                    f"{test_case['name']} request failed with status {response.status_code}",
                )


if __name__ == "__main__":
    unittest.main()
