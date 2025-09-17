#!/usr/bin/env python3
"""
01-envoy-extproc-test.py - Envoy ExtProc interaction tests

This test verifies that Envoy is correctly forwarding requests to the ExtProc,
and that the ExtProc is responding with appropriate routing decisions.
These tests use custom headers to trace request processing.
"""

import json
import os
import sys
import unittest
import uuid

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
ENVOY_URL = "http://localhost:8801"
OPENAI_ENDPOINT = "/v1/chat/completions"
DEFAULT_MODEL = "gemma3:27b"  # Use configured model


class EnvoyExtProcTest(SemanticRouterTestBase):
    """Test Envoy and ExtProc interaction."""

    def setUp(self):
        """Check if the Envoy server is running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that Envoy server is running and accepting requests",
        )

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

    def test_request_headers_propagation(self):
        """Test that request headers are correctly propagated through the ExtProc."""
        self.print_test_header(
            "Request Headers Propagation Test",
            "Verifies that request headers are correctly handled by the ExtProc",
        )

        trace_id = str(uuid.uuid4())

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "assistant", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "temperature": 0.7,
        }

        headers = {
            "Content-Type": "application/json",
            "X-Test-Trace-ID": trace_id,
            "X-Original-Model": DEFAULT_MODEL,
        }

        self.print_request_info(
            payload=payload,
            expectations=(
                "Expect: 2xx status code, Content-Type header in response, "
                "and potential model routing changes"
            ),
        )

        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}", headers=headers, json=payload, timeout=60
        )

        passed = (
            response.status_code < 400
            and "Content-Type" in response.headers
            and "model" in response.json()
        )

        response_json = response.json()
        self.print_response_info(
            response,
            {
                "Original Model": DEFAULT_MODEL,
                "Routed Model": response_json.get("model", "Not specified"),
                "Trace ID Preserved": response.headers.get("X-Test-Trace-ID")
                == trace_id,
            },
        )

        self.print_test_result(
            passed=passed,
            message=(
                "Headers processed correctly, model routing applied"
                if passed
                else "Issues with header processing or model routing"
            ),
        )

        self.assertLess(
            response.status_code,
            400,
            f"Request was rejected with status code {response.status_code}",
        )
        self.assertIn(
            "Content-Type", response.headers, "Response is missing Content-Type header"
        )
        self.assertIn("model", response_json, "Response is missing 'model' field")

    def test_extproc_override(self):
        """Test that the ExtProc can modify the request's target model."""
        self.print_test_header(
            "ExtProc Model Override Test",
            "Verifies that ExtProc correctly routes different query types to appropriate models",
        )

        test_cases = [
            {
                "name": "Math Query",
                "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?",
                "category": "math",
            },
            {
                "name": "Creative Writing Query",
                "content": "Write a short story about a space cat.",
                "category": "creative",
            },
        ]

        results = {}

        for test_case in test_cases:
            self.print_subtest_header(test_case["name"])

            trace_id = str(uuid.uuid4())

            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"You are an expert in {test_case['category']}.",
                    },
                    {"role": "user", "content": test_case["content"]},
                ],
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "X-Test-Trace-ID": trace_id,
                "X-Original-Model": DEFAULT_MODEL,
                "X-Test-Category": test_case["category"],
            }

            self.print_request_info(
                payload=payload,
                expectations=f"Expect: Query to be routed based on {test_case['category']} category",
            )

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=60,
            )

            response_json = response.json()
            results[test_case["name"]] = response_json.get("model", "unknown")

            self.print_response_info(
                response,
                {
                    "Category": test_case["category"],
                    "Original Model": DEFAULT_MODEL,
                    "Routed Model": results[test_case["name"]],
                },
            )

            passed = (
                response.status_code < 400 and results[test_case["name"]] != "unknown"
            )
            self.print_test_result(
                passed=passed,
                message=(
                    f"Successfully routed to model: {results[test_case['name']]}"
                    if passed
                    else f"Routing failed or returned unknown model"
                ),
            )

            self.assertLess(
                response.status_code,
                400,
                f"{test_case['name']} request failed with status {response.status_code}",
            )

        # Final summary of routing results
        if len(results) == 2:
            print("\nRouting Summary:")
            print(f"Math Query → {results['Math Query']}")
            print(f"Creative Writing Query → {results['Creative Writing Query']}")

    def test_extproc_body_modification(self):
        """Test that the ExtProc can modify the request and response bodies."""
        self.print_test_header(
            "ExtProc Body Modification Test",
            "Verifies that ExtProc can modify request and response bodies while preserving essential fields",
        )

        trace_id = str(uuid.uuid4())

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "assistant", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is quantum computing?"},
            ],
            "temperature": 0.7,
            "test_field": "should_be_preserved",
        }

        headers = {
            "Content-Type": "application/json",
            "X-Test-Trace-ID": trace_id,
            "X-Test-Body-Modification": "true",
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: Request processing with body modifications while preserving essential fields",
        )

        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}", headers=headers, json=payload, timeout=60
        )

        response_json = response.json()
        self.print_response_info(
            response,
            {
                "Original Model": DEFAULT_MODEL,
                "Final Model": response_json.get("model", "Not specified"),
                "Test Field Preserved": "test_field" in response_json,
            },
        )

        passed = response.status_code < 400 and "model" in response_json
        self.print_test_result(
            passed=passed,
            message=(
                "Request processed successfully with body modifications"
                if passed
                else "Issues with request processing or body modifications"
            ),
        )

        self.assertLess(
            response.status_code,
            400,
            f"Request was rejected with status code {response.status_code}",
        )


if __name__ == "__main__":
    unittest.main()
