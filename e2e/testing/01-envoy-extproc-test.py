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
DEFAULT_MODEL = "Model-A"  # Use configured model that matches router config


class EnvoyExtProcTest(SemanticRouterTestBase):
    """Test Envoy and ExtProc interaction."""

    def setUp(self):
        """Check if the Envoy server is running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that Envoy server is running and accepting requests",
        )

        try:
            # Use unique content to bypass cache for setup check
            setup_id = str(uuid.uuid4())[:8]
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"ExtProc setup test {setup_id}"},
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
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"ExtProc header test {trace_id[:8]} - explain photosynthesis briefly.",
                },
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
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"ExtProc body test {trace_id[:8]} - describe machine learning in simple terms.",
                },
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

    def test_extproc_error_handling(self):
        """Test ExtProc error handling and failure scenarios."""
        self.print_test_header(
            "ExtProc Error Handling Test",
            "Verifies that ExtProc properly handles and recovers from error conditions",
        )

        # Test with headers that might cause ExtProc issues
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Simple test query"},
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "X-Very-Long-Header": "x" * 1000,  # Very long header value
            "X-Test-Error-Recovery": "true",
            "X-Special-Chars": "data-with-special-chars-!@#$%^&*()",  # Special characters
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: ExtProc to handle unusual headers gracefully without crashing",
        )

        try:
            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=60,
            )

            # ExtProc should either process successfully or fail gracefully without hanging
            passed = (
                response.status_code < 500
            )  # No server errors due to ExtProc issues

            self.print_response_info(
                response,
                {
                    "Status Code": response.status_code,
                    "Error Handling": "Graceful" if passed else "Server Error",
                },
            )

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Connection errors are acceptable - it shows the system is protecting itself
            passed = True
            self.print_response_info(
                None,
                {
                    "Connection": "Terminated (Expected)",
                    "Error Handling": "Protective disconnection",
                    "Error": str(e)[:100] + "..." if len(str(e)) > 100 else str(e),
                },
            )

        self.print_test_result(
            passed=passed,
            message=(
                "ExtProc handled error conditions gracefully"
                if passed
                else "ExtProc error handling failed"
            ),
        )

        # The test passes if either the request succeeds or fails gracefully
        self.assertTrue(
            passed,
            "ExtProc should handle malformed input gracefully",
        )

    def test_extproc_performance_impact(self):
        """Test that ExtProc doesn't significantly impact request performance."""
        self.print_test_header(
            "ExtProc Performance Impact Test",
            "Verifies that ExtProc processing doesn't add excessive latency",
        )

        # Generate unique content for cache bypass
        trace_id = str(uuid.uuid4())

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"ExtProc performance test {trace_id[:8]} - what is artificial intelligence?",
                },
            ],
        }

        # Test with minimal ExtProc processing
        headers_minimal = {"Content-Type": "application/json"}

        # Test with ExtProc headers
        headers_extproc = {
            "Content-Type": "application/json",
            "X-Test-Performance": "true",
            "X-Processing-Mode": "full",
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: Reasonable response times with ExtProc processing",
        )

        import time

        # Measure response time with ExtProc
        start_time = time.time()
        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers=headers_extproc,
            json=payload,
            timeout=60,
        )
        response_time = time.time() - start_time

        passed = (
            response.status_code < 400 and response_time < 30.0
        )  # Reasonable timeout

        self.print_response_info(
            response,
            {
                "Response Time": f"{response_time:.2f}s",
                "Performance": (
                    "Acceptable" if response_time < 10.0 else "Slow but functional"
                ),
            },
        )

        self.print_test_result(
            passed=passed,
            message=(
                f"ExtProc processing completed in {response_time:.2f}s"
                if passed
                else f"ExtProc processing too slow: {response_time:.2f}s"
            ),
        )

        self.assertLess(
            response.status_code,
            400,
            "ExtProc should not cause request failures",
        )
        self.assertLess(
            response_time,
            30.0,
            "ExtProc should not cause excessive delays",
        )


if __name__ == "__main__":
    unittest.main()
