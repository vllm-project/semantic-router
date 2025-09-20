#!/usr/bin/env python3
"""
09-error-handling-test.py - Error handling and edge cases tests

This test validates the router's ability to handle various error conditions,
malformed requests, and edge cases gracefully.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
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

# Malformed request test cases
MALFORMED_REQUEST_CASES = [
    {
        "name": "Empty Request Body",
        "payload": {},
        "expected_status_range": (400, 499),
        "description": "Empty JSON object should return 4xx error",
    },
    {
        "name": "Missing Model Field",
        "payload": {
            "messages": [{"role": "user", "content": "Hello"}],
        },
        "expected_status_range": (400, 499),
        "description": "Missing required model field",
    },
    {
        "name": "Missing Messages Field",
        "payload": {
            "model": DEFAULT_MODEL,
        },
        "expected_status_range": (400, 499),
        "description": "Missing required messages field",
    },
    {
        "name": "Empty Messages Array",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [],
        },
        "expected_status_range": (400, 499),
        "description": "Empty messages array should be rejected",
    },
    {
        "name": "Invalid Message Format",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [{"invalid": "message"}],
        },
        "expected_status_range": (400, 499),
        "description": "Message without role/content should be rejected",
    },
    {
        "name": "Invalid Temperature",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": "invalid",
        },
        "expected_status_range": (400, 499),
        "description": "Non-numeric temperature should be rejected",
    },
    {
        "name": "Extremely Large Temperature",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 999.9,
        },
        "expected_status_range": (400, 499),
        "description": "Temperature out of valid range should be rejected",
    },
]

# Edge case test cases
EDGE_CASE_TESTS = [
    {
        "name": "Very Long Message",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": "A" * 10000}  # 10KB message
            ],
        },
        "expected_status_range": (200, 200),  # Should be processed successfully - no 503 accepted
        "description": "Very long message should be handled gracefully",
    },
    {
        "name": "Many Messages",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": f"Message {i}"}
                for i in range(100)  # 100 messages
            ],
        },
        "expected_status": 200,
        "description": "Large number of messages should be handled",
    },
    {
        "name": "Unicode Content",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": "Hello 世界 🌍 Здравствуй мир"}
            ],
        },
        "expected_status": 200,
        "description": "Unicode characters should be handled correctly",
    },
    {
        "name": "Zero Temperature",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0,
        },
        "expected_status": 200,
        "description": "Zero temperature should be valid",
    },
    {
        "name": "Maximum Valid Temperature",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 2.0,
        },
        "expected_status": 200,
        "description": "Maximum valid temperature should work",
    },
    {
        "name": "Special Characters in Content",
        "payload": {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "user", "content": "Test with \"quotes\" and 'apostrophes' and \n newlines \t tabs"}
            ],
        },
        "expected_status": 200,
        "description": "Special characters should be handled",
    },
]

# Timeout and connection test cases
TIMEOUT_TEST_CASES = [
    {
        "name": "Very Short Timeout",
        "timeout": 0.1,  # 100ms timeout
        "description": "Very short timeout should be handled gracefully",
    },
    {
        "name": "Medium Timeout",
        "timeout": 5.0,  # 5 second timeout
        "description": "Medium timeout should allow processing",
    },
]


class ErrorHandlingEdgeCasesTest(SemanticRouterTestBase):
    """Test error handling and edge cases in the semantic router."""

    def setUp(self):
        """Check if the services are running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that required services (Envoy and Router) are running for error handling tests",
        )

        # Check Envoy with a simple request
        try:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "assistant", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "test"},
                ],
            }

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )

            # For error handling tests, we don't skip on 5xx errors since we're testing error handling
            self.print_response_info(response, {"Setup": "Basic connectivity check"})

        except requests.exceptions.ConnectionError:
            self.skipTest("Cannot connect to Envoy server. Is it running?")

        # Check router metrics endpoint
        try:
            response = requests.get(ROUTER_METRICS_URL, timeout=2)
            if response.status_code != 200:
                self.skipTest(
                    "Router metrics server is not responding. Is the router running?"
                )

        except requests.exceptions.ConnectionError:
            self.skipTest(
                "Cannot connect to router metrics server. Is the router running?"
            )

    def test_malformed_requests(self):
        """Test that malformed requests are properly rejected with appropriate error codes."""
        self.print_test_header(
            "Malformed Requests Test",
            "Verifies that malformed requests are rejected with appropriate 4xx error codes",
        )

        for test_case in MALFORMED_REQUEST_CASES:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                headers = {
                    "Content-Type": "application/json",
                    "X-Session-ID": session_id,
                }

                self.print_request_info(
                    payload=test_case["payload"],
                    expectations=f"Expect: {test_case['expected_status_range'][0]}-{test_case['expected_status_range'][1]} status code",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    json=test_case["payload"],
                    timeout=30,
                )

                expected_status = test_case["expected_status"]
                passed = response.status_code == expected_status

                try:
                    response_json = response.json()
                    error_info = response_json.get("error", {})
                except:
                    error_info = "Non-JSON response"

                self.print_response_info(
                    response,
                    {
                        "Payload": str(test_case["payload"])[:100] + "...",
                        "Expected Status": expected_status,
                        "Actual Status": response.status_code,
                        "Error Info": str(error_info)[:100] + "..." if len(str(error_info)) > 100 else str(error_info),
                        "Session ID": session_id,
                        "Description": test_case["description"],
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"Malformed request properly rejected (status: {response.status_code})"
                        if passed
                        else f"Unexpected status code: {response.status_code} (expected {expected_status})"
                    ),
                )

                self.assertTrue(
                    passed,
                    f"Malformed request '{test_case['name']}' returned status {response.status_code}, "
                    f"expected {expected_status}",
                )

    def test_edge_cases(self):
        """Test edge cases that should be handled gracefully."""
        self.print_test_header(
            "Edge Cases Test",
            "Verifies that edge cases like very long messages and special characters are handled gracefully",
        )

        for test_case in EDGE_CASE_TESTS:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                headers = {
                    "Content-Type": "application/json",
                    "X-Session-ID": session_id,
                }

                self.print_request_info(
                    payload={**test_case["payload"], "messages": [{"content": f"[{len(str(test_case['payload']['messages']))} chars]"}]},  # Show length instead of full content
                    expectations=f"Expect: {test_case['expected_status']} status code",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    json=test_case["payload"],
                    timeout=30,  # Longer timeout for edge cases
                )

                expected_status = test_case["expected_status"]
                passed = response.status_code == expected_status

                try:
                    response_json = response.json()
                    model = response_json.get("model", "unknown")
                except:
                    model = "N/A"

                self.print_response_info(
                    response,
                    {
                        "Test Case": test_case["name"],
                        "Expected Status": expected_status,
                        "Actual Status": response.status_code,
                        "Selected Model": model,
                        "Session ID": session_id,
                        "Description": test_case["description"],
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"Edge case handled gracefully (status: {response.status_code})"
                        if passed
                        else f"Edge case not handled properly: {response.status_code} (expected {min_status}-{max_status})"
                    ),
                )

                self.assertTrue(
                    passed,
                    f"Edge case '{test_case['name']}' returned status {response.status_code}, "
                    f"expected {expected_status}",
                )

    def test_timeout_handling(self):
        """Test various timeout scenarios."""
        self.print_test_header(
            "Timeout Handling Test",
            "Verifies that the router handles various timeout scenarios appropriately",
        )

        for test_case in TIMEOUT_TEST_CASES:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {"role": "user", "content": "This is a test message for timeout handling."}
                    ],
                }

                headers = {
                    "Content-Type": "application/json",
                    "X-Session-ID": session_id,
                }

                self.print_request_info(
                    payload=payload,
                    expectations=f"Expect: Appropriate handling of {test_case['timeout']}s timeout",
                )

                start_time = time.time()
                try:
                    response = requests.post(
                        f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                        headers=headers,
                        json=payload,
                        timeout=test_case["timeout"],
                    )
                    
                    elapsed_time = time.time() - start_time
                    passed = True
                    result_message = f"Request completed in {elapsed_time:.2f}s (status: {response.status_code})"

                except requests.exceptions.Timeout:
                    elapsed_time = time.time() - start_time
                    passed = True  # Timeout is an expected behavior for very short timeouts
                    result_message = f"Request timed out after {elapsed_time:.2f}s (expected for short timeout)"
                    response = None

                except Exception as e:
                    elapsed_time = time.time() - start_time
                    passed = False
                    result_message = f"Unexpected error after {elapsed_time:.2f}s: {str(e)}"
                    response = None

                if response:
                    self.print_response_info(
                        response,
                        {
                            "Timeout": f"{test_case['timeout']}s",
                            "Elapsed Time": f"{elapsed_time:.2f}s",
                            "Status Code": response.status_code,
                            "Session ID": session_id,
                            "Description": test_case["description"],
                        },
                    )
                else:
                    print(f"Response: {result_message}")

                self.print_test_result(
                    passed=passed,
                    message=result_message,
                )

                self.assertTrue(passed, f"Timeout test '{test_case['name']}' failed: {result_message}")

    def test_invalid_content_types(self):
        """Test requests with invalid content types."""
        self.print_test_header(
            "Invalid Content Types Test",
            "Verifies that requests with invalid content types are properly rejected",
        )

        test_cases = [
            {
                "name": "Plain Text Content-Type",
                "content_type": "text/plain",
                "body": '{"model": "' + DEFAULT_MODEL + '", "messages": [{"role": "user", "content": "test"}]}',
            },
            {
                "name": "XML Content-Type",
                "content_type": "application/xml",
                "body": '{"model": "' + DEFAULT_MODEL + '", "messages": [{"role": "user", "content": "test"}]}',
            },
            {
                "name": "Missing Content-Type",
                "content_type": None,
                "body": '{"model": "' + DEFAULT_MODEL + '", "messages": [{"role": "user", "content": "test"}]}',
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                headers = {"X-Session-ID": session_id}
                
                if test_case["content_type"]:
                    headers["Content-Type"] = test_case["content_type"]

                self.print_request_info(
                    payload={"content_type": test_case["content_type"], "body_length": len(test_case["body"])},
                    expectations="Expect: 4xx error for invalid content type",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    data=test_case["body"],
                    timeout=30,
                )

                # Invalid content types should typically return 4xx errors
                passed = response.status_code >= 400

                self.print_response_info(
                    response,
                    {
                        "Content-Type": test_case["content_type"] or "None",
                        "Status Code": response.status_code,
                        "Session ID": session_id,
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=f"Invalid content type properly rejected (status: {response.status_code})" if passed else f"Invalid content type not rejected (status: {response.status_code})",
                )

                self.assertGreaterEqual(
                    response.status_code, 400,
                    f"Invalid content type '{test_case['name']}' should return 4xx error, got {response.status_code}",
                )

    def test_error_response_format(self):
        """Test that error responses follow a consistent format."""
        self.print_test_header(
            "Error Response Format Test",
            "Verifies that error responses follow a consistent, well-structured format",
        )

        # Make a request that should trigger an error
        session_id = str(uuid.uuid4())
        payload = {}  # Empty payload should trigger an error

        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: 4xx error with properly formatted error response",
        )

        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers=headers,
            json=payload,
            timeout=30,
        )

        passed = response.status_code >= 400

        try:
            response_json = response.json()
            has_error_field = "error" in response_json
            error_structure_valid = False
            
            if has_error_field:
                error = response_json["error"]
                # Check if error has expected fields (message, type, etc.)
                error_structure_valid = isinstance(error, (dict, str))
                
        except json.JSONDecodeError:
            response_json = None
            has_error_field = False
            error_structure_valid = False

        self.print_response_info(
            response,
            {
                "Status Code": response.status_code,
                "Has JSON Response": response_json is not None,
                "Has Error Field": has_error_field,
                "Error Structure Valid": error_structure_valid,
                "Session ID": session_id,
            },
        )

        if response_json:
            print(f"\nError response structure:")
            print(f"  {json.dumps(response_json, indent=2)[:200]}...")

        self.print_test_result(
            passed=passed and response_json is not None,
            message=f"Error response format: status {response.status_code}, JSON: {response_json is not None}, error field: {has_error_field}",
        )

        self.assertGreaterEqual(response.status_code, 400, "Should return 4xx error for empty payload")
        self.assertIsNotNone(response_json, "Error response should be valid JSON")


if __name__ == "__main__":
    unittest.main()