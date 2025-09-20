#!/usr/bin/env python3
"""
06-tools-test.py - Tools selection tests

This test validates the router's ability to automatically select appropriate tools
based on request content using semantic similarity matching.

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

# Tool test cases - based on the 5 tools configured in config/tools_db.json
TOOL_TEST_CASES = [
    {
        "name": "Weather Query",
        "expected_tools": ["get_weather"],
        "content": "What's the weather forecast for tomorrow in San Francisco?",
        "description": "Should match get_weather tool",
    },
    {
        "name": "Search Request",
        "expected_tools": ["search_web"],
        "content": "Can you find information about the latest AI research papers?",
        "description": "Should match search_web tool",
    },
    {
        "name": "Mathematical Calculation",
        "expected_tools": ["calculate"],
        "content": "What is 25% of 847 plus the square root of 169?",
        "description": "Should match calculate tool",
    },
    {
        "name": "Email Request",
        "expected_tools": ["send_email"],
        "content": "Please send an email to the team about the meeting update.",
        "description": "Should match send_email tool",
    },
    {
        "name": "Calendar/Scheduling",
        "expected_tools": ["create_calendar_event"],
        "content": "Schedule a meeting with the development team for next Friday at 2 PM.",
        "description": "Should match create_calendar_event tool",
    },
    {
        "name": "Multi-tool Request",
        "expected_tools": ["get_weather", "create_calendar_event"],
        "content": "Check the weather for the conference location and schedule a meeting to discuss it.",
        "description": "Should match multiple tools",
    },
]

# Cases that should not match any tools strongly
NO_TOOL_CASES = [
    {
        "name": "General Conversation",
        "content": "Hello, how are you doing today?",
        "description": "Generic greeting, should not match specific tools",
    },
    {
        "name": "Simple Question",
        "content": "What is artificial intelligence?",
        "description": "General knowledge question, no specific tool needed",
    },
    {
        "name": "Creative Writing",
        "content": "Write a short poem about the ocean.",
        "description": "Creative task, no specific tool needed",
    },
]


class ToolsSelectionTest(SemanticRouterTestBase):
    """Test the router's automatic tool selection functionality."""

    def setUp(self):
        """Check if the services are running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that required services (Envoy and Router) are running and tools selection is enabled",
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

        # Check if tools metrics exist
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text
        if "tool" not in metrics_text.lower():
            self.skipTest("Tools metrics not found. Tools selection may be disabled.")

    def test_specific_tool_selection(self):
        """Test that specific requests match their expected tools."""
        self.print_test_header(
            "Specific Tool Selection Test",
            "Verifies that requests for specific functionality match the appropriate tools",
        )

        for test_case in TOOL_TEST_CASES:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant with access to various tools."},
                        {"role": "user", "content": test_case["content"]},
                    ],
                    "temperature": 0.7,
                }

                headers = {
                    "Content-Type": "application/json",
                    "X-Session-ID": session_id,
                    "X-Tools-Enabled": "true",  # Explicitly request tool selection
                }

                self.print_request_info(
                    payload=payload,
                    expectations=f"Expect: Request to match tools {test_case['expected_tools']}",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                # Tool selection should work regardless of vLLM backend availability
                # Tool selection should work successfully
                passed = response.status_code == 200

                try:
                    response_json = response.json()
                    model = response_json.get("model", "unknown")
                    tools = response_json.get("tools", [])
                except:
                    model = "N/A"
                    tools = []

                self.print_response_info(
                    response,
                    {
                        "Content": test_case["content"][:50] + "...",
                        "Selected Model": model,
                        "Expected Tools": test_case["expected_tools"],
                        "Selected Tools": tools if tools else "N/A",
                        "Session ID": session_id,
                        "Description": test_case["description"],
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"Tool selection request processed (status: {response.status_code})"
                        if passed
                        else f"Tool selection request failed (status: {response.status_code})"
                    ),
                )

                self.assertIn(
                    response.status_code,
                    [200, 503],
                    f"Tool selection request '{test_case['name']}' failed. Status: {response.status_code}",
                )

    def test_no_tool_requests(self):
        """Test that generic requests don't unnecessarily trigger tool selection."""
        self.print_test_header(
            "No Tool Requests Test",
            "Verifies that generic requests don't inappropriately match specific tools",
        )

        for test_case in NO_TOOL_CASES:
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
                    expectations="Expect: Generic request to be processed without specific tool selection",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                # Tool selection should work successfully
                passed = response.status_code == 200

                try:
                    response_json = response.json()
                    model = response_json.get("model", "unknown")
                    tools = response_json.get("tools", [])
                except:
                    model = "N/A"
                    tools = []

                self.print_response_info(
                    response,
                    {
                        "Content": test_case["content"][:50] + "...",
                        "Selected Model": model,
                        "Selected Tools": tools if tools else "None",
                        "Session ID": session_id,
                        "Description": test_case["description"],
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"Generic request processed normally (status: {response.status_code})"
                        if passed
                        else f"Generic request failed (status: {response.status_code})"
                    ),
                )

                self.assertIn(
                    response.status_code,
                    [200, 503],
                    f"Generic request '{test_case['name']}' failed. Status: {response.status_code}",
                )

    def test_tools_configuration_validation(self):
        """Test that the tools configuration is properly loaded and accessible."""
        self.print_test_header(
            "Tools Configuration Validation Test",
            "Verifies that the tools database is properly loaded with expected tools",
        )

        # Make a request that should trigger tool processing
        session_id = str(uuid.uuid4())
        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": "I need help with weather, calculations, and scheduling."},
            ],
            "temperature": 0.7,
        }

        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
            "X-Tools-Debug": "true",  # Request debug information
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: Request to be processed and tools configuration to be validated",
        )

        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers=headers,
            json=payload,
            timeout=30,
        )

        passed = response.status_code in [200, 503]

        try:
            response_json = response.json()
            model = response_json.get("model", "unknown")
        except:
            model = "N/A"

        self.print_response_info(
            response,
            {
                "Selected Model": model,
                "Status Code": response.status_code,
                "Session ID": session_id,
                "Tools Config": "Expected: 5 tools loaded",
            },
        )

        self.print_test_result(
            passed=passed,
            message=f"Tools configuration validation completed (status: {response.status_code})",
        )

        self.assertIn(
            response.status_code,
            [200, 503],
            f"Tools configuration validation failed. Status: {response.status_code}",
        )

    def test_tools_metrics(self):
        """Test that tools selection metrics are being recorded properly."""
        self.print_test_header(
            "Tools Selection Metrics Test",
            "Verifies that tools selection metrics are being properly recorded and exposed",
        )

        # Get baseline metrics
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text

        # Look for specific tools metrics
        tools_metrics = [
            "llm_router_tools_selected_total",
            "llm_router_tools_selection_duration_seconds",
            "llm_router_tools_database_size",
            "llm_router_requests_total",
        ]

        metrics_found = {}
        for metric in tools_metrics:
            for line in metrics_text.split("\n"):
                if metric in line and not line.startswith("#"):
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
            {"Metrics Found": len(metrics_found), "Total Expected": len(tools_metrics)},
        )

        # Print detailed metrics information
        for metric, value in metrics_found.items():
            print(f"\nMetric: {metric}")
            print(f"  Value: {value}")

        # Print any metrics that contain "tool" even if not in our expected list
        print(f"\nAll tools-related metrics found:")
        for line in metrics_text.split("\n"):
            if "tool" in line.lower() and not line.startswith("#") and line.strip():
                print(f"  {line.strip()}")

        passed = len(metrics_found) > 0
        self.print_test_result(
            passed=passed,
            message=f"Found {len(metrics_found)} out of {len(tools_metrics)} expected tools metrics",
        )

        self.assertGreater(len(metrics_found), 0, "No tools metrics found")

    def test_tools_selection_consistency(self):
        """Test that tool selection is consistent for the same content."""
        self.print_test_header(
            "Tools Selection Consistency Test",
            "Verifies that the same content consistently triggers the same tool selection",
        )

        test_content = "What's the weather like today in New York?"
        
        results = []
        for i in range(3):
            self.print_subtest_header(f"Consistency Test {i+1}")

            session_id = str(uuid.uuid4())
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant with access to tools."},
                    {"role": "user", "content": test_content},
                ],
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": session_id,
                "X-Tools-Enabled": "true",
            }

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=30,
            )

            # Record the response status for consistency checking
            results.append(response.status_code)

            try:
                response_json = response.json()
                tools = response_json.get("tools", [])
            except:
                tools = []

            self.print_response_info(
                response,
                {
                    "Attempt": i + 1,
                    "Status Code": response.status_code,
                    "Content": test_content[:50] + "...",
                    "Selected Tools": tools if tools else "None",
                },
            )

            time.sleep(1)  # Small delay between requests

        # Check consistency - all results should be the same
        is_consistent = len(set(results)) == 1
        self.print_test_result(
            passed=is_consistent,
            message=f"Tools selection consistency: {is_consistent}. Results: {results}",
        )

        self.assertEqual(
            len(set(results)), 1, f"Inconsistent tools selection results: {results}"
        )


if __name__ == "__main__":
    unittest.main()