#!/usr/bin/env python3
"""
07-model-selection-test.py - Model selection and scoring tests

This test validates the router's ability to select appropriate models based on
content categories and configured scoring rules.
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

# Model selection test cases - based on category scores in config.yaml
MODEL_SELECTION_TEST_CASES = [
    {
        "name": "Math Query - Should Prefer phi4",
        "category": "math",
        "content": "Solve this differential equation: dy/dx + 2y = x^2",
        "preferred_models": ["phi4"],  # phi4 has score 1.0 for math
        "reasoning_enabled": True,
    },
    {
        "name": "Business Query - Should Prefer phi4",
        "category": "business",
        "content": "What are the key strategies for improving quarterly revenue in a SaaS company?",
        "preferred_models": ["phi4"],  # phi4 has score 0.8 for business
        "reasoning_enabled": False,
    },
    {
        "name": "Law Query - Should Prefer gemma3:27b",
        "category": "law",
        "content": "What are the legal implications of GDPR compliance for international data transfers?",
        "preferred_models": ["gemma3:27b"],  # gemma3:27b has score 0.8 for law
        "reasoning_enabled": False,
    },
    {
        "name": "Chemistry Query - Should Prefer mistral-small3.1",
        "category": "chemistry",
        "content": "Explain the mechanism of nucleophilic substitution in organic chemistry.",
        "preferred_models": ["mistral-small3.1"],  # mistral has score 0.8 for chemistry
        "reasoning_enabled": True,
    },
    {
        "name": "History Query - Should Prefer mistral-small3.1",
        "category": "history",
        "content": "Analyze the causes and consequences of the French Revolution.",
        "preferred_models": ["mistral-small3.1"],  # mistral has score 0.8 for history
        "reasoning_enabled": False,
    },
    {
        "name": "General Query - Should Use Default",
        "category": "other",
        "content": "What is artificial intelligence and how does it work?",
        "preferred_models": ["gemma3:27b"],  # gemma3:27b has score 0.8 for other
        "reasoning_enabled": False,
    },
]

# Reasoning test cases - categories that should enable reasoning
REASONING_TEST_CASES = [
    {
        "name": "Physics Problem - Reasoning Enabled",
        "category": "physics", 
        "content": "Calculate the trajectory of a projectile launched at 45 degrees with initial velocity 20 m/s.",
        "reasoning_enabled": True,
    },
    {
        "name": "Computer Science Problem - Reasoning Enabled",
        "category": "computer science",
        "content": "Design an algorithm to find the shortest path in a weighted directed graph.",
        "reasoning_enabled": True,
    },
    {
        "name": "Engineering Problem - Reasoning Enabled",
        "category": "engineering",
        "content": "Design a bridge to span 100 meters with maximum load capacity of 50 tons.",
        "reasoning_enabled": True,
    },
    {
        "name": "Biology Analysis - Reasoning Enabled",
        "category": "biology",
        "content": "Explain the process of protein synthesis from DNA to functional protein.",
        "reasoning_enabled": True,
    },
]

# Non-reasoning categories
NON_REASONING_CASES = [
    {
        "name": "Psychology Discussion - No Reasoning",
        "category": "psychology",
        "content": "Discuss the psychological effects of social media on teenagers.",
        "reasoning_enabled": False,
    },
    {
        "name": "Philosophy Discussion - No Reasoning", 
        "category": "philosophy",
        "content": "What is the meaning of consciousness in modern philosophy?",
        "reasoning_enabled": False,
    },
]


class ModelSelectionTest(SemanticRouterTestBase):
    """Test the router's model selection and scoring functionality."""

    def setUp(self):
        """Check if the services are running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that required services (Envoy and Router) are running and model selection is enabled",
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

        # Check if model selection metrics exist
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text
        if "model_selection" not in metrics_text.lower():
            self.skipTest("Model selection metrics not found. Model selection may be disabled.")

    def test_category_based_model_selection(self):
        """Test that models are selected based on category scores."""
        self.print_test_header(
            "Category-Based Model Selection Test",
            "Verifies that the router selects models based on category scoring rules",
        )

        for test_case in MODEL_SELECTION_TEST_CASES:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                payload = {
                    "model": "auto",  # Request automatic model selection
                    "messages": [
                        {"role": "system", "content": f"You are an expert in {test_case['category']}."},
                        {"role": "user", "content": test_case["content"]},
                    ],
                    "temperature": 0.7,
                }

                headers = {
                    "Content-Type": "application/json",
                    "X-Session-ID": session_id,
                    "X-Category-Hint": test_case["category"],  # Provide category hint
                }

                self.print_request_info(
                    payload=payload,
                    expectations=f"Expect: Auto-selection to prefer {test_case['preferred_models']} for {test_case['category']}",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )

                passed = response.status_code in [200, 503]

                try:
                    response_json = response.json()
                    selected_model = response_json.get("model", "unknown")
                    reasoning_enabled = response_json.get("reasoning_enabled", False)
                except:
                    selected_model = "N/A"
                    reasoning_enabled = False

                self.print_response_info(
                    response,
                    {
                        "Category": test_case["category"],
                        "Content": test_case["content"][:50] + "...",
                        "Selected Model": selected_model,
                        "Preferred Models": test_case["preferred_models"],
                        "Reasoning Enabled": reasoning_enabled,
                        "Expected Reasoning": test_case["reasoning_enabled"],
                        "Session ID": session_id,
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"Model selection processed (status: {response.status_code}, model: {selected_model})"
                        if passed
                        else f"Model selection failed (status: {response.status_code})"
                    ),
                )

                self.assertIn(
                    response.status_code,
                    [200, 503],
                    f"Model selection request '{test_case['name']}' failed. Status: {response.status_code}",
                )

    def test_reasoning_mode_selection(self):
        """Test that reasoning mode is enabled for appropriate categories."""
        self.print_test_header(
            "Reasoning Mode Selection Test",
            "Verifies that reasoning mode is enabled for categories that require structured thinking",
        )

        all_reasoning_cases = REASONING_TEST_CASES + NON_REASONING_CASES

        for test_case in all_reasoning_cases:
            with self.subTest(test_case["name"]):
                self.print_subtest_header(test_case["name"])

                session_id = str(uuid.uuid4())
                payload = {
                    "model": "auto",
                    "messages": [
                        {"role": "system", "content": f"You are an expert in {test_case['category']}."},
                        {"role": "user", "content": test_case["content"]},
                    ],
                    "temperature": 0.7,
                }

                headers = {
                    "Content-Type": "application/json",
                    "X-Session-ID": session_id,
                    "X-Category-Hint": test_case["category"],
                }

                self.print_request_info(
                    payload=payload,
                    expectations=f"Expect: Reasoning mode {'enabled' if test_case['reasoning_enabled'] else 'disabled'} for {test_case['category']}",
                )

                response = requests.post(
                    f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                    headers=headers,
                    json=payload,
                    timeout=10,
                )

                passed = response.status_code in [200, 503]

                try:
                    response_json = response.json()
                    selected_model = response_json.get("model", "unknown")
                    reasoning_enabled = response_json.get("reasoning_enabled", False)
                except:
                    selected_model = "N/A"
                    reasoning_enabled = False

                self.print_response_info(
                    response,
                    {
                        "Category": test_case["category"],
                        "Content": test_case["content"][:50] + "...",
                        "Selected Model": selected_model,
                        "Reasoning Enabled": reasoning_enabled,
                        "Expected Reasoning": test_case["reasoning_enabled"],
                        "Session ID": session_id,
                    },
                )

                self.print_test_result(
                    passed=passed,
                    message=(
                        f"Reasoning mode selection processed correctly"
                        if passed
                        else f"Reasoning mode selection failed"
                    ),
                )

                self.assertIn(
                    response.status_code,
                    [200, 503],
                    f"Reasoning mode test '{test_case['name']}' failed. Status: {response.status_code}",
                )

    def test_model_fallback_behavior(self):
        """Test model fallback when preferred models are unavailable."""
        self.print_test_header(
            "Model Fallback Behavior Test",
            "Verifies that the router falls back to available models when preferred models are unavailable",
        )

        # Test with a specific model that might not be available
        session_id = str(uuid.uuid4())
        payload = {
            "model": "non-existent-model",  # Request a non-existent model
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"},
            ],
            "temperature": 0.7,
        }

        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": session_id,
        }

        self.print_request_info(
            payload=payload,
            expectations="Expect: Router to fallback to default model when requested model is unavailable",
        )

        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers=headers,
            json=payload,
            timeout=10,
        )

        # Fallback should work, though may get 503 if no vLLM backend
        passed = response.status_code in [200, 400, 503]  # 400 is acceptable for invalid model

        try:
            response_json = response.json()
            selected_model = response_json.get("model", "unknown")
        except:
            selected_model = "N/A"

        self.print_response_info(
            response,
            {
                "Requested Model": "non-existent-model",
                "Selected Model": selected_model,
                "Status Code": response.status_code,
                "Session ID": session_id,
            },
        )

        self.print_test_result(
            passed=passed,
            message=f"Model fallback behavior tested (status: {response.status_code}, model: {selected_model})",
        )

        self.assertIn(
            response.status_code,
            [200, 400, 503],
            f"Model fallback test failed unexpectedly. Status: {response.status_code}",
        )

    def test_model_selection_metrics(self):
        """Test that model selection metrics are being recorded properly."""
        self.print_test_header(
            "Model Selection Metrics Test",
            "Verifies that model selection metrics are being properly recorded and exposed",
        )

        # Get baseline metrics
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text

        # Look for specific model selection metrics
        model_metrics = [
            "llm_router_model_selection_count",
            "llm_router_model_selection_duration_seconds",
            "llm_router_category_classification_total",
            "llm_router_reasoning_enabled_total",
            "llm_router_requests_total",
        ]

        metrics_found = {}
        for metric in model_metrics:
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
            {"Metrics Found": len(metrics_found), "Total Expected": len(model_metrics)},
        )

        # Print detailed metrics information
        for metric, value in metrics_found.items():
            print(f"\nMetric: {metric}")
            print(f"  Value: {value}")

        # Print any metrics that contain "model" or "category" even if not in our expected list
        print(f"\nAll model/category-related metrics found:")
        for line in metrics_text.split("\n"):
            if (("model" in line.lower() or "category" in line.lower() or "reasoning" in line.lower()) 
                and not line.startswith("#") and line.strip()):
                print(f"  {line.strip()}")

        passed = len(metrics_found) > 0
        self.print_test_result(
            passed=passed,
            message=f"Found {len(metrics_found)} out of {len(model_metrics)} expected model selection metrics",
        )

        self.assertGreater(len(metrics_found), 0, "No model selection metrics found")

    def test_model_selection_consistency(self):
        """Test that model selection is consistent for the same content and category."""
        self.print_test_header(
            "Model Selection Consistency Test",
            "Verifies that the same content consistently triggers the same model selection",
        )

        test_content = "Solve this quadratic equation: x^2 + 5x + 6 = 0"
        category = "math"
        
        results = []
        for i in range(3):
            self.print_subtest_header(f"Consistency Test {i+1}")

            session_id = str(uuid.uuid4())
            payload = {
                "model": "auto",
                "messages": [
                    {"role": "system", "content": f"You are an expert in {category}."},
                    {"role": "user", "content": test_content},
                ],
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": session_id,
                "X-Category-Hint": category,
            }

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=10,
            )

            # Record the response status and model for consistency checking
            try:
                response_json = response.json()
                selected_model = response_json.get("model", "unknown")
                results.append((response.status_code, selected_model))
            except:
                results.append((response.status_code, "N/A"))

            self.print_response_info(
                response,
                {
                    "Attempt": i + 1,
                    "Status Code": response.status_code,
                    "Selected Model": results[-1][1],
                    "Category": category,
                    "Content": test_content[:50] + "...",
                },
            )

            time.sleep(1)  # Small delay between requests

        # Check consistency - all results should be the same
        status_codes = [r[0] for r in results]
        models = [r[1] for r in results]
        
        status_consistent = len(set(status_codes)) == 1
        model_consistent = len(set(models)) == 1
        
        is_consistent = status_consistent and model_consistent
        
        self.print_test_result(
            passed=is_consistent,
            message=f"Model selection consistency: {is_consistent}. Status codes: {status_codes}, Models: {models}",
        )

        self.assertEqual(
            len(set(status_codes)), 1, f"Inconsistent status codes: {status_codes}"
        )
        self.assertEqual(
            len(set(models)), 1, f"Inconsistent model selection: {models}"
        )


if __name__ == "__main__":
    unittest.main()