#!/usr/bin/env python3
"""
03-classification-api-test.py - Classification API tests

This test validates the standalone Classification API service,
which provides direct classification capabilities without LLM routing.
The API is separate from the ExtProc router and runs on port 8080.
"""

import json
import sys
import unittest

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
CLASSIFICATION_API_URL = "http://localhost:8080"
INTENT_ENDPOINT = "/api/v1/classify/intent"

# Test cases with expected categories based on config.e2e.yaml
INTENT_TEST_CASES = [
    {
        "name": "Math Query",
        "text": "Solve the quadratic equation x^2 + 5x + 6 = 0",
        "expected_category": "math",
    },
    {
        "name": "Computer Science Query",
        "text": "Write a Python function to implement a linked list",
        "expected_category": "computer science",
    },
    {
        "name": "Business Query",
        "text": "What are the key principles of supply chain management?",
        "expected_category": "business",
    },
    {
        "name": "History Query",
        "text": "Describe the main causes of World War I",
        "expected_category": "history",
    },
]


class ClassificationAPITest(SemanticRouterTestBase):
    """Test the standalone Classification API service."""

    def setUp(self):
        """Check if the Classification API is running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that Classification API is running and accepting requests",
        )

        try:
            # Test health endpoint
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
                "Cannot connect to Classification API on port 8080. Is it running? Start with: make run-router-e2e"
            )
        except requests.exceptions.Timeout:
            self.skipTest("Classification API health check timed out")

    def test_intent_classification(self):
        """Test that intent classification returns correct categories for different query types."""
        self.print_test_header(
            "Intent Classification Test",
            "Verifies that Classification API correctly classifies different query types",
        )

        for test_case in INTENT_TEST_CASES:
            self.print_subtest_header(test_case["name"])

            payload = {
                "text": test_case["text"],
                "options": {"return_probabilities": False},
            }

            self.print_request_info(
                payload=payload,
                expectations=f"Expect: Correctly classified as '{test_case['expected_category']}'",
            )

            response = requests.post(
                f"{CLASSIFICATION_API_URL}{INTENT_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )

            response_json = response.json()
            # The response may be nested in "classification" or at top level
            if "classification" in response_json:
                classification = response_json["classification"]
                actual_category = classification.get("category", "unknown")
                confidence = classification.get("confidence", 0.0)
            else:
                actual_category = response_json.get("category", "unknown")
                confidence = response_json.get("confidence", 0.0)

            # Check if classification is correct
            category_correct = actual_category == test_case["expected_category"]
            is_placeholder = actual_category == "general"
            passed = response.status_code == 200 and category_correct

            self.print_response_info(
                response,
                {
                    "Expected Category": test_case["expected_category"],
                    "Actual Category": actual_category,
                    "Confidence": f"{confidence:.2f}",
                    "Is Placeholder": "⚠️  Yes" if is_placeholder else "No",
                    "Category Match": "✅" if category_correct else "❌",
                },
            )

            if not category_correct:
                if is_placeholder:
                    failure_message = f"Classification failed: returned placeholder 'general' instead of '{test_case['expected_category']}'"
                else:
                    failure_message = (
                        f"Classification incorrect: expected '{test_case['expected_category']}', "
                        f"got '{actual_category}'"
                    )
            else:
                failure_message = None

            self.print_test_result(
                passed=passed,
                message=(
                    f"Correctly classified as '{actual_category}'"
                    if passed
                    else failure_message
                ),
            )

            self.assertEqual(
                response.status_code,
                200,
                f"Request failed with status {response.status_code}",
            )

            self.assertEqual(
                actual_category,
                test_case["expected_category"],
                f"{test_case['name']}: Expected category '{test_case['expected_category']}', got '{actual_category}'",
            )

    def test_batch_classification(self):
        """Test batch classification endpoint works correctly."""
        self.print_test_header(
            "Batch Classification Test",
            "Verifies that batch classification endpoint processes multiple texts correctly",
        )

        texts = [tc["text"] for tc in INTENT_TEST_CASES]
        expected_categories = [tc["expected_category"] for tc in INTENT_TEST_CASES]

        payload = {"texts": texts, "task_type": "intent"}

        self.print_request_info(
            payload={"texts": f"{len(texts)} texts", "task_type": "intent"},
            expectations=f"Expect: {len(texts)} classifications matching expected categories",
        )

        response = requests.post(
            f"{CLASSIFICATION_API_URL}/api/v1/classify/batch",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        response_json = response.json()
        results = response_json.get("results", [])

        self.print_response_info(
            response,
            {
                "Total Texts": len(texts),
                "Results Count": len(results),
                "Processing Time (ms)": response_json.get("processing_time_ms", 0),
            },
        )

        passed = response.status_code == 200 and len(results) == len(texts)

        self.print_test_result(
            passed=passed,
            message=(
                f"Successfully classified {len(results)} texts"
                if passed
                else f"Batch classification failed or returned wrong count"
            ),
        )

        self.assertEqual(response.status_code, 200, "Batch request failed")
        self.assertEqual(len(results), len(texts), "Result count mismatch")


if __name__ == "__main__":
    unittest.main()
