import json
import unittest
from typing import Any, Dict, Optional


class SemanticRouterTestBase(unittest.TestCase):
    def print_test_header(self, test_name: str, description: Optional[str] = None):
        """Print a formatted header for each test."""
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        if description:
            print(f"Description: {description}")
        print(f"{'='*80}")

    def print_request_info(self, payload: Dict[str, Any], expectations: str):
        """Print information about the request being made."""
        print("\nRequest Details:")
        print("-" * 40)
        print("Payload:")
        print(json.dumps(payload, indent=2))
        print("\nExpectations:")
        print(expectations)
        print("-" * 40)

    def print_response_info(
        self, response, additional_info: Optional[Dict[str, Any]] = None
    ):
        """Print information about the response received."""
        print("\nResponse Details:")
        print("-" * 40)
        print(f"Status Code: {response.status_code}")

        try:
            response_json = response.json()
            print("\nResponse Body:")
            print(json.dumps(response_json, indent=2))
        except (json.JSONDecodeError, AttributeError):
            print(f"Raw Response: {response}")

        if additional_info:
            print("\nAdditional Information:")
            for key, value in additional_info.items():
                print(f"{key}: {value}")
        print("-" * 40)

    def print_test_result(self, passed: bool, message: str = ""):
        """Print the test result with a clear pass/fail indicator."""
        result = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\nTest Result: {result}")
        if message:
            print(f"Details: {message}")
        print("-" * 40)

    def print_subtest_header(self, name: str):
        """Print a header for subtests within a test case."""
        print(f"\n{'-'*40}")
        print(f"Subtest: {name}")
        print(f"{'-'*40}")
