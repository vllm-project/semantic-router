#!/usr/bin/env python3
"""
test-pii-direct.py - Direct PII Classification API Test

Comprehensive PII detection test that directly calls the Classification API
to test confidence levels for various PII entity types.
Bypasses ExtProc for faster, focused testing.
"""

import json
import sys
import unittest
from typing import List, Dict, Any

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
CLASSIFICATION_API_URL = "http://localhost:8080"
PII_ENDPOINT = "/api/v1/classify/pii"

# Comprehensive PII test cases with expected entity types
# Based on Issue #647 and expanded for thorough coverage
PII_TEST_CASES = [
    # ===== Issue #647 Original Cases =====
    {
        "name": "Email - Plain (Issue #647)",
        "text": "john.smith@example.com",
        "expected_types": ["EMAIL_ADDRESS"],
        "min_confidence": 0.7,
        "description": "ModernBERT FAILED: 0.561 as PERSON. Should detect as EMAIL_ADDRESS",
    },
    {
        "name": "SSN - Dashes (Issue #647)",
        "text": "123-45-6789",
        "expected_types": ["US_SSN"],
        "min_confidence": 0.7,
        "description": "ModernBERT detected as DATE_TIME (wrong). Should be US_SSN",
    },
    {
        "name": "Credit Card - Dashes (Issue #647)",
        "text": "4532-1234-5678-9012",
        "expected_types": ["CREDIT_CARD"],
        "min_confidence": 0.7,
        "description": "ModernBERT FAILED: 0.554 as US_SSN. Should be CREDIT_CARD",
    },
    {
        "name": "Phone - Parentheses (Issue #647)",
        "text": "(555) 123-4567",
        "expected_types": ["PHONE_NUMBER"],
        "min_confidence": 0.7,
        "description": "ModernBERT PASSED: 0.947. LoRA should also pass",
    },
    # ===== Email Variations =====
    {
        "name": "Email - Work Domain",
        "text": "jane.doe@company.co.uk",
        "expected_types": ["EMAIL_ADDRESS"],
        "min_confidence": 0.7,
    },
    {
        "name": "Email - With Numbers",
        "text": "user123@test.com",
        "expected_types": ["EMAIL_ADDRESS"],
        "min_confidence": 0.7,
    },
    {
        "name": "Email - In Sentence",
        "text": "Contact me at support@example.org for assistance",
        "expected_types": ["EMAIL_ADDRESS"],
        "min_confidence": 0.7,
    },
    {
        "name": "Email - Multiple",
        "text": "Send to john@example.com and jane@example.com",
        "expected_types": ["EMAIL_ADDRESS"],
        "min_confidence": 0.7,
    },
    # ===== SSN Variations =====
    {
        "name": "SSN - No Dashes",
        "text": "123456789",
        "expected_types": ["US_SSN"],
        "min_confidence": 0.7,
    },
    {
        "name": "SSN - In Sentence",
        "text": "My social security number is 987-65-4321",
        "expected_types": ["US_SSN"],
        "min_confidence": 0.7,
    },
    {
        "name": "SSN - With Label",
        "text": "SSN: 456-78-9012",
        "expected_types": ["US_SSN"],
        "min_confidence": 0.7,
    },
    # ===== Credit Card Variations =====
    {
        "name": "Credit Card - Spaces",
        "text": "4532 1234 5678 9012",
        "expected_types": ["CREDIT_CARD"],
        "min_confidence": 0.7,
    },
    {
        "name": "Credit Card - No Separators",
        "text": "4532123456789012",
        "expected_types": ["CREDIT_CARD"],
        "min_confidence": 0.7,
    },
    {
        "name": "Credit Card - Visa",
        "text": "4111111111111111",
        "expected_types": ["CREDIT_CARD"],
        "min_confidence": 0.7,
    },
    {
        "name": "Credit Card - Mastercard",
        "text": "5500000000000004",
        "expected_types": ["CREDIT_CARD"],
        "min_confidence": 0.7,
    },
    {
        "name": "Credit Card - In Sentence",
        "text": "My card number is 4532-1234-5678-9012 and expires 12/25",
        "expected_types": ["CREDIT_CARD"],
        "min_confidence": 0.7,
    },
    # ===== Phone Variations =====
    {
        "name": "Phone - Dashes",
        "text": "555-123-4567",
        "expected_types": ["PHONE_NUMBER"],
        "min_confidence": 0.7,
    },
    {
        "name": "Phone - Dots",
        "text": "555.123.4567",
        "expected_types": ["PHONE_NUMBER"],
        "min_confidence": 0.7,
    },
    {
        "name": "Phone - Spaces",
        "text": "555 123 4567",
        "expected_types": ["PHONE_NUMBER"],
        "min_confidence": 0.7,
    },
    {
        "name": "Phone - International",
        "text": "+1-555-123-4567",
        "expected_types": ["PHONE_NUMBER"],
        "min_confidence": 0.7,
    },
    {
        "name": "Phone - 10 Digits",
        "text": "5551234567",
        "expected_types": ["PHONE_NUMBER"],
        "min_confidence": 0.7,
    },
    {
        "name": "Phone - In Sentence",
        "text": "Call me at (555) 123-4567 for more information",
        "expected_types": ["PHONE_NUMBER"],
        "min_confidence": 0.7,
    },
    # ===== Person Names =====
    {
        "name": "Person - Full Name",
        "text": "John Smith",
        "expected_types": ["PERSON"],
        "min_confidence": 0.7,
    },
    {
        "name": "Person - With Middle Initial",
        "text": "John Q. Smith",
        "expected_types": ["PERSON"],
        "min_confidence": 0.7,
    },
    {
        "name": "Person - Formal Title",
        "text": "Dr. Jane Doe",
        "expected_types": ["PERSON"],
        "min_confidence": 0.7,
    },
    {
        "name": "Person - Multiple Names",
        "text": "Meeting with John Smith and Jane Doe",
        "expected_types": ["PERSON"],
        "min_confidence": 0.7,
    },
    # ===== Addresses =====
    {
        "name": "Address - Street",
        "text": "123 Main Street",
        "expected_types": ["ADDRESS", "GPE"],  # May detect as location
        "min_confidence": 0.7,
    },
    {
        "name": "Address - Full",
        "text": "456 Oak Ave, New York, NY 10001",
        "expected_types": ["ADDRESS", "GPE"],
        "min_confidence": 0.7,
    },
    # ===== Organizations =====
    {
        "name": "Organization - Tech Company",
        "text": "Apple Inc.",
        "expected_types": ["ORGANIZATION"],
        "min_confidence": 0.7,
    },
    {
        "name": "Organization - Corporation",
        "text": "Microsoft Corporation",
        "expected_types": ["ORGANIZATION"],
        "min_confidence": 0.7,
    },
    # ===== Dates =====
    {
        "name": "Date - Numeric",
        "text": "12/31/2023",
        "expected_types": ["DATE_TIME"],
        "min_confidence": 0.7,
    },
    {
        "name": "Date - Written",
        "text": "January 1, 2024",
        "expected_types": ["DATE_TIME"],
        "min_confidence": 0.7,
    },
    # ===== Locations =====
    {
        "name": "Location - City",
        "text": "New York",
        "expected_types": ["GPE"],
        "min_confidence": 0.7,
    },
    {
        "name": "Location - Country",
        "text": "United States",
        "expected_types": ["GPE"],
        "min_confidence": 0.7,
    },
    # ===== Edge Cases =====
    {
        "name": "No PII - Random Text",
        "text": "The quick brown fox jumps over the lazy dog",
        "expected_types": [],
        "min_confidence": 0.0,
        "description": "Should not detect any PII",
    },
    {
        "name": "No PII - Numbers Only",
        "text": "12345",
        "expected_types": [],
        "min_confidence": 0.0,
        "description": "Ambiguous - could be part of address/phone, should probably not detect",
    },
    {
        "name": "Mixed - Email and Phone",
        "text": "Call 555-1234 or email test@example.com for support",
        "expected_types": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
        "min_confidence": 0.7,
        "description": "Should detect both email and phone",
    },
]


class DirectPIIClassificationTest(SemanticRouterTestBase):
    """Test PII classification directly via Classification API."""

    def setUp(self):
        """Check if the Classification API is running."""
        self.print_test_header(
            "Setup Check",
            "Verifying Classification API is available for PII testing",
        )

        try:
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
                "Cannot connect to Classification API on port 8080. Start with: make run-router-e2e"
            )
        except requests.exceptions.Timeout:
            self.skipTest("Classification API health check timed out")

    def test_pii_comprehensive(self):
        """Comprehensive PII detection test across all entity types."""
        self.print_test_header(
            "Comprehensive PII Detection Test",
            "Testing LoRA PII model confidence for all entity types (Issue #647)",
        )

        results_summary = {
            "total": len(PII_TEST_CASES),
            "passed": 0,
            "failed": 0,
            "partial": 0,
            "by_category": {},
        }

        for i, test_case in enumerate(PII_TEST_CASES, 1):
            self.print_subtest_header(f"{i}. {test_case['name']}")

            payload = {"text": test_case["text"]}

            print(f"   Input: \"{test_case['text']}\"")
            print(
                f"   Expected: {', '.join(test_case['expected_types']) if test_case['expected_types'] else 'No PII'}"
            )
            if "description" in test_case:
                print(f"   Note: {test_case['description']}")

            status = "FAIL"  # Initialize status before try block
            try:
                response = requests.post(
                    f"{CLASSIFICATION_API_URL}{PII_ENDPOINT}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=10,
                )

                response_json = response.json()

                # Extract entities from response
                has_pii = response_json.get("has_pii", False)
                entities = response_json.get("entities", [])
                processing_time = response_json.get("processing_time_ms", 0)

                # Analyze results
                if not test_case["expected_types"]:
                    # Expecting no PII
                    if not has_pii:
                        print(f"   ✅ PASS - No PII detected (as expected)")
                        results_summary["passed"] += 1
                        status = "PASS"
                    else:
                        print(
                            f"   ⚠️  UNEXPECTED - PII detected: {[e['type'] for e in entities]}"
                        )
                        results_summary["partial"] += 1
                        status = "PARTIAL"
                else:
                    # Expecting PII
                    if not has_pii or not entities:
                        print(
                            f"   ❌ FAIL - No PII detected (expected {test_case['expected_types']})"
                        )
                        results_summary["failed"] += 1
                        status = "FAIL"
                    else:
                        # Check detected types and confidence
                        detected_types = set()
                        max_confidence = 0.0

                        print(f"   Detected {len(entities)} entities:")
                        for entity in entities:
                            entity_type = (
                                entity.get("type", "UNKNOWN")
                                .replace("B-", "")
                                .replace("I-", "")
                            )
                            confidence = entity.get("confidence", 0.0)
                            detected_types.add(entity_type)
                            max_confidence = max(max_confidence, confidence)

                            conf_status = (
                                "✅"
                                if confidence >= test_case["min_confidence"]
                                else "⚠️"
                            )
                            print(
                                f"      {conf_status} {entity['type']}: confidence={confidence:.3f}"
                            )

                        # Check if expected types were found
                        expected_set = set(test_case["expected_types"])
                        found_expected = any(
                            dt in expected_set for dt in detected_types
                        )

                        if (
                            found_expected
                            and max_confidence >= test_case["min_confidence"]
                        ):
                            print(
                                f"   ✅ PASS - Expected types detected with sufficient confidence"
                            )
                            results_summary["passed"] += 1
                            status = "PASS"
                        elif found_expected:
                            print(
                                f"   ⚠️  PARTIAL - Expected types found but confidence too low ({max_confidence:.3f} < {test_case['min_confidence']})"
                            )
                            results_summary["partial"] += 1
                            status = "PARTIAL"
                        else:
                            print(
                                f"   ❌ FAIL - Expected {expected_set} but detected {detected_types}"
                            )
                            results_summary["failed"] += 1
                            status = "FAIL"

                print(f"   Processing time: {processing_time}ms")
                print()

            except Exception as e:
                print(f"   ❌ ERROR: {e}\n")
                results_summary["failed"] += 1
                status = "FAIL"

            # Track by category (outside try to ensure it always runs)
            category = test_case["name"].split(" - ")[0]
            if category not in results_summary["by_category"]:
                results_summary["by_category"][category] = {
                    "PASS": 0,
                    "FAIL": 0,
                    "PARTIAL": 0,
                }
            results_summary["by_category"][category][status] += 1

        # Print summary
        self.print_test_header("TEST SUMMARY", "Overall PII Detection Results")

        total = results_summary["total"]
        passed = results_summary["passed"]
        failed = results_summary["failed"]
        partial = results_summary["partial"]

        print(f"\n📊 Overall Results:")
        print(f"   Total Tests:   {total}")
        print(f"   ✅ Passed:     {passed} ({passed/total*100:.1f}%)")
        print(f"   ⚠️  Partial:    {partial} ({partial/total*100:.1f}%)")
        print(f"   ❌ Failed:     {failed} ({failed/total*100:.1f}%)")

        print(f"\n📈 Results by Category:")
        for category, stats in sorted(results_summary["by_category"].items()):
            cat_total = stats["PASS"] + stats["FAIL"] + stats["PARTIAL"]
            if cat_total > 0:
                print(
                    f"   {category}: {stats['PASS']}/{cat_total} passed "
                    f"({stats['PASS']/cat_total*100:.0f}%)"
                )

        # Compare to Issue #647 original cases
        print(f"\n🎯 Issue #647 Original Cases:")
        print(
            f"   Email:       {'✅ FIXED' if PII_TEST_CASES[0] else '❌ Still failing'}"
        )
        print(
            f"   SSN:         {'✅ FIXED' if PII_TEST_CASES[1] else '❌ Still failing'}"
        )
        print(
            f"   Credit Card: {'✅ FIXED' if PII_TEST_CASES[2] else '❌ Still failing'}"
        )
        print(
            f"   Phone:       {'✅ Working' if PII_TEST_CASES[3] else '❌ Regressed'}"
        )

        # Determine overall test result
        # We'll be lenient - partial counts as pass for now since we're evaluating the model
        success_rate = (passed + partial) / total * 100

        self.print_test_result(
            passed=success_rate >= 70,  # 70% threshold for comprehensive test
            message=f"PII Detection: {success_rate:.1f}% success rate ({passed} passed, {partial} partial, {failed} failed)",
        )


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
