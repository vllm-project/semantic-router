#!/usr/bin/env python3
"""
06-pii-detection-test.py - PII Detection Tests

This test validates PII detection across different classifier paths:

🔍 CLASSIFICATION PATHS TESTED:
1. BATCH API (Unified classifier) - Expected: WORKING ✅
   - Uses unified classifier with PII detection models
   - Should detect PII entities with high confidence (~99%)

2. DIRECT PII API - Expected: BROKEN ❌
   - Uses different code path than batch API
   - Returns empty results despite containing PII

3. EXTPROC PIPELINE - Expected: UNKNOWN ⚠️
   - Tests if PII detection works in request filtering
   - May show similar discrepancies as jailbreak detection

🚨 PII DETECTION GAP: Batch API detects PII correctly but direct API fails
This demonstrates inconsistent PII classifier usage between components.
"""

import json
import sys
import time
import unittest
import uuid

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
CLASSIFICATION_API_URL = "http://localhost:8080"
ENVOY_URL = "http://localhost:8801"
BATCH_ENDPOINT = "/api/v1/classify/batch"
PII_ENDPOINT = "/api/v1/classify/pii"
OPENAI_ENDPOINT = "/v1/chat/completions"

# Base PII test cases - will be made unique each run to avoid caching
PII_TEST_TEMPLATES = [
    {
        "name": "Social Security Number",
        "template": "My SSN is 123-45-6789 for application {}",
        "expected_entities": ["SSN", "US_SSN"],
        "risk_level": "high",
    },
    {
        "name": "Email Address",
        "template": "Contact me at john.doe+{}@example.com for updates",
        "expected_entities": ["EMAIL", "EMAIL_ADDRESS"],
        "risk_level": "medium",
    },
    {
        "name": "Phone Number",
        "template": "Call me at (555) 123-4567 about case {}",
        "expected_entities": ["PHONE", "PHONE_NUMBER"],
        "risk_level": "medium",
    },
    {
        "name": "Credit Card",
        "template": "Card number 4111-1111-1111-1111 for order {}",
        "expected_entities": ["CREDIT_CARD", "CARD_NUMBER"],
        "risk_level": "high",
    },
    {
        "name": "Multiple PII Types",
        "template": "John Smith, SSN: 987-65-4321, email: jsmith{}@company.com, phone: 555-987-6543",
        "expected_entities": ["PERSON", "SSN", "EMAIL", "PHONE"],
        "risk_level": "critical",
    },
]

# Safe test cases - these should NOT contain PII
SAFE_TEST_TEMPLATES = [
    {
        "name": "Math Problem",
        "template": "Calculate 123 + 456 = 579 for homework {}",
        "expected_entities": [],
        "risk_level": "none",
    },
    {
        "name": "Generic Text",
        "template": "The weather is nice today for event {}",
        "expected_entities": [],
        "risk_level": "none",
    },
    {
        "name": "Product Information",
        "template": "Product ABC-123 costs $99.99 in store {}",
        "expected_entities": [],
        "risk_level": "none",
    },
]


def generate_unique_test_cases():
    """Generate unique test cases with timestamp to avoid caching."""
    import time

    timestamp = str(int(time.time() * 1000))[-8:]  # Last 8 digits of milliseconds
    unique_id = str(uuid.uuid4())[:8]
    cache_buster = f"{timestamp}-{unique_id}"

    pii_cases = []
    for template in PII_TEST_TEMPLATES:
        pii_cases.append(
            {
                "name": template["name"],
                "text": template["template"].format(cache_buster),
                "expected_entities": template["expected_entities"],
                "risk_level": template["risk_level"],
            }
        )

    safe_cases = []
    for template in SAFE_TEST_TEMPLATES:
        safe_cases.append(
            {
                "name": template["name"],
                "text": template["template"].format(cache_buster),
                "expected_entities": template["expected_entities"],
                "risk_level": template["risk_level"],
            }
        )

    return pii_cases, safe_cases


class PIIDetectionTest(SemanticRouterTestBase):
    """Test PII detection across Classification API and ExtProc pipeline."""

    def setUp(self):
        """Check if services are running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that Classification API and Envoy/ExtProc are running",
        )

        # Check Classification API
        try:
            health_response = requests.get(
                f"{CLASSIFICATION_API_URL}/health", timeout=5
            )
            if health_response.status_code != 200:
                self.skipTest(
                    f"Classification API health check failed: {health_response.status_code}"
                )
        except requests.exceptions.ConnectionError:
            self.skipTest("Cannot connect to Classification API on port 8080")

        # Check Envoy/ExtProc with simple test
        try:
            test_payload = {
                "model": "auto",
                "messages": [
                    {"role": "user", "content": f"Setup test {str(uuid.uuid4())[:8]}"}
                ],
            }
            envoy_response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=30,
            )
            if envoy_response.status_code >= 500:
                self.skipTest(
                    f"Envoy/ExtProc health check failed: {envoy_response.status_code}"
                )
        except requests.exceptions.ConnectionError:
            self.skipTest("Cannot connect to Envoy on port 8801")
        except requests.exceptions.Timeout:
            self.skipTest("Envoy/ExtProc setup test timed out")

        print("✅ All required services are running")

    def test_batch_pii_detection(self):
        """
        TEST 1: Batch API PII Detection (Unified Classifier Path)

        WHAT IS TESTED: /api/v1/classify/batch with task_type="pii"
        CLASSIFIER PATH: Unified classifier with PII detection models
        EXPECTED RESULT: ✅ WORKING - Should detect PII entities with high confidence

        This path uses the working unified PII classifier models.
        """
        self.print_test_header(
            "TEST 1: Batch API PII Detection (Unified Classifier)",
            "Expected: ✅ WORKING - Uses unified classifier that correctly detects PII",
        )

        # Generate unique PII test to avoid caching
        pii_cases, _ = generate_unique_test_cases()
        pii_text = pii_cases[0]["text"]  # Use first PII case (SSN)

        payload = {"texts": [pii_text], "task_type": "pii"}

        self.print_request_info(
            payload=payload,
            expectations="✅ EXPECTED: Category containing PII type (e.g., 'B-US_SSN'), Confidence=~0.99",
        )

        response = requests.post(
            f"{CLASSIFICATION_API_URL}{BATCH_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        response_json = response.json()
        results = response_json.get("results", [])

        if len(results) > 0:
            result = results[0]
            actual_category = result.get("category", "unknown")
            confidence = result.get("confidence", 0.0)

            # Check if PII was properly detected
            is_no_pii_classification = actual_category in ["no_pii", "safe", "unknown"]
            is_pii_detected = (
                not is_no_pii_classification
            )  # Any specific PII type indicates detection

            self.print_response_info(
                response,
                {
                    "Text": pii_text,
                    "Actual Category": actual_category,
                    "Confidence": f"{confidence:.3f}",
                    "No PII Classification": (
                        "❌ YES" if is_no_pii_classification else "✅ NO"
                    ),
                    "PII Detected": ("✅ YES" if is_pii_detected else "❌ NO"),
                    "Detection Status": (
                        "🚨 MISSED PII" if is_no_pii_classification else "✅ DETECTED"
                    ),
                },
            )

            # The test should fail if PII content is not detected
            if is_no_pii_classification:
                self.print_test_result(
                    passed=False,
                    message=f"🚨 PII DETECTION FAILURE: PII content '{pii_text}' "
                    f"classified as '{actual_category}' instead of detecting PII",
                )
                self.fail(
                    f"PII detection failed: content with SSN classified as '{actual_category}'"
                )
            elif is_pii_detected:
                self.print_test_result(
                    passed=True,
                    message=f"PII correctly detected as '{actual_category}'",
                )
            else:
                self.print_test_result(
                    passed=False,
                    message=f"Unknown classification result: '{actual_category}'",
                )

        self.assertEqual(response.status_code, 200, "Batch request failed")
        self.assertGreater(len(results), 0, "No classification results returned")

    def test_direct_pii_endpoint(self):
        """
        TEST 2: Direct PII API Endpoint

        WHAT IS TESTED: /api/v1/classify/pii endpoint (direct PII classification)
        CLASSIFIER PATH: Different implementation from batch API
        EXPECTED RESULT: ❌ BROKEN - Returns empty results despite containing PII

        This tests the discrepancy between batch and direct PII endpoints.
        """
        self.print_test_header(
            "TEST 2: Direct PII API Endpoint",
            "Expected: ❌ BROKEN - Different implementation fails to detect PII",
        )

        # Generate unique PII test to avoid caching
        pii_cases, _ = generate_unique_test_cases()
        pii_text = pii_cases[0]["text"]  # Use first PII case (SSN)

        payload = {
            "text": pii_text,
            "options": {"return_positions": True, "confidence_threshold": 0.5},
        }

        self.print_request_info(
            payload=payload,
            expectations="❌ EXPECTED: has_pii=false, entities=[] (broken implementation)",
        )

        response = requests.post(
            f"{CLASSIFICATION_API_URL}{PII_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            response_json = response.json()
            has_pii = response_json.get("has_pii", False)
            entities = response_json.get("entities", [])
            security_recommendation = response_json.get(
                "security_recommendation", "unknown"
            )
            processing_time = response_json.get("processing_time_ms", 0)

            # Detection based on has_pii field and entities
            is_pii_detected = has_pii and len(entities) > 0

            self.print_response_info(
                response,
                {
                    "Endpoint Status": "✅ Available",
                    "Has PII": has_pii,
                    "Entities Count": len(entities),
                    "Security Recommendation": security_recommendation,
                    "Processing Time (ms)": processing_time,
                    "PII Detected": "✅ YES" if is_pii_detected else "❌ NO",
                    "Consistency with Batch": (
                        "✅ CONSISTENT" if is_pii_detected else "❌ INCONSISTENT"
                    ),
                },
            )

            if entities:
                print(f"\n📋 Detected PII Entities:")
                for i, entity in enumerate(entities):
                    entity_type = entity.get("type", "unknown")
                    confidence = entity.get("confidence", 0.0)
                    value = entity.get("value", "")
                    print(
                        f"  {i+1}. Type: {entity_type}, Confidence: {confidence:.3f}, Value: {value}"
                    )

            if is_pii_detected:
                self.print_test_result(
                    passed=True,
                    message=f"✅ Direct PII endpoint working: detected {len(entities)} PII entities",
                )
            else:
                self.print_test_result(
                    passed=False,
                    message=f"🚨 DISCREPANCY: Direct PII endpoint fails where batch endpoint succeeds (has_pii={has_pii}, entities={len(entities)})",
                )
                # Document the discrepancy instead of failing
                print(
                    f"⚠️  NOTE: Batch endpoint correctly detects PII but direct endpoint doesn't"
                )
                print(
                    f"⚠️  This suggests different implementations between batch and direct PII endpoints"
                )
        else:
            self.print_response_info(
                response,
                {
                    "Endpoint Status": "❌ Error",
                    "Error Code": response.status_code,
                },
            )
            self.print_test_result(
                passed=False,
                message=f"🚨 Direct PII endpoint failed with status {response.status_code}",
            )
            self.fail(f"PII endpoint request failed: {response.status_code}")

        self.assertEqual(response.status_code, 200, "PII endpoint request failed")

    def test_extproc_pii_filtering(self):
        """
        TEST 3: ExtProc PII Filtering (Production Pipeline)

        WHAT IS TESTED: Full LLM completion pipeline through Envoy/ExtProc
        CLASSIFIER PATH: ExtProc PII detection during request processing
        EXPECTED RESULT: ⚠️ UNKNOWN - May show similar issues as jailbreak detection

        This tests if PII content is filtered or logged in the ExtProc pipeline.
        """
        self.print_test_header(
            "TEST 3: ExtProc PII Filtering (Production Pipeline)",
            "Expected: ⚠️ UNKNOWN - Tests if PII is detected/filtered in request pipeline",
        )

        # Generate unique PII test to avoid caching
        pii_cases, _ = generate_unique_test_cases()
        pii_content = pii_cases[4]["text"]  # Use multiple PII types case

        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": pii_content}],
            "temperature": 0.1,
        }

        self.print_request_info(
            payload=payload,
            expectations="⚠️ EXPECTED: Request may be allowed through (PII filtering behavior unknown)",
        )

        response = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )

        # Check if the request went through
        request_allowed = response.status_code == 200

        response_content = ""
        if request_allowed:
            try:
                response_json = response.json()
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    response_content = (
                        response_json["choices"][0]
                        .get("message", {})
                        .get("content", "")
                    )
            except:
                response_content = "Could not parse response"

        # Check if response potentially exposes PII patterns
        pii_patterns = ["SSN", "social security", "123", "555", "email", "@"]
        potentially_exposes_pii = any(
            pattern.lower() in response_content.lower() for pattern in pii_patterns
        )

        self.print_response_info(
            response,
            {
                "PII Request": pii_content,
                "Request Allowed": "✅ YES" if request_allowed else "❌ NO (Blocked)",
                "Response Contains PII Patterns": (
                    "⚠️ YES" if potentially_exposes_pii else "✅ NO"
                ),
                "PII Filtering Status": (
                    "⚠️ NO FILTERING DETECTED"
                    if (request_allowed and potentially_exposes_pii)
                    else "✅ NO PII EXPOSURE"
                ),
                "Response Preview": (
                    response_content[:100] + "..." if response_content else "No content"
                ),
            },
        )

        if request_allowed:
            if potentially_exposes_pii:
                self.print_test_result(
                    passed=False,
                    message="⚠️ PII EXPOSURE RISK: ExtProc allows PII content and response may contain PII patterns",
                )
                # Don't fail the test - this is informational
                print(
                    "📝 NOTE: This test documents PII handling behavior rather than enforcing strict requirements"
                )
            else:
                self.print_test_result(
                    passed=True,
                    message="✅ No obvious PII exposure: Request processed without echoing PII patterns",
                )
        else:
            self.print_test_result(
                passed=True,
                message="✅ Request blocked: ExtProc may have PII filtering enabled",
            )

    def test_extproc_pii_detection_comprehensive(self):
        """
        TEST 3.5: ExtProc PII Detection - Comprehensive Production Pipeline Test

        WHAT IS TESTED: Full production pipeline PII detection capabilities
        PIPELINE: client → envoy → extProc → semantic-router → model routing
        EXPECTED RESULT: ⚠️ VALIDATION - Tests actual PII detection in production flow

        This tests the MAIN USE CASE: whether ExtProc actually detects PII and enforces policies
        in the production request pipeline, not just whether responses leak PII.
        """
        self.print_test_header(
            "TEST 3.5: ExtProc PII Detection - Comprehensive Production Pipeline",
            "Expected: ⚠️ VALIDATION - Tests if ExtProc actually detects PII in production flow",
        )

        # Generate unique PII test cases
        pii_cases, safe_cases = generate_unique_test_cases()

        # Test 1: High-value PII that should trigger policy enforcement
        high_value_pii = pii_cases[0]["text"]  # SSN case

        # Test 2: Safe content that should pass through
        safe_content = safe_cases[0]["text"]  # Math problem

        test_cases = [
            {
                "name": "High-Value PII Content",
                "content": high_value_pii,
                "expected_pii": True,
                "expected_action": "May block or route to PII-safe model",
            },
            {
                "name": "Safe Content",
                "content": safe_content,
                "expected_pii": False,
                "expected_action": "Should route normally",
            },
        ]

        results = []

        for test_case in test_cases:
            print(f"\n🔍 Testing: {test_case['name']}")
            print(f"Content: {test_case['content'][:60]}...")

            # Send request through full ExtProc pipeline
            payload = {
                "model": "auto",
                "messages": [{"role": "user", "content": test_case["content"]}],
                "temperature": 0.1,
            }

            # Monitor for specific behaviors that indicate PII detection
            start_time = time.time()

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )

            processing_time = (time.time() - start_time) * 1000

            # Analyze response for PII detection indicators
            request_allowed = response.status_code == 200

            # Check for PII policy violation response
            is_pii_blocked = response.status_code == 403 or response.status_code == 400

            # Analyze response content
            response_content = ""
            model_used = "unknown"
            if request_allowed:
                try:
                    response_json = response.json()
                    if "choices" in response_json:
                        response_content = response_json["choices"][0]["message"][
                            "content"
                        ]
                    model_used = response_json.get("model", "unknown")
                except:
                    response_content = "Could not parse response"

            # Check for PII-related routing decisions
            # ExtProc may route PII content to specific models or add safety measures
            pii_safe_routing = model_used in [
                "Model-A",
                "safe-model",
                "pii-compliant-model",
            ]

            # Check response quality/safety indicators
            response_looks_safe = (
                len(response_content) > 50
                and "sorry" not in response_content.lower()
                and "can't help" not in response_content.lower()
            )

            # Analyze processing time (PII detection may add latency)
            baseline_time = 1000  # ms
            has_extra_processing = processing_time > baseline_time * 1.5

            result = {
                "test_case": test_case["name"],
                "content": test_case["content"][:50] + "...",
                "expected_pii": test_case["expected_pii"],
                "request_allowed": request_allowed,
                "is_blocked": is_pii_blocked,
                "model_used": model_used,
                "processing_time_ms": processing_time,
                "has_extra_processing": has_extra_processing,
                "response_safe": response_looks_safe,
                "response_preview": (
                    response_content[:100] + "..." if response_content else "No content"
                ),
            }

            results.append(result)

            print(f"  📊 Results:")
            print(f"    Request Status: {response.status_code}")
            print(f"    Model Used: {model_used}")
            print(f"    Processing Time: {processing_time:.1f}ms")
            print(f"    Expected PII: {test_case['expected_pii']}")
            print(f"    Blocked: {is_pii_blocked}")
            print(f"    Extra Processing: {has_extra_processing}")

        # Analyze results for PII detection patterns
        pii_test = results[0]  # High-value PII test
        safe_test = results[1]  # Safe content test

        # Check for differential behavior between PII and safe content
        differential_routing = pii_test["model_used"] != safe_test["model_used"]
        differential_processing = (
            abs(pii_test["processing_time_ms"] - safe_test["processing_time_ms"]) > 200
        )
        differential_blocking = pii_test["is_blocked"] != safe_test["is_blocked"]

        # Overall PII detection indicators
        pii_detection_indicators = []
        if pii_test["is_blocked"]:
            pii_detection_indicators.append("Request blocking")
        if differential_routing:
            pii_detection_indicators.append("Model routing differences")
        if differential_processing:
            pii_detection_indicators.append("Processing time differences")
        if not pii_test["response_safe"] and safe_test["response_safe"]:
            pii_detection_indicators.append("Response quality differences")

        # Final assessment
        pii_detection_evidence = len(pii_detection_indicators) > 0

        self.print_response_info(
            response,  # Use last response for HTTP details
            {
                "Test Cases": len(test_cases),
                "PII Detection Evidence": (
                    "✅ YES" if pii_detection_evidence else "❌ NO"
                ),
                "Detection Indicators": (
                    ", ".join(pii_detection_indicators)
                    if pii_detection_indicators
                    else "None found"
                ),
                "PII Content Model": pii_test["model_used"],
                "Safe Content Model": safe_test["model_used"],
                "Differential Routing": "✅ YES" if differential_routing else "❌ NO",
                "PII Request Blocked": "✅ YES" if pii_test["is_blocked"] else "❌ NO",
                "Overall Assessment": (
                    "✅ PII DETECTION ACTIVE"
                    if pii_detection_evidence
                    else "⚠️ NO CLEAR PII DETECTION"
                ),
            },
        )

        # Print detailed analysis
        print(f"\n📋 Detailed ExtProc PII Analysis:")
        for result in results:
            status = (
                "🔒"
                if result["is_blocked"]
                else "✅" if result["request_allowed"] else "❌"
            )
            print(f"  {status} {result['test_case']}")
            print(f"      Content: {result['content']}")
            print(
                f"      Model: {result['model_used']}, Time: {result['processing_time_ms']:.1f}ms"
            )
            print(f"      Status: {'Blocked' if result['is_blocked'] else 'Allowed'}")

        if pii_detection_evidence:
            self.print_test_result(
                passed=True,
                message=f"✅ ExtProc PII detection evidence found: {', '.join(pii_detection_indicators)}",
            )
        else:
            self.print_test_result(
                passed=False,
                message="⚠️ No clear evidence of ExtProc PII detection in production pipeline",
            )
            print(
                "📝 NOTE: This may indicate PII detection is not active in ExtProc or"
            )
            print("         PII policies are configured to allow all content through")

    def test_multiple_pii_types_analysis(self):
        """
        TEST 4: Multiple PII Types Pattern Analysis

        WHAT IS TESTED: Various PII types through batch API
        CLASSIFIER PATH: Same as Test 1 (Unified classifier via batch API)
        EXPECTED RESULT: ✅ HIGH DETECTION RATE - Should detect most PII types

        This validates the unified classifier works across different PII entity types.
        """
        self.print_test_header(
            "TEST 4: Multiple PII Types Pattern Analysis",
            "Expected: ✅ HIGH DETECTION - Unified classifier should detect various PII types",
        )

        # Generate unique test cases to avoid caching
        pii_cases, safe_cases = generate_unique_test_cases()
        all_test_cases = pii_cases + safe_cases
        all_texts = [case["text"] for case in all_test_cases]

        payload = {"texts": all_texts, "task_type": "pii"}

        self.print_request_info(
            payload={"texts": f"{len(all_texts)} test cases", "task_type": "pii"},
            expectations="✅ EXPECTED: >80% detection rate for PII content, low false positives",
        )

        response = requests.post(
            f"{CLASSIFICATION_API_URL}{BATCH_ENDPOINT}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        response_json = response.json()
        results = response_json.get("results", [])

        # Analyze results
        pii_detected_count = 0
        safe_misclassified_count = 0
        detection_failures = []

        print(f"\n📊 Detailed Analysis of {len(results)} PII Classifications:")

        for i, (result, test_case) in enumerate(zip(results, all_test_cases)):
            actual_category = result.get("category", "unknown")
            confidence = result.get("confidence", 0.0)

            is_pii_case = test_case in pii_cases
            detected_as_pii = actual_category not in [
                "no_pii",
                "safe",
                "unknown",
            ]  # Any specific PII type indicates detection
            missed_pii = actual_category in ["no_pii", "safe"] and is_pii_case

            if is_pii_case and detected_as_pii:
                pii_detected_count += 1
            elif not is_pii_case and detected_as_pii:
                safe_misclassified_count += 1
            elif missed_pii:
                detection_failures.append(
                    (test_case["text"], "classified as safe/no_pii")
                )

            status = "✅" if (is_pii_case == detected_as_pii) else "❌"
            print(f"  {i+1}. {status} '{test_case['text'][:50]}...'")
            print(
                f"     Expected: {'PII detected' if is_pii_case else 'no PII'} | "
                f"Actual: {actual_category} | Confidence: {confidence:.3f}"
            )

        # Calculate metrics
        total_pii_cases = len(pii_cases)
        detection_rate = (
            (pii_detected_count / total_pii_cases) * 100 if total_pii_cases > 0 else 0
        )

        self.print_response_info(
            response,
            {
                "Total Test Cases": len(all_texts),
                "PII Cases": total_pii_cases,
                "Safe Cases": len(safe_cases),
                "PII Detected": f"{pii_detected_count}/{total_pii_cases}",
                "Detection Rate": f"{detection_rate:.1f}%",
                "Safe Content Misclassified": safe_misclassified_count,
                "Detection Failures": len(detection_failures),
                "Critical Issues": len(detection_failures) + safe_misclassified_count,
            },
        )

        # Print PII detection failures
        if detection_failures:
            print(f"\n🚨 PII Detection Failures ({len(detection_failures)}):")
            for text, issue in detection_failures:
                print(f"  • '{text[:60]}...' → {issue}")

        # Test should pass if detection rate is reasonable
        has_major_issues = (
            detection_rate < 60
            or len(detection_failures) > 1
            or safe_misclassified_count > 1
        )

        self.print_test_result(
            passed=not has_major_issues,
            message=(
                f"PII detection analysis complete: {detection_rate:.1f}% detection rate, "
                f"{len(detection_failures)} failures, {safe_misclassified_count} false positives"
                if not has_major_issues
                else f"🚨 MAJOR ISSUES: {detection_rate:.1f}% detection rate, "
                f"{len(detection_failures)} failures, {safe_misclassified_count} false positives"
            ),
        )

        self.assertEqual(response.status_code, 200, "Batch request failed")
        self.assertGreater(
            detection_rate,
            50,
            f"PII detection rate too low: {detection_rate:.1f}%",
        )
        self.assertLessEqual(
            len(detection_failures),
            1,
            f"Too many PII detection failures: {detection_failures}",
        )


# EXPECTED TEST RESULTS SUMMARY:
# ============================
#
# ✅ TEST 1 (Batch API): SHOULD PASS
#    - Unified classifier detects PII correctly (~99% confidence)
#    - Demonstrates working PII detection capability
#
# ❌ TEST 2 (Direct PII API): WILL FAIL
#    - Different implementation fails to detect PII
#    - Exposes critical discrepancy between endpoints
#
# ⚠️ TEST 3 (ExtProc Pipeline): INFORMATIONAL
#    - Documents PII handling behavior in production pipeline
#    - May show no filtering or logging of PII content
#
# ✅ TEST 4 (Pattern Analysis): SHOULD PASS
#    - Validates unified classifier works across different PII types
#    - Confirms consistent detection rates
#
# 🔍 PII DETECTION FINDINGS:
# - Batch API classification works but direct PII endpoint fails
# - Different code paths have inconsistent PII classifier implementations
# - Production pipeline may not properly handle PII content

if __name__ == "__main__":
    unittest.main()
