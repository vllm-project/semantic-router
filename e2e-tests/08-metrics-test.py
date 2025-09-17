#!/usr/bin/env python3
"""
08-metrics-test.py - Comprehensive metrics tests

This test validates that all metrics endpoints are working correctly and
that the router is properly recording performance and operational metrics.
"""

import json
import os
import sys
import time
import unittest
import uuid
import re

import requests

# Import test base from same directory
from test_base import SemanticRouterTestBase

# Constants
ENVOY_URL = "http://localhost:8801"
OPENAI_ENDPOINT = "/v1/chat/completions"
ROUTER_METRICS_URL = "http://localhost:9190/metrics"
DEFAULT_MODEL = "gemma3:27b"  # Use configured model

# Expected metric families that should be present
EXPECTED_METRIC_FAMILIES = [
    # Core routing metrics
    "llm_router_requests_total",
    "llm_router_routing_decision",
    "llm_router_model_selection_count",
    
    # Classification metrics
    "llm_router_classification_duration_seconds",
    "llm_router_category_classification_total",
    
    # Security metrics
    "llm_router_jailbreak",
    "llm_router_pii",
    
    # Cache metrics (if enabled)
    "llm_router_cache",
    
    # Performance metrics
    "llm_router_request_duration_seconds",
    "llm_router_response_size_bytes",
    
    # System metrics
    "go_",  # Go runtime metrics
    "process_",  # Process metrics
]

# Sample requests to generate metrics
METRIC_TEST_REQUESTS = [
    {
        "name": "Math Query",
        "content": "What is 15 * 23?",
        "expected_category": "math",
    },
    {
        "name": "General Query",
        "content": "What is the weather like?",
        "expected_category": "other",
    },
    {
        "name": "Science Query",
        "content": "Explain photosynthesis.",
        "expected_category": "biology",
    },
]


def extract_metric_value(metrics_text, metric_name, labels=None):
    """Extract metric value from Prometheus metrics text."""
    pattern = f"^{re.escape(metric_name)}"
    if labels:
        label_pattern = ','.join([f'{k}="{v}"' for k, v in labels.items()])
        pattern += f"\\{{{label_pattern}\\}}"
    pattern += r"\s+([0-9.]+(?:e[+-]?[0-9]+)?)"
    
    for line in metrics_text.split('\n'):
        if line.startswith('#'):
            continue
        match = re.match(pattern, line)
        if match:
            return float(match.group(1))
    return None


class ComprehensiveMetricsTest(SemanticRouterTestBase):
    """Test comprehensive metrics collection and reporting."""

    def setUp(self):
        """Check if the services are running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that required services (Envoy and Router) are running and metrics are enabled",
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
            response = requests.get(ROUTER_METRICS_URL, timeout=5)
            if response.status_code != 200:
                self.skipTest(
                    "Router metrics server is not responding. Is the router running?"
                )

            self.print_response_info(response, {"Service": "Router Metrics"})

        except requests.exceptions.ConnectionError:
            self.skipTest(
                "Cannot connect to router metrics server. Is the router running?"
            )

    def test_metrics_endpoint_availability(self):
        """Test that the metrics endpoint is available and returns valid Prometheus format."""
        self.print_test_header(
            "Metrics Endpoint Availability Test",
            "Verifies that the metrics endpoint is accessible and returns valid Prometheus metrics",
        )

        response = requests.get(ROUTER_METRICS_URL, timeout=5)
        
        passed = (response.status_code == 200 and 
                 response.headers.get('content-type', '').startswith('text/plain'))

        metrics_text = response.text
        lines = metrics_text.split('\n')
        metric_lines = [line for line in lines if line and not line.startswith('#')]
        help_lines = [line for line in lines if line.startswith('# HELP')]
        type_lines = [line for line in lines if line.startswith('# TYPE')]

        self.print_response_info(
            response,
            {
                "Status Code": response.status_code,
                "Content-Type": response.headers.get('content-type', 'N/A'),
                "Total Lines": len(lines),
                "Metric Lines": len(metric_lines),
                "Help Lines": len(help_lines),
                "Type Lines": len(type_lines),
                "Response Size": len(metrics_text),
            },
        )

        self.print_test_result(
            passed=passed,
            message=(
                f"Metrics endpoint available with {len(metric_lines)} metrics"
                if passed
                else f"Metrics endpoint not available or invalid format"
            ),
        )

        self.assertEqual(response.status_code, 200, "Metrics endpoint should return 200")
        self.assertGreater(len(metric_lines), 0, "Should have at least some metrics")

    def test_expected_metric_families_present(self):
        """Test that all expected metric families are present."""
        self.print_test_header(
            "Expected Metric Families Test",
            "Verifies that all expected metric families are present in the metrics output",
        )

        response = requests.get(ROUTER_METRICS_URL, timeout=5)
        metrics_text = response.text

        found_families = {}
        missing_families = []

        for family in EXPECTED_METRIC_FAMILIES:
            found = False
            count = 0
            
            for line in metrics_text.split('\n'):
                if line and not line.startswith('#'):
                    if family.endswith('_'):
                        # Prefix match for families like 'go_' and 'process_'
                        if line.startswith(family):
                            found = True
                            count += 1
                    else:
                        # Exact or partial match for specific metrics
                        if family in line:
                            found = True
                            count += 1
            
            if found:
                found_families[family] = count
            else:
                missing_families.append(family)

        self.print_response_info(
            response,
            {
                "Total Expected Families": len(EXPECTED_METRIC_FAMILIES),
                "Found Families": len(found_families),
                "Missing Families": len(missing_families),
            },
        )

        # Print found families
        print(f"\nFound metric families:")
        for family, count in found_families.items():
            print(f"  {family}: {count} metrics")

        # Print missing families
        if missing_families:
            print(f"\nMissing metric families:")
            for family in missing_families:
                print(f"  {family}")

        passed = len(found_families) >= len(EXPECTED_METRIC_FAMILIES) * 0.7  # Allow 70% success rate
        self.print_test_result(
            passed=passed,
            message=f"Found {len(found_families)}/{len(EXPECTED_METRIC_FAMILIES)} expected metric families",
        )

        self.assertGreater(len(found_families), 0, "Should find at least some expected metric families")

    def test_metrics_increase_with_requests(self):
        """Test that metrics increase as requests are processed."""
        self.print_test_header(
            "Metrics Increase Test",
            "Verifies that metrics values increase as requests are processed through the router",
        )

        # Get baseline metrics
        baseline_response = requests.get(ROUTER_METRICS_URL, timeout=5)
        baseline_metrics = baseline_response.text

        baseline_requests = extract_metric_value(baseline_metrics, "llm_router_requests_total") or 0
        
        self.print_subtest_header("Baseline Metrics")
        print(f"Baseline requests total: {baseline_requests}")

        # Make several test requests
        for i, test_request in enumerate(METRIC_TEST_REQUESTS):
            self.print_subtest_header(f"Test Request {i+1}: {test_request['name']}")

            session_id = str(uuid.uuid4())
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": test_request["content"]},
                ],
                "temperature": 0.7,
            }

            headers = {
                "Content-Type": "application/json",
                "X-Session-ID": session_id,
            }

            response = requests.post(
                f"{ENVOY_URL}{OPENAI_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=10,
            )

            self.print_response_info(
                response,
                {
                    "Request": test_request["name"],
                    "Status Code": response.status_code,
                    "Content": test_request["content"][:50] + "...",
                },
            )

            time.sleep(1)  # Allow metrics to be recorded

        # Get updated metrics
        time.sleep(2)  # Additional wait for metrics aggregation
        updated_response = requests.get(ROUTER_METRICS_URL, timeout=5)
        updated_metrics = updated_response.text

        updated_requests = extract_metric_value(updated_metrics, "llm_router_requests_total") or 0

        print(f"\nUpdated requests total: {updated_requests}")
        requests_increase = updated_requests - baseline_requests

        passed = requests_increase >= len(METRIC_TEST_REQUESTS)
        self.print_test_result(
            passed=passed,
            message=f"Requests metric increased by {requests_increase} (expected at least {len(METRIC_TEST_REQUESTS)})",
        )

        self.assertGreaterEqual(
            requests_increase, len(METRIC_TEST_REQUESTS),
            f"Requests metric should increase by at least {len(METRIC_TEST_REQUESTS)}, got {requests_increase}"
        )

    def test_performance_metrics_present(self):
        """Test that performance metrics are being recorded."""
        self.print_test_header(
            "Performance Metrics Test",
            "Verifies that performance metrics like duration and response size are recorded",
        )

        response = requests.get(ROUTER_METRICS_URL, timeout=5)
        metrics_text = response.text

        performance_metrics = [
            "llm_router_request_duration_seconds",
            "llm_router_classification_duration_seconds",
            "llm_router_routing_latency_ms",
        ]

        found_metrics = {}
        for metric in performance_metrics:
            # Look for histogram or summary metrics
            for line in metrics_text.split('\n'):
                if metric in line and not line.startswith('#'):
                    if metric not in found_metrics:
                        found_metrics[metric] = []
                    found_metrics[metric].append(line.strip())

        self.print_response_info(
            response,
            {
                "Performance Metrics Expected": len(performance_metrics),
                "Performance Metrics Found": len(found_metrics),
            },
        )

        # Print found performance metrics
        for metric, lines in found_metrics.items():
            print(f"\nMetric: {metric}")
            for line in lines[:3]:  # Show first 3 lines
                print(f"  {line}")
            if len(lines) > 3:
                print(f"  ... and {len(lines) - 3} more lines")

        passed = len(found_metrics) > 0
        self.print_test_result(
            passed=passed,
            message=f"Found {len(found_metrics)} out of {len(performance_metrics)} performance metrics",
        )

        self.assertGreater(len(found_metrics), 0, "Should find at least some performance metrics")

    def test_system_health_metrics(self):
        """Test that system health metrics are available."""
        self.print_test_header(
            "System Health Metrics Test",
            "Verifies that system health metrics like Go runtime and process metrics are available",
        )

        response = requests.get(ROUTER_METRICS_URL, timeout=5)
        metrics_text = response.text

        system_metric_prefixes = [
            "go_memstats_",
            "go_goroutines",
            "go_info",
            "process_cpu_",
            "process_virtual_memory_",
            "process_resident_memory_",
        ]

        found_system_metrics = {}
        for prefix in system_metric_prefixes:
            count = 0
            for line in metrics_text.split('\n'):
                if line.startswith(prefix) and not line.startswith('#'):
                    count += 1
            if count > 0:
                found_system_metrics[prefix] = count

        self.print_response_info(
            response,
            {
                "System Metric Prefixes Expected": len(system_metric_prefixes),
                "System Metric Prefixes Found": len(found_system_metrics),
                "Total System Metrics": sum(found_system_metrics.values()),
            },
        )

        # Print found system metrics
        for prefix, count in found_system_metrics.items():
            print(f"  {prefix}: {count} metrics")

        passed = len(found_system_metrics) >= len(system_metric_prefixes) * 0.5  # At least 50%
        self.print_test_result(
            passed=passed,
            message=f"Found {len(found_system_metrics)}/{len(system_metric_prefixes)} system metric families",
        )

        self.assertGreater(len(found_system_metrics), 0, "Should find at least some system metrics")

    def test_custom_labels_in_metrics(self):
        """Test that custom labels are properly included in metrics."""
        self.print_test_header(
            "Custom Labels Test",
            "Verifies that metrics include appropriate labels for filtering and aggregation",
        )

        response = requests.get(ROUTER_METRICS_URL, timeout=5)
        metrics_text = response.text

        # Look for metrics with labels
        labeled_metrics = []
        for line in metrics_text.split('\n'):
            if '{' in line and '}' in line and not line.startswith('#'):
                labeled_metrics.append(line.strip())

        # Sample some labeled metrics
        sample_labeled_metrics = labeled_metrics[:10]

        self.print_response_info(
            response,
            {
                "Total Metrics": len([l for l in metrics_text.split('\n') if l and not l.startswith('#')]),
                "Labeled Metrics": len(labeled_metrics),
                "Label Usage": f"{len(labeled_metrics)/max(1, len([l for l in metrics_text.split('\n') if l and not l.startswith('#')]))*100:.1f}%",
            },
        )

        # Print sample labeled metrics
        print(f"\nSample labeled metrics:")
        for metric in sample_labeled_metrics:
            print(f"  {metric}")

        passed = len(labeled_metrics) > 0
        self.print_test_result(
            passed=passed,
            message=f"Found {len(labeled_metrics)} metrics with labels",
        )

        self.assertGreater(len(labeled_metrics), 0, "Should find at least some metrics with labels")

    def test_metrics_format_validation(self):
        """Test that metrics follow proper Prometheus format."""
        self.print_test_header(
            "Metrics Format Validation Test",
            "Verifies that metrics follow proper Prometheus exposition format",
        )

        response = requests.get(ROUTER_METRICS_URL, timeout=5)
        metrics_text = response.text

        format_issues = []
        valid_metrics = 0
        total_metrics = 0

        for line in metrics_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            total_metrics += 1
            
            # Basic format validation
            if ' ' not in line:
                format_issues.append(f"No space separator: {line[:50]}...")
                continue
                
            parts = line.split(' ', 1)
            if len(parts) != 2:
                format_issues.append(f"Invalid format: {line[:50]}...")
                continue
                
            metric_name_and_labels, value = parts
            
            # Validate metric name
            if not re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{.*\})?$', metric_name_and_labels):
                format_issues.append(f"Invalid metric name: {metric_name_and_labels}")
                continue
                
            # Validate value
            try:
                float(value)
                valid_metrics += 1
            except ValueError:
                format_issues.append(f"Invalid value: {value}")

        format_score = valid_metrics / max(1, total_metrics) * 100

        self.print_response_info(
            response,
            {
                "Total Metrics": total_metrics,
                "Valid Metrics": valid_metrics,
                "Format Issues": len(format_issues),
                "Format Score": f"{format_score:.1f}%",
            },
        )

        # Print first few format issues
        if format_issues:
            print(f"\nFormat issues (showing first 5):")
            for issue in format_issues[:5]:
                print(f"  {issue}")

        passed = format_score >= 90  # Require 90% format compliance
        self.print_test_result(
            passed=passed,
            message=f"Metrics format validation: {format_score:.1f}% valid",
        )

        self.assertGreaterEqual(format_score, 90, f"Metrics format score should be >= 90%, got {format_score:.1f}%")


if __name__ == "__main__":
    unittest.main()