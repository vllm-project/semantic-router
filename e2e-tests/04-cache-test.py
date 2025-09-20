#!/usr/bin/env python3
"""
04-cache-test.py - Semantic cache tests

This test validates the semantic cache functionality by sending similar
queries and checking if cache hits occur as expected.

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


# Helper function to extract a specific metric value
def extract_metric(metrics_text, metric_name):
    for line in metrics_text.split("\n"):
        if line.startswith(metric_name) and not line.startswith("#"):
            return float(line.split()[1])
    return None


class SemanticCacheTest(SemanticRouterTestBase):
    """Test the semantic cache functionality."""

    def setUp(self):
        """Check if the services are running before running tests."""
        self.print_test_header(
            "Setup Check",
            "Verifying that required services (Envoy and Router) are running and cache is enabled",
        )

        # Check Envoy
        try:
            payload = {
                "model": "gemma3:27b",
                "messages": [{"role": "user", "content": "test"}],
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

        # Check if cache is enabled in metrics
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text
        if "llm_router_cache" not in metrics_text:
            self.skipTest("Cache metrics not found. Semantic cache may be disabled.")

    def test_cache_hit_with_identical_query(self):
        """Test that identical queries result in cache hits."""
        self.print_test_header(
            "Identical Query Cache Test",
            "Verifies that identical queries trigger cache hits",
        )

        session_id = str(uuid.uuid4())
        query = "What are the primary causes of climate change?"

        # Get baseline cache metrics
        response = requests.get(ROUTER_METRICS_URL)
        baseline_metrics = response.text
        baseline_hits = (
            extract_metric(baseline_metrics, "llm_router_cache_hits_total") or 0
        )

        self.print_request_info(
            payload={
                "model": "gemma3:27b",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.7,
            },
            expectations="Expect: First request to be a cache miss, second request to be a cache hit",
        )

        # First request should be a cache miss
        payload = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
            "temperature": 0.7,
        }

        headers = {"Content-Type": "application/json", "X-Session-ID": session_id}

        # First request
        self.print_subtest_header("First Request (Expected Cache Miss)")
        response1 = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}", headers=headers, json=payload, timeout=30
        )

        response1_json = response1.json()
        model1 = response1_json.get("model", "unknown")

        self.print_response_info(
            response1,
            {
                "Selected Model": model1,
                "Session ID": session_id,
                "Cache Status": "Expected Miss",
            },
        )

        # Wait a moment to ensure metrics are updated
        time.sleep(2)

        # Second identical request
        self.print_subtest_header("Second Request (Expected Cache Hit)")
        response2 = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}", headers=headers, json=payload, timeout=30
        )

        response2_json = response2.json()
        model2 = response2_json.get("model", "unknown")

        self.print_response_info(
            response2,
            {
                "Selected Model": model2,
                "Session ID": session_id,
                "Cache Status": "Expected Hit",
            },
        )

        # Wait for metrics to update
        time.sleep(2)

        # Check if cache hits increased
        response = requests.get(ROUTER_METRICS_URL)
        updated_metrics = response.text
        updated_hits = (
            extract_metric(updated_metrics, "llm_router_cache_hits_total") or 0
        )

        passed = (model1 == model2) and (updated_hits > baseline_hits)
        self.print_test_result(
            passed=passed,
            message=(
                f"Cache hits increased from {baseline_hits} to {updated_hits}, "
                f"Models consistent: {model1 == model2} ({model1})"
            ),
        )

        self.assertEqual(
            model1,
            model2,
            f"Routed to different models for identical queries: {model1} vs {model2}",
        )

    def test_cache_hit_with_similar_query(self):
        """Test that semantically similar queries can result in cache hits."""
        self.print_test_header(
            "Similar Query Cache Test",
            "Verifies that semantically similar queries can trigger cache hits",
        )

        session_id = str(uuid.uuid4())

        # Two semantically similar but not identical queries
        original_query = "Explain how photosynthesis works in plants."
        similar_query = (
            "How does the process of photosynthesis function in plant cells?"
        )

        # Get baseline cache metrics
        response = requests.get(ROUTER_METRICS_URL)
        baseline_metrics = response.text
        baseline_hits = (
            extract_metric(baseline_metrics, "llm_router_cache_hits_total") or 0
        )

        # First request with original query
        self.print_subtest_header("Original Query")
        payload1 = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": original_query},
            ],
            "temperature": 0.7,
        }

        self.print_request_info(
            payload=payload1, expectations="Expect: First request to be a cache miss"
        )

        headers = {"Content-Type": "application/json", "X-Session-ID": session_id}

        response1 = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}", headers=headers, json=payload1, timeout=30
        )

        response1_json = response1.json()
        model1 = response1_json.get("model", "unknown")

        self.print_response_info(
            response1,
            {
                "Query": original_query,
                "Selected Model": model1,
                "Session ID": session_id,
                "Cache Status": "Expected Miss",
            },
        )

        # Wait a moment to ensure the first request is processed fully
        time.sleep(2)

        # Second request with similar query
        self.print_subtest_header("Similar Query")
        payload2 = {
            "model": "gemma3:27b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": similar_query},
            ],
            "temperature": 0.7,
        }

        self.print_request_info(
            payload=payload2,
            expectations="Expect: Similar query might trigger a cache hit if similarity threshold is met",
        )

        response2 = requests.post(
            f"{ENVOY_URL}{OPENAI_ENDPOINT}", headers=headers, json=payload2, timeout=30
        )

        response2_json = response2.json()
        model2 = response2_json.get("model", "unknown")

        self.print_response_info(
            response2,
            {
                "Query": similar_query,
                "Selected Model": model2,
                "Session ID": session_id,
                "Cache Status": "Potential Hit",
            },
        )

        # Wait for metrics to update
        time.sleep(2)

        # Check cache metrics
        response = requests.get(ROUTER_METRICS_URL)
        updated_metrics = response.text
        updated_hits = (
            extract_metric(updated_metrics, "llm_router_cache_hits_total") or 0
        )

        passed = (model1 == model2) and (updated_hits > baseline_hits)
        self.print_test_result(
            passed=passed,
            message=(
                f"Cache hits: {baseline_hits} → {updated_hits}, "
                f"Models: {model1} → {model2}, "
                f"{'Cache hit detected' if updated_hits > baseline_hits else 'No cache hit'}"
            ),
        )

        self.assertLess(
            response2.status_code,
            400,
            f"Second request failed with status code {response2.status_code}",
        )

    def test_cache_metrics(self):
        """Test that cache metrics are being recorded properly."""
        self.print_test_header(
            "Cache Metrics Test",
            "Verifies that cache metrics are being properly recorded and exposed",
        )

        # Get baseline metrics
        response = requests.get(ROUTER_METRICS_URL)
        metrics_text = response.text

        # Look for specific cache metrics
        cache_metrics = [
            "llm_router_cache_hits_total",
            "llm_router_cache_misses_total",
            "llm_router_cache_size",
            "llm_router_cache_max_size",
        ]

        metrics_found = {}
        for metric in cache_metrics:
            value = extract_metric(metrics_text, metric)
            if value is not None:
                metrics_found[metric] = value

        self.print_response_info(
            response,
            {"Metrics Found": len(metrics_found), "Total Metrics": len(cache_metrics)},
        )

        # Print detailed metrics information
        for metric, value in metrics_found.items():
            print(f"\nMetric: {metric}")
            print(f"  Value: {value}")

        passed = len(metrics_found) > 0
        self.print_test_result(
            passed=passed,
            message=f"Found {len(metrics_found)} out of {len(cache_metrics)} expected cache metrics",
        )

        self.assertGreater(len(metrics_found), 0, "No cache metrics found")


if __name__ == "__main__":
    unittest.main()
