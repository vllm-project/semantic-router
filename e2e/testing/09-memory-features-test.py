#!/usr/bin/env python3
"""
Memory Features E2E Test Suite — Milvus-First

The memory pipeline is direct per-turn storage: each response triggers
"Q: {user_msg}\\nA: {assistant_response}" chunk storage in Milvus (no LLM
extraction call). Session chunks are stored every 3 turns (window size 5).

Test classes live in the memory_tests package:
  - UserIsolationTest: Cross-user memory leak prevention
  - MemoryInjectionPipelineTest: The fundamental store -> inject contract
  - MemoryContentIntegrityTest: Content preserved in Milvus (no truncation/corruption)
  - SimilarityThresholdTest: Irrelevant NOT injected, relevant IS injected
  - StaleMemoryTest: Contradicting facts baseline (soft-insert, no contradiction detection)
  - PluginCombinationTest: Memory + system_prompt coexistence
  - MemoryStorageTest: Conversation turns stored in Milvus
  - PerDecisionMemoryDisabledTest: Decision with memory.enabled=false skips retrieval
  - PerDecisionThresholdOverrideTest: Decision-level threshold overrides global default

Prerequisites:
  - Milvus running
  - Semantic Router running with memory enabled
  - LLM backend with ECHO mode for reliable verification

To start llm-katan with echo backend:
    LLM_KATAN_BACKEND=echo ./start-llm-katan.sh

Usage:
    python e2e/testing/09-memory-features-test.py

    # With custom endpoint
    ROUTER_ENDPOINT=http://localhost:8888 python e2e/testing/09-memory-features-test.py
"""

import os
import sys
import unittest

import requests
from memory_tests import (
    MemoryContentIntegrityTest,
    MemoryInjectionPipelineTest,
    MemoryStorageTest,
    PerDecisionMemoryDisabledTest,
    PerDecisionThresholdOverrideTest,
    PluginCombinationTest,
    SimilarityThresholdTest,
    StaleMemoryTest,
    UserIsolationTest,
)
from memory_tests.base import HTTP_OK


def run_tests():
    """Run all memory feature tests with detailed output."""
    print("\n" + "=" * 60)
    print("Memory Features Integration Test Suite")
    print("=" * 60)

    router_endpoint = os.environ.get("ROUTER_ENDPOINT", "http://localhost:8888")
    print(f"Router endpoint: {router_endpoint}")

    try:
        response = requests.get(f"{router_endpoint}/health", timeout=10)
        if response.status_code == HTTP_OK:
            print("✅ Router is healthy")
        else:
            print(f"⚠️  Router health check returned {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach router: {e}")
        sys.exit(1)

    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        # P0: Security — run first, fail fast on data leaks
        UserIsolationTest,
        # P1: Pipeline correctness
        MemoryInjectionPipelineTest,
        MemoryContentIntegrityTest,
        SimilarityThresholdTest,
        StaleMemoryTest,
        PluginCombinationTest,
        MemoryStorageTest,
        # P1: Per-decision plugin behavior
        PerDecisionMemoryDisabledTest,
        PerDecisionThresholdOverrideTest,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")

    if failures > 0:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    if errors > 0:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")

    if failures == 0 and errors == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
