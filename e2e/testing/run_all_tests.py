#!/usr/bin/env python3
"""
Run all tests in sequence.

This script runs all the test files in the tests directory in order,
providing a complete test of the Semantic Router system.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import argparse
import glob
import json
import os
import sys
import time
import unittest

import requests


def check_services():
    """Check if required services are running."""
    services = [
        # Instead of checking /healthz, we'll check the actual chat endpoint with a minimal POST request
        {"name": "Envoy Proxy", "check_func": check_envoy_running},
        {"name": "Router Metrics", "url": "http://localhost:9190/metrics"},
    ]

    # Check for OpenAI API key if RAG OpenAI tests are enabled
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI API key is set (RAG OpenAI tests can run)")
    else:
        print("⚠️  OpenAI API key not set (RAG OpenAI tests will be skipped)")

    all_running = True
    for service in services:
        try:
            if "url" in service:
                # Standard GET request check
                response = requests.get(service["url"], timeout=2)
                if response.status_code == 200:
                    print(f"✅ {service['name']} is running")
                else:
                    print(
                        f"❌ {service['name']} returned status code {response.status_code}"
                    )
                    all_running = False
            elif "check_func" in service:
                # Custom check function
                if service["check_func"]():
                    print(f"✅ {service['name']} is running")
                else:
                    print(f"❌ {service['name']} is not responding")
                    all_running = False
        except requests.exceptions.ConnectionError:
            print(f"❌ {service['name']} is not running")
            all_running = False

    return all_running


def check_envoy_running():
    """Check if Envoy is running by making a minimal POST request to the chat completions endpoint."""
    try:
        # Simple request with minimal content
        payload = {
            "model": "Qwen/Qwen2-0.5B-Instruct",
            "messages": [{"role": "user", "content": "test"}],
        }
        response = requests.post(
            "http://localhost:8801/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,  # Increased timeout to 30 seconds to match behavior of make test-prompt
        )

        # If we get any response (even an error from the backend), Envoy is running
        return response.status_code < 500
    except requests.exceptions.ConnectionError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all Semantic Router tests in sequence"
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Only check if services are running"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip checking if services are running",
    )
    parser.add_argument("--pattern", default="*.py", help="Test file pattern to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Get the directory where this script is located
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tests_dir)

    # Check if services are running
    if not args.skip_check:
        services_running = check_services()
        if args.check_only:
            return 0 if services_running else 1
        if not services_running:
            print("\n❌ Some required services are not running")
            print("Please start the services with:")
            print("  make run-envoy  # In one terminal")
            print("  make run-router # In another terminal")
            return 1

    # Find all test files matching the pattern
    test_files = sorted(glob.glob(args.pattern))

    # Filter out this script and __init__.py
    test_files = [
        f
        for f in test_files
        if f != os.path.basename(__file__) and not f.startswith("__")
    ]

    if not test_files:
        print(f"No test files found matching pattern '{args.pattern}'")
        return 1

    print(f"\nRunning {len(test_files)} test files:")
    for file in test_files:
        print(f"  - {file}")

    # Run each test file
    results = []
    for i, test_file in enumerate(test_files):
        test_module = os.path.splitext(test_file)[0]

        print(f"\n{'='*80}")
        print(f"Running test file {i+1}/{len(test_files)}: {test_file}")
        print(f"{'='*80}")

        # Add a small delay between test files
        if i > 0:
            time.sleep(2)

        # Create and run a test suite for this file
        try:
            # Import the test module
            __import__(test_module)
            module = sys.modules[test_module]

            # Create a test suite from the module
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)

            # Run the tests
            result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(
                suite
            )
            results.append((test_file, result.wasSuccessful()))
        except Exception as e:
            print(f"Error running {test_file}: {e}")
            results.append((test_file, False))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_file, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_file}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
