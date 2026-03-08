#!/usr/bin/env python3
"""
run_cli_tests.py - Main runner for vLLM-SR CLI end-to-end tests.

This script runs all CLI tests in sequence, providing:
- Pre-flight checks (Docker, vllm-sr CLI installed)
- Test discovery and execution
- Detailed reporting
- Integration test support (when RUN_INTEGRATION_TESTS=true)

Usage:
    python run_cli_tests.py                    # Run all unit tests
    python run_cli_tests.py --integration      # Run including integration tests
    python run_cli_tests.py --verbose          # Verbose output
    python run_cli_tests.py --pattern "init"   # Run tests matching pattern

"""

import argparse
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

DEFAULT_IMAGE = "ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
SECTION_DIVIDER = "=" * 60


def print_section_header(title: str) -> None:
    print(SECTION_DIVIDER)
    print(title)
    print(SECTION_DIVIDER)


def detect_container_runtime() -> str | None:
    """Return the available container runtime, printing the detection result."""
    if shutil.which("docker"):
        print("✅ Docker is installed")
        return "docker"
    if shutil.which("podman"):
        print("✅ Podman is installed")
        return "podman"
    print("❌ Neither Docker nor Podman found")
    return None


def check_container_runtime_access(
    container_runtime: str, *, require_runtime_access: bool
) -> bool:
    """Check whether the detected runtime daemon is reachable."""
    try:
        result = subprocess.run(
            [container_runtime, "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        if require_runtime_access:
            print(f"❌ Failed to check {container_runtime}: {exc}")
            return False
        print(f"⚠️  Failed to check {container_runtime}: {exc}")
        print("   Continuing because integration tests are disabled for this run.")
        return True

    if result.returncode == 0:
        print(f"✅ {container_runtime.capitalize()} daemon is running")
        return True
    if require_runtime_access:
        print(f"❌ {container_runtime.capitalize()} daemon is not accessible")
        return False
    print(f"⚠️  {container_runtime.capitalize()} daemon is not accessible")
    print(
        "   Continuing because fast/unit tests only require the CLI binary and "
        "container runtime client."
    )
    return True


def check_cli_installation() -> bool:
    """Verify that the CLI is installed and print its version when available."""
    if not shutil.which("vllm-sr"):
        print("❌ vllm-sr CLI is not installed")
        print("   Install with: pip install -e src/vllm-sr")
        return False

    print("✅ vllm-sr CLI is installed")
    try:
        result = subprocess.run(
            ["vllm-sr", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return True

    version = result.stdout.strip() or result.stderr.strip()
    if version:
        print(f"   Version: {version}")
    return True


def report_local_image_status(container_runtime: str) -> None:
    """Print whether the default local image is already present."""
    try:
        result = subprocess.run(
            [container_runtime, "images", "-q", DEFAULT_IMAGE],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return

    if result.stdout.strip():
        print("✅ vllm-sr Docker image is available locally")
        return
    print("⚠️  vllm-sr Docker image not found locally")
    print("   Some tests may need to pull the image")


def configure_integration_mode(integration: bool) -> None:
    """Set the integration-test environment toggle for the test suite."""
    if integration:
        os.environ["RUN_INTEGRATION_TESTS"] = "true"
        print("\n🔧 Integration tests ENABLED")
        return
    os.environ["RUN_INTEGRATION_TESTS"] = "false"
    print("\n🔧 Integration tests DISABLED (use --integration to enable)")


def print_test_summary(result: unittest.TestResult) -> bool:
    """Print the final unit test summary and return success state."""
    print(f"\n{SECTION_DIVIDER}")
    print("Test Summary")
    print(SECTION_DIVIDER)

    executed = result.testsRun - len(result.skipped)
    print(f"Tests executed: {executed}")
    print(f"Tests skipped: {len(result.skipped)}")
    print(f"Total tests: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True

    print("\n❌ Some tests failed")
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[0]}")
    return False


def check_prerequisites(*, require_runtime_access: bool) -> bool:
    """Check that all prerequisites are met for running CLI tests."""
    print_section_header("Pre-flight Checks")
    container_runtime = detect_container_runtime()
    all_ok = container_runtime is not None

    if container_runtime:
        all_ok = (
            check_container_runtime_access(
                container_runtime,
                require_runtime_access=require_runtime_access,
            )
            and all_ok
        )
        report_local_image_status(container_runtime)

    all_ok = check_cli_installation() and all_ok
    print(SECTION_DIVIDER)
    return all_ok


def discover_tests(pattern: str | None = None) -> unittest.TestSuite:
    """Discover and load CLI tests."""
    # Get the directory containing this script
    test_dir = Path(__file__).parent

    # Create test loader
    loader = unittest.TestLoader()

    if pattern:
        # Load tests matching pattern
        suite = unittest.TestSuite()
        for test_file in test_dir.glob("test_*.py"):
            if pattern.lower() in test_file.name.lower():
                module_name = test_file.stem
                try:
                    # Import and load tests from module
                    module = __import__(module_name)
                    suite.addTests(loader.loadTestsFromModule(module))
                except Exception as e:
                    print(f"Warning: Failed to load {test_file}: {e}")
    else:
        # Load all tests
        suite = loader.discover(str(test_dir), pattern="test_*.py")

    return suite


def run_tests(
    pattern: str | None = None,
    verbose: bool = False,
    integration: bool = False,
) -> bool:
    """
    Run CLI tests.

    Args:
        pattern: Optional pattern to filter tests
        verbose: Enable verbose output
        integration: Include integration tests

    Returns:
        True if all tests passed, False otherwise
    """
    configure_integration_mode(integration)

    # Change to test directory
    test_dir = Path(__file__).parent
    original_dir = os.getcwd()
    os.chdir(test_dir)

    try:
        # Discover tests
        suite = discover_tests(pattern)

        test_count = suite.countTestCases()
        print(f"\n📋 Found {test_count} test(s) to run")
        if test_count == 0:
            print("No tests found!")
            return False

        print(f"\n{SECTION_DIVIDER}")
        print("Running Tests")
        print(f"{SECTION_DIVIDER}\n")

        verbosity = 2 if verbose else 1
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return print_test_summary(result)

    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM-SR CLI end-to-end tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_cli_tests.py                    # Run all unit tests
    python run_cli_tests.py --integration      # Include integration tests
    python run_cli_tests.py --pattern init     # Run tests matching 'init'
    python run_cli_tests.py -v                 # Verbose output

Integration Tests:
    Integration tests require a working Docker image and may take several
    minutes to complete. They test the full serve workflow including
    container startup and health checks.

    To run integration tests:
        python run_cli_tests.py --integration

    Or set the environment variable:
        RUN_INTEGRATION_TESTS=true python run_cli_tests.py
""",
    )

    parser.add_argument(
        "--pattern",
        "-p",
        help="Pattern to filter test files (e.g., 'init' matches test_vllm_sr_init.py)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--integration",
        "-i",
        action="store_true",
        help="Include integration tests (requires Docker image)",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip pre-flight checks",
    )

    args = parser.parse_args()

    # Run pre-flight checks
    if not args.skip_checks and not check_prerequisites(
        require_runtime_access=args.integration
    ):
        print("\n❌ Pre-flight checks failed. Fix issues above and retry.")
        return 1

    # Run tests
    success = run_tests(
        pattern=args.pattern,
        verbose=args.verbose,
        integration=args.integration,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
