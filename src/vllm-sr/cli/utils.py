"""Utility functions for vLLM Semantic Router CLI."""

import logging
import os
import sys
import time
import requests
import yaml
from pathlib import Path


def getLogger(name):
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def find_config_file(path=".", file=None):
    """
    Find the router config file.

    Args:
        path: Directory path to search
        file: Specific file name (optional)

    Returns:
        Absolute path to config file
    """
    if file:
        return os.path.abspath(file)

    # Look for config.yaml in the specified path
    config_path = os.path.join(path, "config.yaml")
    if os.path.exists(config_path):
        return os.path.abspath(config_path)

    # Look for config/config.yaml
    config_path = os.path.join(path, "config", "config.yaml")
    if os.path.exists(config_path):
        return os.path.abspath(config_path)

    raise FileNotFoundError(
        f"Config file not found in {path}. "
        "Please specify the config file path or ensure config.yaml exists."
    )


def load_config(config_file):
    """Load and parse YAML config file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def health_check_endpoint(url, timeout=5):
    """
    Check if an endpoint is healthy.

    Args:
        url: URL to check
        timeout: Request timeout in seconds

    Returns:
        True if healthy, False otherwise
    """
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def wait_for_healthy(url, timeout=120, interval=2, logger=None):
    """
    Wait for an endpoint to become healthy.

    Args:
        url: URL to check
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds
        logger: Logger instance

    Returns:
        True if healthy, False if timeout
    """
    if logger is None:
        logger = getLogger(__name__)

    start_time = time.time()
    while time.time() - start_time < timeout:
        if health_check_endpoint(url):
            logger.info(f"✓ Endpoint {url} is healthy")
            return True
        time.sleep(interval)

    logger.error(f"✗ Endpoint {url} failed to become healthy after {timeout}s")
    return False


def stream_logs_from_file(log_file, follow=False):
    """
    Stream logs from a file.

    Args:
        log_file: Path to log file
        follow: Whether to follow the file (tail -f behavior)
    """
    if not os.path.exists(log_file):
        return

    with open(log_file, "r") as f:
        # Read existing content
        for line in f:
            print(line, end="")

        # Follow mode
        if follow:
            while True:
                line = f.readline()
                if line:
                    print(line, end="")
                else:
                    time.sleep(0.1)


def get_vllm_endpoints(config):
    """Extract vLLM endpoints from config."""
    return config.get("vllm_endpoints", [])


def get_envoy_port(config):
    """Get Envoy listen port from config or use default."""
    from cli.consts import DEFAULT_ENVOY_PORT

    # Try to read from listeners configuration
    listeners = config.get("listeners", [])
    if listeners and len(listeners) > 0:
        # Get port from first listener
        port = listeners[0].get("port")
        if port:
            return port

    # Fall back to default
    return DEFAULT_ENVOY_PORT
