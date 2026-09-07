"""Utility functions for vLLM Semantic Router CLI."""

import logging
import os

import yaml

from cli.terminal import TerminalLogHandler


def get_logger(name):
    """Get a logger that follows the shared CLI terminal contract."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = TerminalLogHandler()
        formatter = logging.Formatter("%(message)s")
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
    with open(config_file) as f:
        return yaml.safe_load(f)
