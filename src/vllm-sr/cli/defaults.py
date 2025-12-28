"""Default router configuration loader."""

import yaml
from pathlib import Path


def _get_defaults_path() -> Path:
    """
    Get path to router-defaults.yaml template.

    Returns:
        Path: Path to router-defaults.yaml
    """
    # Get the directory where this file is located
    cli_dir = Path(__file__).parent
    # Templates are now in cli/templates/
    templates_dir = cli_dir / "templates"
    defaults_path = templates_dir / "router-defaults.yaml"

    if not defaults_path.exists():
        raise FileNotFoundError(f"Router defaults not found: {defaults_path}")

    return defaults_path


def load_embedded_defaults() -> dict:
    """
    Load default configuration from router-defaults.yaml template.

    Returns:
        dict: Default router configuration
    """
    defaults_path = _get_defaults_path()
    with open(defaults_path, "r") as f:
        return yaml.safe_load(f)


def get_defaults_yaml() -> str:
    """
    Get default configuration as YAML string.

    Returns:
        str: Default configuration in YAML format
    """
    defaults_path = _get_defaults_path()
    with open(defaults_path, "r") as f:
        return f.read()
