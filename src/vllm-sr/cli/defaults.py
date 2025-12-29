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


def load_defaults(output_dir: str = None) -> dict:
    """
    Load default configuration, preferring local router-defaults.yaml if it exists.

    Priority:
    1. Local .vllm-sr/router-defaults.yaml (if output_dir provided and file exists)
    2. Embedded router-defaults.yaml template

    Args:
        output_dir: Optional output directory to check for local router-defaults.yaml

    Returns:
        dict: Default router configuration
    """
    # Check for local router-defaults.yaml first
    if output_dir:
        local_defaults_path = Path(output_dir) / "router-defaults.yaml"
        if local_defaults_path.exists():
            with open(local_defaults_path, "r") as f:
                return yaml.safe_load(f)

    # Fall back to embedded defaults
    return load_embedded_defaults()


def get_defaults_yaml() -> str:
    """
    Get default configuration as YAML string.

    Returns:
        str: Default configuration in YAML format
    """
    defaults_path = _get_defaults_path()
    with open(defaults_path, "r") as f:
        return f.read()
