"""Show defaults command implementation."""

import sys

from cli.defaults import get_defaults_yaml
from cli.utils import getLogger

log = getLogger(__name__)


def show_defaults_command(output_file: str = None):
    """
    Show embedded default configuration.

    Args:
        output_file: Optional output file path
    """
    defaults_yaml = get_defaults_yaml()

    if output_file:
        try:
            with open(output_file, "w") as f:
                f.write(defaults_yaml)
            log.info(f"✓ Defaults written to: {output_file}")
        except Exception as e:
            log.error(f"Failed to write defaults: {e}")
            sys.exit(1)
    else:
        print(defaults_yaml)
