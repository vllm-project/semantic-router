"""
CLI entrypoint for running model_manager as a module.

Usage:
    python -m model_manager --config models.yaml
"""

from .cli import main

if __name__ == "__main__":
    main()
