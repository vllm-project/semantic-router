"""
CLI entrypoint for Model Manager.

Usage:
    python -m model_manager --config models.yaml
    python -m model_manager --config models.yaml --cache-dir /path/to/cache
    python -m model_manager --verify-only
    python -m model_manager --list

    # CI mode - auto-selects minimal config
    CI_MINIMAL_MODELS=true python -m model_manager
"""

import argparse
import logging
import os
import sys

from . import ModelManager, ensure_models
from .errors import ModelManagerError


def get_default_config() -> str:
    """Get default config path based on CI_MINIMAL_MODELS environment variable."""
    ci_minimal = os.environ.get("CI_MINIMAL_MODELS", "").lower()
    if ci_minimal in ("true", "1", "yes"):
        return "config/model_manager/models.minimal.yaml"
    return "config/model_manager/models.yaml"


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Model Manager - Download, verify, and cache ML models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all models defined in models.yaml
    python -m model_manager --config models.yaml

    # Download specific model
    python -m model_manager --config models.yaml --model category_classifier

    # Only verify existing models, don't download
    python -m model_manager --config models.yaml --verify-only

    # List all configured models
    python -m model_manager --config models.yaml --list
""",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to models.yaml configuration file (default: auto-detected based on CI_MINIMAL_MODELS)",
    )
    parser.add_argument(
        "--cache-dir",
        help="Override cache directory from config",
    )
    parser.add_argument(
        "--model",
        help="Only ensure a specific model by ID",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models, don't download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="List all configured models and exit",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove all cached models defined in config",
    )
    parser.add_argument(
        "--clean-model",
        help="Remove a specific cached model by ID",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Determine config file
    config_path = args.config if args.config else get_default_config()
    logger.info(f"Using config: {config_path}")

    try:
        manager = ModelManager.from_config(config_path)

        if args.cache_dir:
            manager.config.cache_dir = args.cache_dir

        # List models and exit
        if args.list_models:
            print(f"Models configured in {config_path}:")
            print(f"Cache directory: {manager.config.cache_dir}")
            print(f"Verification: {manager.config.verify}")
            print()
            for spec in manager.config.models:
                from .cache import is_cached, get_cache_info

                cached = is_cached(spec, manager.config.cache_dir)
                status = "✓ cached" if cached else "✗ not cached"
                print(f"  [{status}] {spec.id}")
                print(f"           repo: {spec.repo_id}")
                print(f"           revision: {spec.revision}")
                if cached:
                    info = get_cache_info(spec, manager.config.cache_dir)
                    if info:
                        size_mb = info["size_bytes"] / (1024 * 1024)
                        print(
                            f"           size: {size_mb:.1f} MB ({info['file_count']} files)"
                        )
                print()
            return 0

        # Clean specific model
        if args.clean_model:
            from .cache import clear_cache

            spec = manager.get_model_spec(args.clean_model)
            if spec is None:
                logger.error(f"Model '{args.clean_model}' not found in configuration")
                return 1
            if clear_cache(spec, manager.config.cache_dir):
                logger.info(f"✓ Removed cached model '{args.clean_model}'")
            else:
                logger.info(f"Model '{args.clean_model}' was not cached")
            return 0

        # Clean all models
        if args.clean:
            from .cache import clear_cache, is_cached

            cleaned = 0
            for spec in manager.config.models:
                if is_cached(spec, manager.config.cache_dir):
                    clear_cache(spec, manager.config.cache_dir)
                    logger.info(f"✓ Removed '{spec.id}'")
                    cleaned += 1
            logger.info(f"Cleaned {cleaned} models")
            return 0

        # Verify only mode
        if args.verify_only:
            from .verifier import verify_model
            from .cache import get_model_path

            all_valid = True
            for spec in manager.config.models:
                model_path = get_model_path(spec, manager.config.cache_dir)
                if verify_model(model_path, manager.config.verify):
                    logger.info(f"✓ Model '{spec.id}' verified")
                else:
                    logger.error(f"✗ Model '{spec.id}' verification failed")
                    all_valid = False

            return 0 if all_valid else 1

        # Ensure specific model
        if args.model:
            path = manager.ensure_model(args.model)
            logger.info(f"Model '{args.model}' ready at {path}")
            return 0

        # Ensure all models
        results = manager.ensure_all()
        logger.info(f"All {len(results)} models ready")
        return 0

    except ModelManagerError as e:
        logger.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
