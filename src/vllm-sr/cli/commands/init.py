"""Init command implementation."""

import os
import shutil
import sys
from pathlib import Path
from cli.utils import getLogger

log = getLogger(__name__)


def get_templates_dir():
    """
    Get the templates directory path.

    Templates are now in cli/templates/ directory.
    """
    # Get cli package directory
    cli_dir = Path(__file__).parent.parent
    templates_dir = cli_dir / "templates"

    return templates_dir if templates_dir.exists() else None


def init_command(force: bool = False):
    """
    Initialize vLLM Semantic Router configuration.

    Creates:
    - config.yaml from config.template.yaml
    - .vllm-sr/ directory with template files

    Args:
        force: Force overwrite existing files
    """
    log.info("=" * 60)
    log.info("vLLM Semantic Router - Initialize Configuration")
    log.info("=" * 60)

    # Get current directory
    current_dir = Path.cwd()
    config_file = current_dir / "config.yaml"
    vllm_sr_dir = current_dir / ".vllm-sr"

    # Get templates directory
    templates_dir = get_templates_dir()

    if templates_dir is None or not templates_dir.exists():
        log.error(f"Templates directory not found: {templates_dir}")
        log.error("Please ensure vllm-sr is installed correctly.")
        return False

    # Copy config.template.yaml to config.yaml
    template_config = templates_dir / "config.template.yaml"
    if not template_config.exists():
        log.error(f"Config template not found: {template_config}")
        return False

    if config_file.exists():
        log.warning(f"config.yaml already exists, overwriting...")

    shutil.copy2(template_config, config_file)
    log.info(f"✓ Created config.yaml")

    # Create .vllm-sr directory
    if vllm_sr_dir.exists():
        log.warning(f".vllm-sr/ directory already exists, overwriting...")
        shutil.rmtree(vllm_sr_dir)

    vllm_sr_dir.mkdir(exist_ok=True)
    log.info(f"✓ Created .vllm-sr/ directory")

    # Copy all template files to .vllm-sr/
    copied_files = []
    for template_file in templates_dir.iterdir():
        if template_file.is_file() and template_file.name != "config.template.yaml":
            dest_file = vllm_sr_dir / template_file.name
            shutil.copy2(template_file, dest_file)
            copied_files.append(template_file.name)
            log.info(f"  • Copied {template_file.name}")

    log.info("=" * 60)
    log.info("✓ Initialization complete!")
    log.info("")
    log.info("Created files:")
    log.info(f"  • config.yaml")
    log.info(f"  • .vllm-sr/ ({len(copied_files)} files)")
    log.info("")
    log.info("Next steps:")
    log.info("  1. Edit config.yaml to configure your setup")
    log.info("  2. Start service: vllm-sr serve")
    log.info("=" * 60)

    return True
