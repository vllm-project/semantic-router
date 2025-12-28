"""vLLM Semantic Router CLI main entry point."""

import click
import os
import sys
from pathlib import Path
from cli import __version__
from cli.utils import getLogger
from cli.core import start_vllm_sr, stop_vllm_sr, show_logs, show_status
from cli.consts import (
    VLLM_SR_DOCKER_IMAGE_DEFAULT,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    DEFAULT_IMAGE_PULL_POLICY,
)
from cli.commands.init import init_command
from cli.commands.config import config_command

log = getLogger(__name__)

# ASCII logo
logo = r"""
       _ _     __  __       ____  ____
__   _| | |_ _|  \/  |     / ___||  _ \
\ \ / / | | | | |\/| |_____\___ \| |_) |
 \ V /| | | |_| | |  |_____|___) |  _ <
  \_/ |_|_|\__,_|_|  |     |____/|_| \_\

vLLM Semantic Router - Intelligent routing for vLLM
"""


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit.")
@click.pass_context
def main(ctx, version):
    """vLLM Semantic Router CLI - Intelligent routing and caching for vLLM endpoints."""
    if version:
        click.echo(f"vllm-sr version: {__version__}")
        ctx.exit()

    if ctx.invoked_subcommand is None:
        click.echo(logo)
        click.echo(ctx.get_help())


@click.command()
def init():
    """
    Initialize vLLM Semantic Router configuration.

    Creates config.yaml and .vllm-sr/ directory with template files.

    Examples:
        vllm-sr init
    """
    try:
        init_command()
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


@click.command()
@click.option(
    "--config",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
@click.option(
    "--image",
    default=None,
    help=f"Docker image to use (default: {VLLM_SR_DOCKER_IMAGE_DEFAULT})",
)
@click.option(
    "--image-pull-policy",
    type=click.Choice(
        [
            IMAGE_PULL_POLICY_ALWAYS,
            IMAGE_PULL_POLICY_IF_NOT_PRESENT,
            IMAGE_PULL_POLICY_NEVER,
        ],
        case_sensitive=False,
    ),
    default=DEFAULT_IMAGE_PULL_POLICY,
    help=f"Image pull policy: always, ifnotpresent, never (default: {DEFAULT_IMAGE_PULL_POLICY})",
)
def serve(config, image, image_pull_policy):
    """
    Start vLLM Semantic Router.

    Ports are configured in config.yaml under 'listeners' section.

    Examples:
        # Basic usage (uses config.yaml)
        vllm-sr serve

        # Custom config file
        vllm-sr serve --config my-config.yaml

        # Custom image
        vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

        # Pull policy
        vllm-sr serve --image-pull-policy always
    """
    try:
        # Check if config file exists
        config_path = Path(config)
        if not config_path.exists():
            log.error(f"Config file not found: {config}")
            log.error("Run 'vllm-sr init' to create a config file")
            sys.exit(1)

        log.info(f"Using config file: {config}")

        # Collect environment variables to pass to container
        env_vars = {}

        # HuggingFace related environment variables
        hf_env_vars = ["HF_ENDPOINT", "HF_TOKEN", "HF_HOME", "HF_HUB_CACHE"]
        for var in hf_env_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
                # Mask sensitive tokens in logs
                if var == "HF_TOKEN":
                    log.info(f"Passing environment variable: {var}=***")
                else:
                    log.info(f"Passing environment variable: {var}={os.environ[var]}")

        # Start container
        start_vllm_sr(
            config_file=str(config_path.absolute()),
            env_vars=env_vars,
            image=image,
            pull_policy=image_pull_policy,
        )

    except KeyboardInterrupt:
        log.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


@click.command()
@click.argument("config_type", type=click.Choice(["envoy", "router"]))
@click.option(
    "--config",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
def config(config_type, config):
    """
    Print generated configuration.

    Examples:
        vllm-sr config envoy
        vllm-sr config router
        vllm-sr config envoy --config my-config.yaml
    """
    try:
        config_command(config_type, config)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


@click.command()
@click.argument("service", type=click.Choice(["envoy", "router", "all"]), default="all")
def status(service):
    """
    Show status of vLLM Semantic Router services.

    Examples:
        vllm-sr status          # Show all services
        vllm-sr status all      # Show all services
        vllm-sr status envoy    # Show envoy status
        vllm-sr status router   # Show router status
    """
    try:
        show_status(service)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


@click.command()
@click.argument("service", type=click.Choice(["envoy", "router"]))
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def logs(service, follow):
    """
    Show logs from vLLM Semantic Router service.

    Examples:
        vllm-sr logs envoy
        vllm-sr logs router
        vllm-sr logs envoy --follow
        vllm-sr logs router -f
    """
    try:
        show_logs(service, follow=follow)
    except KeyboardInterrupt:
        log.info("\nLog streaming stopped")
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


@click.command()
def stop():
    """
    Stop vLLM Semantic Router.

    Examples:
        vllm-sr stop
    """
    try:
        stop_vllm_sr()
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


# Register commands
main.add_command(init)
main.add_command(serve)
main.add_command(config)
main.add_command(status)
main.add_command(logs)
main.add_command(stop)


if __name__ == "__main__":
    main()
