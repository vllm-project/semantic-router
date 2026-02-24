"""vLLM Semantic Router CLI main entry point."""

import atexit
import click
import os
import shutil
import sys
import tempfile
import webbrowser
import yaml
from pathlib import Path
from cli import __version__
from cli.utils import getLogger

# Track temp directories for cleanup
_temp_dirs = []


def _cleanup_temp_dirs():
    """Clean up all temp directories created by the CLI."""
    for temp_dir in _temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass  # Best effort cleanup


atexit.register(_cleanup_temp_dirs)
from cli.core import start_vllm_sr, stop_vllm_sr, show_logs, show_status
from cli.consts import (
    VLLM_SR_DOCKER_IMAGE_DEFAULT,
    VLLM_SR_DOCKER_NAME,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    DEFAULT_IMAGE_PULL_POLICY,
)
from cli.commands.init import init_command
from cli.commands.config import config_command
from cli.commands.validate import validate_command
from cli.docker_cli import docker_container_status

log = getLogger(__name__)


def inject_algorithm_into_config(config_path: Path, algorithm: str) -> Path:
    """
    Inject algorithm type into all decisions in the config file.

    This implements the CLI translation logic required by issue #1103:
    vllm-sr serve --algorithm elo
    => Translates to algorithm.type: elo in router config

    Args:
        config_path: Path to the original config.yaml
        algorithm: Algorithm type (e.g., "elo", "router_dc", "automix")

    Returns:
        Path: Path to the modified config file (temporary file)
    """
    # Load the original config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Inject algorithm into each decision
    decisions = config.get("decisions", [])
    for decision in decisions:
        if "algorithm" not in decision:
            decision["algorithm"] = {}
        decision["algorithm"]["type"] = algorithm
        log.info(
            f"  Injected algorithm.type={algorithm} into decision '{decision.get('name', 'unnamed')}'"
        )

    # Write to a temporary file (tracked for cleanup)
    temp_dir = tempfile.mkdtemp(prefix="vllm-sr-")
    _temp_dirs.append(temp_dir)  # Track for atexit cleanup
    temp_config_path = Path(temp_dir) / "config-with-algorithm.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    log.info(f"✓ Created config with algorithm: {temp_config_path}")

    return temp_config_path


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
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing config.yaml and .vllm-sr without prompting.",
)
def init(force):
    """
    Initialize vLLM Semantic Router configuration.

    Creates config.yaml and .vllm-sr/ directory with template files.

    Examples:
        vllm-sr init
        vllm-sr init --force
    """
    try:
        init_command(force=force)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


# Valid algorithm types for model selection
# Original algorithms (PR #1089, #1104)
# RL-driven algorithms (PR #1196 - issue #994)
ALGORITHM_TYPES = [
    "static",  # First model (default)
    "elo",  # Elo rating with feedback
    "router_dc",  # Embedding similarity
    "automix",  # POMDP cost-quality optimization
    "hybrid",  # Combined methods
    "thompson",  # Thompson Sampling (RL-driven)
    "gmtrouter",  # Graph-based personalized routing
    "router_r1",  # LLM-as-router with think/route
]


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
@click.option(
    "--readonly",
    is_flag=True,
    default=False,
    help="Run dashboard in read-only mode (disable config editing, allow playground only)",
)
@click.option(
    "--minimal",
    is_flag=True,
    default=False,
    help="Start in minimal mode: only router + envoy, no dashboard or observability (Jaeger, Prometheus, Grafana)",
)
@click.option(
    "--platform",
    default=None,
    help="Platform branding (e.g., 'amd' for AMD GPU deployments)",
)
@click.option(
    "--algorithm",
    type=click.Choice(ALGORITHM_TYPES, case_sensitive=False),
    default=None,
    help="Model selection algorithm: static (default), elo (rating-based), "
    "router_dc (embedding similarity), automix (cost-quality optimization), "
    "hybrid (combined methods). Overrides config file setting.",
)
def serve(config, image, image_pull_policy, readonly, minimal, platform, algorithm):
    """
    Start vLLM Semantic Router.

    Ports are configured in config.yaml under 'listeners' section.

    MODEL SELECTION ALGORITHMS:

    \b
    static     - Use first configured model (default, no learning)
    elo        - Rating-based selection using user feedback
    router_dc  - Query-model matching via embedding similarity
    automix    - Cost-quality optimization using POMDP
    hybrid     - Combine multiple methods with configurable weights
    thompson   - Thompson Sampling with exploration/exploitation (RL-driven)
    gmtrouter  - Graph neural network for personalized routing (RL-driven)
    router_r1  - LLM-as-router with think/route actions (RL-driven)

    Examples:
        # Basic usage (uses config.yaml)
        vllm-sr serve

        # Custom config file
        vllm-sr serve --config my-config.yaml

        # Use Elo rating selection (learns from feedback)
        vllm-sr serve --algorithm elo

        # Use cost-optimized selection
        vllm-sr serve --algorithm automix

        # Custom image
        vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

        # Pull policy
        vllm-sr serve --image-pull-policy always

        # Read-only dashboard (for public beta)
        vllm-sr serve --readonly

        # Minimal mode (no dashboard, no observability)
        vllm-sr serve --minimal

        # Platform branding (for AMD deployments)
        vllm-sr serve --platform amd
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

        # API keys for model providers
        api_key_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        for var in api_key_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
                log.info(f"Passing environment variable: {var}=***")

        # Minimal mode: disable dashboard and observability
        if minimal:
            env_vars["DISABLE_DASHBOARD"] = "true"
            log.info("Minimal mode: ENABLED (no dashboard, no observability)")
            if readonly:
                log.warning(
                    "--readonly is ignored in minimal mode (dashboard is disabled)"
                )

        # Dashboard read-only mode
        if readonly and not minimal:
            env_vars["DASHBOARD_READONLY"] = "true"
            log.info("Dashboard read-only mode: ENABLED")

        # Platform branding
        if platform:
            env_vars["DASHBOARD_PLATFORM"] = platform
            log.info(f"Platform branding: {platform}")

        # Model selection algorithm override
        # This injects algorithm.type into all decisions in the config
        # Implements CLI translation logic per issue #1103 acceptance criteria
        effective_config_path = config_path
        if algorithm:
            algo = algorithm.lower()
            log.info(f"Model selection algorithm: {algo}")

            # Inject algorithm into config (creates temp file with modified config)
            effective_config_path = inject_algorithm_into_config(config_path, algo)

            # Log algorithm-specific hints
            if algo == "elo":
                log.info("  Tip: Submit feedback via POST /api/v1/feedback")
            elif algo == "router_dc":
                log.info("  Tip: Ensure models have 'description' fields")
            elif algo == "automix":
                log.info("  Tip: Configure model 'pricing' for cost optimization")
            elif algo == "hybrid":
                log.info("  Tip: Configure weights in decision.algorithm.hybrid")
            elif algo == "thompson":
                log.info("  Tip: Balances exploration vs exploitation automatically")
            elif algo == "gmtrouter":
                log.info("  Tip: Learns user preferences via graph neural network")
            elif algo == "router_r1":
                log.info("  Tip: Requires Router-R1 server (see training docs)")

        # Start container
        start_vllm_sr(
            config_file=str(effective_config_path.absolute()),
            env_vars=env_vars,
            image=image,
            pull_policy=image_pull_policy,
            enable_observability=not minimal,
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
@click.option(
    "--config",
    default="config.yaml",
    help="Path to config file (default: config.yaml)",
)
def validate(config):
    """
    Validate configuration file.

    Examples:
        vllm-sr validate                    # Uses config.yaml
        vllm-sr validate --config my-config.yaml  # Uses my-config.yaml
    """
    try:
        validate_command(config)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


@click.command()
@click.argument(
    "service", type=click.Choice(["envoy", "router", "dashboard", "all"]), default="all"
)
def status(service):
    """
    Show status of vLLM Semantic Router services.

    Examples:
        vllm-sr status              # Show all services
        vllm-sr status all          # Show all services
        vllm-sr status envoy        # Show envoy status
        vllm-sr status router       # Show router status
        vllm-sr status dashboard    # Show dashboard status
    """
    try:
        show_status(service)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


@click.command()
@click.argument("service", type=click.Choice(["envoy", "router", "dashboard"]))
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
def logs(service, follow):
    """
    Show logs from vLLM Semantic Router service.

    Examples:
        vllm-sr logs envoy
        vllm-sr logs router
        vllm-sr logs dashboard
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


@click.command()
@click.option("--no-open", is_flag=True, help="Don't open browser, just show URL")
def dashboard(no_open):
    """
    Open the dashboard in your default web browser.

    Examples:
        vllm-sr dashboard
        vllm-sr dashboard --no-open
    """
    try:
        # Check if container is running
        status = docker_container_status(VLLM_SR_DOCKER_NAME)
        if status != "running":
            log.error("vLLM Semantic Router is not running")
            log.info("Start it with: vllm-sr serve")
            sys.exit(1)

        dashboard_url = "http://localhost:8700"

        if no_open:
            log.info(f"Dashboard URL: {dashboard_url}")
        else:
            log.info(f"Opening dashboard: {dashboard_url}")
            webbrowser.open(dashboard_url)
            log.info("✓ Dashboard opened in browser")

    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


# Register commands
main.add_command(init)
main.add_command(serve)
main.add_command(config)
main.add_command(validate)
main.add_command(status)
main.add_command(logs)
main.add_command(stop)
main.add_command(dashboard)


if __name__ == "__main__":
    main()
