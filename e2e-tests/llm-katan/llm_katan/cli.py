"""
Command Line Interface for LLM Katan

Provides easy-to-use CLI for starting LLM Katan servers with different configurations.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import asyncio
import logging
import sys
from typing import Optional

import click

from .config import ServerConfig
from .server import run_server

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "unknown"

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="Model name to load (e.g., 'Qwen/Qwen3-0.6B')",
)
@click.option(
    "--served-model-name",
    "--name",
    "-n",
    help="Model name to serve via API (defaults to model name)",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    type=int,
    help="Port to serve on (default: 8000)",
)
@click.option(
    "--host",
    "-h",
    default="0.0.0.0",
    help="Host to bind to (default: 0.0.0.0)",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["transformers", "vllm"], case_sensitive=False),
    default="transformers",
    help="Backend to use (default: transformers)",
)
@click.option(
    "--max-tokens",
    "--max",
    default=512,
    type=int,
    help="Maximum tokens to generate (default: 512)",
)
@click.option(
    "--temperature",
    "-t",
    default=0.7,
    type=float,
    help="Sampling temperature (default: 0.7)",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    default="auto",
    help="Device to use (default: auto)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Log level (default: INFO)",
)
@click.version_option(version=__version__, prog_name="LLM Katan")
def main(
    model: str,
    served_model_name: Optional[str],
    port: int,
    host: str,
    backend: str,
    max_tokens: int,
    temperature: float,
    device: str,
    log_level: str,
):
    """
    LLM Katan - Lightweight LLM Server for Testing

    Start a lightweight LLM server using real tiny models for testing and development.

    Examples:
        # Basic usage
        llm-katan --model Qwen/Qwen3-0.6B

        # Custom port and served model name
        llm-katan --model Qwen/Qwen3-0.6B --port 8001 --name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Use vLLM backend
        llm-katan --model Qwen/Qwen3-0.6B --backend vllm

        # Force CPU usage
        llm-katan --model Qwen/Qwen3-0.6B --device cpu
    """
    # Set log level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    # Create configuration
    config = ServerConfig(
        model_name=model,
        served_model_name=served_model_name,
        port=port,
        host=host,
        backend=backend.lower(),
        max_tokens=max_tokens,
        temperature=temperature,
        device=device.lower(),
    )

    # Print startup information
    click.echo("🚀 Starting LLM Katan server...")
    click.echo(f"   Model: {config.model_name}")
    click.echo(f"   Served as: {config.served_model_name}")
    click.echo(f"   Backend: {config.backend}")
    click.echo(f"   Device: {config.device_auto}")
    click.echo(f"   Server: http://{config.host}:{config.port}")
    click.echo("")

    # Validate backend availability
    if config.backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            click.echo(
                "❌ vLLM backend selected but vLLM is not installed. "
                "Install with: pip install vllm",
                err=True,
            )
            sys.exit(1)

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        click.echo(
            "❌ Required dependencies missing. "
            "Install with: pip install transformers torch",
            err=True,
        )
        sys.exit(1)

    # Run the server
    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        click.echo("\n🛑 Server stopped by user")
    except Exception as e:
        click.echo(f"❌ Server error: {str(e)}", err=True)
        sys.exit(1)


@click.command()
@click.option(
    "--model",
    "-m",
    default="Qwen/Qwen3-0.6B",
    help="Default model to use",
)
def quickstart(model: str):
    """Quick start with default settings"""
    click.echo("🚀 Quick starting LLM Katan with default settings...")

    config = ServerConfig(
        model_name=model,
        port=8000,
        backend="transformers",
    )

    click.echo(f"   Model: {config.model_name}")
    click.echo(f"   Server: http://localhost:8000")
    click.echo("")

    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        click.echo("\n🛑 Server stopped")


if __name__ == "__main__":
    main()
