"""vLLM logo printing utilities."""

from __future__ import annotations

from cli.terminal import brand


def build_vllm_logo_lines() -> list[str]:
    """Build the serve banner lines using the installer wordmark style."""

    return [
        "",
        "       █     █     █▄   ▄█",
        " ▄▄ ▄█ █     █     █ ▀▄▀ █",
        "  █▄█▀ █     █     █     █",
        "   ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀",
        "  Semantic Router",
        "  Intelligent Routing for Mixture-of-Models",
        "",
    ]


def print_vllm_logo() -> None:
    """Print the vLLM Semantic Router serve banner."""
    brand("\n".join(build_vllm_logo_lines()))
