"""vLLM logo printing utilities."""

from __future__ import annotations

# ANSI color codes
COLOR_RESET = "\033[0m"
COLOR_WHITE = "\033[97m"
COLOR_MUTED = "\033[38;2;145;158;171m"


def build_vllm_logo_lines() -> list[str]:
    """Build the serve banner lines using the installer wordmark style."""

    return [
        "",
        f"{COLOR_WHITE}       █     █     █▄   ▄█{COLOR_RESET}",
        f"{COLOR_WHITE} ▄▄ ▄█ █     █     █ ▀▄▀ █{COLOR_RESET}",
        f"{COLOR_WHITE}  █▄█▀ █     █     █     █{COLOR_RESET}",
        f"{COLOR_WHITE}   ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀{COLOR_RESET}",
        f"{COLOR_WHITE}  Semantic Router{COLOR_RESET}",
        f"{COLOR_MUTED}  local runtime{COLOR_RESET}",
        "",
    ]


def print_vllm_logo() -> None:
    """Print the vLLM Semantic Router serve banner."""

    for line in build_vllm_logo_lines():
        print(line)
