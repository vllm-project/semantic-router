"""Shell completion script generation for vllm-sr CLI."""

from __future__ import annotations

import os

import click

from cli.commands.common import exit_with_logged_error
from cli.utils import get_logger

log = get_logger(__name__)

_SUPPORTED_SHELLS = ("bash", "zsh", "fish")

_SHELL_SETUP_HINTS: dict[str, str] = {
    "bash": (
        "# Add this to ~/.bashrc:\n"
        '#   eval "$(vllm-sr completion bash)"\n'
        "# Then reload your shell or run:\n"
        "#   source ~/.bashrc"
    ),
    "zsh": (
        "# Add this to ~/.zshrc:\n"
        '#   eval "$(vllm-sr completion zsh)"\n'
        "# Then reload your shell or run:\n"
        "#   source ~/.zshrc"
    ),
    "fish": (
        "# Save the output to the fish completions directory:\n"
        "#   vllm-sr completion fish > ~/.config/fish/completions/vllm-sr.fish\n"
        "# Then restart your fish shell."
    ),
}


def _detect_shell() -> str | None:
    """Attempt to detect the current shell from the SHELL environment variable."""
    shell_env = os.environ.get("SHELL", "")
    for shell in _SUPPORTED_SHELLS:
        if shell in shell_env:
            return shell
    return None


@click.command()
@click.argument(
    "shell",
    required=False,
    type=click.Choice(_SUPPORTED_SHELLS, case_sensitive=False),
    default=None,
)
@exit_with_logged_error(log)
def completion(shell: str | None) -> None:
    """Generate shell completion script.

    Outputs a completion script for the specified shell. If SHELL is omitted,
    the command attempts to detect the current shell automatically.

    Supported shells: bash, zsh, fish.

    \b
    Examples:
        vllm-sr completion bash              # Print bash completion script
        vllm-sr completion zsh               # Print zsh completion script
        vllm-sr completion fish              # Print fish completion script
        eval "$(vllm-sr completion bash)"    # Activate bash completions
        eval "$(vllm-sr completion zsh)"     # Activate zsh completions
    """
    if shell is None:
        shell = _detect_shell()
        if shell is None:
            raise click.UsageError(
                "Could not detect shell. Please specify one of: "
                + ", ".join(_SUPPORTED_SHELLS)
            )

    # Click's shell completion machinery uses an env var convention:
    #   _<PROG_NAME>_COMPLETE=<shell>_source <prog_name>
    # We replicate that by importing the completion class and generating
    # the script source directly.
    from click.shell_completion import get_completion_class  # noqa: PLC0415

    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.UsageError(f"Unsupported shell: {shell}")

    # We need the root CLI group to generate the completion script.
    from cli.main import main  # noqa: PLC0415

    comp = comp_cls(
        cli=main,
        ctx_args={},
        prog_name="vllm-sr",
        complete_var="_VLLM_SR_COMPLETE",
    )

    click.echo(f"# vllm-sr shell completion for {shell}")
    click.echo(f"# {'-' * 40}")
    click.echo(f"# {_SHELL_SETUP_HINTS[shell].splitlines()[0].lstrip('# ')}")
    click.echo(comp.source())
