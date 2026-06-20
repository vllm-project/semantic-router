"""Shell completion script generation and installation for vllm-sr CLI."""

from __future__ import annotations

import os
from pathlib import Path

import click

from cli.commands.common import exit_with_logged_error
from cli.utils import get_logger

log = get_logger(__name__)

_SUPPORTED_SHELLS = ("bash", "zsh", "fish")

_COMPLETION_MARKER = "# vllm-sr shell completion"

_RC_FILES: dict[str, str] = {
    "bash": "~/.bashrc",
    "zsh": "~/.zshrc",
}

_EVAL_LINES: dict[str, str] = {
    "bash": 'eval "$(vllm-sr completion show bash)"',
    "zsh": 'eval "$(vllm-sr completion show zsh)"',
}


def _detect_shell() -> str | None:
    """Attempt to detect the current shell from the SHELL environment variable."""
    shell_env = os.environ.get("SHELL", "")
    for shell in _SUPPORTED_SHELLS:
        if shell in shell_env:
            return shell
    return None


def _resolve_shell(shell: str | None) -> str:
    """Resolve the shell argument, falling back to auto-detection."""
    if shell is not None:
        return shell
    detected = _detect_shell()
    if detected is None:
        raise click.UsageError(
            "Could not detect shell. Please specify one of: "
            + ", ".join(_SUPPORTED_SHELLS)
        )
    return detected


def _generate_completion_script(shell: str) -> str:
    """Generate the completion script source for the given shell."""
    from click.shell_completion import get_completion_class  # noqa: PLC0415

    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.UsageError(f"Unsupported shell: {shell}")

    from cli.main import main  # noqa: PLC0415

    comp = comp_cls(
        cli=main,
        ctx_args={},
        prog_name="vllm-sr",
        complete_var="_VLLM_SR_COMPLETE",
    )
    return comp.source()


def _rc_already_configured(rc_path: Path) -> bool:
    """Check whether the rc file already contains the completion marker."""
    if not rc_path.exists():
        return False
    return _COMPLETION_MARKER in rc_path.read_text(encoding="utf-8")


def _install_for_bash_or_zsh(shell: str) -> str:
    """Append the eval line to ~/.bashrc or ~/.zshrc. Returns the rc path."""
    rc_path = Path(_RC_FILES[shell]).expanduser()
    if _rc_already_configured(rc_path):
        return str(rc_path)

    eval_line = _EVAL_LINES[shell]
    snippet = f"\n{_COMPLETION_MARKER}\n{eval_line}\n"

    with rc_path.open("a", encoding="utf-8") as f:
        f.write(snippet)
    return str(rc_path)


def _install_for_fish(script_source: str) -> str:
    """Write the completion script to the fish completions directory."""
    fish_dir = Path("~/.config/fish/completions").expanduser()
    fish_dir.mkdir(parents=True, exist_ok=True)
    fish_file = fish_dir / "vllm-sr.fish"
    fish_file.write_text(f"{_COMPLETION_MARKER}\n{script_source}\n", encoding="utf-8")
    return str(fish_file)


@click.group(invoke_without_command=True)
@click.pass_context
@exit_with_logged_error(log)
def completion(ctx: click.Context) -> None:
    """Generate or install shell completion for vllm-sr.

    \b
    Examples:
        vllm-sr completion show bash         # Print bash completion script
        vllm-sr completion show zsh          # Print zsh completion script
        vllm-sr completion install           # Auto-install for current shell
        vllm-sr completion install bash      # Install bash completions
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@completion.command("show")
@click.argument(
    "shell",
    required=False,
    type=click.Choice(_SUPPORTED_SHELLS, case_sensitive=False),
    default=None,
)
@exit_with_logged_error(log)
def completion_show(shell: str | None) -> None:
    """Print the completion script for a shell.

    Outputs the completion script to stdout. If SHELL is omitted, the
    command attempts to detect the current shell automatically.

    \b
    Examples:
        vllm-sr completion show bash
        vllm-sr completion show zsh
        vllm-sr completion show fish
        eval "$(vllm-sr completion show zsh)"
    """
    resolved = _resolve_shell(shell)
    script = _generate_completion_script(resolved)
    click.echo(f"# vllm-sr shell completion for {resolved}")
    click.echo(f"# {'-' * 40}")
    click.echo(script)


@completion.command("install")
@click.argument(
    "shell",
    required=False,
    type=click.Choice(_SUPPORTED_SHELLS, case_sensitive=False),
    default=None,
)
@exit_with_logged_error(log)
def completion_install(shell: str | None) -> None:
    """Install shell completions into your shell configuration.

    Automatically appends the completion setup to your shell's rc file
    (~/.bashrc, ~/.zshrc) or writes to the fish completions directory.
    Safe to run multiple times — skips if already configured.

    \b
    Examples:
        vllm-sr completion install           # Auto-detect shell
        vllm-sr completion install bash      # Install for bash
        vllm-sr completion install zsh       # Install for zsh
        vllm-sr completion install fish      # Install for fish
    """
    resolved = _resolve_shell(shell)
    script = _generate_completion_script(resolved)

    if resolved == "fish":
        target = _install_for_fish(script)
    else:
        target = _install_for_bash_or_zsh(resolved)

    rc_path = (
        Path(_RC_FILES.get(resolved, "")).expanduser() if resolved != "fish" else None
    )
    already = rc_path is not None and _rc_already_configured(rc_path)

    if already:
        click.echo(
            f"✓ Shell completions for {resolved} are already configured " f"in {target}"
        )
    else:
        click.echo(f"✓ Shell completions for {resolved} installed to {target}")

    if resolved == "fish":
        click.echo("  Restart your fish shell to activate completions.")
    else:
        click.echo(f"  Restart your shell or run: source {target}")
