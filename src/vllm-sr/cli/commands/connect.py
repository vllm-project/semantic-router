"""Connect Claude Code or Codex CLI to a Router gateway."""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
from pathlib import Path

import click
import yaml

from cli import harness_connect as hc
from cli import harness_router_check as rc
from cli.terminal import echo, heading, success, warning

GATEWAY = "http://localhost:8899"


def _options(command):
    for option in reversed(
        [
            click.option(
                "--gateway",
                default=GATEWAY,
                show_default=True,
                help="The Router listener the harness calls.",
            ),
            click.option(
                "--model",
                default="vllm-sr/auto",
                show_default=True,
                help="The entrypoint the harness asks for.",
            ),
            click.option(
                "--api-key-env",
                default="VLLM_SR_API_KEY",
                show_default=True,
                help="The variable that holds the listener's API key.",
            ),
            click.option(
                "--config",
                "router_config",
                type=click.Path(dir_okay=False, path_type=Path),
                help="The Router configuration to check (default: ./config.yaml when present). Never edited.",
            ),
            click.option(
                "--dry-run",
                is_flag=True,
                help="Print the plan and the changes without writing anything.",
            ),
            click.option(
                "--undo",
                is_flag=True,
                help="Take back what earlier runs set; keys changed since are left as they are.",
            ),
            click.option("--yes", is_flag=True, help="Write without asking."),
        ]
    ):
        command = option(command)
    return command


@click.group()
def connect() -> None:
    """Point an agent harness at a Router gateway (System Two)."""


@connect.command("claude-code")
@_options
@click.option(
    "--scope",
    type=click.Choice(["user", "project"]),
    default="user",
    show_default=True,
    help="user: ~/.claude/settings.json ($CLAUDE_CONFIG_DIR). project: .claude/settings.local.json here.",
)
def claude_code(gateway, model, api_key_env, router_config, dry_run, undo, yes, scope):
    """Route Claude Code's model calls through the gateway."""
    env = hc.environ()
    path = hc.claude_settings_path(scope, Path.cwd(), env)
    if undo:
        _undo(path.parent)
        return
    token, from_env = hc.token_from(env, api_key_env)
    changes = [hc.Change(path, "json-env", hc.claude_env(gateway, model, token))]
    _run(
        "claude-code",
        changes,
        path.parent,
        token,
        from_env,
        api_key_env,
        router_config,
        dry_run,
        yes,
    )
    if dry_run:
        return
    if from_env:
        warning(f"{path} now holds the API key from ${api_key_env}.")
    _check_vsr(env)
    _probe(gateway, "claude-code", model, token)
    echo("Start a new Claude Code session to use it.")


@connect.command("codex")
@_options
@click.option(
    "--profile",
    default="vllm-sr",
    show_default=True,
    help="The Codex profile to write.",
)
def codex(gateway, model, api_key_env, router_config, dry_run, undo, yes, profile):
    """Route Codex CLI through the gateway with a profile of its own."""
    env = hc.environ()
    home = hc.codex_home(env)
    if undo:
        _undo(home)
        return
    token, from_env = hc.token_from(env, api_key_env)
    changes = hc.codex_changes(home, profile, gateway, model, api_key_env)
    _run(
        "codex",
        changes,
        home,
        token,
        from_env,
        api_key_env,
        router_config,
        dry_run,
        yes,
    )
    if dry_run:
        return
    if not from_env:
        warning(
            f"Set ${api_key_env} before starting Codex: it reads the key from there."
        )
    echo(
        "Codex gets the gateway's routing and plugins (System Two); vSR's checks are"
        " not available for Codex yet."
    )
    _probe(gateway, "codex", model, token)
    echo(f"Start Codex with: codex --profile {profile}")


def _run(
    harness,
    changes,
    manifest_dir,
    token,
    from_env,
    key_env,
    router_config,
    dry_run,
    yes,
):
    heading(f"vllm-sr connect {harness}")
    try:
        texts = [(change.old_text(), change.new_text()) for change in changes]
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    for change, (old, new) in zip(changes, texts, strict=True):
        _show_diff(
            change, old, new, hc.secrets_in(change, token if from_env else "", key_env)
        )
    _check_router(harness, router_config)
    if dry_run:
        echo("Dry run: nothing was written.")
        return
    if not yes and not click.confirm("Write these changes?", default=True):
        raise click.Abort()
    manifest = hc.apply_changes(changes, manifest_dir)
    success(f"Written; undo with --undo (record: {manifest}).")


def _show_diff(change: hc.Change, old: str, new: str, secrets: dict[str, str]) -> None:
    label = str(change.path)
    diff = difflib.unified_diff(
        hc.masked(old, secrets).splitlines(keepends=True),
        hc.masked(new, secrets).splitlines(keepends=True),
        fromfile=label if change.path.exists() else "/dev/null",
        tofile=label,
    )
    echo("".join(diff) or f"{label}: no change")


def _check_router(harness: str, router_config: Path | None) -> None:
    path = router_config or (
        Path("config.yaml") if Path("config.yaml").exists() else None
    )
    if path is None:
        echo("Router configuration: not checked (pass --config to check it).")
        return
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    findings = rc.check_router_config(doc, harness)
    if not findings:
        success(f"Router configuration {path}: has what {harness} needs.")
        return
    warning(
        f"Router configuration {path} lacks what {harness} needs; add (not changed here):"
    )
    for finding in findings:
        echo(f"- {finding.need}")
        echo("  " + finding.snippet.replace("\n", "\n  "))


def _check_vsr(env: dict[str, str]) -> None:
    """With the vSR plugin installed, its router app should leave model choice to the gateway."""
    plugin = _vsr_plugin(env)
    if plugin is None:
        return
    inputs = plugin.get("inputs") or {}
    if str(inputs.get("router", "true")) != "false":
        warning(
            "The vSR plugin's router app is on. Turn it off so the plugin and the gateway"
            " do not both choose a subagent's model: run /vsr router off in Claude Code."
            " This also stops its escalate app from moving the next subagent up a model."
        )
    runtime_url = str(inputs.get("runtimeUrl") or "")
    if not runtime_url:
        return
    health = rc.probe_runtime_health(runtime_url, str(inputs.get("runtimeToken") or ""))
    if health.ok:
        success(f"System One: ok ({health.detail})")
    else:
        warning(
            f"System One: offline ({health.detail}). Until the runtime answers, vSR's checks"
            " step aside (guard's fixed rules still hold), and an app set to wait until done"
            " withholds the output or holds the command instead."
        )


def _vsr_plugin(env: dict[str, str]) -> dict | None:
    claude = shutil.which("claude")
    if claude is None:
        return None
    listed = _json_of([claude, "plugin", "list", "--json"], env)
    plugin_id = next(
        (
            p.get("id")
            for p in listed or []
            if str(p.get("id", "")).startswith("vsr@") and p.get("enabled")
        ),
        None,
    )
    if plugin_id is None:
        return None
    return _json_of([claude, "plugin", "configure", plugin_id, "--json"], env) or {}


def _json_of(argv: list[str], env: dict[str, str]):
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, env=env, check=False
        )
        return json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def _probe(gateway: str, harness: str, model: str, token: str) -> None:
    result = rc.probe_gateway(gateway, harness, model, token)
    if not result.ok:
        raise click.ClickException(f"System Two: failed. {result.detail}")
    success(
        f"System Two: ok (selected model {result.model or 'not reported'}; {result.detail})"
    )
    for item in result.warnings:
        echo(f"  x-vsr-warnings: {item}")


def _undo(manifest_dir: Path) -> None:
    try:
        for line in hc.undo_changes(manifest_dir):
            echo(line)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    success("Undone.")
