"""Connect Claude Code or Codex CLI to a Router gateway."""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
from pathlib import Path

import click
import yaml

from cli import harness_connect as hc
from cli import harness_router_check as rc
from cli.chat_client import resolve_chat_base_url
from cli.terminal import echo, heading, success, warning


def _options(command):
    for option in reversed(
        [
            click.option(
                "--config",
                "router_config",
                default="config.yaml",
                show_default=True,
                type=click.Path(dir_okay=False, path_type=Path),
                help="The Router configuration: its first listener is the gateway, and it"
                " is checked for what the harness needs. Never edited.",
            ),
            click.option(
                "--base-url",
                default=None,
                help="The gateway's origin; otherwise derived from --config.",
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
def claude_code(router_config, base_url, model, api_key_env, dry_run, undo, yes, scope):
    """Route Claude Code's model calls through the gateway."""
    env = dict(os.environ)
    path = hc.claude_settings_path(scope, Path.cwd(), env)
    if undo:
        _undo(path.parent)
        return
    gateway, doc = _gateway(router_config, base_url)
    token, from_env = hc.token_from(env, api_key_env)
    change = hc.Change(path, "json-env", hc.claude_env(gateway, model, token))
    label = f"<${api_key_env}>" if from_env else token
    if not _run("claude-code", [change], path.parent, label, doc, dry_run, yes):
        return
    if scope == "project":
        for kept in hc.keep_out_of_git([path, path.parent / hc.MANIFEST], Path.cwd()):
            echo(f"Added {kept} to this repository's .git/info/exclude.")
    if from_env:
        warning(f"{path} now holds the API key from ${api_key_env}.")
    _check_vsr(env, gateway, doc)
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
def codex(router_config, base_url, model, api_key_env, dry_run, undo, yes, profile):
    """Route Codex CLI through the gateway with a profile of its own."""
    env = dict(os.environ)
    home = hc.codex_home(env)
    if undo:
        _undo(home)
        return
    gateway, doc = _gateway(router_config, base_url)
    token, from_env = hc.token_from(env, api_key_env)
    change = hc.codex_change(home, profile, gateway, model, api_key_env)
    if not _run("codex", [change], home, "", doc, dry_run, yes):
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


def _run(harness, changes, manifest_dir, key_label, doc, dry_run, yes):
    """Show the changes, check the Router configuration, and write; False on a dry run."""
    heading(f"vllm-sr connect {harness}")
    try:
        for change in changes:
            label = str(change.path)
            diff = difflib.unified_diff(
                change.shown(change.old_text(), key_label).splitlines(keepends=True),
                change.shown(change.new_text(), key_label).splitlines(keepends=True),
                fromfile=label if change.path.exists() else "/dev/null",
                tofile=label,
            )
            echo("".join(diff) or f"{label}: no change")
    except ValueError as exc:
        raise click.ClickException(f"{change.path}: {exc}") from exc
    _check_router(harness, doc)
    if dry_run:
        echo("Dry run: nothing was written.")
        return False
    if not yes and not click.confirm("Write these changes?", default=True):
        raise click.Abort()
    manifest = hc.apply_changes(changes, manifest_dir)
    success(f"Written; undo with --undo (record: {manifest}).")
    return True


def _gateway(router_config: Path, base_url: str | None) -> tuple[str, dict | None]:
    """The gateway's origin, from --base-url or the configuration's first listener,
    and the configuration when there is one to check."""
    doc = None
    if router_config.is_file():
        try:
            doc = yaml.safe_load(router_config.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise click.ClickException(f"cannot read {router_config}: {exc}") from exc
    try:
        gateway = resolve_chat_base_url(
            config_path=str(router_config), target=None, base_url=base_url
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    return gateway, doc


def _check_router(harness: str, doc: dict | None) -> None:
    if doc is None:
        echo("Router configuration: not checked (pass --config to check it).")
        return
    findings = rc.check_router_config(doc, harness)
    if not findings:
        success(f"Router configuration: has what {harness} needs.")
        return
    warning(
        f"The Router configuration lacks what {harness} needs; add (not changed here):"
    )
    for finding in findings:
        echo(f"- {finding.need}")
        echo("  " + finding.snippet.replace("\n", "\n  "))


def _check_vsr(env: dict[str, str], gateway: str, doc: dict | None) -> None:
    """With the vSR plugin installed, its router app should leave model choice to the
    gateway, and it can ask the System One the same listener serves."""
    plugin = _vsr_plugin(env)
    if plugin is None:
        return
    inputs = plugin.get("inputs") or {}
    # a boolean option may come back as false or "false"
    if str(inputs.get("router", True)).lower() != "false":
        warning(
            "The vSR plugin's router app is on. Turn it off so the plugin and the gateway"
            " do not both choose a subagent's model: run /vsr router off in Claude Code."
            " This also stops its escalate app from moving the next subagent up a model."
        )
    published = rc.system_one_models(doc)
    decisions = gateway + "/v1/decisions"
    if published and inputs.get("runtimeUrl") != decisions:
        values = json.dumps(
            {"runtime": "remote", "runtimeUrl": decisions, "runtimeModel": published[0]}
        )
        echo(
            f"This listener also serves System One ({', '.join(published)}). To let vSR"
            f" ask it instead of {inputs.get('runtimeUrl') or 'its own runtime'}, run:\n"
            f"  echo '{values}' | claude plugin configure vsr@vllm-sr --values-stdin"
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
        raise click.ClickException(
            f"System Two: failed. {result.detail}\n"
            "The changes above stay written; take them back with --undo."
        )
    success(
        f"System Two: ok (selected model {result.model or 'not reported'}; {result.detail})"
    )
    for item in result.warnings:
        echo(f"  x-vsr-warnings: {item}")


def _undo(manifest_dir: Path) -> None:
    try:
        for line in hc.undo_changes(manifest_dir):
            echo(line)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    success("Undone.")
