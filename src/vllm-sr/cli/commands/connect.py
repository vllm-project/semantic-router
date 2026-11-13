"""Connect Claude Code or Codex CLI to a Router gateway."""

from __future__ import annotations

import difflib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import textwrap
from pathlib import Path

import click
import yaml

from cli import harness_connect as hc
from cli import harness_router_check as rc
from cli.chat_client import resolve_chat_base_url
from cli.terminal import echo, heading, output_width, success, warning


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
            click.option(
                "--context-window",
                type=click.IntRange(min=1),
                default=None,
                help="The smallest context window, in tokens, of the models the gateway"
                " can pick; by default read from the configuration's model cards. Claude"
                " Code then neither assumes its own window nor warns that it does not know"
                " the entrypoint, and Codex budgets with it.",
            ),
            click.option("--yes", is_flag=True, help="Write without asking."),
            click.option(
                "--launch",
                is_flag=True,
                help="Start the harness through the gateway for this session only, writing"
                " no file; arguments after -- go to the harness.",
            ),
            click.argument("harness_args", nargs=-1, type=click.UNPROCESSED),
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
@click.option(
    "--status-line",
    is_flag=True,
    help="Also show in Claude Code's status line which models answered, the last first.",
)
def claude_code(
    router_config,
    base_url,
    model,
    api_key_env,
    dry_run,
    undo,
    context_window,
    yes,
    launch,
    harness_args,
    scope,
    status_line,
):
    """Route Claude Code's model calls through the gateway."""
    env = dict(os.environ)
    path = hc.claude_settings_path(scope, Path.cwd(), env)
    _check_launch(launch, harness_args, undo, status_line)
    if undo:
        _undo(path.parent)
        return
    gateway, doc = _gateway(router_config, base_url)
    token, from_env = hc.token_from(env, api_key_env)
    context_window, notes = _context_window(doc, model, context_window)
    values = hc.claude_env(gateway, model, token)
    if context_window:
        values["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context_window)
    if launch:
        # the key goes in a private file, not on the command line
        _launch(
            "claude-code",
            ["claude", *harness_args],
            doc,
            gateway,
            model,
            token,
            notes,
            dry_run,
            {"env": values},
        )
    # the undo manifest beside the settings records the key too
    keyed = (path, path.parent / hc.MANIFEST) if scope == "project" and from_env else ()
    tracked = [p for p in keyed if hc.tracked_by_git(p, Path.cwd())]
    if tracked:
        raise click.ClickException(
            f"git tracks {tracked[0]}, so the API key would reach its history. Untrack"
            f" it (git rm --cached {tracked[0]}) or connect with --scope user."
        )
    changes = []
    if status_line:
        script, values["statusLine"] = hc.status_line(path)
        changes.append(script)
    changes.append(hc.Change(path, "json-env", values))
    label = f"<${api_key_env}>" if from_env else token
    if not _run("claude-code", changes, path.parent, label, doc, dry_run, yes, notes):
        return
    if scope == "project":
        for kept in hc.keep_out_of_git([path, path.parent / hc.MANIFEST], Path.cwd()):
            echo(f"Added {kept} to this repository's .git/info/exclude.")
    if from_env:
        warning(f"{path} now holds the API key from ${api_key_env}.")
    if env.get(hc.API_KEY):
        echo(
            "The ANTHROPIC_API_KEY your shell exports no longer goes to the gateway:"
            " the settings set it empty, and Claude Code takes the settings."
        )
    if scope == "user":
        for found, keys in hc.claude_overrides(Path.cwd(), values):
            warning(
                f"{found} sets {', '.join(keys)}: Claude Code started in this directory"
                f" takes those over {path}. Remove them there, connect here with --scope"
                " project, or start with --launch."
            )
    _check_vsr(env, gateway, doc, api_key_env if from_env else None)
    if not context_window:
        echo(
            f"Claude Code does not know {model}, so it assumes a window of its own"
            " (1,000,000 tokens in 2.1.323) and says so at start; --context-window sets"
            " the real window."
        )
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
def codex(
    router_config,
    base_url,
    model,
    api_key_env,
    dry_run,
    undo,
    context_window,
    yes,
    launch,
    harness_args,
    profile,
):
    """Route Codex CLI through the gateway with a profile of its own."""
    env = dict(os.environ)
    home = hc.codex_home(env)
    _check_launch(launch, harness_args, undo)
    if undo:
        _undo(home)
        return
    gateway, doc = _gateway(router_config, base_url)
    token, from_env = hc.token_from(env, api_key_env)
    key_env = api_key_env if from_env else None
    context_window, notes = _context_window(doc, model, context_window)
    if launch:
        # Codex reads the key from key_env, so it stays off the command line
        overrides = hc.codex_launch_args(gateway, model, key_env, context_window)
        _launch(
            "codex",
            ["codex", *overrides, *harness_args],
            doc,
            gateway,
            model,
            token,
            notes,
            dry_run,
        )
    change = hc.codex_change(home, profile, gateway, model, key_env, context_window)
    if not _run("codex", [change], home, "", doc, dry_run, yes, notes):
        return
    if from_env:
        echo(f"Codex reads the key from ${api_key_env}; keep it set when you start it.")
    echo(
        "Codex gets the gateway's routing and plugins (System Two); vSR's checks are"
        " not available for Codex yet."
    )
    for found in hc.codex_model_overrides(Path.cwd()):
        warning(
            f"{found} sets model, which Codex asks for instead of {model} once it trusts"
            " this project. Remove it there, or start with --launch."
        )
    if not context_window:
        echo(
            f"Codex does not know {model}, so it warns that it falls back to default"
            " model metadata; --context-window gives it the window to budget with."
        )
    _probe(gateway, "codex", model, token)
    echo(f"Start Codex with: codex --profile {profile}")


def _run(harness, changes, manifest_dir, key_label, doc, dry_run, yes, notes=()):
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
    for note in notes:
        echo(note)
    if dry_run:
        echo("Dry run: nothing was written.")
        return False
    if not yes and not click.confirm("Write these changes?", default=True):
        raise click.Abort()
    manifest = hc.apply_changes(changes, manifest_dir)
    success(f"Written; undo with --undo (record: {manifest}).")
    return True


def _check_launch(launch, harness_args, undo, status_line=False) -> None:
    if harness_args and not launch:
        raise click.UsageError(
            "arguments after -- go to the harness only with --launch"
        )
    if launch and (undo or status_line):
        flag = "--undo" if undo else "--status-line"
        raise click.UsageError(f"--launch writes nothing, so it does not take {flag}")


def _context_window(doc, model, given) -> tuple[int | None, list[str]]:
    """The flag, else the smallest window of the models the configuration lets
    model reach, with a line on where it came from or why there is none."""
    if given or doc is None:
        return given, []
    names = rc.entrypoint_models(doc, model)
    if not names:
        return None, [
            f"Context window: the configuration maps {model} to no recipe or model,"
            " so none was read from it."
        ]
    windows = rc.context_windows(doc, names)
    missing = [name for name, window in windows.items() if window is None]
    if missing:
        one = len(missing) == 1
        return None, [
            f"Context window: not set: {', '.join(missing)}, which {model} can reach,"
            f" {'states' if one else 'state'} no context_window_size. Set it"
            f" {'there' if one else 'on each'}, or pass --context-window."
        ]
    window = min(windows.values())
    return window, [
        f"Context window: {window:,} tokens, the smallest of the {len(windows)}"
        f" models {model} can reach in this configuration."
    ]


def _launch(harness, argv, doc, gateway, model, token, notes, dry_run, settings=None):
    """Probe the gateway, then run the harness through it and exit with its status."""
    heading(f"vllm-sr connect {harness} --launch")
    _check_router(harness, doc)
    for note in notes:
        echo(note)
    found = shutil.which(argv[0])
    if found is None:
        raise click.ClickException(f"{argv[0]} is not on PATH.")
    with tempfile.TemporaryDirectory(prefix="vllm-sr-launch-") as private:
        if settings is not None:
            # --settings wins over the user's and the project's settings (2.1.323)
            path = Path(private) / "settings.json"
            hc.write_private(path, json.dumps(settings, indent=2))
            argv = [argv[0], "--settings", str(path), *argv[1:]]
        echo(f"Starts: {shlex.join(argv)}")
        if dry_run:
            echo("Dry run: nothing was started.")
            raise click.exceptions.Exit(0)
        _probe(gateway, harness, model, token, written=False)
        # Ctrl-C belongs to the harness while it runs
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            code = subprocess.call([found, *argv[1:]])
        finally:
            signal.signal(signal.SIGINT, previous)
    raise click.exceptions.Exit(code)


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


def _check_vsr(
    env: dict[str, str], gateway: str, doc: dict | None, key_env: str | None = None
) -> None:
    """With the vSR plugin installed, its router app should leave model choice to the
    gateway, and it can ask the System One the same listener serves. key_env names
    the variable holding the listener's key, which System One checks too."""
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
        # the shell fills in the key, so it is never printed here
        feed = (
            f"printf '{values[:-1].replace('%', '%%')}, \"runtimeToken\": \"%s\"}}'"
            f' "${key_env}"'
            if key_env
            else f"echo '{values}'"
        )
        echo(
            f"This listener also serves System One ({', '.join(published)}). To let vSR"
            f" ask it instead of {inputs.get('runtimeUrl') or 'its own runtime'}, run:\n"
            f"  {feed} | claude plugin configure vsr@vllm-sr --values-stdin"
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


def _probe(
    gateway: str, harness: str, model: str, token: str, written: bool = True
) -> None:
    """A basic reply, then a streamed tool loop, each on its own line."""
    for result in rc.probe_gateway(gateway, harness, model, token):
        if not result.ok:
            raise click.ClickException(
                f"{result.name}: failed. {result.detail}\n"
                + (
                    "The changes above stay written: fix the gateway and run this"
                    " command again, or take them back with --undo."
                    if written
                    else f"{harness} was not started: fix the gateway and run this"
                    " command again."
                )
            )
        success(
            f"{result.name}: ok (model {result.model or 'not reported'}; {result.detail})"
        )
        for note in result.warnings:
            echo(
                textwrap.fill(
                    f"note: {note}",
                    output_width(),
                    initial_indent="  ",
                    subsequent_indent="    ",
                )
            )


def _undo(manifest_dir: Path) -> None:
    try:
        for line in hc.undo_changes(manifest_dir):
            echo(line)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    success("Undone.")
