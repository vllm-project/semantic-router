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
            click.option(
                "--vsr",
                is_flag=True,
                help="Also set up vSR (System One) beside the agent: its plugin,"
                " asking this listener's System One model.",
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
    vsr,
    launch,
    harness_args,
    scope,
    status_line,
):
    """Route Claude Code's model calls through the gateway."""
    env = dict(os.environ)
    path = hc.claude_settings_path(scope, Path.cwd(), env)
    _check_launch(launch, harness_args, undo, status_line, vsr)
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
    options = _vsr_options(gateway, doc) if vsr else None
    if options and not dry_run and shutil.which("claude") is None:
        raise click.ClickException(
            "--vsr runs claude plugin, and claude is not on PATH."
        )
    plugin_id = _vsr_plugin_id(env) if options else None
    plan = _vsr_claude_plan(options, plugin_id, api_key_env if from_env else None)
    if not _run(
        "claude-code", changes, path.parent, label, doc, dry_run, yes, notes, plan
    ):
        return
    if options:
        _setup_vsr_claude(
            env, plugin_id, options, token if from_env else "", path.parent
        )
    if scope == "project":
        for kept in hc.keep_out_of_git([path, path.parent / hc.MANIFEST], Path.cwd()):
            echo(f"Added {kept} to this repository's .git/info/exclude.")
    if from_env:
        warning(f"{path} now holds the API key from ${api_key_env}.")
    overrides = list(hc.claude_overrides(Path.cwd(), values)) if scope == "user" else []
    # a project file that sets the key itself sends it again, and is warned about below
    if env.get(hc.API_KEY) and all(hc.API_KEY not in keys for _, keys in overrides):
        echo(
            "The ANTHROPIC_API_KEY your shell exports no longer goes to the gateway:"
            " the settings set it empty, and Claude Code takes the settings."
        )
    if overrides:
        for found, keys in overrides:
            warning(
                f"{found} sets {', '.join(keys)}: Claude Code started in this directory"
                f" takes those over {path}. Remove them there, connect here with --scope"
                " project, or start with --launch."
            )
    if not vsr:
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
    vsr,
    launch,
    harness_args,
    profile,
):
    """Route Codex CLI through the gateway with a profile of its own."""
    env = dict(os.environ)
    home = hc.codex_home(env)
    _check_launch(launch, harness_args, undo, vsr=vsr)
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
    changes = [change]
    options = _vsr_options(gateway, doc) if vsr else None
    plan = []
    if options:
        # the vSR plugin's own settings: what is there stays, the runtime is the listener's
        held = home / "vsr.json"
        kept = json.loads(held.read_text(encoding="utf-8")) if held.exists() else {}
        text = json.dumps({**kept, **options}, indent=2) + "\n"
        changes.append(hc.Change(held, "file", {"file": text}))
        plan = [shlex.join(argv) for argv in VSR_CODEX_INSTALL]
    if not _run("codex", changes, home, "", doc, dry_run, yes, notes, plan):
        return
    if from_env:
        echo(f"Codex reads the key from ${api_key_env}; keep it set when you start it.")

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


def _run(
    harness, changes, manifest_dir, key_label, doc, dry_run, yes, notes=(), plan=()
):
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
    if plan:
        echo("vSR (System One) beside it, then:")
        for line in plan:
            echo(f"  {line}")
    if dry_run:
        echo("Dry run: nothing was written.")
        return False
    if not yes and not click.confirm("Write these changes?", default=True):
        raise click.Abort()
    manifest = hc.apply_changes(changes, manifest_dir)
    success(f"Written; undo with --undo (record: {manifest}).")
    return True


def _check_launch(launch, harness_args, undo, status_line=False, vsr=False) -> None:
    if harness_args and not launch:
        raise click.UsageError(
            "arguments after -- go to the harness only with --launch"
        )
    if launch and (undo or status_line or vsr):
        flag = "--undo" if undo else "--status-line" if status_line else "--vsr"
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
        path = Path(private) / "settings.json"
        if settings is not None:
            # --settings wins over the user's and the project's settings (2.1.323)
            argv = [argv[0], "--settings", str(path), *argv[1:]]
        echo(f"Starts: {shlex.join(argv)}")
        if dry_run:
            echo("Dry run: nothing was started.")
            raise click.exceptions.Exit(0)
        if settings is not None:
            hc.write_private(path, json.dumps(settings, indent=2))
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


# vSR beside the agent (System One): the plugin, from this repository's marketplace.
VSR_PLUGIN = "vsr@vllm-sr"
VSR_CLAUDE_INSTALL = (
    ["claude", "plugin", "marketplace", "add", "vllm-project/semantic-router"],
    ["claude", "plugin", "install", VSR_PLUGIN],
)
# The Codex plugin installs from a checkout until it is on main (#6413).
VSR_CODEX_INSTALL = (
    ["codex", "plugin", "marketplace", "add", "<semantic-router checkout>"],
    ["codex", "plugin", "add", VSR_PLUGIN],
)


def _vsr_options(gateway: str, doc: dict | None) -> dict[str, str] | None:
    """The runtime the vSR plugin asks: this listener's System One, or None when it
    publishes no model."""
    models = rc.system_one_models(doc)
    if not models:
        echo(
            "System One: this listener publishes no model (listeners[].systemone), so"
            " vSR is not set up."
        )
        return None
    return {
        "runtimeUrl": f"{gateway.rstrip('/')}/v1/decisions",
        "runtimeModel": models[0],
    }


def _vsr_claude_plan(options, plugin_id, key_env) -> list[str]:
    """What --vsr runs for Claude Code, as shown before it runs (the key named, never
    shown)."""
    if not options:
        return []
    shown = {
        **_claude_vsr_values(options),
        **({"runtimeToken": f"<${key_env}>"} if key_env else {}),
    }
    install = [] if plugin_id else [shlex.join(argv) for argv in VSR_CLAUDE_INSTALL]
    configure = f"claude plugin configure {plugin_id or VSR_PLUGIN} --values-stdin"
    return [*install, f"{configure} <<< '{json.dumps(shown)}'"]


def _claude_vsr_values(options: dict[str, str]) -> dict[str, str]:
    # a remote runtime, and the router app off: the gateway picks each call's model
    return {"runtime": "remote", **options, "router": "false"}


def _setup_vsr_claude(env, plugin_id, options, token, manifest_dir) -> None:
    """Install the plugin when it is missing and set its runtime through Claude Code's
    own command, which keeps the token in its credential store. --undo puts back the
    options it changed; the token stays."""
    for argv in () if plugin_id else VSR_CLAUDE_INSTALL:
        _must(argv, env)
    plugin_id = plugin_id or VSR_PLUGIN
    configure = ["claude", "plugin", "configure", plugin_id]
    before = _plugin_inputs(env, plugin_id)
    values = _claude_vsr_values(options)
    secret = {"runtimeToken": token} if token else {}
    _must([*configure, "--values-stdin"], env, json.dumps({**values, **secret}))
    hc.record_plugin(manifest_dir, plugin_id, before, values)
    success(
        f"vSR: {plugin_id} asks {options['runtimeModel']} at {options['runtimeUrl']}."
    )


def _must(argv: list[str], env: dict[str, str], stdin: str | None = None) -> None:
    try:
        subprocess.run(argv, env=env, input=stdin, text=True, check=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise click.ClickException(f"{shlex.join(argv)} failed: {exc}") from exc


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


def _vsr_plugin_id(env: dict[str, str]) -> str | None:
    """The enabled vSR plugin's id (vsr@<marketplace>), or None."""
    claude = shutil.which("claude")
    listed = _json_of([claude, "plugin", "list", "--json"], env) if claude else None
    return next(
        (
            p.get("id")
            for p in listed or []
            if str(p.get("id", "")).startswith("vsr@") and p.get("enabled")
        ),
        None,
    )


def _vsr_plugin(env: dict[str, str]) -> dict | None:
    plugin_id = _vsr_plugin_id(env)
    if plugin_id is None:
        return None
    return _json_of(["claude", "plugin", "configure", plugin_id, "--json"], env) or {}


def _plugin_inputs(env: dict[str, str], plugin_id: str) -> dict[str, str]:
    """A plugin's option values as strings; an option never set reads as its
    default, since `inputs` gives an unset string option as empty."""
    found = _json_of(["claude", "plugin", "configure", plugin_id, "--json"], env) or {}
    inputs, configured = found.get("inputs") or {}, set(found.get("configured") or ())
    schema = found.get("schema") or {}
    values = {
        key: value if key in configured else schema.get(key, {}).get("default", value)
        for key, value in inputs.items()
    }
    return {
        k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in values.items()
    }


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
    env = dict(os.environ)
    for plugin_id, back in hc.plugin_undo(
        manifest_dir, lambda one: _plugin_inputs(env, one)
    ):
        if back:
            _must(
                ["claude", "plugin", "configure", plugin_id, "--values-stdin"],
                env,
                json.dumps(back),
            )
            echo(f"restored {plugin_id}: {', '.join(sorted(back))}")
    try:
        for line in hc.undo_changes(manifest_dir):
            echo(line)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    success("Undone.")
