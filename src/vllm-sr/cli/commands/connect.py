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
import urllib.parse
from pathlib import Path

import click
import yaml

from cli import harness_check as hk
from cli import harness_connect as hc
from cli import harness_router_check as rc
from cli.chat_client import resolve_chat_base_url
from cli.runtime_management_config import _configured_management_port
from cli.runtime_stack import resolve_runtime_stack
from cli.terminal import echo, fields, heading, output_width, success, warning

# The vSR plugin's marketplace: this repository on GitHub.
VSR_SOURCE = "vllm-project/semantic-router"

# Each harness's name, and the provider whose login it sends a gateway.
HARNESS_LOGINS = {
    "claude-code": ("Claude Code", "anthropic"),
    "codex": ("Codex", "openai"),
}


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
                " Code and Codex then budget with it instead of a window of their own.",
            ),
            click.option(
                "--vsr/--no-vsr",
                default=True,
                help="Also set up vSR (System One) beside the agent when the listener"
                " publishes a System One model: install its plugin and point it there.",
            ),
            click.option(
                "--vsr-source",
                default=VSR_SOURCE,
                show_default=True,
                help="Where the vSR plugin installs from: a marketplace's owner/repo,"
                " Git URL, or a local checkout.",
            ),
            click.option("--yes", is_flag=True, help="Write without asking."),
            click.option(
                "--check",
                is_flag=True,
                help="Then run the harness itself once on a tiny task through the"
                " gateway, with what was written (or, with --dry-run, would be), and"
                " report each step. Spends a few short model calls on your login or"
                " key.",
            ),
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
    vsr_source,
    check,
    launch,
    harness_args,
    scope,
    status_line,
):
    """Route Claude Code's model calls through the gateway."""
    env = dict(os.environ)
    path = hc.claude_settings_path(scope, Path.cwd(), env)
    _check_launch(launch, harness_args, undo, status_line, check)
    if undo:
        _undo(path.parent)
        return
    gateway, doc = _gateway(router_config, base_url)
    token, from_env = hc.token_from(env, api_key_env)
    _require_key(
        doc, base_url, from_env, api_key_env, _caller_login("claude-code", doc)
    )
    if _caller_login("claude-code", doc):
        # the Router presents Claude Code's own login to Anthropic, and a gateway
        # token would go in its place
        token, from_env = "", False
    context_window, notes = _context_window(doc, model, context_window)
    values = hc.claude_env(gateway, model, token)
    if context_window:
        values["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context_window)
    if launch:
        # the key stays off the command line and out of the settings file: Claude
        # Code reads it from the variable its own session inherits
        settings = {"env": values}
        if from_env:
            settings = {
                "env": {**values, hc.TOKEN_KEY: ""},
                "apiKeyHelper": f"printenv {shlex.quote(api_key_env)}",
            }
        _launch(
            "claude-code",
            ["claude", *harness_args],
            doc,
            gateway,
            model,
            token,
            notes,
            dry_run,
            settings,
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
    options, plugin_id, state = _vsr_claude_start(
        env, gateway, doc, vsr, from_env, api_key_env
    )
    install = _installs("claude", vsr_source) if options and not plugin_id else []
    plan = _vsr_claude_plan(
        options, plugin_id, install, api_key_env if from_env else None
    )
    secrets = [token] if from_env else []
    if not _run(
        "claude-code",
        changes,
        path.parent,
        label,
        doc,
        dry_run,
        yes,
        notes,
        plan,
        model,
    ):
        if check:
            text = hc.set_json_env("", values)
            _check(
                "claude-code",
                lambda private, _: hk.claude_argv(
                    _private(private, "settings.json", text)
                ),
                env,
                secrets,
                written=False,
            )
        return
    if options:
        before = state[0] if state else None
        secret = token if from_env else ""
        _vsr_step(
            lambda: _setup_vsr_claude(
                env, plugin_id, install, options, secret, path.parent, before
            ),
            plan,
        )
    if scope == "project":
        for kept in hc.keep_out_of_git([path, path.parent / hc.MANIFEST], Path.cwd()):
            echo(f"Added {kept} to this repository's .git/info/exclude.")
    if from_env:
        warning(f"{path} now holds the API key from ${api_key_env}.")
    overrides = (
        list(hc.claude_overrides(Path.cwd(), values, path)) if scope == "user" else []
    )
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
    _probe_or_say_why(gateway, "claude-code", model, token, doc)
    if check:
        settings = path.absolute()  # the harness runs in a directory of its own
        _check("claude-code", lambda *_: hk.claude_argv(settings), env, secrets)
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
    vsr_source,
    check,
    launch,
    harness_args,
    profile,
):
    """Route Codex CLI through the gateway with a profile of its own."""
    env = dict(os.environ)
    home = hc.codex_home(env)
    _check_launch(launch, harness_args, undo, check=check)
    if undo:
        _undo(home)
        return
    gateway, doc = _gateway(router_config, base_url)
    token, from_env = hc.token_from(env, api_key_env)
    login = _caller_login("codex", doc)
    _require_key(doc, base_url, from_env, api_key_env, login)
    if login:
        # the Router presents Codex's own ChatGPT login or API key to OpenAI, and a
        # gateway key would go in its place
        token, from_env = "", False
    key_env = api_key_env if from_env else None
    context_window, notes = _context_window(doc, model, context_window)
    review_model = next(
        (
            entry.get("review_model")
            for entry in (doc or {}).get("entrypoints") or []
            if isinstance(entry, dict)
            and entry.get("api") in (None, "", "chat")
            and model in (entry.get("model_names") or [])
        ),
        None,
    )
    review_model = review_model.strip() if review_model else None
    review_window, review_notes = (
        _context_window(doc, review_model, None) if review_model else (None, [])
    )
    notes += review_notes
    notes += [
        f"Approve for me reviews go to {review_model or model} through the gateway."
        + ("" if review_model else " (session model)")
    ]
    if not review_model and _chatgpt_caller_backend(doc):
        notes.append(_codex_review_hint(model))
    catalog = hc.codex_catalog(
        (
            _json_of(["codex", "debug", "models", "--bundled"], env)
            if shutil.which("codex")
            else None
        ),
        model,
        context_window,
        review_model,
        review_window,
    )
    if launch:
        # Codex reads the key from key_env, so it stays off the command line
        overrides = hc.codex_launch_args(gateway, model, key_env, context_window, login)
        _launch(
            "codex",
            ["codex", *overrides, *harness_args],
            doc,
            gateway,
            model,
            token,
            notes,
            dry_run,
            catalog or None,
        )
    catalog_path = home / f"{profile}.models.json" if catalog else None
    change = hc.codex_change(
        home, profile, gateway, model, key_env, context_window, login, catalog_path
    )
    changes = [change]
    if catalog_path:
        text = json.dumps(catalog, indent=2) + "\n"
        changes.append(hc.Change(catalog_path, "file", {"file": text}))
    options = _vsr_options(gateway, doc) if vsr else None
    if options and shutil.which("codex") is None:
        warning("vSR is not set up: codex is not on PATH.")
        options = None
    install = (
        []
        if not options or _vsr_codex_installed(env)
        else _installs("codex", vsr_source)
    )
    plan = [shlex.join(argv) for argv in install]
    if options:
        # the listener checks its key on System One too: the plugin reads it from the
        # variable named here, so the key itself stays out of vsr.json; without a key
        # none is named, so a variable an earlier listener's key sits in is not sent.
        # The turn line reads each turn's records from the management API.
        options = {
            **options,
            "runtimeTokenEnv": key_env,
            "managementUrl": _management_url(gateway, doc),
        }
    if options:
        # the vSR plugin's own settings: connect owns the runtime keys, the rest stays
        changes.append(hc.Change(home / "vsr.json", "json", options))
    secrets = [token] if from_env else []
    # only the catalog sends reviews to the review entrypoint
    reviewer = review_model if catalog else None
    if not _run("codex", changes, home, "", doc, dry_run, yes, notes, plan, model):
        if check:
            overrides = hc.codex_launch_args(
                gateway, model, key_env, context_window, login
            )

            def unwritten(private: Path, work: Path) -> list[str]:
                config = list(overrides)
                if catalog:
                    text = json.dumps(catalog, indent=2)
                    found = _private(private, "models.json", text)
                    config += ["-c", f"model_catalog_json={json.dumps(str(found))}"]
                return hk.codex_argv(config, work, bool(reviewer))

            _check("codex", unwritten, env, secrets, False, reviewer, gateway)
        return
    if options and _vsr_step(lambda: [_must(argv, env) for argv in install], plan):
        success(
            f"vSR: the Codex plugin asks {options['runtimeModel']} at {options['runtimeUrl']}."
        )
    if from_env:
        reader = "Codex and its vSR plugin read" if options else "Codex reads"
        echo(f"{reader} the key from ${api_key_env}; keep it set when you start it.")

    for found in hc.codex_model_overrides(Path.cwd()):
        warning(
            f"{found} sets model, which Codex asks for instead of {model} once it trusts"
            " this project. Remove it there, or start with --launch."
        )
    if catalog is None:
        echo(
            f"Codex does not know {model}, so it warns that it falls back to default"
            f" model metadata: this Codex bundles no {hc.CODEX_CATALOG_BASE} entry to"
            " describe it with."
        )
    if not context_window:
        echo(
            f"Codex budgets {model} with a window of its own; --context-window gives"
            " it the gateway's."
        )
    _probe_or_say_why(gateway, "codex", model, token, doc)
    if check:
        _check(
            "codex",
            lambda _, work: hk.codex_argv(["--profile", profile], work, bool(reviewer)),
            env,
            secrets,
            True,
            reviewer,
            gateway,
        )
    echo(f"Start Codex with: codex --profile {profile}")


# Seconds the status waits for the gateway's basic reply.
STATUS_PROBE_TIMEOUT = 5


@connect.command("status")
@click.option("--json", "as_json", is_flag=True, help="Print the status as JSON.")
def status(as_json):
    """Show what connect manages for each harness now, and whether it still holds.

    Read from the undo records beside Claude Code's settings (user and project)
    and in the Codex home; the gateway gets one short request per connection.
    """
    env = dict(os.environ)
    cwd = Path.cwd()
    places = {
        "claude-code": [
            hc.claude_settings_path(scope, cwd, env).parent
            for scope in ("user", "project")
        ],
        "codex": [hc.codex_home(env)],
    }
    found = []
    for harness, listed in places.items():
        folders = list(dict.fromkeys(folder.resolve() for folder in listed))
        entries = [
            {
                "harness": harness,
                "connected": True,
                "record": str(folder / hc.MANIFEST),
                **entry,
            }
            for folder in folders
            for entry in _connections(folder)
            if entry["harness"] == harness
        ]
        for entry in entries:
            entry["probe"] = _status_probe(entry, env)
        found += entries or [
            {
                "harness": harness,
                "connected": False,
                "looked_in": [str(folder) for folder in folders],
            }
        ]
    if as_json:
        echo(json.dumps({"connections": found}, indent=2))
        return
    for entry in found:
        _show_status(entry)


def _connections(folder: Path) -> list[dict]:
    try:
        return hc.connections(folder)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(
            f"cannot read {folder / hc.MANIFEST}: {exc!r}"
        ) from exc


def _status_probe(entry: dict, env: dict[str, str]) -> dict[str, str]:
    """The gateway's basic reply now, sent with the key the harness would send,
    while the harness still sends to the gateway and model connect wrote: a key
    the settings now hold for another gateway never goes to the one recorded."""
    harness, gateway, model = entry["harness"], entry["gateway"], entry["model"]
    if not gateway or not model:
        return {"state": "not probed", "detail": "the record names no gateway or model"}
    if entry["login"] == "own login":
        name = HARNESS_LOGINS[harness][0]
        return {
            "state": "not probed",
            "detail": f"the gateway answers only on the login {name} itself sends",
        }
    # the profile is the one file of a Codex connection that says where it goes
    if harness == "codex" and any(
        item["state"] != "as written"
        for item in entry["files"]
        if item["path"].endswith(".config.toml")
    ):
        return {
            "state": "not probed",
            "detail": "the profile no longer holds what connect wrote, so no key is"
            f" sent to {gateway}",
        }
    key_env = entry.get("key_env")
    if key_env and not env.get(key_env):
        return {
            "state": "not probed",
            "detail": f"${key_env} is not set, and Codex sends the key from it",
        }
    token = env.get(key_env) if key_env else hc.NO_KEY
    if harness == "claude-code":
        # where to, for which model and with which key, from one read of the settings
        recorded = {"ANTHROPIC_BASE_URL": gateway, "ANTHROPIC_MODEL": model}
        keys = dict.fromkeys((*recorded, hc.TOKEN_KEY), "")
        settings = hc.Change(Path(entry["settings"]), "json-env", keys)
        try:
            now = settings.raw(settings.old_text())
        except ValueError:
            now = {}
        if moved := [key for key, value in recorded.items() if now.get(key) != value]:
            return {
                "state": "not probed",
                "detail": f"the settings no longer hold the {' and '.join(moved)}"
                f" connect wrote, so their key is not sent to {gateway}",
            }
        token = now[hc.TOKEN_KEY] or hc.NO_KEY
    result = rc.basic_reply(gateway, harness, model, token, STATUS_PROBE_TIMEOUT)
    if not result.ok:
        return {"state": "failed", "detail": hk.shown(result.detail, [token])}
    return {"state": "ok", "detail": f"model {result.model or 'not reported'}"}


def _show_status(entry: dict) -> None:
    harness = entry["harness"]
    if not entry["connected"]:
        echo(f"{harness}: not connected (no record in {', '.join(entry['looked_in'])})")
        return
    name = HARNESS_LOGINS[harness][0]
    codex = harness == "codex"
    where = f"profile {entry['profile']}" if codex else f"{entry['scope']} scope"
    heading(f"{harness} ({where})")
    window = entry["context_window"]
    login = {
        "own login": f"{name}'s own login",
        "no key": "no key",
        "gateway key": (
            f"gateway key from ${entry['key_env']}"
            if codex
            else "gateway key, in the settings"
        ),
    }[entry["login"]]
    rows = [
        ("record", entry["record"]),
        ("gateway", entry["gateway"] or "not recorded"),
        ("model", entry["model"] or "not recorded"),
    ]
    if codex:
        rows.append(("review model", entry["review_model"] or "the session model"))
    rows += [
        ("context window", f"{window:,} tokens" if window else "not set"),
        ("login", login),
    ]
    for index, item in enumerate(entry["files"]):
        state = item["state"]
        if item.get("changed"):
            state += f": {', '.join(item['changed'])}"
        rows.append(("files" if index == 0 else "", f"{item['path']}: {state}"))
    if entry.get("plugins"):
        rows.append(("vSR plugin", ", ".join(entry["plugins"])))
    probe = entry["probe"]
    rows.append(("gateway now", f"{probe['state']}: {probe['detail']}"))
    # not wrapped, so a path stays whole to copy
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        echo(f"  {label:<{width}}  {value}")


def _chatgpt_caller_backend(doc) -> bool:
    return any(
        ref.get("provider") == "openai"
        and str(ref.get("endpoint", "")).rstrip("/")
        == "https://chatgpt.com/backend-api/codex"
        and (ref.get("auth") or {}).get("credential") == "caller"
        for model in (doc or {}).get("providers", {}).get("models") or []
        for ref in model.get("backend_refs") or []
    )


def _codex_review_hint(model: str) -> str:
    """The config that sends Approve for me reviews to Codex's own reviewer on
    the person's ChatGPT login, for a gateway that already takes that login."""
    return f"""To review with Codex's own reviewer on your ChatGPT login, add this to the Router config:
  entrypoints:
    - model_names: [{model}]
      default: true
      review_model: vllm-sr/review
      # recipe: as before
    - model_names: [vllm-sr/review]
      recipe: review
  recipes:
    - name: review
      decisions:
        - name: review
          priority: 100
          rules: {{}}
          modelRefs: [{{model: codex-auto-review}}]
  providers:
    models:
      - name: codex-auto-review
        provider_model_id: codex-auto-review
        api_format: responses
        backend_refs:
          - name: chatgpt
            provider: openai
            endpoint: https://chatgpt.com/backend-api/codex
            auth: {{credential: caller}}
Any other model the Router serves can review instead."""


def _run(
    harness,
    changes,
    manifest_dir,
    key_label,
    doc,
    dry_run,
    yes,
    notes=(),
    plan=(),
    model=None,
):
    """Show the changes, check the Router configuration, and write; False on a dry run."""
    heading(f"vllm-sr connect {harness}")
    # what the diffs are worked out from: a file that holds anything else by the
    # time it is written is refused, not merged
    seen = {}
    try:
        for change in changes:
            seen[change.path] = old = hc.read(change.path)
            label, new = str(change.path), change.new_text(old=old or "")
            diff = difflib.unified_diff(
                change.shown(old or "", key_label).splitlines(keepends=True),
                change.shown(new, key_label).splitlines(keepends=True),
                fromfile=label if old is not None else "/dev/null",
                tofile=label,
            )
            echo("".join(diff) or f"{label}: no change")
    except ValueError as exc:
        raise click.ClickException(f"{change.path}: {exc}") from exc
    _check_router(harness, doc, model)
    for note in notes:
        echo(note)
    if plan:
        echo("vSR (System One) beside it, then:")
        for line in plan:
            echo(f"  {line}")
    if dry_run:
        echo("Dry run: nothing was written.")
        return False
    if all(change.old_text() == change.new_text() for change in changes):
        success("Nothing to change: the files already hold these settings.")
        return True
    if not yes and not click.confirm("Write these changes?", default=True):
        raise click.Abort()
    try:
        manifest = hc.apply_changes(changes, manifest_dir, seen)
    except hc.StaleWriteError as exc:
        raise click.ClickException(str(exc)) from exc
    except OSError as exc:
        raise click.ClickException(f"Writing failed: {exc}") from exc
    success(f"Written; undo with --undo (record: {manifest}).")
    return True


def _check_launch(launch, harness_args, undo, status_line=False, check=False) -> None:
    if harness_args and not launch:
        raise click.UsageError(
            "arguments after -- go to the harness only with --launch"
        )
    if launch and check:
        raise click.UsageError(
            "--launch starts the harness itself, so it does not take --check"
        )
    if undo and check:
        raise click.UsageError(
            "--undo leaves nothing to check, so it does not take --check"
        )
    if launch and (undo or status_line):
        flag = "--undo" if undo else "--status-line"
        raise click.UsageError(f"--launch writes nothing, so it does not take {flag}")


def _private(directory: Path, name: str, text: str) -> Path:
    """A file only its owner can read, in a temporary directory."""
    hc.write_private(directory / name, text)
    return directory / name


def _check(
    harness, argv_for, env, secrets, written=True, review_model=None, gateway=""
) -> None:
    """Run the harness once on a tiny task through the gateway, in a temporary
    directory, and report each step. argv_for(private, work) gives its command
    line and writes any file it needs into the private directory, which the
    harness does not work in."""
    name = HARNESS_LOGINS[harness][0]
    heading(f"vllm-sr connect {harness} --check")
    echo(
        f"Runs {name} once on a tiny task through the gateway. This spends a few"
        " short model calls on your login or key."
    )
    with (
        tempfile.TemporaryDirectory(prefix="vllm-sr-check-") as private,
        tempfile.TemporaryDirectory(prefix="vllm-sr-check-") as work,
    ):
        argv = argv_for(Path(private), Path(work))
        rows = hk.run(harness, argv, env, Path(work), secrets, review_model, gateway)
    fields((row.name, f"{row.state:<13}{row.detail}".rstrip()) for row in rows)
    if any(row.state == "failed" for row in rows):
        raise click.ClickException(
            f"The {name} check failed. "
            + (
                "The changes above stay written: fix what failed and run this"
                " command again, or take them back with --undo."
                if written
                else "Nothing was written."
            )
        )
    success(f"{name} works through the gateway.")


def _context_window(doc, model, given) -> tuple[int | None, list[str]]:
    """The flag, else the window every request for model fits in this
    configuration, with a line on where it came from or why there is none. That
    is the smallest window of the models model can reach, or, when its recipe
    drops the candidates a request no longer fits, the smallest over decisions
    of the window that still leaves the decision the candidates it needs."""
    if given or doc is None:
        return given, []
    groups = rc.entrypoint_candidates(doc, model)
    if not groups:
        return None, [
            f"Context window: the configuration maps {model} to no recipe or model,"
            " so none was read from it."
        ]
    windows = rc.context_windows(doc, [n for names, _ in groups for n in names])
    falls_over = rc.recipe_known_limits(doc, model)
    if falls_over:
        # a candidate without a known window never stays eligible
        groups = [([n for n in names if windows[n]], need) for names, need in groups]
    missing = sorted({n for names, _ in groups for n in names if windows[n] is None})
    if missing or any(len(names) < need for names, need in groups):
        one = len(missing) == 1
        return None, [
            f"Context window: not set: {', '.join(missing) or 'a decision'}, which"
            f" {model} can reach, {'states' if one else 'state'} no"
            f" context_window_size. Set it {'there' if one else 'on each'}, or pass"
            " --context-window."
        ]
    if falls_over:
        window = min(
            sorted((windows[n] for n in names), reverse=True)[need - 1]
            for names, need in groups
        )
        return window, [
            f"Context window: {window:,} tokens, the most every decision still"
            f" serves, since {model}'s recipe moves a long request to the"
            " decision's larger models."
        ]
    window = min(windows[n] for names, _ in groups for n in names)
    return window, [
        f"Context window: {window:,} tokens, "
        + (
            f"the smallest of the {len(windows)} models"
            if len(windows) > 1
            else "that of the one model"
        )
        + f" {model} can reach in this configuration."
    ]


def _launch(harness, argv, doc, gateway, model, token, notes, dry_run, settings=None):
    """Probe the gateway, then run the harness through it and exit with its status."""
    heading(f"vllm-sr connect {harness} --launch")
    _check_router(harness, doc, model)
    for note in notes:
        echo(note)
    found = shutil.which(argv[0])
    if found is None:
        raise click.ClickException(f"{argv[0]} is not on PATH.")
    with tempfile.TemporaryDirectory(prefix="vllm-sr-launch-") as private:
        path = Path(private) / "settings.json"
        if settings is not None:
            # --settings wins over the user's and the project's settings (2.1.323);
            # Codex reads its model catalog from the file an override names
            argv = [
                argv[0],
                *(
                    ["--settings", str(path)]
                    if harness == "claude-code"
                    else ["-c", f"model_catalog_json={json.dumps(str(path))}"]
                ),
                *argv[1:],
            ]
        echo(f"Starts: {shlex.join(argv)}")
        if dry_run:
            echo("Dry run: nothing was started.")
            raise click.exceptions.Exit(0)
        if settings is not None:
            hc.write_private(path, json.dumps(settings, indent=2))
        _probe_or_say_why(gateway, harness, model, token, doc, written=False)
        child = subprocess.Popen([found, *argv[1:]])

        # Ctrl-C belongs to the harness while it runs, and a closed terminal or a
        # stop reaches it through this process, which outlives it to remove the
        # private directory
        def forward(signum, _frame):
            child.send_signal(signum)

        handlers = {signal.SIGINT: signal.SIG_IGN, signal.SIGHUP: forward}
        handlers[signal.SIGTERM] = forward
        previous = {signum: signal.signal(signum, h) for signum, h in handlers.items()}
        try:
            code = child.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
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


def _require_key(doc, base_url, from_env, api_key_env, login) -> None:
    """Refuse to write a placeholder key for a listener that takes keys: it would
    replace a key written before, and every request would be refused."""
    listeners = (doc or {}).get("listeners") or [{}]
    if base_url or from_env or login or not (listeners[0] or {}).get("api_keys"):
        return
    raise click.ClickException(
        f"The listener takes an API key and ${api_key_env} is not set. Export it"
        " (or name another variable with --api-key-env) and run this again."
    )


def _check_router(harness: str, doc: dict | None, model: str | None = None) -> None:
    if doc is None:
        echo("Router configuration: not checked (pass --config to check it).")
        return
    findings = rc.check_router_config(doc, harness, model)
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


def _installs(harness_cli: str, source: str) -> list[list[str]]:
    """The commands that add the marketplace and install the plugin, in the agent's
    own CLI (Claude Code says install, Codex add)."""
    verb = "install" if harness_cli == "claude" else "add"
    return [
        [harness_cli, "plugin", "marketplace", "add", source],
        [harness_cli, "plugin", verb, VSR_PLUGIN],
    ]


def _vsr_step(step, plan: list[str]) -> bool:
    """Run a vSR step after the gateway settings are written, and whether it ran; a
    failure leaves those in place and says what to run by hand."""
    try:
        step()
    except click.ClickException as exc:
        warning(f"vSR is not set up: {exc.message}")
        for line in plan:
            echo(f"  {line}")
        return False
    return True


def _vsr_codex_installed(env: dict[str, str]) -> bool:
    found = _json_of(["codex", "plugin", "list", "--json"], env) or {}
    return any(
        str(p.get("pluginId", "")).startswith("vsr@") and p.get("enabled")
        for p in found.get("installed") or []
    )


def _vsr_claude_start(env, gateway, doc, vsr, from_env, key_env):
    """The runtime options to set, the installed plugin's id, and its current state,
    or no options with a line saying why vSR is left out. Read before anything is
    written, so a plugin that cannot be read or that holds a token is left alone."""
    options = _vsr_options(gateway, doc) if vsr else None
    if options and shutil.which("claude") is None:
        warning("vSR is not set up: claude is not on PATH.")
        return None, None, None
    plugin_id = _vsr_plugin_id(env) if options else None
    try:
        state = _plugin_state(env, plugin_id) if options and plugin_id else None
    except click.ClickException as exc:
        warning(f"vSR is not set up: {exc.message}")
        return None, None, None
    if state and not from_env and "runtimeToken" in state[1]:
        # Claude Code cannot clear a stored token, and the plugin sends it to any URL
        warning(
            f"vSR is not set up: {plugin_id} holds a runtime token from before, which"
            f" it would send to {options['runtimeUrl']}. Set ${key_env} to this"
            f" listener's key, or remove the token with /plugin configure {plugin_id}."
        )
        return None, None, None
    return options, plugin_id, state


def _vsr_options(gateway: str, doc: dict | None) -> dict[str, str] | None:
    """The runtime the vSR plugin asks: this listener's System One, or None when it
    publishes no model."""
    # the listener the gateway URL names, which --base-url may set to any of them
    models = rc.system_one_models(doc, urllib.parse.urlsplit(gateway).port)
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


def _management_url(gateway: str, doc: dict | None) -> str:
    """The Router management API beside the gateway, as `vllm-sr serve` publishes
    it: the gateway's host at the configured management port, 8080 by default,
    moved by the stack's port offset like the listener."""
    port = _configured_management_port(doc or {})
    host = urllib.parse.urlsplit(gateway).hostname or "localhost"
    host = f"[{host}]" if ":" in host else host
    return f"http://{host}:{port + resolve_runtime_stack().port_offset}"


def _vsr_claude_plan(options, plugin_id, install, key_env) -> list[str]:
    """What vSR's setup runs for Claude Code, as shown before it runs (the key named,
    never shown)."""
    if not options:
        return []
    shown = {
        **_claude_vsr_values(options),
        **({"runtimeToken": f"<${key_env}>"} if key_env else {}),
    }
    configure = f"claude plugin configure {plugin_id or VSR_PLUGIN} --values-stdin"
    return [*map(shlex.join, install), f"{configure} <<< '{json.dumps(shown)}'"]


def _claude_vsr_values(options: dict[str, str]) -> dict[str, str]:
    # a remote runtime, and the router app off: the gateway picks each call's model
    return {"runtime": "remote", **options, "router": "false"}


def _setup_vsr_claude(env, plugin_id, install, options, token, manifest_dir, before):
    """Install the plugin when it is missing and set its runtime through Claude Code's
    own command, which keeps the token in its credential store. --undo puts back the
    options it changed; the token stays."""
    for argv in install:
        _must(argv, env)
    plugin_id = plugin_id or VSR_PLUGIN
    configure = ["claude", "plugin", "configure", plugin_id]
    if before is None:
        before = _plugin_state(env, plugin_id)[0]
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
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"{shlex.join(argv)} exited with status {exc.returncode}"
        ) from exc
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


def _plugin_state(
    env: dict[str, str], plugin_id: str
) -> tuple[dict[str, str], set[str]]:
    """A plugin's option values as strings, and the options set at all; an option
    never set reads as its default, since `inputs` gives an unset string as empty.
    Fails rather than guess, since --vsr and --undo record and restore these."""
    found = _json_of(["claude", "plugin", "configure", plugin_id, "--json"], env)
    if not isinstance(found, dict) or not isinstance(found.get("inputs"), dict):
        raise click.ClickException(
            f"claude plugin configure {plugin_id} --json gave no options, so the"
            " plugin's settings were left as they were."
        )
    inputs, configured = found["inputs"], set(found.get("configured") or ())
    schema = found.get("schema") or {}
    values = {
        key: value if key in configured else schema.get(key, {}).get("default", value)
        for key, value in inputs.items()
    }
    strings = {
        k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in values.items()
    }
    return strings, configured


def _json_of(argv: list[str], env: dict[str, str]):
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, env=env, check=False
        )
        return json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def _caller_login(harness: str, doc: dict | None) -> bool:
    """Whether the gateway presents the harness's own login upstream. A gateway
    that presents another provider's login refuses every call of the harness, so
    nothing is written for it."""
    provider = rc.caller_provider(doc)
    if provider is None:
        return False
    name, own = HARNESS_LOGINS[harness]
    if provider != own:
        raise click.ClickException(
            f"The gateway presents each client's own {provider} login upstream, and"
            f" {name} sends a {own} login, so every call would be refused. Point"
            f" {name} at a gateway whose caller backends name {own}."
        )
    return True


def _probe_or_say_why(gateway, harness, model, token, doc, written=True) -> None:
    if rc.caller_provider(doc) is not None:
        echo(
            f"Not probed: the gateway answers only on the login {HARNESS_LOGINS[harness][0]}"
            " itself sends, which this command does not hold."
        )
        return
    _probe(gateway, harness, model, token, written)


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
    # every plugin is read before anything is undone, so a failed read keeps the record
    for plugin_id, back in hc.plugin_undo(
        manifest_dir, lambda one: _plugin_state(env, one)[0]
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
