"""`vllm-sr connect pi`: route the Pi coding agent through a Router gateway."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import click

from cli import harness_check as hk
from cli import harness_connect as hc
from cli import harness_pi as hp
from cli import harness_router_check as rc
from cli.commands import connect as base
from cli.terminal import echo


def _pi_options(command):
    return base._options(command, launch_and_vsr=False)


@click.command("pi")
@_pi_options
def pi(
    router_config,
    base_url,
    model,
    api_key_env,
    dry_run,
    undo,
    context_window,
    yes,
    check,
    key_command,
):
    """Route Pi's model calls through the gateway, as a provider in its models.json."""
    env = dict(os.environ)
    home = hp.agent_dir(env)
    base._check_launch(False, (), undo, check=check)
    if undo:
        base._undo(home)
        return
    gateway, doc = base._gateway(router_config, base_url)
    caller = rc.caller_provider(doc, model)
    if caller:
        raise click.ClickException(
            f"The gateway presents each client's own {caller} login upstream, and Pi"
            " sends only the key its models.json gives the provider, so calls to"
            " those models would be refused. Point Pi at a gateway without caller"
            " backends."
        )
    if hp.stored_key(home):
        raise click.ClickException(
            f"{hp.STORED_KEY} ({home / 'auth.json'}). Remove its {hp.PROVIDER} entry"
            " and run this again."
        )
    token, from_env = hc.token_from(env, api_key_env)
    if key_command:
        token, from_env = "", False
    else:
        base._require_key(doc, base_url, from_env, api_key_env, False)
    key = base._key("pi", key_command, False, env, Path.cwd(), check or not dry_run)
    key_env = api_key_env if from_env else None
    context_window, notes = base._context_window(doc, model, context_window)
    entry = hp.provider(
        gateway,
        model,
        key_env,
        context_window,
        key_command,
        base._output_limit(doc, model),
    )
    secrets = [token] if from_env else [key] if key else []
    change = hp.models_change(home, entry)
    if not base._run("pi", [change], home, "", doc, dry_run, yes, notes, (), model):
        if check:
            _check_unwritten(entry, model, env, secrets)
        return
    if from_env:
        echo(f"Pi reads the key from ${api_key_env}; keep it set when you start it.")
    if key_command:
        echo(
            f'Pi runs {key_command} ("!" in its apiKey) for the key itself, before'
            " each request, and waits up to 10 seconds for it."
        )
    if not context_window:
        echo(
            f"Pi does not know {model}, so it assumes a window of 128,000 tokens;"
            " --context-window gives it the gateway's."
        )
    base._probe_or_say_why(gateway, "pi", model, key or token, doc)
    if check:
        base._check("pi", lambda *_: hk.pi_argv(hp.PROVIDER, model), env, secrets)
    echo(f"Start Pi with: pi --provider {hp.PROVIDER} --model {model}")


def _check_unwritten(entry: dict, model: str, env: dict, secrets: list[str]) -> None:
    """Check the provider a dry run would write. Pi reads models.json only from
    its agent directory, so the check runs with one of its own, which holds that
    file alone: no login or setting of yours is copied into it."""
    with tempfile.TemporaryDirectory(prefix="vllm-sr-check-") as agent:
        text = json.dumps({"providers": {hp.PROVIDER: entry}}, indent=2)
        hc.write_private(Path(agent) / "models.json", text)
        base._check(
            "pi",
            lambda *_: hk.pi_argv(hp.PROVIDER, model),
            {**env, hp.AGENT_DIR_VARIABLE: agent},
            secrets,
            written=False,
        )
