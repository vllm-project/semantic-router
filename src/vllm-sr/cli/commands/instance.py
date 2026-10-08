"""Inspect or deploy one CLI-owned instance without replacing its Dashboard."""

import json
import uuid

import click

from cli.instance_controller import run
from cli.instance_setup import (
    attach_controller,
    controller_request,
    ensure_controller,
    instance_directory,
)


@click.group()
@click.option(
    "--config",
    "config_file",
    default="config.yaml",
    type=click.Path(exists=True, dir_okay=False),
)
@click.pass_context
def instance(ctx, config_file):
    """Manage the Router/Engine mode of an existing local instance."""
    ctx.ensure_object(dict)
    ctx.obj["instance_directory"] = instance_directory(config_file)
    ctx.obj["instance_config_file"] = config_file


@instance.command()
@click.option(
    "--runtime-config", type=click.Path(exists=True, dir_okay=False), default=None
)
@click.option(
    "--gateway", type=click.Choice(["standalone", "extproc"]), default="extproc"
)
@click.option("--startup-timeout", type=click.IntRange(min=1), default=600)
@click.pass_context
def attach(ctx, runtime_config, gateway, startup_timeout):
    """Attach control to an existing canonical local stack after an image rollout."""
    source = ctx.obj["instance_config_file"]
    attach_controller(source, runtime_config or source, {}, gateway, startup_timeout)
    click.echo(json.dumps(controller_request(ctx.obj["instance_directory"], "/status")))


@instance.command()
@click.pass_context
def models(ctx):
    """Print actual native model cards for readiness checks, without inference."""
    try:
        click.echo(
            json.dumps(controller_request(ctx.obj["instance_directory"], "/models"))
        )
    except (OSError, ValueError) as error:
        raise click.ClickException("Instance model inventory is unavailable") from error


@instance.command()
@click.pass_context
def status(ctx):
    """Print desired/observed mode and durable operation state."""
    try:
        click.echo(
            json.dumps(controller_request(ctx.obj["instance_directory"], "/status"))
        )
    except (OSError, ValueError) as error:
        raise click.ClickException("No reachable local instance controller") from error


@instance.command()
@click.option("--mode", type=click.Choice(["router", "engine"]), required=True)
@click.option(
    "--deployment",
    default=None,
    help="Configured model deployment; omit to preserve the selected model.",
)
@click.option(
    "--request-id", default=None, help="Reuse this ID to safely retry an operation."
)
@click.pass_context
def deploy(ctx, mode, deployment, request_id):
    """Publish a rollback-protected frontend capability generation."""
    try:
        ensure_controller(ctx.obj["instance_directory"])
        state = controller_request(
            ctx.obj["instance_directory"],
            "/deploy",
            {
                "mode": mode,
                "deployment": deployment,
                "request_id": request_id or uuid.uuid4().hex,
            },
        )
        click.echo(json.dumps(state))
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error


@instance.command()
@click.pass_context
def controller(ctx):
    """Run the persistent controller in a host service supervisor."""
    run(ctx.obj["instance_directory"])
