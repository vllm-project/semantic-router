"""Inspect a CLI-owned instance and attach its private host controller."""

import json

import click

from cli.instance_controller import run
from cli.instance_setup import (
    attach_controller,
    controller_request,
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
    """Inspect the serving state of an existing local instance."""
    ctx.ensure_object(dict)
    ctx.obj["instance_directory"] = instance_directory(config_file)
    ctx.obj["instance_config_file"] = config_file


@instance.command(hidden=True)
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


@instance.command(hidden=True)
@click.pass_context
def controller(ctx):
    """Run the persistent controller in a host service supervisor."""
    run(ctx.obj["instance_directory"])
