"""Organize reusable evaluation baselines and recipe candidates."""

import json

import click

from cli.sr_bench.experiments import EXPERIMENT_ID, ROLES


def _request(client, method, path, body=None):
    try:
        value = (
            client.request(method, path, body)
            if body is not None
            else client.request(method, path)
        )
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


@click.group()
def experiment():
    """Group durable baseline, routing checks, candidates and validation runs."""


@experiment.command("create")
@click.argument("name")
@click.option("--idempotency-key")
@click.pass_obj
def create(client, name, idempotency_key):
    """Create an experiment without submitting model work."""
    _request(
        client,
        "POST",
        "/experiments",
        {"name": name, "idempotency_key": idempotency_key},
    )


@experiment.command("list")
@click.option("--after", type=click.IntRange(min=0), default=0)
@click.option("--limit", type=click.IntRange(1, 50), default=20)
@click.pass_obj
def list_experiments(client, after, limit):
    """Read one page of experiments."""
    _request(client, "GET", f"/experiments?after={after}&limit={limit}")


@experiment.command("show")
@click.argument("experiment_id")
@click.option("--after", type=click.IntRange(min=0), default=0)
@click.option("--limit", type=click.IntRange(1, 50), default=20)
@click.pass_obj
def show(client, experiment_id, after, limit):
    """Read the experiment and one page of its linked runs."""
    _identifier(experiment_id)
    _request(
        client, "GET", f"/experiments/{experiment_id}/runs?after={after}&limit={limit}"
    )


@experiment.command("attach")
@click.argument("experiment_id")
@click.option("--run", "run_id", required=True)
@click.option("--role", type=click.Choice(sorted(ROLES)), required=True)
@click.option("--hypothesis", default="")
@click.pass_obj
def attach(client, experiment_id, run_id, role, hypothesis):
    """Link existing evidence without changing or rerunning it."""
    _identifier(experiment_id)
    _request(
        client,
        "POST",
        f"/experiments/{experiment_id}/runs",
        {"run_id": run_id, "role": role, "hypothesis": hypothesis},
    )


@experiment.command("delete")
@click.argument("experiment_id")
@click.pass_obj
def delete(client, experiment_id):
    """Delete a finished experiment's grouping and links; keep every run and result."""
    _identifier(experiment_id)
    _request(client, "DELETE", f"/experiments/{experiment_id}")


def _identifier(value):
    if not EXPERIMENT_ID.fullmatch(value):
        raise click.ClickException("Invalid experiment identity")
