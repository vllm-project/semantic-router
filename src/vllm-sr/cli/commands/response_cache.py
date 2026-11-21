"""Router-managed response cache inspection and lifecycle commands."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from cli.commands.common import exit_with_logged_error
from cli.router_management_client import (
    RESPONSE_CACHE_PATH,
    RouterManagementClient,
)
from cli.terminal import echo, fields, heading, success
from cli.utils import get_logger

log = get_logger(__name__)


def _build_selector(
    *,
    recipe: str | None = None,
    decision: str | None = None,
    request_model: str | None = None,
    selected_model: str | None = None,
    protocol: str | None = None,
    namespace: str | None = None,
    epoch: str | None = None,
) -> dict[str, Any] | None:
    partition: dict[str, str] = {}
    if recipe:
        partition["recipe"] = recipe
    if decision:
        partition["decision"] = decision
    if request_model:
        partition["request_model"] = request_model
    if selected_model:
        partition["selected_model"] = selected_model
    if protocol:
        partition["protocol"] = protocol
    if namespace:
        partition["namespace"] = namespace
    if epoch:
        partition["epoch"] = epoch

    selector: dict[str, Any] = {}
    if partition:
        selector["partition"] = partition
    if epoch:
        selector["epoch"] = epoch
    return selector or None


def _is_interactive() -> bool:
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _selector_options(func):
    options = [
        click.option(
            "--recipe", default=None, help="Scope operation to a recipe partition."
        ),
        click.option(
            "--decision", default=None, help="Scope operation to a decision partition."
        ),
        click.option(
            "--request-model",
            default=None,
            help="Scope operation to an incoming requested model.",
        ),
        click.option(
            "--selected-model",
            default=None,
            help="Scope operation to a routed target model.",
        ),
        click.option("--protocol", default=None, help="Scope operation to a protocol."),
        click.option(
            "--namespace",
            default=None,
            help="Scope operation to a partition namespace.",
        ),
        click.option(
            "--epoch", default=None, help="Scope operation to a partition epoch."
        ),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func


@click.group("response-cache")
def response_cache_group() -> None:
    """Inspect or manage the Router response cache."""


@response_cache_group.command("capabilities")
@click.option(
    "--endpoint",
    default=None,
    help="Router management origin or /api/v1 root.",
)
@click.option("--timeout", type=float, default=15.0, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Print the raw JSON response payload.",
)
@exit_with_logged_error(log)
def response_cache_capabilities(
    endpoint: str | None,
    timeout: float,
    token_env: str,
    output_json: bool,
) -> None:
    """Get response cache backend capabilities."""

    client = RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)
    data = client.get_response_cache_capabilities().payload
    if output_json:
        click.echo(json.dumps(data, indent=2, sort_keys=True))
        return

    payload = data if isinstance(data, dict) else {}
    heading("Response cache capabilities")
    fields((("Endpoint", client.base_url + RESPONSE_CACHE_PATH + "/capabilities"),))
    echo()
    fields(
        [
            ("Backend", str(payload.get("backend", "-"))),
            ("Exact", str(payload.get("exact", False)).lower()),
            ("Semantic", str(payload.get("semantic", False)).lower()),
            ("Per-entry TTL", str(payload.get("per_entry_ttl", False)).lower()),
            ("Stats", str(payload.get("stats", False)).lower()),
            ("Invalidate", str(payload.get("invalidate", False)).lower()),
            ("Flush", str(payload.get("flush", False)).lower()),
            ("Shared", str(payload.get("shared", False)).lower()),
            ("Consistency model", str(payload.get("consistency_model") or "-")),
        ]
    )


@response_cache_group.command("stats")
@click.option(
    "--endpoint",
    default=None,
    help="Router management origin or /api/v1 root.",
)
@click.option("--timeout", type=float, default=15.0, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Print the raw JSON response payload.",
)
@exit_with_logged_error(log)
def response_cache_stats(
    endpoint: str | None,
    timeout: float,
    token_env: str,
    output_json: bool,
) -> None:
    """Get response cache performance and usage statistics."""

    client = RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)
    data = client.get_response_cache_stats().payload
    if output_json:
        click.echo(json.dumps(data, indent=2, sort_keys=True))
        return

    payload = data if isinstance(data, dict) else {}
    heading("Response cache statistics")
    fields((("Endpoint", client.base_url + RESPONSE_CACHE_PATH + "/stats"),))
    echo()
    fields(
        [
            ("Total entries", str(payload.get("total_entries", 0))),
            ("Hit count", str(payload.get("hit_count", 0))),
            ("Miss count", str(payload.get("miss_count", 0))),
            ("Hit ratio", str(payload.get("hit_ratio", 0.0))),
            ("Exact hit count", str(payload.get("exact_hit_count", 0))),
            ("Semantic hit count", str(payload.get("semantic_hit_count", 0))),
            ("L1 hit count", str(payload.get("l1_hit_count", 0))),
            ("L2 hit count", str(payload.get("l2_hit_count", 0))),
            ("L1 entries", str(payload.get("l1_entries", 0))),
            ("Singleflight waiters", str(payload.get("singleflight_waiters", 0))),
            ("Invalidation count", str(payload.get("invalidation_count", 0))),
        ]
    )


@response_cache_group.command("health")
@click.option(
    "--endpoint",
    default=None,
    help="Router management origin or /api/v1 root.",
)
@click.option("--timeout", type=float, default=15.0, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Print the raw JSON response payload.",
)
@exit_with_logged_error(log)
def response_cache_health(
    endpoint: str | None,
    timeout: float,
    token_env: str,
    output_json: bool,
) -> None:
    """Check response cache backend health."""

    client = RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)
    data = client.get_response_cache_health().payload
    if output_json:
        click.echo(json.dumps(data, indent=2, sort_keys=True))
        return

    payload = data if isinstance(data, dict) else {}
    status = payload.get("status", "unknown")
    caps = payload.get("capabilities") or {}
    backend = caps.get("backend", "-")

    heading("Response cache health")
    fields((("Endpoint", client.base_url + RESPONSE_CACHE_PATH + "/health"),))
    echo()
    fields(
        [
            ("Status", str(status)),
            ("Backend", str(backend)),
        ]
    )


@response_cache_group.command("flush")
@click.option(
    "--yes",
    "-y",
    "confirmed",
    is_flag=True,
    help="Confirm flush without interactive prompt.",
)
@_selector_options
@click.option(
    "--endpoint",
    default=None,
    help="Router management origin or /api/v1 root.",
)
@click.option("--timeout", type=float, default=30.0, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Print the raw JSON response payload.",
)
@exit_with_logged_error(log)
def response_cache_flush(
    confirmed: bool,
    recipe: str | None,
    decision: str | None,
    request_model: str | None,
    selected_model: str | None,
    protocol: str | None,
    namespace: str | None,
    epoch: str | None,
    endpoint: str | None,
    timeout: float,
    token_env: str,
    output_json: bool,
) -> None:
    """Advance response-cache epoch or flush scoped cache entries."""

    if not confirmed:
        if not _is_interactive():
            raise click.ClickException(
                "Confirmation required. Run with --yes in non-interactive environments."
            )
        if not click.confirm("Are you sure you want to flush the response cache?"):
            echo("Aborted.")
            return

    client = RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)
    selector = _build_selector(
        recipe=recipe,
        decision=decision,
        request_model=request_model,
        selected_model=selected_model,
        protocol=protocol,
        namespace=namespace,
        epoch=epoch,
    )
    data = client.flush_response_cache(selector=selector).payload
    if output_json:
        click.echo(json.dumps(data, indent=2, sort_keys=True))
        return

    payload = data if isinstance(data, dict) else {}
    deleted = payload.get("deleted", 0)
    matched = payload.get("matched_estimate", 0)
    success("Response cache flushed")
    fields(
        [
            ("Deleted", str(deleted)),
            ("Matched estimate", str(matched)),
        ]
    )


@response_cache_group.command("invalidate")
@click.option(
    "--execute/--dry-run",
    default=False,
    show_default=True,
    help="Execute invalidation (default is dry-run mode for safety).",
)
@_selector_options
@click.option(
    "--endpoint",
    default=None,
    help="Router management origin or /api/v1 root.",
)
@click.option("--timeout", type=float, default=30.0, show_default=True)
@click.option("--token-env", default="VSR_MGMT_TOKEN", show_default=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Print the raw JSON response payload.",
)
@exit_with_logged_error(log)
def response_cache_invalidate(
    execute: bool,
    recipe: str | None,
    decision: str | None,
    request_model: str | None,
    selected_model: str | None,
    protocol: str | None,
    namespace: str | None,
    epoch: str | None,
    endpoint: str | None,
    timeout: float,
    token_env: str,
    output_json: bool,
) -> None:
    """Dry-run or invalidate scoped response-cache partitions."""

    client = RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)
    selector = _build_selector(
        recipe=recipe,
        decision=decision,
        request_model=request_model,
        selected_model=selected_model,
        protocol=protocol,
        namespace=namespace,
        epoch=epoch,
    )
    dry_run = not execute
    data = client.invalidate_response_cache(selector=selector, dry_run=dry_run).payload
    if output_json:
        click.echo(json.dumps(data, indent=2, sort_keys=True))
        return

    payload = data if isinstance(data, dict) else {}
    matched = payload.get("matched_estimate", 0)
    deleted = payload.get("deleted", 0)
    if dry_run:
        heading("Response cache invalidation (dry-run)")
        fields(
            [
                ("Matched estimate", str(matched)),
                ("Deleted", str(deleted)),
            ]
        )
        echo("Pass --execute to perform actual invalidation.")
    else:
        success("Response cache invalidation completed")
        fields(
            [
                ("Deleted", str(deleted)),
                ("Matched estimate", str(matched)),
            ]
        )
