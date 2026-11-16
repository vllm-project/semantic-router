"""A quick look at what the Router served, from Router Replay."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from http import HTTPStatus

import click

from cli.commands.recipe_learning import fetch_replay_payload
from cli.durations import parse_duration
from cli.router_management_client import (
    OBSERVABILITY_REPLAYS_PATH,
    RouterManagementClient,
    RouterManagementError,
)
from cli.terminal import echo, fields, heading

AGGREGATE_PATH = f"{OBSERVABILITY_REPLAYS_PATH}/aggregate"
# Models and entrypoints shown, those with the most requests.
TOP = 3
LIFECYCLE = ("completed", "failed", "aborted", "in_progress", "unknown")


@click.command("usage")
@click.option(
    "--since",
    help="Start of the window, as a time back from now such as 30m, 1h or 7d."
    "  [default: midnight today, local time]",
)
@click.option(
    "--endpoint",
    default=None,
    help="Router management base URL; defaults to the local Router API port.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the summary as JSON.")
def usage(since: str | None, endpoint: str | None, as_json: bool) -> None:
    """Show the requests, spend and savings the Router recorded in a window.

    Read from Router Replay's list and aggregate, the figures Insights shows,
    with $VSR_MGMT_TOKEN as the bearer token when it is set. Top models and
    entrypoints are those with the most requests, by selected model and by the
    model the client asked for; each row adds its requests' own priced costs,
    the Router's calls for them included, as the window's spend does.
    """

    start = _window_start(since)
    try:
        client = RouterManagementClient(endpoint)
        summary = _summary(client, start)
    except RouterManagementError as exc:
        if exc.status == HTTPStatus.NOT_FOUND and exc.code == "REPLAY_UNAVAILABLE":
            raise click.ClickException(
                f"Router Replay is off at {client.base_url}, so no usage is"
                " recorded: set enabled: true on a router_replay plugin in"
                " global.plugins and restart the Router."
            ) from exc
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:  # an endpoint the client refuses
        raise click.ClickException(str(exc)) from exc
    if as_json:
        echo(json.dumps(summary, indent=2))
    else:
        _show(summary)


def _window_start(since: str | None) -> datetime:
    if since is None:
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.astimezone()
    try:
        days = re.fullmatch(r"(\d+)d", since)
        back = (
            timedelta(days=int(days[1]))
            if days
            else timedelta(seconds=float(parse_duration(since)))
        )
        # elapsed time, so a clock change inside the window does not move it
        start = datetime.now(timezone.utc) - back
    except (ValueError, OverflowError) as exc:
        raise click.BadParameter(
            "use a time back from now such as 30m, 1h or 7d", param_hint="--since"
        ) from exc
    return start.replace(microsecond=0).astimezone()


def _summary(client: RouterManagementClient, start: datetime) -> dict:
    window = {"since": start.isoformat()}
    aggregate = client.request("GET", AGGREGATE_PATH, params=window).payload
    # a Router without the since filter lists a record even from tomorrow on
    tomorrow = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=1)
    probe = {"since": tomorrow.isoformat(), "limit": "1"}
    if client.request("GET", OBSERVABILITY_REPLAYS_PATH, params=probe).payload.get(
        "data"
    ):
        raise _older_router(client, "ignores the since filter")
    summary = aggregate.get("summary") or {}
    if any("gross_saved" not in group for group in summary.get("by_currency") or []):
        raise _older_router(client, "reports savings with its own calls left in")
    records = fetch_replay_payload(
        client.base_url,
        max(1, aggregate.get("record_count", 0)),
        client.timeout,
        window,
    )["data"]
    lifecycle = aggregate.get("lifecycle") or {}
    return {
        "endpoint": client.base_url,
        "since": window["since"],
        "requests": {
            "total": aggregate.get("record_count", 0),
            **{state: lifecycle.get(state, 0) for state in LIFECYCLE},
        },
        "sessions": len({r["session_id"] for r in records if r.get("session_id")}),
        "cost": summary.get("by_currency") or [],
        "prompt_cache": (aggregate.get("provider_cache") or {}).get("by_currency")
        or [],
        "reviews": aggregate.get("reviews"),
        "models": _top(records, "selected_model"),
        "entrypoints": _top(records, "original_model"),
    }


def _older_router(client: RouterManagementClient, why: str) -> click.ClickException:
    return click.ClickException(
        f"The Router at {client.base_url} {why}: it predates this command and"
        " would give wrong figures. Upgrade the Router."
    )


def _top(records: list[dict], key: str) -> list[dict]:
    """The TOP names under key with the most requests, each with its requests'
    priced costs by currency and how many have none."""
    rows: dict[str, dict] = {}
    for record in records:
        if not record.get(key):
            continue
        row = rows.setdefault(
            record[key],
            {"name": record[key], "requests": 0, "spend": {}, "unpriced": 0},
        )
        row["requests"] += 1
        costs = _costs(record)
        row["unpriced"] += costs == {}
        for currency, amount in (costs or {}).items():
            row["spend"][currency] = row["spend"].get(currency, 0) + amount
    ranked = sorted(rows.values(), key=lambda row: (-row["requests"], row["name"]))
    for row in ranked:
        row["spend"] = dict(sorted(row["spend"].items()))
    return ranked[:TOP]


def _costs(record: dict) -> dict[str, float] | None:
    """A record's priced costs by currency, as the aggregate's total_spend adds
    them: the selected model's and the Router's own calls'. None for a Looper
    hop, whose cost the record of the request that made it carries."""
    accounting = record.get("cost_accounting") or {}
    if accounting.get("unpriced_reason") == "counted_in_parent":
        return None
    costs: dict[str, float] = {}
    for priced in [record, *(accounting.get("router_calls") or [])]:
        # a record's own cost is actual_cost, a Router call's is cost
        cost = priced.get("actual_cost" if priced is record else "cost")
        code = (priced.get("currency") or "").strip().upper()
        if code and isinstance(cost, (int, float)):
            costs[code] = costs.get(code, 0) + cost
    return costs


def _requests(count: int) -> str:
    return f"{count} request" if count == 1 else f"{count} requests"


def _money(amount: float, currency: str) -> str:
    return f"{amount:.6f} {currency}"


def _show(summary: dict) -> None:
    requests = summary["requests"]
    counted = f"{requests['total']}: " + ", ".join(
        f"{requests[state]} {state.replace('_', ' ')}"
        for state in LIFECYCLE
        if state in ("completed", "failed", "aborted") or requests[state]
    )
    rows = [("Requests", counted), ("Sessions", summary["sessions"])]
    for cost in summary["cost"]:
        currency = cost["currency"]
        spend = _money(cost["total_spend"], currency)
        if cost["router_call_spend"]:
            spend += f" (Router calls {_money(cost['router_call_spend'], currency)})"
        saved = "n/a: no request was compared with a baseline"
        if cost["cost_record_count"]:
            saved = (
                f"{_money(cost['total_saved'], currency)} (gross"
                f" {_money(cost['gross_saved'], currency)}, Router calls netted"
                f" {_money(cost['netted_router_call_spend'], currency)},"
                f" {_requests(cost['cost_record_count'])} compared)"
            )
        rows += [("Spend", spend), ("Net savings", saved)]
    if not summary["cost"]:
        rows.append(("Spend", "no priced request"))
    for cache in summary["prompt_cache"]:
        rows.append(
            (
                "Prompt cache",
                f"saved {_money(cache['net_savings'], cache['currency'])} over"
                f" {_requests(cache['priced_records'])}",
            )
        )
    reviews = summary["reviews"]
    if reviews and reviews.get("reviews"):
        verdicts = "no verdict"
        if reviews.get("approval_rate") is not None:
            verdicts = (
                f"approval {reviews['approval_rate']:.0%} ({reviews['allowed']}"
                f" allowed, {reviews['denied']} denied)"
            )
        rows.append(("Reviews", f"{reviews['reviews']}, {verdicts}"))
    heading(f"Usage since {summary['since']} ({summary['endpoint']})")
    fields(rows)
    for title, key in (("Top models", "models"), ("Top entrypoints", "entrypoints")):
        if summary[key]:
            heading(title)
            fields((row["name"], _row_text(row)) for row in summary[key])


def _row_text(row: dict) -> str:
    priced = [_money(amount, currency) for currency, amount in row["spend"].items()]
    if not priced:
        return f"{_requests(row['requests'])}, no price"
    if row["unpriced"]:
        priced.append(f"{row['unpriced']} with no price")
    return f"{_requests(row['requests'])}, " + ", ".join(priced)
