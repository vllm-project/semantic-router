"""The sr-bench 1.0 CLI shares one durable service with the Dashboard."""

from __future__ import annotations

import json
import os
import time
from functools import wraps
from pathlib import Path

import click

from cli.sr_bench.client import Client
from cli.sr_bench.contracts import catalog, load_document, plan
from cli.sr_bench.service import DEFAULT_STORE, DEFAULT_URL, serve
from cli.sr_bench.store import TERMINAL


def guarded(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (OSError, ValueError, KeyError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapped


def output(value):
    click.echo(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


@click.group()
@click.option(
    "--url",
    envvar="SR_BENCH_URL",
    help="Shared sr-bench service URL; discovers the current local stack by default.",
)
@click.option(
    "--store",
    type=click.Path(path_type=Path),
    envvar="SR_BENCH_STORE",
    help="Durable service store; discovers the current local stack by default.",
)
@click.option(
    "--no-autostart", is_flag=True, help="Require an already running service."
)
@click.pass_context
def benchmark(ctx, url, store, no_autostart):
    """Prepare, run, inspect, and compare sr-bench 1.0 evaluations."""
    from cli.runtime_stack import resolve_runtime_stack

    stack = resolve_runtime_stack()
    root = (
        Path(os.environ.get("VLLM_SR_STATE_ROOT_DIR", str(Path.cwd())))
        .expanduser()
        .resolve()
        / ".sr-bench"
        / stack.stack_name
    )
    managed = root / "store"
    if store is None and managed.is_dir():
        store = managed
    store = Path(store or DEFAULT_STORE).expanduser().resolve()
    token_file = store.parent / "service-token"
    managed_service = token_file.is_file()
    if url is None:
        url = (
            f"http://127.0.0.1:{stack.sr_bench_port}"
            if managed_service
            else DEFAULT_URL
        )
    client = Client(url, store, not no_autostart and not managed_service)
    if managed_service and "Authorization" not in client.headers:
        if token_file.is_symlink() or token_file.stat().st_mode & 0o077:
            raise click.ClickException(
                "Managed service token must be a private regular file"
            )
        client.headers["Authorization"] = "Bearer " + token_file.read_text().strip()
    ctx.obj = client


@benchmark.command("catalog")
def catalog_command():
    """Show the nine benchmark adapters and evaluation profiles."""
    output(catalog())


@benchmark.command("serve")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8090, type=int)
@click.option(
    "--store-identity",
    help="Canonical host store path SHA256 for an isolated runtime container.",
)
@click.pass_obj
@guarded
def serve_command(client, host, port, store_identity):
    """Own the durable journal and workers independently of a browser."""
    serve(client.store, host, port, store_identity)


@benchmark.group("dataset")
def dataset():
    """Prepare reproducible fixed benchmark case sets."""


@dataset.command("prepare")
@click.option("--benchmark", "benchmark_id", required=True)
@click.option(
    "--profile", type=click.Choice(["smoke", "quick", "standard"]), default="quick"
)
@click.option("--source-path", type=click.Path(path_type=Path))
@click.option("--revision")
@click.option("--seed", default=20260918, type=int)
@click.option(
    "--limit",
    type=int,
    help="Custom case cap; cannot be represented as an upstream full benchmark.",
)
@click.pass_obj
@guarded
def dataset_prepare(client, benchmark_id, profile, source_path, revision, seed, limit):
    """Download or read a pinned source and freeze a reusable dataset."""
    from cli.sr_bench.sources import prepare_dataset

    kwargs = dict(
        benchmark=benchmark_id,
        profile=profile,
        store=client.store,
        source_path=str(source_path) if source_path else None,
        revision=revision,
        seed=seed,
    )
    if limit is not None:
        kwargs["limit"] = limit
    output(prepare_dataset(**kwargs))


@dataset.command("combine")
@click.argument(
    "manifests", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True
)
@click.pass_obj
@guarded
def dataset_combine(client, manifests):
    """Create a reusable multi-benchmark dataset from prepared manifests."""
    from cli.sr_bench.sources import combine_datasets

    output(combine_datasets([load_document(path) for path in manifests], client.store))


@dataset.command("show")
@click.argument("path", type=click.Path(exists=True, path_type=Path), required=False)
@click.pass_obj
@guarded
def dataset_show(client, path):
    """Inspect a frozen dataset or list datasets in the shared store."""
    output(load_document(path) if path else client.request("GET", "/datasets"))


@benchmark.command("plan")
@click.option("--manifest", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "destination", type=click.Path(path_type=Path))
@guarded
def plan_command(manifest, destination):
    """Validate and freeze all cases, targets, profiles, and limits without inference."""
    frozen = plan(load_document(manifest))
    if destination:
        with destination.open("x") as file:
            file.write(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    output(
        {
            "manifest": frozen,
            "plan_sha256": frozen["plan_sha256"],
            "total": len(frozen["cases"]) * len(frozen["targets"]),
        }
    )


def _run(client, manifest, detach, idempotency_key, preview=False):
    document = load_document(manifest)
    if preview:
        document["mode"] = "preview"
    run = client.request(
        "POST", "/runs", {"manifest": document, "idempotency_key": idempotency_key}
    )
    if detach:
        output(run)
        return
    click.echo(
        f"Run {run['id']} submitted; Ctrl-C leaves the service running. Use benchmark cancel to stop.",
        err=True,
    )
    while run["status"] not in TERMINAL:
        time.sleep(0.5)
        run = client.request("GET", "/runs/" + run["id"])
    output(client.request("GET", "/runs/" + run["id"] + "/report"))
    if run["status"] != "completed":
        raise click.exceptions.Exit(2)


def run_options(fn):
    fn = click.option(
        "--manifest", type=click.Path(exists=True, path_type=Path), required=True
    )(fn)
    fn = click.option(
        "--detach", is_flag=True, help="Return immediately with a durable run ID."
    )(fn)
    fn = click.option(
        "--idempotency-key",
        help="Bind repeated submissions to the same frozen plan, without reissuing calls.",
    )(fn)
    return fn


@benchmark.command("run")
@run_options
@click.pass_obj
@guarded
def run_command(client, manifest, detach, idempotency_key):
    """Execute one frozen live evaluation through the shared service."""
    _run(client, manifest, detach, idempotency_key)


@benchmark.command("preview")
@run_options
@click.pass_obj
@guarded
def preview_command(client, manifest, detach, idempotency_key):
    """Inspect routing decisions without producing quality scores."""
    _run(client, manifest, detach, idempotency_key, True)


@benchmark.command("runs")
@click.pass_obj
@guarded
def runs_command(client):
    output(client.request("GET", "/runs"))


@benchmark.command("show")
@click.argument("run_id")
@click.option("--results", is_flag=True)
@click.option("--calls", is_flag=True)
@click.option("--events", is_flag=True)
@click.pass_obj
@guarded
def show_command(client, run_id, results, calls, events):
    """Inspect status, case results, calls, or durable events."""
    if sum((results, calls, events)) > 1:
        raise ValueError("Choose one of results/calls/events")
    suffix = (
        "/results" if results else "/calls" if calls else "/events" if events else ""
    )
    output(client.request("GET", "/runs/" + run_id + suffix))


@benchmark.command("cancel")
@click.argument("run_id")
@click.pass_obj
@guarded
def cancel_command(client, run_id):
    """Cancel remaining work while retaining all existing evidence."""
    output(client.request("POST", "/runs/" + run_id + "/cancel", {}))


@benchmark.command("report")
@click.argument("run_id")
@click.option("--output", "destination", type=click.Path(path_type=Path))
@click.pass_obj
@guarded
def report_command(client, run_id, destination):
    """Show quality, four-bucket usage, cost, latency, time, and limitations."""
    report = client.request("GET", "/runs/" + run_id + "/report")
    if destination:
        with destination.open("x") as file:
            file.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    output(report)


@benchmark.command("compare")
@click.argument("baseline_run_id")
@click.argument("candidate_run_id")
@click.pass_obj
@guarded
def compare_command(client, baseline_run_id, candidate_run_id):
    """Compare matched cases against the best observed single model."""
    output(
        client.request(
            "POST",
            "/comparisons",
            {"baseline_run_id": baseline_run_id, "candidate_run_id": candidate_run_id},
        )
    )


@benchmark.group("target")
def target_group():
    """Manage the operator-owned target registry used by the Dashboard."""


@target_group.command("list")
@click.pass_obj
@guarded
def target_list(client):
    output(client.request("GET", "/targets"))


@target_group.command("register")
@click.option(
    "--file", "source", type=click.Path(exists=True, path_type=Path), required=True
)
@click.pass_obj
@guarded
def target_register(client, source):
    """Replace the local registry from a JSON list of credential references."""
    document = load_document(source)
    targets = document.get("targets") if isinstance(document, dict) else document
    validated = plan(
        {
            "version": "sr-bench-1.0",
            "cost_policy": "capability_only",
            "targets": targets,
            "cases": [
                {
                    "id": "registry-validation",
                    "benchmark": "mmlu-pro",
                    "messages": [
                        {"role": "user", "content": "Registry validation only"}
                    ],
                    "answer": "A",
                }
            ],
        }
    )
    client.store.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = client.store / "targets.json"
    temporary = client.store / ("targets-" + str(os.getpid()) + ".tmp")
    with temporary.open("x") as file:
        file.write(json.dumps(validated["targets"], indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(destination)
    output({"targets": validated["targets"], "registered": len(validated["targets"])})
