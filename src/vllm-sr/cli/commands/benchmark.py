"""The sr-bench 1.0 CLI shares one durable service with the Dashboard."""

from __future__ import annotations

import json
import os
import time
from functools import wraps
from pathlib import Path
from urllib.parse import urlencode

import click
from click.core import ParameterSource

from cli.commands.benchmark_experiments import experiment
from cli.commands.benchmark_preparations import (
    preparation_path,
    prepare_remote_dataset,
    prepare_remote_datasets,
)
from cli.runtime_stack import resolve_runtime_stack
from cli.sr_bench import setup, sources
from cli.sr_bench.client import Client
from cli.sr_bench.contracts import catalog, load_document, plan, planned_cells
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
@guarded
def benchmark(ctx, url, store, no_autostart):
    """Prepare, run, inspect, and compare sr-bench 1.0 evaluations."""
    explicit_url, explicit_store = url is not None, store is not None
    if not explicit_url:
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
    managed_service = not explicit_url and token_file.is_file()
    if url is None:
        url = (
            f"http://127.0.0.1:{stack.sr_bench_port}"
            if managed_service
            else DEFAULT_URL
        )
    client = Client(
        url,
        store,
        not no_autostart and not managed_service and not explicit_url,
        verify_store=explicit_store or not explicit_url,
    )
    if managed_service and "Authorization" not in client.headers:
        if token_file.is_symlink() or token_file.stat().st_mode & 0o077:
            raise click.ClickException(
                "Managed service token must be a private regular file"
            )
        client.headers["Authorization"] = "Bearer " + token_file.read_text().strip()
    ctx.obj = client
    ctx.meta["benchmark_explicit_url"] = explicit_url


@benchmark.command("catalog")
def catalog_command():
    """Show the nine benchmark adapters and evaluation profiles."""
    output(catalog())


@benchmark.command("setup")
@click.option("--benchmark", "benchmark_id", default="all")
@click.option(
    "--install",
    is_flag=True,
    help="Install pinned optional harnesses and task sources; makes no model requests.",
)
@click.option(
    "--build-sandbox",
    is_flag=True,
    help="Build a local offline grading image and record its content digest.",
)
@guarded
def setup_command(benchmark_id, install, build_sandbox):
    """Inspect prerequisites or explicitly install optional benchmark harnesses."""
    output(setup.setup(benchmark_id, install, build_sandbox))


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
@click.option(
    "--benchmark",
    "benchmark_ids",
    required=True,
    multiple=True,
    help="Benchmark ID; repeat to prepare a shared collection.",
)
@click.option(
    "--profile", type=click.Choice(["smoke", "quick", "standard"]), default="quick"
)
@click.option("--source-path", type=click.Path(path_type=Path))
@click.option(
    "--local",
    is_flag=True,
    help="Prepare on this host; required for source files and history options.",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Return the shared service preparation job immediately.",
)
@click.option("--revision")
@click.option("--seed", default=20260918, type=int)
@click.option(
    "--source-partition",
    help="Frozen upstream partition for native task identity; never an evaluation split.",
)
@click.option("--exclusion-snapshot", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--evaluation-role",
    type=click.Choice(["holdout", "retest"]),
    help="Explicit family role; retest is never selected automatically.",
)
@click.option(
    "--limit",
    type=int,
    help="Custom case cap; cannot be represented as an upstream full benchmark.",
)
@click.pass_obj
@guarded
def dataset_prepare(
    client,
    benchmark_ids,
    profile,
    source_path,
    revision,
    seed,
    limit,
    source_partition,
    exclusion_snapshot,
    evaluation_role,
    local,
    no_wait,
):
    """Download and freeze a dataset through the shared service by default."""
    local_options = (
        source_path,
        revision,
        source_partition,
        exclusion_snapshot,
        evaluation_role,
    )
    if len(benchmark_ids) > 1:
        if (
            local
            or limit is not None
            or any(value is not None for value in local_options)
        ):
            raise ValueError(
                "Multiple benchmarks require shared collection preparation; "
                "--local, --limit, source, and history options are not supported"
            )
        if (
            click.get_current_context().get_parameter_source("seed")
            == ParameterSource.DEFAULT
        ):
            seed = None
        output(prepare_remote_datasets(client, benchmark_ids, profile, seed, no_wait))
        return
    benchmark_id = benchmark_ids[0]
    if not local:
        if any(value is not None for value in local_options):
            raise ValueError(
                "Source files, revisions, partitions, and history options require --local; "
                "local files are not uploaded to the service"
            )
        output(
            prepare_remote_dataset(client, benchmark_id, profile, seed, limit, no_wait)
        )
        return
    if click.get_current_context().meta.get("benchmark_explicit_url"):
        raise ValueError(
            "--local cannot be combined with --url or SR_BENCH_URL; unset the service URL to prepare locally"
        )
    if no_wait:
        raise ValueError(
            "--no-wait requires shared service preparation; remove --local"
        )
    kwargs = {
        "benchmark": benchmark_id,
        "profile": profile,
        "store": client.store,
        "source_path": str(source_path) if source_path else None,
        "revision": revision,
        "seed": seed,
    }
    if limit is not None:
        kwargs["limit"] = limit
    for key, value in {
        "source_partition": source_partition,
        "exclusion_snapshot": exclusion_snapshot,
        "evaluation_role": evaluation_role,
    }.items():
        if value is not None:
            kwargs[key] = value
    output(sources.prepare_dataset(**kwargs))


@dataset.command("options")
@click.pass_obj
@guarded
def dataset_options(client):
    """List downloadable sources, profiles, access notes, and dependencies."""
    output(client.request("GET", "/dataset-preparations/options"))


@dataset.command("preparations")
@click.argument("preparation_id", required=False)
@click.pass_obj
@guarded
def dataset_preparations(client, preparation_id):
    """List shared download jobs or inspect one preparation by ID."""
    path = (
        preparation_path(preparation_id) if preparation_id else "/dataset-preparations"
    )
    output(client.request("GET", path))


@dataset.command("exclusions")
@click.option(
    "--dataset",
    "dataset_ids",
    multiple=True,
    help="Named prepared dataset whose entire membership is reserved.",
)
@click.option(
    "--run",
    "run_ids",
    multiple=True,
    help="Named frozen run whose entire planned membership is reserved.",
)
@click.option("--output", "destination", required=True, type=click.Path(path_type=Path))
@click.pass_obj
@guarded
def dataset_exclusions(client, dataset_ids, run_ids, destination):
    """Freeze finite named history without inspecting outcomes or running models."""
    from cli.sr_bench.history_exclusions import compile_snapshot  # noqa: PLC0415
    from cli.sr_bench.history_snapshot import save_snapshot  # noqa: PLC0415

    snapshot = compile_snapshot(client.store, dataset_ids=dataset_ids, run_ids=run_ids)
    output(save_snapshot(snapshot, destination))


@dataset.command("combine")
@click.argument(
    "manifests", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True
)
@click.pass_obj
@guarded
def dataset_combine(client, manifests):
    """Create a reusable multi-benchmark dataset from prepared manifests."""
    output(
        sources.combine_datasets(
            [load_document(path) for path in manifests], client.store
        )
    )


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
            "total": len(planned_cells(frozen)),
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
@click.option(
    "--active", is_flag=True, help="Read only in-progress calls; requires --calls."
)
@click.option("--events", is_flag=True)
@click.option(
    "--after",
    type=click.IntRange(min=0),
    default=0,
    help="Evidence cursor from the previous page.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1, max=500),
    default=100,
    help="Calls/results per page.",
)
@click.option(
    "--call-id", help="Read one full saved call including prompt and final response."
)
@click.pass_obj
@guarded
def show_command(client, run_id, results, calls, active, events, after, limit, call_id):
    """Read a run, bounded evidence page, or one complete saved call."""
    if sum((results, calls, events, bool(call_id))) > 1:
        raise ValueError("Choose one evidence view")
    if active and not calls:
        raise ValueError("--active requires --calls")
    path = "/runs/" + run_id
    if call_id:
        path += "/calls/" + call_id
    elif results or calls:
        path += ("/results" if results else "/calls") + f"?after={after}&limit={limit}"
        if active:
            path += "&active=true"
    elif events:
        path += f"/events?after={after}"
    output(client.request("GET", path))


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


@benchmark.command("replay-options")
@click.argument("baseline", required=False)
@click.option("--after", help="Opaque cursor from the previous eligible options page.")
@click.option("--limit", type=click.IntRange(1, 25), default=10, show_default=True)
@click.pass_obj
@guarded
def replay_options_command(client, baseline, after, limit):
    """List eligible baselines, or compatible previews for BASELINE; no model calls."""
    _saved_run_options(client, "replay", baseline, after, limit)


@benchmark.command("comparison-options")
@click.argument("baseline", required=False)
@click.option("--after", help="Opaque cursor from the previous eligible options page.")
@click.option("--limit", type=click.IntRange(1, 25), default=10, show_default=True)
@click.pass_obj
@guarded
def comparison_options_command(client, baseline, after, limit):
    """List eligible baselines, or comparable live runs for BASELINE; no model calls."""
    _saved_run_options(client, "comparison", baseline, after, limit)


def _saved_run_options(client, kind, baseline, after, limit):
    query = {"limit": limit}
    if baseline is not None:
        query["baseline_run_id"] = baseline
    if after is not None:
        query["after"] = after
    output(client.request("GET", f"/{kind}-options?" + urlencode(query)))


@benchmark.command("replay")
@click.option(
    "--baseline", required=True, help="Completed single-model answer matrix run ID."
)
@click.option(
    "--preview", required=True, help="Completed deterministic routing preview run ID."
)
@click.option("--idempotency-key")
@click.pass_obj
@guarded
def replay_command(client, baseline, preview, idempotency_key):
    """Estimate eligible static routes from saved answers without inference."""
    output(
        client.request(
            "POST",
            "/replays",
            {
                "baseline_run_id": baseline,
                "preview_run_id": preview,
                "idempotency_key": idempotency_key,
            },
        )
    )


@benchmark.command("regrade")
@click.argument("run_id")
@click.option("--output", "destination", type=click.Path(path_type=Path), required=True)
@click.pass_obj
@guarded
def regrade_command(client, run_id, destination):
    """Regrade saved MCQ/grid final outputs without mutating original evidence."""
    artifact = client.request("POST", "/runs/" + run_id + "/regrade", {})
    with destination.open("x") as file:
        file.write(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    output(
        {
            "path": str(destination),
            "changed_count": artifact["changed_count"],
            "model_requests": 0,
        }
    )


@benchmark.command("reconcile-usage")
@click.argument("run_id")
@click.pass_obj
@guarded
def reconcile_usage_command(client, run_id):
    """Append an offline accounting correction from saved streams; no inference."""
    output(client.request("POST", "/runs/" + run_id + "/reconcile-usage", {}))


@benchmark.command("export")
@click.argument("run_id")
@click.option("--output", "destination", type=click.Path(path_type=Path), required=True)
@click.pass_obj
@guarded
def export_command(client, run_id, destination):
    """Export a dev response matrix for training; holdout export is rejected."""
    artifact = client.request("POST", "/runs/" + run_id + "/export", {})
    with destination.open("x") as file:
        file.write(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    output(
        {
            "path": str(destination),
            "case_count": len(artifact["cases"]),
            "split": artifact["split"],
        }
    )


@benchmark.command("recover-plan")
@click.argument("run_id")
@click.option(
    "--mode", type=click.Choice(["undispatched", "failed"]), default="undispatched"
)
@click.option("--output", "destination", type=click.Path(path_type=Path))
@click.pass_obj
@guarded
def recover_plan_command(client, run_id, mode, destination):
    """Inspect eligible continuation cells without making model requests."""
    proposed = client.request(
        "POST", "/runs/" + run_id + "/recover-plan", {"mode": mode}
    )
    if destination:
        with destination.open("x") as handle:
            handle.write(json.dumps(proposed, indent=2) + "\n")
    output(proposed)


@benchmark.command("recover")
@click.argument("run_id")
@click.option(
    "--plan", "plan_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--idempotency-key", required=True)
@click.option(
    "--acknowledge-new-attempt",
    is_flag=True,
    help="Authorize new paid attempts for the exact reviewed failed cells.",
)
@click.pass_obj
@guarded
def recover_command(
    client, run_id, plan_path, idempotency_key, acknowledge_new_attempt
):
    """Create a separate attempt from a reviewed recovery plan; never auto-retry."""
    proposed = load_document(plan_path)
    if proposed.get("parent_run_id") != run_id:
        raise click.ClickException("Recovery plan belongs to another parent run")
    output(
        client.request(
            "POST",
            "/runs/" + run_id + "/recover",
            {
                "mode": proposed["mode"],
                "plan_sha256": proposed["plan_sha256"],
                "cells": proposed.get("selected_cells", proposed["eligible_cells"]),
                "idempotency_key": idempotency_key,
                "acknowledge_new_attempt": acknowledge_new_attempt,
            },
        )
    )


@benchmark.command("candidate-plan")
@click.argument("baseline_run_id")
@click.option(
    "--target",
    "target_ids",
    multiple=True,
    required=True,
    help="Registered MoM target; may be repeated.",
)
@click.option(
    "--mode", type=click.Choice(["live", "preview"]), default="live", show_default=True
)
@click.option("--name")
@click.option("--experiment", "experiment_id")
@click.option("--hypothesis", default="")
@click.pass_obj
@guarded
def candidate_plan_command(
    client, baseline_run_id, target_ids, mode, name, experiment_id, hypothesis
):
    """Reuse a terminal baseline's frozen protocol without repeating its requests.

    Failed, cancelled and interrupted full-plan baselines may supply the same
    questions and settings. This does not qualify their measurements for Compare.
    """
    body = {"target_ids": list(target_ids), "mode": mode}
    if name:
        body["name"] = name
    if experiment_id:
        baseline = client.request("GET", "/runs/" + baseline_run_id)
        body["experiment"] = {
            "id": experiment_id,
            "role": (
                "preview"
                if mode == "preview"
                else (
                    "validation"
                    if baseline["manifest"]["profile"] == "standard"
                    else "candidate"
                )
            ),
            "hypothesis": hypothesis,
        }
    output(client.request("POST", "/runs/" + baseline_run_id + "/candidate-plan", body))


benchmark.add_command(experiment)
