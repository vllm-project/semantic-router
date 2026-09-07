"""Evaluation plane subcommands registered under ``vllm-sr eval``."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import click

from cli.evaluation.benchmark_normalization import normalize_benchmark_suite
from cli.evaluation.benchmark_normalization_registry import (
    get_benchmark_normalizers,
)
from cli.evaluation.benchmark_registry import get_benchmark_registry
from cli.evaluation.benchmark_sources import verify_benchmark_source
from cli.evaluation.builtin_executors import DEFAULT_EXECUTOR_REGISTRY
from cli.evaluation.canonical import json_value
from cli.evaluation.catalog import get_catalog
from cli.evaluation.compare import compare_runs
from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.executor_contracts import BUILTIN_NORMALIZED_SUITE_EXECUTORS
from cli.evaluation.orchestrator import load_manifest, run_evaluation, validate_manifest
from cli.evaluation.store import LocalArtifactStore
from cli.evaluation.suite_catalog import NormalizedSuiteCatalog
from cli.evaluation.suite_install_contract import BenchmarkSuiteInstallRequest
from cli.evaluation.suite_store import NormalizedSuiteStore
from cli.evaluation.worker_report import WorkerReportDraft

DEFAULT_STORE = Path(".vllm-sr/evaluation-store")
DEFAULT_SUITE_STORE = Path(".vllm-sr/evaluation-suites")


def _suite_catalog(store: NormalizedSuiteStore) -> NormalizedSuiteCatalog:
    return NormalizedSuiteCatalog(
        store,
        DEFAULT_EXECUTOR_REGISTRY,
        BUILTIN_NORMALIZED_SUITE_EXECUTORS,
    )


def _dump(value: object) -> str:
    return json.dumps(
        json_value(value, exclude_none=False),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _user_errors(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except click.ClickException:
            raise
        except (OSError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper


def _load_suite_install_request(path: Path) -> BenchmarkSuiteInstallRequest:
    with path.open("rb") as handle:
        payload = json.load(handle)
    return BenchmarkSuiteInstallRequest.model_validate(payload)


@click.command("catalog")
@click.option(
    "--suite-store",
    "suite_store_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_SUITE_STORE,
    show_default=True,
)
@_user_errors
def catalog_command(suite_store_path: Path) -> None:
    """Print the versioned evaluation suites, tracks, and targets."""

    installed = _suite_catalog(NormalizedSuiteStore(suite_store_path)).list()
    click.echo(_dump(get_catalog(installed_suites=installed)))


@click.command("benchmarks")
def benchmarks_command() -> None:
    """Print all exact-pinned external benchmark adapter descriptors."""

    click.echo(_dump(get_benchmark_registry()))


@click.command("normalizers")
@click.option("--runnable-only", is_flag=True, default=False)
def normalizers_command(runnable_only: bool) -> None:
    """Print the closed native-export contracts and explicit blockers."""

    descriptors = get_benchmark_normalizers()
    if runnable_only:
        descriptors = tuple(item for item in descriptors if item.executable)
    click.echo(
        _dump(
            {
                "schema_version": "benchmark-normalizer.v1",
                "normalizers": descriptors,
            }
        )
    )


@click.command("verify-source")
@click.option("--adapter", "adapter_id", required=True)
@click.option(
    "--source-root",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
)
@_user_errors
def verify_source_command(adapter_id: str, source_root: Path) -> None:
    """Verify an ignored external source checkout against its exact pin."""

    receipt = verify_benchmark_source(adapter_id, source_root)
    click.echo(_dump(receipt))
    if not receipt.verified:
        raise click.exceptions.Exit(2)


@click.command("suite-normalize")
@click.option("--adapter", "adapter_id", required=True)
@click.option("--suite-id", required=True)
@click.option("--suite-name")
@click.option(
    "--source-root",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False, readable=True),
    help="Ignored directory containing the exact-pinned benchmark checkout(s).",
)
@click.option(
    "--export-root",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False, readable=True),
    help=(
        "Directory containing the adapter's frozen native export shape. Its origin "
        "is not attested by this command."
    ),
)
@click.option(
    "--output",
    "output_root",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="New immutable directory for request.json and the normalized bundle.",
)
@_user_errors
def suite_normalize_command(
    adapter_id: str,
    suite_id: str,
    suite_name: str | None,
    source_root: Path,
    export_root: Path,
    output_root: Path,
) -> None:
    """Parse one native export into exploratory E0 replay evidence."""

    result = normalize_benchmark_suite(
        adapter_id=adapter_id,
        source_root=source_root,
        export_root=export_root,
        output_root=output_root,
        suite_id=suite_id,
        suite_name=suite_name,
    )
    click.echo(
        _dump(
            {
                "schema_version": result.request.schema_version,
                "adapter_id": result.request.adapter_id,
                "suite_id": result.request.id,
                "case_count": result.request.case_count,
                "track_ids": result.request.track_ids,
                "request": str(result.request_path),
                "bundle": str(result.bundle_path),
            }
        )
    )


@click.command("suite-install")
@click.option(
    "--request",
    "request_path",
    required=True,
    type=click.Path(
        path_type=Path,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
)
@click.option(
    "--bundle",
    "bundle_path",
    required=True,
    type=click.Path(
        path_type=Path,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
)
@click.option(
    "--source-root",
    required=True,
    type=click.Path(
        path_type=Path,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    help="Ignored directory containing the exact-pinned benchmark checkout(s).",
)
@click.option(
    "--export-root",
    "native_export_root",
    type=click.Path(
        path_type=Path,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    help=(
        "Frozen export re-parsed to verify deterministic normalization; this does "
        "not attest upstream benchmark execution."
    ),
)
@click.option(
    "--suite-store",
    "suite_store_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_SUITE_STORE,
    show_default=True,
)
@_user_errors
def suite_install_command(
    request_path: Path,
    bundle_path: Path,
    source_root: Path,
    native_export_root: Path | None,
    suite_store_path: Path,
) -> None:
    """Install a normalized suite as exploratory E0 evidence."""

    request = _load_suite_install_request(request_path)
    manifest = NormalizedSuiteStore(suite_store_path).install(
        request,
        bundle_path,
        source_root=source_root,
        native_export_root=native_export_root,
    )
    click.echo(_dump(manifest))


@click.command("benchmark-install")
@click.option(
    "--pack",
    "pack_root",
    required=True,
    type=click.Path(
        path_type=Path,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    help="Clean exact-revision checkout containing benchmark.yaml and bundle/.",
)
@click.option(
    "--suite-store",
    "suite_store_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_SUITE_STORE,
    show_default=True,
)
@_user_errors
def benchmark_install_command(pack_root: Path, suite_store_path: Path) -> None:
    """Install a data-only Benchmark Pack for replay or supported live tracks."""

    manifest = NormalizedSuiteStore(suite_store_path).install_pack(pack_root)
    click.echo(_dump(manifest))


@click.command("suite-list")
@click.option(
    "--suite-store",
    "suite_store_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_SUITE_STORE,
    show_default=True,
)
@_user_errors
def suite_list_command(suite_store_path: Path) -> None:
    """List browser-safe suite metadata without private artifact references."""

    suites = _suite_catalog(NormalizedSuiteStore(suite_store_path)).list()
    click.echo(_dump({"schema_version": SCHEMA_VERSION, "suites": suites}))


@click.command("suite-show")
@click.argument("suite_id")
@click.option(
    "--suite-store",
    "suite_store_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_SUITE_STORE,
    show_default=True,
)
@_user_errors
def suite_show_command(suite_id: str, suite_store_path: Path) -> None:
    """Print the immutable operator manifest for one installed suite."""

    manifest = NormalizedSuiteStore(suite_store_path).get_suite_manifest(suite_id)
    click.echo(_dump(manifest))


@click.command("validate")
@click.option(
    "--manifest", "manifest_path", required=True, type=click.Path(path_type=Path)
)
@click.option(
    "--suite-store",
    "suite_store_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_SUITE_STORE,
    show_default=True,
)
@_user_errors
def validate_command(manifest_path: Path, suite_store_path: Path) -> None:
    """Validate a fixed evaluation manifest without executing it."""

    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, NormalizedSuiteStore(suite_store_path))
    click.echo(
        _dump(
            {
                "schema_version": SCHEMA_VERSION,
                "valid": True,
                "run_id": manifest.run_id,
                "track_ids": manifest.track_ids,
            }
        )
    )


@click.command("run")
@click.option(
    "--manifest", "manifest_path", required=True, type=click.Path(path_type=Path)
)
@click.option(
    "--store",
    "store_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_STORE,
    show_default=True,
)
@click.option(
    "--suite-store",
    "suite_store_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=DEFAULT_SUITE_STORE,
    show_default=True,
)
@_user_errors
def run_command(manifest_path: Path, store_path: Path, suite_store_path: Path) -> None:
    """Execute a manifest and print its finalized worker report draft."""

    manifest = load_manifest(manifest_path)
    report = run_evaluation(
        manifest,
        LocalArtifactStore(store_path),
        suite_store=NormalizedSuiteStore(suite_store_path),
    )
    click.echo(_dump(report))


@click.command("report")
@click.argument("run_id")
@click.option(
    "--store",
    "store_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_STORE,
    show_default=True,
)
@_user_errors
def report_command(run_id: str, store_path: Path) -> None:
    """Read the local worker report draft."""

    store = LocalArtifactStore(store_path)
    click.echo(
        _dump(
            WorkerReportDraft.model_validate(store.read_run_json(run_id, "report.json"))
        )
    )


@click.command("compare")
@click.option("--baseline", "baseline_run_id", required=True)
@click.option("--candidate", "candidate_run_id", required=True)
@click.option(
    "--store",
    "store_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_STORE,
    show_default=True,
)
@_user_errors
def compare_command(
    baseline_run_id: str, candidate_run_id: str, store_path: Path
) -> None:
    """Compare two local immutable drafts without rerunning either workload."""

    store = LocalArtifactStore(store_path)
    comparison = compare_runs(store, baseline_run_id, candidate_run_id)
    click.echo(_dump(comparison))


@click.command("gate")
@click.argument("run_id")
@click.option(
    "--store",
    "store_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_STORE,
    show_default=True,
)
@click.option("--allow-unavailable", is_flag=True, default=False)
@_user_errors
def gate_command(run_id: str, store_path: Path, allow_unavailable: bool) -> None:
    """Emit standalone gate evidence and fail CI on blocking draft verdicts."""

    store = LocalArtifactStore(store_path)
    report = WorkerReportDraft.model_validate(
        store.read_run_json(run_id, "report.json")
    )
    click.echo(
        _dump(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "verdict": report.summary.verdict,
                "gates": report.gates,
            }
        )
    )
    if report.summary.verdict == "fail" or (
        report.summary.verdict == "unavailable" and not allow_unavailable
    ):
        raise click.exceptions.Exit(2)


EVALUATION_COMMANDS = (
    catalog_command,
    benchmarks_command,
    normalizers_command,
    verify_source_command,
    suite_normalize_command,
    suite_install_command,
    benchmark_install_command,
    suite_list_command,
    suite_show_command,
    validate_command,
    run_command,
    report_command,
    compare_command,
    gate_command,
)
