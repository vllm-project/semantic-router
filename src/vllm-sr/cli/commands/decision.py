"""Launch and safely manage standalone Decision model runtimes."""

from __future__ import annotations

import click

from cli.consts import (
    HEALTH_CHECK_TIMEOUT,
    IMAGE_PULL_POLICY_ALWAYS,
    IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    IMAGE_PULL_POLICY_NEVER,
    SUPPORTED_CONTAINER_RUNTIMES,
)
from cli.decision_runtime.catalog import (
    SUPPORTED_DECISION_BACKENDS,
    DecisionCatalogError,
    default_catalog_resolver,
)
from cli.decision_runtime.container import DecisionContainerError
from cli.decision_runtime.lifecycle import (
    DecisionLaunchReceipt,
    DecisionLifecycleError,
    DecisionServeOptions,
    run_decision_runtime,
)
from cli.decision_runtime.management import (
    DecisionDiscoveryStatus,
    DecisionManagedStatus,
    DecisionManagementError,
    DecisionOrphanStatus,
    forget_decision_instance,
    list_decision_instances,
    status_decision_instance,
    stop_decision_instance,
)
from cli.decision_runtime.registry import DecisionRegistryError

DECISION_HELP = """Serve and manage standalone Decision models.

Launching requires the integrated Decision catalog. Recovery commands remain
available without it. ``vllm-sr decision serve MODEL`` starts one model as a
standalone HTTP service. Detached instances remain registered for the
ownership-checked list, status, stop, and forget commands.
Current source builds have no default CPU or ROCm image; build one locally and
pass its image ID with ``--image`` and ``--image-pull-policy never``.

List and status report authoritative registry lifecycle separately from observed
container state. List does not probe service readiness; status probes the strict
``/ready`` contract for a registered running instance. Native MLX launch support
remains explicit integration work and is not emulated by the container driver.

\b
Examples:
  vllm-sr decision serve llm-semantic-router/Decision-1.0-Kai-0.6B \\
    --backend cpu --image "$DECISION_IMAGE_ID" --image-pull-policy never \\
    --instance-name kai-demo --detach
  vllm-sr decision list
  vllm-sr decision status kai-demo
  vllm-sr decision stop kai-demo
"""

_LIFECYCLE_ERRORS = (
    DecisionCatalogError,
    DecisionContainerError,
    DecisionLifecycleError,
    DecisionManagementError,
    DecisionRegistryError,
    ValueError,
)


@click.group("decision", help=DECISION_HELP)
def decision() -> None:
    """Launch and safely manage standalone Decision model runtimes."""


@decision.command("serve")
@click.argument("model", required=True)
@click.option(
    "--revision",
    help="Exact model revision; the catalog revision is used when omitted.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host IP used to publish the SystemOne endpoint.",
)
@click.option(
    "--port",
    type=click.IntRange(min=1, max=65535),
    default=8000,
    show_default=True,
    help="Host port used to publish the SystemOne endpoint.",
)
@click.option(
    "--backend",
    type=click.Choice(SUPPORTED_DECISION_BACKENDS, case_sensitive=False),
    default="auto",
    show_default=True,
    help="Decision inference backend.",
)
@click.option(
    "--dtype",
    help="Runtime dtype override; the selected backend profile decides when omitted.",
)
@click.option(
    "--max-batch",
    type=click.IntRange(min=1),
    help="Maximum physical inference batch override.",
)
@click.option(
    "--max-concurrency",
    type=click.IntRange(min=1),
    help="Maximum concurrent request override.",
)
@click.option(
    "--max-queue",
    type=click.IntRange(min=0),
    help="Maximum queued request override.",
)
@click.option(
    "--experimental-qwen-rocm-graph-b8",
    is_flag=True,
    help="Opt in to short-shape B8 ROCm backbone graphs for Decision Sol only.",
)
@click.option(
    "--cpu-threads",
    type=click.IntRange(min=1, max=256),
    help="CPU-only Torch/BLAS threads; default is min(8, container CPU allowance).",
)
@click.option(
    "--gpu-device",
    help="ROCm-only GPU index visible to this instance (0-9999); default sees all GPUs.",
)
@click.option(
    "--instance-name",
    help="Stable lowercase name for this managed Decision runtime instance.",
)
@click.option(
    "--image",
    help=(
        "Digest-qualified image, or exact local Docker sha256 image ID with "
        "--image-pull-policy never."
    ),
)
@click.option(
    "--image-pull-policy",
    type=click.Choice(
        (
            IMAGE_PULL_POLICY_ALWAYS,
            IMAGE_PULL_POLICY_IF_NOT_PRESENT,
            IMAGE_PULL_POLICY_NEVER,
        ),
        case_sensitive=False,
    ),
    default=IMAGE_PULL_POLICY_IF_NOT_PRESENT,
    show_default=True,
    help="Container image pull policy.",
)
@click.option(
    "--runtime",
    type=click.Choice(SUPPORTED_CONTAINER_RUNTIMES, case_sensitive=False),
    help="Docker-compatible container runtime.",
)
@click.option(
    "--startup-timeout",
    type=click.IntRange(min=1),
    default=HEALTH_CHECK_TIMEOUT,
    show_default=True,
    help="Seconds allowed for model loading and readiness.",
)
@click.option(
    "--detach",
    is_flag=True,
    help="Leave the managed runtime running after readiness succeeds.",
)
@click.option(
    "--restart-policy",
    type=click.Choice(("no", "unless-stopped"), case_sensitive=False),
    default="no",
    show_default=True,
    help="Docker restart policy; unless-stopped requires --detach.",
)
def serve(
    model: str,
    revision: str | None,
    host: str,
    port: int,
    backend: str,
    dtype: str | None,
    max_batch: int | None,
    max_concurrency: int | None,
    max_queue: int | None,
    experimental_qwen_rocm_graph_b8: bool,
    cpu_threads: int | None,
    gpu_device: str | None,
    instance_name: str | None,
    image: str | None,
    image_pull_policy: str,
    runtime: str | None,
    startup_timeout: int,
    detach: bool,
    restart_policy: str,
) -> None:
    """Launch one exact MODEL as a standalone SystemOne service."""

    options = DecisionServeOptions(
        model=model,
        revision=revision,
        host=host,
        port=port,
        backend=backend.lower(),
        dtype=dtype,
        max_batch=max_batch,
        max_concurrency=max_concurrency,
        max_queue=max_queue,
        experimental_qwen_rocm_graph_b8=experimental_qwen_rocm_graph_b8,
        cpu_threads=cpu_threads,
        gpu_device=gpu_device,
        instance_name=instance_name,
        image=image,
        image_pull_policy=image_pull_policy.lower(),
        runtime=runtime.lower() if runtime is not None else None,
        startup_timeout=startup_timeout,
        detach=detach,
        restart_policy=restart_policy.lower(),
    )
    try:
        resolver = default_catalog_resolver()
        run_decision_runtime(
            options,
            resolver=resolver,
            on_ready=_show_ready_receipt,
        )
    except _LIFECYCLE_ERRORS as error:
        raise click.ClickException(str(error)) from error


@decision.command("list")
def list_instances() -> None:
    """List registry lifecycle, runtime state, and cross-runtime orphans."""

    try:
        statuses = list_decision_instances()
    except _LIFECYCLE_ERRORS as error:
        raise click.ClickException(str(error)) from error
    if not statuses:
        click.echo("No managed Decision runtime instances.")
        return
    for managed_status in statuses:
        if isinstance(managed_status, DecisionDiscoveryStatus):
            _show_discovery_status(managed_status)
        elif isinstance(managed_status, DecisionOrphanStatus):
            _show_orphan_status(managed_status)
        else:
            _show_managed_status(managed_status)


@decision.command("status")
@click.argument("instance_name", required=True)
def status(instance_name: str) -> None:
    """Inspect lifecycle, runtime state, and strict readiness for INSTANCE_NAME."""

    try:
        _show_managed_status(status_decision_instance(instance_name))
    except _LIFECYCLE_ERRORS as error:
        raise click.ClickException(str(error)) from error


@decision.command("stop")
@click.argument("instance_name", required=True)
def stop(instance_name: str) -> None:
    """Stop one owned runtime or clear a proven container-missing record."""

    try:
        receipt = stop_decision_instance(instance_name)
    except _LIFECYCLE_ERRORS as error:
        raise click.ClickException(str(error)) from error
    if receipt.action == "stopped":
        click.echo(f"Stopped Decision runtime instance {receipt.instance_name}.")
    else:
        click.echo(f"Removed stale registry record {receipt.instance_name}.")
    _show_registry_durability(receipt.registry_durable)


@decision.command("forget")
@click.argument("instance_name", required=True)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "Forget a starting reservation after independently confirming its launch "
        "process is no longer active; no container is changed."
    ),
)
def forget(instance_name: str, force: bool) -> None:
    """Forget registry evidence without stopping any container."""

    try:
        receipt = forget_decision_instance(instance_name, force=force)
    except _LIFECYCLE_ERRORS as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"Forgot Decision runtime registry record {receipt.instance_name}; "
        "no container was changed."
    )
    _show_registry_durability(receipt.registry_durable)


def _show_ready_receipt(receipt: DecisionLaunchReceipt) -> None:
    click.echo("Decision runtime ready")
    click.echo(f"  Instance: {receipt.instance_name}")
    click.echo(f"  Model: {receipt.model}@{receipt.revision}")
    click.echo(f"  Endpoint: {receipt.endpoint}")
    click.echo(f"  Backend: {receipt.backend} ({receipt.dtype})")
    click.echo(f"  Artifact: {receipt.artifact_digest}")
    click.echo(f"  Mode: {'detached' if receipt.detached else 'foreground'}")
    click.echo(f"  Restart policy: {receipt.restart_policy}")


def _show_managed_status(managed_status: DecisionManagedStatus) -> None:
    record = managed_status.record
    ownership = "verified" if managed_status.ownership_verified else "unverified"
    click.echo(
        f"{record.instance_name}\tregistry={record.state}\t"
        f"runtime={managed_status.runtime_state}\t"
        f"readiness={managed_status.readiness_state}\townership={ownership}\t"
        f"restart={managed_status.restart_policy}\t"
        f"{record.model}\t{record.endpoint}"
    )
    _show_registry_durability(managed_status.registry_durable)


def _show_orphan_status(orphan_status: DecisionOrphanStatus) -> None:
    candidate = orphan_status.candidate
    display_name = candidate.instance_name or candidate.container_name
    label_state = (
        "label-owned-unregistered"
        if candidate.label_contract_valid
        else "invalid-managed-labels"
    )
    click.echo(
        f"{display_name}\tregistry=unregistered\truntime={candidate.state}\t"
        f"readiness=not-probed\tengine={candidate.runtime}\tlabels={label_state}\t-\t-"
    )
    click.echo(
        "Warning: this container is not present in the registry; it was not "
        "adopted or changed.",
        err=True,
    )


def _show_discovery_status(discovery_status: DecisionDiscoveryStatus) -> None:
    runtime = discovery_status.runtime or "container runtime"
    click.echo(
        f"Warning: {runtime} managed-label discovery is unavailable; registry "
        "entries above remain authoritative, but unregistered containers could "
        "not be enumerated.",
        err=True,
    )


def _show_registry_durability(durable: bool) -> None:
    if not durable:
        click.echo(
            "Warning: the registry update is visible, but crash durability could "
            "not be confirmed.",
            err=True,
        )
