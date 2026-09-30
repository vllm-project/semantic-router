"""EXPERIMENTAL sr-bench-nano commands: frozen questions, graders and run protocol."""

from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

import click

from cli.sr_bench import nano
from cli.sr_bench.contracts import load_document, plan, planned_cells

GRADER_ID = "simpleqa-grader"


def _fail(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (OSError, ValueError, KeyError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapped


def _echo(value):
    click.echo(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _pairs(values, label, parse):
    result = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"{label} must be KEY=VALUE")
        result[key] = parse(value)
    return result


def _sandbox_image(explicit):
    if explicit:
        return explicit
    from cli.sr_bench.setup import home  # noqa: PLC0415

    receipt = home() / "sandbox" / "manifest.json"
    return (
        json.loads(receipt.read_text())["sandbox_image"] if receipt.is_file() else None
    )


@click.group("nano")
def nano_group():
    """EXPERIMENTAL sr-bench-nano: frozen five-benchmark ids for your own models."""


@nano_group.command("show")
@_fail
def show_command():
    """Verify the installed frozen id list and print its sha256 and split sizes."""
    document = nano.frozen_ids()
    _echo(
        {
            "version": document["version"],
            "sha256": document["sha256"],
            "seed": document["seed"],
            "weights": nano.WEIGHTS,
            "benchmarks": {
                name: {
                    "revision": entry["source"]["revision"],
                    "population": entry["population"]["count"],
                    "splits": {
                        profile: {
                            "count": split["count"],
                            "ids_sha256": split["ids_sha256"],
                        }
                        for profile, split in entry["splits"].items()
                    },
                }
                for name, entry in document["benchmarks"].items()
            },
        }
    )


@nano_group.command("prepare")
@click.option(
    "--split",
    "profile",
    type=click.Choice(sorted(nano.PROFILES)),
    default="nano",
    show_default=True,
    help="nano-holdout is a frozen holdout; run it only when explicitly requested.",
)
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Offline pinned files as DIR/<benchmark>/<file>; default downloads from the Hub.",
)
@click.pass_obj
@_fail
def prepare_command(client, profile, source_dir):
    """Download pinned sources, verify every frozen hash, and store one dataset."""
    _echo(
        nano_prepare().prepare(
            profile,
            client.store,
            source_dir,
            lambda name: click.echo(f"verifying {name}", err=True),
        )
    )


def nano_prepare():
    from cli.sr_bench import nano_prepare as module  # noqa: PLC0415

    return module


@nano_group.command("manifest")
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Dataset manifest JSON printed by 'nano prepare'.",
)
@click.option(
    "--targets",
    "targets_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON/YAML list (or {targets: [...]}) of your own OpenAI-compatible targets.",
)
@click.option("--output", "destination", type=click.Path(path_type=Path), required=True)
@click.option("--name", default="sr-bench-nano")
@click.option("--concurrency", type=click.IntRange(1, 32), default=4, show_default=True)
@click.option(
    "--sample",
    type=click.IntRange(1),
    help="Smoke test: run only the first N frozen tasks per benchmark (no nano score).",
)
@click.option("--grader-base-url", envvar="SR_BENCH_NANO_GRADER_BASE_URL")
@click.option("--grader-model", envvar="SR_BENCH_NANO_GRADER_MODEL")
@click.option("--grader-api-key-env", envvar="SR_BENCH_NANO_GRADER_API_KEY_ENV")
@click.option(
    "--grader-header",
    multiple=True,
    envvar="SR_BENCH_NANO_GRADER_HEADERS",
    help="HEADER=ENV_VAR; the header value is read from ENV_VAR at call time.",
)
@click.option(
    "--grader-param",
    multiple=True,
    envvar="SR_BENCH_NANO_GRADER_PARAMS",
    help='KEY=JSON request parameter, e.g. chat_template_kwargs={"thinking":false}.',
)
@click.option(
    "--grader-no-stream",
    is_flag=True,
    envvar="SR_BENCH_NANO_GRADER_NO_STREAM",
    help="Call the grader without streaming.",
)
@click.option(
    "--lcb-sandbox-image",
    envvar="SR_BENCH_NANO_LCB_SANDBOX_IMAGE",
    help="Digest-pinned image from 'benchmark setup --build-sandbox' (read from its receipt by default).",
)
@_fail
def manifest_command(
    dataset_path,
    targets_path,
    destination,
    name,
    concurrency,
    sample,
    grader_base_url,
    grader_model,
    grader_api_key_env,
    grader_header,
    grader_param,
    grader_no_stream,
    lcb_sandbox_image,
):
    """Write and validate a nano run manifest for your targets and grader."""
    dataset = json.loads(dataset_path.read_text())
    if dataset.get("profile") not in nano.PROFILES:
        raise ValueError("Dataset was not prepared by 'vllm-sr benchmark nano prepare'")
    document = load_document(targets_path)
    targets = document.get("targets") if isinstance(document, dict) else document
    manifest = {
        "version": "sr-bench-1.0",
        "name": name,
        "mode": "live",
        "profile": dataset["profile"],
        "seed": nano.SEED,
        "dataset": {"path": dataset["path"], "sha256": dataset["sha256"]},
        "targets": targets,
        "cost_policy": "capability_only",
        "output_policy": "uncapped",
        "limits": {"concurrency": concurrency},
        "benchmark_options": {},
    }
    if "simpleqa-verified" in dataset["benchmarks"]:
        if not grader_base_url or not grader_model:
            raise ValueError(
                "SimpleQA grader is not configured and nano has no default: set "
                "--grader-base-url and --grader-model (or SR_BENCH_NANO_GRADER_BASE_URL "
                "and SR_BENCH_NANO_GRADER_MODEL), plus --grader-api-key-env if needed"
            )
        grader = {
            "id": GRADER_ID,
            "kind": "single",
            "base_url": grader_base_url,
            "model": grader_model,
            "request_params": {
                **_pairs(grader_param, "--grader-param", json.loads),
                "temperature": 0,
            },
        }
        if grader_api_key_env:
            grader["api_key_env"] = grader_api_key_env
        if grader_header:
            grader["header_env"] = _pairs(grader_header, "--grader-header", str)
        if grader_no_stream:
            grader["stream"] = False
        manifest["auxiliary_targets"] = {GRADER_ID: grader}
        manifest["benchmark_options"]["simpleqa-verified"] = {
            "judge": GRADER_ID,
            "grader_version": nano.SIMPLEQA_GRADER,
        }
    if "livecodebench" in dataset["benchmarks"]:
        from cli.sr_bench.setup import PACKAGES  # noqa: PLC0415

        image = _sandbox_image(lcb_sandbox_image)
        if not image:
            raise ValueError(
                "LiveCodeBench needs a sandbox: run 'vllm-sr benchmark setup "
                "--benchmark livecodebench --install --build-sandbox' or pass --lcb-sandbox-image"
            )
        manifest["benchmark_options"]["livecodebench"] = {
            "source_revision": PACKAGES["livecodebench"]["revision"],
            "sandbox_image": image,
        }
    if sample:
        manifest["execution_cells"] = [
            {"case_id": task["id"], "target_id": target["id"]}
            for benchmark in sorted(dataset["benchmarks"])
            for task in nano.split_tasks(benchmark, dataset["profile"])[:sample]
            for target in targets
        ]
    frozen = plan(manifest)
    with destination.open("x") as file:
        file.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _echo(
        {
            "manifest": str(destination),
            "plan_sha256": frozen["plan_sha256"],
            "cells": len(planned_cells(frozen)),
            "ids_sha256": nano.frozen_ids()["sha256"],
        }
    )


@nano_group.command("freeze")
@click.option("--output", "destination", type=click.Path(path_type=Path), required=True)
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Offline pinned files as DIR/<benchmark>/<file>.",
)
@click.pass_obj
@_fail
def freeze_command(client, destination, source_dir):
    """Maintainer: regenerate the frozen id list (ids and hashes only, no text)."""
    document = nano_prepare().freeze(client.store, source_dir)
    with destination.open("x") as file:
        file.write(_render(document))
    _echo({"output": str(destination), "sha256": document["sha256"]})


def _render(document):
    """Readable, diffable layout: one task object per line."""
    placeholder = {}
    shell = json.loads(json.dumps(document))
    for name, entry in shell["benchmarks"].items():
        for profile, split in entry["splits"].items():
            key = f"@@{name}/{profile}@@"
            placeholder[json.dumps(key)] = split["tasks"]
            split["tasks"] = key
    text = json.dumps(shell, indent=2, ensure_ascii=False)
    for key, tasks in placeholder.items():
        rows = ",\n".join(
            "        " + json.dumps(task, ensure_ascii=False, sort_keys=True)
            for task in tasks
        )
        text = text.replace(key, "[\n" + rows + "\n      ]")
    return text + "\n"
