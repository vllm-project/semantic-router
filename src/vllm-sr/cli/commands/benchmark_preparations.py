"""CLI submission and observation of shared dataset preparation jobs."""

from __future__ import annotations

import re
import time

import click


def preparation_path(preparation_id):
    if not isinstance(preparation_id, str) or not re.fullmatch(
        r"prep-[0-9a-f]{32}", preparation_id
    ):
        raise ValueError(
            "Preparation ID must be prep- followed by 32 lowercase hexadecimal characters"
        )
    return "/dataset-preparations/" + preparation_id


def prepare_remote_dataset(client, benchmark, profile, seed, limit, no_wait):
    request = {"benchmark": benchmark, "profile": profile, "seed": seed}
    if limit is not None:
        request["limit"] = limit
    return _submit_preparation(client, request, no_wait)


def prepare_remote_datasets(client, benchmarks, profile, seed, no_wait):
    request = {"benchmarks": list(benchmarks), "profile": profile}
    if seed is not None:
        request["seed"] = seed
    return _submit_preparation(client, request, no_wait)


def _submit_preparation(client, request, no_wait):
    preparation = client.request("POST", "/dataset-preparations", request)[
        "preparation"
    ]
    path = preparation_path(preparation["id"])
    if no_wait:
        return {"preparation": preparation}
    click.echo(
        f"Preparation {preparation['id']} submitted; Ctrl-C leaves the service working. "
        "Inspect it with benchmark dataset preparations.",
        err=True,
    )
    while preparation["status"] in {"queued", "running"}:
        time.sleep(0.5)
        preparation = client.request("GET", path)["preparation"]
    if preparation["status"] != "completed":
        raise ValueError(
            f"Preparation {preparation['id']} {preparation['status']}: "
            + (preparation.get("error") or "inspect the preparation for details")
        )
    return preparation["dataset"]
