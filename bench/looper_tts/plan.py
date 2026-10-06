"""Deterministic experiment identity and expansion (no model or scorer calls)."""

import copy
from itertools import product
from pathlib import Path

from . import SCHEMA_VERSION
from .config import validate_config
from .validation import (
    digest,
    fields,
    require,
    sequence,
    sha256,
    string,
    strings,
    version,
)


def _command(command):
    # Commands are argument vectors; repeated arguments are valid.
    sequence(command, "command")
    for argument in command:
        string(argument, "command argument")


def _matrix(config, experiment_id):
    matrix = []
    for arm, budget, seed in product(
        config["arms"], config["budgets"], config["seeds"]
    ):
        coordinates = {"arm_id": arm["id"], "budget_id": budget["id"], "seed": seed}
        matrix.append(
            {
                "id": digest({"experiment_id": experiment_id, **coordinates}),
                **coordinates,
                "algorithm": arm["algorithm"],
                "item_ids": [item["id"] for item in config["dataset"]["items"]],
            }
        )
    strings([cell["id"] for cell in matrix], "matrix ids")
    return matrix


def validate_plan(plan):
    """Check a saved manifest's internal integrity, including all derived fields."""
    fields(
        plan,
        "schema_version experiment_id config_sha256 config code_revision "
        "planner_sha256 command status matrix",
        "plan",
    )
    version(plan["schema_version"])
    config = validate_config(plan["config"])
    string(plan["code_revision"], "code_revision")
    _command(plan["command"])
    require(plan["status"] == "planned", "invalid plan status")
    for key in ("config_sha256", "planner_sha256", "experiment_id"):
        sha256(plan[key], key)
    require(plan["config_sha256"] == digest(config), "config_sha256 mismatch")
    identity = {key: plan[key] for key in ("config", "code_revision", "planner_sha256")}
    experiment_id = digest(identity)
    require(plan["experiment_id"] == experiment_id, "plan experiment identity mismatch")
    # Use the saved source digest: historical manifests need not match this checkout.
    require(
        digest(plan["matrix"]) == digest(_matrix(config, experiment_id)),
        "plan matrix mismatch",
    )
    return plan


def build_plan(config, code_revision, command):
    validate_config(config)
    string(code_revision, "code_revision")
    _command(command)
    source = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    identity = {
        "config": copy.deepcopy(config),
        "code_revision": code_revision,
        "planner_sha256": digest(source),
    }
    experiment_id = digest(identity)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "config_sha256": digest(config),
        **identity,
        "command": list(command),
        "status": "planned",
        "matrix": _matrix(config, experiment_id),
    }
