"""Deterministic experiment identity and expansion (no model or scorer calls)."""

import copy
from itertools import product
from pathlib import Path

from . import SCHEMA_VERSION
from .config import validate_config
from .validation import digest, string, strings


def build_plan(config, code_revision, command):
    validate_config(config)
    string(code_revision, "code_revision")
    # Commands are argument vectors; repeated arguments are valid.
    if not isinstance(command, list) or not command:
        raise ValueError("command must be a nonempty argument vector")
    for argument in command:
        string(argument, "command argument")
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
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "config_sha256": digest(config),
        **identity,
        "command": list(command),
        "status": "planned",
        "matrix": matrix,
    }
