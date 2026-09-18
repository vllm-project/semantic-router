"""Validate complete fixed-budget experiment matrices without executing them."""

import hashlib
from datetime import date

from .validation import (
    fields,
    indexed,
    number,
    require,
    sequence,
    sha256,
    string,
    strings,
    version,
)

MIN_BUDGETS = 2


def _dataset(dataset):
    fields(dataset, "id revision evidence_kind items calibration_ids", "dataset")
    string(dataset["id"], "dataset.id")
    string(dataset["revision"], "dataset.revision")
    require(dataset["evidence_kind"] in ("synthetic", "benchmark"), "evidence_kind")
    items = indexed(dataset["items"], "dataset.items")
    for item in items.values():
        fields(item, "id prompt prompt_sha256", "item")
        string(item["prompt"], "item.prompt")
        sha256(item["prompt_sha256"], "item.prompt_sha256")
        # Hash the exact UTF-8 prompt bytes, not its JSON representation.
        require(
            hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()
            == item["prompt_sha256"],
            "prompt_sha256 mismatch",
        )
    strings(dataset["calibration_ids"], "calibration_ids", nonempty=False)
    require(
        not set(items).intersection(dataset["calibration_ids"]), "calibration overlap"
    )


def _model(model):
    fields(model, "id provider model revision sampling pricing", "model")
    for key in ("provider", "model", "revision"):
        string(model[key], "model." + key)
    sampling = model["sampling"]
    fields(sampling, "temperature top_p", "sampling")
    number(sampling["temperature"], "temperature")
    number(sampling["top_p"], "top_p")
    require(0 < sampling["top_p"] <= 1, "top_p must be in (0, 1]")
    price = model["pricing"]
    fields(price, "as_of currency input_per_million output_per_million", "pricing")
    string(price["as_of"], "pricing.as_of")
    try:
        date.fromisoformat(price["as_of"])
    except ValueError:
        require(False, "pricing.as_of must be an ISO date")
    require(price["currency"] == "USD", "pricing currency must be USD")
    for key in ("input_per_million", "output_per_million"):
        number(price[key], "pricing." + key, nullable=True)


def _arm(arm, models, calibration_ids):
    fields(arm, "id algorithm model_ids parameters", "arm")
    strings(arm["model_ids"], "arm.model_ids")
    require(set(arm["model_ids"]) <= set(models), "unknown arm model reference")
    algorithm, params = arm["algorithm"], arm["parameters"]
    if algorithm == "direct":
        fields(params, "", "direct.parameters")
        require(len(arm["model_ids"]) == 1, "direct requires one model")
    elif algorithm == "confidence":
        fields(params, "threshold calibration_revision", "confidence.parameters")
        number(params["threshold"], "threshold")
        require(params["threshold"] <= 1, "threshold must be <= 1")
        string(params["calibration_revision"], "calibration_revision")
        require(bool(calibration_ids), "confidence requires calibration_ids")
    elif algorithm == "remom":
        fields(params, "breadth", "remom.parameters")
        sequence(params["breadth"], "breadth")
        for width in params["breadth"]:
            number(width, "breadth", minimum=1, integer=True)
    elif algorithm == "fusion":
        fields(
            params,
            "panel_model_ids judge_model_id synthesis_model_id",
            "fusion.parameters",
        )
        strings(params["panel_model_ids"], "panel_model_ids")
        for key in ("judge_model_id", "synthesis_model_id"):
            string(params[key], key)
        refs = params["panel_model_ids"] + [
            params["judge_model_id"],
            params["synthesis_model_id"],
        ]
        require(
            set(refs) == set(arm["model_ids"]), "fusion model_ids must match all stages"
        )
    else:
        require(False, "unknown algorithm")


def validate_config(config):
    fields(
        config, "schema_version id dataset models arms budgets scorer seeds", "config"
    )
    version(config["schema_version"])
    string(config["id"], "config.id")
    _dataset(config["dataset"])
    models = indexed(config["models"], "models")
    for model in models.values():
        _model(model)
    arms = indexed(config["arms"], "arms")
    for arm in arms.values():
        _arm(arm, models, config["dataset"]["calibration_ids"])
    require(
        {arm["algorithm"] for arm in arms.values()}
        == {"direct", "confidence", "remom", "fusion"},
        "matrix requires direct, confidence, remom and fusion",
    )
    budgets = indexed(config["budgets"], "budgets")
    require(len(budgets) >= MIN_BUDGETS, "matrix requires at least two budgets")
    limits = set()
    for budget in budgets.values():
        fields(budget, "id max_calls max_total_tokens exhaustion_policy", "budget")
        for key in ("max_calls", "max_total_tokens"):
            number(budget[key], key, minimum=1, integer=True)
        require(budget["exhaustion_policy"] == "stop_and_record", "exhaustion_policy")
        limits.add((budget["max_calls"], budget["max_total_tokens"]))
    require(len(limits) >= MIN_BUDGETS, "budget limits must be distinct")
    scorer = config["scorer"]
    fields(scorer, "id revision protocol_sha256", "scorer")
    string(scorer["id"], "scorer.id")
    string(scorer["revision"], "scorer.revision")
    sha256(scorer["protocol_sha256"], "scorer.protocol_sha256")
    sequence(config["seeds"], "seeds")
    for seed in config["seeds"]:
        number(seed, "seed", integer=True)
    require(len(set(config["seeds"])) == len(config["seeds"]), "duplicate seeds")
    return config
