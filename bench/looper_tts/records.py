"""Validate normalized evidence and its links to a frozen experiment plan."""

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


def _usage(usage):
    fields(usage, "prompt_tokens completion_tokens total_tokens", "usage")
    for key, value in usage.items():
        number(value, "usage." + key, integer=True, nullable=True)
    # Preserve provider total: some backends account for additional token classes.


def _coordinates(record, cells, item_ids, experiment_id):
    for key in ("experiment_id", "cell_id", "item_id"):
        string(record[key], key)
    require(record["experiment_id"] == experiment_id, "experiment identity mismatch")
    require(record["cell_id"] in cells, "unknown matrix cell")
    require(record["item_id"] in item_ids, "unknown item")


def _call(call, cells, arms):
    fields(
        call,
        "id experiment_id cell_id item_id stage model_id attempt status usage "
        "latency_ms raw_output_path error cache_id",
        "call",
    )
    require(
        call["stage"] in ("generate", "verify", "select", "judge", "synthesize"),
        "call stage",
    )
    arm = arms[cells[call["cell_id"]]["arm_id"]]
    require(call["model_id"] in arm["model_ids"], "call model outside arm")
    number(call["attempt"], "attempt", minimum=1, integer=True)
    require(call["status"] in ("success", "error", "cached"), "call status")
    _usage(call["usage"])
    number(call["latency_ms"], "latency_ms", nullable=True)
    if call["status"] == "error":
        string(call["error"], "call.error")
    else:
        require(call["error"] is None, "non-error call must have null error")
        string(call["raw_output_path"], "raw_output_path")
    if call["raw_output_path"] is not None:
        string(call["raw_output_path"], "raw_output_path")
    if call["status"] == "cached":
        sha256(call["cache_id"], "cache_id")
        require(call["latency_ms"] is None, "cached call has no live latency")
    else:
        require(call["cache_id"] is None, "live call must have null cache_id")


def _result(result, calls, scorer_id):
    fields(
        result,
        "id experiment_id cell_id item_id status final_answer score scorer_id "
        "call_ids candidate_scores panel_sha256 budget_status error",
        "result",
    )
    require(
        result["status"] in ("success", "error", "budget_exhausted"), "result status"
    )
    require(result["budget_status"] in ("within", "exhausted"), "budget status")
    require(
        (result["status"] == "budget_exhausted")
        == (result["budget_status"] == "exhausted"),
        "inconsistent budget status",
    )
    require(result["scorer_id"] == scorer_id, "scorer mismatch")
    number(result["score"], "score", nullable=True)
    if result["score"] is not None:
        require(result["score"] <= 1, "normalized score must be <= 1")
    if result["status"] == "success":
        string(result["final_answer"], "final_answer")
        require(result["error"] is None, "successful result has an error")
    else:
        string(result["error"], "result.error")
        require(result["score"] is None, "failed result cannot have a score")
    if result["final_answer"] is not None:
        string(result["final_answer"], "final_answer")
    strings(result["call_ids"], "call_ids", nonempty=False)
    require(
        result["status"] != "success" or bool(result["call_ids"]),
        "success requires calls",
    )
    require(set(result["call_ids"]) <= set(calls), "unknown call reference")
    for call_id in result["call_ids"]:
        call = calls[call_id]
        require(
            all(
                call[key] == result[key]
                for key in ("experiment_id", "cell_id", "item_id")
            ),
            "cross-item or cross-cell call reference",
        )
    sequence(result["candidate_scores"], "candidate_scores", nonempty=False)
    candidate_ids = []
    for candidate in result["candidate_scores"]:
        fields(candidate, "call_id score", "candidate_score")
        require(candidate["call_id"] in result["call_ids"], "unknown candidate call")
        require(
            calls[candidate["call_id"]]["stage"] == "generate",
            "candidate must be generated",
        )
        number(candidate["score"], "candidate score", nullable=True)
        require(
            candidate["score"] is None or candidate["score"] <= 1, "candidate score > 1"
        )
        candidate_ids.append(candidate["call_id"])
    strings(candidate_ids, "candidate call ids", nonempty=False)
    if result["panel_sha256"] is not None:
        sha256(result["panel_sha256"], "panel_sha256")


def validate_records(bundle, plan):
    """Require one terminal result for every planned (cell, item) pair."""
    fields(
        bundle, "schema_version experiment_id evidence_kind calls results", "records"
    )
    version(bundle["schema_version"])
    config = plan["config"]
    require(
        bundle["experiment_id"] == plan["experiment_id"], "experiment identity mismatch"
    )
    require(
        bundle["evidence_kind"] == config["dataset"]["evidence_kind"],
        "evidence kind mismatch",
    )
    cells = indexed(plan["matrix"], "matrix")
    arms = indexed(config["arms"], "arms")
    items = indexed(config["dataset"]["items"], "items")
    sequence(bundle["calls"], "calls", nonempty=False)
    calls = indexed(bundle["calls"], "calls") if bundle["calls"] else {}
    results = indexed(bundle["results"], "results")
    for call in calls.values():
        require(isinstance(call, dict), "call must be an object")
        require(
            all(key in call for key in ("experiment_id", "cell_id", "item_id")),
            "missing call coordinates",
        )
        _coordinates(call, cells, items, plan["experiment_id"])
        _call(call, cells, arms)
    pairs, referenced = set(), []
    for result in results.values():
        require(
            all(key in result for key in ("experiment_id", "cell_id", "item_id")),
            "missing result coordinates",
        )
        _coordinates(result, cells, items, plan["experiment_id"])
        _result(result, calls, config["scorer"]["id"])
        pair = (result["cell_id"], result["item_id"])
        require(pair not in pairs, "duplicate result coordinates")
        pairs.add(pair)
        referenced.extend(result["call_ids"])
    require(
        pairs == {(cell, item) for cell in cells for item in items},
        "incomplete result matrix",
    )
    require(set(referenced) == set(calls), "orphan call records")
    require(
        len(referenced) == len(set(referenced)), "call referenced by multiple results"
    )
    return bundle
