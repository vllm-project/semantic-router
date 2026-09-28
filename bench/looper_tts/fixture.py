"""Hand-authored synthetic evidence cases; never executes or scores an algorithm."""

from . import SCHEMA_VERSION
from .validation import digest, require

FIXTURE_CASES = (
    ("success", False),
    ("error", False),
    ("budget_exhausted", False),
    ("success", True),
)


def fixture_records(plan):
    """Populate every cell with deterministic contract examples, not measurements."""
    require(
        plan["config"]["dataset"]["evidence_kind"] == "synthetic",
        "fixture requires synthetic data",
    )
    arms = {arm["id"]: arm for arm in plan["config"]["arms"]}
    calls, results = [], []
    for index, cell in enumerate(plan["matrix"]):
        for item_id in cell["item_ids"]:
            coordinates = {
                "experiment_id": plan["experiment_id"],
                "cell_id": cell["id"],
                "item_id": item_id,
            }
            status, cached = FIXTURE_CASES[index % len(FIXTURE_CASES)]
            call_id = digest({**coordinates, "fixture": "call"})
            usage = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
            if status == "success":
                usage = {"prompt_tokens": 8, "completion_tokens": 0, "total_tokens": 8}
            call = {
                "id": call_id,
                **coordinates,
                "stage": "generate",
                "model_id": arms[cell["arm_id"]]["model_ids"][0],
                "attempt": 1,
                "status": (
                    "cached"
                    if cached
                    else ("error" if status == "error" else "success")
                ),
                "usage": usage,
                "latency_ms": None if cached else 1.0,
                "raw_output_path": "fixture://synthetic-answer",
                "error": "synthetic provider failure" if status == "error" else None,
                "cache_id": digest("synthetic-panel") if cached else None,
            }
            calls.append(call)
            results.append(
                {
                    "id": digest({**coordinates, "fixture": "result"}),
                    **coordinates,
                    "status": status,
                    "final_answer": "synthetic answer" if status == "success" else None,
                    "score": 1.0 if status == "success" else None,
                    "scorer_id": plan["config"]["scorer"]["id"],
                    "call_ids": [call_id],
                    "candidate_scores": [],
                    "panel_sha256": digest("synthetic-panel") if cached else None,
                    "budget_status": (
                        "exhausted" if status == "budget_exhausted" else "within"
                    ),
                    "error": None if status == "success" else "synthetic " + status,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": plan["experiment_id"],
        "evidence_kind": "synthetic",
        "calls": calls,
        "results": results,
    }
