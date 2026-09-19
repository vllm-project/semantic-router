"""Read-only replay eligibility shared by discovery and materialization."""

from collections import defaultdict

from cli.routing_preview import PROMPT_FIELDS, case_request_fields

from . import VERSION
from .accounting import effective_calls
from .contracts import canonical, digest

REPLAYABLE = {"mmlu-pro", "gpqa-diamond", "arc-agi-2", "hle", "simpleqa-verified"}


class ReplayEligibilityError(ValueError):
    """No replay was created because frozen evidence is incompatible."""

    def __init__(self, reasons):
        self.reasons = reasons
        super().__init__("; ".join(reason["message"] for reason in reasons))


def replay_summary(run):
    manifest = run["manifest"]
    return {
        "run_id": run["id"],
        "name": manifest.get("name", "sr-bench"),
        "profile": manifest.get("profile"),
        "case_count": len(manifest.get("cases", [])),
        "status": run["status"],
        "mode": manifest.get("mode"),
    }


class ReplayValidator:
    """Read one baseline matrix once per bounded page; never dispatch or write."""

    def __init__(self, store, baseline):
        self.baseline = baseline
        self.results = {
            (row["case_id"], row["target_id"]): row
            for row in store.results(baseline["id"])
        }
        self.calls = defaultdict(list)
        for call in effective_calls(store, baseline["id"]):
            if call["role"] == "subject":
                self.calls[(call["case_id"], call["target_id"])].append(call)
        self.store = store

    def validate(self, preview):
        reasons = {}

        def reject(code, message):
            reasons.setdefault(code, {"code": code, "message": message})

        bm, pm = self.baseline["manifest"], preview["manifest"]
        if self.baseline["status"] != "completed" or bm.get("mode") != "live":
            reject(
                "baseline_not_completed_live",
                "Baseline must be a completed live single-model matrix.",
            )
        if preview["status"] != "completed" or pm.get("mode") != "preview":
            reject("preview_not_completed", "Routing preview must be completed.")
        if bm.get("version") != VERSION or pm.get("version") != VERSION:
            reject(
                "version_mismatch",
                "Both runs must use the supported sr-bench protocol version.",
            )
        if any(m.get("execution_cells") or m.get("recovery") for m in (bm, pm)):
            reject(
                "partial_matrix",
                "Recovery subsets cannot stand in for a complete replay matrix.",
            )
        cases = []
        for manifest in (bm, pm):
            rows = manifest.get("cases", [])
            indexed = {c["id"]: c for c in rows}
            if not rows or len(indexed) != len(rows):
                reject(
                    "duplicate_case_ids",
                    "Frozen cases must have distinct IDs and a nonempty population.",
                )
            if manifest.get("case_sha256") != digest(rows):
                reject(
                    "case_receipt_mismatch",
                    "Saved case content differs from its frozen hash.",
                )
            cases.append(indexed)
        baseline_cases, preview_cases = cases
        if baseline_cases.keys() != preview_cases.keys():
            reject(
                "case_set_mismatch",
                "Replay requires the same case IDs and population in both runs.",
            )
        elif any(
            canonical(baseline_cases[key]) != canonical(value)
            for key, value in preview_cases.items()
        ):
            reject(
                "case_content_mismatch",
                "Cases with the same ID have different frozen content, answers, or metadata.",
            )
        if any(
            c["benchmark"] not in REPLAYABLE
            for c in list(baseline_cases.values()) + list(preview_cases.values())
        ):
            reject(
                "benchmark_not_replayable",
                "Agent and code harness trajectories cannot be replayed as single selections.",
            )
        for key in ("adapter_versions", "benchmark_weights"):
            if not bm.get(key) or bm.get(key) != pm.get(key):
                reject(
                    "grader_protocol_mismatch",
                    "Frozen adapter versions or benchmark weights differ or are unavailable.",
                )
        singles = [target for target in bm["targets"] if target["kind"] == "single"]
        by_model = {target["model"]: target for target in singles}
        if not singles or len(by_model) != len(singles):
            reject(
                "baseline_models_not_unique",
                "Baseline must contain unique single-model identities.",
            )
        preview_rows = {
            (row["case_id"], row["target_id"]): row
            for row in self.store.results(preview["id"])
        }
        materialized = []
        for target in pm["targets"]:
            for case in pm["cases"]:
                row = preview_rows.get((case["id"], target["id"]))
                if row is None or row["status"] != "completed":
                    reject(
                        "preview_rows_incomplete",
                        "Preview does not contain every completed case/target selection.",
                    )
                    continue
                routing = row.get("details", {}).get("routing", {})
                provenance = routing.get("selection_provenance") or {}
                if (
                    provenance.get("state_dependent")
                    or provenance.get("mode") == "read_only_snapshot"
                ):
                    reject(
                        "state_dependent_preview",
                        "State-dependent learning previews cannot be replayed as stateless single selections.",
                    )
                if routing.get("selection_status") != "selected" or routing.get(
                    "selection_method"
                ) not in {"static", "single", "route_action"}:
                    reject(
                        "route_not_deterministic",
                        "Replay requires a deterministic selected single-model route; dynamic or execution-required choices are excluded.",
                    )
                decision = routing.get("decision_result")
                if not isinstance(decision, dict) or decision.get("plugins"):
                    reject(
                        "plugins_not_replayable",
                        "Replay excludes routes with request or response plugins or missing decision evidence.",
                    )
                selected = by_model.get(routing.get("selected_model"))
                if selected is None:
                    reject(
                        "model_missing",
                        "Preview selected a model absent from the live single-model matrix.",
                    )
                    continue
                baseline_params = {
                    **bm["sampling"],
                    **selected.get("request_params", {}),
                }
                preview_params = {**pm["sampling"], **target.get("request_params", {})}
                if canonical(baseline_params) != canonical(preview_params):
                    reject(
                        "sampling_mismatch",
                        "Replay selected model has different frozen effective request parameters.",
                    )
                result = self.results.get((case["id"], selected["id"]))
                calls = self.calls.get((case["id"], selected["id"]), [])
                if (
                    result is None
                    or result["status"] != "completed"
                    or len(calls) != 1
                    or calls[0]["status"] != "completed"
                ):
                    reject(
                        "saved_generation_incomplete",
                        "Replay requires exactly one saved complete subject generation per selected case.",
                    )
                    continue
                if not isinstance(result.get("correct"), bool):
                    reject(
                        "saved_grade_missing",
                        "The saved generation has no completed correctness grade.",
                    )
                request = calls[0].get("request", {}).get("effective_body", {})
                expected_extras = case_request_fields(case)
                if request.get("messages") != case.get("messages") or any(
                    request.get(key) != expected_extras.get(key)
                    for key in PROMPT_FIELDS | {"metadata"}
                ):
                    reject(
                        "saved_prompt_mismatch",
                        "Saved generated prompt differs from the replay case.",
                    )
                if any(
                    request.get(key) != value for key, value in baseline_params.items()
                ):
                    reject(
                        "saved_request_parameters_mismatch",
                        "Saved generation parameters differ from the frozen selected-model profile.",
                    )
                if request.get("model") != selected["model"]:
                    reject(
                        "saved_model_mismatch",
                        "Saved request identity differs from the selected single-model target.",
                    )
                materialized.append((case, target, result, calls[0], selected["id"]))
        receipt = {
            "version": "sr-bench-replay-compatibility-v1",
            "case_matching": "stable_id_full_content",
            "case_count": len(preview_cases),
            "baseline_case_sha256": bm.get("case_sha256"),
            "preview_case_sha256": pm.get("case_sha256"),
            "baseline_plan_sha256": bm["plan_sha256"],
            "preview_plan_sha256": pm["plan_sha256"],
            "order_only_difference": not reasons
            and bm.get("case_sha256") != pm.get("case_sha256"),
            "case_content_sha256": digest(
                [preview_cases[key] for key in sorted(preview_cases)]
            ),
            "accounting_basis": "saved_baseline_calls_and_prices",
            "model_requests": 0,
        }
        return {
            "eligible": not reasons,
            "reasons": list(reasons.values()),
            "receipt": receipt,
            "materialized": materialized if not reasons else [],
        }
