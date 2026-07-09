import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_PATH = (
    REPO_ROOT / "src/vllm-sr/cli/templates/llm-router-dashboard.serve.json"
)


def test_grafana_dashboard_contains_router_observability_panels():
    dashboard = json.loads(DASHBOARD_PATH.read_text())
    exprs = "\n".join(
        target.get("expr", "")
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
    )

    for metric in [
        "llm_model_routing_latency_seconds_bucket",
        "llm_model_selection_duration_seconds_bucket",
        "llm_model_selection_total",
        "llm_model_inflight_requests",
        "llm_model_latency_p95_windowed_seconds",
        "llm_model_error_rate_windowed",
        "llm_model_queue_depth_estimated",
        "llm_model_utilization_percentage",
        "llm_session_model_transitions_total",
        "llm_cache_warmth_estimate_bucket",
        "llm_request_errors_total",
    ]:
        assert metric in exprs
