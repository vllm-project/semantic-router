import importlib.util
from pathlib import Path


def load_plot_module():
    module_path = Path(__file__).with_name("plot_session_routing_figures.py")
    spec = importlib.util.spec_from_file_location(
        "plot_session_routing_figures", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_read_csv_parses_numbers_and_missing_cells(tmp_path):
    plot = load_plot_module()
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "policy,switches,quality_delta,drift_switches\n"
        "single-turn,9709,0.0,\n"
        "acr-full,2011,-0.0453,574\n"
    )

    rows = plot.read_csv(csv_path)

    assert rows == [
        {
            "policy": "single-turn",
            "switches": 9709,
            "quality_delta": 0.0,
            "drift_switches": "",
        },
        {
            "policy": "acr-full",
            "switches": 2011,
            "quality_delta": -0.0453,
            "drift_switches": 574,
        },
    ]


def test_ordered_keeps_publication_policy_order():
    plot = load_plot_module()
    rows = [
        {"policy": "acr-full"},
        {"policy": "single-turn"},
        {"policy": "acr-initial"},
    ]

    ordered = plot.ordered(rows, "policy", plot.POLICY_ORDER)

    assert [row["policy"] for row in ordered] == [
        "single-turn",
        "acr-initial",
        "acr-full",
    ]


def test_agent_task_readiness_metrics_count_stale_summary():
    plot = load_plot_module()
    summary = {
        "requests": 96,
        "task_count": 6,
        "task_instances": 18,
        "missing_router_header_counts": {
            "x-vsr-selected-model": 0,
            "x-vsr-selected-decision": 0,
            "x-vsr-replay-id": 0,
            "x-vsr-selected-confidence": 96,
            "x-vsr-context-token-count": 96,
        },
        "invalid_router_header_counts": {},
    }

    rows = plot.agent_task_readiness_metrics(
        summary,
        list(plot.DEFAULT_AGENT_TASK_HEADERS),
        plot.DEFAULT_MIN_AGENT_TASK_REQUESTS,
        plot.DEFAULT_MIN_AGENT_TASK_COUNT,
        plot.DEFAULT_MIN_AGENT_TASK_INSTANCES,
    )

    assert rows == [
        {
            "label": "requests",
            "actual": 96,
            "target": 345,
            "text": "96/345",
        },
        {
            "label": "task types",
            "actual": 6,
            "target": 20,
            "text": "6/20",
        },
        {
            "label": "scored instances",
            "actual": 18,
            "target": 60,
            "text": "18/60",
        },
        {
            "label": "diagnostics",
            "actual": 3,
            "target": 6,
            "text": "3/6 present",
        },
    ]


def test_agent_task_readiness_metrics_treat_missing_header_map_as_no_evidence():
    plot = load_plot_module()
    summary = {"requests": 345, "task_count": 20, "task_instances": 60}

    rows = plot.agent_task_readiness_metrics(
        summary,
        list(plot.DEFAULT_AGENT_TASK_HEADERS),
        plot.DEFAULT_MIN_AGENT_TASK_REQUESTS,
        plot.DEFAULT_MIN_AGENT_TASK_COUNT,
        plot.DEFAULT_MIN_AGENT_TASK_INSTANCES,
    )

    assert rows[-1]["label"] == "diagnostics"
    assert rows[-1]["actual"] == 0
