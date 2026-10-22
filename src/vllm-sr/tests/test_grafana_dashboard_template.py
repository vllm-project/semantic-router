import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_PATH = REPO_ROOT / "src/vllm-sr/cli/templates/llm-router-dashboard.serve.json"
METRICS_ROOT = REPO_ROOT / "src/semantic-router/pkg/observability/metrics"


def dashboard():
    return json.loads(DASHBOARD_PATH.read_text())


def all_panels(panels=None):
    for current in dashboard()["panels"] if panels is None else panels:
        yield current
        yield from all_panels(current.get("panels", []))


def data_panels():
    return [panel for panel in all_panels() if panel["type"] != "row"]


def panel(title):
    return next(p for p in data_panels() if p["title"] == title)


def metric_contracts():
    """Read production metric names/types/labels, independently of panel helpers."""
    contracts = {}
    for path in METRICS_ROOT.parent.rglob("*.go"):
        if path.name.endswith("_test.go"):
            continue
        source = path.read_text()
        for match in re.finditer(
            r"prometheus\.(Counter|Gauge|Histogram)Opts\{((?:[^{}]|\{[^{}]*\})*)\}\s*,\s*\[\]string\{([^}]*)\}",
            source,
            re.S,
        ):
            name = re.search(r'Name:\s*"([^"]+)"', match[2])
            if name:
                contracts[name[1]] = (match[1], set(re.findall(r'"([^"]+)"', match[3])))
        # The in-flight collector derives a gauge from the authoritative registry.
        for match in re.finditer(
            r'prometheus\.NewDesc\(\s*"([^"]+)"\s*,\s*"[^"]*"\s*,\s*\[\]string\{([^}]+)\}',
            source,
            re.S,
        ):
            contracts[match[1]] = ("Gauge", set(re.findall(r'"([^"]+)"', match[2])))
    return contracts


def test_generated_dashboard_document_is_reproducible_and_layout_is_coherent():
    source = DASHBOARD_PATH.with_name("generate_dashboard.py")
    spec = importlib.util.spec_from_file_location("dashboard_generator", source)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    actual = dashboard()
    assert actual == generator.generate_dashboard_document()
    assert actual["uid"] == "vllm-semantic-router"
    rows = [p["title"] for p in actual["panels"] if p["type"] == "row"]
    assert rows == [
        "Inference overview",
        "Recipe workflow",
        "Model responses and usage",
        "Plugins and response cache",
        "Accounting and MoM execution",
        "Telemetry health",
    ]
    assert len({p["id"] for p in all_panels()}) == len(list(all_panels()))
    assert_panel_layout(actual["panels"])
    assert all(p.get("description") for p in data_panels())


def assert_panel_layout(panels):
    for index, first in enumerate(panels):
        a = first["gridPos"]
        assert 0 <= a["x"] < 24 and a["x"] + a["w"] <= 24
        for second in panels[index + 1 :]:
            b = second["gridPos"]
            assert not (
                a["x"] < b["x"] + b["w"]
                and b["x"] < a["x"] + a["w"]
                and a["y"] < b["y"] + b["h"]
                and b["y"] < a["y"] + a["h"]
            ), (first["title"], second["title"])
        assert_panel_layout(first.get("panels", []))


@pytest.mark.parametrize("current", data_panels(), ids=lambda item: item["title"])
def test_queries_reference_production_metric_labels_and_histogram_units(current):
    contracts = metric_contracts()
    for target in current["targets"]:
        expr = target["expr"]
        assert "or vector(0)" not in expr
        if expr.startswith("up{"):
            assert current["title"] == "Telemetry scrape health"
            continue
        references = re.findall(r'(llm_[a-z0-9_]+)\{((?:"[^"]*"|[^"}])+)\}', expr)
        assert references, current["title"]
        groups = set()
        for group in re.findall(r"by\s*\(([^)]+)\)", expr):
            groups.update(part.strip() for part in group.split(","))
        for name, selector in references:
            base = re.sub(r"_(bucket|sum|count)$", "", name)
            kind, labels = contracts[name if name in contracts else base]
            if name not in contracts:
                assert kind == "Histogram", name
            valid_labels = labels | {"job", "instance"}
            if name.endswith("_bucket"):
                valid_labels.add("le")
            selected = set(re.findall(r"([a-zA-Z_]\w*)\s*(?:=~|!~|!=|=)", selector))
            assert selected <= valid_labels, (name, selected - valid_labels)
            assert groups <= valid_labels, (name, groups - valid_labels)
            assert 'job="semantic-router"' in selector
            assert 'instance=~"${router_instance:regex}"' in selector
        if "rate(" in expr:
            assert "[$__rate_interval]" in expr
        if "histogram_quantile" in expr or "_seconds_sum" in expr:
            assert current["fieldConfig"]["defaults"]["unit"] == "s"
        # Do not turn full-range rolling increases into time-series 'counts'.
        if "increase(" in expr:
            assert current["type"] == "stat"
            assert target.get("instant") is True


def test_success_fraction_uses_only_terminal_inference_and_preserves_empty_population():
    expr = panel("Inference success rate")["targets"][0]["expr"]
    selectors = re.findall(r'llm_request_outcomes_total\{((?:"[^"]*"|[^"}])+)\}', expr)
    assert len(selectors) == 3  # success, zero anchored to total, positive denominator
    assert all('traffic_kind="inference"' in item for item in selectors)
    assert sum('outcome="success"' in item for item in selectors) == 1
    assert "llm_model_requests" not in expr and "llm_request_errors" not in expr
    assert "or (0 * sum(increase(llm_request_outcomes_total" in expr
    assert expr.endswith(" > 0)")
    assert (
        "No observed denominator remains No data"
        in panel("Inference success rate")["description"]
    )


def test_mean_durations_use_recorded_sums_and_positive_matching_counts():
    for title in [
        "Recipe stage duration",
        "Response duration by model",
        "First response observation by model",
        "Response duration per output token",
        "Plugin duration",
        "Cache operation duration",
    ]:
        expr = panel(title)["targets"][0]["expr"]
        assert "_sum{" in expr and "_count{" in expr
        assert expr.endswith(" > 0)")
        assert "histogram_quantile" not in expr
    assert (
        "not decode inter-token"
        in panel("Response duration per output token")["description"]
    )
    assert (
        "nonstreaming response headers"
        in panel("First response observation by model")["description"]
    )


def test_obsolete_and_unmeasured_contracts_cannot_reappear_as_dashboard_metrics():
    expressions = "\n".join(t["expr"] for p in data_panels() for t in p["targets"])
    for obsolete in [
        "_windowed",
        "llm_model_utilization_percentage",
        "llm_model_queue_depth_estimated",
        "llm_model_ttft_seconds",
        "llm_model_tpot_seconds",
        "llm_decision_confidence",
        "llm_cache_warmth_estimate",
        "llm_cache_plugin_hits_total",
        "llm_cache_plugin_misses_total",
        "llm_cache_operation_duration_seconds",
        "llm_model_requests_total",
    ]:
        assert obsolete not in expressions
    cache = panel("Cache lookup outcomes")
    assert 'operation=~"lookup_exact|lookup_semantic"' in cache["targets"][0]["expr"]
    assert "status" in cache["targets"][0]["expr"]
    assert "do not define a request-level hit percentage" in cache["description"]
    assert "_count{" in panel("Projection evaluations")["targets"][0]["expr"]
    assert "not a quality probability" in panel("Projection evaluations")["description"]


def test_accounting_is_optional_currency_scoped_and_never_claims_complete_cost():
    row = next(
        p for p in dashboard()["panels"] if p["title"] == "Accounting and MoM execution"
    )
    assert row["collapsed"] is True and len(row["panels"]) == 6
    for title in ["Known model cost rate", "Known Looper attempt cost rate"]:
        current = panel(title)
        expression = current["targets"][0]["expr"]
        assert "currency" in re.search(r"by \(([^)]+)\)", expression)[1]
        assert "{{currency}}" in current["targets"][0]["legendFormat"]
        assert "not invoices or hardware cost" in current["description"]
        assert "does not prove complete accounting" in current["description"]
        assert " or " not in expression
    for current in row["panels"]:
        if "Looper" in current["title"]:
            assert "currently covers Confidence" in current["description"]
            assert "absent series" in current["description"]
