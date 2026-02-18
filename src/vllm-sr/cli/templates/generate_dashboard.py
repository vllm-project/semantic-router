#!/usr/bin/env python3
"""
Generate Grafana Dashboard JSON for vLLM Semantic Router
"""

import json


def create_stat_panel(title, expr, unit="short", x=0, y=0, w=6, h=6, panel_id=1):
    """Create a stat panel"""
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "thresholds"},
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
                "unit": unit,
            }
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": panel_id,
        "options": {
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "auto",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto",
        },
        "pluginVersion": "11.5.1",
        "targets": [
            {
                "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
                "expr": expr,
                "refId": "A",
            }
        ],
        "title": title,
        "type": "stat",
    }


def create_timeseries_panel(
    title, targets, x=0, y=0, w=12, h=8, panel_id=1, unit="short"
):
    """Create a time series panel"""
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "line",
                    "fillOpacity": 10,
                    "gradientMode": "none",
                    "hideFrom": {"tooltip": False, "viz": False, "legend": False},
                    "insertNulls": False,
                    "lineInterpolation": "linear",
                    "lineWidth": 1,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": "never",
                    "spanNulls": False,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "off"},
                },
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
                "unit": unit,
            }
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": panel_id,
        "options": {
            "legend": {
                "calcs": [],
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi", "sort": "none"},
        },
        "pluginVersion": "11.5.1",
        "targets": targets,
        "title": title,
        "type": "timeseries",
    }


def create_row_panel(title, y=0, panel_id=100):
    """Create a row panel"""
    return {
        "collapsed": False,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "id": panel_id,
        "panels": [],
        "title": title,
        "type": "row",
    }


def create_target(expr, legend="", ref_id="A"):
    """Create a query target"""
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "expr": expr,
        "legendFormat": legend,
        "refId": ref_id,
    }


def create_bar_chart_panel(
    title, targets, x=0, y=0, w=24, h=8, panel_id=1, unit="short"
):
    """Create a bar chart panel"""
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "axisBorderShow": False,
                    "axisCenteredZero": False,
                    "axisColorMode": "text",
                    "axisLabel": "",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "bars",
                    "fillOpacity": 80,
                    "gradientMode": "none",
                    "hideFrom": {"tooltip": False, "viz": False, "legend": False},
                    "insertNulls": False,
                    "lineInterpolation": "linear",
                    "lineWidth": 1,
                    "pointSize": 5,
                    "scaleDistribution": {"type": "linear"},
                    "showPoints": "never",
                    "spanNulls": False,
                    "stacking": {"group": "A", "mode": "none"},
                    "thresholdsStyle": {"mode": "off"},
                },
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [{"color": "green", "value": None}],
                },
                "unit": unit,
            }
        },
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "id": panel_id,
        "options": {
            "legend": {
                "calcs": ["lastNotNull"],
                "displayMode": "table",
                "placement": "right",
                "showLegend": True,
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "pluginVersion": "11.5.1",
        "targets": targets,
        "title": title,
        "type": "timeseries",
    }


def generate_dashboard():
    """Generate the complete dashboard"""
    panels = []
    panel_id = 1
    y_pos = 0

    # ========== 1. Overall Request Metrics ==========
    panels.append(create_row_panel("Overall Request Metrics", y=y_pos, panel_id=100))
    y_pos += 1

    # Total Requests (increase over time range)
    panels.append(
        create_stat_panel(
            "Total Requests",
            "sum(increase(llm_model_requests_total[$__range]))",
            unit="short",
            x=0,
            y=y_pos,
            w=8,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1

    # Average QPS (over time range)
    panels.append(
        create_stat_panel(
            "Average QPS",
            "sum(rate(llm_model_requests_total[$__range]))",
            unit="reqps",
            x=8,
            y=y_pos,
            w=8,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1

    # Success Rate (over time range)
    panels.append(
        create_stat_panel(
            "Success Rate",
            "sum(increase(llm_model_requests_total[$__range])) / (sum(increase(llm_model_requests_total[$__range])) + (sum(increase(llm_request_errors_total[$__range])) or vector(0))) * 100",
            unit="percent",
            x=16,
            y=y_pos,
            w=8,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1
    y_pos += 6

    # Request Count Trend
    panels.append(
        create_timeseries_panel(
            "Request Count Trend",
            [
                create_target(
                    "sum(rate(llm_model_requests_total[5m]))", "Requests/sec", "A"
                )
            ],
            x=0,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="reqps",
        )
    )
    panel_id += 1

    # Request Latency Trend
    panels.append(
        create_timeseries_panel(
            "Request Latency (P50/P95/P99)",
            [
                create_target(
                    "histogram_quantile(0.50, sum(rate(llm_model_completion_latency_seconds_bucket[5m])) by (le))",
                    "P50",
                    "A",
                ),
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_model_completion_latency_seconds_bucket[5m])) by (le))",
                    "P95",
                    "B",
                ),
                create_target(
                    "histogram_quantile(0.99, sum(rate(llm_model_completion_latency_seconds_bucket[5m])) by (le))",
                    "P99",
                    "C",
                ),
            ],
            x=12,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="s",
        )
    )
    panel_id += 1
    y_pos += 8

    # ========== 2. Token Usage ==========
    panels.append(create_row_panel("LLM Token Usage", y=y_pos, panel_id=200))
    y_pos += 1

    # Total Tokens (increase over time range)
    panels.append(
        create_stat_panel(
            "Total Tokens",
            "sum(increase(llm_model_tokens_total[$__range]))",
            unit="short",
            x=0,
            y=y_pos,
            w=6,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1

    # Average Total Tokens/sec (over time range)
    panels.append(
        create_stat_panel(
            "Avg Tokens/sec",
            "sum(rate(llm_model_tokens_total[$__range]))",
            unit="tps",
            x=6,
            y=y_pos,
            w=6,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1

    # Average Prompt Tokens/sec (over time range)
    panels.append(
        create_stat_panel(
            "Avg Prompt Tokens/sec",
            "sum(rate(llm_model_prompt_tokens_total[$__range]))",
            unit="tps",
            x=12,
            y=y_pos,
            w=6,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1

    # Average Completion Tokens/sec (over time range)
    panels.append(
        create_stat_panel(
            "Avg Completion Tokens/sec",
            "sum(rate(llm_model_completion_tokens_total[$__range]))",
            unit="tps",
            x=18,
            y=y_pos,
            w=6,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1
    y_pos += 6

    # Token Usage Trend
    panels.append(
        create_timeseries_panel(
            "Token Usage Trend",
            [
                create_target("sum(rate(llm_model_tokens_total[5m]))", "Total", "A"),
                create_target(
                    "sum(rate(llm_model_prompt_tokens_total[5m]))", "Prompt", "B"
                ),
                create_target(
                    "sum(rate(llm_model_completion_tokens_total[5m]))",
                    "Completion",
                    "C",
                ),
            ],
            x=0,
            y=y_pos,
            w=24,
            h=8,
            panel_id=panel_id,
            unit="tps",
        )
    )
    panel_id += 1
    y_pos += 8

    # ========== 3. Signal Extraction ==========
    panels.append(create_row_panel("Signal Extraction", y=y_pos, panel_id=300))
    y_pos += 1

    # Signal Extraction Rate by Type
    panels.append(
        create_timeseries_panel(
            "Signal Extraction Rate by Type",
            [
                create_target(
                    "sum(rate(llm_signal_extraction_total[5m])) by (signal_type)",
                    "{{signal_type}}",
                    "A",
                )
            ],
            x=0,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="ops",
        )
    )
    panel_id += 1

    # Signal Match Rate by Type
    panels.append(
        create_timeseries_panel(
            "Signal Match Rate by Type",
            [
                create_target(
                    "sum(rate(llm_signal_match_total[5m])) by (signal_type)",
                    "{{signal_type}}",
                    "A",
                )
            ],
            x=12,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="ops",
        )
    )
    panel_id += 1
    y_pos += 8

    # Signal Extraction Latency P95
    panels.append(
        create_timeseries_panel(
            "Signal Extraction Latency (P95) by Type",
            [
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_signal_extraction_latency_seconds_bucket[5m])) by (le, signal_type))",
                    "{{signal_type}}",
                    "A",
                )
            ],
            x=0,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="s",
        )
    )
    panel_id += 1

    # Top Matched Signals
    panels.append(
        create_timeseries_panel(
            "Top 10 Matched Signals",
            [
                create_target(
                    "topk(10, sum(rate(llm_signal_match_total[5m])) by (signal_name))",
                    "{{signal_name}}",
                    "A",
                )
            ],
            x=12,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="ops",
        )
    )
    panel_id += 1
    y_pos += 8

    # ========== 4. Decision Matching ==========
    panels.append(create_row_panel("Decision Matching", y=y_pos, panel_id=400))
    y_pos += 1

    # Decision Evaluation Rate
    panels.append(
        create_stat_panel(
            "Decision Evaluation Rate",
            "rate(llm_decision_evaluation_total[5m])",
            unit="ops",
            x=0,
            y=y_pos,
            w=8,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1

    # Decision Match Rate
    panels.append(
        create_stat_panel(
            "Decision Match Rate",
            "sum(rate(llm_decision_match_total[5m]))",
            unit="ops",
            x=8,
            y=y_pos,
            w=8,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1

    # Decision Evaluation Latency P95
    panels.append(
        create_stat_panel(
            "Decision Evaluation Latency (P95)",
            "histogram_quantile(0.95, sum(rate(llm_decision_evaluation_latency_seconds_bucket[5m])) by (le))",
            unit="s",
            x=16,
            y=y_pos,
            w=8,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1
    y_pos += 6

    # Decision Match Trend by Decision
    panels.append(
        create_timeseries_panel(
            "Decision Match Trend by Decision",
            [
                create_target(
                    "sum(rate(llm_decision_match_total[5m])) by (decision_name)",
                    "{{decision_name}}",
                    "A",
                )
            ],
            x=0,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="ops",
        )
    )
    panel_id += 1

    # Decision Confidence (P50) by Decision
    panels.append(
        create_timeseries_panel(
            "Decision Confidence (P50) by Decision",
            [
                create_target(
                    "histogram_quantile(0.5, sum(rate(llm_decision_confidence_bucket[5m])) by (le, decision_name))",
                    "{{decision_name}}",
                    "A",
                )
            ],
            x=12,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="percentunit",
        )
    )
    panel_id += 1
    y_pos += 8

    # ========== 5. Model Distribution ==========
    panels.append(create_row_panel("Model Distribution", y=y_pos, panel_id=500))
    y_pos += 1

    # Model Request Count (Bar Chart - increase over time range)
    panels.append(
        create_bar_chart_panel(
            "Model Request Count",
            [
                create_target(
                    "sum(increase(llm_model_requests_total[$__range])) by (model)",
                    "{{model}}",
                    "A",
                )
            ],
            x=0,
            y=y_pos,
            w=24,
            h=8,
            panel_id=panel_id,
            unit="short",
        )
    )
    panel_id += 1
    y_pos += 8

    # Requests by Model
    panels.append(
        create_timeseries_panel(
            "Requests by Model",
            [
                create_target(
                    "sum(rate(llm_model_requests_total[5m])) by (model)",
                    "{{model}}",
                    "A",
                )
            ],
            x=0,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="reqps",
        )
    )
    panel_id += 1

    # Errors by Model
    panels.append(
        create_timeseries_panel(
            "Errors by Model",
            [
                create_target(
                    "sum(rate(llm_request_errors_total[5m])) by (model, reason)",
                    "{{model}}-{{reason}}",
                    "A",
                )
            ],
            x=12,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="ops",
        )
    )
    panel_id += 1
    y_pos += 8

    # TTFT by Model
    panels.append(
        create_timeseries_panel(
            "TTFT (Time to First Token) by Model - P95",
            [
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_model_ttft_seconds_bucket[5m])) by (le, model))",
                    "{{model}}",
                    "A",
                )
            ],
            x=0,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="s",
        )
    )
    panel_id += 1

    # TPOT by Model
    panels.append(
        create_timeseries_panel(
            "TPOT (Time per Output Token) by Model - P95",
            [
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_model_tpot_seconds_bucket[5m])) by (le, model))",
                    "{{model}}",
                    "A",
                )
            ],
            x=12,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="s",
        )
    )
    panel_id += 1
    y_pos += 8

    # ========== 6. Cache Plugin Metrics ==========
    panels.append(create_row_panel("Cache Plugin Metrics", y=y_pos, panel_id=600))
    y_pos += 1

    # Cache Hit Rate by Decision
    panels.append(
        create_timeseries_panel(
            "Cache Hit Rate by Decision",
            [
                create_target(
                    "sum(rate(llm_cache_plugin_hits_total[5m])) by (decision_name) / (sum(rate(llm_cache_plugin_hits_total[5m])) by (decision_name) + sum(rate(llm_cache_plugin_misses_total[5m])) by (decision_name)) * 100",
                    "{{decision_name}}",
                    "A",
                )
            ],
            x=0,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="percent",
        )
    )
    panel_id += 1

    # Cache Hits/Misses by Decision
    panels.append(
        create_timeseries_panel(
            "Cache Hits/Misses by Decision",
            [
                create_target(
                    "sum(rate(llm_cache_plugin_hits_total[$__range])) by (decision_name)",
                    "{{decision_name}} - hits",
                    "A",
                ),
                create_target(
                    "sum(rate(llm_cache_plugin_misses_total[$__range])) by (decision_name)",
                    "{{decision_name}} - misses",
                    "B",
                ),
            ],
            x=12,
            y=y_pos,
            w=12,
            h=8,
            panel_id=panel_id,
            unit="ops",
        )
    )
    panel_id += 1
    y_pos += 8

    # Current Cache Items by Backend
    panels.append(
        create_stat_panel(
            "Current Cache Items (Total)",
            "sum(llm_cache_entries_total)",
            unit="short",
            x=0,
            y=y_pos,
            w=6,
            h=6,
            panel_id=panel_id,
        )
    )
    panel_id += 1

    # Cache Items by Backend (Gauge)
    panels.append(
        create_timeseries_panel(
            "Cache Items by Backend",
            [
                create_target(
                    "llm_cache_entries_total",
                    "{{backend}}",
                    "A",
                )
            ],
            x=6,
            y=y_pos,
            w=18,
            h=6,
            panel_id=panel_id,
            unit="short",
        )
    )
    panel_id += 1
    y_pos += 6

    # Cache Item Operations Over Time (add/expired/evicted)
    panels.append(
        create_timeseries_panel(
            "Cache Item Operations (add/cleanup_expired/evict)",
            [
                create_target(
                    'sum(rate(llm_cache_operations_total{operation="add_entry", status="success"}[$__range])) by (backend)',
                    "{{backend}} - add",
                    "A",
                ),
                create_target(
                    'sum(rate(llm_cache_operations_total{operation="cleanup_expired", status="success"}[$__range])) by (backend)',
                    "{{backend}} - expired",
                    "B",
                ),
                create_target(
                    'sum(rate(llm_cache_operations_total{operation="evict", status="success"}[$__range])) by (backend)',
                    "{{backend}} - evicted",
                    "C",
                ),
            ],
            x=0,
            y=y_pos,
            w=24,
            h=8,
            panel_id=panel_id,
            unit="ops",
        )
    )
    panel_id += 1
    y_pos += 8

    # Cache Operation Latency
    panels.append(
        create_timeseries_panel(
            "Cache Operation Latency (p95)",
            [
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_cache_operation_duration_seconds_bucket[$__range])) by (backend, operation, le))",
                    "{{backend}} - {{operation}}",
                    "A",
                )
            ],
            x=0,
            y=y_pos,
            w=24,
            h=8,
            panel_id=panel_id,
            unit="s",
        )
    )
    panel_id += 1
    y_pos += 8

    return panels


def main():
    """Main function to generate and save the dashboard"""
    panels = generate_dashboard()

    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "panels": panels,
        "refresh": "10s",
        "schemaVersion": 39,
        "tags": ["llm", "router", "semantic"],
        "templating": {
            "list": [
                {
                    "current": {
                        "selected": False,
                        "text": "Prometheus",
                        "value": "Prometheus",
                    },
                    "hide": 0,
                    "includeAll": False,
                    "label": "Datasource",
                    "multi": False,
                    "name": "DS_PROMETHEUS",
                    "options": [],
                    "query": "prometheus",
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": False,
                    "type": "datasource",
                }
            ]
        },
        "time": {"from": "now-3h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": "vLLM Semantic Router Dashboard",
        "uid": "vllm-semantic-router",
        "version": 1,
        "weekStart": "",
    }

    # Save to file
    output_file = "llm-router-dashboard.serve.json"
    with open(output_file, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"Dashboard generated successfully: {output_file}")
    print(f"Total panels: {len(panels)}")


if __name__ == "__main__":
    main()
