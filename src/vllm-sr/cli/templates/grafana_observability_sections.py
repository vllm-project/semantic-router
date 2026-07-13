"""Grafana dashboard sections for semantic router observability."""

from grafana_panel_factories import (
    create_bar_chart_panel,
    create_row_panel,
    create_target,
    create_timeseries_panel,
)


def append_routing_selection_section(ctx) -> None:
    ctx.panels.append(
        create_row_panel("Routing and Model Selection", y=ctx.y_pos, panel_id=550)
    )
    ctx.y_pos += 1

    append_routing_latency_panels(ctx)
    append_selection_inflight_panels(ctx)
    append_request_error_panels(ctx)


def append_routing_latency_panels(ctx) -> None:
    ctx.panels.append(
        create_timeseries_panel(
            "Router Routing Latency (P50/P95/P99)",
            [
                create_target(
                    "histogram_quantile(0.50, sum(rate(llm_model_routing_latency_seconds_bucket[5m])) by (le))",
                    "P50",
                    "A",
                ),
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_model_routing_latency_seconds_bucket[5m])) by (le))",
                    "P95",
                    "B",
                ),
                create_target(
                    "histogram_quantile(0.99, sum(rate(llm_model_routing_latency_seconds_bucket[5m])) by (le))",
                    "P99",
                    "C",
                ),
            ],
            x=0,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="s",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_timeseries_panel(
            "Model Selection Duration (P95)",
            [
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_model_selection_duration_seconds_bucket[5m])) by (le, method, tier))",
                    "{{method}} {{tier}}",
                    "A",
                )
            ],
            x=12,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="s",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_selection_inflight_panels(ctx) -> None:
    ctx.panels.append(
        create_bar_chart_panel(
            "Selection Count by Method and Tier",
            [
                create_target(
                    "sum(increase(llm_model_selection_total[$__range])) by (method, tier)",
                    "{{method}} {{tier}}",
                    "A",
                )
            ],
            x=0,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="short",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_timeseries_panel(
            "Inflight Requests by Model",
            [
                create_target(
                    "sum(llm_model_inflight_requests) by (model)",
                    "{{model}}",
                    "A",
                )
            ],
            x=12,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="short",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_request_error_panels(ctx) -> None:
    ctx.panels.append(
        create_timeseries_panel(
            "Request Errors by Reason",
            [
                create_target(
                    "sum(rate(llm_request_errors_total[5m])) by (reason)",
                    "{{reason}}",
                    "A",
                )
            ],
            x=0,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="ops",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_bar_chart_panel(
            "Top Failing Models",
            [
                create_target(
                    "topk(10, sum(increase(llm_request_errors_total[$__range])) by (model))",
                    "{{model}}",
                    "A",
                )
            ],
            x=12,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="short",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_windowed_model_health_section(ctx) -> None:
    ctx.panels.append(
        create_row_panel("Windowed Model Health", y=ctx.y_pos, panel_id=575)
    )
    ctx.y_pos += 1

    append_windowed_latency_error_panels(ctx)
    append_windowed_queue_utilization_panels(ctx)


def append_windowed_latency_error_panels(ctx) -> None:
    ctx.panels.append(
        create_timeseries_panel(
            "Windowed Model Latency",
            [
                create_target(
                    "llm_model_latency_windowed_seconds",
                    "{{model}} {{time_window}} avg",
                    "A",
                ),
                create_target(
                    "llm_model_latency_p95_windowed_seconds",
                    "{{model}} {{time_window}} p95",
                    "B",
                ),
                create_target(
                    "llm_model_latency_p99_windowed_seconds",
                    "{{model}} {{time_window}} p99",
                    "C",
                ),
            ],
            x=0,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="s",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_timeseries_panel(
            "Windowed Model Error Rate",
            [
                create_target(
                    "llm_model_error_rate_windowed",
                    "{{model}} {{time_window}}",
                    "A",
                )
            ],
            x=12,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="percentunit",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_windowed_queue_utilization_panels(ctx) -> None:
    ctx.panels.append(
        create_timeseries_panel(
            "Estimated Queue Depth by Model",
            [
                create_target(
                    "llm_model_queue_depth_estimated",
                    "{{model}}",
                    "A",
                )
            ],
            x=0,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="short",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_timeseries_panel(
            "Model Utilization by Window",
            [
                create_target(
                    "llm_model_utilization_percentage",
                    "{{model}} {{time_window}}",
                    "A",
                )
            ],
            x=12,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="percent",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_session_cache_warmth_section(ctx) -> None:
    ctx.panels.append(
        create_row_panel("Session and Cache Warmth", y=ctx.y_pos, panel_id=590)
    )
    ctx.y_pos += 1

    ctx.panels.append(
        create_timeseries_panel(
            "Session Model Transitions",
            [
                create_target(
                    "sum(rate(llm_session_model_transitions_total[5m])) by (from_model, to_model)",
                    "{{from_model}} to {{to_model}}",
                    "A",
                )
            ],
            x=0,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="ops",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_timeseries_panel(
            "Cache Warmth Estimate (P50/P95)",
            [
                create_target(
                    "histogram_quantile(0.50, sum(rate(llm_cache_warmth_estimate_bucket[5m])) by (le, model))",
                    "{{model}} p50",
                    "A",
                ),
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_cache_warmth_estimate_bucket[5m])) by (le, model))",
                    "{{model}} p95",
                    "B",
                ),
            ],
            x=12,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="percentunit",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8
