"""Composable Grafana dashboard sections for the semantic router."""

from dataclasses import dataclass, field

from grafana_panel_factories import (
    create_bar_chart_panel,
    create_row_panel,
    create_stat_panel,
    create_target,
    create_timeseries_panel,
)


@dataclass
class DashboardBuildContext:
    panels: list = field(default_factory=list)
    panel_id: int = 1
    y_pos: int = 0


def append_overall_request_header(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(
        create_row_panel("Overall Request Metrics", y=ctx.y_pos, panel_id=100)
    )
    ctx.y_pos += 1

    ctx.panels.append(
        create_stat_panel(
            "Total Requests",
            "sum(increase(llm_model_requests_total[$__range]))",
            unit="short",
            x=0,
            y=ctx.y_pos,
            w=8,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_stat_panel(
            "Average QPS",
            "sum(rate(llm_model_requests_total[$__range]))",
            unit="reqps",
            x=8,
            y=ctx.y_pos,
            w=8,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_stat_panel(
            "Success Rate",
            "sum(increase(llm_model_requests_total[$__range])) / (sum(increase(llm_model_requests_total[$__range])) + (sum(increase(llm_request_errors_total[$__range])) or vector(0))) * 100",
            unit="percent",
            x=16,
            y=ctx.y_pos,
            w=8,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 6


def append_overall_request_timeseries(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(
        create_timeseries_panel(
            "Request Count Trend",
            [
                create_target(
                    "sum(rate(llm_model_requests_total[5m]))", "Requests/sec", "A"
                )
            ],
            x=0,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="reqps",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="s",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_token_usage_section(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(create_row_panel("LLM Token Usage", y=ctx.y_pos, panel_id=200))
    ctx.y_pos += 1

    ctx.panels.append(
        create_stat_panel(
            "Total Tokens",
            "sum(increase(llm_model_tokens_total[$__range]))",
            unit="short",
            x=0,
            y=ctx.y_pos,
            w=6,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_stat_panel(
            "Avg Tokens/sec",
            "sum(rate(llm_model_tokens_total[$__range]))",
            unit="tps",
            x=6,
            y=ctx.y_pos,
            w=6,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_stat_panel(
            "Avg Prompt Tokens/sec",
            "sum(rate(llm_model_prompt_tokens_total[$__range]))",
            unit="tps",
            x=12,
            y=ctx.y_pos,
            w=6,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_stat_panel(
            "Avg Completion Tokens/sec",
            "sum(rate(llm_model_completion_tokens_total[$__range]))",
            unit="tps",
            x=18,
            y=ctx.y_pos,
            w=6,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 6

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=24,
            h=8,
            panel_id=ctx.panel_id,
            unit="tps",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_signal_extraction_section(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(create_row_panel("Signal Extraction", y=ctx.y_pos, panel_id=300))
    ctx.y_pos += 1

    ctx.panels.append(
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
            "Signal Match Rate by Type",
            [
                create_target(
                    "sum(rate(llm_signal_match_total[5m])) by (signal_type)",
                    "{{signal_type}}",
                    "A",
                )
            ],
            x=12,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="ops",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8

    ctx.panels.append(
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
            "Top 10 Matched Signals",
            [
                create_target(
                    "topk(10, sum(rate(llm_signal_match_total[5m])) by (signal_name))",
                    "{{signal_name}}",
                    "A",
                )
            ],
            x=12,
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="ops",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_decision_matching_section(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(create_row_panel("Decision Matching", y=ctx.y_pos, panel_id=400))
    ctx.y_pos += 1

    ctx.panels.append(
        create_stat_panel(
            "Decision Evaluation Rate",
            "rate(llm_decision_evaluation_total[5m])",
            unit="ops",
            x=0,
            y=ctx.y_pos,
            w=8,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_stat_panel(
            "Decision Match Rate",
            "sum(rate(llm_decision_match_total[5m]))",
            unit="ops",
            x=8,
            y=ctx.y_pos,
            w=8,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
        create_stat_panel(
            "Decision Evaluation Latency (P95)",
            "histogram_quantile(0.95, sum(rate(llm_decision_evaluation_latency_seconds_bucket[5m])) by (le))",
            unit="s",
            x=16,
            y=ctx.y_pos,
            w=8,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 6

    ctx.panels.append(
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
            "Decision Confidence (P50) by Decision",
            [
                create_target(
                    "histogram_quantile(0.5, sum(rate(llm_decision_confidence_bucket[5m])) by (le, decision_name))",
                    "{{decision_name}}",
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


def append_model_distribution_section(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(create_row_panel("Model Distribution", y=ctx.y_pos, panel_id=500))
    ctx.y_pos += 1

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=24,
            h=8,
            panel_id=ctx.panel_id,
            unit="short",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="reqps",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="ops",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def append_model_latency_section(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(
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
            "TPOT (Time per Output Token) by Model - P95",
            [
                create_target(
                    "histogram_quantile(0.95, sum(rate(llm_model_tpot_seconds_bucket[5m])) by (le, model))",
                    "{{model}}",
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


def append_cache_plugin_section_part1(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(
        create_row_panel("Cache Plugin Metrics", y=ctx.y_pos, panel_id=600)
    )
    ctx.y_pos += 1

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="percent",
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=12,
            h=8,
            panel_id=ctx.panel_id,
            unit="ops",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8

    ctx.panels.append(
        create_stat_panel(
            "Current Cache Items (Total)",
            "sum(llm_cache_entries_total)",
            unit="short",
            x=0,
            y=ctx.y_pos,
            w=6,
            h=6,
            panel_id=ctx.panel_id,
        )
    )
    ctx.panel_id += 1

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=18,
            h=6,
            panel_id=ctx.panel_id,
            unit="short",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 6


def append_cache_plugin_section_part2(ctx: DashboardBuildContext) -> None:
    ctx.panels.append(
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
            y=ctx.y_pos,
            w=24,
            h=8,
            panel_id=ctx.panel_id,
            unit="ops",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8

    ctx.panels.append(
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
            y=ctx.y_pos,
            w=24,
            h=8,
            panel_id=ctx.panel_id,
            unit="s",
        )
    )
    ctx.panel_id += 1
    ctx.y_pos += 8


def generate_all_dashboard_panels():
    ctx = DashboardBuildContext()
    append_overall_request_header(ctx)
    append_overall_request_timeseries(ctx)
    append_token_usage_section(ctx)
    append_signal_extraction_section(ctx)
    append_decision_matching_section(ctx)
    append_model_distribution_section(ctx)
    append_model_latency_section(ctx)
    append_cache_plugin_section_part1(ctx)
    append_cache_plugin_section_part2(ctx)
    return ctx.panels
