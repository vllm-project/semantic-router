"""Dashboard composition: measured outcomes before routing diagnostics."""

from dataclasses import dataclass, field

from grafana_observability_sections import (
    append_accounting_section,
    append_backend_section,
    append_plugin_cache_section,
    append_telemetry_section,
)
from grafana_panel_factories import (
    create_row_panel,
    create_stat_panel,
    create_target,
    create_timeseries_panel,
    metric,
    rate_sum,
)
from grafana_routing_contract_sections import append_routing_contract_section


@dataclass
class DashboardBuildContext:
    panels: list = field(default_factory=list)
    panel_id: int = 1
    y_pos: int = 0

    def row(self, title):
        self.panels.append(create_row_panel(title, self.y_pos, self.panel_id))
        self.panel_id += 1
        self.y_pos += 1

    def pair(self, definitions):
        for index, (title, targets, unit, description) in enumerate(definitions):
            panel = create_timeseries_panel(
                title,
                targets,
                x=(index % 2) * 12,
                y=self.y_pos,
                panel_id=self.panel_id,
                unit=unit,
            )
            panel["description"] = description
            self.panels.append(panel)
            self.panel_id += 1
            if index % 2 or index == len(definitions) - 1:
                self.y_pos += 8

    def collapsed(self, title, append_content):
        self.row(title)
        row = self.panels[-1]
        nested = DashboardBuildContext(panel_id=self.panel_id, y_pos=self.y_pos)
        append_content(nested)
        row.update(collapsed=True, panels=nested.panels)
        self.panel_id = nested.panel_id


def append_request_section(ctx):
    ctx.row("Inference overview")
    outcomes = metric("llm_request_outcomes_total", 'traffic_kind="inference"')
    successes = metric(
        "llm_request_outcomes_total", 'traffic_kind="inference",outcome="success"'
    )
    total = f"sum(increase({outcomes}[$__range]))"
    success = f"sum(increase({successes}[$__range]))"
    stats = [
        (
            "Completed inference requests",
            total,
            "short",
            "Terminal requests in the selected interval, including errors and cancellations. "
            "Excludes catalog, health checks and response-object operations. Prometheus increase estimates changes between scrapes.",
        ),
        (
            "Inference success rate",
            f"100 * ({success} or (0 * {total})) / ({total} > 0)",
            "percent",
            "Successful terminal inference requests / all terminal inference outcomes. Success requires a normal 2xx/3xx response "
            "without an aborted stream or processing error. No observed denominator remains No data, never zero success.",
        ),
        (
            "Inference throughput",
            rate_sum("llm_request_outcomes_total", labels='traffic_kind="inference"'),
            "reqps",
            "Terminal inference requests per second over Grafana's scrape-aware rate interval.",
        ),
        (
            "Requests in flight",
            f'sum({metric("llm_model_inflight_requests")})',
            "short",
            "Requests currently tracked against logical models. This is not backend queue depth or hardware utilization.",
        ),
    ]
    for index, (title, expression, unit, description) in enumerate(stats):
        panel = create_stat_panel(
            title,
            expression,
            unit,
            x=index * 6,
            y=ctx.y_pos,
            h=4,
            panel_id=ctx.panel_id,
        )
        panel["description"] = description
        panel["targets"][0].update(instant=True, range=False)
        panel["options"]["graphMode"] = "none"
        ctx.panels.append(panel)
        ctx.panel_id += 1
    ctx.y_pos += 4
    duration = metric("llm_request_duration_seconds_bucket", 'traffic_kind="inference"')
    ctx.pair(
        [
            (
                "Inference outcomes",
                [
                    create_target(
                        rate_sum(
                            "llm_request_outcomes_total",
                            "outcome",
                            'traffic_kind="inference"',
                        ),
                        "{{outcome}}",
                    )
                ],
                "reqps",
                "One terminal outcome per processed inference request. Errors, cancellation, timeout and incomplete streams remain separate.",
            ),
            (
                "Inference request duration (P50/P95/P99)",
                [
                    create_target(
                        f"histogram_quantile({q}, sum(rate({duration}[$__rate_interval])) by (le))",
                        name,
                        ref,
                    )
                    for q, name, ref in [
                        (0.50, "P50", "A"),
                        (0.95, "P95", "B"),
                        (0.99, "P99", "C"),
                    ]
                ],
                "s",
                "Time from starting request processing to its terminal outcome, including routing and response streaming. "
                "All inference outcomes contribute. Histogram estimates use finite buckets through 1800 seconds; "
                "quantiles in the overflow bucket are capped at that boundary.",
            ),
        ]
    )


def generate_all_dashboard_panels():
    ctx = DashboardBuildContext()
    append_request_section(ctx)
    append_routing_contract_section(ctx)
    append_backend_section(ctx)
    append_plugin_cache_section(ctx)
    ctx.collapsed("Accounting and MoM execution", append_accounting_section)
    append_telemetry_section(ctx)
    return ctx.panels
