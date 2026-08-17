"""RAG retrieval Grafana dashboard section for the semantic router."""

from grafana_panel_factories import (
    create_row_panel,
    create_stat_panel,
    create_target,
    create_timeseries_panel,
)


def append_rag_section(ctx) -> None:
    """Append the full RAG retrieval section to the dashboard context."""
    _append_rag_overview(ctx)
    _append_rag_details(ctx)


def _append_rag_overview(ctx) -> None:
    ctx.panels.append(
        create_row_panel("RAG Retrieval Metrics", y=ctx.y_pos, panel_id=700)
    )
    ctx.y_pos += 1

    ctx.panels.append(
        create_stat_panel(
            "RAG Retrieval Rate",
            "sum(rate(rag_retrieval_attempts_total[5m]))",
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
            "RAG Success Rate",
            'sum(rate(rag_retrieval_attempts_total{status="success"}[$__range])) / sum(rate(rag_retrieval_attempts_total[$__range])) * 100',
            unit="percent",
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
            "RAG Cache Hit Rate",
            "sum(rate(rag_cache_hits_total[$__range])) / (sum(rate(rag_cache_hits_total[$__range])) + sum(rate(rag_cache_misses_total[$__range]))) * 100",
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

    ctx.panels.append(
        create_timeseries_panel(
            "RAG Retrieval Latency (P50/P95/P99) by Backend",
            [
                create_target(
                    "histogram_quantile(0.50, sum(rate(rag_retrieval_latency_seconds_bucket[5m])) by (le, backend))",
                    "{{backend}} - P50",
                    "A",
                ),
                create_target(
                    "histogram_quantile(0.95, sum(rate(rag_retrieval_latency_seconds_bucket[5m])) by (le, backend))",
                    "{{backend}} - P95",
                    "B",
                ),
                create_target(
                    "histogram_quantile(0.99, sum(rate(rag_retrieval_latency_seconds_bucket[5m])) by (le, backend))",
                    "{{backend}} - P99",
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


def _append_rag_details(ctx) -> None:
    ctx.panels.append(
        create_timeseries_panel(
            "RAG Retrieval Attempts by Status",
            [
                create_target(
                    "sum(rate(rag_retrieval_attempts_total[5m])) by (status)",
                    "{{status}}",
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
            "RAG Cache Hits/Misses by Backend",
            [
                create_target(
                    "sum(rate(rag_cache_hits_total[5m])) by (backend)",
                    "{{backend}} - hits",
                    "A",
                ),
                create_target(
                    "sum(rate(rag_cache_misses_total[5m])) by (backend)",
                    "{{backend}} - misses",
                    "B",
                ),
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
            "RAG Similarity Score by Backend/Decision",
            [
                create_target(
                    "rag_similarity_score",
                    "{{backend}} - {{decision}}",
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

    ctx.panels.append(
        create_timeseries_panel(
            "RAG Context Length (P95) by Backend",
            [
                create_target(
                    "histogram_quantile(0.95, sum(rate(rag_context_length_chars_bucket[5m])) by (le, backend))",
                    "{{backend}}",
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
