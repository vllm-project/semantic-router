"""Reported model usage, instrumented plugin work, and telemetry health."""

from grafana_panel_factories import create_target, histogram_mean, metric, rate_sum


def append_backend_section(ctx):
    ctx.row("Model responses and usage")
    ctx.pair(
        [
            (
                "Reported tokens by model",
                [
                    create_target(
                        rate_sum("llm_model_tokens_total", "model"), "{{model}}"
                    )
                ],
                "tps",
                "Tokens reported by response usage. Missing usage is not measured as zero; cache responses may include reused usage. "
                "Not billed cost, physical token generation or capability. Use Evaluation for frozen-price comparisons.",
            ),
            (
                "Requests in flight by model",
                [
                    create_target(
                        f'sum({metric("llm_model_inflight_requests")}) by (model)',
                        "{{model}}",
                    )
                ],
                "short",
                "Requests tracked against each logical model, collected at scrape time. No waiting-queue length or utilization is inferred.",
            ),
            (
                "Response duration by model",
                [
                    create_target(
                        histogram_mean("llm_model_completion_latency_seconds", "model"),
                        "{{model}}",
                    )
                ],
                "s",
                "Mean duration from request-body processing through recorded response completion, including routing. "
                "This observation can include cache responses and is not the terminal-request population above.",
            ),
            (
                "First response observation by model",
                [
                    create_target(
                        histogram_mean(
                            "llm_model_first_response_observation_seconds", "model"
                        ),
                        "{{model}}",
                    )
                ],
                "s",
                "Mean time from request-body processing to the first observed streaming chunk or nonstreaming response headers. "
                "Includes routing and may precede a content token; not a uniform backend time-to-first-token measurement.",
            ),
            (
                "Response duration per output token",
                [
                    create_target(
                        histogram_mean(
                            "llm_model_response_duration_per_output_token_seconds",
                            "model",
                        ),
                        "{{model}}",
                    )
                ],
                "s",
                "Mean of per-response duration / reported output-token count, for responses with output tokens. "
                "Includes routing and first-response wait; not decode inter-token latency or a token-weighted aggregate.",
            ),
            (
                "Recorded model errors",
                [
                    create_target(
                        rate_sum("llm_request_errors_total", "model, reason"),
                        "{{model}} / {{reason}}",
                    )
                ],
                "ops",
                "Instrumented model error events. Use Inference outcomes for request-level denominators; "
                "these diagnostic events are not a disjoint terminal-request counter.",
            ),
        ]
    )


def append_plugin_cache_section(ctx):
    ctx.row("Plugins and response cache")
    ctx.pair(
        [
            (
                "Plugin executions",
                [
                    create_target(
                        rate_sum(
                            "llm_plugin_execution_total",
                            "plugin_type, decision_name, status",
                        ),
                        "{{plugin_type}} / {{decision_name}} / {{status}}",
                    )
                ],
                "ops",
                "Instrumented plugin executions and recorded outcomes. Missing series do not imply success or that every configured plugin ran.",
            ),
            (
                "Plugin duration",
                [
                    create_target(
                        histogram_mean(
                            "llm_plugin_execution_latency_seconds", "plugin_type"
                        ),
                        "{{plugin_type}}",
                    )
                ],
                "s",
                "Mean measured duration of instrumented plugin executions. Unknown or absent observations stay absent.",
            ),
            (
                "Cache lookup outcomes",
                [
                    create_target(
                        rate_sum(
                            "llm_response_cache_operation_duration_seconds_count",
                            "backend, operation, status",
                            'operation=~"lookup_exact|lookup_semantic"',
                        ),
                        "{{backend}} / {{operation}} / {{status}}",
                    )
                ],
                "ops",
                "Measured response-cache lookup hit, miss and error operations. Exact and semantic lookups stay distinct. "
                "One request can perform multiple lookups; these counts do not define a request-level hit percentage.",
            ),
            (
                "Cache operation duration",
                [
                    create_target(
                        histogram_mean(
                            "llm_response_cache_operation_duration_seconds",
                            "backend, operation",
                        ),
                        "{{backend}} / {{operation}}",
                    )
                ],
                "s",
                "Mean wall-clock duration measured by the response-cache service. Uses timed canonical operations, "
                "not legacy backend callbacks that can report placeholder zero durations.",
            ),
        ]
    )


def append_accounting_section(ctx):
    coverage = (
        "Detailed attempt instrumentation currently covers Confidence execution. "
        "An absent series for another algorithm does not prove zero attempts. "
    )
    price_basis = (
        "Known configured-price amounts only, not invoices or hardware cost. "
        "Unknown usage or prices are omitted; a known subtotal does not prove complete accounting. "
        "Currencies remain separate; no savings percentage is inferred. "
    )
    ctx.pair(
        [
            (
                "Known model cost rate",
                [
                    create_target(
                        rate_sum("llm_model_cost_total", "model, currency"),
                        "{{model}} / {{currency}} per second",
                    )
                ],
                "short",
                price_basis
                + "Each series is currency units per second attributed to a logical model. "
                "Do not add this series to Looper attempt cost: accounting scopes can overlap.",
            ),
            (
                "Reported prompt and completion tokens",
                [
                    create_target(
                        rate_sum("llm_model_prompt_tokens_total", "model"),
                        "{{model}} / prompt",
                        "A",
                    ),
                    create_target(
                        rate_sum("llm_model_completion_tokens_total", "model"),
                        "{{model}} / completion",
                        "B",
                    ),
                ],
                "tps",
                "Reported usage breakdown per second. Responses with only total-token usage do not enter this breakdown. "
                "Prompt tokens can include cache reads/writes; these series alone do not establish billed-token buckets.",
            ),
            (
                "Looper attempt outcomes",
                [
                    create_target(
                        rate_sum(
                            "llm_looper_attempts_total", "algorithm, stage, status"
                        ),
                        "{{algorithm}} / {{stage}} / {{status}}",
                    )
                ],
                "ops",
                coverage
                + "One terminal event per instrumented attempt. Internal model/verifier attempts are distinct from public inference request outcomes.",
            ),
            (
                "Looper attempt duration",
                [
                    create_target(
                        histogram_mean(
                            "llm_looper_attempt_duration_seconds",
                            "algorithm, stage, status",
                        ),
                        "{{algorithm}} / {{stage}} / {{status}}",
                    )
                ],
                "s",
                coverage
                + "Mean measured end-to-end attempt duration by stage and outcome. Milliseconds are converted to seconds by the emitter.",
            ),
            (
                "Known Looper attempt cost rate",
                [
                    create_target(
                        rate_sum(
                            "llm_looper_attempt_cost_total",
                            "algorithm, stage, currency",
                        ),
                        "{{algorithm}} / {{stage}} / {{currency}} per second",
                    )
                ],
                "short",
                coverage
                + price_basis
                + "Each series is known currency units per second. Do not sum with model cost counters.",
            ),
            (
                "Looper attempt tokens",
                [
                    create_target(
                        rate_sum(
                            "llm_looper_attempt_tokens_total",
                            "algorithm, stage, token_type",
                        ),
                        "{{algorithm}} / {{stage}} / {{token_type}}",
                    )
                ],
                "tps",
                coverage
                + "Reported prompt/completion usage for internal model and verifier stages. "
                "These attempt observations may overlap model usage counters and must not be added to them.",
            ),
        ]
    )


def append_telemetry_section(ctx):
    ctx.row("Telemetry health")
    ctx.pair(
        [
            (
                "Telemetry scrape health",
                [
                    create_target(
                        'up{job=~"semantic-router|jaeger"}', "{{job}} / {{instance}}"
                    )
                ],
                "short",
                "All router and Jaeger scrape targets, independent of the Router instance filter. "
                "1 means scrape success, 0 means failure; missing means no scrape evidence. Does not prove inference traces exist.",
            ),
            (
                "Router trace export outcomes",
                [
                    create_target(
                        rate_sum("llm_trace_export_spans_total", "result"), "{{result}}"
                    )
                ],
                "ops",
                "Spans in exporter batches. Excludes unsampled and SDK queue-dropped spans and does not prove durable collector storage. "
                "No data may mean no batch was exported in this interval.",
            ),
        ]
    )
