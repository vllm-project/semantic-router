"""Current entrypoint → recipe → signals → decision → model workflow."""

from grafana_panel_factories import create_target, histogram_mean, rate_sum


def append_routing_contract_section(ctx):
    ctx.row("Recipe workflow")
    ctx.pair(
        [
            (
                "Entrypoint requests by recipe",
                [
                    create_target(
                        rate_sum("llm_entrypoint_requests_total", "entrypoint, recipe"),
                        "{{entrypoint}} → {{recipe}}",
                    )
                ],
                "reqps",
                "Requests resolved to a configured entrypoint and recipe before selection or backend execution. "
                "This boundary is distinct from terminal outcomes above.",
            ),
            (
                "Recipe model selections",
                [
                    create_target(
                        rate_sum(
                            "llm_recipe_selections_total",
                            "recipe, decision, algorithm, model",
                        ),
                        "{{recipe}} / {{decision}} / {{algorithm}} → {{model}}",
                    )
                ],
                "ops",
                "Successful model selections, not completed requests or quality measurements. Logical configuration names identify each recipe's selected decision and algorithm.",
            ),
            (
                "Recipe stage duration",
                [
                    create_target(
                        histogram_mean(
                            "llm_routing_stage_duration_seconds", "recipe, stage"
                        ),
                        "{{recipe}} / {{stage}}",
                    )
                ],
                "s",
                "Observed mean duration by recipe stage. Signals includes projection evaluation; algorithm measures selection, not model generation. "
                "Means use measured sums and counts without clipping long observations to histogram bucket bounds.",
            ),
            (
                "Projection evaluations",
                [
                    create_target(
                        rate_sum("llm_projection_score_count", "recipe, projection"),
                        "{{recipe}} / {{projection}}",
                    )
                ],
                "ops",
                "Recorded projection score observations per second. A projection score is not a quality probability; "
                "inspect the request's Projection Trace in Insights for its exact score and rule inputs.",
            ),
            (
                "Signal matches by family",
                [
                    create_target(
                        rate_sum("llm_signal_match_total", "signal_type"),
                        "{{signal_type}}",
                    )
                ],
                "ops",
                "Recorded rule matches, not a percentage of requests. One request can match several rules and signal families.",
            ),
            (
                "Decision rule matches",
                [
                    create_target(
                        rate_sum("llm_decision_match_total", "decision_name"),
                        "{{decision_name}}",
                    )
                ],
                "ops",
                "All matched decision rules with recipe-scoped names. Multiple rules can match one request; "
                "the winning route appears in Recipe model selections. Rule confidence is not model-answer quality.",
            ),
        ]
    )
