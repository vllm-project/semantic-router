"""Cold-start C4+C5 (template path): per-lane bindings from the C3 profile.

Replaces the legacy ``result_to_config.py`` scaffold as the main generation
path (sub-issue-8 §2.2 steps 5-7). For every ``--lambdas`` value this module:

1. **Checks the routing space** (step 5). Either condition ends the flow
   with a "no benefit to routing" report and no candidates:
   - on every measured lane the best model's advantage over a common cheapest
     model is below ``bootstrap.min_advantage_pp``;
   - every measured lane offers the same eligible models and the same
     accuracy and normalized cost differences between them, so the ranking
     cannot vary by lane at any lambda.
2. **Plans the bindings** (step 6). Per lane, in the template's order:
   capability filter (only models whose user-declared capabilities satisfy
   the lane requirement, sub-issue-8 §2.1 - never guessed) -> rank by
   ``accuracy - lambda * normalized_cost`` (cost normalized per lane by the
   max across models, the ``cost_aware_calib.py`` convention) -> take the
   top ``minimum_candidates`` (at least one; frontier-style lanes take
   their declared minimum). Lanes observed on fewer than
   ``bootstrap.min_lane_cases`` cases are **unmeasured**: their candidates
   are ranked by the lambda-adjusted overall score and marked "assigned by
   the default rule, untested". A lane with no model satisfying its
   capability requirement must be explicitly excluded via
   ``--exclude-decision`` (which makes ``recipe builtin init`` drop the
   lane) - the report names every exclusion.
3. **Generates the config** (step 7). ``vllm-sr recipe builtin init`` binds
   the template to the planned refs over the spec's ``providers_config``
   (endpoints and credentials come only from there), then
   ``vllm-sr config validate`` must accept the output before it is emitted.
   The derived eval copy of the provider config disables the response cache
   (a warm cache would poison replay/live evidence) and records the reason.

Anti-leak: only the calib split feeds the ranking; ``dev`` belongs to the
answer bank and ``frozen_test`` is quarantined (enforced by
``bootstrap.load_profile``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .bootstrap import load_profile, parse_lambdas, validate_config
from .contract import ContractError, ExperimentSpec, dataset_content_hash, lambda_tag
from .observe_lanes import (
    OBSERVE_MODEL_NAME,
    capability_satisfied,
    lane_requirements,
    template_info,
)
from .observe_lanes import _run_json as _default_run_json

BINDINGS_SCHEMA = "recipe-lab-bindings-v1"
CACHE_DISABLED_REASON = (
    "disabled for evaluation: a warm cache would let repeated cases hit the "
    "cache instead of the routed model and poison replay/live evidence. "
    "Re-enable for production if wanted."
)


def resolve_lambdas(spec: ExperimentSpec, raw: str | None) -> list[float]:
    """The lambda list: CLI override, else the contract's bootstrap.lambdas.

    A raw string is the legacy comma-separated form (parse_lambdas
    semantics: deduped, order kept); None falls back to the contract, and a
    contract without lambdas defaults to ``[0.0]`` (pure accuracy).
    """
    if raw is not None and raw.strip():
        return parse_lambdas(raw)
    contract = spec.bootstrap_config.lambdas
    if contract:
        return [float(lam) for lam in contract]
    return [0.0]


def _calib_stats(profile: dict[str, Any]) -> dict[str, Any]:
    stats = (profile.get("model_split_stats") or {}).get("calib") or {}
    if not stats:
        raise ContractError(
            "profile has no calib stats: bindings consumes the C3 profile's "
            "calib split only (rerun profile with --splits calib)"
        )
    return stats


def lane_case_counts(profile: dict[str, Any]) -> dict[str, int]:
    """calib lane -> case count (union over models; all models share tasks)."""
    counts: dict[str, int] = {}
    for alias, block in _calib_stats(profile).items():
        labels = (block or {}).get("labels") or {}
        if not isinstance(labels, dict):
            raise ContractError(
                f"profile calib stats for {alias!r} have no labels block"
            )
        for lane, entry in labels.items():
            n = entry.get("n") if isinstance(entry, dict) else None
            if not isinstance(n, int) or n < 0:
                raise ContractError(
                    f"profile calib stats for {alias!r} lane {lane!r} has no case count"
                )
            counts[lane] = max(counts.get(lane, 0), n)
    if not counts:
        raise ContractError("profile has no calib lanes: nothing to bind")
    return counts


def _lane_stat(
    profile: dict[str, Any], alias: str, lane: str, key: str
) -> float | None:
    block = ((_calib_stats(profile).get(alias) or {}).get("labels") or {}).get(lane)
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _overall_stat(profile: dict[str, Any], alias: str, key: str) -> float | None:
    block = (_calib_stats(profile).get(alias) or {}).get("total")
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _normalize_costs(
    costs: dict[str, float | None],
) -> dict[str, float | None] | None:
    """Per-lane cost normalization (``cost_aware_calib`` convention: divide
    by the max across models). None when no model has a cost."""
    values = {alias: cost for alias, cost in costs.items() if cost is not None}
    if not values:
        return None
    top = max(values.values())
    if top <= 0:
        return {
            alias: 0.0 if cost is not None else None for alias, cost in costs.items()
        }
    return {
        alias: (cost / top if cost is not None else None)
        for alias, cost in costs.items()
    }


def check_routing_space(
    *,
    measured_lanes: list[str],
    accuracies: dict[str, dict[str, float | None]],
    costs: dict[str, dict[str, float | None]],
    min_advantage_pp: float,
) -> dict[str, Any]:
    """Sub-issue-8 §2.2 step 5: is there anything to route?

    ``accuracies[lane][alias]`` / ``costs[lane][alias]`` come from the calib
    profile, filtered to models eligible for the lane. The comparison to a
    fixed cheapest model requires it to attain the lowest cost on every
    measured lane. Different lane minima can themselves make routing useful.
    """
    if not measured_lanes:
        return {
            "verdict": "no_routing_space",
            "reason": "no routable lane meets bootstrap.min_lane_cases",
            "cheapest_model": None,
        }

    minima: list[set[str]] = []
    for lane in measured_lanes:
        lane_costs = costs.get(lane) or {}
        lane_accuracies = accuracies.get(lane) or {}
        if (
            not lane_costs
            or lane_accuracies.keys() - lane_costs.keys()
            or any(cost is None for cost in lane_costs.values())
        ):
            # Accuracy-only candidates are valid. Cost-aware ranking requires
            # complete pricing separately in plan_lane.
            return {
                "verdict": "route",
                "reason": f"incomplete costs on {lane}: no fixed cheapest model established",
                "cheapest_model": None,
            }
        if lane_costs.keys() - lane_accuracies.keys() or any(
            acc is None for acc in lane_accuracies.values()
        ):
            return {
                "verdict": "route",
                "reason": f"incomplete accuracy on {lane}: lane trade-offs are unknown",
                "cheapest_model": None,
            }
        floor = min(lane_costs.values())
        minima.append({alias for alias, cost in lane_costs.items() if cost == floor})
    common = set.intersection(*minima)
    if not common:
        return {
            "verdict": "route",
            "reason": "the cheapest eligible model varies across measured lanes",
            "cheapest_model": None,
        }
    cheapest = min(common)

    lane_advantages: dict[str, float] = {}
    tradeoffs: list[dict[str, tuple[float, float]]] = []
    for lane in measured_lanes:
        accs = accuracies[lane]
        normalized = _normalize_costs(costs[lane])
        lane_advantages[lane] = max(accs.values()) - accs[cheapest]
        # Relative to one fixed model, these are the intercept and the cost
        # coefficient of every eligible model's score at any lambda.
        tradeoffs.append(
            {
                alias: (acc - accs[cheapest], normalized[alias] - normalized[cheapest])
                for alias, acc in accs.items()
            }
        )

    if all(
        advantage * 100 < min_advantage_pp for advantage in lane_advantages.values()
    ):
        largest = max(lane_advantages.values())
        return {
            "verdict": "no_routing_space",
            "reason": (
                f"no measured lane shows a {min_advantage_pp:g}pp advantage of "
                f"the best model over the cheapest ({cheapest}); the largest "
                f"advantage is {largest * 100:.1f}pp"
            ),
            "cheapest_model": cheapest,
            "lane_advantages": {
                lane: round(advantage, 6)
                for lane, advantage in sorted(lane_advantages.items())
            },
        }
    if all(lane == tradeoffs[0] for lane in tradeoffs[1:]):
        scores = accuracies[measured_lanes[0]]
        winner = min(scores, key=lambda alias: (-scores[alias], alias))
        return {
            "verdict": "no_routing_space",
            "reason": (
                "all measured lanes have identical eligible models and relative "
                "accuracy and normalized cost: every lambda ranks them identically"
            ),
            "cheapest_model": cheapest,
            "first_places": [winner],
            "lane_advantages": {
                lane: round(advantage, 6)
                for lane, advantage in sorted(lane_advantages.items())
            },
        }
    return {"verdict": "route", "cheapest_model": cheapest}


def plan_lane(
    *,
    lane: str,
    template_lane: dict[str, Any],
    models: Any,
    profile: dict[str, Any],
    lam: float,
    min_lane_cases: int,
    lane_counts: dict[str, int],
) -> dict[str, Any]:
    """One lane's binding plan (sub-issue-8 §2.2 step 6).

    Returns the plan dict with ``selected`` refs, the ``measured`` flag, the
    selection basis and the per-model scores the sidecar records.
    """
    requirement = lane_requirements(lane)
    eligible = [
        model
        for model in models
        if capability_satisfied(requirement, model.capabilities)
    ]
    basis = "accuracy - lambda * normalized_cost"
    if template_lane["immediate"]:
        return {
            "lane": lane,
            "immediate": True,
            "measured": False,
            "n": lane_counts.get(lane, 0),
            "selected": [],
            "basis": "immediate response (no backend candidates)",
            "scores": {},
        }
    minimum = template_lane["minimum_candidates"]
    if not eligible:
        raise ContractError(
            f"lane {lane!r} requires capabilities "
            f"{requirement.get('capabilities') or ()} / context window "
            f"{requirement.get('min_context_window')} that none of the declared "
            "models satisfy; pass --exclude-decision "
            f"{lane} explicitly or declare the capability"
        )
    n = lane_counts.get(lane, 0)
    measured = n >= min_lane_cases
    if measured:
        costs = {
            model.name: _lane_stat(profile, model.name, lane, "cost_per_question")
            for model in eligible
        }
        normalized = _normalize_costs(costs)
        if lam > 0 and (normalized is None or any(c is None for c in costs.values())):
            raise ContractError(
                f"lane {lane!r} has incomplete cost data: a lambda > 0 would "
                "silently rank cost-less models as free; declare cost_basis "
                "assumed_pricing (or rerun profile) or drop the lambda"
            )

        def score(model) -> tuple[float, str]:
            acc = _lane_stat(profile, model.name, lane, "accuracy")
            acc = acc if acc is not None else 0.0
            norm = (normalized or {}).get(model.name)
            return acc - lam * (norm if norm is not None else 0.0), model.name

        ranked = sorted(eligible, key=lambda model: (-score(model)[0], score(model)[1]))
        basis_scores = {
            model.name: {
                "accuracy": _lane_stat(profile, model.name, lane, "accuracy"),
                "normalized_cost": (normalized or {}).get(model.name),
                "score": score(model)[0],
            }
            for model in eligible
        }
    else:
        # Unmeasured lane: default rule on the lambda-adjusted overall score,
        # marked untested (sub-issue-8 §2.2 step 6).
        basis = "default rule (unmeasured lane): overall accuracy - lambda * normalized overall cost"
        costs = {
            model.name: _overall_stat(profile, model.name, "cost_per_question")
            for model in eligible
        }
        normalized = _normalize_costs(costs)
        if lam > 0 and (normalized is None or any(c is None for c in costs.values())):
            raise ContractError(
                f"lane {lane!r} (unmeasured) has incomplete overall cost data: "
                "a lambda > 0 would silently rank cost-less models as free; "
                "declare cost_basis assumed_pricing or drop the lambda"
            )

        def oscore(model) -> tuple[float, str]:
            acc = _overall_stat(profile, model.name, "accuracy")
            acc = acc if acc is not None else 0.0
            norm = (normalized or {}).get(model.name)
            return acc - lam * (norm if norm is not None else 0.0), model.name

        ranked = sorted(
            eligible, key=lambda model: (-oscore(model)[0], oscore(model)[1])
        )
        basis_scores = {
            model.name: {
                "overall_accuracy": _overall_stat(profile, model.name, "accuracy"),
                "normalized_overall_cost": (normalized or {}).get(model.name),
                "score": oscore(model)[0],
            }
            for model in eligible
        }
    if len(ranked) < minimum:
        raise ContractError(
            f"lane {lane!r} requires {minimum} distinct candidates but only "
            f"{len(ranked)} model(s) satisfy its capability requirement; pass "
            f"--exclude-decision {lane} explicitly or declare more capable models"
        )
    selected = [model.name for model in ranked[:minimum]]
    return {
        "lane": lane,
        "immediate": False,
        "measured": measured,
        "n": n,
        "minimum_candidates": minimum,
        "selected": selected,
        "basis": basis,
        "scores": basis_scores,
    }


def plan_bindings(
    *,
    spec: ExperimentSpec,
    profile: dict[str, Any],
    template: dict[str, Any],
    lam: float,
    exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Every lane's plan for one lambda, plus the routing-space verdict."""
    lanes = template["lanes"]
    unknown = sorted(set(exclude) - set(lanes))
    if unknown:
        raise ContractError(
            "--exclude-decision names lanes the template does not declare: "
            + ", ".join(unknown)
        )
    counts = lane_case_counts(profile)
    min_lane_cases = spec.bootstrap_config.min_lane_cases
    plans: dict[str, dict[str, Any]] = {}
    for lane in sorted(lanes):
        if lane in exclude:
            plans[lane] = {
                "lane": lane,
                "excluded": True,
                "measured": counts.get(lane, 0) >= min_lane_cases,
                "n": counts.get(lane, 0),
                "selected": [],
                "basis": "explicitly excluded (--exclude-decision)",
                "scores": {},
            }
            continue
        plans[lane] = plan_lane(
            lane=lane,
            template_lane=lanes[lane],
            models=spec.models,
            profile=profile,
            lam=lam,
            min_lane_cases=min_lane_cases,
            lane_counts=counts,
        )
    measured = {
        lane: plan["scores"]
        for lane, plan in plans.items()
        if plan["measured"] and not plan.get("excluded")
    }
    space = check_routing_space(
        measured_lanes=list(measured),
        accuracies={
            lane: {alias: stats["accuracy"] for alias, stats in scores.items()}
            for lane, scores in measured.items()
        },
        costs={
            lane: {alias: stats["normalized_cost"] for alias, stats in scores.items()}
            for lane, scores in measured.items()
        },
        min_advantage_pp=spec.bootstrap_config.min_advantage_pp,
    )
    bindings = {
        lane: [{"model": name} for name in plan["selected"]]
        for lane, plan in sorted(plans.items())
        if not plan.get("excluded")
    }
    return {
        "plans": plans,
        "bindings": bindings,
        "routing_space": space,
        "measured_lanes": list(measured),
        "unmeasured_lanes": sorted(
            lane
            for lane, plan in plans.items()
            if not plan.get("immediate") and not plan["measured"]
        ),
        "excluded_lanes": sorted(exclude),
    }


def _derived_eval_providers(
    providers: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """The eval copy of the provider config: response cache off.

    Returns ``(config, cache_disabled)``; everything else is preserved
    verbatim (endpoints and credentials come only from the user's file).
    """
    config = copy.deepcopy(providers)
    global_section = config.setdefault("global", {})
    stores = global_section.setdefault("stores", {})
    cache = stores.get("response_cache")
    disabled = False
    if isinstance(cache, dict) and cache.get("enabled"):
        cache["enabled"] = False
        disabled = True
    elif not isinstance(cache, dict):
        stores["response_cache"] = {"enabled": False, "backend_type": "memory"}
        disabled = True
    return config, disabled


def _decision_enables_replay(document: dict[str, Any]) -> bool:
    """A decision can opt in when the global replay recorder is disabled."""
    for profile in [document, *(document.get("recipes") or [])]:
        # v0.5 recipes hold decisions at the recipe level; donor-era
        # documents nest them under routing.
        decisions = (profile.get("routing") or {}).get("decisions") or []
        for decision in decisions or profile.get("decisions") or []:
            for plugin in decision.get("plugins") or []:
                if plugin.get("type") == "router_replay":
                    if (plugin.get("configuration") or {}).get("enabled") is True:
                        return True
                    break
    return False


def _derive_eval_replay_store(
    candidate: Path, providers: dict[str, Any]
) -> list[dict[str, str]]:
    """Keep configured storage; supply memory for the template's unbound Postgres."""

    def replay_settings(doc):
        return ((doc.get("global") or {}).get("services") or {}).get(
            "router_replay"
        ) or {}

    document = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    rendered = replay_settings(document)
    source = replay_settings(providers)
    backend = str(rendered.get("store_backend") or "").strip().lower()
    pg = rendered.get("postgres")
    if backend != "postgres" and (backend or pg is None):
        return []
    if not rendered.get("enabled") and not _decision_enables_replay(document):
        return []  # Go initializes no replay store when every recorder is disabled.
    path = "global.services.router_replay"
    if isinstance(pg, dict) and all(
        isinstance(pg.get(k), str) and pg[k].strip() for k in ("database", "user")
    ):
        return (
            []
        )  # The Router defaults host and port; the connection itself is live-checked.
    if not pg and not any(
        key in source
        for key in ("store_backend", "postgres", "redis", "milvus", "qdrant")
    ):
        rendered["store_backend"] = "memory"
        candidate.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return [
            {
                "field": f"{path}.store_backend",
                "old": backend or "postgres",
                "new": "memory",
                "reason": "The template selected PostgreSQL without a connection and providers_config supplies no replay backend or connection.",
            }
        ]
    raise ContractError(
        f"rendered {path}.store_backend selects postgres without a usable connection. "
        f"Set {path}.postgres.database and .user in providers_config or explicitly "
        f"select the intended {path}.store_backend."
    )


def _check_bound_model_cards(candidate: Path, template: dict[str, Any]) -> None:
    """Check statically known routing requirements of the emitted recipe."""
    doc = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    recipe = next(
        (r for r in doc.get("recipes", []) if r.get("name") == template["recipe"]),
        {},
    )
    # v0.5 recipes carry candidate_requirements and decisions at the recipe
    # level; donor-era candidates nest both under the recipe's routing.
    legacy_routing = recipe.get("routing") or {}
    strict = (
        recipe.get("candidate_requirements")
        or legacy_routing.get("candidate_requirements")
        or {}
    )
    limits = strict.get("context") == "known_limits"
    declared = strict.get("capabilities") == "declared"
    if not (limits or declared):
        return

    providers = {p["name"]: p for p in doc["providers"]["models"]}
    cards = {c["name"]: c for c in (doc.get("routing") or {}).get("modelCards", [])}
    issues = []
    for decision in recipe.get("decisions") or legacy_routing.get("decisions") or []:
        lane = decision["name"]
        metadata = template["lanes"].get(lane)
        if not metadata or metadata["immediate"]:
            continue
        minimum = max(1, metadata["minimum_candidates"])
        need = lane_requirements(lane)
        eligible = set()
        reasons = []
        for ref in decision.get("modelRefs", []):
            alias = ref["model"]
            provider = providers.get(alias)
            if provider is None:
                # A unique external ID can also resolve to the provider's card.
                matches = [
                    p
                    for p in providers.values()
                    if alias == p.get("provider_model_id")
                    or alias in (p.get("external_model_ids") or {}).values()
                ]
                provider = matches[0] if len(matches) == 1 else {}
            catalog = provider.get("catalog")
            card_id = catalog or provider.get("name", alias)
            card = cards.get(card_id, {})
            missing = []
            if limits:
                for field in ("context_window_size", "max_output_tokens"):
                    value = card.get(field)
                    if type(value) is not int or value <= 0:
                        # Catalog overlays only override a limit with a positive
                        # value. The Router supplies the inherited value.
                        if not catalog:
                            missing.append(f"positive {field}")
                    elif field == "context_window_size" and value < (
                        need["min_context_window"] or 0
                    ):
                        missing.append(f"{field} >= {need['min_context_window']}")
            if declared and (not catalog or card.get("capabilities") is not None):
                caps = {str(c).strip().lower() for c in card.get("capabilities") or []}
                # ModelCapabilities in the Router accepts these aliases. All
                # recipe-lab replay probes send a text message.
                if not caps.intersection({"text", "chat"}):
                    missing.append("capabilities: text/chat")
                aliases = {
                    "vision": {"vision", "image_input"},
                    "tools": {"tools", "tool_use"},
                }
                for cap in need["capabilities"]:
                    if not caps.intersection(aliases.get(cap, {cap})):
                        missing.append(f"capabilities: {cap}")
            if missing:
                reasons.append(f"{alias} (card {card_id}: {', '.join(missing)})")
            else:
                eligible.add(provider.get("name", alias))
        if len(eligible) < minimum:
            issues.append(
                f"{lane}: {len(eligible)}/{minimum} eligible; " + ", ".join(reasons)
            )
    if issues:
        raise ContractError(
            "bound models cannot serve this recipe; correct providers_config "
            "routing.modelCards or change the lane bindings: " + "; ".join(issues)
        )


def run_bindings(
    *,
    spec: ExperimentSpec,
    profile_dir: str | Path,
    out_dir: str | Path,
    lambdas_raw: str | None,
    exclude: tuple[str, ...] = (),
    vllm_sr_bin: str = "vllm-sr",
    catalog_version: str = "latest",
    model_name: str = OBSERVE_MODEL_NAME,
    timeout_s: float = 300.0,
    run_json: Any = None,
) -> dict[str, Any]:
    """Run the bindings chain for every lambda (sub-issue-8 §2.2 steps 5-7).

    Per lambda: plan -> write ``bindings-lambda-<tag>.yaml`` -> derive the
    eval copy of ``providers_config`` -> ``recipe builtin init`` ->
    ``config validate`` -> emit ``candidate-lambda-<tag>.yaml`` plus the
    provenance sidecar. The routing-space check runs once (it does not
    depend on lambda) and a ``no_routing_space`` verdict ends the command
    with the report and no candidates.
    """
    runner = run_json or _default_run_json
    if spec.template is None:
        raise ContractError(
            "spec declares no template section: the bindings path generates "
            "configs from a built-in recipe template (sub-issue-8 §2.1)"
        )
    if not spec.providers_config:
        raise ContractError(
            "spec declares no providers_config: recipe builtin init consumes "
            "it (endpoints and credentials come only from there)"
        )
    providers_path = Path(spec.providers_config)
    if (
        not providers_path.is_file()
        and spec.spec_dir is not None
        and (Path(spec.spec_dir) / providers_path).is_file()
    ):
        providers_path = Path(spec.spec_dir) / providers_path
    if not providers_path.is_file():
        raise ContractError(f"providers_config not found: {providers_path}")

    profile_path = Path(profile_dir) / "profile.json"
    profile = load_profile(profile_path)

    template = template_info(
        bundle=spec.template.bundle,
        recipe=spec.template.recipe,
        vllm_sr_bin=vllm_sr_bin,
        catalog_version=catalog_version,
        timeout_s=timeout_s,
        run_json=runner,
    )
    lam_values = resolve_lambdas(spec, lambdas_raw)

    space_probe = plan_bindings(
        spec=spec,
        profile=profile,
        template=template,
        lam=lam_values[0],
        exclude=exclude,
    )["routing_space"]
    if space_probe["verdict"] != "route":
        return {
            "schema": BINDINGS_SCHEMA,
            "template": {
                "bundle": template["bundle"],
                "recipe": template["recipe"],
                "sha256": template["sha256"],
                "catalog_version": template["catalog_version"],
            },
            "profile": str(profile_path),
            "routing_space": space_probe,
            "candidates": [],
            "report": ("No benefit to routing: " + space_probe["reason"]),
        }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    providers = yaml.safe_load(providers_path.read_text(encoding="utf-8"))
    if not isinstance(providers, dict):
        raise ContractError(f"{providers_path} is not a YAML mapping")
    eval_providers, cache_disabled = _derived_eval_providers(providers)
    eval_providers_path = out / "providers-eval.yaml"
    eval_providers_path.write_text(
        yaml.safe_dump(eval_providers, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    candidates: list[dict[str, Any]] = []
    for lam in lam_values:
        tag = lambda_tag(lam)
        plan = plan_bindings(
            spec=spec,
            profile=profile,
            template=template,
            lam=lam,
            exclude=exclude,
        )
        bindings_path = out / f"bindings-lambda-{tag}.yaml"
        bindings_path.write_text(
            yaml.safe_dump(plan["bindings"], sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        candidate_path = out / f"candidate-lambda-{tag}.yaml"
        init_argv = [
            str(vllm_sr_bin),
            "recipe",
            "builtin",
            "init",
            str(template["recipe"]),
            "--bundle",
            str(template["bundle"]),
            "--catalog-version",
            str(template["catalog_version"]),
            "--config",
            str(eval_providers_path),
            "--bindings",
            str(bindings_path),
            "--model-name",
            str(model_name),
            "--output",
            str(candidate_path),
        ]
        for lane in plan["excluded_lanes"]:
            init_argv += ["--exclude-decision", lane]
        runner(init_argv, timeout_s=timeout_s, what="recipe builtin init")
        if not candidate_path.is_file():
            raise ContractError(
                "recipe builtin init reported success but wrote no config: "
                f"{candidate_path}"
            )
        evaluation_overrides = _derive_eval_replay_store(candidate_path, providers)
        reason = validate_config(candidate_path, vllm_sr_bin)
        if reason is not None:
            raise ContractError(
                f"candidate for lambda {tag} was rejected by config validate: {reason}"
            )
        _check_bound_model_cards(candidate_path, template)

        sidecar = {
            "schema": BINDINGS_SCHEMA,
            "proposer": "bindings",
            "lambda": lam,
            "evaluation_overrides": evaluation_overrides,
            "template": {
                "bundle": template["bundle"],
                "recipe": template["recipe"],
                "sha256": template["sha256"],
                "catalog_version": template["catalog_version"],
            },
            "profile": {
                "path": str(profile_path),
                "sha256": dataset_content_hash(profile_path),
            },
            "providers": {
                "input": str(providers_path),
                "input_sha256": dataset_content_hash(providers_path),
                "eval_copy": str(eval_providers_path),
                "eval_copy_sha256": dataset_content_hash(eval_providers_path),
                "cache_disabled": cache_disabled,
                "cache_disabled_reason": (
                    CACHE_DISABLED_REASON if cache_disabled else None
                ),
            },
            "bindings": {
                "path": str(bindings_path),
                "sha256": dataset_content_hash(bindings_path),
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": dataset_content_hash(candidate_path),
            },
            "init_command": init_argv,
            "routing_space": plan["routing_space"],
            "measured_lanes": plan["measured_lanes"],
            "unmeasured_lanes": plan["unmeasured_lanes"],
            "excluded_lanes": plan["excluded_lanes"],
            "lanes": plan["plans"],
        }
        sidecar_path = out / f"bindings-lambda-{tag}.provenance.json"
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        candidates.append(
            {
                "lambda": lam,
                "tag": tag,
                "bindings": str(bindings_path),
                "candidate": str(candidate_path),
                "provenance": str(sidecar_path),
                "measured_lanes": plan["measured_lanes"],
                "unmeasured_lanes": plan["unmeasured_lanes"],
                "excluded_lanes": plan["excluded_lanes"],
            }
        )

    return {
        "schema": BINDINGS_SCHEMA,
        "template": {
            "bundle": template["bundle"],
            "recipe": template["recipe"],
            "sha256": template["sha256"],
            "catalog_version": template["catalog_version"],
        },
        "profile": str(profile_path),
        "providers_eval_copy": str(eval_providers_path),
        "routing_space": space_probe,
        "candidates": candidates,
    }
