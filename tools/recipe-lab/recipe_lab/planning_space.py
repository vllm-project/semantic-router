"""Describe the knobs on one deployed recipe and derive a guarded count handoff."""

import re

if __package__:
    from .casebook import load_cases
    from .first_tool_action import first_tool_edit, verify_guards
    from .loop_evidence import check, digest, require_isolated, target
    from .semantic_depth import threshold_sites
    from .window_action import (
        apply_window,
        load_space,
        named,
        space_mapping,
        window_bounds,
    )
else:
    from casebook import load_cases
    from first_tool_action import first_tool_edit, verify_guards
    from loop_evidence import check, digest, require_isolated, target
    from semantic_depth import threshold_sites
    from window_action import (
        apply_window,
        load_space,
        named,
        space_mapping,
        window_bounds,
    )


REQUIREMENTS = {"context": "known_limits", "capabilities": "declared"}
SOURCE_TYPES = {"message": "all_messages", "assistant_tool_cycle": "tool_results"}


def _model_cards(doc, strong, fast):
    check(strong != fast, "strong and fast require distinct model bindings")
    cards = doc["providers"]["models"]
    return {
        role: named(cards, name, "provider model")
        for role, name in (("strong", strong), ("fast", fast))
    }


def _refs(lane):
    return [r.get("model") for r in lane.get("modelRefs", [])]


def _guarded(recipe):
    return recipe.get("candidate_requirements") == REQUIREMENTS


def _count_predicate(recipe, lane):
    """Recognize exactly a conjunction of simple count bounds, including NOT gte."""
    rules = lane.get("rules") or {}
    if (
        rules.get("operator") != "AND"
        or set(rules) - {"operator", "conditions", "on_unknown"}
        or rules.get("on_unknown", "no_match") != "no_match"
        or not rules.get("conditions")
    ):
        return None
    signals = recipe.get("signals", {}).get("conversation", [])
    feature, lower, upper = None, 0, None
    for leaf in rules["conditions"]:
        negated = leaf.get("operator") == "NOT"
        if negated:
            if set(leaf) != {"operator", "conditions"} or len(leaf["conditions"]) != 1:
                return None
            condition = leaf["conditions"][0]
        else:
            condition = leaf
        if set(condition) != {"type", "name"} or condition["type"] != "conversation":
            return None
        matches = [s for s in signals if s.get("name") == condition["name"]]
        if len(matches) != 1:
            return None
        signal = matches[0]
        f = signal.get("feature", {})
        source = f.get("source", {})
        kind = SOURCE_TYPES.get(source.get("type"))
        if (
            set(f) != {"type", "source"}
            or f["type"] != "count"
            or set(source) != {"type"}
            or not kind
            or feature not in (None, kind)
        ):
            return None
        feature = kind
        bounds = signal.get("predicate") or {}
        if not bounds or (negated and len(bounds) != 1):
            return None
        for op, n in bounds.items():
            if (
                op not in ("gt", "gte", "lt", "lte")
                or type(n) is not int
                or not 0 <= n < 2**53
            ):
                return None
            if negated:
                flipped = {"gt": "lte", "gte": "lt", "lt": "gte", "lte": "gt"}[op]
            else:
                flipped = op
            if flipped in ("gt", "gte"):
                lower = max(lower, n + (flipped == "gt"))
            else:
                limit = n - (flipped == "lt")
                upper = limit if upper is None else min(upper, limit)
    if upper is not None and upper < lower:
        return None
    return {"feature": feature, "lower": lower, "upper": upper}


def _descriptor(kind, reason, predicate=None):
    return {
        "id": "parent",
        "source": "config",
        "kind": kind,
        "predicate": predicate,
        "reason": reason,
    }


def _routing_shape(recipe, strong, fast, *, historical):
    lanes = recipe.get("decisions", [])
    static = {"type": "static"}
    if (
        recipe.get("strategy") != "priority"
        or not lanes
        or any(d.get("algorithm") != static or d.get("action") for d in lanes)
    ):
        return _descriptor(
            "requires_probe", "requires priority routing and static model bindings"
        )
    catchall = [d for d in lanes if not d.get("rules")]
    if len(catchall) != 1 or _refs(catchall[0]) != [strong]:
        return _descriptor(
            "requires_probe", "strong singleton catch-all not established"
        )
    if len(lanes) == 1:
        return _descriptor("fixed_strong", "the only decision binds the strong model")
    targets = [d for d in lanes if _refs(d) in ([fast], [fast, strong])]
    if len(targets) != 1 or any(
        d is not targets[0] and _refs(d) != [strong] for d in lanes
    ):
        return _descriptor(
            "requires_probe", "multiple route mechanisms need gateway probes"
        )
    lane = targets[0]
    if any(type(d.get("priority", 0)) is not int for d in lanes) or any(
        d is not lane and d.get("priority", 0) >= lane.get("priority", 0) for d in lanes
    ):
        return _descriptor(
            "requires_probe", "a higher or tied decision may override the count"
        )
    predicate = _count_predicate(recipe, lane)
    if predicate is None or (
        not historical and (len(lanes) != 2 or not _guarded(recipe))
    ):
        return _descriptor(
            "requires_probe", "requires complete rules and target eligibility checks"
        )
    result = _descriptor(
        "count_handoff",
        "visible count predicate; confirm effective routing on the gateway",
        predicate,
    )
    result.update(decision=lane["name"], priority=lane.get("priority", 0))
    return result


def extract_candidate(doc, *, public_model, strong, fast):
    """Describe an observed predicate. A predicate does not attest to candidate delivery."""
    return _routing_shape(target(doc, public_model), strong, fast, historical=False)


def extract_reference(doc, *, public_model, strong, fast):
    """Read the dominant count branch of a reference, independent of its old provider settings."""
    recipe = target(doc, public_model)
    result = _routing_shape(recipe, strong, fast, historical=True)
    result.update(
        source="reference",
        reference_recipe=recipe["name"],
        reference_requirements_declared=_guarded(recipe),
    )
    return result


def _render_blocker(recipe, cards):
    if not _guarded(recipe):
        return "requires declared capabilities and known context limits"
    if (
        set(recipe)
        - {
            "name",
            "description",
            "strategy",
            "candidate_requirements",
            "signals",
            "decisions",
        }
        or recipe.get("signals")
        or len(recipe.get("decisions", [])) != 1
    ):
        return "automatic handoff insertion requires an isolated fixed parent with no active signals"
    strong = cards["strong"]["name"]
    lane = recipe["decisions"][0]
    allowed_fields = {
        "name",
        "description",
        "priority",
        "algorithm",
        "modelRefs",
        "rules",
        "adaptations",
        "reliability",
    }
    if (
        recipe.get("strategy") != "priority"
        or set(lane) - allowed_fields
        or lane.get("rules")
        or lane.get("algorithm") != {"type": "static"}
        or lane.get("modelRefs") != [{"model": strong}]
        or type(lane.get("priority", 0)) is not int
        or not 0 <= lane.get("priority", 0) < 2**31 - 1
        or lane.get("adaptations")
        not in (None, {"adaptation": {"candidate_set": "decision"}})
    ):
        return "existing decision has behavior outside the guarded static handoff edit"
    if lane["name"] == "lab_late_fast":
        return "generated decision name already exists"
    return _tool_capability_blocker(cards)


def _tool_capability_blocker(cards):
    for card in cards.values():
        caps = {
            c.strip().lower()
            for c in card.get("capabilities", [])
            if isinstance(c, str)
        }
        if not ({"text", "chat"} & caps and {"tool_use", "tools"} & caps):
            return "both selected models require declared text and tool capability"
        if any(
            type(card.get(k)) is not int or card[k] <= 0
            for k in ("context_window_size", "max_output_tokens")
        ):
            return (
                "both selected models require known positive context and output limits"
            )
    return None


def render_handoff(doc, *, public_model, strong, fast, threshold):
    """Render a single >= tool-results action against the target parent's controls."""
    check(
        type(threshold) is int and 0 < threshold < 2**53,
        "invalid tool-result threshold",
    )
    recipe = require_isolated(doc, public_model)
    blocker = _render_blocker(recipe, _model_cards(doc, strong, fast))
    check(blocker is None, blocker or "")
    base = recipe["decisions"][0]
    signal = {
        "name": "lab_late_tool_results",
        "feature": {"type": "count", "source": {"type": "assistant_tool_cycle"}},
        "predicate": {"gte": threshold},
    }
    branch = {
        **base,
        "name": "lab_late_fast",
        "description": "Prefer the fast model after the declared tool-result count.",
        "priority": base.get("priority", 0) + 1,
        "rules": {
            "operator": "AND",
            "on_unknown": "no_match",
            "conditions": [{"type": "conversation", "name": signal["name"]}],
        },
        "modelRefs": [{"model": fast}, {"model": strong}],
    }
    changed = {
        **recipe,
        "signals": {"conversation": [signal]},
        "decisions": [branch, base],
    }
    return {**doc, "recipes": [changed if r is recipe else r for r in doc["recipes"]]}


def _dependencies(rule):
    if "type" in rule and "name" in rule:
        return [{"type": rule["type"], "name": rule["name"]}]
    return [
        leaf for child in rule.get("conditions", []) for leaf in _dependencies(child)
    ]


def _existing_edits(doc, recipe, cards, public):
    """Use the native editors' shape validators on the actual signal references."""
    windows, first = [], []
    signals = recipe.get("signals", {}).get("conversation", [])
    sources = {s["name"]: s.get("feature", {}).get("source", {}) for s in signals}
    for lane in recipe.get("decisions", []):
        rules = lane.get("rules") or {}
        if rules.get("operator") != "AND":
            continue
        positives, negatives = [], []
        for node in rules.get("conditions", []):
            if node.get("type") == "conversation":
                positives.append(node["name"])
            elif node.get("operator") == "NOT" and len(node.get("conditions", [])) == 1:
                leaf = node["conditions"][0]
                if leaf.get("type") == "conversation":
                    negatives.append(leaf["name"])
        where = {
            "public_model": public,
            "recipe": recipe["name"],
            "decision": lane["name"],
        }
        for lo in positives:
            for hi in negatives:
                if not all(
                    sources.get(k) == {"type": "assistant_tool_cycle"} for k in (lo, hi)
                ):
                    continue
                fields = {**where, "lower_signal": lo, "upper_signal": hi}
                try:
                    lower, upper = window_bounds(doc, fields)
                except ValueError:
                    continue
                windows.append(
                    {**fields, "current_lower": lower, "current_upper": upper}
                )
        if (
            not _guarded(recipe)
            or _tool_capability_blocker(cards)
            or len(positives) != 1
            or len(negatives) != 2
            or lane.get("algorithm") != {"type": "static"}
        ):
            continue
        offered = positives[0]
        disabled = [
            n for n in negatives if sources.get(n) == {"type": "tool_choice_none"}
        ]
        assistant = [
            n
            for n in negatives
            if sources.get(n) == {"type": "message", "role": "assistant"}
        ]
        if len(disabled) != 1 or len(assistant) != 1:
            continue
        bindings = [c["name"] for c in cards.values()]
        refs = lane.get("modelRefs", [])
        if not any(refs == [{"model": backend}] for backend in bindings):
            continue
        fields = {
            **where,
            "tools_signal": offered,
            "disabled_signal": disabled[0],
            "assistant_signal": assistant[0],
        }
        try:
            verify_guards(recipe, lane, fields)
        except ValueError:
            continue
        first.append(
            {**fields, "current_model": refs[0]["model"], "allowed_models": bindings}
        )
    return windows, first


def _inventory(doc, recipe, cards, public):
    decisions = []
    signals = recipe.get("signals", {})
    for lane in recipe.get("decisions", []):
        plugins = lane.get("plugins", [])
        reliability = lane.get("reliability") or {}
        durations = {
            k: v
            for k, v in reliability.items()
            if k
            in (
                "total_timeout",
                "per_try_timeout",
                "idle_timeout",
                "first_byte_timeout",
            )
            and isinstance(v, str)
            and re.fullmatch(r"(?:[0-9.]+(?:h|m|s|ms|us|µs|ns))+", v)
        }
        decisions.append(
            {
                "name": lane["name"],
                "priority": lane.get("priority", 0),
                "algorithm": lane.get("algorithm", {}).get("type"),
                "models": _refs(lane),
                "dependencies": _dependencies(lane.get("rules") or {}),
                "plugin_types": [p.get("type") for p in plugins],
                "generation": [
                    {
                        k: v
                        for k, v in p.get("configuration", {}).items()
                        if k in {"default_max_tokens", "max_tokens_limit"}
                        and type(v) is int
                    }
                    for p in plugins
                    if p.get("type") == "request_params"
                ],
                "reliability_fields": sorted(lane.get("reliability") or {}),
                "timeouts": durations,
                "retention_fields": sorted(lane.get("retention") or {}),
                "retention": {
                    k: v
                    for k, v in (lane.get("retention") or {}).items()
                    if k
                    in {
                        "drop",
                        "ttl_turns",
                        "keep_current_model",
                        "prefer_prefix_retention",
                    }
                    and type(v) in (int, bool)
                },
            }
        )
    numbers = []
    for sig in signals.get("conversation", []):
        f = sig.get("feature", {})
        numbers.append(
            {
                "name": sig["name"],
                "feature": f.get("type"),
                "source": {
                    k: v
                    for k, v in f.get("source", {}).items()
                    if k in ("type", "role")
                },
                "bounds": {
                    k: v
                    for k, v in (sig.get("predicate") or {}).items()
                    if k in ("gt", "gte", "lt", "lte") and type(v) in (int, float)
                },
            }
        )
    windows, first = _existing_edits(doc, recipe, cards, public)
    blocker = _render_blocker(recipe, cards)
    semantic = threshold_sites(recipe, cards["strong"]["name"], cards["fast"]["name"])
    return {
        "public_model": public,
        "recipe": recipe["name"],
        "strategy": recipe.get("strategy"),
        "signal_families": sorted(signals),
        "conversation_signals": numbers,
        **({"semantic_thresholds": semantic} if semantic else {}),
        "decisions": decisions,
        "candidate_requirements": {
            k: v
            for k, v in recipe.get("candidate_requirements", {}).items()
            if k in REQUIREMENTS
        },
        "providers": {
            role: {
                k: card.get(k)
                for k in (
                    "name",
                    "provider_model_id",
                    "capabilities",
                    "context_window_size",
                    "max_output_tokens",
                )
            }
            for role, card in cards.items()
        },
        "families": {
            "trigger": {
                "review_when": "the target trace contains discriminating count boundaries",
                "source": "recipe.signals.conversation and decision.rules",
                "status": (
                    "editable"
                    if blocker is None or windows
                    else "no_supported_parent_shape"
                ),
                "late_tool_results": {
                    "parent_compatible": blocker is None,
                    "reason": blocker,
                    "requires": "a source threshold and the guarded fixed-strong parent",
                },
                "tool_result_window_upper": {
                    "targets": windows,
                    "requires": "an existing AND window with a private negated upper signal and a declared --space domain",
                },
            },
            "binding": {
                "review_when": "an eligible alternative can serve the guarded branch",
                "source": "decision.modelRefs, decision.rules and providers.models",
                "status": "editable" if first else "no_supported_parent_shape",
                "first_tool_model": {
                    "targets": first,
                    "requires": "tools offered, tool choice enabled, zero visible assistant messages and two tool-capable models",
                },
            },
            "priority": {
                "review_when": "gateway selection shows another matching rule won",
                "source": "decision.priority and decision.rules",
                "status": "probe_only",
                "requires": "an overlapping rule and a gateway decision trace",
            },
            "generation": {
                "review_when": "effective generation settings explain a measured difference",
                "source": "decision.plugins.request_params",
                "status": "fixed_for_handoff",
                "requires": "actual provider request, including explicit caller values",
            },
            "reliability": {
                "review_when": "a transport failure affects completed task grades",
                "source": "decision.reliability",
                "status": "fixed_for_handoff",
                "requires": "stream completion status and timing on the same call",
            },
            "session": {
                "review_when": "routing disagrees with the rule across turns of a session",
                "source": "decision.retention and session routing settings",
                "status": "probe_only",
                "requires": "an identified session and observed handoff across turns",
            },
        },
        "fixed_controls": [
            "provider_capabilities",
            "context_window_size",
            "max_output_tokens",
            "caller_generation_parameters",
            "DSH",
            "grader",
        ],
    }


def build_space(doc, *, public_model, strong, fast):
    """Enumerate the project's active configuration and applicable recorded handoffs."""
    recipe = require_isolated(doc, public_model)
    cards = _model_cards(doc, strong, fast)
    args = {"public_model": public_model, "strong": strong, "fast": fast}
    candidates = [extract_candidate(doc, **args)]
    historical = []
    references = (
        (case["id"], ref)
        for case in load_cases()
        for ref in case.get("routing_references", [])
    )
    for case_id, ref in references:
        pair_match = all(
            (card.get("provider_model_id") or card["name"]).casefold()
            in {s.casefold() for s in ref[role + "_ids"]}
            for role, card in cards.items()
        )
        blocker = _render_blocker(recipe, cards)
        if not pair_match:
            blocker = "model identity not matched; use as a transfer lead"
        elif ref["feature"] != "tool_results":
            blocker = "feature requires a dedicated materializer and gateway probe"
        elif ref["required_overrides"]:
            blocker = "requires additional protection: " + ", ".join(
                ref["required_overrides"]
            )
        historical.append(
            {
                **ref,
                "case": case_id,
                "model_pair_match": pair_match,
                "renderable": blocker is None,
                "reason": blocker,
            }
        )
        if blocker is None:
            child = render_handoff(doc, **args, threshold=ref["lower"])
            candidate = extract_candidate(child, **args)
            candidate.update(id=ref["id"], source=case_id)
            candidates.append(candidate)
    return {
        "inventory": _inventory(doc, recipe, cards, public_model),
        "candidates": candidates,
        "historical_references": historical,
    }


def _window_change(recipe, lane, edit, strong, fast):
    before, after = edit["before_upper"], edit["after_upper"]
    interval = {
        "feature": "tool_results",
        "lower": min(before, after) + 1,
        "upper": max(before, after),
    }
    result = _descriptor(
        "requires_probe",
        "window narrowing may alter quality or transfer work to another branch",
    )
    result.update(
        affected_interval=interval,
        change_direction="narrow" if after < before else "expand",
        predicate_scope="changed_window_only",
    )
    if after < before:
        return result
    # More AND conditions or a higher-priority branch may leave the same
    # backend active across the entire interval. Do not score those calls as
    # an observable fast-lane opportunity.
    actual = _count_predicate(recipe, lane)
    if (
        recipe.get("strategy") != "priority"
        or len(lane["rules"]["conditions"]) != 2
        or actual
        != {"feature": "tool_results", "lower": edit["lower"], "upper": before}
    ):
        result["reason"] = (
            "evaluate the other rule conditions or strategy on the public route"
        )
        return result
    if (
        lane.get("algorithm") != {"type": "static"}
        or _refs(lane) not in ([fast], [fast, strong])
        or lane.get("action")
    ):
        result["reason"] = (
            "the edited window does not statically prefer the nominated fast backend"
        )
        return result
    priority = lane.get("priority", 0)
    if any(
        other is not lane and other.get("priority", 0) >= priority
        for other in recipe["decisions"]
    ):
        result["reason"] = (
            "test whether a competing decision takes precedence over the enlarged window"
        )
        return result
    result.update(
        kind="count_handoff",
        predicate=interval,
        reason="newly included count interval; actual backend eligibility and delivery require a probe",
    )
    return result


def native_edits(doc, raw_space, *, base_sha, public_model, strong, fast):
    """Enumerate the declared action domain using the same editors as propose --space."""
    space = space_mapping(raw_space)
    check(
        space.get("base_config_sha256") == base_sha
        and space.get("public_model") == public_model,
        "edit space does not bind this parent configuration and public model",
    )
    recipe = require_isolated(doc, public_model)
    check(space.get("recipe") == recipe["name"], "edit space addresses another recipe")
    lane = named(recipe["decisions"], space.get("decision"), "decision")
    source = digest(raw_space)
    args = {"public_model": public_model, "config_sha256": base_sha}
    candidates = []
    if space.get("action") == "tool_result_window_upper":
        space = load_space(raw_space)
        _, before = window_bounds(doc, space)
        for upper in space["allowed_upper"]:
            if upper == before:
                continue
            changed, edit = apply_window(doc, space, upper=upper, **args)
            candidate = _window_change(recipe, lane, edit, strong, fast)
            candidates.append((upper, changed, edit, candidate))
    elif space.get("action") == "first_tool_model":
        # This planner compares two declared roles. Refuse a third backend
        # rather than mislabelling it as a fast/strong change in the report.
        allowed = space.get("allowed_models")
        check(
            isinstance(allowed, list)
            and len(allowed) == 2
            and set(allowed) == {strong, fast},
            "the first-tool edit must declare the two planning models",
        )
        for backend in allowed:
            if _refs(lane) == [backend]:
                continue
            changed, edit = first_tool_edit(doc, raw_space, model=backend, **args)
            candidate = _descriptor(
                "requires_probe",
                "verify offered tools, enabled tool choice and zero "
                "visible assistant messages, then check the actual destination",
            )
            candidate["predicate_scope"] = "first_tool_guard"
            candidates.append((backend, changed, edit, candidate))
    else:
        raise ValueError("unsupported native edit space")
    result = []
    for value, changed, edit, candidate in candidates:
        candidate.update(
            id=f"edit-{source[:12]}-{value}",
            source="declared_edit_space",
            space_sha256=source,
            target_decision=lane["name"],
            edit=edit,
        )
        result.append((candidate, changed))
    return result
