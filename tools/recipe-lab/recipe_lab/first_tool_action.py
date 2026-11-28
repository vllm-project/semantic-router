"""Edit one v0.5 model reference for tools with no visible assistant history."""

if __package__:
    from .loop_evidence import check, require_isolated, sha
    from .window_action import named, space_mapping
else:
    from loop_evidence import check, require_isolated, sha
    from window_action import named, space_mapping


def first_tool_edit(doc, raw, *, public_model, config_sha256, model):
    space = space_mapping(raw)
    identity = ("public_model", "recipe", "decision")
    signals = ("tools_signal", "disabled_signal", "assistant_signal")
    check(
        set(space)
        == {
            "schema",
            "action",
            "base_config_sha256",
            "allowed_models",
            *identity,
            *signals,
        }
        and space["schema"] == "recipe-lab-first-tool-space-v1"
        and space["action"] == "first_tool_model",
        "expected a first-tool model action declaration",
    )
    check(
        sha(space["base_config_sha256"]) == config_sha256
        and space["public_model"] == public_model,
        "first-tool space differs from the parent",
    )
    check(
        all(
            isinstance(space[k], str) and space[k].strip()
            for k in (*identity, *signals)
        )
        and len({space[k] for k in signals}) == 3,
        "expected named targets and three distinct conversation signals",
    )
    allowed = space["allowed_models"]
    check(
        isinstance(allowed, list)
        and all(isinstance(v, str) and v.strip() for v in allowed)
        and len(allowed) >= 2
        and len(set(allowed)) == len(allowed)
        and model in allowed,
        "choose a model from the declared first-tool space",
    )
    check(isinstance(doc, dict), "expected a v0.5 configuration mapping")
    recipe = require_isolated(doc, public_model)
    check(
        recipe["name"] == space["recipe"], "first-tool space points to another recipe"
    )
    requirements = recipe.get("candidate_requirements")
    check(
        isinstance(requirements, dict)
        and requirements.get("capabilities") == "declared"
        and requirements.get("context") == "known_limits",
        "the recipe must enforce declared capabilities and known context limits",
    )
    providers = doc.get("providers")
    check(isinstance(providers, dict), "expected a provider registry")
    for backend in allowed:
        entry = named(providers.get("models"), backend, "provider model")
        capabilities = entry.get("capabilities")
        check(
            isinstance(capabilities, list)
            and all(isinstance(tag, str) for tag in capabilities),
            f"{backend}: expected model capability names",
        )
        # Match the Router's ModelCapabilities aliases for these two demands.
        tags = {tag.strip().lower() for tag in capabilities}
        check(
            tags.intersection(("chat", "text"))
            and tags.intersection(("tool_use", "tools")),
            f"{backend}: text and tools capabilities required",
        )
        check(
            all(
                type(entry.get(k)) is int and entry[k] > 0
                for k in ("context_window_size", "max_output_tokens")
            ),
            f"{backend}: positive integer context and output limits required",
        )
    decision = named(recipe["decisions"], space["decision"], "decision")
    refs = decision.get("modelRefs")
    check(
        decision.get("algorithm") == {"type": "static"}
        and isinstance(refs, list)
        and len(refs) == 1
        and isinstance(refs[0], dict)
        and set(refs[0]) == {"model"},
        "first-tool action requires one plain static model reference",
    )
    before = refs[0]["model"]
    check(
        before in allowed and before != model,
        "first-tool action must change a declared model",
    )
    verify_guards(recipe, decision, space)
    changed = {**decision, "modelRefs": [{"model": model}]}
    replacement = {
        **recipe,
        "decisions": [changed if d is decision else d for d in recipe["decisions"]],
    }
    return {
        **doc,
        "recipes": [replacement if r is recipe else r for r in doc["recipes"]],
    }, {
        "action": space["action"],
        "decision": space["decision"],
        "before_model": before,
        "after_model": model,
        "request_scope": "tools offered, tool use enabled, zero assistant messages visible to the Router",
    }


def verify_guards(recipe, decision, space):
    sources = (
        ("tools_signal", {"type": "tool_definition"}, True),
        ("disabled_signal", {"type": "tool_choice_none"}, False),
        ("assistant_signal", {"type": "message", "role": "assistant"}, True),
    )
    rules = decision.get("rules")
    check(isinstance(rules, dict), "expected a first-tool decision rule mapping")
    conditions = rules.get("conditions")
    check(
        set(rules) == {"operator", "on_unknown", "conditions"}
        and rules.get("operator") == "AND"
        and rules.get("on_unknown") == "no_match"
        and isinstance(conditions, list)
        and len(conditions) == len(sources),
        "expected the three first-tool guards in an AND that declines unknown inputs",
    )
    signals = recipe.get("signals")
    check(isinstance(signals, dict), "first-tool recipe requires conversation signals")
    for key, source, counted in sources:
        name = space[key]
        signal = named(signals.get("conversation"), name, "conversation signal")
        feature = {"type": "count" if counted else "exists", "source": source}
        expected = {"name": name, "feature": feature}
        if counted:
            expected["predicate"] = {"gte": 1}
        check(
            {k: v for k, v in signal.items() if k != "description"} == expected
            and (not counted or type(signal["predicate"]["gte"]) in (int, float)),
            f"{name}: incompatible first-tool signal",
        )
        reference = {"type": "conversation", "name": name}
        condition = (
            reference
            if key == "tools_signal"
            else {"operator": "NOT", "conditions": [reference]}
        )
        check(
            conditions.count(condition) == 1,
            f"{name}: missing or repeated first-tool guard",
        )
