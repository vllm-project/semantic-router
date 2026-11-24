"""Edit the upper bound of an existing v0.5 tool-result count window."""

import yaml

if __package__:
    from .loop_evidence import check, require_isolated, sha
else:
    from loop_evidence import check, require_isolated, sha


def space_mapping(raw):
    # This declaration is flat. Reject duplicate keys before SafeLoader silently
    # picks a value from two incompatible target or domain declarations.
    node = yaml.compose(raw, Loader=yaml.SafeLoader)
    check(isinstance(node, yaml.MappingNode), "space must be a mapping")
    keys = [key.value for key, _ in node.value]
    check(all(isinstance(k, str) for k in keys), "space keys must be strings")
    check(len(keys) == len(set(keys)), "duplicate field in space")
    return yaml.safe_load(raw)


def load_space(raw):
    space = space_mapping(raw)
    names = ("public_model", "recipe", "decision", "lower_signal", "upper_signal")
    fields = {"schema", "action", "base_config_sha256", "allowed_upper", *names}
    check(
        set(space) == fields,
        f"unexpected or missing space fields: {set(space) ^ fields}",
    )
    check(space["schema"] == "recipe-lab-window-space-v1", "unsupported space schema")
    check(space["action"] == "tool_result_window_upper", "unsupported edit action")
    sha(space["base_config_sha256"])
    for key in names:
        check(isinstance(space[key], str) and space[key].strip(), f"invalid {key}")
    check(
        space["lower_signal"] != space["upper_signal"],
        "window needs two distinct signals",
    )
    values = space["allowed_upper"]
    # The Router stores gte in float64. Both upper and upper+1 must be exact.
    check(
        isinstance(values, list)
        and values
        and all(type(v) is int and 0 <= v < 2**53 for v in values),
        "allowed_upper must contain nonnegative exactly representable count bounds",
    )
    check(len(values) == len(set(values)), "duplicate allowed_upper value")
    return space


def named(rows, name, kind):
    check(isinstance(rows, list), f"missing {kind} list")
    matches = [r for r in rows if isinstance(r, dict) and r.get("name") == name]
    check(len(matches) == 1, f"expected exactly one {kind} named {name!r}")
    return matches[0]


def threshold(rule):
    name = rule["name"]
    check(
        rule.get("feature")
        == {"type": "count", "source": {"type": "assistant_tool_cycle"}},
        f"{name}: expected count of assistant_tool_cycle",
    )
    predicate = rule.get("predicate")
    check(
        isinstance(predicate, dict) and set(predicate) == {"gte"},
        f"{name}: expected a single gte predicate",
    )
    value = predicate["gte"]
    check(
        type(value) in (int, float) and 0 <= value <= 2**53 and int(value) == value,
        f"{name}: expected an exact nonnegative integer gte",
    )
    return int(value)


def reference_count(tree, name):
    if isinstance(tree, list):
        return sum(reference_count(item, name) for item in tree)
    if isinstance(tree, dict):
        own = tree.get("type") == "conversation" and tree.get("name") == name
        return int(own) + sum(reference_count(item, name) for item in tree.values())
    return 0


def window_bounds(doc, space):
    """Resolve the two real signal predicates and the direct AND / NOT consumer."""
    recipe = require_isolated(doc, space["public_model"])
    check(recipe["name"] == space["recipe"], "space points to another recipe")
    signals = recipe.get("signals")
    check(isinstance(signals, dict), "window recipe requires signals")
    rules = signals.get("conversation")
    lower_name, upper_name = space["lower_signal"], space["upper_signal"]
    lower = threshold(named(rules, lower_name, "conversation signal"))
    upper = threshold(named(rules, upper_name, "conversation signal")) - 1
    check(lower <= upper, "inverted or empty tool-result window")
    decision = named(recipe["decisions"], space["decision"], "decision")
    rules = decision.get("rules")
    check(isinstance(rules, dict), "window decision requires rules")
    conditions = rules.get("conditions", [])
    check(
        rules.get("operator") == "AND" and isinstance(conditions, list),
        "window decision must conjoin its bounds",
    )
    lower_ref = {"type": "conversation", "name": lower_name}
    upper_ref = {"type": "conversation", "name": upper_name}
    check(
        lower_ref in conditions
        and {"operator": "NOT", "conditions": [upper_ref]} in conditions,
        "window requires a positive lower signal and a negated upper signal",
    )
    check(
        reference_count(recipe, upper_name) == 1,
        "upper signal is shared by other rules",
    )
    return lower, upper


def apply_window(doc, space, *, public_model, config_sha256, upper):
    check(
        public_model == space["public_model"],
        "space public model differs from the parent",
    )
    check(
        config_sha256 == space["base_config_sha256"],
        "space baseline fingerprint mismatch",
    )
    lower, before = window_bounds(doc, space)
    check(
        type(upper) is int and upper in space["allowed_upper"],
        "upper outside declared space",
    )
    check(
        upper >= lower and upper != before,
        "upper must change and keep the window nonempty",
    )

    # Copy just the path to the field. YAML anchors can share dictionaries with
    # other signals or recipes; mutating one in place would edit those too.
    recipe = require_isolated(doc, public_model)
    signals = recipe["signals"]
    rules = signals["conversation"]
    signal = named(rules, space["upper_signal"], "conversation signal")
    changed = {**signal, "predicate": {"gte": upper + 1}}
    rules = [changed if r is signal else r for r in rules]
    replacement = {**recipe, "signals": {**signals, "conversation": rules}}
    result = {
        **doc,
        "recipes": [replacement if r is recipe else r for r in doc["recipes"]],
    }
    return result, {
        "action": space["action"],
        "decision": space["decision"],
        "lower_signal": space["lower_signal"],
        "upper_signal": space["upper_signal"],
        "lower": lower,
        "before_upper": before,
        "after_upper": upper,
        "before_gte": before + 1,
        "after_gte": upper + 1,
    }
