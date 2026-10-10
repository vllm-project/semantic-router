"""Bound decision trees before recursive schema and Pydantic processing."""

from typing import Any

DEFAULT_MAX_DEPTH = 16
DEFAULT_MAX_NODES = 256


def _field(value: Any, name: str, default: Any = None) -> Any:
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _items(value: Any) -> list:
    # Malformed containers belong to structural validation, not this preflight.
    return value if isinstance(value, list) else []


def validate_decision_rule_limits(document: Any) -> None:
    """Validate raw documents or models without recursively copying either."""
    global_config = _field(document, "global", _field(document, "global_"))
    settings = _field(_field(global_config, "router"), "decision_rule_limits") or {}
    limits = {}
    for name, default in (
        ("max_depth", DEFAULT_MAX_DEPTH),
        ("max_nodes", DEFAULT_MAX_NODES),
    ):
        value = _field(settings, name, default)
        if type(value) is not int or value < 1:
            raise ValueError(
                f"global.router.decision_rule_limits.{name} must be a positive integer"
            )
        limits[name] = value

    profiles = [("default", _field(document, "routing"))]
    profiles.extend(
        (_field(recipe, "name", ""), _field(recipe, "routing"))
        for recipe in _items(_field(document, "recipes"))
    )
    for recipe, routing in profiles:
        for decision in _items(_field(routing, "decisions")):
            root = _field(decision, "rules")
            # Rules keeps a convenience AND wrapper in memory for consumers,
            # but a bare leaf is serialized and counted in its authored form.
            if getattr(root, "_bare_leaf", False):
                root = root.conditions[0]
            try:
                _validate_tree(root, limits["max_depth"], limits["max_nodes"])
            except ValueError as error:
                raise ValueError(
                    f'routing recipe "{recipe}": decision "{_field(decision, "name", "")}": {error}'
                ) from error


def _validate_tree(root: Any, max_depth: int, max_nodes: int) -> None:
    # Cursor frames avoid an auxiliary allocation proportional to tree width.
    stack = [(iter(_items(_field(root, "conditions"))), "rules", 0)]
    count = 1
    while stack:
        children, path, index = stack[-1]
        try:
            child = next(children)
        except StopIteration:
            stack.pop()
            continue
        stack[-1] = (children, path, index + 1)
        child_path = f"{path}.conditions[{index}]"
        depth = len(stack) + 1
        count += 1
        if depth > max_depth:
            raise ValueError(
                f"{child_path}: depth {depth} exceeds max_depth={max_depth}"
            )
        if count > max_nodes:
            raise ValueError(
                f"{child_path}: node count {count} exceeds max_nodes={max_nodes}"
            )
        stack.append((iter(_items(_field(child, "conditions"))), child_path, 0))
