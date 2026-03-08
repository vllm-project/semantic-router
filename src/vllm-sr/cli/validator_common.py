"""Shared validation helpers for the CLI config validators."""


class ValidationError:
    """Validation error."""

    def __init__(self, message: str, field: str | None = None):
        self.message = message
        self.field = field

    def __str__(self) -> str:
        if self.field:
            return f"[{self.field}] {self.message}"
        return self.message


def iter_condition_nodes(conditions):
    """Depth-first traversal over recursive condition trees."""
    if not conditions:
        return
    for condition in conditions:
        yield condition
        children = getattr(condition, "conditions", None)
        if children:
            yield from iter_condition_nodes(children)


def iter_merged_condition_nodes(conditions):
    """Depth-first traversal over merged router condition dicts."""
    if not conditions:
        return
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        yield condition
        children = condition.get("conditions")
        if children:
            yield from iter_merged_condition_nodes(children)
