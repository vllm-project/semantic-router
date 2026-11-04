"""Product layers and reviewed execution budgets for hosted CI workers."""

from __future__ import annotations

LAYERS = ("source", "router", "runtime", "product", "integrations", "performance")
SCOPES = frozenset(
    {"affected", "models", "storage", "product", "e2e", "operator", "performance"}
)


def approved(record: dict, authorization: dict) -> bool:
    if not authorization.get("authorized"):
        return False
    scopes = set(authorization.get("scopes", []))
    if scopes - SCOPES:
        raise ValueError("unknown manual CI scope")
    if authorization.get("full") or "affected" in scopes:
        return True
    checks = {
        "models": record["layer"] == "runtime" and record["executor"] != "storage",
        "storage": record["executor"] == "storage",
        "product": record["layer"] == "product",
        "e2e": record["executor"] == "e2e",
        "operator": record["executor"] == "operator",
        "performance": record["executor"] == "performance",
    }
    return any(checks[scope] for scope in scopes)


def partition(
    ids: list[str], records: dict, authorization: dict
) -> tuple[list[str], list[str]]:
    execute, deferred = [], []
    for name in ids:
        record = records[name]
        target = (
            execute
            if record["execution"] == "automatic" or approved(record, authorization)
            else deferred
        )
        target.append(name)
    return execute, deferred
