"""Progressively disclose the canonical Router JSON Schema."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from typing import Any
from urllib.parse import quote_plus

SCHEMA_ENDPOINT = "/api/v1/config/schema"
SUPPORTED_VIEWS = ("full", "index", "section", "surface")


def schema_view(
    document: dict[str, Any],
    *,
    view: str = "index",
    path: str | None = None,
    surface_kind: str | None = None,
    surface_name: str | None = None,
    expanded: bool = False,
) -> dict[str, Any]:
    """Return one full, index, section, or routing-surface representation."""

    normalized_view = view.strip().lower()
    if normalized_view == "full":
        return deepcopy(document)
    if normalized_view == "index":
        return _index(document)
    if normalized_view == "section":
        return _section(document, path or "", expanded=expanded)
    if normalized_view == "surface":
        return _surface(document, surface_kind or "", surface_name or "")
    raise ValueError(
        f"unsupported schema view {view!r}; use {', '.join(SUPPORTED_VIEWS)}"
    )


def parse_surface_selector(selector: str) -> tuple[str, str]:
    """Parse KIND:NAME without constraining future surface names."""

    kind, separator, name = selector.partition(":")
    if not separator or not kind.strip() or not name.strip():
        raise ValueError("surface must use KIND:NAME, for example algorithm:static")
    return kind.strip(), name.strip()


def _index(document: dict[str, Any]) -> dict[str, Any]:
    extension = document.get("x-vllm-sr")
    if not isinstance(extension, dict):
        raise RuntimeError("Router config schema is missing x-vllm-sr")
    required = set(document.get("required", []))
    properties = document.get("properties", {})
    sections = []
    for name in sorted(properties):
        node = properties[name]
        resolved = _resolve(document, node)
        sections.append(
            {
                "path": name,
                "title": resolved.get("title") or _display_name(name),
                **(
                    {"description": resolved["description"]}
                    if resolved.get("description")
                    else {}
                ),
                "required": name in required,
                "href": f"{SCHEMA_ENDPOINT}?view=section&path={quote_plus(name)}",
            }
        )

    surfaces: dict[str, Any] = {}
    for kind, collection, name_key in (
        ("signal", "signals", "type"),
        ("algorithm", "algorithms", "type"),
        ("plugin", "plugins", "type"),
        ("projection", "projections", "collection"),
    ):
        names = sorted(
            entry[name_key]
            for entry in extension.get(collection, [])
            if isinstance(entry, dict) and isinstance(entry.get(name_key), str)
        )
        surfaces[kind] = {
            "count": len(names),
            "names": names,
            "href_template": (
                f"{SCHEMA_ENDPOINT}?view=surface&kind={kind}&name={{name}}"
            ),
        }

    return {
        "contract_version": extension.get("contract_version"),
        "config_version": extension.get("config_version"),
        "schema_id": document.get("$id"),
        "schema_etag": _schema_etag(document),
        "default_view": "index",
        "views": [
            {
                "name": "index",
                "description": "Compact section and routing-surface directory.",
                "href": f"{SCHEMA_ENDPOINT}?view=index",
            },
            {
                "name": "section",
                "description": (
                    "Compact field directory for one config path; request "
                    "expanded=true for its self-contained JSON Schema."
                ),
                "href": f"{SCHEMA_ENDPOINT}?view=section&path={{path}}",
            },
            {
                "name": "surface",
                "description": "One signal, algorithm, plugin, or projection contract.",
                "href": f"{SCHEMA_ENDPOINT}?view=surface&kind={{kind}}&name={{name}}",
            },
            {
                "name": "full",
                "description": "Complete canonical JSON Schema.",
                "href": f"{SCHEMA_ENDPOINT}?view=full",
            },
        ],
        "sections": sections,
        "surfaces": surfaces,
    }


def _section(document: dict[str, Any], path: str, *, expanded: bool) -> dict[str, Any]:
    segments = [segment.strip() for segment in path.replace("/", ".").split(".")]
    segments = [segment for segment in segments if segment]
    if not segments:
        raise ValueError("section view requires a non-empty path")
    node: dict[str, Any] = document
    traversed: list[str] = []
    for segment in segments:
        resolved = _resolve(document, node)
        if resolved.get("type") == "array":
            resolved = _resolve(document, resolved.get("items", {}))
        child = resolved.get("properties", {}).get(segment)
        traversed.append(segment)
        if not isinstance(child, dict):
            raise ValueError(
                f"unknown config section path {'.'.join(segments)!r} "
                f"at {'.'.join(traversed)!r}"
            )
        node = child
    normalized_path = ".".join(segments)
    if expanded:
        return _focused(
            document,
            node,
            {"view": "section", "path": normalized_path, "detail": "expanded"},
        )
    return _section_summary(document, node, normalized_path)


def _section_summary(
    document: dict[str, Any], node: dict[str, Any], path: str
) -> dict[str, Any]:
    root = _resolve(document, node)
    field_root = root
    if root.get("type") == "array":
        field_root = _resolve(document, root.get("items", {}))

    required = set(field_root.get("required", []))
    properties = field_root.get("properties", {})
    fields = []
    for name in sorted(properties):
        child = properties[name]
        if not isinstance(child, dict):
            continue
        resolved = _resolve(document, child)
        child_path = f"{path}.{name}"
        field = {
            "name": name,
            "path": child_path,
            "type": _schema_type(document, child),
            "required": name in required,
            "href": (f"{SCHEMA_ENDPOINT}?view=section&path={quote_plus(child_path)}"),
        }
        if description := resolved.get("description") or child.get("description"):
            field["description"] = description
        _copy_constraints(resolved, field)
        fields.append(field)

    summary = {
        "x-vllm-sr-view": {
            "view": "section",
            "path": path,
            "detail": "summary",
        },
        "shape": _schema_type(document, node),
        "title": field_root.get("title") or root.get("title") or _display_name(path),
        "fields": fields,
        "expanded_href": (
            f"{SCHEMA_ENDPOINT}?view=section&path={quote_plus(path)}&expanded=true"
        ),
    }
    if description := field_root.get("description") or root.get("description"):
        summary["description"] = description
    _copy_constraints(root, summary)
    return summary


def _schema_type(document: dict[str, Any], node: dict[str, Any]) -> str:
    resolved = _resolve(document, node)
    node_type = resolved.get("type")
    if isinstance(node_type, list):
        return " | ".join(str(value) for value in node_type)
    if node_type == "array":
        item = resolved.get("items")
        item_type = _schema_type(document, item) if isinstance(item, dict) else "value"
        return f"array<{item_type}>"
    if isinstance(node_type, str):
        return node_type
    for alternatives_key in ("oneOf", "anyOf"):
        alternatives = resolved.get(alternatives_key)
        if not isinstance(alternatives, list):
            continue
        types = []
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            alternative_type = _schema_type(document, alternative)
            if alternative_type not in types:
                types.append(alternative_type)
        if types:
            return " | ".join(types)
    if "const" in resolved:
        constant = resolved["const"]
        if constant is None:
            return "null"
        if isinstance(constant, bool):
            return "boolean"
        if isinstance(constant, str):
            return "string"
        if isinstance(constant, (int, float)):
            return "number"
    if isinstance(resolved.get("properties"), dict):
        return "object"
    return "value"


def _copy_constraints(source: dict[str, Any], target: dict[str, Any]) -> None:
    for key in ("const", "default", "enum", "minimum", "maximum"):
        if key in source:
            target[key] = deepcopy(source[key])


def _surface(document: dict[str, Any], kind: str, name: str) -> dict[str, Any]:
    normalized_kind = kind.strip().lower()
    catalogs = {
        "signal": ("signals", "type"),
        "signals": ("signals", "type"),
        "algorithm": ("algorithms", "type"),
        "algorithms": ("algorithms", "type"),
        "plugin": ("plugins", "type"),
        "plugins": ("plugins", "type"),
        "projection": ("projections", "collection"),
        "projections": ("projections", "collection"),
    }
    if normalized_kind not in catalogs:
        raise ValueError(
            f"unsupported surface kind {kind!r}; use signal, algorithm, plugin, or projection"
        )
    if not name.strip():
        raise ValueError("surface view requires a non-empty name")
    collection, name_key = catalogs[normalized_kind]
    extension = document.get("x-vllm-sr", {})
    for entry in extension.get(collection, []):
        if isinstance(entry, dict) and entry.get(name_key) == name.strip():
            reference = entry.get("schema_ref")
            node = (
                {"$ref": reference}
                if isinstance(reference, str)
                else {"type": "object"}
            )
            focused = _focused(
                document,
                node,
                {
                    "view": "surface",
                    "kind": normalized_kind.rstrip("s"),
                    "name": name.strip(),
                },
            )
            focused["x-vllm-sr-surface"] = deepcopy(entry)
            return focused
    raise ValueError(f"unknown {normalized_kind.rstrip('s')} surface {name!r}")


def _resolve(document: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    reference = node.get("$ref")
    if not reference:
        return node
    prefix = "#/$defs/"
    if not isinstance(reference, str) or not reference.startswith(prefix):
        raise ValueError(f"unsupported config schema reference {reference!r}")
    definition = document.get("$defs", {}).get(reference.removeprefix(prefix))
    if not isinstance(definition, dict):
        raise ValueError(f"missing config schema definition {reference!r}")
    return definition


def _focused(
    document: dict[str, Any], node: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    focused = deepcopy(node)
    if "$schema" in document:
        focused["$schema"] = document["$schema"]
    focused["x-vllm-sr-view"] = metadata
    definitions = _referenced_definitions(document, node)
    if definitions:
        focused["$defs"] = definitions
    return focused


def _referenced_definitions(
    document: dict[str, Any], node: dict[str, Any]
) -> dict[str, Any]:
    all_definitions = document.get("$defs", {})
    selected: dict[str, Any] = {}
    queue = _local_refs(node)
    while queue:
        name = queue.pop(0)
        if name in selected:
            continue
        definition = all_definitions.get(name)
        if definition is None:
            raise ValueError(f"missing config schema definition {name!r}")
        selected[name] = deepcopy(definition)
        queue.extend(_local_refs(definition))
    return selected


def _local_refs(value: Any) -> list[str]:
    references: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key == "$ref"
                and isinstance(child, str)
                and child.startswith("#/$defs/")
            ):
                references.append(child.removeprefix("#/$defs/"))
            else:
                references.extend(_local_refs(child))
    elif isinstance(value, list):
        for child in value:
            references.extend(_local_refs(child))
    return references


def _schema_etag(document: dict[str, Any]) -> str:
    payload = (json.dumps(document, indent=2) + "\n").encode()
    return f'"sha256:{sha256(payload).hexdigest()}"'


def _display_name(value: str) -> str:
    return value.replace("_", " ").capitalize()
