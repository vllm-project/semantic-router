"""YAML scalar rules shared by every user-authored v0.3 document loader."""

from __future__ import annotations

import re
from typing import Any

import yaml


class YAML12SafeLoader(yaml.SafeLoader):
    """Safe loader with YAML 1.2 boolean spelling.

    PyYAML otherwise applies the YAML 1.1 resolver and turns mapping keys such
    as ``on`` into booleans. The public v0.3 schema uses ``on`` for retry and
    fallback conditions, so all authoring paths must agree on YAML 1.2 here.
    """


YAML12SafeLoader.yaml_implicit_resolvers = {
    first: [
        resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
YAML12SafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """Construct one mapping while rejecting duplicate or non-scalar keys."""

    keys: set[object] = set()
    for key_node, _ in node.value:
        if key_node.value == "<<" or key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in keys
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found a non-scalar mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        keys.add(key)
    loader.flatten_mapping(node)
    return yaml.constructor.BaseConstructor.construct_mapping(loader, node, deep=deep)


YAML12SafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_unique_mapping,
)


def load_yaml(source: Any) -> Any:
    """Load one trusted-size YAML value with canonical scalar and key rules."""

    return yaml.load(source, Loader=YAML12SafeLoader)
