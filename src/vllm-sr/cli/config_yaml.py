"""YAML loading for Router config files that types token counts like the Router.

The Router decodes config.yaml with gopkg.in/yaml.v2 into an untyped map,
re-marshals it, and only then decodes the typed config, so a plain scalar
under ``routing.signals.context[].min_tokens`` or ``max_tokens`` reaches the
Router's ``TokenCount`` as yaml.v2's rendering of its implicit type (see
cli.goyaml_scalars). PyYAML applies its own implicit typing, which differs
from yaml.v2 for spellings such as ``1:30`` (a base-60 number to PyYAML, a
string to yaml.v2) or ``0o17`` and ``0X10`` (numbers to yaml.v2 only).
``vllm-sr config router`` forwards the original file, so validating PyYAML's
reading would accept a config the Router rejects or read one differently.

``safe_load_router_config`` composes the document with ``yaml.SafeLoader``
and, before construction, replaces each plain token-count scalar with the
text yaml.v2 would hand the Router. Quoted scalars are already literal on
both sides. Everything else loads exactly as ``yaml.safe_load`` would.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import yaml

from cli.goyaml_scalars import router_scalar_text

TOKEN_COUNT_KEYS = frozenset({"min_tokens", "max_tokens"})

_STR_TAG = "tag:yaml.org,2002:str"
_NULL_TAG = "tag:yaml.org,2002:null"


def safe_load_router_config(stream: str | bytes | IO[str] | IO[bytes]) -> Any:
    """Load one YAML document like ``yaml.safe_load``, with context band
    token counts typed as the Router's decoder types them."""
    loader = yaml.SafeLoader(stream)
    try:
        node = loader.get_single_node()
        if node is None:
            return None
        type_token_counts_like_router(node, loader)
        return loader.construct_document(node)
    finally:
        loader.dispose()


def type_token_counts_like_router(root: yaml.Node, loader: yaml.SafeLoader) -> None:
    """Rewrite every plain context band token count under *root* in place.

    A plain scalar whose tag came from implicit resolution becomes the string
    yaml.v2 would emit for it, or null when yaml.v2 reads it as null. Quoted
    scalars, block scalars, and explicit tags are left alone: yaml.v2 does not
    type those either, and PyYAML already constructs them literally. Mappings
    and sequences are left for the schema to reject on both sides.
    """
    for rule in _context_rule_nodes(root, loader):
        for key, value_node in _mapping_items(rule, loader):
            if key in TOKEN_COUNT_KEYS and _is_plain_implicit_scalar(
                value_node, loader
            ):
                text = router_scalar_text(value_node.value)
                value_node.tag = _NULL_TAG if text is None else _STR_TAG
                value_node.value = "" if text is None else text


def _is_plain_implicit_scalar(node: yaml.Node, loader: yaml.SafeLoader) -> bool:
    """A plain scalar carrying the tag implicit resolution gives it.

    An explicit tag that happens to match the implicit one (``!!int 8001``) is
    treated as implicit, which yaml.v2 resolves the same way.
    """
    return (
        isinstance(node, yaml.ScalarNode)
        and node.style is None
        and node.tag == loader.resolve(yaml.ScalarNode, node.value, (True, False))
    )


def _context_rule_nodes(
    root: yaml.Node, loader: yaml.SafeLoader
) -> Iterator[yaml.Node]:
    """Yield the context rule nodes of the default profile and every recipe."""
    yield from _sequence_items(_child(root, loader, "routing", "signals", "context"))
    for recipe in _sequence_items(_child(root, loader, "recipes")):
        yield from _sequence_items(
            _child(recipe, loader, "routing", "signals", "context")
        )


def _child(
    node: yaml.Node | None, loader: yaml.SafeLoader, *keys: str
) -> yaml.Node | None:
    """Descend *keys* through mapping nodes; a duplicate key keeps its last value
    as ``construct_mapping`` does."""
    for key in keys:
        found = None
        for candidate, value_node in _mapping_items(node, loader):
            if candidate == key:
                found = value_node
        node = found
    return node


def _mapping_items(
    node: yaml.Node | None, loader: yaml.SafeLoader
) -> Iterator[tuple[str, yaml.Node]]:
    if not isinstance(node, yaml.MappingNode):
        return
    # Fold ``<<`` merge keys in place, as construct_mapping does later, so a
    # token count inherited through an anchor is typed too.
    loader.flatten_mapping(node)
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode):
            yield key_node.value, value_node


def _sequence_items(node: yaml.Node | None) -> Iterator[yaml.Node]:
    if isinstance(node, yaml.SequenceNode):
        yield from node.value
