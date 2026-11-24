"""Y2: the classification sections the observe seed and candidates share.

Cold-start observes the router's own domain labels under a seed config, then
generates candidates whose decisions route by those labels. The premise (the
cold-start contract, §3.2-4) is that the classification machinery behaves the
same way in both configs: if the seed and a candidate carry different
classifier/prompt-guard/embedding settings, the labels the candidates route by
can silently diverge from the labels the profile was grouped by.

The same-source rule: both sides derive their classification-relevant
``global.model_catalog`` sections from ONE explicit classification source
config (the maintained ``config/config.yaml`` in the cold-start flow) -

- ``deployments`` and ``system``: always the C2 seed's own pinning (the
  candidates inherit it; a scaffold copy must never override it);
- ``modules.classifier`` / ``modules.prompt_guard`` / ``embeddings``: read
  from the classification source by ``observe-labels`` (into the seed) and by
  ``bootstrap`` (into every candidate, overriding the scaffold's copies).

``classification_projection`` names exactly the sections that decide the
observed labels; its canonical-JSON sha256 is written into the bootstrap
provenance sidecar and verified by ``bundle``, so a packaged config cannot
carry classification settings other than the ones the evidence was produced
with.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .contract import ContractError

# The model_catalog sections observe-labels injects into the seed config and
# bootstrap injects into every candidate (overriding the scaffold's copies).
INJECTED_SECTIONS = ("modules", "embeddings")


def classification_projection(config: dict[str, Any]) -> dict[str, Any]:
    """The classification-relevant sections of one config (Y2).

    Exactly the five parts the review named: ``deployments``, ``system``,
    ``modules.classifier``, ``modules.prompt_guard`` and ``embeddings``.
    Missing sections project as null so a config that carries none of them
    still has a stable identity.
    """
    catalog = (config.get("global") or {}).get("model_catalog") or {}
    modules = catalog.get("modules") or {}
    return {
        "deployments": catalog.get("deployments"),
        "system": catalog.get("system"),
        "classifier": modules.get("classifier"),
        "prompt_guard": modules.get("prompt_guard"),
        "embeddings": catalog.get("embeddings"),
    }


def classification_projection_hash(config: dict[str, Any]) -> str:
    """Canonical-JSON sha256 of the classification projection."""
    payload = json.dumps(
        classification_projection(config), sort_keys=True, ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_classification_source(path: str | Path) -> dict[str, Any]:
    """Read the classification sections the seed and candidates must share.

    The source must declare BOTH ``global.model_catalog.modules`` and
    ``global.model_catalog.embeddings`` mappings: an incomplete source would
    silently reintroduce the default-vs-scaffold divergence Y2 closes.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read the classification source: {exc}") from exc
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ContractError(f"{path} is not a YAML mapping")
    catalog = (config.get("global") or {}).get("model_catalog") or {}
    sections: dict[str, Any] = {}
    for key in INJECTED_SECTIONS:
        value = catalog.get(key)
        if not isinstance(value, dict):
            raise ContractError(
                f"the classification source {path} has no global.model_catalog.{key} "
                "mapping: the observe seed and every bootstrap candidate must "
                "derive their classification sections from the same source (Y2)"
            )
        sections[key] = value
    return sections
