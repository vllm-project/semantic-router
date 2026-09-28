"""Licence policy for model cards and package metadata (fail closed).

Card roster: a comparator appears on a card only with a known licence that is
neither non-commercial (CC BY-NC) nor research-only; unknown licences and
internal-only controls are excluded, with the reason recorded. Package
metadata says ``apache-2.0`` only when every upstream component of the direct
weight lineage carries an Apache-compatible licence; otherwise ``other`` with a
component table in ``LICENSING.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

APACHE_COMPATIBLE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause"}
CARD_ALLOWED = APACHE_COMPATIBLE | {
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "openrail",
    "llama3",
    "gemma",
}
RESEARCH_ONLY_NAMES = {"research-and-demo", "research-only", "non-commercial"}
OWN_MODELS = {
    "llm-semantic-router/Decision-1.0-Kai-0.6B": "apache-2.0",
    "llm-semantic-router/Decision-1.0-Lex-0.6B": "apache-2.0",
    "llm-semantic-router/Decision-1.0-Eos-0.8B": "apache-2.0",
    "llm-semantic-router/Decision-1.0-Sol-2B": "apache-2.0",
    "llm-semantic-router/Decision-1.0-Nox-4B": "apache-2.0",
    "llm-semantic-router/Decision-1.0-Lux-9B": "apache-2.0",
}
# Internal controls that must never appear on a card (coordinator 2026-09-28).
INTERNAL_ONLY_MODELS = {
    "research/qwen3-06b-official-full466",
    "llm-semantic-router/DEV2.0-0.6B@5380e01e",
}
INTERNAL_ONLY_LABELS = {"DEV2.0-0.6B (private)"}


def load_roster(path: Path) -> dict[str, dict[str, Any]]:
    roster = json.loads(Path(path).read_text(encoding="utf-8"))
    return {peer["repo"]: peer for peer in roster["peers"]}


def peer_licence(peer: dict[str, Any]) -> tuple[str | None, str | None]:
    licence = peer.get("license") or {}
    return licence.get("card"), licence.get("card_license_name")


def card_eligibility(
    entry: dict[str, Any], roster: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Decide whether one same-panel report may appear on a public-facing card."""
    repo, family, label = entry.get("repo_id"), entry.get("family"), entry.get("label")
    if label in INTERNAL_ONLY_LABELS or repo in INTERNAL_ONLY_MODELS:
        return {"eligible": False, "licence": None, "reason": "internal-only control"}
    if entry.get("candidate"):
        return {"eligible": True, "licence": "package", "reason": "release candidate"}
    if family == "decision2":
        return {
            "eligible": False,
            "licence": None,
            "reason": "unreleased Decision 2.0 model",
        }
    if repo in OWN_MODELS:
        return {
            "eligible": True,
            "licence": OWN_MODELS[repo],
            "reason": "own Decision 1.0",
        }
    peer = roster.get(repo or "")
    if peer is None:
        return {
            "eligible": False,
            "licence": None,
            "reason": "licence unknown (not in the pinned roster)",
        }
    card, name = peer_licence(peer)
    card = (card or "").lower()
    if card.startswith("cc-by-nc") or "-nc" in card:
        return {"eligible": False, "licence": card, "reason": "non-commercial licence"}
    if (name or "").lower() in RESEARCH_ONLY_NAMES or (card == "other" and not name):
        return {
            "eligible": False,
            "licence": name or card,
            "reason": "research-only or unnamed licence",
        }
    if card not in CARD_ALLOWED:
        return {
            "eligible": False,
            "licence": card or None,
            "reason": "licence not on the card allowlist",
        }
    return {"eligible": True, "licence": card, "reason": "roster licence"}


def package_licence(components: list[dict[str, Any]]) -> dict[str, Any]:
    """Card metadata licence from the direct weight lineage components."""
    if not components:
        raise ValueError("Declare every upstream component of the weight lineage")
    for component in components:
        if not component.get("component") or not component.get("licence"):
            raise ValueError("Each lineage component needs a name and a licence")
    compatible = all(c["licence"].lower() in APACHE_COMPATIBLE for c in components)
    return {
        "spdx": "apache-2.0" if compatible else "other",
        "license_name": None if compatible else "decision-2.0-component-licences",
        "apache_compatible": compatible,
        "components": components,
    }
