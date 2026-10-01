"""Jev Decision Index values for the model cards (stdlib only).

The values live in one private input file (schema ``dev2-card-index/1``) that the
card build reads at build time; they appear only on the private Hugging Face cards
and never in a commit, record, gist or test fixture with real values. Committed
receipts carry the file's SHA-256 only.

File layout::

    {"schema": "dev2-card-index/1", "edition": "0.2.1", "footnote": "...",
     "family":    [{"tier", "name", "parameters", "balanced_skill", "areas", "model_sha256"}, ...],
     "decision1": [{"name", "tier" (or null), "parameters", "balanced_skill", "areas"}, ...],
     "entrants":  [{"parameters", "balanced_skill"}, ...]}

``areas`` maps every id of ``AREAS`` to a balanced skill (x100). ``family`` holds
the six Decision 2.0 tiers, scored on exactly the released weights
(``model_sha256``); ``decision1`` and ``entrants`` come from one public board
snapshot.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from v2.release.layout import CODENAMES, TIERS, sha_file

SCHEMA = "dev2-card-index/1"
AREAS = (
    ("knowledge", "Knowledge"),
    ("language", "Language"),
    ("retrieval", "Retrieval"),
    ("tools", "Tools"),
    ("arts", "Arts"),
)
# The Decision 1.0 model of each size slot; the 27B slot has none and compares with the 9B of its own family.
COUNTERPART_1_0 = {
    "0.6B": "Decision 1.0 Kai",
    "0.8B": "Decision 1.0 Eos",
    "2B": "Decision 1.0 Sol",
    "4B": "Decision 1.0 Nox",
    "9B": "Decision 1.0 Lux",
}
FAMILY_COMPARISON = {"27B": "9B"}


def _number(value: Any, where: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{where}: not a finite number")
    return float(value)


def _areas(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {a for a, _ in AREAS}:
        raise ValueError(f"{where}: areas must be exactly {[a for a, _ in AREAS]}")
    return {a: _number(value[a], f"{where}.{a}") for a, _ in AREAS}


def _point(value: Any, where: str, areas: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where}: not an object")
    parameters = value.get("parameters")
    if type(parameters) is not int or parameters < 1:
        raise ValueError(f"{where}: parameters must be a positive integer")
    point = {
        **value,
        "parameters": parameters,
        "balanced_skill": _number(value.get("balanced_skill"), where),
    }
    if areas:
        point["areas"] = _areas(value.get("areas"), where)
    return point


def load(path: Path) -> dict[str, Any]:
    """Read and validate the private Index input; adds its SHA-256 as ``sha256``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file")
    if not data.get("footnote") or not data.get("edition"):
        raise ValueError(f"{path}: needs edition and footnote")
    family = [
        _point(p, f"family[{i}]", True) for i, p in enumerate(data.get("family") or [])
    ]
    if sorted(p.get("tier") for p in family) != sorted(TIERS):
        raise ValueError(f"family must hold exactly the tiers {sorted(TIERS)}")
    for point in family:
        expected = f"Decision-2.0-{CODENAMES[point['tier']]}-"
        if not str(point.get("name", "")).startswith(expected):
            raise ValueError(f"family {point['tier']}: name must start with {expected}")
        if (
            not isinstance(point.get("model_sha256"), str)
            or len(point["model_sha256"]) != 64
        ):
            raise ValueError(f"family {point['tier']}: needs the scored model_sha256")
    decision1 = [
        _point(p, f"decision1[{i}]", True)
        for i, p in enumerate(data.get("decision1") or [])
    ]
    names = {p.get("name") for p in decision1}
    missing = set(COUNTERPART_1_0.values()) - names
    if missing:
        raise ValueError(f"decision1 lacks {sorted(missing)}")
    entrants = [
        _point(p, f"entrants[{i}]", False)
        for i, p in enumerate(data.get("entrants") or [])
    ]
    if not entrants:
        raise ValueError("entrants must not be empty")
    return {
        **data,
        "family": family,
        "decision1": decision1,
        "entrants": entrants,
        "sha256": sha_file(Path(path)),
    }


def view(index: dict[str, Any], tier: str, model_sha256: str) -> dict[str, Any]:
    """This tier's point, its comparison point and the family, checked against the package weights."""
    family = {p["tier"]: p for p in index["family"]}
    own = family[tier]
    if own["model_sha256"] != model_sha256:
        raise ValueError(
            f"Index values for {tier} were scored on other weights ({own['model_sha256'][:12]})"
        )
    if tier in COUNTERPART_1_0:
        name = COUNTERPART_1_0[tier]
        compare = next(p for p in index["decision1"] if p["name"] == name)
        kind = "decision1"
    else:
        compare = family[FAMILY_COMPARISON[tier]]
        kind = "family"
    return {
        "tier": tier,
        "own": own,
        "compare": compare,
        "compare_kind": kind,
        "compare_name": compare["name"],
        "delta": own["balanced_skill"] - compare["balanced_skill"],
        "family": [family[t] for t in TIERS],
        "decision1": index["decision1"],
        "entrants": index["entrants"],
        "footnote": index["footnote"],
        "edition": index["edition"],
        "sha256": index["sha256"],
    }
