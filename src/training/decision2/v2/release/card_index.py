"""Jev Decision Index values for the model cards (stdlib only).

The values live in one private input file (schema ``dev2-card-index/1``) that the
card build reads at build time; they appear only on the private Hugging Face cards
and never in a commit, record, gist or test fixture with real values. Committed
receipts carry the file's SHA-256 only.

File layout::

    {"schema": "dev2-card-index/1", "edition": "0.2.1", "snapshot": "YYYY-MM-DD",
     "footnote": FOOTNOTE.format(edition, snapshot),
     "family":    [{"tier", "name", "parameters", "parameters_basis", "loaded_parameters",
                    "balanced_skill", "areas", "model_sha256"}, ...],
     "decision1": [{"name", "tier" (or null), "parameters", "balanced_skill", "areas"}, ...],
     "entrants":  [{"parameters", "balanced_skill"}, ...]}

``areas`` maps every id of ``AREAS`` to a balanced skill (x100). ``family`` holds
the six Decision 2.0 tiers, scored on exactly the released weights
(``model_sha256``); ``decision1`` and ``entrants`` come from one public board
snapshot. Every ``parameters`` follows the board's served-parameter convention:
a family point takes the board's count for its own base (``BOARD_BASES``), so
the size axis compares like with like; ``loaded_parameters`` is the package's
own count, which the card's at-a-glance table shows.

Build the file from the kit runs and the board snapshot (private inputs)::

    python -m v2.release.card_index --runs runs.json ... --board index-latest.json \
        --snapshot 2026-09-28 --manifests DIR --out decision-index-card.json

``DIR`` holds ``<tier>/MODEL_MANIFEST.json`` of the scored packages.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from v2.release.layout import CODENAMES, TIERS, sha_file

SCHEMA = "dev2-card-index/1"
FOOTNOTE = (
    "Decision 2.0: independent reproduction with the official {edition} kit on the "
    "released weights; others: public board snapshot, {snapshot}. Training data "
    "audited at row level against all Index test items."
)
PARAMETERS_BASIS = "board-served"
# The base of each size slot under the names the board uses for it (base and instruct share the served count).
BOARD_BASES = {
    "0.6B": ("Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"),
    "0.8B": ("Qwen/Qwen3.5-0.8B-Base", "Qwen/Qwen3.5-0.8B"),
    "2B": ("Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-2B"),
    "4B": ("Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B"),
    "9B": ("Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B"),
    "27B": ("Qwen/Qwen3.8-27B",),
}
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
DECISION1_TIERS = {
    **{v: k for k, v in COUNTERPART_1_0.items()},
    "Decision 1.0 Lex": None,
}


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
    if not data.get("edition") or not data.get("snapshot"):
        raise ValueError(f"{path}: needs edition and snapshot")
    footnote = FOOTNOTE.format(edition=data["edition"], snapshot=data["snapshot"])
    if data.get("footnote") != footnote:
        raise ValueError(f"{path}: footnote must be {footnote!r}")
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
        loaded = point.get("loaded_parameters")
        if point.get("parameters_basis") != PARAMETERS_BASIS or type(loaded) is not int:
            raise ValueError(
                f"family {point['tier']}: parameters must follow the board's served "
                f"convention ({PARAMETERS_BASIS}) next to loaded_parameters"
            )
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


def board_parameters(board: dict[str, Any], tier: str) -> int:
    """The board's served-parameter count for this tier's base; all its same-base entrants must agree."""
    counts = {
        m["meta"]["served_params"]
        for m in board["models"]
        if m["meta"].get("base_model") in BOARD_BASES[tier]
    }
    if len(counts) != 1:
        raise ValueError(
            f"{tier}: the board's same-base entrants ({BOARD_BASES[tier]}) give "
            f"{sorted(counts)} served parameters, not one count"
        )
    return counts.pop()


def _board_areas(model: dict[str, Any]) -> dict[str, float]:
    skills = {c["id"]: c["skill"] for c in model["categories"]}
    return {a: round(100 * skills[a], 2) for a, _ in AREAS}


def build(
    runs: list[dict[str, Any]],
    board: dict[str, Any],
    snapshot: str,
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """The Index input from one kit run per released tier (matched by model_sha256) and the board snapshot."""
    family, editions = [], set()
    for tier in TIERS:
        manifest = manifests[tier]
        model_sha256 = manifest["identity"]["model_sha256"]
        scored = [r for r in runs if r["model_sha256"] == model_sha256]
        if len(scored) != 1:
            raise ValueError(
                f"{tier}: {len(scored)} kit runs scored the weights {model_sha256[:12]}"
            )
        run = scored[0]
        editions.add(run["edition"])
        family.append(
            {
                "tier": tier,
                "name": f"Decision-2.0-{CODENAMES[tier]}-{tier}",
                "parameters": board_parameters(board, tier),
                "parameters_basis": PARAMETERS_BASIS,
                "loaded_parameters": manifest["parameters"]["loaded"],
                "balanced_skill": run["balanced_skill"],
                "areas": run["areas"],
                "model_sha256": model_sha256,
                "kit_index_sha256": run["index_sha256"],
            }
        )
    if len(editions) != 1:
        raise ValueError(f"mixed kit editions {sorted(editions)}")
    edition = editions.pop()
    decision1, entrants = [], []
    for model in board["models"]:
        point = {
            "parameters": model["meta"]["served_params"],
            "balanced_skill": model["scores"]["balanced_skill"],
        }
        if model["name"] in DECISION1_TIERS:
            decision1.append(
                {
                    "name": model["name"],
                    "tier": DECISION1_TIERS[model["name"]],
                    **point,
                    "areas": _board_areas(model),
                }
            )
        elif model["name"].startswith("Decision"):
            raise ValueError(f"unexpected own model on the board: {model['name']}")
        else:
            entrants.append({"name": model["name"], **point})
    return {
        "schema": SCHEMA,
        "edition": edition,
        "snapshot": snapshot,
        "board_generated_utc": board["generated_utc"],
        "footnote": FOOTNOTE.format(edition=edition, snapshot=snapshot),
        "family": family,
        "decision1": sorted(decision1, key=lambda p: p["parameters"]),
        "entrants": sorted(entrants, key=lambda p: p["parameters"]),
    }


def _runs(path: Path) -> list[dict[str, Any]]:
    """A run file holds one JSON object of runs keyed by name, possibly after other output lines."""
    lines = [l for l in path.read_text().splitlines() if l.startswith("{")]
    return list(json.loads(lines[-1]).values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the private Index input of the cards."
    )
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--board", type=Path, required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--manifests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    runs = [r for path in args.runs for r in _runs(path)]
    manifests = {
        tier: json.loads((args.manifests / tier / "MODEL_MANIFEST.json").read_text())
        for tier in TIERS
    }
    board = json.loads(args.board.read_text())
    out = build(runs, board, args.snapshot, manifests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    loaded = load(args.out)
    print(json.dumps({"out": str(args.out), "sha256": loaded["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
