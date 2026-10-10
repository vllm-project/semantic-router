"""Assemble the private card input (``d25-card-input/1``) of a Decision 2.5 card (stdlib only).

    python -m d25.vega.release.card_input --board index.json --snapshot 2026-10-07 --roster roster.json \
        --manifest MODEL_MANIFEST.json --own own.json [--latency latency.json] --teachers teachers.json \
        --out card-input.json

- ``--board``: the board's ``data/index.json`` (Space multimodalart/jev-decision-index), 0.3 format; re-download
  it right before a release so the peers are the live board's.
- ``--roster``: ``{"<engine>": {"name", "repo", "licence"}}`` for named peers; a peer appears on the card only
  with a card-eligible licence (fail closed, as for 2.0), the anonymous size chart shows every entrant.
- ``--own``: ``{"status": "estimate", "full", "public", "same_skill", "new_domain", "uncertainty", "areas"}`` with
  public area skills x100 from the kit run (``index.json`` areas), or ``{"status": "board", "engine": ...}`` once
  the board has scored the model (all values then come from ``--board``).
- ``--latency``: ``compare.py latency`` output of the RTX PRO 6000 job; ``--teachers``: ``[{"model", "licence", "use"}]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from d25.vega.release.card import CARD_LICENCES, SCHEMA

AREAS = ("knowledge", "language", "retrieval", "tools", "arts")
PREVIOUS_ENGINE = "decision-2.0-vega-27b"
FAMILY_PREFIX = "decision-2.0-"
OWN_PREFIX = "decision-2.5-"


def decision_name(engine: str) -> str:
    """``decision-2.0-vega-27b`` -> ``Decision-2.0-Vega-27B``."""
    _, generation, codename, size = engine.split("-", 3)
    return f"Decision-{generation}-{codename.capitalize()}-{size.upper()}"


def board_point(model: dict[str, Any]) -> dict[str, Any]:
    v03 = model["v03"]
    served = model["meta"].get("served_params")
    return {
        "engine": model["engine"],
        "parameters": int(served) if served else None,
        "full": v03["v"],
        "public": v03["pub"],
        "same_skill": v03["same_skills"],
        "new_domain": v03["new_domains"],
        "rank": v03.get("rank_tie"),
        "position": v03.get("rank_v"),
        "areas": {
            c["id"]: round(100 * c["skill"], 2)
            for c in model.get("categories", [])
            if c["id"] in AREAS
        },
    }


def build(
    board: dict[str, Any],
    snapshot: str,
    roster: dict[str, Any],
    manifest: dict[str, Any],
    own: dict[str, Any],
    teachers: list[dict[str, Any]],
    latency: dict[str, Any] | None,
    peers: int = 3,
    select: list[str] | None = None,
) -> dict[str, Any]:
    models = [
        m
        for m in [*board["models"], *([board["jev"]] if board.get("jev") else [])]
        if m.get("v03") and m["v03"].get("v") is not None
    ]
    points = {m["engine"]: board_point(m) for m in models}
    if PREVIOUS_ENGINE not in points:
        raise ValueError(f"the board has no {PREVIOUS_ENGINE}")
    previous = {**points[PREVIOUS_ENGINE], "name": decision_name(PREVIOUS_ENGINE)}
    if own["status"] == "board":
        mine = points[own["engine"]]
        own_values = {
            k: mine[k]
            for k in ("full", "public", "same_skill", "new_domain", "rank", "areas")
        }
    elif own["status"] == "pending":
        own_values = {k: None for k in ("full", "public", "same_skill", "new_domain")}
    else:
        own_values = {
            k: own.get(k)
            for k in (
                "full",
                "public",
                "same_skill",
                "new_domain",
                "uncertainty",
                "areas",
                "full_unadjusted",
                "paired_with",
            )
        }
    others = [
        p for e, p in points.items() if not e.startswith((FAMILY_PREFIX, OWN_PREFIX))
    ]
    ranked = sorted(others, key=lambda p: -p["full"])
    eligible = []
    for point in ranked:
        entry = roster.get(point["engine"])
        if entry and (entry.get("licence") or "").lower() in CARD_LICENCES:
            eligible.append(
                {
                    **point,
                    "name": entry["name"],
                    "repo": entry.get("repo"),
                    "licence": entry["licence"].lower(),
                }
            )
    if select:
        eligible = sorted(
            (
                {
                    **points[e],
                    "name": roster[e]["name"],
                    "repo": roster[e].get("repo"),
                    "licence": (roster[e].get("licence") or "unverified").lower(),
                    "selected": "lead",
                }
                for e in select
            ),
            key=lambda p: -p["full"],
        )
        peers = len(eligible)
    top = ranked[0]
    top_name = (roster.get(top["engine"]) or {}).get("name") or next(
        (m["name"] for m in models if m["engine"] == top["engine"]), top["engine"]
    )
    runner_up = None
    if own["status"] == "board":
        better = [p for p in ranked if p["full"] < own_values["full"]]
        if better:
            runner_up = {
                "name": (roster.get(better[0]["engine"]) or {}).get("name")
                or better[0]["engine"],
                "full": better[0]["full"],
            }
    family = [
        {**points[e], "name": decision_name(e)}
        for e in points
        if e.startswith(FAMILY_PREFIX) and points[e]["parameters"]
    ]
    return {
        "schema": SCHEMA,
        "model_name": manifest["model_name"],
        "repo_id": manifest["repo_id"],
        "model_sha256": manifest["identity"]["model_sha256"],
        "base_model": manifest["base_model"]["repo_id"],
        "parameters_loaded": manifest["parameters"]["loaded"],
        "parameters_served": previous["parameters"],
        "max_input_tokens": manifest["max_input_tokens"],
        "collection_url": own.get("collection_url"),
        "index": {
            "edition": "0.3",
            "status": own["status"],
            "snapshot": snapshot,
            "board_generated_utc": board.get("generated_utc"),
            "own": own_values,
            "previous": previous,
            "peers": eligible[:peers],
            "board_top": {"name": top_name, "full": top["full"]},
            "runner_up": runner_up,
            "family": family,
            "entrants": [
                {"parameters": p["parameters"], "full": p["full"]}
                for p in others
                if p["parameters"]
            ],
        },
        "speed": (
            None
            if latency is None
            else {
                "median_ms": latency["median_ms"],
                "mean_ms": latency["mean_ms"],
                "p80_ms": latency["p80_ms"],
                "gpu": "NVIDIA RTX PRO 6000",
            }
        ),
        "teachers": teachers,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("board", "roster", "manifest", "own", "teachers", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--latency", type=Path)
    ap.add_argument("--peers", type=int, default=3)
    ap.add_argument(
        "--select",
        nargs="+",
        help="peer engines chosen by the lead (shown regardless of licence)",
    )
    args = ap.parse_args(argv)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    data = build(
        read(args.board),
        args.snapshot,
        read(args.roster),
        read(args.manifest),
        read(args.own),
        read(args.teachers),
        read(args.latency) if args.latency else None,
        args.peers,
        args.select,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "peers": [p["name"] for p in data["index"]["peers"]],
                "board_top": data["index"]["board_top"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
