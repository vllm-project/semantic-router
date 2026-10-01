"""Assemble the private Index input for the round-2 cards (schema dev2-card-index/1).

Inputs are private and never committed: the independent kit runs of the six
released tiers (one JSON object per run, keyed by run name) and the public board
snapshot. The output stays under the private program directory and on the release
node's private tree; only its SHA-256 enters a commit.

    python make_index.py --runs runs-c.json runs-d.json --board index-latest.json \
        --snapshot 2026-09-28 --out decision-index-card.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2.release import card_index
from v2.release.layout import CODENAMES, TIERS

FOOTNOTE = (
    "Decision 2.0: independent reproduction with the official {edition} kit on the "
    "released weights; others: public board snapshot, {snapshot}."
)
DECISION1 = {
    "Decision 1.0 Kai": "0.6B",
    "Decision 1.0 Eos": "0.8B",
    "Decision 1.0 Sol": "2B",
    "Decision 1.0 Nox": "4B",
    "Decision 1.0 Lux": "9B",
    "Decision 1.0 Lex": None,
}


def last_json(path: Path) -> dict:
    """A run file may carry a directory listing before its one JSON line."""
    lines = [l for l in path.read_text().splitlines() if l.startswith("{")]
    return json.loads(lines[-1])


def board_areas(model: dict) -> dict[str, float]:
    skills = {c["id"]: c["skill"] for c in model["categories"]}
    return {a: round(100 * skills[a], 2) for a, _ in card_index.AREAS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--board", type=Path, required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument(
        "--manifests",
        type=Path,
        required=True,
        help="dir with <tier>/MODEL_MANIFEST.json",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = {}
    for path in args.runs:
        runs.update(last_json(path))
    family, editions = [], set()
    for tier in TIERS:
        old = f"llm-semantic-router/DEV2.0-{tier}"
        run = next(
            r
            for r in runs.values()
            if r["model_id"]
            in (old, f"llm-semantic-router/Decision-2.0-{CODENAMES[tier]}-{tier}")
        )
        manifest = json.loads(
            (args.manifests / tier / "MODEL_MANIFEST.json").read_text()
        )
        if run["model_sha256"] != manifest["identity"]["model_sha256"]:
            raise SystemExit(f"{tier}: kit run scored other weights")
        editions.add(run["edition"])
        family.append(
            {
                "tier": tier,
                "name": f"Decision-2.0-{CODENAMES[tier]}-{tier}",
                "parameters": manifest["parameters"]["loaded"],
                "balanced_skill": run["balanced_skill"],
                "areas": run["areas"],
                "model_sha256": run["model_sha256"],
                "kit_index_sha256": run["index_sha256"],
            }
        )
    if len(editions) != 1:
        raise SystemExit(f"mixed kit editions {editions}")
    edition = editions.pop()

    board = json.loads(args.board.read_text())
    decision1, entrants = [], []
    for model in board["models"]:
        point = {
            "parameters": model["meta"]["served_params"],
            "balanced_skill": model["scores"]["balanced_skill"],
        }
        if model["name"] in DECISION1:
            decision1.append(
                {
                    "name": model["name"],
                    "tier": DECISION1[model["name"]],
                    **point,
                    "areas": board_areas(model),
                }
            )
        elif model["name"].startswith("Decision"):
            raise SystemExit(f"unexpected own model on the board: {model['name']}")
        else:
            entrants.append({"name": model["name"], **point})
    out = {
        "schema": card_index.SCHEMA,
        "edition": edition,
        "snapshot": args.snapshot,
        "board_generated_utc": board["generated_utc"],
        "footnote": FOOTNOTE.format(edition=edition, snapshot=args.snapshot),
        "family": family,
        "decision1": sorted(decision1, key=lambda p: p["parameters"]),
        "entrants": sorted(entrants, key=lambda p: p["parameters"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    loaded = card_index.load(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "sha256": loaded["sha256"],
                "entrants": len(entrants),
            }
        )
    )


if __name__ == "__main__":
    main()
