"""Card inputs for tests: a synthetic Index file and placeholder assets with their receipt (stdlib).

The Index values here are made up; real values live only in the private input file.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

from v2.release import card, card_index, layout

ROOT = Path(__file__).resolve().parents[3]
ROSTER = ROOT / "v2/eval/records/decision-index-peer-roster-2026-09-28.json"
EDITION, SNAPSHOT = "0.0-test", "2000-01-01"
FOOTNOTE = card_index.FOOTNOTE.format(edition=EDITION, snapshot=SNAPSHOT)
# Synthetic balanced skills: family tier i -> 10.5 + 10 i; Decision 1.0 -> 2.25 below the same tier.
FAMILY = {tier: 10.5 + 10 * i for i, tier in enumerate(layout.TIERS)}
DECISION1 = {
    name: FAMILY[tier] - 2.25 for tier, name in card_index.COUNTERPART_1_0.items()
}


def _areas(skill: float) -> dict[str, float]:
    return {area: skill + i - 2 for i, (area, _) in enumerate(card_index.AREAS)}


def index_file(
    path: Path,
    model_sha256: dict[str, str] | None = None,
    family: dict[str, float] | None = None,
) -> Path:
    family = {**FAMILY, **(family or {})}
    data = {
        "schema": card_index.SCHEMA,
        "edition": EDITION,
        "snapshot": SNAPSHOT,
        "footnote": FOOTNOTE,
        "family": [
            {
                "tier": tier,
                "name": layout.tier_name(tier),
                "parameters": int(layout.TIERS[tier] * 1.1),
                "parameters_basis": card_index.PARAMETERS_BASIS,
                "loaded_parameters": int(layout.TIERS[tier]),
                "balanced_skill": family[tier],
                "areas": _areas(family[tier]),
                "model_sha256": (model_sha256 or {}).get(tier, "a" * 64),
            }
            for tier in layout.TIERS
        ],
        "decision1": [
            {
                "name": name,
                "tier": tier,
                "parameters": int(layout.TIERS[tier] * 1.1),
                "balanced_skill": DECISION1[name],
                "areas": _areas(DECISION1[name]),
            }
            for tier, name in card_index.COUNTERPART_1_0.items()
        ],
        "entrants": [
            {
                "name": f"entrant-{i}",
                "parameters": 10**8 * (i + 1),
                "balanced_skill": 1.5 * i,
            }
            for i in range(8)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def png(path: Path, seed: int = 0) -> None:
    """A 1x1 PNG without text chunks."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    pixel = zlib.compress(bytes([0, seed % 256, 255, 255]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", pixel)
        + chunk(b"IEND", b"")
    )


def tier_of(model_name: str) -> str:
    codename = layout.MODEL_NAME.match(model_name)["codename"]
    return next(t for t, c in layout.CODENAMES.items() if c == codename)


def relabel(entries: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    return [
        {**e, "label": model_name} if e["role"] == "candidate" else dict(e)
        for e in entries
    ]


def card_inputs(
    scratch: Path,
    entries: list[dict[str, Any]],
    model_name: str,
    model_sha256: str,
    comparison: str = "own-1.0",
    roster: Path = ROSTER,
    family: dict[str, float] | None = None,
) -> tuple[Path, Path]:
    """(index file, assets dir) matching these reports and weights, as card_assets would write."""
    index = index_file(
        scratch / "card-index.json", {tier_of(model_name): model_sha256}, family
    )
    shown = card.select_reports(entries, roster, comparison)["shown"]
    assets = scratch / "card-assets"
    files = {}
    for i, name in enumerate(layout.CARD_ASSETS):
        png(assets / name, i)
        files[name] = layout.sha_file(assets / name)
    receipt = {
        "schema": card.ASSETS_SCHEMA,
        "model_name": model_name,
        "model_sha256": model_sha256,
        "inputs": {
            "reports": {e["key"]: e["sha256"] for e in shown},
            "index_sha256": layout.sha_file(index),
        },
        "files": files,
    }
    (assets / card.ASSETS_RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
    return index, assets


def pinned_card(
    scratch: Path,
    entries: list[dict[str, Any]],
    model_name: str,
    model_sha256: str,
    text: dict[str, str] | None = None,
    comparison: str = "own-1.0",
) -> dict[str, Any]:
    """A spec ``card`` block whose Index input and assets are pinned like a release spec's."""
    entries = relabel(entries, model_name)
    index, assets = card_inputs(scratch, entries, model_name, model_sha256, comparison)
    return {
        "reports": entries,
        "roster": str(ROSTER),
        "text": text or {},
        "index": {"path": str(index), "sha256": layout.sha_file(index)},
        "assets": {
            "dir": str(assets),
            "receipt_sha256": layout.sha_file(assets / card.ASSETS_RECEIPT),
        },
    }
