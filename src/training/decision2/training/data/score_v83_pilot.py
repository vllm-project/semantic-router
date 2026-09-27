"""CPU-only Score v8.3 source-disjoint decision dossier quality pilot.

The generated rows are candidates, not admitted training data. Their private
keys and labeled rows must remain separate from the gold-free review packets.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from training.data import score_v8_pilot as packet_tools
from training.model.data import INPUT_FIELDS, digest, file_sha256, validate_row

VERSION = "decision2-score-v8.3-dossiers/1"
MECHANISMS = ("dependency", "connection", "fulfillment")
GROUPS = {"train": 4, "select": 2}
LEVELS = (0, 1, 2)

WORK_SETTINGS = (
    (
        "regional archive migration",
        "catalog freeze",
        "rights scan",
        "checksum approval",
        "vendor billing",
    ),
    (
        "community theater lighting handoff",
        "cue export",
        "rig inspection",
        "operator signoff",
        "poster print",
    ),
    (
        "field sensor release",
        "firmware image",
        "battery certification",
        "telemetry rehearsal",
        "site catering",
    ),
    (
        "public library kiosk rollout",
        "accessibility pass",
        "network review",
        "desk training",
        "lobby plants",
    ),
    (
        "coastal lab sampler restart",
        "pump service",
        "water blank",
        "custody review",
        "newsletter",
    ),
    (
        "museum audio guide update",
        "translation check",
        "device pairing",
        "gallery approval",
        "gift shop",
    ),
)
TRAVEL_SETTINGS = (
    ("harbor terminal", "ferry", "quay", "boarding desk"),
    ("mountain rail junction", "train", "platform", "station office"),
    ("airport shuttle hub", "shuttle", "bay", "gate agent"),
    ("regional coach station", "coach", "stand", "dispatch window"),
    ("river crossing depot", "waterbus", "pier", "ticket booth"),
    ("city tram interchange", "tram", "platform", "service counter"),
)
STOCK_SETTINGS = (
    ("theater prop store", "dispatch sheet", "bin count", "carrier note"),
    ("school science stockroom", "request slip", "shelf card", "receiving email"),
    ("regional print shop", "job ticket", "cycle count", "freight update"),
    ("museum conservation store", "pull list", "stock snapshot", "delivery docket"),
    ("coastal field depot", "packing request", "rack audit", "inbound manifest"),
    ("library repair hub", "service order", "inventory export", "courier message"),
)
OPTIONS = {
    "dependency": (
        "Required work is blocked",
        "Required work is pending",
        "Required work is ready",
    ),
    "connection": (
        "Connection is impossible",
        "Connection is uncertain",
        "Connection is reliable",
    ),
    "fulfillment": (
        "Order cannot be filled",
        "Order awaits tentative stock",
        "Order can be filled",
    ),
}
INSTRUCTIONS = {
    "dependency": (
        "Judge the named handoff through all transitive required tasks. A failed required task blocks it; "
        "otherwise a queued required task leaves it pending; all required tasks complete means ready. "
        "Ignore the unrelated neighboring task."
    ),
    "connection": (
        "Use the named incoming service's earliest and latest arrival, the walking time between its arrival "
        "and the onward gate, and the onward boarding cutoff. If even earliest arrival plus walking time "
        "misses cutoff, the connection is impossible. If only some arrivals make it, it is uncertain. "
        "If even latest arrival makes cutoff, it is reliable."
    ),
    "fulfillment": (
        "Assess only the requested SKU. Available confirmed units are counted stock minus existing reservations "
        "plus confirmed inbound units. Tentative inbound may arrive but is not confirmed. If even the "
        "optimistic total is short, the order cannot be filled; if only tentative inbound closes the gap, "
        "it awaits stock; otherwise it can be filled."
    ),
}


def _rng(secret: bytes, role: str, mechanism: str, index: int) -> random.Random:
    payload = f"{VERSION}\0{role}\0{mechanism}\0{index}".encode()
    value = hmac.new(secret, payload, hashlib.sha256).digest()
    return random.Random(int.from_bytes(value[:16], "big"))


def _tag(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=3))}{rng.randrange(100, 999)}"


def _minute(value: int) -> str:
    assert 0 <= value < 24 * 60
    return f"{value // 60:02d}:{value % 60:02d}"


def _base(rng: random.Random, style: int, mechanism: str) -> dict[str, Any]:
    if mechanism == "dependency":
        setting, first, second, third, decoy = WORK_SETTINGS[style]
        return {
            "style": style,
            "setting": setting,
            "case": _tag(rng, "JOB"),
            "root": _tag(rng, "HANDOFF"),
            "required": [first, second, third],
            "decoy": decoy,
            "decoy_status": rng.choice(("failed", "queued", "complete")),
            "failed_index": style % 3,
            "queued_index": (style + 1) % 3,
        }
    if mechanism == "connection":
        setting, vehicle, platform, desk = TRAVEL_SETTINGS[style]
        departure = rng.randrange(10 * 60, 18 * 60)
        cutoff_gap = rng.randrange(4, 8)
        walk = rng.randrange(5, 13)
        return {
            "style": style,
            "setting": setting,
            "vehicle": vehicle,
            "platform": platform,
            "desk": desk,
            "case": _tag(rng, "CONN"),
            "incoming": _tag(rng, "IN"),
            "onward": _tag(rng, "OUT"),
            "decoy": _tag(rng, "OTHER"),
            "departure": departure,
            "cutoff": departure - cutoff_gap,
            "walk": walk,
        }
    if mechanism == "fulfillment":
        setting, request_doc, count_doc, inbound_doc = STOCK_SETTINGS[style]
        quantity = rng.randrange(19, 37)
        reserved = rng.randrange(2, 7)
        confirmed = rng.randrange(1, 5)
        gap = rng.randrange(4, 8)
        return {
            "style": style,
            "setting": setting,
            "request_doc": request_doc,
            "count_doc": count_doc,
            "inbound_doc": inbound_doc,
            "case": _tag(rng, "PICK"),
            "sku": _tag(rng, "SKU"),
            "decoy_sku": _tag(rng, "SKU"),
            "quantity": quantity,
            "reserved": reserved,
            "confirmed": confirmed,
            "gap": gap,
            "tentative_low": rng.randrange(1, gap),
        }
    raise ValueError(mechanism)


def _facts(base: dict[str, Any], mechanism: str, level: int) -> dict[str, Any]:
    facts = dict(base)
    if mechanism == "dependency":
        statuses = ["complete", "complete", "complete"]
        if level == 0:
            statuses[base["failed_index"]] = "failed"
        elif level == 1:
            statuses[base["queued_index"]] = "queued"
        facts["statuses"] = statuses
    elif mechanism == "connection":
        threshold = base["cutoff"] - base["walk"]
        if level == 0:
            earliest = threshold + 2
            latest = threshold + 6
        elif level == 1:
            earliest = threshold - 3
            latest = threshold + 6
        else:
            earliest = threshold - 3
            latest = threshold - 1
        facts["earliest"] = earliest
        facts["latest"] = latest
    elif mechanism == "fulfillment":
        onhand_low = (
            base["quantity"] + base["reserved"] - base["confirmed"] - base["gap"]
        )
        facts["onhand"] = onhand_low if level < 2 else onhand_low + base["gap"] + 1
        facts["tentative"] = base["tentative_low"] if level != 1 else base["gap"] + 2
    else:
        raise ValueError(mechanism)
    return facts


def oracle(f: dict[str, Any], mechanism: str) -> int:
    if mechanism == "dependency":
        statuses = f["statuses"]
        return 0 if "failed" in statuses else 1 if "queued" in statuses else 2
    if mechanism == "connection":
        first = f["earliest"] + f["walk"]
        last = f["latest"] + f["walk"]
        return 0 if first > f["cutoff"] else 1 if last > f["cutoff"] else 2
    if mechanism == "fulfillment":
        confirmed = f["onhand"] - f["reserved"] + f["confirmed"]
        return (
            0
            if confirmed + f["tentative"] < f["quantity"]
            else 1 if confirmed < f["quantity"] else 2
        )
    raise ValueError(mechanism)


def _render_dependency(f: dict[str, Any]) -> str:
    a, b, c = f["required"]
    root, case = f["root"], f["case"]
    graph = f"Dependency map [{case}]: {root} needs {a} and {b}; {b} also needs {c}."
    status = (
        f"Status cards [{case}]: {a}={f['statuses'][0]}; {b}={f['statuses'][1]}; "
        f"{c}={f['statuses'][2]}; {f['decoy']}={f['decoy_status']}."
    )
    context = (
        f"{f['setting'].title()} — coordinator's handoff request for {root}. "
        "The neighboring item has its own owner and is not a dependency of this handoff."
    )
    styles = (
        ("Request memo", "Work breakdown", "Technician board"),
        ("Stage manager note", "Cue chain", "Floor checklist"),
        ("Field ticket", "Prerequisite sketch", "Bench log"),
        ("IT rollout note", "Required work", "Service desk export"),
        ("Restart brief", "Order of operations", "Shift statuses"),
        ("Curator memo", "Approval chain", "Device board"),
    )
    h1, h2, h3 = styles[f["style"]]
    sections = [f"{h1}: {context}", f"{h2}: {graph}", f"{h3}: {status}"]
    shift = f["style"] % 3
    return "\n\n".join(sections[shift:] + sections[:shift])


def _render_connection(f: dict[str, Any]) -> str:
    case = f["case"]
    intro = (
        f"{f['setting'].title()} transfer inquiry {case}: riders arrive on {f['incoming']} "
        f"and want {f['onward']}. A notice for {f['decoy']} concerns a different service."
    )
    arrival = (
        f"Service update [{case}]: {f['incoming']} arrival window "
        f"{_minute(f['earliest'])} to {_minute(f['latest'])}."
    )
    walking = (
        f"Route guide [{case}]: allow {f['walk']} minutes from the incoming "
        f"{f['platform']} to the onward boarding point."
    )
    board = (
        f"{f['desk'].title()} [{case}]: {f['onward']} departs at "
        f"{_minute(f['departure'])}; boarding closes {_minute(f['cutoff'])}."
    )
    decoy = (
        f"Separate notice: {f['decoy']} is listed for {_minute(f['departure'] + 16)}."
    )
    layouts = (
        [intro, arrival, walking, board, decoy],
        [intro, board, decoy, walking, arrival],
        [arrival, intro, walking, decoy, board],
        [board, intro, arrival, walking, decoy],
        [walking, intro, board, arrival, decoy],
        [intro, decoy, arrival, board, walking],
    )
    return "\n\n".join(layouts[f["style"]])


def _render_fulfillment(f: dict[str, Any]) -> str:
    case, sku = f["case"], f["sku"]
    request = (
        f"{f['request_doc'].title()} [{case}] requests {f['quantity']} units of {sku}. "
        f"A nearby order is for {f['decoy_sku']}, not this pick."
    )
    stock = (
        f"{f['count_doc'].title()} [{case}]: {sku} counted {f['onhand']} on hand; "
        f"{f['reserved']} already reserved. The count belongs to the requested SKU."
    )
    inbound = (
        f"{f['inbound_doc'].title()} [{case}]: {sku} has {f['confirmed']} confirmed "
        f"incoming units and {f['tentative']} tentative units."
    )
    context = f"{f['setting'].title()} — order reconciliation {case}."
    layouts = (
        [context, request, stock, inbound],
        [request, context, inbound, stock],
        [stock, context, request, inbound],
        [context, inbound, stock, request],
        [inbound, context, request, stock],
        [context, stock, inbound, request],
    )
    return "\n\n".join(layouts[f["style"]])


def render(f: dict[str, Any], mechanism: str) -> str:
    if mechanism == "dependency":
        return _render_dependency(f)
    if mechanism == "connection":
        return _render_connection(f)
    if mechanism == "fulfillment":
        return _render_fulfillment(f)
    raise ValueError(mechanism)


def rendered_oracle(state: str, mechanism: str, case: str) -> int:
    """Recompute from visible prose, without structured answer facts."""
    if mechanism == "dependency":
        graph = re.search(
            rf"Dependency map \[{case}\]: ([A-Z-]+\d+) needs (.+?) and (.+?); \3 also needs (.+?)\.",
            state,
        )
        if graph is None:
            raise ValueError("Dependency map cannot be read")
        required = graph.group(2, 3, 4)
        cards = re.search(rf"Status cards \[{case}\]: ([^\n]+)\.", state)
        if cards is None:
            raise ValueError("Status cards cannot be read")
        statuses = dict(
            re.findall(r"([^;=]+)=(failed|queued|complete)", cards.group(1))
        )
        statuses = {k.strip(): v for k, v in statuses.items()}
        if any(name not in statuses for name in required):
            raise ValueError("Required ancestor lacks a status")
        return oracle({"statuses": [statuses[name] for name in required]}, mechanism)
    if mechanism == "connection":
        arrival = re.search(
            rf"Service update \[{case}\]: [A-Z-]+\d+ arrival window (\d\d:\d\d) to (\d\d:\d\d)\.",
            state,
        )
        walk = re.search(rf"Route guide \[{case}\]: allow (\d+) minutes", state)
        board = re.search(
            rf"\[{case}\]: [A-Z-]+\d+ departs at (\d\d:\d\d); boarding closes (\d\d:\d\d)\.",
            state,
        )
        if arrival is None or walk is None or board is None:
            raise ValueError("Connection evidence cannot be read")
        to_min = lambda value: int(value[:2]) * 60 + int(value[3:])
        return oracle(
            {
                "earliest": to_min(arrival[1]),
                "latest": to_min(arrival[2]),
                "walk": int(walk[1]),
                "cutoff": to_min(board[2]),
            },
            mechanism,
        )
    if mechanism == "fulfillment":
        request = re.search(
            rf"\[{case}\] requests (\d+) units of (SKU-[A-Z]+\d+)\.", state
        )
        if request is None:
            raise ValueError("Order request cannot be read")
        sku = re.escape(request[2])
        stock = re.search(
            rf"\[{case}\]: {sku} counted (\d+) on hand; (\d+) already reserved\.",
            state,
        )
        inbound = re.search(
            rf"\[{case}\]: {sku} has (\d+) confirmed incoming units and (\d+) tentative units\.",
            state,
        )
        if stock is None or inbound is None:
            raise ValueError("Stock or carrier evidence cannot be read")
        return oracle(
            {
                "quantity": int(request[1]),
                "onhand": int(stock[1]),
                "reserved": int(stock[2]),
                "confirmed": int(inbound[1]),
                "tentative": int(inbound[2]),
            },
            mechanism,
        )
    raise ValueError(mechanism)


def build(secret: bytes, role: str) -> list[dict[str, Any]]:
    if len(secret) != 32 or role not in GROUPS:
        raise ValueError("Need a private 32-byte seed and a known role")
    rows = []
    for mechanism in MECHANISMS:
        for index in range(GROUPS[role]):
            style = index + (4 if role == "select" else 0)
            rng = _rng(secret, role, mechanism, index)
            base = _base(rng, style, mechanism)
            group = f"score83-{role}-{mechanism}-{hashlib.sha256((base['case'] + role).encode()).hexdigest()[:14]}"
            for level in LEVELS:
                facts = _facts(base, mechanism, level)
                state = render(facts, mechanism)
                if (
                    oracle(facts, mechanism) != level
                    or rendered_oracle(state, mechanism, base["case"]) != level
                ):
                    raise AssertionError(
                        f"Structured/rendered oracle disagreement: {group}, level {level}"
                    )
                row = {
                    "id": f"{group}_l{level}",
                    "state": state,
                    "instructions": INSTRUCTIONS[mechanism],
                    "options": [
                        {"key": str(i), "description": text}
                        for i, text in enumerate(OPTIONS[mechanism])
                    ],
                    "label": level,
                    "task_type": "score",
                    "family": f"score_v83_{mechanism}",
                    "group_id": group,
                    "language": "en",
                    "split": role,
                    "source": VERSION,
                    "evaluation_role": role,
                    "render_template": f"v8.3-{mechanism}-genre-{style}",
                    "audit_metadata": {
                        "mechanism": mechanism,
                        "case": base["case"],
                        "style": style,
                        "oracle_version": VERSION,
                    },
                }
                row["input_sha256"] = digest(
                    {field: row[field] for field in INPUT_FIELDS}
                )
                validate_row(row, role)
                rows.append(row)
    return rows


def write(seed_file: Path, output_dir: Path) -> dict[str, Any]:
    if seed_file.stat().st_mode & 0o077:
        raise PermissionError("Seed file must be mode 0600")
    secret = seed_file.read_bytes()
    if len(secret) != 32 or output_dir.exists():
        raise ValueError("Need a 32-byte seed and a fresh output directory")
    output_dir.mkdir(parents=True, mode=0o700)
    manifest: dict[str, Any] = {
        "schema_version": VERSION,
        "status": "PENDING_INDEPENDENT_BLIND_REVIEW",
        "seed_sha256": hashlib.sha256(secret).hexdigest(),
        "roles": {},
    }
    for role in GROUPS:
        rows = build(secret, role)
        packet, key = packet_tools.blind_packet(rows, secret)
        data_path = output_dir / f"{role}.jsonl"
        packet_path = output_dir / f"{role}-blind-packet.jsonl"
        key_path = output_dir / f"{role}-sealed-key.json"
        packet_tools._write_jsonl(data_path, rows)
        packet_tools._write_jsonl(packet_path, packet)
        with key_path.open("x", encoding="utf-8") as stream:
            json.dump(key, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        key_path.chmod(0o600)
        manifest["roles"][role] = {
            "rows": len(rows),
            "groups": len(packet),
            "labels": dict(collections.Counter(str(row["label"]) for row in rows)),
            "rows_sha256": file_sha256(data_path),
            "packet_sha256": file_sha256(packet_path),
            "key_sha256": file_sha256(key_path),
        }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = write(args.seed_file, args.output_dir)
    print(
        json.dumps(
            {"status": manifest["status"], "roles": manifest["roles"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
