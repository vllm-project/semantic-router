"""Deterministic, private Score v8 multimechanism quality pilot.

This produces candidate TRAIN/SELECT rows and separate gold-free blind packets.
It does not admit a corpus, select a model, or read any benchmark answer.
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

from training.model.data import INPUT_FIELDS, digest, file_sha256, validate_row

VERSION = "decision2-score-v8.1-multimechanism-pilot/1"
MECHANISMS = (
    "dated_update",
    "numeric_limits",
    "evidence_sufficiency",
    "scoped_exception",
    "long_memo",
)
GROUPS = {"train": 3, "select": 2}
LEVELS = (0, 1, 2)
OPTIONS = {
    "dated_update": (
        "Authorization is suspended",
        "Authorization awaits follow-up",
        "Authorization is active",
    ),
    "numeric_limits": (
        "Release limits are violated",
        "Release evidence is incomplete",
        "Release limits are verified",
    ),
    "evidence_sufficiency": (
        "Claim is contradicted",
        "Claim lacks corroboration",
        "Claim is corroborated",
    ),
    "scoped_exception": (
        "Restriction still applies",
        "Exception awaits approval",
        "Valid exception is in force",
    ),
    "long_memo": (
        "Delivery is late",
        "Delivery is unconfirmed",
        "Delivery is timely and confirmed",
    ),
}
ROLE_ORGANIZATIONS = {
    "train": (
        "borough fleet office",
        "regional sample laboratory",
        "district utility desk",
        "coastal works unit",
        "school procurement board",
    ),
    "select": (
        "museum equipment office",
        "mountain field station",
        "clinic facilities desk",
        "forest works unit",
        "civic library procurement board",
    ),
}


def _rng(secret: bytes, role: str, mechanism: str, index: int) -> random.Random:
    message = f"{VERSION}\0{role}\0{mechanism}\0{index}".encode()
    value = hmac.new(secret, message, hashlib.sha256).digest()
    return random.Random(int.from_bytes(value[:16], "big"))


def _base(rng: random.Random, role: str, mechanism: str, index: int) -> dict[str, Any]:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    suffix = "".join(rng.choices(alphabet, k=4)) + str(rng.randrange(100, 999))
    other = "".join(rng.choices(alphabet, k=4)) + str(rng.randrange(100, 999))
    while other == suffix:
        other = "".join(rng.choices(alphabet, k=4)) + str(rng.randrange(100, 999))
    other2 = "".join(rng.choices(alphabet, k=4)) + str(rng.randrange(100, 999))
    while other2 in {suffix, other}:
        other2 = "".join(rng.choices(alphabet, k=4)) + str(rng.randrange(100, 999))
    deadline = rng.randrange(18, 22)
    return {
        "record": "R-" + suffix,
        "other": "R-" + other,
        "other2": "R-" + other2,
        "day": rng.randrange(22, 28),
        "organization": ROLE_ORGANIZATIONS[role][MECHANISMS.index(mechanism)],
        "site": "Site " + alphabet[rng.randrange(len(alphabet))],
        "other_site": "Site " + alphabet[rng.randrange(len(alphabet))],
        "activity": rng.choice(
            ("night access", "equipment transfer", "sample collection")
        ),
        "other_activity": "archive disposal",
        "temp_limit": rng.randrange(6, 10),
        "shock_limit": rng.randrange(4, 7),
        "fail_dimension": rng.randrange(2),
        "missing_dimension": rng.randrange(2),
        "window": (rng.randrange(8, 12), rng.randrange(13, 17)),
        "deadline": deadline,
        "amended_deadline": deadline + rng.randrange(2, 5),
        "target_position": rng.randrange(3),
        "index": index,
    }


def _facts(
    base: dict[str, Any], mechanism: str, level: int, rng: random.Random
) -> dict[str, Any]:
    f = dict(base)
    if mechanism == "dated_update":
        actions = ("suspended", "review", "active")
        f["events"] = [
            (base["day"] - 9, base["record"], "active", "signed"),
            (base["day"] - 2, base["record"], actions[level], "signed"),
            (base["day"] - 1, base["record"], actions[(level + 1) % 3], "unsigned"),
            (base["day"] + 2, base["record"], actions[(level + 2) % 3], "signed"),
            (base["day"] - 1, base["other"], "review", "signed"),
        ]
        rng.shuffle(f["events"])
    elif mechanism == "numeric_limits":
        f["temp"] = f["temp_limit"] - 1
        f["shock"] = f["shock_limit"] - 1
        f["calibration"] = "current"
        if level == 0:
            if f["fail_dimension"]:
                f["temp"] = f["temp_limit"] + 1
            else:
                f["shock"] = f["shock_limit"] + 1
        elif level == 1:
            if f["missing_dimension"]:
                f["temp"] = None
            else:
                f["calibration"] = "missing"
        f["readings"] = [(f["record"], f["temp"], f["shock"], f["calibration"])]
        for decoy_id, decoy_level in zip(
            (f["other"], f["other2"]), (i for i in LEVELS if i != level)
        ):
            if decoy_level == 0:
                reading = (
                    decoy_id,
                    f["temp_limit"] + 1,
                    f["shock_limit"] - 1,
                    "current",
                )
            elif decoy_level == 1:
                reading = (
                    (decoy_id, None, f["shock_limit"] - 1, "current")
                    if f["missing_dimension"]
                    else (
                        decoy_id,
                        f["temp_limit"] - 1,
                        f["shock_limit"] - 1,
                        "missing",
                    )
                )
            else:
                reading = (
                    decoy_id,
                    f["temp_limit"] - 1,
                    f["shock_limit"] - 1,
                    "current",
                )
            f["readings"].append(reading)
        rng.shuffle(f["readings"])
    elif mechanism == "evidence_sufficiency":
        f["dispatch"] = "outage"
        f["telemetry"] = ("normal", "unavailable", "outage")[level]
        f["evidence_rows"] = [(f["record"], f["dispatch"], f["telemetry"])]
        for decoy_id, decoy_level in zip(
            (f["other"], f["other2"]), (i for i in LEVELS if i != level)
        ):
            f["evidence_rows"].append(
                (decoy_id, "outage", ("normal", "unavailable", "outage")[decoy_level])
            )
        rng.shuffle(f["evidence_rows"])
    elif mechanism == "scoped_exception":
        if f["site"] == f["other_site"]:
            f["other_site"] += " East"
        if level == 0:
            target = (f["site"], f["activity"], f["day"] - 8, f["day"] - 1, "signed")
            decoys = [
                (f["other_site"], f["activity"], f["day"] - 3, f["day"] + 3, "pending"),
                (f["site"], f["other_activity"], f["day"] - 3, f["day"] + 3, "signed"),
            ]
        elif level == 1:
            target = (f["site"], f["activity"], f["day"] - 2, f["day"] + 2, "pending")
            decoys = [
                (f["other_site"], f["activity"], f["day"] - 3, f["day"] + 3, "signed"),
                (f["site"], f["other_activity"], f["day"] - 8, f["day"] - 1, "signed"),
            ]
        else:
            target = (f["site"], f["activity"], f["day"] - 2, f["day"] + 2, "signed")
            decoys = [
                (f["other_site"], f["activity"], f["day"] - 3, f["day"] + 3, "pending"),
                (f["site"], f["other_activity"], f["day"] - 8, f["day"] - 1, "signed"),
            ]
        f["register"] = [target, *decoys]
        rng.shuffle(f["register"])
    elif mechanism == "long_memo":
        f["arrival"] = (
            f["amended_deadline"] + 2 if level == 0 else f["amended_deadline"] - 1
        )
        f["arrival_status"] = "unconfirmed" if level == 1 else "confirmed"
        f["other_delivery"] = []
        for decoy_id, decoy_level in zip(
            (f["other"], f["other2"]), (i for i in LEVELS if i != level)
        ):
            decoy_arrival = (
                f["amended_deadline"] + 2
                if decoy_level == 0
                else f["amended_deadline"] - 1
            )
            status = "unconfirmed" if decoy_level == 1 else "confirmed"
            f["other_delivery"].append((decoy_id, decoy_arrival, status))
    else:
        raise ValueError(mechanism)
    return f


def oracle(f: dict[str, Any], mechanism: str) -> int:
    if mechanism == "dated_update":
        applicable = [
            event
            for event in f["events"]
            if event[1] == f["record"] and event[0] <= f["day"] and event[3] == "signed"
        ]
        status = max(applicable, key=lambda event: event[0])[2]
        return {"suspended": 0, "review": 1, "active": 2}[status]
    if mechanism == "numeric_limits":
        if any(
            value is not None and value > limit
            for value, limit in (
                (f["temp"], f["temp_limit"]),
                (f["shock"], f["shock_limit"]),
            )
        ):
            return 0
        return (
            1
            if f["temp"] is None or f["shock"] is None or f["calibration"] != "current"
            else 2
        )
    if mechanism == "evidence_sufficiency":
        if f["dispatch"] == "normal" or f["telemetry"] == "normal":
            return 0
        return (
            1
            if f["dispatch"] == "unavailable" or f["telemetry"] == "unavailable"
            else 2
        )
    if mechanism == "scoped_exception":
        matched = [
            row
            for row in f["register"]
            if row[0] == f["site"]
            and row[1] == f["activity"]
            and row[2] <= f["day"] <= row[3]
        ]
        if any(row[4] == "signed" for row in matched):
            return 2
        return 1 if any(row[4] == "pending" for row in matched) else 0
    if mechanism == "long_memo":
        if f["arrival"] > f["amended_deadline"]:
            return 0
        return 1 if f["arrival_status"] != "confirmed" else 2
    raise ValueError(mechanism)


_FILLER = (
    "The committee retained the prior quarter's filing convention for receipts and courier stamps.",
    "A separate inquiry concerned the archive shelves and did not amend any delivery obligation.",
    "The finance secretary catalogued several duplicate notices before closing the minutes.",
    "The appendices list inspection visits by building, including visits unrelated to this contract.",
    "Staff compared handwritten copies with the digital register and kept both in the record.",
    "A draft circular addressed equipment storage, but its effective date was deferred.",
    "The public summary omitted personal names while retaining reference identifiers.",
    "The board moved a routine inventory discussion to the next monthly meeting.",
    "A clerical correction changed only the postal routing address on older letters.",
    "The office placed superseded forms in a separate folder for historical lookup.",
    "Several teams reported maintenance work at locations outside the named project.",
    "The attendance sheet confirms that the relevant liaison received a copy of the minutes.",
    "The insurance schedule covers transport and handling but does not set the arrival deadline.",
    "Earlier purchase requests remain attached solely for budget reconciliation.",
    "The filing clerk marked late duplicate submissions as reference copies only.",
    "The annex contains calibration notes for a different equipment purchase.",
    "A subsequent phone note requested clarification without changing the signed amendment.",
    "The records office separated draft proposals from instruments carrying an approval signature.",
)


def render(f: dict[str, Any], mechanism: str, role: str) -> tuple[str, str]:
    record, day, org = f["record"], f["day"], f["organization"]
    if mechanism == "dated_update":
        events = sorted(f["events"], key=lambda item: item[0], reverse=role == "select")
        lines = [
            f"{date} | {item} | {action} | {signature}"
            for date, item, action, signature in events
        ]
        state = (
            f"{org.title()} request note: determine the status of {record} on day {day}.\nTransaction ledger (day | record | action | signature):\n"
            + "\n".join(lines)
        )
        instructions = "Use the latest signed event for the named record no later than the review day. Ignore later, unsigned and other-record entries. Which status governs?"
    elif mechanism == "numeric_limits":
        reading = lambda value: "NA" if value is None else str(value)
        lines = [
            f"{item} | {reading(temp)} | {reading(shock)} | {calibration}"
            for item, temp, shock, calibration in f["readings"]
        ]
        if role == "select":
            lines.reverse()
        state = (
            f"{org.title()} release sheet for lot {record}. Specification: temperature at most {f['temp_limit']} C, shock at most {f['shock_limit']} g, both inclusive; calibration must be current.\nInstrument log (lot | temperature_C | shock_g | calibration):\n"
            + "\n".join(lines)
        )
        instructions = "An observed limit violation blocks release even if other evidence is absent. Otherwise missing reading or calibration needs review; complete passing evidence clears release. Assess only the named lot."
    elif mechanism == "evidence_sufficiency":
        start, end = f["window"]
        lines = [
            f"Dispatch diary for {item}, hours {start}-{end}: {dispatch} observed. Meter trace for {item}, hours {start}-{end}: {telemetry} observed."
            for item, dispatch, telemetry in f["evidence_rows"]
        ]
        if role == "select":
            lines.reverse()
        state = (
            f"Claim file at the {org}: a service interruption is alleged for {record} during hours {start}-{end}.\nIndependently maintained extracts follow:\n"
            + "\n".join(lines)
        )
        instructions = "For the named site and same hours, normal service in either independent source contradicts the claim. If no source contradicts but one is unavailable, corroboration is incomplete. Two outage observations corroborate it."
    elif mechanism == "scoped_exception":
        lines = [
            f"{site} | {activity} | {start} | {end} | {status}"
            for site, activity, start, end, status in f["register"]
        ]
        state = (
            f"{org.title()} policy: {f['activity']} is restricted by default. Request concerns {f['site']} on day {day}.\nException register (site | activity | first_day | last_day | authorization):\n"
            + "\n".join(lines)
        )
        instructions = "Only an entry for the exact site and activity whose inclusive dates cover the review day can override the restriction. A signed entry authorizes; a matching pending entry awaits approval; otherwise the restriction remains."
    elif mechanism == "long_memo":
        snippets = [
            f"The original signed schedule for {record} set a delivery deadline of day {f['deadline']}.",
            f"A later signed amendment for {record} sets the controlling deadline to day {f['amended_deadline']}.",
            f"The delivery notice for {record} reports arrival on day {f['arrival']}; evidence status: {f['arrival_status']}.",
        ]
        pos = f["target_position"]
        blocks = [list(_FILLER[:6]), list(_FILLER[6:12]), list(_FILLER[12:])]
        for index, snippet in enumerate(snippets):
            blocks[(pos + index) % 3].insert(2 + index, snippet)
        paragraphs = [
            f"Section {i + 1}. " + " ".join(block) for i, block in enumerate(blocks)
        ]
        decoys = [
            f"For unrelated {item}, a signed amendment sets the controlling deadline to day {f['amended_deadline']}; its delivery notice reports arrival on day {arrival}, evidence status: {status}."
            for item, arrival, status in f["other_delivery"]
        ]
        if role == "select":
            decoys.reverse()
        paragraphs.insert(1, decoys[0])
        paragraphs.insert(3, decoys[1])
        state = (
            f"{org.title()} contract dossier: decide delivery for {record}.\n"
            + "\n\n".join(paragraphs)
        )
        instructions = "Use the controlling signed amendment for the named record, not the older schedule or another record. Arrival after that deadline is late; an on-time but unconfirmed arrival remains unresolved; confirmed on-time arrival is timely."
    else:
        raise ValueError(mechanism)
    return state, instructions


def rendered_oracle(
    state: str,
    mechanism: str,
    record: str,
    day: int,
    *,
    site: str | None = None,
    activity: str | None = None,
) -> int:
    """Recompute the verdict from rendered fields rather than audit facts."""
    if mechanism == "dated_update":
        rows = [
            (int(date), item, action, signature)
            for date, item, action, signature in re.findall(
                r"(?m)^(\d+) \| (R-[A-Z0-9]+) \| (suspended|review|active) \| (signed|unsigned)$",
                state,
            )
        ]
        if len(rows) != 5:
            raise ValueError("Dated ledger parse failed")
        return oracle({"events": rows, "record": record, "day": day}, mechanism)
    if mechanism == "numeric_limits":
        spec = re.search(r"temperature at most (\d+) C, shock at most (\d+) g", state)
        matches = re.findall(
            r"(?m)^(R-[A-Z0-9]+) \| (NA|\d+) \| (NA|\d+) \| (current|missing)$", state
        )
        if spec is None or len(matches) != 3:
            raise ValueError("Numeric document parse failed")
        _, temp, shock, calibration = next(row for row in matches if row[0] == record)
        parsed = {
            "temp_limit": int(spec[1]),
            "shock_limit": int(spec[2]),
            "temp": None if temp == "NA" else int(temp),
            "shock": None if shock == "NA" else int(shock),
            "calibration": calibration,
        }
        return oracle(parsed, mechanism)
    if mechanism == "evidence_sufficiency":
        matches = re.findall(
            r"(?m)^Dispatch diary for (R-[A-Z0-9]+), hours (\d+-\d+): (outage|normal|unavailable) observed\. Meter trace for \1, hours \2: (outage|normal|unavailable) observed\.$",
            state,
        )
        if len(matches) != 3:
            raise ValueError("Evidence document parse failed")
        _, _, dispatch, telemetry = next(row for row in matches if row[0] == record)
        return oracle({"dispatch": dispatch, "telemetry": telemetry}, mechanism)
    if mechanism == "scoped_exception":
        matches = re.findall(
            r"(?m)^(Site [A-Z](?: East)?) \| ([a-z ]+) \| (\d+) \| (\d+) \| (signed|pending)$",
            state,
        )
        if len(matches) != 3 or site is None or activity is None:
            raise ValueError("Exception register parse failed")
        rows = [
            (s, a, int(first), int(last), status)
            for s, a, first, last, status in matches
        ]
        return oracle(
            {"register": rows, "site": site, "activity": activity, "day": day},
            mechanism,
        )
    if mechanism == "long_memo":
        amendment = re.search(
            rf"signed amendment for {re.escape(record)} sets the controlling deadline to day (\d+)",
            state,
        )
        delivery = re.search(
            rf"delivery notice for {re.escape(record)} reports arrival on day (\d+); evidence status: (confirmed|unconfirmed)",
            state,
        )
        if amendment is None or delivery is None:
            raise ValueError("Long memorandum parse failed")
        return oracle(
            {
                "amended_deadline": int(amendment[1]),
                "arrival": int(delivery[1]),
                "arrival_status": delivery[2],
            },
            mechanism,
        )
    raise ValueError(mechanism)


def build(secret: bytes, role: str) -> list[dict[str, Any]]:
    if len(secret) != 32 or role not in GROUPS:
        raise ValueError("Need a private 32-byte seed and known role")
    rows = []
    for mechanism in MECHANISMS:
        for index in range(GROUPS[role]):
            rng = _rng(secret, role, mechanism, index)
            base = _base(rng, role, mechanism, index)
            group = f"d2sv81_{role}_{mechanism}_{index:02d}"
            for level in LEVELS:
                facts = _facts(base, mechanism, level, rng)
                state, instructions = render(facts, mechanism, role)
                options = [
                    {"key": str(i), "description": description}
                    for i, description in enumerate(OPTIONS[mechanism])
                ]
                if (
                    oracle(facts, mechanism) != level
                    or rendered_oracle(
                        state,
                        mechanism,
                        base["record"],
                        base["day"],
                        site=base["site"],
                        activity=base["activity"],
                    )
                    != level
                ):
                    raise AssertionError(f"Oracle mismatch for {group} level {level}")
                row = {
                    "id": f"{group}_l{level}",
                    "state": state,
                    "instructions": instructions,
                    "options": options,
                    "label": level,
                    "task_type": "score",
                    "family": f"score_v8_{mechanism}",
                    "group_id": group,
                    "language": "en",
                    "split": role,
                    "source": VERSION,
                    "evaluation_role": role,
                    "render_template": f"v8.1-{mechanism}-{role}",
                    "audit_metadata": {
                        "mechanism": mechanism,
                        "record": base["record"],
                        "review_day": base["day"],
                        "site": base["site"],
                        "activity": base["activity"],
                        "oracle_version": VERSION,
                    },
                }
                row["input_sha256"] = digest(
                    {field: row[field] for field in INPUT_FIELDS}
                )
                validate_row(row, role)
                rows.append(row)
    return rows


def blind_packet(
    rows: list[dict[str, Any]], secret: bytes
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    packet, key = [], {}
    for group, items in sorted(groups.items()):
        alias = hmac.new(
            secret, f"blind-group\0{group}".encode(), hashlib.sha256
        ).hexdigest()[:16]
        ordered = sorted(
            items,
            key=lambda item: hmac.new(
                secret, f"blind-order\0{item['id']}".encode(), hashlib.sha256
            ).digest(),
        )
        blind_items = []
        for index, item in enumerate(ordered):
            review_id = f"{alias}-{index + 1}"
            blind_items.append(
                {
                    "review_id": review_id,
                    "state": item["state"],
                    "instructions": item["instructions"],
                    "options": item["options"],
                }
            )
            key[review_id] = {
                "source_id": item["id"],
                "label": item["label"],
                "group_id": group,
            }
        packet.append({"review_group": alias, "items": blind_items})
    return packet, key


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def write(secret_path: Path, output_dir: Path) -> dict[str, Any]:
    if secret_path.stat().st_mode & 0o077:
        raise PermissionError("Seed file must be mode 0600")
    secret = secret_path.read_bytes()
    if len(secret) != 32 or output_dir.exists():
        raise ValueError("Seed must be 32 bytes and output directory must not exist")
    output_dir.mkdir(parents=True, mode=0o700)
    manifest: dict[str, Any] = {
        "schema_version": VERSION,
        "status": "PENDING_INDEPENDENT_BLIND_REVIEW",
        "seed_sha256": hashlib.sha256(secret).hexdigest(),
        "roles": {},
    }
    for role in GROUPS:
        rows = build(secret, role)
        packet, key = blind_packet(rows, secret)
        data_path = output_dir / f"{role}.jsonl"
        packet_path = output_dir / f"{role}-blind-packet.jsonl"
        key_path = output_dir / f"{role}-sealed-key.json"
        _write_jsonl(data_path, rows)
        _write_jsonl(packet_path, packet)
        with key_path.open("x", encoding="utf-8") as stream:
            json.dump(key, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        key_path.chmod(0o600)
        manifest["roles"][role] = {
            "rows": len(rows),
            "groups": len(packet),
            "label_counts": dict(collections.Counter(row["label"] for row in rows)),
            "mechanism_counts": dict(
                collections.Counter(row["audit_metadata"]["mechanism"] for row in rows)
            ),
            "rows_sha256": file_sha256(data_path),
            "packet_sha256": file_sha256(packet_path),
            "key_sha256": file_sha256(key_path),
            "state_char_range": [
                min(len(row["state"]) for row in rows),
                max(len(row["state"]) for row in rows),
            ],
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path.chmod(0o600)
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
