"""Fresh, CPU-only Score v8.2 quality pilot; never a training admission.

The v8.1 outputs and keys are immutable. This version repairs two observed
shortcuts, creates a distinct seed namespace, and leaves independent blind
review as a separate required step.
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

from training.data import score_v8_pilot as previous
from training.model.data import INPUT_FIELDS, digest, file_sha256, validate_row

VERSION = "decision2-score-v8.2-multimechanism-pilot/1"
MECHANISMS = previous.MECHANISMS
GROUPS = previous.GROUPS
LEVELS = previous.LEVELS
OPTIONS = previous.OPTIONS

# Each document has a different source format and actual authority/evidence
# context. The common decisive facts are rendered into different locations.
DOSSIERS = (
    (
        "Transit depot parts acceptance file",
        "Depot minutes",
        "The receiving office distinguishes a supplier's shipment promise from a board-approved delivery obligation. The earlier schedule remains on file because it explains the purchase order, but only the later signed instrument changes the operative date. Staff filed an unsigned supplier proposal separately so that its suggested date cannot be mistaken for approval.",
        "Change register",
        "The change register identifies each order by its full reference. A notice for another vehicle batch may look similar, but it carries no authority for this order. The clerk checked the signature ledger before carrying a changed date into the receiving queue; a phone inquiry alone was not entered as an amendment.",
        "Receiving desk",
        "The desk records physical arrival apart from acknowledgment. A courier timestamp can show when a crate reached the depot but does not certify that the receiving official accepted the documentation. The desk therefore keeps the confirmation field alongside the arrival date, and unresolved paperwork does not turn an on-time delivery into a confirmed one.",
    ),
    (
        "Museum conservation equipment acquisition",
        "Acquisition folder",
        "The registrar preserved the original accession schedule to explain the planned installation window. Changes to the vendor's obligation require a countersigned addendum indexed by acquisition reference. A draft conservation note circulated later, but circulation did not amend the contract or extend the delivery date.",
        "Signed instruments",
        "The registrar compared document scope before entering any later date. Two nearby exhibit projects use the same supplier and share a filing cabinet, so their addenda cannot be applied to this acquisition. The controlling record is the signed change for the named unit; an unsigned planning sheet has no equivalent status.",
        "Intake evidence",
        "The intake desk distinguishes a crate's arrival from its verification. A guard's gate entry supplies an arrival day, while the conservation team supplies the confirmation status after checking the seal and accompanying inventory. An unconfirmed arrival remains unresolved even if it occurred before the signed deadline.",
    ),
    (
        "Watershed sensor kit delivery dossier",
        "Field order history",
        "The original dispatch calendar was printed for the sampling season and remains attached for audit. Weather planning messages proposed several alternatives, but the procurement officer treated only a signed order amendment as capable of replacing the original deadline. Record numbers matter because multiple stations received similar kits.",
        "Authority trail",
        "An appendix carries amendments for neighboring stations as well as the named kit. The field coordinator checked the approved signature and the kit reference before updating the route schedule. A draft routing note for the named kit was never signed, so its proposed day cannot supersede the executed amendment.",
        "Handover log",
        "Drivers record gate arrival separately from handover confirmation. A load can reach the watershed store on time yet remain unconfirmed while serial numbers are checked. The signed deadline is compared with the arrival day first; confirmation then decides whether an on-time handover is complete.",
    ),
    (
        "School kitchen refrigeration contract",
        "Contract chronology",
        "The district retained the first purchase schedule as a historical attachment. The facilities lead requested a later date, but the request itself did not change the delivery obligation; a signed amendment was needed. Similar refrigeration units at another school have separate order references and independent delivery notices.",
        "Approval ledger",
        "The signed change record for the named unit is filed with the governing contract. An unsigned maintenance draft mentions a different proposed date and remains nonbinding. The reviewer must distinguish that draft from the approved instrument and avoid importing a deadline from the other school's order.",
        "Acceptance record",
        "The receiving clerk noted when the unit reached the loading bay and whether the installation packet was confirmed. Arrival and confirmation are separate facts: a timely but unconfirmed unit is not yet accepted, while a confirmed arrival after the operative deadline remains late. Both fields belong to the named order only.",
    ),
    (
        "Civic archive digitization equipment file",
        "Order index",
        "The archive keeps the original vendor schedule beside later approvals so an auditor can reconstruct the sequence. A proposed change in an unsigned email was not adopted. The project register lists several scanners with similar labels, making the complete record reference essential when reading a later deadline.",
        "Amendment register",
        "A signed amendment controls only the scanner identified in its text. The register also contains changes for unrelated scanners that arrived in the same week. Staff relied on the executed record rather than a draft timeline, and they kept both documents to explain why the working date differs from the older schedule.",
        "Receipt reconciliation",
        "The courier receipt supplies an arrival day, and the archive desk records whether the accompanying equipment list was confirmed. A box's presence alone is not confirmation. The review compares the named scanner's arrival with its approved deadline and then checks the evidence status for an on-time delivery.",
    ),
)


def _rng(secret: bytes, role: str, mechanism: str, index: int) -> random.Random:
    payload = f"{VERSION}\0{role}\0{mechanism}\0{index}".encode()
    value = hmac.new(secret, payload, hashlib.sha256).digest()
    return random.Random(int.from_bytes(value[:16], "big"))


def _evidence_facts(
    base: dict[str, Any], role: str, index: int, level: int, rng: random.Random
) -> dict[str, Any]:
    facts = dict(base)
    phase = (index + int(role == "select")) % 2
    if phase == 0:
        target = (
            ("normal", "outage"),
            ("outage", "unavailable"),
            ("outage", "outage"),
        )[level]
    else:
        target = (
            ("outage", "normal"),
            ("unavailable", "outage"),
            ("outage", "outage"),
        )[level]
    facts["dispatch"], facts["telemetry"] = target
    # Two stable distractors do not complement the target answer. A third
    # unrelated observation varies independently, defeating simple global
    # status counts without defining the target label.
    extra = (
        "R-"
        + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=4))
        + str(rng.randrange(100, 999))
    )
    while extra in {base["record"], base["other"], base["other2"]}:
        extra = (
            "R-"
            + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=4))
            + str(rng.randrange(100, 999))
        )
    statuses = ("normal", "unavailable", "outage")
    facts["evidence_rows"] = [
        (base["record"], *target),
        (base["other"], "normal", "outage"),
        (base["other2"], "outage", "unavailable"),
        (extra, rng.choice(statuses), rng.choice(statuses)),
    ]
    rng.shuffle(facts["evidence_rows"])
    return facts


def _long_facts(
    base: dict[str, Any], role: str, index: int, rng: random.Random
) -> dict[str, Any]:
    facts = dict(base)
    facts["dossier_style"] = index + (3 if role == "select" else 0)
    facts["organization"] = DOSSIERS[facts["dossier_style"]][0]
    facts["draft_deadline"] = facts["amended_deadline"] + rng.choice((-3, 2, 4))
    return facts


def _render_evidence(f: dict[str, Any]) -> tuple[str, str]:
    start, end = f["window"]
    notes = []
    for record, dispatch, telemetry in f["evidence_rows"]:
        notes.append(
            f"Dispatch diary for {record}, hours {start}-{end}: {dispatch} observed.\n"
            f"Meter trace for {record}, hours {start}-{end}: {telemetry} observed."
        )
    state = (
        f"Claim at the {f['organization']}: interruption alleged for {f['record']} during hours {start}-{end}.\n"
        "The dispatch diary and meter trace are independently maintained. Match the named site and time window; nearby entries concern other sites.\n"
        + "\n".join(notes)
    )
    instruction = (
        "For the named site and same hours, normal service in either independent source contradicts the claim. "
        "If neither contradicts but one is unavailable, corroboration is incomplete. "
        "Two outage observations corroborate it."
    )
    return state, instruction


def _render_long(f: dict[str, Any]) -> tuple[str, str]:
    title, heading1, intro, heading2, authority, heading3, intake = DOSSIERS[
        f["dossier_style"]
    ]
    record = f["record"]
    old = f"The original signed schedule for {record} set a delivery deadline of day {f['deadline']}."
    amendment = f"A later signed amendment for {record} sets the controlling deadline to day {f['amended_deadline']}."
    receipt = f"The delivery notice for {record} reports arrival on day {f['arrival']}; evidence status: {f['arrival_status']}."
    draft = f"An unsigned draft for {record} proposed day {f['draft_deadline']}; it was not executed."
    decoys = [
        f"A separate signed amendment for {item} sets day {f['amended_deadline']}; its delivery notice reports arrival on day {arrival}, evidence status: {status}."
        for item, arrival, status in f["other_delivery"]
    ]
    sections = [
        f"{heading1}. {intro} {old} {decoys[0]}",
        f"{heading2}. {authority} {amendment} {draft}",
        f"{heading3}. {intake} {receipt} {decoys[1]}",
    ]
    # Vary the physical placement of the controlling evidence without changing
    # its authority or selecting a favourable label.
    shift = f["target_position"]
    sections = sections[shift:] + sections[:shift]
    state = f"{title}: decide delivery for {record}.\n\n" + "\n\n".join(sections)
    instruction = (
        "Use the controlling signed amendment for the named record, not the older schedule, an unsigned draft, or another record. "
        "Arrival after that deadline is late; an on-time but unconfirmed arrival remains unresolved; "
        "confirmed on-time arrival is timely."
    )
    return state, instruction


def _rendered_evidence_oracle(state: str, record: str) -> int:
    matches = re.findall(
        r"(?m)^Dispatch diary for (R-[A-Z0-9]+), hours (\d+-\d+): (outage|normal|unavailable) observed\.\n"
        r"Meter trace for \1, hours \2: (outage|normal|unavailable) observed\.$",
        state,
    )
    if len(matches) != 4:
        raise ValueError("Evidence source notes cannot be parsed")
    target = [row for row in matches if row[0] == record]
    if len(target) != 1:
        raise ValueError("Named evidence source is missing or duplicated")
    _, _, dispatch, telemetry = target[0]
    return previous.oracle(
        {"dispatch": dispatch, "telemetry": telemetry}, "evidence_sufficiency"
    )


def build(secret: bytes, role: str) -> list[dict[str, Any]]:
    if len(secret) != 32 or role not in GROUPS:
        raise ValueError("Need a private 32-byte seed and known role")
    rows = []
    for mechanism in MECHANISMS:
        for index in range(GROUPS[role]):
            rng = _rng(secret, role, mechanism, index)
            base = previous._base(rng, role, mechanism, index)
            group = f"d2sv82_{role}_{mechanism}_{index:02d}"
            for level in LEVELS:
                facts = (
                    _evidence_facts(base, role, index, level, rng)
                    if mechanism == "evidence_sufficiency"
                    else previous._facts(base, mechanism, level, rng)
                )
                if mechanism == "long_memo":
                    facts = _long_facts(facts, role, index, rng)
                    state, instructions = _render_long(facts)
                elif mechanism == "evidence_sufficiency":
                    state, instructions = _render_evidence(facts)
                else:
                    state, instructions = previous.render(facts, mechanism, role)
                options = [
                    {"key": str(i), "description": description}
                    for i, description in enumerate(OPTIONS[mechanism])
                ]
                rendered = (
                    _rendered_evidence_oracle(state, base["record"])
                    if mechanism == "evidence_sufficiency"
                    else previous.rendered_oracle(
                        state,
                        mechanism,
                        base["record"],
                        base["day"],
                        site=base["site"],
                        activity=base["activity"],
                    )
                )
                if previous.oracle(facts, mechanism) != level or rendered != level:
                    raise AssertionError(
                        f"Structured/rendered oracle disagreement for {group}"
                    )
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
                    "render_template": f"v8.2-{mechanism}-{role}",
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
        packet, key = previous.blind_packet(rows, secret)
        data_path = output_dir / f"{role}.jsonl"
        packet_path = output_dir / f"{role}-blind-packet.jsonl"
        key_path = output_dir / f"{role}-sealed-key.json"
        previous._write_jsonl(data_path, rows)
        previous._write_jsonl(packet_path, packet)
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
