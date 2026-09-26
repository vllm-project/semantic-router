"""Independent gold-blind reading of the frozen Score SELECT r2 packet.

Only the reviewer packet and manifest are inputs. This script does not import
the author builder, the two author oracles, the private join, or the key.
The mechanical reading needs a separate editorial seal before key comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PACKET_SHA = "091e3023b84d64131a72b23b90b3eacf837027ed23d58045de801e92a331f683"
MANIFEST_SHA = "ee3045f852478b743d24af9c076590b4cbec0bc60999e6dd8b44f308e55fbf9b"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def match(pattern: str, text: str) -> tuple[str, ...]:
    found = re.search(pattern, text)
    if found is None:
        raise ValueError(f"Visible state missing expected form: {pattern}")
    return found.groups()


def read_caps(state: str, language: str) -> tuple[int, dict]:
    if language == "en":
        patterns = [
            rf"{side} budget ledger: capacity (\d+) units; committed (\d+); requested (\d+); protected reserve (\d+)\."
            for side in ("East", "West")
        ]
    else:
        patterns = [
            rf"{side}预算账：容量 (\d+) 单位；已承诺 (\d+)；本次申请 (\d+)；保留额度 (\d+)。"
            for side in ("东区", "西区")
        ]
    ledgers = [tuple(map(int, match(pattern, state))) for pattern in patterns]
    margins = [
        capacity - committed - requested - reserve
        for capacity, committed, requested, reserve in ledgers
    ]
    feasible = [margin >= 0 for margin in margins]
    return sum(feasible), {"margins": margins, "feasible": feasible}


def read_coverage(state: str, language: str) -> tuple[int, dict]:
    if language == "en":
        requested = tuple(
            map(
                int,
                match(r"Requested inclusive period: day (\d+) to day (\d+)\.", state),
            )
        )
        patterns = [
            rf"Active office {side} lists days (\d+) to (\d+)\."
            for side in ("North", "South")
        ]
        if "A retired listing shows" not in state or "it has no effect" not in state:
            raise ValueError("Retired listing status unclear")
    else:
        requested = tuple(
            map(
                int, match(r"申请时段（首尾均计入）：第 (\d+) 天到第 (\d+) 天。", state)
            )
        )
        patterns = [
            rf"现行{side}机构登记：第 (\d+) 天到第 (\d+) 天。"
            for side in ("北区", "南区")
        ]
        if "已撤销的旧登记" not in state or "不再生效" not in state:
            raise ValueError("Retired listing status unclear")
    offices = [tuple(map(int, match(pattern, state))) for pattern in patterns]
    if requested[0] > requested[1] or any(start > end for start, end in offices):
        raise ValueError("Invalid visible interval")
    covered = [start <= requested[0] and end >= requested[1] for start, end in offices]
    return sum(covered), {"requested": requested, "covered": covered}


def read_quorum(state: str, language: str) -> tuple[int, dict]:
    origins: set[str] = set()
    sheet_count = 0
    signed_count = 0
    qualifying_reports = 0
    for line in state.splitlines():
        if language == "en" and line.startswith("Evidence sheet "):
            origin, signed, current, support = match(
                r"Evidence sheet \d+: origin ([A-Z]); signature (present|absent); (in force|retired); (supports|does not support) the request\.",
                line,
            )
            signed_ok = signed == "present"
            qualified = signed_ok and current == "in force" and support == "supports"
        elif language == "zh" and line.startswith("证据单 "):
            origin, signed, current, support = match(
                r"证据单 \d+：来源 ([A-Z])；(已签|未签)；(有效|失效)；(支持申请|不支持申请)。",
                line,
            )
            signed_ok = signed == "已签"
            qualified = signed_ok and current == "有效" and support == "支持申请"
        else:
            continue
        sheet_count += 1
        signed_count += signed_ok
        if qualified:
            qualifying_reports += 1
            origins.add(origin)
    if sheet_count != 6:
        raise ValueError("Expected six visible evidence sheets")
    return min(len(origins), 2), {
        "qualifying_origins": sorted(origins),
        "qualifying_reports": qualifying_reports,
        "signed_count": signed_count,
        "sheet_count": sheet_count,
    }


def read_waiver(state: str, language: str) -> tuple[int, dict]:
    if language == "en":
        day = int(match(r"Review is on day (\d+)\.", state)[0])
        lead, second, signed_on, expiry = match(
            r"Waiver entry: lead signature (present|absent); second signature (present|absent); signed day (\d+); final valid day (\d+)\.",
            state,
        )
        active = "The live hold is active." in state
        retired = "notice is retired" in state
        pending = "Follow-up check: pending." in state
        passed = "Follow-up check: passed." in state
        signers = lead == second == "present"
    else:
        day = int(match(r"审核日为第 (\d+) 天。", state)[0])
        lead, second, signed_on, expiry = match(
            r"豁免登记：主负责人(已签|未签)；另一负责人(已签|未签)；第 (\d+) 天签署；最后有效日是第 (\d+) 天。",
            state,
        )
        active = "现行暂停状态：生效。" in state
        retired = "现已撤销" in state
        pending = "后续核查：未完成。" in state
        passed = "后续核查：已完成。" in state
        signers = lead == second == "已签"
    if not active or not retired or pending == passed or int(signed_on) > int(expiry):
        raise ValueError("Hold/waiver state is incomplete")
    waiver_valid = signers and int(signed_on) <= day <= int(expiry)
    return ((2 if passed else 1) if waiver_valid else 0), {
        "waiver_valid": waiver_valid,
        "follow_up_passed": passed,
        "review_day": day,
        "signed_on": int(signed_on),
        "expiry": int(expiry),
    }


READERS = {
    "allocation_caps": read_caps,
    "inclusive_coverage": read_coverage,
    "independent_quorum": read_quorum,
    "waiver_precedence": read_waiver,
}


def review(packet: Path, manifest: Path) -> tuple[list[dict], list[dict], dict]:
    if sha(packet) != PACKET_SHA or sha(manifest) != MANIFEST_SHA:
        raise ValueError("Frozen gold-free input bytes differ")
    m = json.loads(manifest.read_text())
    if m["rows"] != 240 or m["groups"] != 80 or m["packet_sha256"] != PACKET_SHA:
        raise ValueError("Frozen roster differs")
    inputs = [json.loads(line) for line in packet.read_text().splitlines() if line]
    if len(inputs) != 240 or len({row["review_id"] for row in inputs}) != 240:
        raise ValueError("Duplicate or absent review IDs")
    rows: list[dict] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in inputs:
        if [option["key"] for option in row["options"]] != ["0", "1", "2"]:
            raise ValueError("Native Score choices differ")
        answer, facts = READERS[row["operation"]](row["state"], row["language"])
        decision = {
            "review_id": row["review_id"],
            "group_id": row["group_id"],
            "operation": row["operation"],
            "language": row["language"],
            "blind_answer": answer,
            "facts": facts,
            "row_issue": None,
        }
        rows.append(decision)
        grouped[row["group_id"]].append(decision)
    groups: list[dict] = []
    for gid, members in sorted(grouped.items()):
        flags: list[str] = []
        if len(members) != 3 or {r["blind_answer"] for r in members} != {0, 1, 2}:
            flags.append("missing_three_level_triplet")
        if len({(r["operation"], r["language"]) for r in members}) != 1:
            flags.append("mixed_operation_or_language")
        operation = members[0]["operation"]
        if operation == "allocation_caps":
            for pool in range(2):
                margins = {r["facts"]["margins"][pool] for r in members}
                if len(margins) != 2:
                    flags.append(f"pool_{pool + 1}_has_{len(margins)}_raw_margins")
        elif operation == "independent_quorum":
            if len({r["facts"]["signed_count"] for r in members}) != 1:
                flags.append("signed_count_changes")
            if len({r["facts"]["qualifying_reports"] for r in members}) == 3:
                flags.append("qualifying_report_count_solves_group")
        groups.append(
            {
                "group_id": gid,
                "operation": operation,
                "language": members[0]["language"],
                "answer_levels": sorted(r["blind_answer"] for r in members),
                "flags": flags,
            }
        )
    if len(groups) != 80:
        raise ValueError("Expected 80 independent source groups")
    counts = Counter((r["operation"], r["language"]) for r in rows)
    summary = {
        "review_schema": "decision20-score-select-r2-independent-blind/1",
        "packet_sha256": PACKET_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "rows_reviewed": len(rows),
        "groups_reviewed": len(groups),
        "operation_language_counts": {
            f"{operation}/{language}": count
            for (operation, language), count in sorted(counts.items())
        },
        "answer_balance": dict(Counter(str(r["blind_answer"]) for r in rows)),
        "group_flags": dict(
            Counter(flag for group in groups for flag in group["flags"])
        ),
        "editorial_status": "PENDING_MANUAL_EDITORIAL_REVIEW",
        "qualified_zh_review": False,
        "models_run": False,
        "author_source_or_gold_access": False,
    }
    return rows, groups, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, groups, summary = review(args.packet, args.manifest)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in (
        (
            "row_judgments.jsonl",
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        ),
        (
            "group_judgments.jsonl",
            "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in groups),
        ),
        ("summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n"),
    ):
        path = args.output / name
        path.write_text(content)
        path.chmod(0o600)


if __name__ == "__main__":
    main()
