"""Independent, gold-blind reading of the frozen Score SELECT r1 packet.

This intentionally uses only the reviewer packet/manifest. It does not import
the author builder, oracle or any target. It is a mechanical first pass;
editorial findings and language qualifications must be added before sealing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PACKET_SHA = "d38702cee5ef1b50458a4ee11d4370a7fda44013321b2dac43b40d01600ea88a"
MANIFEST_SHA = "a9868c0e8562c4450a2be57556327bdaba84719713cfbc77d8227a5ecff6ef59"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(pattern: str, value: str) -> tuple[str, ...]:
    match = re.search(pattern, value)
    if match is None:
        raise ValueError(
            f"Gold-free text did not match expected visible form: {pattern}"
        )
    return match.groups()


def answer(row: dict) -> tuple[int, dict]:
    state = row["state"]
    language = row["language"]
    operation = row["operation"]
    if operation == "inclusive_coverage":
        if language == "en":
            start, end = map(
                int, capture(r"Requested days: (\d+) through (\d+)", state)
            )
            first = tuple(
                map(
                    int,
                    capture(
                        r"Current first authority: days (\d+) through (\d+)", state
                    ),
                )
            )
            second = tuple(
                map(
                    int,
                    capture(
                        r"Current second authority: days (\d+) through (\d+)", state
                    ),
                )
            )
        else:
            start, end = map(int, capture(r"申请时段：第 (\d+) 天至第 (\d+) 天", state))
            first = tuple(
                map(int, capture(r"第一家现行机构：第 (\d+) 天至第 (\d+) 天", state))
            )
            second = tuple(
                map(int, capture(r"第二家现行机构：第 (\d+) 天至第 (\d+) 天", state))
            )
        if start > end or any(a > b for a, b in (first, second)):
            raise ValueError("Invalid inclusive interval")
        flags = (
            first[0] <= start and first[1] >= end,
            second[0] <= start and second[1] >= end,
        )
        return sum(flags), {"requested": [start, end], "coverage": list(flags)}

    if operation == "independent_quorum":
        qualified: set[str] = set()
        parsed: list[tuple[str, bool]] = []
        for line in state.splitlines():
            if language == "en" and line.startswith("Report "):
                lineage, signed, current, affirmative = capture(
                    r"origin lineage ([A-Z]); signed=(yes|no); current=(yes|no); affirmative=(yes|no)",
                    line,
                )
                good = signed == current == affirmative == "yes"
            elif language == "zh" and line.startswith("报告 "):
                lineage = capture(r"来源谱系 ([A-Z])", line)[0]
                good = "已签署" in line and "仍有效" in line and "；支持结论" in line
            else:
                continue
            parsed.append((lineage, good))
            if good:
                qualified.add(lineage)
        if len(parsed) != 5:
            raise ValueError("Expected five visible reports")
        return min(len(qualified), 2), {
            "qualifying_lineages": sorted(qualified),
            "report_count": len(parsed),
        }

    if operation == "allocation_caps":
        pools: list[tuple[int, int, int, int]] = []
        for line in state.splitlines():
            if language == "en" and line.startswith("Resource pool "):
                values = capture(
                    r"capacity (\d+); already committed (\d+); this request (\d+); mandatory reserve floor (\d+)",
                    line,
                )
            elif language == "zh" and line.startswith("资源池 "):
                values = capture(
                    r"容量 (\d+)；已承诺 (\d+)；本次申请 (\d+)；必须保留底线 (\d+)",
                    line,
                )
            else:
                continue
            pools.append(tuple(map(int, values)))
        if len(pools) != 2:
            raise ValueError("Expected two resource pools")
        flags = [
            committed + request + reserve <= capacity
            for capacity, committed, request, reserve in pools
        ]
        return sum(flags), {
            "feasible": flags,
            "margins": [c - a - r - f for c, a, r, f in pools],
        }

    if operation == "waiver_precedence":
        if language == "en":
            day = int(capture(r"Review day: (\d+)", state)[0])
            lead, second, start, expiry = capture(
                r"Waiver register: lead signed=(yes|no); second signer=(yes|no); signed on day (\d+); expires after day (\d+)",
                state,
            )
            passed = "Secondary check: passed." in state
            pending = "Secondary check: pending." in state
            active = "Current veto notice: active." in state
        else:
            day = int(capture(r"审核日：第 (\d+) 天", state)[0])
            lead, second, start, expiry = capture(
                r"豁免登记：主签署人(已签|未签)；第二签署人(已签|未签)；第 (\d+) 天签署；有效至第 (\d+) 天（含当日）",
                state,
            )
            lead = "yes" if lead == "已签" else "no"
            second = "yes" if second == "已签" else "no"
            passed = "次级核查：已通过。" in state
            pending = "次级核查：待完成。" in state
            active = "现行否决通知：生效。" in state
        if not active or passed == pending or int(start) > int(expiry):
            raise ValueError("Visible veto or secondary-check contract differs")
        waiver_valid = lead == second == "yes" and int(start) <= day <= int(expiry)
        return (2 if passed else 1) if waiver_valid else 0, {
            "waiver_valid": waiver_valid,
            "secondary_passed": passed,
            "review_day": day,
            "expiry": int(expiry),
        }

    raise ValueError(f"Unknown operation {operation}")


def review(packet: Path, manifest: Path) -> tuple[list[dict], list[dict], dict]:
    if digest(packet) != PACKET_SHA or digest(manifest) != MANIFEST_SHA:
        raise ValueError("Frozen reviewer packet or manifest hash differs")
    meta = json.loads(manifest.read_text())
    if (
        meta["rows"] != 240
        or meta["groups"] != 80
        or meta["packet_sha256"] != PACKET_SHA
    ):
        raise ValueError("Frozen reviewer manifest counts differ")
    rows = [json.loads(line) for line in packet.read_text().splitlines() if line]
    if len(rows) != 240 or len({row["review_id"] for row in rows}) != 240:
        raise ValueError("Frozen reviewer row roster differs")
    decisions: list[dict] = []
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in rows:
        if [option["key"] for option in row["options"]] != ["0", "1", "2"]:
            raise ValueError("Native Score criteria differ")
        value, facts = answer(row)
        record = {
            "review_id": row["review_id"],
            "group_id": row["group_id"],
            "language": row["language"],
            "operation": row["operation"],
            "blind_answer": value,
            "directly_solvable": True,
            "ambiguity": False,
            "native_valid": True,
            "facts": facts,
        }
        decisions.append(record)
        grouped[row["group_id"]].append((row, record))
    if len(grouped) != 80:
        raise ValueError("Frozen reviewer group roster differs")
    groups: list[dict] = []
    for group_id, members in sorted(grouped.items()):
        if len(members) != 3:
            raise ValueError("Incomplete reviewer triplet")
        ops = {row["operation"] for row, _ in members}
        langs = {row["language"] for row, _ in members}
        answers = {record["blind_answer"] for _, record in members}
        flags: list[str] = []
        if len(ops) != 1 or len(langs) != 1:
            flags.append("mixed_operation_or_language")
        if answers != {0, 1, 2}:
            flags.append("incomplete_level_triplet")
        groups.append(
            {
                "group_id": group_id,
                "operation": next(iter(ops)),
                "language": next(iter(langs)),
                "answer_levels": sorted(answers),
                "flags": flags,
            }
        )
    count = Counter((row["operation"], row["language"]) for row in rows)
    summary = {
        "review_schema": "decision20-score-select-r1-independent-blind/1",
        "packet_sha256": PACKET_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "rows_reviewed": len(decisions),
        "groups_reviewed": len(groups),
        "operation_language_counts": {
            f"{operation_name}/{language_name}": n
            for (operation_name, language_name), n in sorted(count.items())
        },
        "parsed_rows": len(decisions),
        "ambiguous_rows": sum(bool(row["ambiguity"]) for row in decisions),
        "incomplete_triplets": sum(bool(group["flags"]) for group in groups),
        "editorial_status": "PENDING_MANUAL_EDITORIAL_REVIEW",
        "qualified_zh_review": False,
        "models_run": False,
        "gold_or_author_source_access": False,
    }
    return decisions, groups, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decisions, groups, summary = review(args.packet, args.manifest)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in (
        (
            "row_judgments.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions),
        ),
        (
            "group_judgments.jsonl",
            "".join(json.dumps(group, ensure_ascii=False) + "\n" for group in groups),
        ),
        ("summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n"),
    ):
        path = args.output / name
        path.write_text(content)
        path.chmod(0o600)


if __name__ == "__main__":
    main()
