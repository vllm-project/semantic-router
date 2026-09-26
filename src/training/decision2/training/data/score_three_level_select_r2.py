"""Freeze independent, gold-blind three-level Score SELECT r2.

This is a SELECT diagnostic, never a release benchmark or TRAIN curriculum.
The local code has only rule and generation logic. A fresh private seed and
review salt, source records, gold and reviewer packet stay off source control.
No model is loaded or run here.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import hmac
import json
import os
import random
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.data.build_score_curriculum_v6 import REQUIRED_PROTECTED_ROLES
from training.data import score_three_level_select_r2_oracle as oracle
from training.data import score_three_level_select_r2_render_oracle as render_oracle
from training.model.data import digest, validate_row
from training.model.infer import question_to_row

VERSION = "decision20-score-three-level-select/2"
METHOD_SHA256 = "96cd806cd2bfcfba8c56dc750fb225b044b44f5873bb0bf394e413d7bdfeed61"
FROZEN_SCORE_V6_SHA256 = (
    "6aee966cc5499a87d2a77241676586c9c9f801b3c662a078daf025f001169f54"
)
FROZEN_R1_PACKET_SHA256 = (
    "d38702cee5ef1b50458a4ee11d4370a7fda44013321b2dac43b40d01600ea88a"
)
OPS = (
    "waiver_precedence",
    "inclusive_coverage",
    "independent_quorum",
    "allocation_caps",
)
GROUPS_PER_OP = 20
FROZEN_PARENT_SHA256 = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
SCENES = (
    ("city greenhouse access", "城市温室准入"),
    ("mountain rescue exercise", "山地救援演练"),
    ("mobile clinic dispatch", "流动诊所调度"),
    ("harbour ferry rehearsal", "港口渡轮演练"),
    ("library conservation shift", "图书保护班次"),
    ("film festival equipment", "电影节器材"),
    ("floodgate inspection", "防洪闸检查"),
    ("planetarium night booking", "天文馆夜场预约"),
    ("orchestra tour permit", "乐团巡演许可"),
    ("aqueduct maintenance", "引水渠维护"),
    ("ski patrol roster", "滑雪巡逻排班"),
    ("urban bee survey", "城市蜜蜂调查"),
    ("solar farm visit", "太阳能场站参观"),
    ("historic tram exhibit", "历史电车展览"),
    ("food safety audit", "食品安全审查"),
    ("community theatre invoice", "社区剧院账单"),
    ("geology lab transfer", "地质实验室转运"),
    ("tunnel ventilation test", "隧道通风测试"),
    ("island clinic boat", "岛屿诊所渡船"),
    ("ceramic kiln reservation", "陶瓷窑炉预约"),
)
OPTION_TEXT = {
    "en": ("Level 0", "Level 1", "Level 2"),
    "zh": ("0 档", "1 档", "2 档"),
}
RULE = {
    ("waiver_precedence", "en"): (
        "Decide using the live hold notice; the old notice has been superseded. "
        "A live hold is lifted only if both people signed the waiver and the "
        "review date lies between signing and expiry, including both dates. "
        "An unlifted hold is 0; after it is lifted, an open follow-up is 1 and "
        "a completed follow-up is 2."
    ),
    ("waiver_precedence", "zh"): (
        "以现行暂停通知为准，旧通知已经失效。现行暂停只有在两位负责人都签署豁免，且审核日在签署日到截止日的闭区间内时才能解除。"
        "未解除判 0；解除后，后续核查未完成判 1，已完成判 2。"
    ),
    ("inclusive_coverage", "en"): (
        "The requested period includes both end dates. An active office covers "
        "it only when that office's period starts on or before the request and "
        "ends on or after it. Disregard the retired listing. Return 0, 1 or 2 "
        "for the number of active offices that fully cover the request."
    ),
    ("inclusive_coverage", "zh"): (
        "申请时段首尾两天都计入。现行机构只有在其时段开始不晚于申请开始、结束不早于申请结束时，才完整覆盖申请。"
        "忽略已撤销的旧时段。完整覆盖的现行机构为零、一、两家时，分别选 0、1、2。"
    ),
    ("independent_quorum", "en"): (
        "A report qualifies only when it supports the request, remains in force "
        "and bears a signature. Several qualifying reports from one origin "
        "count as one independent origin. Return 0 for none, 1 for one, and 2 "
        "for at least two independent origins."
    ),
    ("independent_quorum", "zh"): (
        "报告须支持申请、仍有效且已签署才合格。同一来源出具的多份合格报告只算一个独立来源。"
        "合格独立来源为零个选 0，一个选 1，至少两个选 2。"
    ),
    ("allocation_caps", "en"): (
        "Check the two budgets separately. A budget can fund the request when "
        "committed units plus requested units plus its protected reserve are "
        "no greater than its capacity. Equality is allowed. Return 0, 1 or 2 "
        "for the number of budgets that can fund it."
    ),
    ("allocation_caps", "zh"): (
        "两个预算分别核算。某预算的已承诺用量、本次申请用量和保留额度之和不大于容量时，该预算可以承担申请；恰好相等也可以。"
        "可承担申请的预算有零个、一个、两个时，分别选 0、1、2。"
    ),
}


def _private_bytes(path: Path) -> bytes:
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"Private input file has broad permissions: {path.name}")
    value = path.read_bytes()
    if len(value) != 32:
        raise ValueError("A fresh 32-byte private seed and review salt are required")
    return value


def _hmac(secret: bytes, domain: str, value: str) -> str:
    return hmac.new(secret, f"{domain}\0{value}".encode(), hashlib.sha256).hexdigest()


def _rng(secret: bytes, operation: str, index: int) -> random.Random:
    return random.Random(int(_hmac(secret, operation, str(index))[:16], 16))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        for row in rows
    )


def _write(path: Path, body: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)


def _base_facts(operation: str, rng: random.Random) -> dict[str, Any]:
    if operation == "waiver_precedence":
        day = rng.randint(11, 38)
        return {
            "review_day": day,
            "veto_active": True,
            "waiver": {
                "lead_signed": True,
                "second_signed": True,
                "signed_day": day - rng.randint(1, 4),
                "expires_day": day + rng.randint(0, 3),
            },
            "secondary_check": "pending",
            "archived_notice": "no veto" if rng.randrange(2) else "veto",
        }
    if operation == "inclusive_coverage":
        start = rng.randint(13, 33)
        end = start + rng.randint(2, 7)
        covering = [
            (start - rng.randint(0, 2), end + rng.randint(0, 2)) for _ in range(2)
        ]
        missing = [
            (start + rng.randint(1, 2), end + rng.randint(0, 2)),
            (start - rng.randint(0, 2), end - rng.randint(1, 2)),
        ]
        rng.shuffle(missing)
        return {
            "request": [start, end],
            "current_windows": {"first": list(missing[0]), "second": list(missing[1])},
            "archived_window": [start - 3, end + 3],
            "_covering": [list(pair) for pair in covering],
        }
    if operation == "independent_quorum":
        lineages = rng.sample(("J", "K", "L", "U", "V", "W", "X", "Y"), 5)
        reports = [
            {
                "lineage": lineages[0],
                "signed": False,
                "current": True,
                "affirmative": True,
            },
            {
                "lineage": lineages[0],
                "signed": False,
                "current": True,
                "affirmative": True,
            },
            {
                "lineage": lineages[1],
                "signed": False,
                "current": True,
                "affirmative": True,
            },
            {
                "lineage": lineages[2],
                "signed": False,
                "current": True,
                "affirmative": False,
            },
            {
                "lineage": lineages[3],
                "signed": False,
                "current": False,
                "affirmative": True,
            },
            {
                "lineage": lineages[4],
                "signed": False,
                "current": False,
                "affirmative": False,
            },
        ]
        rng.shuffle(reports)
        return {"reports": reports}
    if operation == "allocation_caps":
        pools: dict[str, dict[str, int]] = {}
        good_requests = {}
        for name in ("first", "second"):
            capacity = rng.randint(20, 39)
            committed = rng.randint(3, 9)
            reserve = rng.randint(2, 6)
            available = capacity - committed - reserve
            bad_margin = -rng.randint(1, 3)
            good_margin = rng.choice((0, 0, 1, 2))
            pools[name] = {
                "capacity": capacity,
                "committed": committed,
                "reserve_floor": reserve,
                "request": available - bad_margin,
            }
            good_requests[name] = available - good_margin
        return {"pools": pools, "_good_requests": good_requests}
    raise ValueError(operation)


def _variant_facts(
    operation: str, base: dict[str, Any], level: int, rng: random.Random, index: int
) -> dict[str, Any]:
    facts = copy.deepcopy(base)
    if operation == "waiver_precedence":
        if level == 0:
            day = facts["review_day"]
            failure = rng.choice(("late_signature", "expired", "missing_cosigner"))
            if failure == "late_signature":
                facts["waiver"]["signed_day"] = day + 1
                facts["waiver"]["expires_day"] = day + 2
            elif failure == "expired":
                facts["waiver"]["expires_day"] = day - 1
            else:
                facts["waiver"]["second_signed"] = False
        if level == 2:
            facts["secondary_check"] = "passed"
    elif operation == "inclusive_coverage":
        covering = facts.pop("_covering")
        names = ["first", "second"] if index % 2 == 0 else ["second", "first"]
        for name in names[:level]:
            facts["current_windows"][name] = covering[0 if name == "first" else 1]
    elif operation == "independent_quorum":
        qualifying = [
            report
            for report in facts["reports"]
            if report["current"] and report["affirmative"]
        ]
        decoys = [
            report
            for report in facts["reports"]
            if not (report["current"] and report["affirmative"])
        ]
        repeated = collections.Counter(report["lineage"] for report in qualifying)
        same_lineage = next(name for name, count in repeated.items() if count == 2)
        same = [report for report in qualifying if report["lineage"] == same_lineage]
        different = next(
            report for report in qualifying if report["lineage"] != same_lineage
        )
        if len(decoys) != 3 or len(same) != 2:
            raise AssertionError("Quorum source inventory must be 3+3")
        if level == 0:
            selected = decoys
        elif level == 1:
            selected = [*same, rng.choice(decoys)]
        else:
            selected = [rng.choice(same), different, rng.choice(decoys)]
        for report in selected:
            report["signed"] = True
    elif operation == "allocation_caps":
        good_requests = facts.pop("_good_requests")
        names = ["first", "second"] if index % 2 == 0 else ["second", "first"]
        for name in names[:level]:
            facts["pools"][name]["request"] = good_requests[name]
    else:
        raise ValueError(operation)
    return facts


def _render(
    operation: str, facts: dict[str, Any], locale: str, scene: str, code: str
) -> str:
    """Present operative case facts without target, oracle or source ID metadata."""
    if locale == "en":
        header = f"Case {code} concerns {scene}."
    else:
        header = f"受理号 {code}，事项：{scene}。"
    if operation == "waiver_precedence":
        waiver = facts["waiver"]
        if locale == "en":
            parts = [
                header,
                f"Review is on day {facts['review_day']}. The archived hold notice said {facts['archived_notice']}; that notice is retired.",
                f"The live hold is {'active' if facts['veto_active'] else 'inactive'}.",
                f"Waiver entry: lead signature {'present' if waiver['lead_signed'] else 'absent'}; "
                f"second signature {'present' if waiver['second_signed'] else 'absent'}; "
                f"signed day {waiver['signed_day']}; final valid day {waiver['expires_day']}.",
                f"Follow-up check: {facts['secondary_check']}.",
            ]
        else:
            parts = [
                header,
                f"审核日为第 {facts['review_day']} 天。旧暂停通知记为{'无暂停' if facts['archived_notice'] == 'no veto' else '有暂停'}，现已撤销。",
                f"现行暂停状态：{'生效' if facts['veto_active'] else '未生效'}。",
                f"豁免登记：主负责人{'已签' if waiver['lead_signed'] else '未签'}；另一负责人{'已签' if waiver['second_signed'] else '未签'}；"
                f"第 {waiver['signed_day']} 天签署；最后有效日是第 {waiver['expires_day']} 天。",
                f"后续核查：{'未完成' if facts['secondary_check'] == 'pending' else '已完成'}。",
            ]
    elif operation == "inclusive_coverage":
        first, last = facts["request"]
        retired = facts["archived_window"]
        offices = facts["current_windows"]
        if locale == "en":
            parts = [
                header,
                f"Requested inclusive period: day {first} to day {last}.",
                f"A retired listing shows days {retired[0]} to {retired[1]}; it has no effect.",
                f"Active office North lists days {offices['first'][0]} to {offices['first'][1]}.",
                f"Active office South lists days {offices['second'][0]} to {offices['second'][1]}.",
            ]
        else:
            parts = [
                header,
                f"申请时段（首尾均计入）：第 {first} 天到第 {last} 天。",
                f"已撤销的旧登记列出第 {retired[0]} 天到第 {retired[1]} 天，不再生效。",
                f"现行北区机构登记：第 {offices['first'][0]} 天到第 {offices['first'][1]} 天。",
                f"现行南区机构登记：第 {offices['second'][0]} 天到第 {offices['second'][1]} 天。",
            ]
    elif operation == "independent_quorum":
        parts = [header]
        for index, report in enumerate(facts["reports"], 1):
            if locale == "en":
                parts.append(
                    f"Evidence sheet {index}: origin {report['lineage']}; "
                    f"signature {'present' if report['signed'] else 'absent'}; "
                    f"{'in force' if report['current'] else 'retired'}; "
                    f"{'supports' if report['affirmative'] else 'does not support'} the request."
                )
            else:
                parts.append(
                    f"证据单 {index}：来源 {report['lineage']}；"
                    f"{'已签' if report['signed'] else '未签'}；"
                    f"{'有效' if report['current'] else '失效'}；"
                    f"{'支持' if report['affirmative'] else '不支持'}申请。"
                )
    elif operation == "allocation_caps":
        parts = [header]
        for name, display in (("first", "East"), ("second", "West")):
            pool = facts["pools"][name]
            if locale == "en":
                parts.append(
                    f"{display} budget ledger: capacity {pool['capacity']} units; "
                    f"committed {pool['committed']}; requested {pool['request']}; "
                    f"protected reserve {pool['reserve_floor']}."
                )
            else:
                direction = "东区" if name == "first" else "西区"
                parts.append(
                    f"{direction}预算账：容量 {pool['capacity']} 单位；"
                    f"已承诺 {pool['committed']}；本次申请 {pool['request']}；"
                    f"保留额度 {pool['reserve_floor']}。"
                )
    else:
        raise ValueError(operation)
    return "\n".join(parts)


def generate(
    seed: bytes, salt: bytes
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(seed) != len(salt) or len(seed) != 32:
        raise ValueError("Private seed and review salt must each be 32 bytes")
    specs = []
    rows = []
    for operation in OPS:
        for index in range(GROUPS_PER_OP):
            locale = "en" if index < 16 else "zh"
            rng = _rng(seed, operation, index)
            group = f"s3g2-{_hmac(salt, 'r2-group', f'{operation}/{index}')[:24]}"
            scene = SCENES[index][0 if locale == "en" else 1]
            code = _hmac(salt, "r2-case", f"{operation}/{index}")[:10].upper()
            base = _base_facts(operation, rng)
            level_order = list(range(3))
            rng.shuffle(level_order)
            source = {
                "group_id": group,
                "operation": operation,
                "locale": locale,
                "scene": scene,
                "variants": [],
            }
            for slot, level in enumerate(level_order):
                variant_rng = _rng(seed, f"{operation}/{index}", level)
                facts = _variant_facts(operation, base, level, variant_rng, index)
                actual = oracle.score(operation, facts)
                if actual != level:
                    raise AssertionError(
                        f"Mechanical oracle disagrees: {operation}/{index}/{level}: {actual}"
                    )
                row_id = (
                    f"s3r2-{_hmac(salt, 'r2-row', f'{operation}/{index}/{slot}')[:24]}"
                )
                state = _render(operation, facts, locale, scene, code)
                if render_oracle.score(operation, state, locale) != level:
                    raise AssertionError(
                        "Independent visible-text derivation disagrees"
                    )
                options = [
                    {"key": str(key), "description": OPTION_TEXT[locale][key]}
                    for key in range(3)
                ]
                row = {
                    "id": row_id,
                    "group_id": group,
                    "family": f"score_select_{operation}",
                    "language": locale,
                    "split": "select",
                    "evaluation_role": "select",
                    "source": "decision2_internal_three_level_score_select_r2",
                    "render_template": "score-three-level-select/2",
                    "task_type": "score",
                    "state": state,
                    "instructions": RULE[(operation, locale)],
                    "options": options,
                    "label": level,
                    "audit_metadata": {"operation": operation, "case_index": index},
                }
                row["input_sha256"] = digest(
                    {
                        field: row[field]
                        for field in ("state", "instructions", "options", "task_type")
                    }
                )
                validate_row(row, "select")
                projected = question_to_row(
                    {"id": row_id, "state": state},
                    "decision",
                    {
                        "type": "score",
                        "instructions": row["instructions"],
                        "criteria": [option["description"] for option in options],
                    },
                )
                if projected["options"] != options or projected["task_type"] != "score":
                    raise AssertionError(
                        "Native Score adapter changes the declared levels"
                    )
                rows.append(row)
                source["variants"].append({"row_id": row_id, "facts": facts})
            specs.append(source)
    if len(rows) != 240 or len(specs) != 80:
        raise AssertionError("Three-level Score SELECT cardinality changed")
    return specs, rows


def _one_field_features(operation: str, facts: dict[str, Any]) -> dict[str, str]:
    """Predeclared single-field and one-subrecord shortcut projections."""
    raw: dict[str, Any] = {}
    if operation == "waiver_precedence":
        waiver = facts["waiver"]
        raw.update(
            review_day=facts["review_day"],
            veto_active=facts["veto_active"],
            archived_notice=facts["archived_notice"],
            waiver=waiver,
            secondary_check=facts["secondary_check"],
            signed_day_offset=waiver["signed_day"] - facts["review_day"],
            expiry_day_offset=waiver["expires_day"] - facts["review_day"],
            waiver_valid=(
                waiver["lead_signed"]
                and waiver["second_signed"]
                and waiver["signed_day"] <= facts["review_day"] <= waiver["expires_day"]
            ),
        )
        raw.update({f"waiver_{key}": value for key, value in waiver.items()})
    elif operation == "inclusive_coverage":
        raw.update(request=facts["request"], archived_window=facts["archived_window"])
        for name, window in facts["current_windows"].items():
            raw[f"{name}_window"] = window
            raw[f"{name}_start"] = window[0]
            raw[f"{name}_end"] = window[1]
            raw[f"{name}_start_gap"] = window[0] - facts["request"][0]
            raw[f"{name}_end_gap"] = window[1] - facts["request"][1]
            raw[f"{name}_covers"] = (
                window[0] <= facts["request"][0] and facts["request"][1] <= window[1]
            )
    elif operation == "independent_quorum":
        reports = facts["reports"]
        raw["report_count"] = len(reports)
        raw["signed_count"] = sum(report["signed"] for report in reports)
        raw["signed_qualifying_count"] = sum(
            report["signed"] and report["current"] and report["affirmative"]
            for report in reports
        )
        raw["origin_multiplicity"] = sorted(
            collections.Counter(report["lineage"] for report in reports).values()
        )
        for index, report in enumerate(reports):
            raw[f"report_{index}"] = report
            for key, value in report.items():
                raw[f"report_{index}_{key}"] = value
    elif operation == "allocation_caps":
        for name, pool in facts["pools"].items():
            margin = (
                pool["capacity"]
                - pool["committed"]
                - pool["request"]
                - pool["reserve_floor"]
            )
            raw[f"{name}_pool"] = pool
            raw[f"{name}_margin"] = margin
            raw[f"{name}_sign"] = -1 if margin < 0 else (0 if margin == 0 else 1)
            raw[f"{name}_feasible"] = margin >= 0
            for key, value in pool.items():
                raw[f"{name}_{key}"] = value
    else:
        raise ValueError(operation)
    return {name: pilot.canonical(value) for name, value in raw.items()}


def shortcut_audit(specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Block every within-triplet single-field key and over-strong LOO heuristic."""
    by_operation: dict[str, list[tuple[str, list[dict[str, Any]]]]] = (
        collections.defaultdict(list)
    )
    for source in specs:
        variants = sorted(
            (variant["facts"] for variant in source["variants"]),
            key=lambda facts: oracle.score(source["operation"], facts),
        )
        if [oracle.score(source["operation"], facts) for facts in variants] != [
            0,
            1,
            2,
        ]:
            raise ValueError("Incomplete source triplet")
        by_operation[source["operation"]].append((source["locale"], variants))
    report: dict[str, Any] = {}
    for operation in OPS:
        groups = by_operation[operation]
        if len(groups) != 20:
            raise ValueError("Operation has incorrect source-group count")
        vectors = [
            [_one_field_features(operation, facts) for facts in variants]
            for _, variants in groups
        ]
        names = set(vectors[0][0])
        if any(set(row) != names for group in vectors for row in group):
            raise ValueError("Single-field inventory changes across source groups")
        for group in vectors:
            for name in names:
                if len({row[name] for row in group}) > 2:
                    raise ValueError(
                        f"{operation} has a one-field three-level shortcut: {name}"
                    )
        if operation == "allocation_caps":
            for group in vectors:
                for pool in ("first", "second"):
                    if (
                        len({row[f"{pool}_margin"] for row in group}) != 2
                        or len({row[f"{pool}_sign"] for row in group}) != 2
                    ):
                        raise ValueError(
                            "Allocation pool must have exactly two raw margins/signs"
                        )
        if operation == "independent_quorum":
            for group in vectors:
                if [json.loads(row["signed_count"]) for row in group] != [3, 3, 3]:
                    raise ValueError("Quorum signature count changed within group")
                if [json.loads(row["signed_qualifying_count"]) for row in group] != [
                    0,
                    2,
                    2,
                ]:
                    raise ValueError("Quorum qualifying-report count changed")
        if operation in ("allocation_caps", "inclusive_coverage"):
            orientation = collections.Counter()
            for (locale, _), group in zip(groups, vectors):
                key = (
                    "first_feasible"
                    if operation == "allocation_caps"
                    else "first_covers"
                )
                first_changes = json.loads(group[1][key])
                orientation[(locale, "first" if first_changes else "second")] += 1
            if orientation != {
                ("en", "first"): 8,
                ("en", "second"): 8,
                ("zh", "first"): 2,
                ("zh", "second"): 2,
            }:
                raise ValueError(
                    "Counterfactual first-change orientation is unbalanced"
                )
        accuracies = {}
        cap = 2 * len(groups)
        for name in sorted(names):
            correct = 0
            for held_index, held in enumerate(vectors):
                lookup: dict[str, collections.Counter[int]] = collections.defaultdict(
                    collections.Counter
                )
                for group_index, group in enumerate(vectors):
                    if group_index == held_index:
                        continue
                    for level, row in enumerate(group):
                        lookup[row[name]][level] += 1
                for level, row in enumerate(held):
                    counts = lookup[row[name]]
                    guess = (
                        min(counts, key=lambda answer: (-counts[answer], answer))
                        if counts
                        else 0
                    )
                    correct += guess == level
            accuracies[name] = correct
            if correct > cap:
                raise ValueError(
                    f"{operation} single-field held-out shortcut exceeds 2/3: {name}"
                )
        report[operation] = {
            "groups": len(groups),
            "rows": len(groups) * 3,
            "features": len(names),
            "shallow_cap_correct": cap,
            "best_shallow_correct": max(accuracies.values()),
            "orientation": (
                {
                    f"{locale}/{pool}": count
                    for (locale, pool), count in sorted(orientation.items())
                }
                if operation in ("allocation_caps", "inclusive_coverage")
                else None
            ),
        }
    return report


def _reference_rows(
    path: Path, *, require_gold_free: bool = False
) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if require_gold_free and (
                "label" in item or "gold" in item or "target" in item
            ):
                raise ValueError(f"Protected roster is not gold-free: {path.name}")
            if "state" not in item or ("id" not in item and "review_id" not in item):
                raise ValueError(f"Reference lacks ID/state: {path.name}")
            result.append(
                {
                    "id": item.get("id", item.get("review_id")),
                    "group_id": item.get("group_id"),
                    "input_sha256": item.get("input_sha256"),
                    "state": item["state"],
                    "instructions": (
                        pilot.canonical(item["questions"])
                        if "questions" in item
                        else item.get("instructions", "")
                    ),
                    "options": item.get("options", []),
                    "task_type": item.get("task_type", "context"),
                }
            )
    if not result:
        raise ValueError(f"Empty reference: {path.name}")
    return result


def audit_overlap(
    rows: list[dict[str, Any]], references: list[dict[str, Any]]
) -> dict[str, Any]:
    all_reference = []
    inventory = []
    for entry in references:
        name = entry["role"]
        path = Path(entry["path"])
        source = _reference_rows(path, require_gold_free=bool(entry.get("gold_free")))
        actual_sha = _sha(path)
        if entry.get("sha256") and actual_sha != entry["sha256"]:
            raise ValueError(f"Protected reference SHA mismatch: {name}")
        inventory.append({"role": name, "sha256": actual_sha, "rows": len(source)})
        all_reference.extend(source)
    ids = {row["id"] for row in all_reference}
    groups = {row.get("group_id") for row in all_reference if row.get("group_id")}
    inputs = {
        row.get("input_sha256") for row in all_reference if row.get("input_sha256")
    }
    contexts = {targeted.text_hashes(row["state"]) for row in all_reference}
    raw = {item[0] for item in contexts}
    normalized = {item[1] for item in contexts}
    exact = {
        "id": sum(row["id"] in ids for row in rows),
        "group_id": sum(row["group_id"] in groups for row in rows),
        "input_sha256": sum(row["input_sha256"] in inputs for row in rows),
        "raw_context": sum(
            targeted.text_hashes(row["state"])[0] in raw for row in rows
        ),
        "normalized_context": sum(
            targeted.text_hashes(row["state"])[1] in normalized for row in rows
        ),
    }
    near = pilot.near_duplicates(
        targeted.context_rows(rows),
        targeted.context_rows(all_reference),
        collect_left_ids=True,
    )
    rejected_ids = set(near.pop("left_ids"))
    near_full = pilot.near_duplicates(rows, all_reference, collect_left_ids=True)
    rejected_ids.update(near_full.pop("left_ids"))
    matched_groups = {row["group_id"] for row in rows if row["id"] in rejected_ids}
    if any(exact.values()) or near["count"] or near_full["count"]:
        raise ValueError(
            f"Source group overlap blocks the version: exact={exact}, near={near['count']}, full={near_full['count']}, "
            f"groups={len(matched_groups)}"
        )
    return {
        "reference_inventory": sorted(inventory, key=lambda item: item["role"]),
        "reference_rows": len(all_reference),
        "exact": exact,
        "near": near,
        "near_full_prompt": near_full,
        "limitations": "Approximate near-match search cannot prove semantic independence.",
    }


def _reviewer_packet(rows: list[dict[str, Any]], salt: bytes) -> list[dict[str, Any]]:
    packet = [
        {
            "review_id": row["id"],
            "group_id": row["group_id"],
            "operation": row["audit_metadata"]["operation"],
            "language": row["language"],
            "state": row["state"],
            "instructions": row["instructions"],
            "options": row["options"],
        }
        for row in rows
    ]
    packet.sort(key=lambda row: _hmac(salt, "review_order", row["review_id"]))
    return packet


def build(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.private_output_dir.exists()
        or args.reviewer_output_dir.exists()
        or args.reviewer_output_dir.is_relative_to(args.private_output_dir)
        or args.private_output_dir.is_relative_to(args.reviewer_output_dir)
    ):
        raise FileExistsError(
            "Private key and reviewer packet need distinct new directories"
        )
    if _sha(args.method) != METHOD_SHA256:
        raise ValueError("Prospective r2 method SHA mismatch")
    if (
        args.max_row_tokens != 1024
        or args.tokenizer_revision != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    ):
        raise ValueError("Frozen tokenizer revision or 1,024-token cap changed")
    seed, salt = _private_bytes(args.private_seed), _private_bytes(args.review_salt)
    if seed == salt:
        raise ValueError("Author seed and review salt must differ")
    for role, expected in FROZEN_PARENT_SHA256.items():
        path = getattr(args, f"parent_{role}")
        if _sha(path) != expected:
            raise ValueError(f"Frozen parent {role} bytes differ")
    if _sha(args.score_v6_train) != FROZEN_SCORE_V6_SHA256:
        raise ValueError("Approved Score v6 TRAIN candidate SHA mismatch")
    if _sha(args.r1_packet) != FROZEN_R1_PACKET_SHA256:
        raise ValueError("Frozen r1 gold-free packet SHA mismatch")
    if len(args.attempted_score) != 3:
        raise ValueError("All three earlier serialized Score attempts are required")
    specs, rows = generate(seed, salt)
    shallow = shortcut_audit(specs)
    group_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        group_rows[row["group_id"]].append(row)
    if any(
        len(group) != 3 or {row["label"] for row in group} != {0, 1, 2}
        for group in group_rows.values()
    ):
        raise AssertionError("Incomplete counterfactual triplet")
    references = [
        {"role": f"parent_{role}", "path": str(getattr(args, f"parent_{role}"))}
        for role in FROZEN_PARENT_SHA256
    ]
    references.extend(
        {"role": f"attempted_score_{i}", "path": str(path)}
        for i, path in enumerate(args.attempted_score)
    )
    references.extend(
        (
            {
                "role": "score_v6_train",
                "path": str(args.score_v6_train),
                "sha256": FROZEN_SCORE_V6_SHA256,
            },
            {
                "role": "score_r1_packet",
                "path": str(args.r1_packet),
                "sha256": FROZEN_R1_PACKET_SHA256,
                "gold_free": True,
            },
        )
    )
    protected = json.loads(args.protected_inventory.read_text(encoding="utf-8"))
    if (
        not isinstance(protected, list)
        or len({entry["role"] for entry in protected}) != len(protected)
        or not {entry["role"] for entry in protected}.issuperset(
            REQUIRED_PROTECTED_ROLES
        )
    ):
        raise ValueError("Protected inventory must have unique roles")
    references.extend(
        {
            "role": entry["role"],
            "path": entry["path"],
            "sha256": entry["sha256"],
            "gold_free": True,
        }
        for entry in protected
    )
    if len({entry["role"] for entry in references}) != len(references):
        raise ValueError("Reference roles are duplicated")
    overlap = audit_overlap(rows, references)
    from transformers import AutoTokenizer
    from transformers import __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = [pilot.count_tokens(row, tokenizer) for row in rows]
    if max(lengths) > args.max_row_tokens:
        raise ValueError(
            f"Tokenizer cap exceeded: {max(lengths)} > {args.max_row_tokens}"
        )
    packet = _reviewer_packet(rows, salt)
    if any("label" in row or "gold" in row or "target" in row for row in packet):
        raise AssertionError("Reviewer packet contains a key")
    by_operation_locale = collections.Counter(
        (s["operation"], s["locale"]) for s in specs
    )
    if any(
        by_operation_locale[(op, "en")] != 16 or by_operation_locale[(op, "zh")] != 4
        for op in OPS
    ):
        raise AssertionError("Preregistered language balance changed")
    args.private_output_dir.mkdir(parents=True, mode=0o700)
    args.reviewer_output_dir.mkdir(parents=True, mode=0o700)
    for directory in ("author", "oracle"):
        (args.private_output_dir / directory).mkdir(mode=0o700)
    _write(
        args.private_output_dir / "author" / "source_specs.jsonl", _jsonl_bytes(specs)
    )
    _write(args.private_output_dir / "oracle" / "select.jsonl", _jsonl_bytes(rows))
    _write(args.reviewer_output_dir / "packet.jsonl", _jsonl_bytes(packet))
    reviewer_manifest = {
        "schema_version": VERSION,
        "status": "gold_free_frozen_pending_independent_review",
        "packet_sha256": _sha(args.reviewer_output_dir / "packet.jsonl"),
        "rows": len(packet),
        "groups": len(group_rows),
        "review_instructions": "Solve every row without key access, inspect complete triplets and seal judgments and shortcut findings before key comparison.",
    }
    _write(
        args.reviewer_output_dir / "manifest.json",
        (json.dumps(reviewer_manifest, sort_keys=True, indent=2) + "\n").encode(),
    )
    files = {
        str(path.relative_to(args.private_output_dir)): _sha(path)
        for path in args.private_output_dir.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": VERSION,
        "status": "frozen_unreviewed_do_not_run_model",
        "source_commit": args.source_commit,
        "method_sha256": _sha(args.method),
        "builder_sha256": _sha(Path(__file__)),
        "oracle_sha256": _sha(Path(oracle.__file__)),
        "render_oracle_sha256": _sha(Path(render_oracle.__file__)),
        "seed_commitment_sha256": hashlib.sha256(seed).hexdigest(),
        "review_salt_commitment_sha256": hashlib.sha256(salt).hexdigest(),
        "parent_sha256": FROZEN_PARENT_SHA256,
        "protected_inventory_sha256": _sha(args.protected_inventory),
        "score_v6_train_sha256": FROZEN_SCORE_V6_SHA256,
        "score_r1_packet_sha256": FROZEN_R1_PACKET_SHA256,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_config_sha256": _sha(args.tokenizer / "tokenizer_config.json"),
        "tokenizer_json_sha256": _sha(args.tokenizer / "tokenizer.json"),
        "transformers_version": transformers_version,
        "max_row_tokens": args.max_row_tokens,
        "token_lengths": {
            "min": min(lengths),
            "max": max(lengths),
            "total": sum(lengths),
        },
        "counts": {
            "groups": len(specs),
            "rows": len(rows),
            "by_operation_locale": {
                f"{op}/{locale}": count
                for (op, locale), count in sorted(by_operation_locale.items())
            },
        },
        "overlap": overlap,
        "single_field_shortcut_audit": shallow,
        "reviewer_packet_sha256": _sha(args.reviewer_output_dir / "packet.jsonl"),
        "reviewer_manifest_sha256": _sha(args.reviewer_output_dir / "manifest.json"),
        "files_sha256": files,
        "limitations": [
            "Selection only, never release benchmark.",
            "Chinese rows require qualified bilingual review for transfer claims.",
            "Approximate near overlap cannot prove semantic independence.",
        ],
    }
    _write(
        args.private_output_dir / "manifest.json",
        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-seed", type=Path, required=True)
    parser.add_argument("--review-salt", type=Path, required=True)
    parser.add_argument("--parent-train", type=Path, required=True)
    parser.add_argument("--parent-select", type=Path, required=True)
    parser.add_argument("--parent-cal", type=Path, required=True)
    parser.add_argument("--attempted-score", type=Path, action="append", default=[])
    parser.add_argument("--score-v6-train", type=Path, required=True)
    parser.add_argument("--r1-packet", type=Path, required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--max-row-tokens", type=int, default=1024)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--reviewer-output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "groups": manifest["counts"]["groups"],
                "rows": manifest["counts"]["rows"],
                "manifest_sha256": _sha(args.private_output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
