"""Independently derive Score SELECT r2 answers from reviewer-visible prose.

This module never imports the private case generator or structured oracle.
"""

from __future__ import annotations

import re


def _one(pattern: str, text: str) -> tuple[str, ...]:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError("A required visible fact is missing or ambiguous")
    return match.groups()


def score(operation: str, text: str, locale: str) -> int:
    if operation == "waiver_precedence":
        if locale == "en":
            day = int(_one(r"Review is on day (\d+)\.", text)[0])
            veto = _one(r"The live hold is (active|inactive)\.", text)[0] == "active"
            lead, second, signed, expiry = _one(
                r"Waiver entry: lead signature (present|absent); second signature "
                r"(present|absent); signed day (\d+); final valid day (\d+)\.",
                text,
            )
            follow = _one(r"Follow-up check: (pending|passed)\.", text)[0]
            valid = lead == second == "present" and int(signed) <= day <= int(expiry)
            return 0 if veto and not valid else (1 if follow == "pending" else 2)
        day = int(_one(r"审核日为第 (\d+) 天", text)[0])
        veto = _one(r"现行暂停状态：(生效|未生效)", text)[0] == "生效"
        lead, second, signed, expiry = _one(
            r"豁免登记：主负责人(已签|未签)；另一负责人(已签|未签)；"
            r"第 (\d+) 天签署；最后有效日是第 (\d+) 天。",
            text,
        )
        follow = _one(r"后续核查：(未完成|已完成)", text)[0]
        valid = lead == second == "已签" and int(signed) <= day <= int(expiry)
        return 0 if veto and not valid else (1 if follow == "未完成" else 2)
    if operation == "inclusive_coverage":
        if locale == "en":
            start, end = map(
                int, _one(r"Requested inclusive period: day (\d+) to day (\d+)\.", text)
            )
            windows = [
                tuple(
                    map(
                        int,
                        _one(
                            rf"Active office {name} lists days (\d+) to (\d+)\.", text
                        ),
                    )
                )
                for name in ("North", "South")
            ]
        else:
            start, end = map(
                int, _one(r"申请时段（首尾均计入）：第 (\d+) 天到第 (\d+) 天", text)
            )
            windows = [
                tuple(
                    map(
                        int,
                        _one(rf"现行{name}区机构登记：第 (\d+) 天到第 (\d+) 天", text),
                    )
                )
                for name in ("北", "南")
            ]
        return sum(first <= start and end <= last for first, last in windows)
    if operation == "independent_quorum":
        if locale == "en":
            reports = re.findall(
                r"Evidence sheet \d+: origin (\w+); signature (present|absent); "
                r"(in force|retired); (supports|does not support) the request\.",
                text,
            )
            if len(reports) != 6:
                raise ValueError("Expected six visible evidence sheets")
            accepted = {
                lineage
                for lineage, signed, current, affirmative in reports
                if signed == "present"
                and current == "in force"
                and affirmative == "supports"
            }
        else:
            reports = re.findall(
                r"证据单 \d+：来源 (\w+)；(已签|未签)；(有效|失效)；(支持|不支持)申请。",
                text,
            )
            if len(reports) != 6:
                raise ValueError("Expected six visible Chinese evidence sheets")
            accepted = {
                lineage
                for lineage, signed, current, affirmative in reports
                if signed == "已签" and current == "有效" and affirmative == "支持"
            }
        return min(2, len(accepted))
    if operation == "allocation_caps":
        if locale == "en":
            ledgers = [
                tuple(
                    map(
                        int,
                        _one(
                            rf"{name} budget ledger: capacity (\d+) units; committed (\d+); "
                            r"requested (\d+); protected reserve (\d+)\.",
                            text,
                        ),
                    )
                )
                for name in ("East", "West")
            ]
        else:
            ledgers = [
                tuple(
                    map(
                        int,
                        _one(
                            rf"{name}区预算账：容量 (\d+) 单位；已承诺 (\d+)；本次申请 (\d+)；保留额度 (\d+)。",
                            text,
                        ),
                    )
                )
                for name in ("东", "西")
            ]
        return sum(
            committed + request + reserve <= capacity
            for capacity, committed, request, reserve in ledgers
        )
    raise ValueError(f"Unknown operation {operation}")
