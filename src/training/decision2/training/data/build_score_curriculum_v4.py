"""Build TRAIN-only Score v4 with count-invariant ordinal mechanisms.

No benchmark generator or held-out gold is imported. A protected inventory
contains only prompt files; matching source groups are quarantined in full.
The output is a research candidate, not an approved training release.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

SEED = "decision20-score-curriculum-v4-20260927"
SOURCE = "decision2_internal_score_curriculum_v4"
FAMILIES = ("obligation_review", "weighted_points", "route_depth", "timely_streak")
GROUPS_PER_FAMILY = 81
EXPECTED_ROWS = len(FAMILIES) * GROUPS_PER_FAMILY * 3
FROZEN = {
    "base": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "base_manifest": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
REQUIRED_PROTECTED_ROLES = {
    "typed_dev",
    "css_pilot",
    "css15_goldfree",
    "jevbench_public231",
    "decision_bench_v4_text",
    "pressure_rq1",
    "pressure_rq2",
    "pressure_rq3",
    "authored_v7",
    "authored_v8",
    "authored_v9",
    "authored_v10",
    "authored_v11_r1",
    "authored_v11_r2",
    "authored_v11_r3",
    "multilingual_hard_r6",
}
SCENES = (
    "community workshop",
    "language center",
    "archive project",
    "field survey",
    "science club",
    "library program",
    "garden project",
    "training cohort",
    "repair clinic",
    "public lecture",
    "museum workshop",
    "cycling program",
)
SCENES_ZH = (
    "社区工坊",
    "语言中心",
    "档案项目",
    "实地调查",
    "科学社团",
    "图书馆项目",
    "园艺项目",
    "培训课程",
    "维修门诊",
    "公开讲座",
    "博物馆工坊",
    "骑行项目",
)
REVIEW_TOPICS_EN = (
    "identity",
    "access",
    "consent",
    "budget",
    "schedule",
    "safety",
    "staffing",
    "notice",
    "translation",
    "records",
    "insurance",
    "supplies",
    "location",
    "privacy",
    "training",
    "maintenance",
    "equipment",
    "attendance",
    "inspection",
    "handoff",
)
REVIEW_TOPICS_ZH = (
    "身份",
    "访问",
    "同意",
    "预算",
    "日程",
    "安全",
    "人员",
    "通知",
    "翻译",
    "档案",
    "保险",
    "物资",
    "场地",
    "隐私",
    "培训",
    "维护",
    "设备",
    "出席",
    "检查",
    "交接",
)
SIGNAL_TOPICS_EN = (
    "clarity",
    "coverage",
    "relevance",
    "traceability",
    "freshness",
    "consistency",
    "usefulness",
    "precision",
    "reach",
    "timeliness",
    "readability",
    "accessibility",
    "completeness",
    "organization",
    "repeatability",
    "durability",
    "specificity",
    "stability",
    "fit",
    "response",
)
SIGNAL_TOPICS_ZH = (
    "清晰度",
    "覆盖度",
    "相关性",
    "可追溯性",
    "时效性",
    "一致性",
    "实用性",
    "准确度",
    "覆盖范围",
    "及时性",
    "可读性",
    "无障碍性",
    "完整性",
    "组织性",
    "可重复性",
    "耐久性",
    "具体性",
    "稳定性",
    "契合度",
    "响应度",
)
STREAK_TOPICS_EN = (
    "response note",
    "stock report",
    "lesson plan",
    "inspection log",
    "repair ticket",
    "survey upload",
    "access review",
    "safety check",
    "schedule update",
    "reading summary",
    "garden log",
    "service memo",
    "staff rota",
    "event recap",
    "equipment check",
    "expense note",
    "meeting record",
    "handoff memo",
    "arrival report",
    "training log",
)
STREAK_TOPICS_ZH = (
    "响应记录",
    "库存报告",
    "课程计划",
    "检查日志",
    "维修工单",
    "调查上传",
    "访问审核",
    "安全核查",
    "日程更新",
    "阅读摘要",
    "园艺日志",
    "服务备忘",
    "人员排班",
    "活动总结",
    "设备核查",
    "费用记录",
    "会议纪要",
    "交接备忘",
    "到达报告",
    "培训日志",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rng(family: str, index: int) -> random.Random:
    return random.Random(int(_sha(f"{SEED}\0{family}\0{index}")[:16], 16))


def _case(rng: random.Random, language: str) -> str:
    choices = SCENES_ZH if language == "zh" else SCENES
    return (
        f"{rng.choice(choices)} {''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=4))}"
    )


def oracle(family: str, state: dict[str, Any]) -> int:
    """Derive the label from the displayed state, independent of metadata."""
    if family == "obligation_review":
        latest: dict[str, dict[str, Any]] = {}
        for event in state["events"]:
            if event["scope"] != "core":
                continue
            name = event["control"]
            if name not in latest or event["timestamp"] > latest[name]["timestamp"]:
                latest[name] = event
        if len(latest) != 2:
            raise ValueError("Obligation review requires two core controls")
        statuses = [event["assessment"] for event in latest.values()]
        if "rejected" in statuses:
            return 0
        return 1 if "unresolved" in statuses else 2
    if family == "weighted_points":
        points = sum(x["weight"] * x["mark"] for x in state["signals"])
        return 0 if points < state["lower"] else (1 if points < state["upper"] else 2)
    if family == "route_depth":
        frontier = [(state["start"], 0)]
        visited = {state["start"]}
        edges = collections.defaultdict(list)
        for link in state["links"]:
            edges[link["from"]].append(link["to"])
        for node, depth in frontier:
            if node == state["finish"]:
                return 2 if depth <= 2 else 1
            for neighbour in edges[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    frontier.append((neighbour, depth + 1))
        return 0
    if family == "timely_streak":
        records = sorted(
            (x for x in state["days"] if 1 <= x["day"] <= 12),
            key=lambda x: x["day"],
        )
        if [x["day"] for x in records] != list(range(1, 13)):
            raise ValueError("timely_streak requires exactly days 1 through 12")
        best = streak = 0
        for record in records:
            streak = streak + 1 if record["on_time"] else 0
            best = max(best, streak)
        return 0 if best <= 2 else (1 if best == 3 else 2)
    raise ValueError(f"Unknown Score family {family}")


def _obligation_states(
    rng: random.Random, case: str, language: str, index: int
) -> list[dict[str, Any]]:
    vocabulary = REVIEW_TOPICS_ZH if language == "zh" else REVIEW_TOPICS_EN
    names = rng.sample(vocabulary, 3)
    core_times = sorted(rng.sample(range(2, 30), 3))
    info_times = sorted(rng.sample(range(2, 30), 3))
    info_statuses = rng.sample(("accepted", "unresolved", "rejected"), 3)
    information = [
        {
            "control": names[2],
            "scope": "informational",
            "timestamp": timestamp,
            "assessment": assessment,
        }
        for timestamp, assessment in zip(info_times, info_statuses)
    ]
    rejected_core = index % 2
    unresolved_core = 1 - rejected_core
    states = []
    for level in range(3):
        current = ["accepted", "accepted"]
        if level == 0:
            current[rejected_core] = "rejected"
        elif level == 1:
            current[unresolved_core] = "unresolved"
        core_events = []
        for core_index in range(2):
            older = [
                status
                for status in ("accepted", "unresolved", "rejected")
                if status != current[core_index]
            ]
            rng.shuffle(older)
            for timestamp, assessment in zip(core_times, (*older, current[core_index])):
                core_events.append(
                    {
                        "control": names[core_index],
                        "scope": "core",
                        "timestamp": timestamp,
                        "assessment": assessment,
                    }
                )
        middle = [*core_events, information[1]]
        rng.shuffle(middle)
        events = [information[0], *middle, information[2]]
        states.append({"case": case, "events": events})
    return states


def _weighted_states(
    rng: random.Random,
    case: str,
    language: str,
    balance: collections.Counter[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    """Fix shallow totals and keep every single signal binary across levels."""
    words = rng.sample(SIGNAL_TOPICS_ZH if language == "zh" else SIGNAL_TOPICS_EN, 5)
    weight_pools = ((1, 2, 3, 4, 5), (1, 2, 2, 4, 5), (1, 2, 3, 3, 5))
    mark_pools = (
        (0, 1, 2, 3, 4),
        (0, 1, 1, 3, 4),
        (0, 1, 2, 2, 4),
        (0, 1, 2, 3, 5),
        (0, 0, 2, 3, 5),
    )
    for _ in range(16):
        weights = list(rng.choice(weight_pools))
        rng.shuffle(weights)
        marks = list(rng.choice(mark_pools))
        anchor_index = weights.index(max(weights))
        anchor_mark = max(marks)
        marks.remove(anchor_mark)
        remaining_indices = [index for index in range(5) if index != anchor_index]
        pairings = []
        for remainder in sorted(set(itertools.permutations(marks))):
            paired = [0] * 5
            paired[anchor_index] = anchor_mark
            for index, mark in zip(remaining_indices, remainder):
                paired[index] = mark
            total = sum(weight * mark for weight, mark in zip(weights, paired))
            pairings.append((tuple(paired), total))
        totals = sorted({total for _, total in pairings})
        if len(totals) < 3:
            continue
        lower, upper = totals[len(totals) // 3], totals[2 * len(totals) // 3]
        buckets = (
            [paired for paired, total in pairings if total < lower],
            [paired for paired, total in pairings if lower <= total < upper],
            [paired for paired, total in pairings if total >= upper],
        )
        eligible = [
            trio
            for trio in itertools.product(*buckets)
            if all(
                len({trio[level][index] for level in range(3)}) <= 2
                for index in range(5)
            )
        ]
        if eligible:
            break
    else:
        raise AssertionError(
            "Could not construct three single-signal-safe weighted buckets"
        )

    def imbalance(trio: tuple[tuple[int, ...], ...]) -> int:
        penalty = 0
        for level, marks_by_signal in enumerate(trio):
            for position, mark in enumerate(marks_by_signal):
                bucket = (position, weights[position], mark)
                own = balance[(*bucket, level)]
                other = sum(balance[(*bucket, label)] for label in range(3))
                penalty += 3 * own - other
        return penalty

    chosen = min(
        eligible,
        key=lambda trio: (
            imbalance(trio),
            _sha(f"{SEED}\0{case}\0{trio}"),
        ),
    )
    states = []
    for level, paired in enumerate(chosen):
        signals = [
            {"name": name, "weight": weight, "mark": mark}
            for name, weight, mark in zip(words, weights, paired)
        ]
        for position, mark in enumerate(paired):
            balance[(position, weights[position], mark, level)] += 1
        states.append(
            {"case": case, "signals": signals, "lower": lower, "upper": upper}
        )
    return states


def _route_states(rng: random.Random, case: str) -> list[dict[str, Any]]:
    """Use two equal directed cycles, hiding labels from graph counts/degrees."""
    nodes = [f"N{number}" for number in rng.sample(range(1000, 9999), 8)]
    start, finish, *others = nodes
    states = []
    for level in range(3):
        if level == 0:
            first = rng.sample(others, 3)
            second = [node for node in others if node not in first]
            rng.shuffle(first)
            rng.shuffle(second)
            cycles = ([start, *first], [finish, *second])
        else:
            first = rng.sample(others, 2)
            second = [node for node in others if node not in first]
            rng.shuffle(first)
            rng.shuffle(second)
            # Level 1 is three directed hops; level 2 is two.
            primary = (
                [start, first[0], first[1], finish]
                if level == 1
                else [start, first[0], finish, first[1]]
            )
            cycles = (primary, second)
        links = [
            {"from": cycle[index], "to": cycle[(index + 1) % 4]}
            for cycle in cycles
            for index in range(4)
        ]
        rng.shuffle(links)
        states.append({"case": case, "start": start, "finish": finish, "links": links})
    return states


def _streak_states(
    rng: random.Random, case: str, language: str
) -> list[dict[str, Any]]:
    on_run_multisets = ((2, 2, 2), (3, 2, 1), (4, 1, 1))
    off_run_multiset = (1, 1, 1, 3)
    schedules = []
    for on_multiset in on_run_multisets:
        candidates = []
        for on_runs in sorted(set(itertools.permutations(on_multiset))):
            for off_runs in sorted(set(itertools.permutations(off_run_multiset))):
                pattern = []
                for position in range(3):
                    pattern.extend([False] * off_runs[position])
                    pattern.extend([True] * on_runs[position])
                pattern.extend([False] * off_runs[3])
                if len(pattern) != 12:
                    raise AssertionError("Streak pattern must have twelve days")
                candidates.append(tuple(pattern))
        schedules.append(rng.choice(candidates))
    process = rng.choice(STREAK_TOPICS_ZH if language == "zh" else STREAK_TOPICS_EN)
    day_zero = rng.choice((True, False))
    day_thirteen = rng.choice((True, False))
    states = []
    for schedule in schedules:
        middle = [
            {"day": day, "on_time": value} for day, value in enumerate(schedule, 1)
        ]
        rng.shuffle(middle)
        days = [
            {"day": 0, "on_time": day_zero},
            *middle,
            {"day": 13, "on_time": day_thirteen},
        ]
        states.append({"case": case, "tracked_process": process, "days": days})
    return states


RULE_ALTERNATIVES = {
    ("obligation_review", "en"): (
        "For each core control, use its event with the greatest timestamp; informational controls do not count. A latest rejection takes precedence and means 0. Otherwise a latest unresolved status means 1, and all latest core statuses accepted means 2.",
        "First resolve the most recent event separately for each core control, ignoring informational events. Give 0 if any current core status is rejected, else 1 if any is unresolved, else 2 when both are accepted.",
    ),
    ("obligation_review", "zh"): (
        "每个核心控制项只取时间戳最大的最新事件，信息性项目不计入。最新核心状态有拒绝时优先判 0；没有拒绝但有未解决状态判 1；最新核心状态全为接受判 2。",
        "先分别找出每个核心控制项时间最新的事件，忽略信息性事件。当前核心状态有拒绝为 0，否则有未解决为 1，两个核心状态都接受为 2。",
    ),
    ("weighted_points", "en"): (
        "Compute the sum of mark times weight for every signal. Return 0 below the lower boundary, 1 from the lower boundary up to but excluding the upper boundary, or 2 from the upper boundary onward.",
        "For each signal, multiply its mark by its weight, then total the products. Compare the total to the shown cutoffs: less than lower is 0, lower to below upper is 1, and upper or above is 2.",
    ),
    ("weighted_points", "zh"): (
        "对每个信号计算分数乘权重，再将乘积相加。低于下界判 0；达到下界但未达到上界判 1；达到上界及以上判 2。",
        "先逐项把分数与权重相乘并求总和，再比较给定界值：总和小于下界为 0，不小于下界且小于上界为 1，不小于上界为 2。",
    ),
    ("route_depth", "en"): (
        "Treat every displayed link as one-way. If finish cannot be reached from start, return 0. Otherwise return 1 for a shortest path of three or more links, and 2 for a shortest path of one or two links.",
        "Trace directed travel from start to finish; link order on this page does not matter. Give 0 if unreachable, 1 if the minimum path takes at least three steps, and 2 if it takes no more than two.",
    ),
    ("route_depth", "zh"): (
        "每条连接都是单向的。若从起点无法到达终点，判 0；可达且最短路径至少三条边，判 1；最短路径为一或两条边，判 2。",
        "沿有向边追踪起点到终点，表中边的排列顺序不重要。不可达为 0；最短需三步及以上为 1；最短不超过两步为 2。",
    ),
    ("timely_streak", "en"): (
        "Sort only days 1–12 by day number and measure the longest uninterrupted run of on-time entries. Ignore days 0 and 13. Return 0 for a run of at most two, 1 for exactly three, and 2 for four or more.",
        "Days 0 and 13 are outside the review window. Across days 1 through 12 in chronological order, find the maximum consecutive on-time stretch: up to two means 0, three means 1, and at least four means 2.",
    ),
    ("timely_streak", "zh"): (
        "按天数排列第 1 至第 12 天，只计算连续按时完成的最长段，忽略第 0 天和第 13 天。最长至多两天判 0，恰好三天判 1，四天及以上判 2。",
        "第 0 天和第 13 天不在评估窗口。依次查看第 1 至第 12 天，找出连续按时天数的最大值：至多两天为 0，恰好三天为 1，至少四天为 2。",
    ),
}

OPTION_ALTERNATIVES = {
    ("obligation_review", "en"): (
        (
            "A latest core status is rejected",
            "No latest core rejection, but one is unresolved",
            "Both latest core statuses are accepted",
        ),
        (
            "Current core veto exists",
            "Current core item pending, none vetoed",
            "Current core items have no veto or pending status",
        ),
    ),
    ("obligation_review", "zh"): (
        (
            "最新核心状态有拒绝",
            "最新核心状态无拒绝但有未解决",
            "两个最新核心状态均已接受",
        ),
        ("当前核心项有否决", "当前核心项待处理且无否决", "当前核心项没有否决或待处理"),
    ),
    ("weighted_points", "en"): (
        (
            "Weighted sum is below lower cutoff",
            "Weighted sum is at least lower and below upper cutoff",
            "Weighted sum is at least upper cutoff",
        ),
        (
            "Total below lower boundary",
            "Total at least lower but below upper",
            "Total at least upper boundary",
        ),
    ),
    ("weighted_points", "zh"): (
        ("加权和低于下界", "加权和达到下界但低于上界", "加权和达到或超过上界"),
        ("总分低于下限", "总分不低于下限且低于上限", "总分不低于上限"),
    ),
    ("route_depth", "en"): (
        (
            "Finish is unreachable",
            "Minimum directed path is three or more links",
            "Minimum directed path is one or two links",
        ),
        (
            "No one-way path to finish",
            "Shortest one-way route exceeds two steps",
            "Shortest one-way route needs at most two steps",
        ),
    ),
    ("route_depth", "zh"): (
        ("终点不可达", "最短有向路径至少三条边", "最短有向路径至多两条边"),
        ("不存在通往终点的单向路径", "最短路线超过两步", "最短路线不超过两步"),
    ),
    ("timely_streak", "en"): (
        (
            "Longest timely run is at most two days",
            "Longest timely run is exactly three days",
            "Longest timely run is at least four days",
        ),
        (
            "Longest on-time stretch is no more than two",
            "Longest on-time stretch is three",
            "Longest on-time stretch is at least four",
        ),
    ),
    ("timely_streak", "zh"): (
        ("最长按时段至多两天", "最长按时段恰好三天", "最长按时段至少四天"),
        ("最长连续按时不超过两天", "最长连续按时恰好三天", "最长连续按时至少四天"),
    ),
}


def _question(
    family: str, language: str, rng: random.Random, case: str
) -> tuple[str, list[dict[str, str]]]:
    if family == "obligation_review":
        instructions = (
            "For each core control, select its event with the greatest timestamp. Ignore informational controls. A latest rejected core status gives level 0; otherwise a latest unresolved status gives level 1; both latest statuses accepted gives level 2."
            if language == "en"
            else "每个核心控制项只取时间戳最大的事件，忽略信息性项目。最新核心状态有拒绝为 0 档；否则有未解决状态为 1 档；两个最新核心状态均已接受为 2 档。"
        )
        labels = (
            (
                "A latest core status is rejected",
                "No latest core rejection, but one is unresolved",
                "Both latest core statuses are accepted",
            )
            if language == "en"
            else (
                "最新核心状态有拒绝",
                "最新核心状态无拒绝但仍有未解决项",
                "两个最新核心状态均已接受",
            )
        )
    elif family == "weighted_points":
        instructions = (
            "Multiply each displayed mark by its weight and add the products. Use the displayed lower and upper cutoffs to assign level 0 below lower, level 1 from lower up to but not including upper, or level 2 at/above upper."
            if language == "en"
            else "每项分数乘以权重后求和。总分低于下界为 0 档；达到下界但低于上界为 1 档；达到或高于上界为 2 档。"
        )
        labels = (
            (
                "Total below lower cutoff",
                "Total at least lower cutoff but below upper cutoff",
                "Total at or above upper cutoff",
            )
            if language == "en"
            else ("总分低于下界", "总分达到下界但低于上界", "总分达到或高于上界")
        )
    elif family == "route_depth":
        instructions = (
            "Follow directed links from start to finish. Give level 0 when no route exists, level 1 when the shortest route uses at least three links, and level 2 when a route of at most two links exists. Link listing order is irrelevant."
            if language == "en"
            else "沿有向连接从起点走到终点。无法到达为 0 档；最短路线至少需要三条连接为 1 档；存在不超过两条连接的路线为 2 档。连接的列出顺序无关紧要。"
        )
        labels = (
            (
                "No directed route",
                "Shortest route is at least three links",
                "A route uses at most two links",
            )
            if language == "en"
            else ("不存在有向路线", "最短路线至少三条连接", "存在不超过两条连接的路线")
        )
    elif family == "timely_streak":
        instructions = (
            "Consider numbered days 1 through 12 only, in day order. Find the longest consecutive streak marked on_time. Give level 0 for at most two days, level 1 for exactly three, and level 2 for four or more. Ignore days 0 and 13."
            if language == "en"
            else "仅按日期顺序查看第 1 至 12 天，找出按时完成的最长连续天数。最长至多 2 天是 0 档，恰好 3 天是 1 档，至少 4 天是 2 档。忽略第 0 天和第 13 天。"
        )
        labels = (
            (
                "Longest on-time streak is at most two",
                "Longest streak is exactly three",
                "Longest streak is at least four",
            )
            if language == "en"
            else ("最长按时连续天数至多两天", "最长连续恰好三天", "最长连续至少四天")
        )
    else:
        raise ValueError(family)
    instructions = rng.choice((instructions, *RULE_ALTERNATIVES[(family, language)]))
    labels = rng.choice((labels, *OPTION_ALTERNATIVES[(family, language)]))
    introduction = (
        rng.choice(("Decision record for", "Review note for", "Classify", "Evaluate"))
        if language == "en"
        else rng.choice(("决策记录", "审核记录", "分类对象", "评估对象"))
    )
    instructions = f"{introduction} {case}. {instructions}"
    return instructions, [
        {"key": str(index), "description": value} for index, value in enumerate(labels)
    ]


def _route_shortcut_features(state: dict[str, Any]) -> tuple[Any, ...]:
    """Observable nontraversal cues and cycle sizes; exclude reachability itself."""
    edges = state["links"]
    nodes = {end for edge in edges for end in (edge["from"], edge["to"])}
    indegree = collections.Counter(edge["to"] for edge in edges)
    outdegree = collections.Counter(edge["from"] for edge in edges)
    if len(edges) != 8 or len(nodes) != 8:
        raise ValueError("Route graph must have eight distinct nodes and links")
    if any(indegree[node] != 1 or outdegree[node] != 1 for node in nodes):
        raise ValueError(
            "Route graph must have one incoming and outgoing link per node"
        )
    successor = {edge["from"]: edge["to"] for edge in edges}
    remaining = set(nodes)
    cycle_sizes = []
    node_cycle_size = {}
    while remaining:
        first = min(remaining)
        current = first
        cycle = []
        while current not in cycle:
            cycle.append(current)
            current = successor[current]
        if current != first:
            raise ValueError("Unexpected non-cycle component")
        remaining.difference_update(cycle)
        cycle_sizes.append(len(cycle))
        node_cycle_size.update({node: len(cycle) for node in cycle})
    direct = {"from": state["start"], "to": state["finish"]} in edges
    return (
        len(edges),
        len(nodes),
        tuple(sorted(cycle_sizes)),
        indegree[state["start"]],
        outdegree[state["start"]],
        indegree[state["finish"]],
        outdegree[state["finish"]],
        node_cycle_size[state["start"]],
        node_cycle_size[state["finish"]],
        direct,
    )


def _streak_shortcut_features(state: dict[str, Any]) -> tuple[Any, ...]:
    days = state["days"]
    by_day = {record["day"]: record["on_time"] for record in days}
    if sorted(by_day) != list(range(14)) or len(days) != 14:
        raise ValueError("Streak state must list day 0 through day 13 exactly once")
    within = [bool(by_day[day]) for day in range(1, 13)]
    adjacent = sum(within[index] and within[index + 1] for index in range(11))
    runs = sum(
        value and (index == 0 or not within[index - 1])
        for index, value in enumerate(within)
    )
    return (
        len(days),
        sum(within),
        adjacent,
        runs,
        sum(not within[index] and not within[index + 1] for index in range(11)),
        sum(within[index] != within[index + 1] for index in range(11)),
        within[0],
        within[-1],
        bool(by_day[0]),
        bool(by_day[13]),
        days[0]["day"],
        days[-1]["day"],
    )


def _obligation_shortcut_features(row: dict[str, Any]) -> tuple[Any, ...]:
    """Surface counts only; latest-per-control resolution is the target."""
    events = row["state"]["events"]
    by_control: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for event in events:
        by_control[event["control"]].append(event)
    return (
        len(events),
        tuple(
            sorted(collections.Counter(event["assessment"] for event in events).items())
        ),
        tuple(sorted(collections.Counter(event["scope"] for event in events).items())),
        tuple(
            sorted(
                (
                    name,
                    records[0]["scope"],
                    tuple(sorted(record["timestamp"] for record in records)),
                    tuple(sorted(record["assessment"] for record in records)),
                )
                for name, records in by_control.items()
            )
        ),
        (events[0]["scope"], events[0]["assessment"]),
        (events[-1]["scope"], events[-1]["assessment"]),
        row["case"] if "case" in row else row["state"]["case"],
        row["instructions"],
    )


def _weighted_shortcut_features(row: dict[str, Any]) -> tuple[Any, ...]:
    state = row["state"]
    signals = state["signals"]
    marks = [signal["mark"] for signal in signals]
    weights = [signal["weight"] for signal in signals]
    anchor = weights.index(max(weights))
    return (
        len(signals),
        tuple(sorted(marks)),
        tuple(sorted(weights)),
        sum(marks),
        min(marks),
        max(marks),
        state["lower"],
        state["upper"],
        marks[anchor],
        max(weight * mark for weight, mark in zip(weights, marks)),
        row["language"],
        row["instructions"],
        json.dumps(row["options"], ensure_ascii=False, sort_keys=True),
        state["case"],
        tuple(signal["name"] for signal in signals),
    )


def _unweighted_rule_audit(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Fit mark-sum thresholds and group-held-out lookup without weights."""
    sums = {
        row["id"]: sum(signal["mark"] for signal in row["state"]["signals"])
        for row in rows
    }
    floor, ceiling = min(sums.values()), max(sums.values()) + 2
    best = 0
    for lower in range(floor, ceiling):
        for upper in range(lower + 1, ceiling + 1):
            correct = sum(
                (0 if sums[row["id"]] < lower else 1 if sums[row["id"]] < upper else 2)
                == row["label"]
                for row in rows
            )
            best = max(best, correct)
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    heldout_correct = 0
    for held, test_rows in groups.items():
        train_counts: dict[int, collections.Counter[int]] = collections.defaultdict(
            collections.Counter
        )
        for group, train_rows in groups.items():
            if group != held:
                for row in train_rows:
                    train_counts[sums[row["id"]]][row["label"]] += 1
        for row in test_rows:
            counts = train_counts[sums[row["id"]]]
            guess = (
                min(counts, key=lambda level: (-counts[level], level)) if counts else 0
            )
            heldout_correct += guess == row["label"]
    if best * 5 > len(rows) * 2 or heldout_correct * 5 > len(rows) * 2:
        raise ValueError("Weighted family has an unweighted shortcut above 40%")
    return {
        "best_unweighted_sum_threshold_correct": best,
        "leave_one_group_out_mark_sum_lookup_correct": heldout_correct,
        "total": len(rows),
    }


def _single_signal_heldout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a fixed signal position from other groups' weight/mark records."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    by_position = {}
    for position in range(5):
        correct = 0
        for held, test_rows in groups.items():
            lookup: dict[tuple[int, int], collections.Counter[int]] = (
                collections.defaultdict(collections.Counter)
            )
            for group, train_rows in groups.items():
                if group == held:
                    continue
                for row in train_rows:
                    signal = row["state"]["signals"][position]
                    lookup[(signal["weight"], signal["mark"])][row["label"]] += 1
            for row in test_rows:
                signal = row["state"]["signals"][position]
                counts = lookup[(signal["weight"], signal["mark"])]
                guess = (
                    min(counts, key=lambda level: (-counts[level], level))
                    if counts
                    else 0
                )
                correct += guess == row["label"]
        by_position[str(position)] = correct
    best = max(by_position.values())
    if best * 5 > len(rows) * 2:
        raise ValueError(
            "Weighted single-signal held-out classifier exceeds 40%: "
            f"best={best}/{len(rows)}, by_position={by_position}"
        )
    return {
        "correct_by_position": by_position,
        "best_correct": best,
        "total": len(rows),
    }


def _streak_day_heldout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure, but do not select on, any one calendar-position bit."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    by_day = {}
    for day in range(1, 13):
        correct = 0
        for held, test_rows in groups.items():
            lookup: dict[bool, collections.Counter[int]] = collections.defaultdict(
                collections.Counter
            )
            for group, train_rows in groups.items():
                if group == held:
                    continue
                for row in train_rows:
                    timed = next(
                        record["on_time"]
                        for record in row["state"]["days"]
                        if record["day"] == day
                    )
                    lookup[bool(timed)][row["label"]] += 1
            for row in test_rows:
                timed = next(
                    record["on_time"]
                    for record in row["state"]["days"]
                    if record["day"] == day
                )
                counts = lookup[bool(timed)]
                guess = (
                    min(counts, key=lambda level: (-counts[level], level))
                    if counts
                    else 0
                )
                correct += guess == row["label"]
        by_day[str(day)] = correct
    return {
        "correct_by_day": by_day,
        "best_correct": max(by_day.values()),
        "total": len(rows),
    }


def shortcut_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject fixed-state and unweighted features that separate levels."""
    report = {}
    for family, feature_fn in (
        ("obligation_review", _obligation_shortcut_features),
        ("weighted_points", _weighted_shortcut_features),
        ("route_depth", lambda row: _route_shortcut_features(row["state"])),
        ("timely_streak", lambda row: _streak_shortcut_features(row["state"])),
    ):
        family_rows = [row for row in rows if row["family"] == f"score_{family}"]
        by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        by_feature: dict[tuple[Any, ...], collections.Counter[int]] = (
            collections.defaultdict(collections.Counter)
        )
        for row in family_rows:
            feature = feature_fn(row)
            by_group[row["group_id"]].append(row)
            by_feature[feature][row["label"]] += 1
        if len(by_group) != GROUPS_PER_FAMILY or any(
            len(group) != 3 or len({feature_fn(row) for row in group}) != 1
            for group in by_group.values()
        ):
            raise ValueError(f"{family} has a within-group count-only shortcut")
        count_classifier_correct = sum(
            max(counts.values()) for counts in by_feature.values()
        )
        if count_classifier_correct * 3 != len(family_rows):
            raise ValueError(f"{family} count-only classifier exceeds chance")
        report[family] = {
            "rows": len(family_rows),
            "groups": len(by_group),
            "feature_buckets": len(by_feature),
            "count_only_correct": count_classifier_correct,
            "count_only_total": len(family_rows),
        }
        if family == "weighted_points":
            report[family]["unweighted_rule"] = _unweighted_rule_audit(family_rows)
            report[family]["single_signal_heldout"] = _single_signal_heldout(
                family_rows
            )
            for group in by_group.values():
                if any(
                    len({row["state"]["signals"][position]["mark"] for row in group})
                    == 3
                    for position in range(5)
                ):
                    raise ValueError(
                        "Weighted group has a three-way single-signal rank"
                    )
        if family == "timely_streak":
            report[family]["per_day_heldout"] = _streak_day_heldout(family_rows)
    return report


def generate() -> list[dict[str, Any]]:
    rows = []
    weighted_balance: collections.Counter[tuple[int, int, int, int]] = (
        collections.Counter()
    )
    for family_offset, family in enumerate(FAMILIES):
        for index in range(GROUPS_PER_FAMILY):
            rng = _rng(family, index)
            language = "zh" if (index + family_offset) % 4 == 0 else "en"
            case = _case(rng, language)
            if family == "obligation_review":
                states = _obligation_states(rng, case, language, index)
            elif family == "weighted_points":
                states = _weighted_states(rng, case, language, weighted_balance)
            elif family == "route_depth":
                states = _route_states(rng, case)
            else:
                states = _streak_states(rng, case, language)
            instructions, options = _question(family, language, rng, case)
            stem = _sha(f"{SEED}\0{family}\0{index}")[:20]
            for level, state in enumerate(states):
                if oracle(family, state) != level:
                    raise AssertionError(
                        f"Oracle disagrees with construction: {family}/{index}/{level}"
                    )
                row = {
                    "id": f"d2scv4_{stem}_{level}",
                    "state": state,
                    "instructions": instructions,
                    "options": options,
                    "label": level,
                    "task_type": "score",
                    "family": f"score_{family}",
                    "group_id": f"d2scg_v4_{stem}",
                    "language": language,
                    "split": "train",
                    "source": SOURCE,
                    "evaluation_role": "train",
                    "render_template": f"score_curriculum_{family}_v4",
                    "audit_metadata": {
                        "generation": "internal deterministic oracle",
                        "seed_sha256": _sha(SEED),
                        "group_ordinal": index,
                        "variant_level": level,
                    },
                }
                row["input_sha256"] = pilot.input_sha256(row)
                pilot.validate_train_row(row)
                rows.append(row)
    rows.sort(key=lambda row: (_sha(f"{SEED}\0order\0{row['id']}"), row["id"]))
    if len(rows) != EXPECTED_ROWS:
        raise AssertionError("Score curriculum size changed")
    shortcut_audit(rows)
    return rows


def _load_protected(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inventory, list):
        raise ValueError("Protected inventory must be a list of role/path objects")
    roles = {entry.get("role") for entry in inventory}
    if not roles.issuperset(REQUIRED_PROTECTED_ROLES) or len(roles) != len(inventory):
        raise ValueError("Protected roles missing or duplicated")
    references, evidence = {}, []
    for entry in inventory:
        role, file = entry["role"], Path(entry["path"])
        if "gold" in file.name.lower() or not (
            file.name.endswith(".prompts.jsonl") or file.name == "prompts.jsonl"
        ):
            raise ValueError(f"Only gold-free prompt inputs permitted for {role}")
        rows, receipt = targeted.load_context_reference(file)
        if not all(
            set(json.loads(line)) == {"id", "state", "questions"}
            for line in file.read_text(encoding="utf-8").splitlines()
        ):
            raise ValueError(f"Protected {role} prompt contains non-prompt fields")
        references[role] = rows
        evidence.append(
            {
                "role": role,
                "basename": file.name,
                "sha256": receipt["sha256"],
                "rows": receipt["rows"],
            }
        )
    return references, sorted(evidence, key=lambda item: item["role"])


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    files = {
        "base": args.base_train,
        "base_manifest": args.base_manifest,
        "select": args.select_file,
        "cal": args.cal_file,
    }
    for role, path in files.items():
        if pilot.sha_file(path) != FROZEN[role]:
            raise ValueError(f"Frozen {role} SHA mismatch")
    base = load_partition(args.base_train, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    if (len(base), len(select), len(cal)) != (7455, 700, 700):
        raise ValueError("Frozen partition cardinality changed")
    original = generate()
    shortcut_report = shortcut_audit(original)
    references, reference_receipts = _load_protected(args.protected_list)
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in original:
        by_group[row["group_id"]].append(row)
    quarantine = collections.Counter()
    audit = {}
    for role, protected in (
        ("base", base),
        ("select", select),
        ("cal", cal),
        *sorted(references.items()),
    ):
        current = [row for group in sorted(by_group) for row in by_group[group]]
        current_context = targeted.context_rows(current)
        protected_context = targeted.context_rows(protected)
        exact_ids = {row.get("id") for row in protected}
        exact_groups = {row.get("group_id") for row in protected}
        exact_inputs = {row.get("input_sha256") for row in protected}
        protected_pairs = {targeted.text_hashes(row["state"]) for row in protected}
        raw_hashes, normalized_hashes = (
            {x[0] for x in protected_pairs},
            {x[1] for x in protected_pairs},
        )
        rejected_ids = {
            row["id"]
            for row in current
            if row["id"] in exact_ids
            or row["group_id"] in exact_groups
            or row["input_sha256"] in exact_inputs
            or (pair := targeted.text_hashes(row["state"]))[0] in raw_hashes
            or pair[1] in normalized_hashes
        }
        near = pilot.near_duplicates(
            current_context, protected_context, collect_left_ids=True
        )
        near_ids = set(near.pop("left_ids"))
        rejected_ids.update(near_ids)
        dropped = [
            group
            for group, rows in by_group.items()
            if any(row["id"] in rejected_ids for row in rows)
        ]
        for group in dropped:
            quarantine[role] += len(by_group.pop(group))
        audit[role] = {
            "matched_rows": len(rejected_ids),
            "quarantined_groups": len(dropped),
            "quarantined_rows": 3 * len(dropped),
            "near": near,
        }
    candidate = [row for group in sorted(by_group) for row in by_group[group]]
    if len(candidate) < 900 or len(candidate) % 3:
        raise ValueError(f"Too few group-complete Score rows remain: {len(candidate)}")
    family_groups = collections.Counter(rows[0]["family"] for rows in by_group.values())
    if any(family_groups[f"score_{family}"] < 72 for family in FAMILIES):
        raise ValueError(
            f"A Score mechanism lost too many source groups: {family_groups}"
        )
    for rows in by_group.values():
        if len(rows) != 3 or {row["label"] for row in rows} != {0, 1, 2}:
            raise ValueError("A Score source group was split or unbalanced")
    if pilot.train_consistency_audit([*base, *candidate])["conflicting_gold_groups"]:
        raise ValueError("Conflicting labels in merged TRAIN")
    check_partition_isolation(
        {"train": [*base, *candidate], "select": select, "cal": cal}
    )
    for role, reference in references.items():
        targeted.context_overlap(candidate, reference, approximate=True)
    if args.tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer.resolve()),
            local_files_only=True,
            trust_remote_code=False,
        )
        lengths = [pilot.count_tokens(row, tokenizer) for row in candidate]
        if max(lengths) > args.max_row_tokens:
            raise ValueError("Score candidate exceeds tokenizer length cap")
        token_audit = {
            "tokenizer_revision": args.tokenizer_revision,
            "max_row_tokens": args.max_row_tokens,
            "min": min(lengths),
            "max": max(lengths),
            "total": sum(lengths),
        }
    else:
        token_audit = {
            "status": "unverified_no_tokenizer",
            "max_row_tokens": args.max_row_tokens,
        }
    merged = [*base, *candidate]
    merged.sort(key=lambda row: (_sha(f"{SEED}\0merged\0{row['id']}"), row["id"]))
    payloads = {
        "score_curriculum.train.jsonl": pilot.jsonl_bytes(candidate),
        "score_augmented.train.jsonl": pilot.jsonl_bytes(merged),
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for filename, payload in payloads.items():
        pilot._atomic_write(args.output_dir / filename, payload)
    rights_manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "decision20-score-curriculum/4",
        "status": "research_candidate_not_training_approved",
        "builder_sha256": pilot.sha_file(Path(__file__)),
        "seed_sha256": _sha(SEED),
        "input_sha256": {role: FROZEN[role] for role in files},
        "protected_inventory_sha256": pilot.sha_file(args.protected_list),
        "protected_sources": reference_receipts,
        "original_rows": len(original),
        "candidate_rows": len(candidate),
        "candidate_groups": len(by_group),
        "candidate_counts": {
            field: dict(
                sorted(collections.Counter(row[field] for row in candidate).items())
            )
            for field in ("family", "language", "label", "source")
        },
        "quarantined_rows_by_role": dict(sorted(quarantine.items())),
        "overlap_audit": audit,
        "token_audit": token_audit,
        "shortcut_audit": shortcut_report,
        "rights": {
            "added_source": "internally generated by Decision 2.0 research; private TRAIN only",
            "base_source_rights": rights_manifest["source_rights"],
            "external_text_in_added_rows": False,
        },
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "rows": len(payload.splitlines()),
            }
            for name, payload in payloads.items()
        },
        "limitations": [
            "Synthetic deterministic labels do not prove natural decision transfer.",
            "Obligation review shares an abstract three-level precedence skill with typed DEV, but has independently authored schemas and prompts.",
            "Approximate near-overlap cannot certify semantic independence.",
            "No sealed FINAL or heldout CSS labels were accessed.",
        ],
    }
    pilot._atomic_write(
        args.output_dir / "manifest.json",
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--select-file", required=True, type=Path)
    parser.add_argument("--cal-file", required=True, type=Path)
    parser.add_argument("--protected-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-row-tokens", type=int, default=1024)
    args = parser.parse_args()
    if bool(args.tokenizer) != bool(args.tokenizer_revision):
        parser.error("--tokenizer and --tokenizer-revision must be supplied together")
    manifest = build(args)
    print(
        json.dumps(
            {
                "rows": manifest["candidate_rows"],
                "groups": manifest["candidate_groups"],
                "sha256": manifest["outputs"]["score_curriculum.train.jsonl"]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
