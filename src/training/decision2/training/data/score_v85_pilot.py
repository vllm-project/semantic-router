"""Prospective CPU-only Score v8.5 deep-dossier quality pilot.

The private seed and signed preregistration must exist before this module is run.
Candidate rows are not admitted training data. The combined reviewer packet has
no labels, role names, or aggregate class counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from training.data.score_v8_pilot import _write_jsonl, blind_packet
from training.model.data import INPUT_FIELDS, digest, file_sha256, validate_row

VERSION = "decision2-score-v8.5-deep-dossier-pilot/1"
MECHANISMS = ("workflow", "connection", "stock", "policy", "evidence", "service_level")
GROUPS = {"train": 3, "select": 2}
LEVELS = (0, 1, 2)
SETTINGS = {
    "workflow": (
        ("harbor archive transfer", "en"),
        ("community theatre relighting", "en"),
        ("山城医院设备交接", "zh"),
        ("botanical seed vault move", "en"),
        ("河口气象站启用", "zh"),
    ),
    "connection": (
        ("island ferry terminal", "en"),
        ("mountain rail interchange", "en"),
        ("西江客运换乘站", "zh"),
        ("university shuttle hub", "en"),
        ("东湾渡轮码头", "zh"),
    ),
    "stock": (
        ("regional conservation store", "en"),
        ("school science stockroom", "en"),
        ("东城图书修复库", "zh"),
        ("field laboratory depot", "en"),
        ("海港剧院道具仓", "zh"),
    ),
    "policy": (
        ("county service office", "en"),
        ("community health van", "en"),
        ("南湖公共档案室", "zh"),
        ("coastal research vessel", "en"),
        ("青川流动图书车", "zh"),
    ),
    "evidence": (
        ("river sample registry", "en"),
        ("museum loan archive", "en"),
        ("古城文物修护中心", "zh"),
        ("grid sensor archive", "en"),
        ("海湾湿地监测组", "zh"),
    ),
    "service_level": (
        ("regional repair desk", "en"),
        ("university instrument office", "en"),
        ("东岭医疗器械服务台", "zh"),
        ("coastal communications team", "en"),
        ("石桥公共设施值守处", "zh"),
    ),
}
OPTIONS = {
    "workflow": (("Blocked", "Pending", "Ready"), ("受阻", "待完成", "已就绪")),
    "connection": (
        ("Cannot connect", "Connection uncertain", "Can connect"),
        ("无法接驳", "能否接驳未定", "可以接驳"),
    ),
    "stock": (
        (
            "Short even with pending units",
            "Depends on pending units",
            "Covered by confirmed units",
        ),
        ("即使待到货也不足", "取决于待到货", "已确认数量足够"),
    ),
    "policy": (
        ("Ineligible", "Evidence pending", "Eligible"),
        ("不符合", "材料待补", "符合"),
    ),
    "evidence": (
        ("Contradicted", "Not established", "Supported"),
        ("受到反证", "尚无定论", "得到支持"),
    ),
    "service_level": (
        ("Late", "Timely but conditional", "Timely and confirmed"),
        ("超过时限", "按期但条件未确认", "按期且条件已确认"),
    ),
}
INSTRUCTIONS = {
    "workflow": (
        "Use the named handoff chain and the latest signed reconciliation. A failed prerequisite gives 0; otherwise an unfinished prerequisite gives 1; all prerequisites complete gives 2. The older board is historical.",
        "按指定事项的依赖链和最新签署的核对记录判断：有前置任务失败记 0；没有失败但仍有任务待办记 1；全部完成记 2。旧看板只供追溯。",
    ),
    "connection": (
        "For the named incoming and onward services, add the interchange walk to both arrival-window endpoints. Use the later signed boarding cutoff for that onward service. Earliest beyond cutoff gives 0; a window crossing it gives 1; latest at or before it gives 2.",
        "只判断指定的抵达班次和后续班次。到达区间两端均加上换乘步行时间，并采用该后续班次最新签署的检票截止时间。最早也来不及记 0；区间跨越截止时间记 1；最迟仍能赶上记 2。",
    ),
    "stock": (
        "Use the requested SKU only. Confirmed usable units are on hand minus reserved plus signed carrier-confirmed arrivals. Pending arrivals are not guaranteed. Even the optimistic total short gives 0; only pending units closing the gap gives 1; confirmed units meeting demand gives 2.",
        "只按申请中的货号计算：已确认可用量为现货减已预留量，再加承运方签收确认的到货量。待确认到货不作保证。乐观合计仍不足记 0；仅靠待到货补足记 1；已确认数量足够记 2。",
    ),
    "policy": (
        "Apply the base permit and inspection rule, the region/date-limited exception, and the dated training amendment. An expired permit or confirmed failure of a current mandatory check gives 0; pending proof for a current requirement gives 1; all current requirements met gives 2.",
        "按申请地区和日期适用许可证、现场检查、地域例外及培训修订。许可证过期或现行必备检查明确未通过记 0；现行要求的证明仍待核实记 1；全部适用要求已满足记 2。",
    ),
    "evidence": (
        "Check the named claim against both independent source observations. A contrary observation gives 0; if neither contradicts but a required observation is missing, give 1; if both affirm the claim, give 2. Match full identifiers and use the latest attributed record.",
        "核对目标编号的两份独立观察记录。任一记录明确相反记 0；没有反证但缺少必要观察记 1；两份记录都支持结论记 2。须核对完整编号及有署名的最新记录。",
    ),
    "service_level": (
        "Count the stated business days after the request date, excluding weekends and the listed local holiday. Completion on the due date is timely. A later completion gives 0; timely completion with an unconfirmed prerequisite gives 1; timely completion with it confirmed gives 2.",
        "从请求次日起计算约定工作日，不计周末及所列本地假日；到期当天完成算按期。逾期记 0；按期但前置条件未确认记 1；按期且前置条件已确认记 2。",
    ),
}


def _rng(secret: bytes, role: str, mechanism: str, index: int) -> random.Random:
    message = f"{VERSION}\0{role}\0{mechanism}\0{index}".encode()
    return random.Random(
        int.from_bytes(hmac.new(secret, message, hashlib.sha256).digest()[:16], "big")
    )


def _tag(rng: random.Random, prefix: str) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{prefix}-{''.join(rng.choices(letters, k=4))}{rng.randrange(100, 999)}"


def _near(identifier: str) -> str:
    return identifier[:-1] + str((int(identifier[-1]) + 1) % 10)


def _clock(minute: int) -> str:
    if not 0 <= minute < 1440:
        raise ValueError("Time crossed day boundary")
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _minute(clock: str) -> int:
    hour, minute = map(int, clock.split(":"))
    return hour * 60 + minute


def _business_due(start: dt.date, days: int, holidays: set[dt.date]) -> dt.date:
    current = start
    while days:
        current += dt.timedelta(days=1)
        if current.weekday() < 5 and current not in holidays:
            days -= 1
    return current


def _base(rng: random.Random, role: str, mechanism: str, index: int) -> dict[str, Any]:
    style = index + (0 if role == "train" else 3)
    setting, language = SETTINGS[mechanism][style]
    target = _tag(rng, "ITEM")
    result: dict[str, Any] = {
        "role": role,
        "mechanism": mechanism,
        "style": style,
        "setting": setting,
        "language": language,
        "case": _tag(rng, "CASE"),
        "target": target,
        "near": _near(target),
        "long": style != 3,
        "very_long": style == 0 and mechanism in ("workflow", "policy"),
        "core_order": rng.sample(range(4), 4),
        "dossier_seed": rng.getrandbits(64),
    }
    if mechanism == "workflow":
        result["tasks"] = [_tag(rng, "TASK") for _ in range(2 + style % 3)]
        result["pivot"] = rng.randrange(len(result["tasks"]))
    elif mechanism == "connection":
        result.update(
            incoming=_tag(rng, "IN"), onward=_tag(rng, "OUT"), walk=rng.randrange(4, 12)
        )
        result["near"] = _near(result["onward"])
        result["target"] = result["onward"]
        arrival = rng.randrange(9 * 60, 17 * 60)
        result.update(earliest=arrival, latest=arrival + rng.randrange(8, 16))
        result["old_cutoff"] = arrival + result["walk"] + rng.randrange(-2, 8)
    elif mechanism == "stock":
        result.update(
            demand=rng.randrange(22, 54),
            reserved=rng.randrange(2, 9),
            gap=rng.randrange(6, 11),
        )
        result["onhand"] = result["demand"] + result["reserved"] - result["gap"]
        result["sku_order"] = rng.sample(
            [result["target"], result["near"], _tag(rng, "ITEM")], 3
        )
    elif mechanism == "policy":
        start = dt.date(2026, 4, 3) + dt.timedelta(days=4 * style)
        result.update(
            exception_date=(start + dt.timedelta(days=2)).isoformat(),
            amendment_date=(start + dt.timedelta(days=5)).isoformat(),
        )
        result["exception_region"] = "North" if language == "en" else "北区"
        result["region"] = (
            result["exception_region"]
            if style in (0, 2, 4)
            else ("South" if language == "en" else "南区")
        )
        offset = {0: 9, 1: 10, 2: 1, 3: 4, 4: 11}[style]
        result["request_date"] = (start + dt.timedelta(days=offset)).isoformat()
    elif mechanism == "evidence":
        result.update(source_a=_tag(rng, "DOC"), source_b=_tag(rng, "DOC"))
    elif mechanism == "service_level":
        start = dt.date(2026, 5, 5) + dt.timedelta(days=(0, 8, 16, 23, 30)[style])
        while start.weekday() >= 5:
            start += dt.timedelta(days=1)
        days = 2 + style % 3
        holiday = start + dt.timedelta(days=1 + style % 3)
        result.update(
            request_date=start.isoformat(), holiday=holiday.isoformat(), days=days
        )
        result["due"] = _business_due(start, days, {holiday}).isoformat()
    else:
        raise ValueError(mechanism)
    if result["long"]:
        count = 23 if result["very_long"] else 12
        result["deep_position"] = rng.randrange(6, count - 1)
    return result


def _facts(base: dict[str, Any], level: int) -> dict[str, Any]:
    f = dict(base)
    mechanism = f["mechanism"]
    if mechanism == "workflow":
        statuses = ["complete"] * len(f["tasks"])
        statuses[f["pivot"]] = ("failed", "queued", "complete")[level]
        if level < 2:
            for descendant in range(f["pivot"] + 1, len(statuses)):
                statuses[descendant] = "queued"
        f["statuses"] = statuses
    elif mechanism == "connection":
        first = f["earliest"] + f["walk"]
        last = f["latest"] + f["walk"]
        f["cutoff"] = (first - 2, first + (last - first) // 2, last)[level]
    elif mechanism == "stock":
        gap = f["gap"]
        f["confirmed"], f["pending"] = ((gap - 5, 2), (gap - 5, 7), (gap + 1, 2))[level]
    elif mechanism == "policy":
        exception = (
            f["region"] == f["exception_region"]
            and f["request_date"] >= f["exception_date"]
        )
        training = f["request_date"] >= f["amendment_date"]
        f["permit"] = "expired" if level == 0 else "valid"
        f["inspection"] = (
            "waived"
            if exception
            else ("pending" if level == 1 and not training else "complete")
        )
        f["training"] = "pending" if level == 1 and training else "filed"
    elif mechanism == "evidence":
        f["finding"] = ("contrary", "unobserved", "affirmed")[level]
    elif mechanism == "service_level":
        due = dt.date.fromisoformat(f["due"])
        f["service_date"] = (
            due + dt.timedelta(days=1 if level == 0 else -(f["style"] % 2))
        ).isoformat()
        f["prerequisite"] = "pending" if level == 1 else "signed"
    else:
        raise ValueError(mechanism)
    return f


def oracle(f: dict[str, Any]) -> int:
    mechanism = f["mechanism"]
    if mechanism == "workflow":
        return 0 if "failed" in f["statuses"] else 1 if "queued" in f["statuses"] else 2
    if mechanism == "connection":
        first, last = f["earliest"] + f["walk"], f["latest"] + f["walk"]
        return 0 if first > f["cutoff"] else 1 if last > f["cutoff"] else 2
    if mechanism == "stock":
        confirmed = f["onhand"] - f["reserved"] + f["confirmed"]
        return (
            0
            if confirmed + f["pending"] < f["demand"]
            else 1 if confirmed < f["demand"] else 2
        )
    if mechanism == "policy":
        exception = (
            f["region"] == f["exception_region"]
            and f["request_date"] >= f["exception_date"]
        )
        training = f["request_date"] >= f["amendment_date"]
        if f["permit"] != "valid" or (not exception and f["inspection"] == "failed"):
            return 0
        if (not exception and f["inspection"] != "complete") or (
            training and f["training"] != "filed"
        ):
            return 1
        return 2
    if mechanism == "evidence":
        return {"contrary": 0, "unobserved": 1, "affirmed": 2}[f["finding"]]
    if mechanism == "service_level":
        due = _business_due(
            dt.date.fromisoformat(f["request_date"]),
            f["days"],
            {dt.date.fromisoformat(f["holiday"])},
        )
        service = dt.date.fromisoformat(f["service_date"])
        return 0 if service > due else 1 if f["prerequisite"] != "signed" else 2
    raise ValueError(mechanism)


# These notes stay within each case domain. The case-specific signed update is
# inserted among them and is necessary for every long-row decision.
TOPICS_EN = {
    "workflow": (
        "The coordinator indexed the handoff under {case} and checked that the task identifiers on the plan match the equipment register. The index is useful for finding the signed reconciliation, but it was prepared before the final task review and does not itself close any ancestor.",
        "A shift leader compared the plan with the work order and retained both versions. The earlier version explains why a task appears on the board, while the later signed task record controls its actual completion state for {target}.",
        "The receiving desk logged who may authorize a change to this handoff. A verbal comment in the corridor was not entered as a task completion; the desk requires the author, timestamp, and complete identifier on the formal update.",
        "The adjacent handoff {near} uses the same crew and a similar sequence of checks. Its sign-off is filed in this binder to help the next shift, but its task results cannot be transferred to {target}.",
        "The supervisor checked the chain arrows against the original plan because a completed downstream action can appear on a stale display. Staff were told to follow the signed ancestor statuses rather than infer readiness from a single green tile.",
        "A facilities call changed the room used for the final handoff. The room change was circulated with {case}, but it did not remove any prerequisite from the approved chain or certify that the chain was complete.",
    ),
    "connection": (
        "The station posted the platform route for passengers changing to {target}. Staff walked the route and verified that the published transfer minutes still include the corridor and boarding gate, rather than only the platform-to-platform distance.",
        "A timetable printout retained the former booking cutoff for audit. The later signed service bulletin is filed separately because the cutoff can change without changing the arrival forecast for the incoming service.",
        "The nearby onward service {near} shares a platform with the requested departure. Its separate boarding notice is kept in the same folder; the two services must not be merged merely because the signs are next to each other.",
        "The dispatch team reviewed the lower and upper ends of the arrival interval. An early arrival is possible but not guaranteed, so staff did not collapse the forecast into a single average time when advising a transfer passenger.",
        "A station agent noted that a gate queue may fluctuate during the hour. The published interchange time remains the minimum used for this case; there is no authorized extra buffer in the booking rule.",
        "The service desk archived a call about the adjacent route after checking the full onward identifier. That call did not amend the cutoff for {target}; a signed bulletin tied to the named service is required for that change.",
    ),
    "stock": (
        "The storekeeper matched the requisition for {target} to the shelf ledger and the reservation book. A bulk count for the whole room would hide units already promised to other work orders, so the item-level figures remain separate.",
        "Receiving staff checked packaging labels against the carrier reference before opening cartons. A delivery for {near} occupied the next shelf, and its quantities were logged under that full SKU instead of being pooled with {target}.",
        "The order clerk retained a preliminary supplier email because it explains the planned shipment. The email is not a signed carrier receipt and does not turn expected units into confirmed usable stock for this requisition.",
        "The daily reservation snapshot was copied into this dossier with its timestamp. Later fulfillment must use the requested SKU and the actual reservation count rather than subtracting every reservation in the warehouse.",
        "A stock auditor checked whether any counted units were damaged or returned. The usable on-hand figure in the current sheet already excludes those units, so the carrier notice is the remaining source for inbound availability.",
        "The neighboring request uses a similar quantity and a similarly numbered item. Staff attached both request forms for traceability, but the allocation decision here is tied to {target} and the demand stated on its own form.",
    ),
    "policy": (
        "The policy desk preserved the previous edition of the eligibility guide under {case}. It is useful for explaining the sequence of amendments, but an application must be tested against the provisions effective for its own date and region.",
        "A clerk checked the exact permit identifier before linking the registry response. The neighboring application {near} was received on the same day, yet its region and inspection file remain separate from those of {target}.",
        "The legal team circulated a short reading note for field staff. It repeats that the regional inspection exception has a defined start date and does not waive a valid permit or any later training requirement that is in force.",
        "The filing office retained an unsigned draft of an earlier amendment. Staff marked it historical because the signed effective-date notice, not the draft, determines whether training proof is required for this request.",
        "The intake register records the applicant region as submitted, rather than the location of the office receiving the form. This prevents an application from borrowing the exception merely because the processing desk is in a different region.",
        "A routine review verified that the permit, inspection, and training checks are independently attributed. A general approval summary cannot substitute for the current signed check tied to the full identifier of {target}.",
    ),
    "evidence": (
        "The archivist matched the requested item {target} to its intake image before reviewing the second observation. A nearby number {near} appears in the same binder, so staff kept each observation beside the full item identifier.",
        "The source register records who collected each observation and when it entered the file. A later upload time does not by itself reverse a signed finding; the reviewer must read the observation associated with this claim.",
        "A prior worksheet for {target} is kept for traceability after the follow-up inspection. The worksheet shows what was planned, while the current attributed observation records what was actually found.",
        "The archive clerk noted that an empty observation box is different from a negative result. A required test that was never performed leaves the claim open, whereas a documented contrary observation weighs against it.",
        "A second team checked that the two source documents were independent rather than copies of one report. The claim requires both observations and cannot be established from the intake note alone.",
        "The neighboring item {near} was examined during the same shift. Similar handling and the same staff signatures do not make its result evidence for {target}; the full identifier on each sheet controls.",
    ),
    "service_level": (
        "The service desk filed the request date under {case} and confirmed that the contract starts counting on the next eligible business day. Staff kept the calculation separate from the time a copy was uploaded to the archive.",
        "The calendar clerk checked the listed local holiday against the office schedule. A different locality may operate that day, but this obligation uses the calendar named in the case file rather than a national generalization.",
        "The dispatch team attached the technician log for the nearby item {near}. Its completion date does not establish timeliness for {target}, even though both visits were assigned to the same service route.",
        "A coordinator marked weekends on the planning sheet and counted only the promised number of working days. Delivery on the last counted day still satisfies the date component of the service obligation.",
        "The prerequisite is documented separately from the visit date because a timely appointment may still depend on an unsigned access or safety check. The current signed service record must be read for both facts.",
        "The archive retained an older provisional appointment for {target}. The provisional entry explains dispatch planning, but it was not the completed service record and does not settle the final delivery date.",
    ),
}
TOPICS_ZH = {
    "workflow": (
        "值班主管按案号 {case} 核对交接单和任务编号，确认各前置环节在同一依赖链内。索引只方便寻找正式核对记录，不能把旧看板上的绿色状态直接当作最终完成证明。",
        "交接组保留了前一版计划及修订后的工单，两份材料标明各自日期。旧计划解释任务为何进入排期，针对 {target} 的现行状态仍以签署后的核对记录为准。",
        "相邻事项 {near} 由同一班组处理，编号也十分接近。档案员分别装订两份签收页，提醒下一班不能把相邻事项的任务完成情况套用到 {target}。",
        "现场换班时有人口头提到进度，记录员没有据此更改状态栏。只有列明任务编号、签署人和形成时间的正式记录，才能更新本案前置任务的状态。",
        "组长复核了依赖箭头的方向：后续工作在旧看板上显示完成，不代表更早的必要环节已通过。交接结论需要逐项读取最新核对内容。",
        "总务处更换了交接房间，并将新门牌写入通知。地点调整不改变原定依赖链，也不能代替任何一项技术验收结果。",
    ),
    "connection": (
        "车站实地核对了通往 {target} 检票口的步行路线。公布的换乘分钟数已经包含走廊和闸口之间的行程，工作人员没有另行添加未经批准的缓冲时间。",
        "调度室留存旧版检票公告以便追溯，后来签署的班次通知单独归档。检票截止时间可能变化，而抵达班次的预报区间并不会因此自动调整。",
        "相邻后续班次 {near} 与本次班车共用候车区，两张公告张贴得很近。工作人员按完整班次编号归档，避免把邻班的截止时间用于 {target}。",
        "值班员同时查看抵达区间的最早和最迟时间。较早抵达只是可能情形，不能用平均到站时间替代整个预报区间。",
        "现场引导员记录了高峰时段的排队情况，但本案的计算规则采用公布的最低步行时间。排队观察没有形成更改检票规则的正式通知。",
        "服务台接到关于邻线的咨询后，核对了完整班次编号。该通话没有变更 {target} 的检票时间；相关变更必须以针对本班次的签署公告为准。",
    ),
    "stock": (
        "仓管员按 {target} 的完整货号核对领料申请、货架账和预留记录。全库合计数量会掩盖已经分配给其他工单的物料，因此不能直接用于本次配货。",
        "收货人员核对了外箱标签和承运单号。相邻货号 {near} 的箱子放在同一区域，但它的件数仍单独入账，不并入 {target} 的可用数量。",
        "采购员保留了供应商的初步邮件，以说明预计发货安排。邮件不是承运方签署的到货回执，不能把预计件数直接算作已确认库存。",
        "每日预留快照附有形成时间。计算本次可用量时，应扣除目标货号的预留件数，而不是将仓库里所有预留数量一并扣除。",
        "盘点员将破损和退回物料从现货数中剔除，当前库存栏已经反映这一步处理。后续是否有可用到货，还须查看针对目标货号的承运记录。",
        "相邻申请的需求数量与本次接近，工作人员因此在封面写明两个完整货号。类似数字不能代替对 {target} 的逐项核算。",
    ),
    "policy": (
        "政策窗口在案号 {case} 下保留了旧版细则，方便解释修订前后的区别。申请是否符合条件仍要按自身日期、地区以及已正式生效的条款判断。",
        "窗口核对了目标 {target} 的完整许可证编号。相邻申请 {near} 虽在同一天收件，但地区和现场检查材料均单独归档，不能互相借用。",
        "法务组向办理人员说明地域例外的起始日期和适用范围。例外只涉及现场检查，不免除有效许可证，也不覆盖随后已经生效的培训要求。",
        "一份未签署的修订草稿留在历史附件中。工作人员已注明草稿不产生效力，培训材料是否必需应以正式公告的生效日期为准。",
        "登记簿填写的是申请事项所在地区，而非受理窗口地址。这样的区分可避免因办事处位置不同而错误套用地域例外。",
        "例行复核分别检查许可证、现场记录和培训材料的来源。汇总审批表不能代替针对 {target} 的最新签署核查单。",
    ),
    "evidence": (
        "档案员先按完整编号找到 {target} 的入库影像，再查看另一份观察记录。同一卷内还有编号相近的 {near}，两者的结论须分别对应原件。",
        "来源登记簿记载了观察人员和资料归档时间。文件上传较晚不等于观察形成较晚，审查时仍应阅读与本项主张直接相关的签署内容。",
        "目标 {target} 的早期工作表保留在附件中，用于说明当时计划检查哪些项目。工作表不能代替后来实际完成的观察。",
        "记录员区分了空白检测栏与明确的阴性结果。尚未实施必要检测只能让结论待定，实际观察到相反情况才构成反证。",
        "质控组确认两份来源记录分别形成，并非同一报告的复印件。只凭第一份记录无法完成对目标主张的双来源核查。",
        "相邻编号 {near} 在同一班次接受检查，签署人员也相同。处理流程相似并不使它的检测结果成为 {target} 的证据。",
    ),
    "service_level": (
        "服务台按案号 {case} 登记请求日期，确认合同从次一个有效工作日起计数。扫描件上传的时间不作为服务时限的起点。",
        "排班人员核对了本地假日表。其他地区即使在当天照常办公，本案仍须采用合同所指的当地日历。",
        "相邻设备 {near} 的维修记录附在同一趟派工资料中。它的完成日期不能证明 {target} 何时交付，即使两次上门由同一名技师处理。",
        "调度员在计划表上标出周末和假日，只把合格工作日计入承诺期限。最后一个工作日完成，日期条件仍算满足。",
        "前置确认与上门日期分别留痕：如安全检查尚未签字，即使派工日期及时，交付结论仍带有条件。应核对最新签署的服务记录。",
        "档案中还有 {target} 的早期预约时间，用于解释排班过程。预约并不等于实际交付，不能据此确定最终完成日期。",
    ),
}


def _sections(f: dict[str, Any]) -> tuple[list[str], str, str, str]:
    """Return four core sections, a decisive note, near-item note, and old note."""
    z = f["language"] == "zh"
    case, target, near = f["case"], f["target"], f["near"]
    m = f["mechanism"]
    if m == "workflow":
        chain = " <- ".join([target, *reversed(f["tasks"])])
        latest = "; ".join(
            f"{task}={status}" for task, status in zip(f["tasks"], f["statuses"])
        )
        latest_zh = "; ".join(
            f"{task}={dict(complete='完成', failed='失败', queued='待办')[status]}"
            for task, status in zip(f["tasks"], f["statuses"])
        )
        old = "; ".join(f"{task}=complete" for task in f["tasks"])
        if z:
            core = [
                f"交接请求：案号 {case}，目标 {target}。",
                f"任务依赖链：{chain}。",
                f"旧看板（2026-06-01）：{old.replace('=complete', '=完成')}。",
                "核对约定：新签署的任务核对记录覆盖旧看板，须逐项检查依赖链。",
            ]
            critical = f"最新签署核对（2026-07-01）[{case}] {target}：{latest_zh}。"
            near_note = f"最新签署核对（2026-07-01）[{case}] {near}：{old.replace('=complete', '=完成')}。"
            stale = f"旧版核对（2026-06-30）[{case}] {target}：{old.replace('=complete', '=完成')}。"
        else:
            core = [
                f"Handoff request: case {case}, target {target}.",
                f"Dependency chain: {chain}.",
                f"Old board (2026-06-01): {old}.",
                "Reconciliation rule: the later signed task record supersedes the old board; inspect every ancestor in the chain.",
            ]
            critical = f"Latest signed reconciliation (2026-07-01) [{case}] {target}: {latest}."
            near_note = (
                f"Latest signed reconciliation (2026-07-01) [{case}] {near}: {old}."
            )
            stale = f"Earlier reconciliation (2026-06-30) [{case}] {target}: {old}."
    elif m == "connection":
        incoming, onward = f["incoming"], f["onward"]
        nearby_cutoff = _clock(f["old_cutoff"] + 11)
        if z:
            core = [
                f"换乘请求：案号 {case}，抵达班次 {incoming}，后续班次 {onward}。",
                f"抵达预报 [{case}] {incoming}：{_clock(f['earliest'])} 至 {_clock(f['latest'])}。",
                f"站内步行 [{case}] {incoming} 至 {onward}：{f['walk']} 分钟。",
                f"旧检票公告 [{case}] {onward}：截止 {_clock(f['old_cutoff'])}；后续调整以签署通知为准。",
            ]
            critical = f"最新签署检票通知（2026-07-01）[{case}] {onward}：检票截止 {_clock(f['cutoff'])}。"
            near_note = f"最新签署检票通知（2026-07-01）[{case}] {near}：检票截止 {nearby_cutoff}。"
            stale = f"早期检票通知（2026-06-30）[{case}] {onward}：检票截止 {_clock(f['old_cutoff'])}。"
        else:
            core = [
                f"Transfer request: case {case}, incoming {incoming}, onward {onward}.",
                f"Arrival forecast [{case}] {incoming}: {_clock(f['earliest'])} to {_clock(f['latest'])}.",
                f"Interchange walk [{case}] {incoming} to {onward}: {f['walk']} minutes.",
                f"Old boarding notice [{case}] {onward}: cutoff {_clock(f['old_cutoff'])}; a later signed notice controls any change.",
            ]
            critical = f"Latest signed boarding notice (2026-07-01) [{case}] {onward}: cutoff {_clock(f['cutoff'])}."
            near_note = f"Latest signed boarding notice (2026-07-01) [{case}] {near}: cutoff {nearby_cutoff}."
            stale = f"Earlier boarding notice (2026-06-30) [{case}] {onward}: cutoff {_clock(f['old_cutoff'])}."
    elif m == "stock":
        target_row = f"{target}: on hand {f['onhand']}, reserved {f['reserved']}"
        near_row = f"{near}: on hand {f['onhand'] + 9}, reserved 2"
        other = next(item for item in f["sku_order"] if item not in (target, near))
        other_row = f"{other}: on hand {f['onhand'] + 4}, reserved 1"
        rows = {target: target_row, near: near_row, other: other_row}
        listing = "; ".join(rows[item] for item in f["sku_order"])
        if z:
            listing = listing.replace("on hand", "现货").replace("reserved", "已预留")
            core = [
                f"领料申请：案号 {case}，目标货号 {target}，需求 {f['demand']} 件。",
                f"库存与预留 [{case}]：{listing}。",
                "到货口径：仅承运方已签收确认的件数计入保证量，待确认件数只用于乐观估计。",
                f"邻项申请：货号 {near} 另有独立需求，不与 {target} 合并。",
            ]
            critical = f"最新签署承运回执（2026-07-01）[{case}] {target}：已确认 {f['confirmed']} 件；待确认 {f['pending']} 件。"
            near_note = f"最新签署承运回执（2026-07-01）[{case}] {near}：已确认 9 件；待确认 3 件。"
            stale = f"早期承运预报（2026-06-30）[{case}] {target}：预计到货 12 件，尚无签收。"
        else:
            core = [
                f"Requisition: case {case}, requested SKU {target}, demand {f['demand']} units.",
                f"Inventory and reservations [{case}]: {listing}.",
                "Inbound accounting: only carrier-signed arrivals count as confirmed; pending arrivals are used for an optimistic bound only.",
                f"Adjacent request: {near} has a separate demand and cannot be pooled with {target}.",
            ]
            critical = f"Latest signed carrier receipt (2026-07-01) [{case}] {target}: confirmed {f['confirmed']} units; pending {f['pending']} units."
            near_note = f"Latest signed carrier receipt (2026-07-01) [{case}] {near}: confirmed 9 units; pending 3 units."
            stale = f"Earlier carrier forecast (2026-06-30) [{case}] {target}: 12 units expected, with no signed receipt."
    elif m == "policy":
        region, exception_region = f["region"], f["exception_region"]
        if z:
            core = [
                f"申请记录：案号 {case}，目标 {target}，地区 {region}，申请日期 {f['request_date']}。",
                "基本规则：许可证须在有效期内；没有适用豁免时，现场检查也须完成。",
                f"地域例外：自 {f['exception_date']} 起，{exception_region} 的申请免现场检查；许可证要求仍有效。",
                f"培训修订：自 {f['amendment_date']} 起，新申请还须提交培训证明。",
            ]
            permit_zh = {"valid": "有效", "expired": "过期"}[f["permit"]]
            inspection_zh = {
                "complete": "已完成",
                "pending": "待核验",
                "waived": "已豁免",
            }[f["inspection"]]
            training_zh = {"filed": "已归档", "pending": "待核验"}[f["training"]]
            critical = f"最新签署资格核查（2026-07-01）[{case}] {target}：许可证 {permit_zh}；现场检查 {inspection_zh}；培训证明 {training_zh}。"
            near_note = f"最新签署资格核查（2026-07-01）[{case}] {near}：许可证 有效；现场检查 已完成；培训证明 已归档。"
            stale = f"早期资格草表（2026-06-30）[{case}] {target}：尚未核对许可证及其他材料。"
        else:
            core = [
                f"Application: case {case}, target {target}, region {region}, filed {f['request_date']}.",
                "Base rule: the permit must be valid; a site inspection is required unless an applicable exception waives it.",
                f"Regional exception: from {f['exception_date']}, applications in {exception_region} waive inspection but still need a valid permit.",
                f"Training amendment: applications filed from {f['amendment_date']} also need training proof.",
            ]
            critical = f"Latest signed eligibility check (2026-07-01) [{case}] {target}: permit {f['permit']}; inspection {f['inspection']}; training {f['training']}."
            near_note = f"Latest signed eligibility check (2026-07-01) [{case}] {near}: permit valid; inspection complete; training filed."
            stale = f"Earlier eligibility draft (2026-06-30) [{case}] {target}: permit and supporting files had not yet been checked."
    elif m == "evidence":
        finding_en = {
            "contrary": "the measured seal differs from the intake image",
            "unobserved": "the scheduled comparison was not performed",
            "affirmed": "the measured seal matches the intake image",
        }[f["finding"]]
        finding_zh = {
            "contrary": "实测封签与入库影像不一致",
            "unobserved": "原定的比对尚未实施",
            "affirmed": "实测封签与入库影像一致",
        }[f["finding"]]
        if z:
            core = [
                f"核实请求：案号 {case}，目标 {target}；主张为封签与入库影像一致。",
                f"第一来源（入库照片，{f['source_a']}）[{case}] {target}：照片清楚显示登记时的封签。",
                "核实要求：还须有独立的后续实测；未实施检测不同于观察到相反结果。",
                f"相邻编号提示：{near} 的照片和检测另行归档。",
            ]
            critical = f"最新签署实测记录（2026-07-01）[{case}] {target}，文件 {f['source_b']}：{finding_zh}。"
            near_note = f"最新签署实测记录（2026-07-01）[{case}] {near}：实测封签与该项入库影像一致。"
            stale = f"早期检测安排（2026-06-30）[{case}] {target}：列明比对计划，但没有实测结论。"
        else:
            core = [
                f"Verification request: case {case}, target {target}; claim: the seal matches its intake image.",
                f"First source, intake photograph {f['source_a']} [{case}] {target}: the image clearly shows the seal at registration.",
                "Verification rule: an independent later measurement is also required; an unperformed comparison is different from an observed mismatch.",
                f"Nearby record: {near} has its own photograph and measurement file.",
            ]
            critical = f"Latest signed measurement (2026-07-01) [{case}] {target}, file {f['source_b']}: {finding_en}."
            near_note = f"Latest signed measurement (2026-07-01) [{case}] {near}: the measured seal matches that item intake image."
            stale = f"Earlier test plan (2026-06-30) [{case}] {target}: comparison scheduled, with no measurement result."
    elif m == "service_level":
        if z:
            core = [
                f"服务请求：案号 {case}，目标 {target}，请求日期 {f['request_date']}。",
                f"合同期限：请求次日起 {f['days']} 个工作日完成；周末不计，到期当天有效。",
                f"本地日历：{f['holiday']} 为假日，不计入工作日。",
                f"早期派工：{target} 曾安排临时预约，实际交付以签署的服务记录为准。",
            ]
            prerequisite_zh = {"signed": "已签署", "pending": "待确认"}[
                f["prerequisite"]
            ]
            critical = f"最新签署服务记录（2026-07-01）[{case}] {target}：交付日期 {f['service_date']}；前置条件 {prerequisite_zh}。"
            near_note = f"最新签署服务记录（2026-07-01）[{case}] {near}：交付日期 {f['due']}；前置条件 已签署。"
            stale = f"早期派工预报（2026-06-30）[{case}] {target}：计划在 {f['due']} 上门，未记实际交付。"
        else:
            core = [
                f"Service request: case {case}, target {target}, request date {f['request_date']}.",
                f"Contract term: complete within {f['days']} business days after the request, excluding weekends; completion on the due date counts.",
                f"Local calendar: {f['holiday']} is a holiday and is not a business day.",
                f"Provisional dispatch: {target} had an appointment; the signed service record controls actual completion.",
            ]
            critical = f"Latest signed service record (2026-07-01) [{case}] {target}: completion {f['service_date']}; prerequisite {f['prerequisite']}."
            near_note = f"Latest signed service record (2026-07-01) [{case}] {near}: completion {f['due']}; prerequisite signed."
            stale = f"Earlier dispatch forecast (2026-06-30) [{case}] {target}: visit planned for {f['due']}, with no completion recorded."
    else:
        raise ValueError(m)
    return core, critical, near_note, stale


def render(f: dict[str, Any]) -> str:
    core, critical, near_note, stale = _sections(f)
    ordered = [core[i] for i in f["core_order"]]
    z = f["language"] == "zh"
    title = (
        f"{f['setting']} · 案卷 {f['case']}"
        if z
        else f"{f['setting']} · case file {f['case']}"
    )
    if not f["long"]:
        ordered[-1] = ordered[-1] + "\n" + critical
        return (
            title + "\n\n" + "\n\n".join(f"[{i}] {s}" for i, s in enumerate(ordered, 1))
        )
    rng = random.Random(f["dossier_seed"])
    note_count = 23 if f["very_long"] else 12
    topics = TOPICS_ZH[f["mechanism"]] if z else TOPICS_EN[f["mechanism"]]
    others = [near_note, stale]
    for i in range(note_count - 3):
        topic = topics[(i + f["style"]) % len(topics)]
        note = topic.format(
            case=f["case"], target=f["target"], near=f["near"], setting=f["setting"]
        )
        if i >= len(topics):
            suffix = f"归档批次 {i + 1}。" if z else f"Filed in batch {i + 1}."
            note += " " + suffix
        others.append(note)
    rng.shuffle(others)
    others.insert(f["deep_position"], critical)
    if len(others) != note_count:
        raise AssertionError("Dossier length changed")
    header = "随案补充材料" if z else "Filed case dossier"
    dossier = "\n\n".join(f"[{i + 5}] {note}" for i, note in enumerate(others))
    return (
        title
        + "\n\n"
        + "\n\n".join(f"[{i}] {s}" for i, s in enumerate(ordered, 1))
        + "\n\n"
        + header
        + "\n\n"
        + dossier
    )


def _one(pattern: str, text: str) -> tuple[str, ...]:
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one controlling source for pattern {pattern!r}; found {len(matches)}"
        )
    value = matches[0]
    return (value,) if isinstance(value, str) else value


def rendered_oracle(state: str, mechanism: str, case: str, target: str) -> int:
    """Compute the score solely from parsed rendered records, never metadata."""
    c, t = re.escape(case), re.escape(target)
    zh = "· 案卷 " in state.splitlines()[0]
    if mechanism == "workflow":
        chain = _one(
            r"任务依赖链：([^。]+)" if zh else r"Dependency chain: ([^.]+)", state
        )[0]
        tasks = set(re.findall(r"TASK-[A-Z]{4}\d{3}", chain))
        pattern = (
            rf"最新签署核对（2026-07-01）\[{c}\] {t}：([^\n]+)"
            if zh
            else rf"Latest signed reconciliation \(2026-07-01\) \[{c}\] {t}: ([^\n]+)"
        )
        raw = _one(pattern, state)[0].rstrip(".。")
        statuses = dict(
            re.findall(
                r"(TASK-[A-Z]{4}\d{3})=(complete|failed|queued|完成|失败|待办)", raw
            )
        )
        if not tasks or set(statuses) != tasks:
            raise ValueError("Signed workflow statuses do not cover the named chain")
        values = set(statuses.values())
        return (
            0
            if values & {"failed", "失败"}
            else 1 if values & {"queued", "待办"} else 2
        )
    if mechanism == "connection":
        arrival = _one(
            (
                rf"抵达预报 \[{c}\] IN-[A-Z]{{4}}\d{{3}}：(\d\d:\d\d) 至 (\d\d:\d\d)"
                if zh
                else rf"Arrival forecast \[{c}\] IN-[A-Z]{{4}}\d{{3}}: (\d\d:\d\d) to (\d\d:\d\d)"
            ),
            state,
        )
        walk = int(
            _one(
                (
                    rf"站内步行 \[{c}\] IN-[A-Z]{{4}}\d{{3}} 至 {t}：(\d+) 分钟"
                    if zh
                    else rf"Interchange walk \[{c}\] IN-[A-Z]{{4}}\d{{3}} to {t}: (\d+) minutes"
                ),
                state,
            )[0]
        )
        cutoff = _minute(
            _one(
                (
                    rf"最新签署检票通知（2026-07-01）\[{c}\] {t}：检票截止 (\d\d:\d\d)"
                    if zh
                    else rf"Latest signed boarding notice \(2026-07-01\) \[{c}\] {t}: cutoff (\d\d:\d\d)"
                ),
                state,
            )[0]
        )
        first, last = _minute(arrival[0]) + walk, _minute(arrival[1]) + walk
        return 0 if first > cutoff else 1 if last > cutoff else 2
    if mechanism == "stock":
        demand = int(
            _one(
                (
                    rf"领料申请：案号 {c}，目标货号 {t}，需求 (\d+) 件"
                    if zh
                    else rf"Requisition: case {c}, requested SKU {t}, demand (\d+) units"
                ),
                state,
            )[0]
        )
        onhand, reserved = map(
            int,
            _one(
                (
                    rf"{t}: 现货 (\d+), 已预留 (\d+)"
                    if zh
                    else rf"{t}: on hand (\d+), reserved (\d+)"
                ),
                state,
            ),
        )
        confirmed, pending = map(
            int,
            _one(
                (
                    rf"最新签署承运回执（2026-07-01）\[{c}\] {t}：已确认 (\d+) 件；待确认 (\d+) 件"
                    if zh
                    else rf"Latest signed carrier receipt \(2026-07-01\) \[{c}\] {t}: confirmed (\d+) units; pending (\d+) units"
                ),
                state,
            ),
        )
        usable = onhand - reserved + confirmed
        return 0 if usable + pending < demand else 1 if usable < demand else 2
    if mechanism == "policy":
        if zh:
            region, date = _one(
                rf"申请记录：案号 {c}，目标 {t}，地区 (北区|南区)，申请日期 (\d{{4}}-\d\d-\d\d)",
                state,
            )
            exception_date, exception_region = _one(
                r"地域例外：自 (\d{4}-\d\d-\d\d) 起，(北区|南区) 的申请免现场检查",
                state,
            )
            amendment_date = _one(r"培训修订：自 (\d{4}-\d\d-\d\d) 起", state)[0]
            permit, inspection, training = _one(
                rf"最新签署资格核查（2026-07-01）\[{c}\] {t}：许可证 (有效|过期)；现场检查 (已完成|待核验|已豁免)；培训证明 (已归档|待核验)",
                state,
            )
            permit = {"有效": "valid", "过期": "expired"}[permit]
            inspection = {
                "已完成": "complete",
                "待核验": "pending",
                "已豁免": "waived",
            }[inspection]
            training = {"已归档": "filed", "待核验": "pending"}[training]
        else:
            region, date = _one(
                rf"Application: case {c}, target {t}, region (North|South), filed (\d{{4}}-\d\d-\d\d)",
                state,
            )
            exception_date, exception_region = _one(
                r"Regional exception: from (\d{4}-\d\d-\d\d), applications in (North|South) waive inspection",
                state,
            )
            amendment_date = _one(
                r"Training amendment: applications filed from (\d{4}-\d\d-\d\d)", state
            )[0]
            permit, inspection, training = _one(
                rf"Latest signed eligibility check \(2026-07-01\) \[{c}\] {t}: permit (valid|expired); inspection (complete|pending|waived); training (filed|pending)",
                state,
            )
        scoped = region == exception_region and date >= exception_date
        training_required = date >= amendment_date
        if (scoped and inspection != "waived") or (
            not scoped and inspection == "waived"
        ):
            raise ValueError("Rendered inspection conflicts with dated regional scope")
        if permit != "valid":
            return 0
        if (not scoped and inspection != "complete") or (
            training_required and training != "filed"
        ):
            return 1
        return 2
    if mechanism == "evidence":
        if zh:
            _one(
                rf"第一来源（入库照片，DOC-[A-Z]{{4}}\d{{3}}）\[{c}\] {t}：照片清楚显示登记时的封签",
                state,
            )
            phrase = _one(
                rf"最新签署实测记录（2026-07-01）\[{c}\] {t}，文件 DOC-[A-Z]{{4}}\d{{3}}：(实测封签与入库影像不一致|原定的比对尚未实施|实测封签与入库影像一致)",
                state,
            )[0]
            return {
                "实测封签与入库影像不一致": 0,
                "原定的比对尚未实施": 1,
                "实测封签与入库影像一致": 2,
            }[phrase]
        _one(
            rf"First source, intake photograph DOC-[A-Z]{{4}}\d{{3}} \[{c}\] {t}: the image clearly shows the seal at registration",
            state,
        )
        phrase = _one(
            rf"Latest signed measurement \(2026-07-01\) \[{c}\] {t}, file DOC-[A-Z]{{4}}\d{{3}}: (the measured seal differs from the intake image|the scheduled comparison was not performed|the measured seal matches the intake image)",
            state,
        )[0]
        return {
            "the measured seal differs from the intake image": 0,
            "the scheduled comparison was not performed": 1,
            "the measured seal matches the intake image": 2,
        }[phrase]
    if mechanism == "service_level":
        if zh:
            request = _one(
                rf"服务请求：案号 {c}，目标 {t}，请求日期 (\d{{4}}-\d\d-\d\d)", state
            )[0]
            days = int(_one(r"合同期限：请求次日起 (\d+) 个工作日完成", state)[0])
            holiday = _one(r"本地日历：(\d{4}-\d\d-\d\d) 为假日", state)[0]
            service, prerequisite = _one(
                rf"最新签署服务记录（2026-07-01）\[{c}\] {t}：交付日期 (\d{{4}}-\d\d-\d\d)；前置条件 (已签署|待确认)",
                state,
            )
            prerequisite = {"已签署": "signed", "待确认": "pending"}[prerequisite]
        else:
            request = _one(
                rf"Service request: case {c}, target {t}, request date (\d{{4}}-\d\d-\d\d)",
                state,
            )[0]
            days = int(
                _one(
                    r"Contract term: complete within (\d+) business days after the request",
                    state,
                )[0]
            )
            holiday = _one(r"Local calendar: (\d{4}-\d\d-\d\d) is a holiday", state)[0]
            service, prerequisite = _one(
                rf"Latest signed service record \(2026-07-01\) \[{c}\] {t}: completion (\d{{4}}-\d\d-\d\d); prerequisite (signed|pending)",
                state,
            )
        due = _business_due(
            dt.date.fromisoformat(request), days, {dt.date.fromisoformat(holiday)}
        )
        return (
            0
            if dt.date.fromisoformat(service) > due
            else 1 if prerequisite != "signed" else 2
        )
    raise ValueError(mechanism)


def build(secret: bytes, role: str) -> list[dict[str, Any]]:
    if len(secret) != 32 or role not in GROUPS:
        raise ValueError("Need private 32-byte seed and train/select role")
    rows: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        for index in range(GROUPS[role]):
            rng = _rng(secret, role, mechanism, index)
            base = _base(rng, role, mechanism, index)
            group = hmac.new(
                secret, f"group\0{role}\0{mechanism}\0{index}".encode(), hashlib.sha256
            ).hexdigest()[:24]
            for level in LEVELS:
                facts = _facts(base, level)
                if oracle(facts) != level:
                    raise AssertionError(
                        f"Structured oracle mismatch: {group} level {level}"
                    )
                state = render(facts)
                if (
                    rendered_oracle(state, mechanism, base["case"], base["target"])
                    != level
                ):
                    raise AssertionError(
                        f"Rendered oracle mismatch: {group} level {level}"
                    )
                if base["long"]:
                    core = state.split(
                        (
                            "随案补充材料"
                            if base["language"] == "zh"
                            else "Filed case dossier"
                        ),
                        1,
                    )[0]
                    try:
                        rendered_oracle(core, mechanism, base["case"], base["target"])
                    except ValueError:
                        pass
                    else:
                        raise AssertionError(
                            f"Core alone reveals signed decisive source: {group}"
                        )
                lang_index = 1 if base["language"] == "zh" else 0
                row = {
                    "id": f"{group}-{level}",
                    "state": state,
                    "instructions": INSTRUCTIONS[mechanism][lang_index],
                    "options": [
                        {"key": str(i), "description": description}
                        for i, description in enumerate(OPTIONS[mechanism][lang_index])
                    ],
                    "label": level,
                    "task_type": "score",
                    "family": f"score_v85_{mechanism}",
                    "group_id": group,
                    "language": base["language"],
                    "split": role,
                    "source": VERSION,
                    "evaluation_role": role,
                    "render_template": f"v8.5-{mechanism}-{role}-{base['style']}",
                    "audit_metadata": {
                        "mechanism": mechanism,
                        "case": base["case"],
                        "target": base["target"],
                        "near": base["near"],
                        "style": base["style"],
                        "long": base["long"],
                        "deep_position": base.get("deep_position"),
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
        raise PermissionError("Seed must be mode 0600")
    secret = seed_file.read_bytes()
    if len(secret) != 32 or output_dir.exists():
        raise ValueError("Need 32-byte seed and a new private output directory")
    output_dir.mkdir(parents=True, mode=0o700)
    all_rows = []
    private_manifest: dict[str, Any] = {
        "schema_version": VERSION,
        "seed_sha256": hashlib.sha256(secret).hexdigest(),
        "roles": {},
    }
    for role in GROUPS:
        rows = build(secret, role)
        all_rows.extend(rows)
        path = output_dir / f"{role}.jsonl"
        _write_jsonl(path, rows)
        private_manifest["roles"][role] = {
            "rows": len(rows),
            "sha256": file_sha256(path),
        }
    packet, key = blind_packet(all_rows, secret)
    packet = sorted(packet, key=lambda group: group["review_group"])
    packet_path = output_dir / "blind-packet.jsonl"
    _write_jsonl(packet_path, packet)
    key_path = output_dir / "sealed-key.json"
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(key, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    private_manifest["packet_sha256"] = file_sha256(packet_path)
    private_manifest["key_sha256"] = file_sha256(key_path)
    manifest_path = output_dir / "private-manifest.json"
    fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(private_manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return private_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write(args.seed_file, args.output_dir)
    print(
        json.dumps(
            {
                "schema_version": VERSION,
                "packet_sha256": manifest["packet_sha256"],
                "rows": sum(x["rows"] for x in manifest["roles"].values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
