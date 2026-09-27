"""Prospective, CPU-only Score v8.4 source-document quality pilot.

This builds candidate rows, not admitted training data. A private seed is
required. Labeled rows/keys and gold-free blind packets remain separate.
No model, benchmark gold, or teacher output is consulted.
"""

from __future__ import annotations

import argparse
import collections
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

VERSION = "decision2-score-v8.4-multisource-pilot/1"
MECHANISMS = (
    "workflow",
    "connection",
    "stock",
    "policy",
    "evidence",
    "service_level",
)
GROUPS = {"train": 3, "select": 2}
LEVELS = (0, 1, 2)

# Each of the five case settings has a different document genre. Settings 2
# and 4 are original Chinese cases, not translated copies of English cases.
SETTINGS = {
    "workflow": (
        ("maritime archive migration", "handoff memo", "status export", "en"),
        ("community theatre lighting refit", "stage call sheet", "crew diary", "en"),
        ("山城医院设备交接", "值班交接单", "维修工单", "zh"),
        ("botanical seed bank relocation", "custody plan", "shift ledger", "en"),
        ("河口气象站启用", "巡检记录", "站务公告", "zh"),
    ),
    "connection": (
        ("island ferry interchange", "tide service note", "boarding circular", "en"),
        ("mountain rail junction", "station platform brief", "change bulletin", "en"),
        ("西江客运换乘站", "调度播报", "检票通告", "zh"),
        ("university shuttle hub", "route operations sheet", "booking update", "en"),
        ("东湾渡轮码头", "航班通知", "现场换乘指引", "zh"),
    ),
    "stock": (
        ("regional conservation store", "pick request", "cycle count", "en"),
        ("school science stockroom", "class requisition", "receiving diary", "en"),
        ("东城图书修复库", "修复领料单", "库存快照", "zh"),
        ("field laboratory depot", "expedition order", "carrier manifest", "en"),
        ("海港剧院道具仓", "演出备料单", "物流通知", "zh"),
    ),
    "policy": (
        ("county records office", "eligibility circular", "amendment notice", "en"),
        ("community health van", "service authorization", "legal addendum", "en"),
        ("南湖公共档案室", "准入章程", "修订公告", "zh"),
        ("coastal research vessel", "permit handbook", "dated rider", "en"),
        ("青川流动图书车", "巡回服务细则", "生效补充条款", "zh"),
    ),
    "evidence": (
        ("river-water sample registry", "chain-of-custody note", "assay sheet", "en"),
        ("museum loan archive", "registrar memo", "condition survey", "en"),
        ("古城文物修护中心", "入库凭证", "检测记录", "zh"),
        ("community grid audit", "field observation", "meter archive", "en"),
        ("海湾湿地监测组", "采样笔记", "实验室回执", "zh"),
    ),
    "service_level": (
        ("regional repair desk", "service agreement", "dispatch diary", "en"),
        (
            "university instrument office",
            "maintenance schedule",
            "campus calendar",
            "en",
        ),
        ("东岭医疗器械服务台", "服务合同", "节假日表", "zh"),
        ("coastal communications team", "response charter", "technician log", "en"),
        ("石桥公共设施值守处", "时限约定", "排班日历", "zh"),
    ),
}

OPTIONS = {
    "workflow": (("Blocked", "Pending", "Ready"), ("受阻", "待完成", "已就绪")),
    "connection": (
        ("Impossible", "Uncertain", "Reliable"),
        ("无法接驳", "接驳不确定", "可以接驳"),
    ),
    "stock": (
        ("Insufficient", "Tentative", "Confirmed"),
        ("无法配齐", "等待未确认补货", "已确认可配齐"),
    ),
    "policy": (
        ("Ineligible", "Evidence pending", "Eligible"),
        ("不符合", "待补证", "符合"),
    ),
    "evidence": (
        ("Contradicted", "Undetermined", "Supported"),
        ("证据反驳", "证据不足", "证据支持"),
    ),
    "service_level": (
        ("Misses deadline", "Conditional", "Assured"),
        ("超过时限", "有条件", "确定达标"),
    ),
}


def _rng(secret: bytes, role: str, mechanism: str, index: int) -> random.Random:
    msg = f"{VERSION}\0{role}\0{mechanism}\0{index}".encode()
    return random.Random(
        int.from_bytes(hmac.new(secret, msg, hashlib.sha256).digest()[:16], "big")
    )


def _tag(rng: random.Random, prefix: str) -> str:
    return (
        prefix
        + "-"
        + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=4))
        + str(rng.randrange(100, 999))
    )


def _time(minute: int) -> str:
    if not 0 <= minute < 1440:
        raise ValueError("time crossed date boundary")
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _minutes(value: str) -> int:
    hh, mm = value.split(":")
    return int(hh) * 60 + int(mm)


def _business_due(start: dt.date, days: int, holidays: set[dt.date]) -> dt.date:
    date = start
    remaining = days
    while remaining:
        date += dt.timedelta(days=1)
        if date.weekday() < 5 and date not in holidays:
            remaining -= 1
    return date


def _base(rng: random.Random, role: str, mechanism: str, index: int) -> dict[str, Any]:
    style = index if role == "train" else index + 3
    setting, genre1, genre2, language = SETTINGS[mechanism][style]
    case = _tag(rng, "CASE")
    base: dict[str, Any] = {
        "case": case,
        "role": role,
        "mechanism": mechanism,
        "style": style,
        "setting": setting,
        "genre1": genre1,
        "genre2": genre2,
        "language": language,
        "target": _tag(rng, "ITEM"),
        "near": _tag(rng, "ITEM"),
        "long": style in (0, 1, 2, 4),
        "very_long": style == 0 and mechanism in ("workflow", "policy"),
        "doc_order": tuple(rng.sample(range(4), 4)),
    }
    if base["target"] == base["near"]:
        raise AssertionError("ID collision")
    if mechanism == "workflow":
        base.update(a=_tag(rng, "TASK"), b=_tag(rng, "TASK"), c=_tag(rng, "TASK"))
    elif mechanism == "connection":
        depart = rng.randrange(10 * 60, 18 * 60)
        walk = rng.randrange(5, 13)
        base.update(
            incoming=_tag(rng, "IN"),
            onward=_tag(rng, "OUT"),
            near_service=_tag(rng, "OUT"),
            depart=depart,
            walk=walk,
            original_cutoff=depart - 8,
            cutoff=depart - 6,
        )
        base["target"], base["near"] = base["incoming"], base["near_service"]
    elif mechanism == "stock":
        qty = rng.randrange(22, 48)
        reserved = rng.randrange(2, 8)
        confirmed = rng.randrange(2, 7)
        base.update(
            qty=qty,
            reserved=reserved,
            confirmed=confirmed,
            near_qty=qty + 3,
            near_stock=qty + 10,
        )
    elif mechanism == "policy":
        start = dt.date(2026, 4, rng.randrange(3, 12))
        base.update(
            region="North" if language == "en" else "北区",
            near_region="South" if language == "en" else "南区",
            request_date=(start + dt.timedelta(days=12)).isoformat(),
            amendment_date=(start + dt.timedelta(days=5)).isoformat(),
            exception_date=(start + dt.timedelta(days=3)).isoformat(),
        )
    elif mechanism == "evidence":
        base.update(
            source_a=_tag(rng, "DOC"),
            source_b=_tag(rng, "DOC"),
            source_near=_tag(rng, "DOC"),
        )
    elif mechanism == "service_level":
        # Monday request; a local holiday on Wednesday must be counted.
        start = dt.date(2026, 5, 4) + dt.timedelta(days=7 * style)
        holiday = start + dt.timedelta(days=2)
        due = _business_due(start, 3, {holiday})
        base.update(
            request_date=start.isoformat(),
            holiday=holiday.isoformat(),
            due=due.isoformat(),
            days=3,
        )
    else:
        raise ValueError(mechanism)
    return base


def _facts(base: dict[str, Any], level: int) -> dict[str, Any]:
    f = dict(base)
    mechanism = f["mechanism"]
    if mechanism == "workflow":
        # Reconciliation supersedes the earlier snapshot. No child is marked
        # complete while its prerequisite is unresolved.
        status = (
            ("failed", "queued"),
            ("complete", "queued"),
            ("complete", "complete"),
        )[level]
        f["a_status"], f["b_status"] = status
    elif mechanism == "connection":
        threshold = f["cutoff"] - f["walk"]
        if level == 0:
            f["earliest"], f["latest"] = threshold + 2, threshold + 5
        elif level == 1:
            f["earliest"], f["latest"] = threshold - 4, threshold + 5
        else:
            # The first case exercises equality at cutoff, rather than a
            # strict inequality shortcut.
            f["earliest"], f["latest"] = threshold - 4, (
                threshold if f["style"] == 0 else threshold - 1
            )
    elif mechanism == "stock":
        base_usable = f["qty"] + f["reserved"] - f["confirmed"] - 5
        f["onhand"] = base_usable if level < 2 else base_usable + 7
        f["tentative"] = 2 if level == 0 else 7
    elif mechanism == "policy":
        f["license"] = "expired" if level == 0 else "valid"
        f["training"] = "pending" if level == 1 else "verified"
        f["inspection"] = "missing"  # waived for the applicable region/date
    elif mechanism == "evidence":
        f["provenance"] = "contradicted" if level == 0 else "confirmed"
        f["stability"] = "not recorded" if level < 2 else "confirmed"
        f["near_stability"] = "confirmed"
    elif mechanism == "service_level":
        due = dt.date.fromisoformat(f["due"])
        f["service_date"] = (
            due + dt.timedelta(days=1) if level == 0 else due
        ).isoformat()
        f["prerequisite"] = "unconfirmed" if level == 1 else "confirmed"
    else:
        raise ValueError(mechanism)
    return f


def oracle(f: dict[str, Any]) -> int:
    mechanism = f["mechanism"]
    if mechanism == "workflow":
        if f["a_status"] == "failed" or f["b_status"] == "failed":
            return 0
        if f["a_status"] != "complete" or f["b_status"] != "complete":
            return 1
        return 2
    if mechanism == "connection":
        first, last = f["earliest"] + f["walk"], f["latest"] + f["walk"]
        return 0 if first > f["cutoff"] else 1 if last > f["cutoff"] else 2
    if mechanism == "stock":
        usable = f["onhand"] - f["reserved"] + f["confirmed"]
        return (
            0 if usable + f["tentative"] < f["qty"] else 1 if usable < f["qty"] else 2
        )
    if mechanism == "policy":
        current = f["request_date"] >= f["amendment_date"]
        scoped = (
            f["region"] in {"North", "北区"}
            and f["request_date"] >= f["exception_date"]
        )
        if f["license"] != "valid":
            return 0
        if current and f["training"] != "verified":
            return 1
        return 2 if scoped or f["inspection"] == "verified" else 1
    if mechanism == "evidence":
        if f["provenance"] == "contradicted" or f["stability"] == "contradicted":
            return 0
        return (
            2 if f["provenance"] == "confirmed" and f["stability"] == "confirmed" else 1
        )
    if mechanism == "service_level":
        due = _business_due(
            dt.date.fromisoformat(f["request_date"]),
            f["days"],
            {dt.date.fromisoformat(f["holiday"])},
        )
        service = dt.date.fromisoformat(f["service_date"])
        return 0 if service > due else 1 if f["prerequisite"] != "confirmed" else 2
    raise ValueError(mechanism)


HISTORY_EN = (
    "The morning shift logged a handover with the facilities team, compared the room access register against its previous edition, and left the same named custodian responsible for any late correction. The archived copy remains useful for chronology but is not the current confirmation.",
    "A routine inspection found no damage to the filing cabinet or terminal. Staff checked that the envelope numbers on the desk copy matched the bound volume, then placed the desk copy back into controlled storage before the evening opening.",
    "The operations diary records a call about staffing for the following week. The caller asked that any change be entered into the ordinary schedule, because verbal remarks made during a shift are easily mistaken for an approved update.",
    "A neighboring team moved its records to a different shelf after the quarterly inventory. The move did not alter the named request, but the proximity of the old shelf label has confused staff who relied on a photograph rather than the current register.",
    "The service counter noted a minor delay in scanning attachments during the previous cycle. The paper originals were legible, and the records clerk logged the scan delay separately from the decision fields on the active request.",
    "The night coordinator reconciled older messages in chronological order, retaining both the signed original and the later response. The coordinator recorded the author, date, and affected record so that an earlier status would not silently replace a later one.",
    "A maintenance crew tested the local printer, label stock, and backup power before the office opened. Their checklist concerns document availability; it does not itself certify the outcome of a request entered later that day.",
    "During the monthly review, two staff members checked the identifier used on envelopes and on the online register. They corrected a transposed digit in a historical example and retained the correction note for future audit.",
    "The public desk keeps a separate notebook for telephone inquiries. It includes approximate times and informal names, so the case file relies on the attributed documents rather than that notebook when confirming a technical fact.",
    "The team discussed how an unexpected absence would be covered at the next rotation. The staffing plan names the replacement desk and escalation channel but leaves the individual case decision to the documented evidence.",
    "A previous delivery used a similar reference prefix. Its scanned cover sheet was moved into the archive after the receiving clerk verified the complete identifier, avoiding an accidental merge of two active folders.",
    "A supervisor asked staff to preserve the source documents in their issued order and to record later corrections explicitly. The process reduces ambiguity when the same topic appears in a plan, a status board, and a dated bulletin.",
    "The quarter-end report contains aggregate counts from several unrelated desks. Those counts describe workload and staffing pressure; they are not substitutes for the item-level values needed for an individual decision.",
    "A trainee annotated the margin of a practice printout, and the trainer marked it as educational material. The actual case packet retains its own date, record number, and source attribution so the practice annotations cannot be read as evidence.",
    "The records team confirmed that the paper and electronic copies were synchronized before close of business. Any later event is expected to appear in a dated supplement rather than silently modifying an archived page.",
    "A small change to the building entrance altered where couriers leave sealed packets. Reception documented the route in the general diary, while the case-specific receiving sheet still carries the relevant handoff time and identifier.",
    "An unrelated inquiry from a nearby organization requested the same style of form. Staff assigned it a distinct reference and kept the response in a separate folder, even though the general process description was shared.",
    "The audit team sampled a prior week's records for legibility and source signatures. This check addressed the reliability of the filing process, not whether the current named request meets the substantive criteria.",
    "Before the next shift, the desk clerk listed unanswered administrative questions and their responsible contacts. The list helps locate records but cannot establish the status of a technical prerequisite without the relevant source document.",
    "The workspace changed its internal folder naming convention after an archive migration. Historical labels remain visible in correspondence, whereas the complete case identifier is the stable reference for joining current documents.",
    "The procurement office retained an older route diagram because staff still use its room names in conversation. The revised floor plan has different entrance labels, and the desk records both names when explaining where a source document was physically received.",
    "At a public meeting, several attendees asked about the processing calendar in general terms. The minutes summarize these questions but omit item identifiers, so they are background for policy communication rather than a replacement for a dated case record.",
    "An internal training session compared two historical situations with superficially similar dates. In the training copy, the instructor emphasized how a later bulletin can change an operational deadline without changing an earlier request or the identifier of the service involved.",
    "The quality officer checked the binding of a thick appendix after a page was found loose. The officer restored the page in its numbered position, wrote a short custody note, and left the substantive figures in the signed source sheet unchanged.",
    "The regional office kept a paper log of returned envelopes because the electronic receipt feed had intermittent outages. Each envelope was marked with its physical arrival date, and the desk later reconciled that date with the system import timestamp.",
    "A contractor asked whether the ordinary storage rules applied to a temporary work area. The supervisor answered in a general bulletin, then directed individual teams to their own signed permits for any exception affecting a particular activity or location.",
    "The field coordinator photographed a faded wall chart before it was replaced. The photograph helps explain older room references in correspondence, while the current service folder gives the controlling identifiers and document dates for active cases.",
    "Staff reviewed a list of routine service interruptions from the previous season. The list helped plan coverage for lunch periods and holidays, but the current deadline still depends on the request date and the applicable local calendar.",
    "An archivist noticed a loose attachment in a historical bundle and restored it to its numbered sequence. The attachment described a different unit, so the archivist recorded its origin rather than folding its measurements into the active case.",
    "The purchasing team opened a separate communication channel for freight questions after several suppliers used the same subject line. They now require a complete purchase reference on every carrier response before relating an inbound quantity to a requested item.",
    "A senior technician checked whether a trial instrument could share spare parts with the production system. The trial used a different connector and was documented in a training file rather than in the service record for the named production unit.",
    "The compliance clerk annotated a superseded paragraph with the date on which an amendment began. The annotation preserves historical context and points readers to the signed later notice when they assess an application made after that date.",
    "A routine building inspection reported that the alarm panel and fire doors remained accessible. That report concerns safe working conditions; it does not establish the individual inventory, document evidence, or service timing for a separate request.",
    "Two operators compared handwritten labels with the equipment registry after a recent move. Where the abbreviated names differed, they wrote the full identifiers on a reconciliation sheet and kept both originals rather than erasing either source.",
    "The training coordinator summarized a case from the prior quarter in a workshop handout. The summary omitted some timings to keep the exercise short, so it is useful for staff orientation but cannot settle a current time-window calculation.",
    "A volunteer recorded general visitor traffic while the public counter was busy. Those counts assist shift planning and explain why a packet was filed late; they do not supply the missing technical observation for a named evidence bundle.",
    "The internal risk register lists weather, staff absence, and transport disruption as possible operational hazards. Each hazard is tracked separately from the dated bulletin that confirms whether a particular booked connection was actually changed.",
    "An IT migration preserved links to old case numbers in the search index. When staff open one of those links, a banner reminds them to compare the complete identifier with the printed dossier before using figures from a scanned page.",
    "The desk signed off on a backup telephone rota for weekends and public holidays. The rota identifies who can answer questions, but it does not move a contract deadline unless the agreement or an authorized change notice says so.",
    "The inventory supervisor documented how partly opened cartons are counted at the end of a shift. The procedure explains which units are usable and how reservations are recorded, leaving the quantity for each named SKU to its own stock sheet.",
    "A sample courier delivered containers to a shared reception area used by several laboratories. Reception assigned separate intake references so laboratory results would remain tied to the correct sample even if the containers arrived together.",
    "The stage crew revised its call-time board after a rehearsal overran. The revised board covers staff attendance, while the technical handoff checklist retains distinct dependencies for wiring, safety inspection, and operator acknowledgement.",
    "A policy office distributed an FAQ after applicants confused a general rule with a regional exception. The FAQ explains where to find the signed exception; it does not itself broaden the exception to a different region or earlier application date.",
    "The facilities manager scheduled cleaning near the document room and warned that access might be delayed. The notice does not change signed facts in the dossier; staff placed a copy inside the general operations folder for historical context.",
    "During quality sampling, an analyst found that one old scan omitted the reverse side of a form. The original was rescanned, and both timestamps were retained so a later reviewer could distinguish the document date from the scan date.",
    "An accounts clerk matched monthly supplier statements to the archive but did not approve any shipment quantities. Shipment confirmations were left with the transport team, where they could be checked against a specific order and its reservation ledger.",
    "A transport supervisor compared boarding notices from two adjacent services that share a platform. The supervisor kept the notices separate because identical route names can hide different cutoffs, dates, and booking conditions.",
    "The public inquiry log contains a broad description of the service promised by the office. The controlling contract schedule, local calendar, and request record are maintained as distinct documents for an item-level timeliness decision.",
)

HISTORY_ZH = (
    "早班人员检查了值班室的收件登记与纸质卷宗编号，并把需要复核的旧记录交给档案员。旧登记保留在卷内用于追溯时间顺序，真正生效的变更仍须查阅带日期和署名的本次补充记录。",
    "设备室完成例行通风和电源检查后，值守人员重新核对了借用单上的完整编号。相邻编号曾在上季度发生誊写错误，因此他们将订正单与原件一并归档，避免只凭编号前缀关联两个案卷。",
    "服务窗口接到一通询问轮班安排的电话。接线人把问题转到排班簿，但没有把口头描述写成目标事项的结论；真正涉及该事项的事实仍保存在具名文件与正式回执中。",
    "仓管员盘点了隔壁架位的旧标签，并记录了搬运过程中发现的磨损。该记录用于解释货架位置变化，不能代替目标编号的实际数量、预约占用或后来收到的承运确认。",
    "午后扫描仪出现短暂延迟，纸质凭证仍清晰可读。档案室把扫描延迟列为技术事件，并要求工作人员按凭证形成时间核查内容，不把文件上传的先后顺序当成业务生效顺序。",
    "组长在例会中复核了前一周期的签收流程，确认了责任人和备用联络渠道。他强调后续更正必须注明适用记录与日期，以免旧状态栏与新的调度说明同时被误认为现行事实。",
    "总务处整理了旧文件柜的索引，把若干已结束项目移至历史区。索引变更只影响检索路径；每一项正在处理的申请仍以其完整编号、适用条款及对应证据确定结果。",
    "夜间值守人员在交班表里注明门禁位置和来访人员登记方式。这些行政信息有助于寻找原件，却不能证明样品、服务请求或审批事项已经满足具体的技术条件。",
    "一名新员工在练习用表格上做了批注，导师随后将练习材料与正式案卷分开存放。正式文件上的编号、时间和来源说明需要完整核对，练习用的示例数字不能并入当前事项。",
    "邻近单位曾借用相同格式的申请表，窗口给它单独分配了编号。虽然办理步骤近似，两个单位的服务日期、附件状态和审批范围不同，汇总报表也不替代单项记录。",
    "质控人员抽查了上周的卷宗，重点检查签字页是否齐全和页面是否可读。抽查说明存档过程正常，但并不构成本次目标事项符合规则的独立证明。",
    "现场人员清点了备用封套、打印纸和送件袋，并在值班日志中留下记录。封套准备情况属于后勤保障，不应与针对具体编号的承运通知或截止时间相混。",
    "季度统计提到了不同站点的工作量变化，但只给出汇总数字。对单一编号作判断时，工作人员仍须找到该编号对应的原始请求、核对单及后来形成的正式更正。",
    "临时换班造成记录录入晚于事实发生。值守主管因此要求在补录时注明实际发生日期、录入日期和签名，以免只按系统排序导致前后文书关系颠倒。",
    "档案员复查了相似名称的两处场地，确认它们的地址和责任部门各不相同。办理人员必须使用案卷上的完整名称，不可把另一场地的日历、库存或授权直接搬来使用。",
    "办公室为下周培训准备了文件目录，并提醒参训人员逐份核对来源。目录帮助确定查找顺序，但只有原始文件中的具体数字、状态和适用范围才能支持最终判断。",
    "同一业务流程曾在另一个站点试行，试行期间积累了不少经验记录。那些记录解释流程背景，却不能覆盖本案文书中明确写出的时间、对象和义务。",
    "收件员将一份历史副本与新回执一起装订，并在封面标明两者的形成日期。保存历史副本是为了追踪变化，决定当前状态时仍须按照有效日期和目标编号读取新回执。",
    "行政人员核对了档案室的备用联系电话及会签路线。电话簿有时滞后于实际值班安排，所以技术结论必须以具名记录为准，不能由一般联络信息推断。",
    "月末审计发现过去使用的缩写容易混淆两类业务。随后所有新案卷均采用完整字段名称，旧缩写只在历史附件中保留，并在索引表中附有解释。",
    "物资科整理了往年供应商的运单，发现部分邮件主题相似但货号不同。此后每份到货确认必须填写完整货号和收货日期，旧邮件仅用于解释沟通过程。",
    "站务处将临时改道信息写入当周公告，同时保留原排班表供乘客查询。核查具体接驳时，工作人员仍须看命名班次的最新截票记录与到站区间。",
    "实验室检查了多个样品盒的封签，把磨损严重的封条拍照存档。照片用于追溯盒子的流转，但不能代替具名样品的来源链记录和稳定性检测结论。",
    "值班经理收到关于楼层照明的反馈，随后安排了维护人员。维护单与设备交接的依赖链分开存放，后者的完成状态仍以正式核对页为准。",
    "法务室编写了一份常见问题说明，帮助申请人辨认基本条款与地域例外。说明书本身不扩大例外适用地区，也不改变后续修订的生效日期。",
    "修复库定期清点包装耗材，并把开封但可用的材料单列。具体领料能否完成仍须结合同一货号的库存、占用和承运回执，而非耗材总量。",
    "排班员提前公布了节假日值守联系人，避免服务请求无人接听。联系人名单不改变合同的工作日计算方法，也不能确认尚未完成的前置检查。",
    "档案室迁移时保留了旧索引号码的转接表，以便查询历史卷宗。新旧号码可能出现在同一检索页面，办理人员必须核对文件封面上的完整案号。",
    "船务处存放了上季度风浪预警和应急靠泊记录，供后续演练复盘。旧预警说明通常风险，却不能推断本次指定船次是否已经取消或改动。",
    "行政办公室将来自不同站点的统计报表分别归档。合计数字可以说明工作量，却不能作为某个站点的库存、许可证或交付时间的替代证据。",
    "质控组复核了电子文档的签名时间和上传时间，发现两者偶尔相隔一个班次。工作人员因此按业务形成时间梳理文书先后，再记录后补上传的原因。",
    "公共服务台更新了读者询问的分类目录，方便将问题转给正确部门。分类目录只提供检索线索，真正的个案结论仍依赖目标编号对应的原始资料。",
)


def _history(f: dict[str, Any]) -> str:
    """Attributed operational history, not a repeated answer-bearing line."""
    if not f["long"]:
        return ""
    bank = HISTORY_ZH if f["language"] == "zh" else HISTORY_EN
    count = 24 if f["very_long"] else 12
    if f["very_long"]:
        offset = 0 if f["mechanism"] == "workflow" else 24
        selected = list(bank[offset : offset + count])
    else:
        order_seed = hashlib.sha256((f["case"] + "\0archived-notes").encode()).digest()
        selected = random.Random(int.from_bytes(order_seed[:16], "big")).sample(
            bank, count
        )
    if f["language"] == "zh":
        return "\n".join(
            f"卷宗背景札记 {j + 1}（{f['setting']}）：{text}"
            for j, text in enumerate(selected)
        )
    return "\n".join(
        f"Historical file note {j + 1} ({f['setting']}): {text}"
        for j, text in enumerate(selected)
    )


def _docs(f: dict[str, Any]) -> list[tuple[str, str]]:
    zh = f["language"] == "zh"
    case, target, near = f["case"], f["target"], f["near"]
    mech = f["mechanism"]
    if mech == "workflow":
        old = "complete" if not zh else "完成"
        a = (
            {"failed": "失败", "queued": "待办", "complete": "完成"}[f["a_status"]]
            if zh
            else f["a_status"]
        )
        b = (
            {"held": "搁置", "queued": "待办", "complete": "完成"}[f["b_status"]]
            if zh
            else f["b_status"]
        )
        return [
            (
                ("交接请求" if zh else "Handoff request"),
                f"{'案号' if zh else 'Case'} {case}：{'目标' if zh else 'Target'} {target}。{f['setting']}。",
            ),
            (
                ("依赖说明" if zh else f["genre1"]),
                (
                    f"{'依赖链' if zh else 'Dependency chain'} [{case}]: {target} <- {f['b']} <- {f['a']}。"
                    if zh
                    else f"Dependency chain [{case}]: {target} <- {f['b']} <- {f['a']}."
                ),
            ),
            (
                ("旧状态与更正" if zh else f["genre2"]),
                (
                    f"旧看板 [{case}]（2026-05-06 09:00）：{f['a']}={old}; {f['b']}={old}。"
                    f"当前核对 [{case}]（2026-05-07 11:00）：{f['a']}={a}; {f['b']}={b}。"
                    if zh
                    else f"Prior board [{case}] (2026-05-06 09:00): {f['a']}={old}; {f['b']}={old}. "
                    f"Reconciled status [{case}] (2026-05-07 11:00): {f['a']}={a}; {f['b']}={b}."
                ),
            ),
            (
                ("相关排班" if zh else "Duty roster"),
                f"{'另项' if zh else 'Nearby item'} {near}：{f['c']} {'安排维护窗口' if zh else 'has a maintenance window next week'}。",
            ),
        ]
    if mech == "connection":
        return [
            (
                ("行程请求" if zh else "Travel request"),
                f"{'案号' if zh else 'Case'} {case}：{'到站班次' if zh else 'Incoming'} {f['incoming']}，{'衔接班次' if zh else 'onward'} {f['onward']}。",
            ),
            (
                ("到站预报" if zh else f["genre1"]),
                f"{'到站区间' if zh else 'Arrival window'} [{case}] {f['incoming']}: {_time(f['earliest'])} — {_time(f['latest'])}。",
            ),
            (
                ("站内路线" if zh else "Station route guide"),
                f"{'步行分钟' if zh else 'Transfer walk minutes'} [{case}] {f['incoming']} -> {f['onward']}: {f['walk']}。",
            ),
            (
                ("改签与截票公告" if zh else f["genre2"]),
                f"{'原截票' if zh else 'Original cutoff'} [{case}] {f['onward']}: {_time(f['original_cutoff'])}；"
                f"{'变更截票' if zh else 'Changed cutoff'} [{case}] {f['onward']}: {_time(f['cutoff'])}；"
                f"{'邻班' if zh else 'Nearby service'} {f['near_service']}: {_time(f['cutoff'] + 18)}。",
            ),
        ]
    if mech == "stock":
        return [
            (
                ("领用申请" if zh else f["genre1"]),
                f"{'案号' if zh else 'Case'} {case}：SKU {target}，{'需求' if zh else 'demand'} {f['qty']} {'件' if zh else 'units'}。",
            ),
            (
                ("盘点与占用" if zh else f["genre2"]),
                f"{'库存' if zh else 'Stock'} [{case}] {target}: {f['onhand']}；{'已预约' if zh else 'reserved'} {f['reserved']}。"
                f"{'库存' if zh else 'Stock'} [{case}] {near}: {f['near_stock']}；{'已预约' if zh else 'reserved'} 2。",
            ),
            (
                ("承运回执" if zh else "Carrier confirmation"),
                f"{'到货' if zh else 'Inbound'} [{case}] {target}: {'已确认' if zh else 'confirmed'} {f['confirmed']}；"
                f"{'未确认' if zh else 'tentative'} {f['tentative']}。"
                f"{'到货' if zh else 'Inbound'} [{case}] {near}: {'已确认' if zh else 'confirmed'} 1；"
                f"{'未确认' if zh else 'tentative'} 3。",
            ),
            (
                ("备用订单" if zh else "Adjacent request"),
                f"{'另项' if zh else 'Nearby item'} {near}：{'需求' if zh else 'demand'} {f['near_qty']} {'件' if zh else 'units'}。",
            ),
        ]
    if mech == "policy":
        license_text = (
            {"valid": "有效", "expired": "过期"}[f["license"]] if zh else f["license"]
        )
        inspection_text = (
            {"verified": "已核实", "missing": "缺失"}[f["inspection"]]
            if zh
            else f["inspection"]
        )
        training_text = (
            {"verified": "已核实", "pending": "待提交"}[f["training"]]
            if zh
            else f["training"]
        )
        return [
            (
                ("申请记录" if zh else "Application record"),
                f"{'案号' if zh else 'Case'} {case}：{'目标' if zh else 'Target'} {target}；"
                f"{'地区' if zh else 'region'} {f['region']}；{'申请日' if zh else 'request date'} {f['request_date']}。"
                f"{'同日另项' if zh else 'Same-day neighboring application'} {near}：{f['near_region']}。",
            ),
            (
                ("基本资格" if zh else f["genre1"]),
                f"{'规则' if zh else 'Base rule'} [{case}]: {'许可证有效不可豁免；现场检查须完成。' if zh else 'a valid license is non-waivable; site inspection is required.'}"
                f"{'许可证' if zh else 'License'} [{case}] {target}: {license_text}；"
                f"{'现场检查' if zh else 'inspection'} {inspection_text}。",
            ),
            (
                ("地域例外" if zh else "Scoped exception"),
                f"{'例外' if zh else 'Exception'} [{case}]: {'仅' if zh else 'only'} {f['region']}；"
                f"{'生效日' if zh else 'effective'} {f['exception_date']}；"
                f"{'豁免现场检查，不豁免有效许可证。' if zh else 'waives inspection but not the valid license.'}"
                f"{'邻近地区' if zh else 'Adjacent region'} {f['near_region']} {'沿用基本规则' if zh else 'retains the base rule'}。",
            ),
            (
                ("后续修订" if zh else f["genre2"]),
                f"{'修订' if zh else 'Amendment'} [{case}]: {'生效日' if zh else 'effective'} {f['amendment_date']}；"
                f"{'需培训凭证' if zh else 'training proof required'}。"
                f"{'培训凭证' if zh else 'Training proof'} [{case}] {target}: {training_text}。",
            ),
        ]
    if mech == "evidence":
        prov = (
            {"confirmed": "已确认", "contradicted": "反驳"}[f["provenance"]]
            if zh
            else f["provenance"]
        )
        stability = (
            {"confirmed": "已确认", "contradicted": "反驳", "not recorded": "未记录"}[
                f["stability"]
            ]
            if zh
            else f["stability"]
        )
        near_stability = (
            {"confirmed": "已确认", "contradicted": "反驳", "not recorded": "未记录"}[
                f["near_stability"]
            ]
            if zh
            else f["near_stability"]
        )
        return [
            (
                ("核实请求" if zh else "Verification request"),
                f"{'案号' if zh else 'Case'} {case}：{'目标' if zh else 'Target'} {target}。"
                f"{'结论需同时有来源链与稳定性记录，明确反证优先；缺任一项则未定。' if zh else 'The claim needs both chain-of-custody and stability support; direct contradiction defeats it, and missing material evidence leaves it undetermined.'}",
            ),
            (
                ("来源链凭证" if zh else f["genre1"]),
                f"{'来源' if zh else 'Provenance'} [{case}] {target}: {prov}；{'文件' if zh else 'document'} {f['source_a']}。",
            ),
            (
                ("稳定性检测" if zh else f["genre2"]),
                f"{'稳定性' if zh else 'Stability'} [{case}] {target}: {stability}；{'文件' if zh else 'document'} {f['source_b']}。",
            ),
            (
                ("相邻编号记录" if zh else "Nearby record"),
                f"{'稳定性' if zh else 'Stability'} [{case}] {near}: {near_stability}；{'文件' if zh else 'document'} {f['source_near']}。",
            ),
        ]
    if mech == "service_level":
        prereq = (
            {"confirmed": "已确认", "unconfirmed": "未确认"}[f["prerequisite"]]
            if zh
            else f["prerequisite"]
        )
        return [
            (
                ("服务请求" if zh else "Service request"),
                f"{'案号' if zh else 'Case'} {case}：{'目标' if zh else 'Target'} {target}；"
                f"{'请求日' if zh else 'request date'} {f['request_date']}。",
            ),
            (
                ("时限表" if zh else f["genre1"]),
                f"{'承诺' if zh else 'Schedule'} [{case}]: {f['days']} {'个工作日，不含请求当日、周末及公布的假日；到期当日交付仍有效。' if zh else 'business days after request, excluding weekends and listed holidays; delivery on the due date counts.'}",
            ),
            (
                ("服务日历" if zh else "Service calendar"),
                f"{'假日' if zh else 'Holiday'} [{case}]: {f['holiday']}。"
                f"{'常规周末为周六和周日。' if zh else 'Ordinary weekends are Saturday and Sunday.'}",
            ),
            (
                ("派工和前置条件" if zh else f["genre2"]),
                f"{'交付日' if zh else 'Service date'} [{case}] {target}: {f['service_date']}；"
                f"{'前置条件' if zh else 'prerequisite'} {prereq}。"
                f"{'邻项' if zh else 'Nearby item'} {near}: {'另列下周排期' if zh else 'listed on next week’s schedule'}。",
            ),
        ]
    raise ValueError(mech)


def render(f: dict[str, Any]) -> str:
    docs = _docs(f)
    zh = f["language"] == "zh"
    title = f"{f['setting']} · {'案卷' if zh else 'case file'} {f['case']}"
    order = f["doc_order"]
    source = "\n\n".join(
        f"[{i + 1}] {docs[index][0]}\n"
        + (
            docs[index][1]
            if zh
            else docs[index][1].translate(str.maketrans("：；，。", ":;,."))
        )
        for i, index in enumerate(order)
    )
    history = _history(f)
    if history:
        source += (
            "\n\n"
            + ("随卷归档的业务背景" if zh else "Archived operational background")
            + "\n"
            + history
        )
    return title + "\n\n" + source


def rendered_oracle(state: str, mechanism: str, case: str) -> int:
    """Parse the emitted source text, independently of the structured facts."""
    state = state.translate(str.maketrans("：；，。（）", ":;,.()"))
    esc = re.escape(case)

    def find(pattern: str) -> tuple[str, ...]:
        match = re.search(pattern, state, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing rendered evidence for {mechanism}: {pattern}")
        return match.groups()

    if mechanism == "workflow":
        target, b, a = find(
            rf"(?:Dependency chain|依赖链) \[{esc}\]: (ITEM-[A-Z]+\d+) <- (TASK-[A-Z]+\d+) <- (TASK-[A-Z]+\d+)"
        )
        line = find(
            rf"(?:Reconciled status|当前核对) \[{esc}\]\s*\([^)]*\)\s*:([^\n]+)"
        )
        values = dict(
            re.findall(
                r"(TASK-[A-Z]+\d+)=(failed|held|queued|complete|失败|搁置|待办|完成)",
                line[0],
            )
        )
        if target not in state or a not in values or b not in values:
            raise ValueError("Unresolved workflow chain")
        normalized = {
            "失败": "failed",
            "搁置": "held",
            "待办": "queued",
            "完成": "complete",
        }
        return oracle(
            {
                "mechanism": mechanism,
                "a_status": normalized.get(values[a], values[a]),
                "b_status": normalized.get(values[b], values[b]),
            }
        )
    if mechanism == "connection":
        incoming, onward = find(
            r"(?:Incoming|到站班次) (IN-[A-Z]+\d+),\s*(?:onward|衔接班次) (OUT-[A-Z]+\d+)"
        )
        early, late = find(
            rf"(?:Arrival window|到站区间) \[{esc}\] {incoming}: (\d\d:\d\d) — (\d\d:\d\d)"
        )
        (walk,) = find(
            rf"(?:Transfer walk minutes|步行分钟) \[{esc}\] {incoming} -> {onward}: (\d+)"
        )
        (cutoff,) = find(
            rf"(?:Changed cutoff|变更截票) \[{esc}\] {onward}: (\d\d:\d\d)"
        )
        return oracle(
            {
                "mechanism": mechanism,
                "earliest": _minutes(early),
                "latest": _minutes(late),
                "walk": int(walk),
                "cutoff": _minutes(cutoff),
            }
        )
    if mechanism == "stock":
        sku, qty = find(r"SKU (ITEM-[A-Z]+\d+),\s*(?:demand|需求) (\d+)")
        onhand, reserved = find(
            rf"(?:Stock|库存) \[{esc}\] {sku}: (\d+);\s*(?:reserved|已预约) (\d+)"
        )
        confirmed, tentative = find(
            rf"(?:Inbound|到货) \[{esc}\] {sku}: (?:confirmed|已确认) (\d+);\s*(?:tentative|未确认) (\d+)"
        )
        return oracle(
            {
                "mechanism": mechanism,
                "qty": int(qty),
                "onhand": int(onhand),
                "reserved": int(reserved),
                "confirmed": int(confirmed),
                "tentative": int(tentative),
            }
        )
    if mechanism == "policy":
        target, region, request_date = find(
            r"(?:Target|目标) (ITEM-[A-Z]+\d+);\s*(?:region|地区) ([^;]+);\s*(?:request date|申请日) (\d{4}-\d\d-\d\d)"
        )
        license, inspection = find(
            rf"(?:License|许可证) \[{esc}\] {target}: (valid|expired|有效|过期);\s*(?:inspection|现场检查) (verified|missing|已核实|缺失)"
        )
        exception_date = find(
            rf"(?:Exception|例外) \[{esc}\]: [^;]+;\s*(?:effective|生效日) (\d{{4}}-\d\d-\d\d)"
        )[0]
        amendment_date = find(
            rf"(?:Amendment|修订) \[{esc}\]: (?:effective|生效日) (\d{{4}}-\d\d-\d\d)"
        )[0]
        (training,) = find(
            rf"(?:Training proof|培训凭证) \[{esc}\] {target}: (verified|pending|已核实|待提交)"
        )
        return oracle(
            {
                "mechanism": mechanism,
                "region": region,
                "request_date": request_date,
                "license": {"有效": "valid", "过期": "expired"}.get(license, license),
                "inspection": {"已核实": "verified", "缺失": "missing"}.get(
                    inspection, inspection
                ),
                "exception_date": exception_date,
                "amendment_date": amendment_date,
                "training": {"已核实": "verified", "待提交": "pending"}.get(
                    training, training
                ),
            }
        )
    if mechanism == "evidence":
        (target,) = find(r"(?:Target|目标) (ITEM-[A-Z]+\d+)")
        (provenance,) = find(
            rf"(?:Provenance|来源) \[{esc}\] {target}: (confirmed|contradicted|not recorded|已确认|反驳|未记录)"
        )
        (stability,) = find(
            rf"(?:Stability|稳定性) \[{esc}\] {target}: (confirmed|contradicted|not recorded|已确认|反驳|未记录)"
        )
        normalized = {
            "已确认": "confirmed",
            "反驳": "contradicted",
            "未记录": "not recorded",
        }
        return oracle(
            {
                "mechanism": mechanism,
                "provenance": normalized.get(provenance, provenance),
                "stability": normalized.get(stability, stability),
            }
        )
    if mechanism == "service_level":
        target, request_date = find(
            r"(?:Target|目标) (ITEM-[A-Z]+\d+);\s*(?:request date|请求日) (\d{4}-\d\d-\d\d)"
        )
        (days,) = find(
            rf"(?:Schedule|承诺) \[{esc}\]: (\d+) (?:business days|个工作日)"
        )
        (holiday,) = find(rf"(?:Holiday|假日) \[{esc}\]: (\d{{4}}-\d\d-\d\d)")
        service_date, prereq = find(
            rf"(?:Service date|交付日) \[{esc}\] {target}: (\d{{4}}-\d\d-\d\d);\s*(?:prerequisite|前置条件) (confirmed|unconfirmed|已确认|未确认)"
        )
        return oracle(
            {
                "mechanism": mechanism,
                "request_date": request_date,
                "holiday": holiday,
                "days": int(days),
                "service_date": service_date,
                "prerequisite": {"已确认": "confirmed", "未确认": "unconfirmed"}.get(
                    prereq, prereq
                ),
            }
        )
    raise ValueError(mechanism)


INSTRUCTIONS = {
    "workflow": (
        "Judge the named handoff using the dependency chain and the latest dated reconciliation. A failed ancestor blocks the handoff (0); with no failure, a queued ancestor leaves it pending (1); all ancestors complete makes it ready (2). An earlier board entry does not override the reconciliation.",
        "判断指定交接事项。结合依赖链和有日期的最新核对记录：任一前置环节失败记 0；没有失败但仍有待办环节记 1；全部前置环节完成记 2。较早的看板不能覆盖后来的正式核对。",
    ),
    "connection": (
        "Evaluate the named incoming and onward services. Add the station walking time to both endpoints of the incoming arrival interval, and use the changed booking cutoff. Earliest possible transfer after cutoff means 0; an interval crossing cutoff means 1; latest transfer at or before cutoff means 2. Equality is feasible.",
        "只评估指定的到站班次和衔接班次。到站区间的两个端点均须加上站内步行分钟，并采用变更后的截票时间。最早也超过截票为 0；可能赶上也可能赶不上为 1；最迟仍不超过截票为 2。恰好等于截票时间可行。",
    ),
    "stock": (
        "For the requested SKU, confirmed usable units equal on-hand minus reservations plus confirmed inbound. Tentative inbound is not confirmed. If even confirmed plus tentative is short of demand score 0; if only tentative closes the gap score 1; if confirmed usable units suffice score 2.",
        "按申请中的 SKU 计算：可确认使用量为现货减已预约数量再加已确认到货。未确认到货不能算作保证。连同未确认数量仍不足为 0；只有未确认到货才能补齐为 1；可确认使用量已经足够为 2。",
    ),
    "policy": (
        "Apply the dated base rule, the exception only to its stated region and effective period, and the later amendment when effective. A valid license cannot be waived: without one score 0. Required training evidence still pending scores 1. If every current non-waived requirement is satisfied, score 2.",
        "按日期和地区适用基本规则、地域例外及后续修订。有效许可证属于不可豁免条件，缺少时记 0；现行要求的培训凭证仍待提交时记 1；其余当期未被豁免的要求全部满足时记 2。",
    ),
    "evidence": (
        "Assess the requested record, joining its provenance and stability sources. A direct contradiction in either necessary source scores 0; missing material evidence without contradiction scores 1; both sources affirm the claim scores 2. A similar record number is a separate case.",
        "核查指定编号的来源链和稳定性两份证据。任一必要来源直接反驳结论记 0；没有反证但缺少必要材料记 1；两份来源均确认记 2。编号相近的案卷须按其完整编号分别核对。",
    ),
    "service_level": (
        "Compute three business days after the dated request, excluding its own date, weekends, and the listed local holiday. Delivery on the deadline counts. A service date after the deadline scores 0; a timely date with an unconfirmed prerequisite scores 1; timely service with the prerequisite confirmed scores 2.",
        "从请求日期次日起计算三个工作日；请求当日、周末和所列本地假日不计。到期当日完成仍有效。交付日超过时限记 0；日期达标但前置条件未确认记 1；日期达标且前置条件已确认记 2。",
    ),
}


def build(secret: bytes, role: str) -> list[dict[str, Any]]:
    if len(secret) != 32 or role not in GROUPS:
        raise ValueError("Need one private 32-byte seed and train/select role")
    rows: list[dict[str, Any]] = []
    for mechanism in MECHANISMS:
        for index in range(GROUPS[role]):
            rng = _rng(secret, role, mechanism, index)
            base = _base(rng, role, mechanism, index)
            group = f"score84-{role}-{mechanism}-{hashlib.sha256(base['case'].encode()).hexdigest()[:14]}"
            for level in LEVELS:
                facts = _facts(base, level)
                if oracle(facts) != level:
                    raise AssertionError(
                        f"Structured oracle disagrees: {group} level {level}"
                    )
                state = render(facts)
                if rendered_oracle(state, mechanism, base["case"]) != level:
                    raise AssertionError(
                        f"Rendered oracle disagrees: {group} level {level}"
                    )
                lang_index = 1 if base["language"] == "zh" else 0
                row = {
                    "id": f"{group}_l{level}",
                    "state": state,
                    "instructions": INSTRUCTIONS[mechanism][lang_index],
                    "options": [
                        {"key": str(i), "description": description}
                        for i, description in enumerate(OPTIONS[mechanism][lang_index])
                    ],
                    "label": level,
                    "task_type": "score",
                    "family": f"score_v84_{mechanism}",
                    "group_id": group,
                    "language": base["language"],
                    "split": role,
                    "source": VERSION,
                    "evaluation_role": role,
                    "render_template": f"v8.4-{mechanism}-{role}-{base['style']}",
                    "audit_metadata": {
                        "mechanism": mechanism,
                        "case": base["case"],
                        "target": base["target"],
                        "near": base["near"],
                        "style": base["style"],
                        "document_order": list(base["doc_order"]),
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
        raise PermissionError("Seed file must be private mode 0600")
    secret = seed_file.read_bytes()
    if len(secret) != 32 or output_dir.exists():
        raise ValueError("Need a 32-byte seed and new private output directory")
    output_dir.mkdir(parents=True, mode=0o700)
    manifest: dict[str, Any] = {
        "schema_version": VERSION,
        "status": "PENDING_AUTOMATED_AUDIT_AND_INDEPENDENT_BLIND_REVIEW",
        "seed_sha256": hashlib.sha256(secret).hexdigest(),
        "roles": {},
    }
    for role in GROUPS:
        rows = build(secret, role)
        packet, key = blind_packet(rows, secret)
        row_path = output_dir / f"{role}.jsonl"
        packet_path = output_dir / f"{role}-blind-packet.jsonl"
        key_path = output_dir / f"{role}-sealed-key.json"
        _write_jsonl(row_path, rows)
        _write_jsonl(packet_path, packet)
        with key_path.open("x", encoding="utf-8") as stream:
            json.dump(key, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        key_path.chmod(0o600)
        manifest["roles"][role] = {
            "rows": len(rows),
            "groups": len(packet),
            "level_counts": dict(collections.Counter(str(r["label"]) for r in rows)),
            "language_counts": dict(collections.Counter(r["language"] for r in rows)),
            "rows_sha256": file_sha256(row_path),
            "packet_sha256": file_sha256(packet_path),
            "key_sha256": file_sha256(key_path),
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
            {
                "status": manifest["status"],
                "seed_sha256": manifest["seed_sha256"],
                "roles": manifest["roles"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
