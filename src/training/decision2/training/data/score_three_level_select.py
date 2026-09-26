"""Freeze a private, gold-blind three-level Score checkpoint-selection pilot.

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
from training.data import score_three_level_select_oracle as oracle
from training.model.data import digest, validate_row
from training.model.infer import question_to_row

VERSION = "decision20-score-three-level-select/1"
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
    ("reef-monitoring permit", "珊瑚礁监测许可"),
    ("observatory booking", "天文台预约"),
    ("river-ferry notice", "渡河船通知"),
    ("seed-bank dispatch", "种子库调拨"),
    ("heritage-map loan", "遗产地图借用"),
    ("seabird census", "海鸟普查"),
    ("theatre lighting call", "剧院灯光调度"),
    ("glacier station supply", "冰川站物资"),
    ("marine buoy repair", "海洋浮标维修"),
    ("archaeology field pass", "考古现场通行"),
    ("orchard frost watch", "果园霜冻监测"),
    ("rail-signal inspection", "铁路信号检查"),
    ("volcanic ash sample", "火山灰取样"),
    ("wetland boardwalk access", "湿地步道通行"),
    ("astronomy night session", "夜间天文观测"),
    ("coastal beacon test", "海岸信标测试"),
    ("aquarium quarantine", "水族馆隔离"),
    ("historical film transfer", "历史胶片转运"),
    ("forest canopy survey", "森林冠层调查"),
    ("harbour tide display", "港口潮位显示"),
)
OPTION_TEXT = {
    "en": ("Level 0", "Level 1", "Level 2"),
    "zh": ("0 档", "1 档", "2 档"),
}
RULE = {
    ("waiver_precedence", "en"): (
        "Use the current veto notice, not the archived notice. An active veto blocks "
        "the file unless both waiver signers signed and the waiver's inclusive "
        "signed-through-expiry period contains the review day. A remaining block "
        "means level 0. Once unblocked, a pending secondary check means level 1 "
        "and a passed secondary check means level 2."
    ),
    ("waiver_precedence", "zh"): (
        "只依据现行否决通知，不依据归档通知。现行否决生效时，仅当豁免有两名签署人，且审核日位于签署日至到期日的闭区间内，才解除阻断。"
        "仍被阻断判 0 档；解除阻断后，次级核查待完成判 1 档，已通过判 2 档。"
    ),
    ("inclusive_coverage", "en"): (
        "The requested window includes both endpoint days. Each of the two current "
        "authorities covers it only if its own inclusive window starts no later "
        "than the request and ends no earlier. Ignore the archived window. Give "
        "level 0, 1 or 2 for the number of current authorities that cover it."
    ),
    ("inclusive_coverage", "zh"): (
        "申请时段的首日和末日均计入。两家现行机构分别只有在其闭区间起点不晚于申请起点、终点不早于申请终点时才算覆盖。"
        "忽略归档时段。覆盖的现行机构数为 0、1、2 时，分别判对应档位。"
    ),
    ("independent_quorum", "en"): (
        "Count only affirmative reports that are signed and still current. Reports "
        "sharing one origin lineage count as one independent source, even if the "
        "document identifiers differ. Zero distinct qualifying lineages means "
        "level 0; one means level 1; two or more mean level 2."
    ),
    ("independent_quorum", "zh"): (
        "只计入支持结论、已签署且仍有效的报告。来源谱系相同的报告即使文号不同也只算一个独立来源。"
        "合格独立谱系数为零判 0 档，为一判 1 档，至少为二判 2 档。"
    ),
    ("allocation_caps", "en"): (
        "Assess each of the two resource pools separately. A pool is feasible "
        "only when its already committed amount plus this request plus the "
        "mandatory reserve floor does not exceed its own capacity. Give level "
        "0, 1 or 2 for the number of feasible pools. Equality is feasible."
    ),
    ("allocation_caps", "zh"): (
        "分别核算两个资源池。只有已承诺用量加本次申请量，再加必须保留的底线，未超过该池容量时，该池才可行。"
        "可行资源池数为 0、1、2 时，分别判对应档位；恰好等于容量也算可行。"
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
        lineages = rng.sample(("M", "N", "P", "R", "S", "T"), 4)
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
                "current": False,
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
        ]
        rng.shuffle(reports)
        return {"reports": reports}
    if operation == "allocation_caps":
        pools: dict[str, dict[str, int]] = {}
        for index, name in enumerate(("first", "second")):
            capacity = rng.randint(17, 36)
            committed = rng.randint(3, 8)
            reserve = rng.randint(2, 5)
            available = capacity - committed - reserve
            pools[name] = {
                "capacity": capacity,
                "committed": committed,
                "reserve_floor": reserve,
                "request": available + rng.randint(1, 3),
            }
        if rng.randrange(2):
            pools = {"first": pools["second"], "second": pools["first"]}
        return {"pools": pools}
    raise ValueError(operation)


def _variant_facts(
    operation: str, base: dict[str, Any], level: int, rng: random.Random
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
        names = ["first", "second"]
        rng.shuffle(names)
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
        rng.shuffle(qualifying)
        rng.shuffle(decoys)
        for report in [*qualifying[:level], *decoys[: 3 - level]]:
            report["signed"] = True
    elif operation == "allocation_caps":
        names = ["first", "second"]
        rng.shuffle(names)
        for name in names[:level]:
            pool = facts["pools"][name]
            available = pool["capacity"] - pool["committed"] - pool["reserve_floor"]
            pool["request"] = available - rng.randint(0, 2)
    else:
        raise ValueError(operation)
    return facts


def _render(
    operation: str, facts: dict[str, Any], locale: str, scene: str, code: str
) -> str:
    if operation == "waiver_precedence":
        waiver = facts["waiver"]
        if locale == "en":
            parts = [
                f"Dossier {code}: {scene}. Review day: {facts['review_day']}.",
                f"Archived notice: {facts['archived_notice']}; superseded by the current notice.",
                f"Current veto notice: {'active' if facts['veto_active'] else 'inactive'}.",
                f"Waiver register: lead signed={'yes' if waiver['lead_signed'] else 'no'}; "
                f"second signer={'yes' if waiver['second_signed'] else 'no'}; "
                f"signed on day {waiver['signed_day']}; expires after day {waiver['expires_day']}.",
                f"Secondary check: {facts['secondary_check']}.",
            ]
        else:
            parts = [
                f"档案 {code}：{scene}。审核日：第 {facts['review_day']} 天。",
                f"归档通知：{'无否决' if facts['archived_notice'] == 'no veto' else '有否决'}；已由现行通知取代。",
                f"现行否决通知：{'生效' if facts['veto_active'] else '未生效'}。",
                f"豁免登记：主签署人{'已签' if waiver['lead_signed'] else '未签'}；第二签署人{'已签' if waiver['second_signed'] else '未签'}；"
                f"第 {waiver['signed_day']} 天签署；有效至第 {waiver['expires_day']} 天（含当日）。",
                f"次级核查：{'待完成' if facts['secondary_check'] == 'pending' else '已通过'}。",
            ]
    elif operation == "inclusive_coverage":
        a, b = facts["request"]
        x, y = facts["archived_window"]
        windows = facts["current_windows"]
        if locale == "en":
            parts = [
                f"Dossier {code}: {scene}. Requested days: {a} through {b}, inclusive.",
                f"Archived authority window {x} through {y} is no longer operative.",
                f"Current first authority: days {windows['first'][0]} through {windows['first'][1]}.",
                f"Current second authority: days {windows['second'][0]} through {windows['second'][1]}.",
            ]
        else:
            parts = [
                f"档案 {code}：{scene}。申请时段：第 {a} 天至第 {b} 天，包含两端。",
                f"归档机构时段为第 {x} 天至第 {y} 天，现已失效。",
                f"第一家现行机构：第 {windows['first'][0]} 天至第 {windows['first'][1]} 天。",
                f"第二家现行机构：第 {windows['second'][0]} 天至第 {windows['second'][1]} 天。",
            ]
    elif operation == "independent_quorum":
        parts = [
            f"Dossier {code}: {scene}." if locale == "en" else f"档案 {code}：{scene}。"
        ]
        for index, report in enumerate(facts["reports"], 1):
            if locale == "en":
                parts.append(
                    f"Report {index}: origin lineage {report['lineage']}; "
                    f"signed={'yes' if report['signed'] else 'no'}; "
                    f"current={'yes' if report['current'] else 'no'}; "
                    f"affirmative={'yes' if report['affirmative'] else 'no'}."
                )
            else:
                parts.append(
                    f"报告 {index}：来源谱系 {report['lineage']}；"
                    f"{'已签署' if report['signed'] else '未签署'}；"
                    f"{'仍有效' if report['current'] else '已失效'}；"
                    f"{'支持结论' if report['affirmative'] else '不支持结论'}。"
                )
    elif operation == "allocation_caps":
        parts = [
            f"Dossier {code}: {scene}." if locale == "en" else f"档案 {code}：{scene}。"
        ]
        for index, name in enumerate(("first", "second"), 1):
            pool = facts["pools"][name]
            if locale == "en":
                parts.append(
                    f"Resource pool {index}: capacity {pool['capacity']}; already committed "
                    f"{pool['committed']}; this request {pool['request']}; "
                    f"mandatory reserve floor {pool['reserve_floor']}."
                )
            else:
                parts.append(
                    f"资源池 {index}：容量 {pool['capacity']}；已承诺 {pool['committed']}；"
                    f"本次申请 {pool['request']}；必须保留底线 {pool['reserve_floor']}。"
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
            group = f"s3g-{_hmac(salt, 'group', f'{operation}/{index}')[:24]}"
            scene = SCENES[index][0 if locale == "en" else 1]
            code = _hmac(salt, "dossier", f"{operation}/{index}")[:8].upper()
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
                facts = _variant_facts(operation, base, level, variant_rng)
                actual = oracle.score(operation, facts)
                if actual != level:
                    raise AssertionError(
                        f"Mechanical oracle disagrees: {operation}/{index}/{level}: {actual}"
                    )
                row_id = f"s3r-{_hmac(salt, 'row', f'{operation}/{index}/{slot}')[:24]}"
                state = _render(operation, facts, locale, scene, code)
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
                    "source": "decision2_internal_three_level_score_select",
                    "render_template": "score-three-level-select/1",
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
                    "instructions": item.get("instructions", ""),
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
        inventory.append({"role": name, "sha256": _sha(path), "rows": len(source)})
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
    matched_groups = {row["group_id"] for row in rows if row["id"] in rejected_ids}
    if any(exact.values()) or near["count"]:
        raise ValueError(
            f"Source group overlap blocks the version: exact={exact}, near={near['count']}, "
            f"groups={len(matched_groups)}"
        )
    return {
        "reference_inventory": sorted(inventory, key=lambda item: item["role"]),
        "reference_rows": len(all_reference),
        "exact": exact,
        "near": near,
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
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    seed, salt = _private_bytes(args.private_seed), _private_bytes(args.review_salt)
    if seed == salt:
        raise ValueError("Author seed and review salt must differ")
    for role, expected in FROZEN_PARENT_SHA256.items():
        path = getattr(args, f"parent_{role}")
        if _sha(path) != expected:
            raise ValueError(f"Frozen parent {role} bytes differ")
    specs, rows = generate(seed, salt)
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
    protected = json.loads(args.protected_inventory.read_text(encoding="utf-8"))
    if not isinstance(protected, list) or len(
        {entry["role"] for entry in protected}
    ) != len(protected):
        raise ValueError("Protected inventory must have unique roles")
    references.extend(
        {"role": entry["role"], "path": entry["path"], "gold_free": True}
        for entry in protected
    )
    overlap = audit_overlap(rows, references)
    from transformers import AutoTokenizer

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
    args.output_dir.mkdir(parents=True, mode=0o700)
    for directory in ("author", "oracle", "reviewer"):
        (args.output_dir / directory).mkdir(mode=0o700)
    _write(args.output_dir / "author" / "source_specs.jsonl", _jsonl_bytes(specs))
    _write(args.output_dir / "oracle" / "select.jsonl", _jsonl_bytes(rows))
    _write(args.output_dir / "reviewer" / "packet.jsonl", _jsonl_bytes(packet))
    reviewer_manifest = {
        "schema_version": VERSION,
        "status": "gold_free_frozen_pending_independent_review",
        "packet_sha256": _sha(args.output_dir / "reviewer" / "packet.jsonl"),
        "rows": len(packet),
        "groups": len(group_rows),
        "review_instructions": "Solve every row without key access, inspect complete triplets and seal judgments and shortcut findings before key comparison.",
    }
    _write(
        args.output_dir / "reviewer" / "manifest.json",
        (json.dumps(reviewer_manifest, sort_keys=True, indent=2) + "\n").encode(),
    )
    files = {
        str(path.relative_to(args.output_dir)): _sha(path)
        for path in args.output_dir.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": VERSION,
        "status": "frozen_unreviewed_do_not_run_model",
        "source_commit": args.source_commit,
        "method_sha256": _sha(args.method),
        "builder_sha256": _sha(Path(__file__)),
        "oracle_sha256": _sha(Path(oracle.__file__)),
        "seed_commitment_sha256": hashlib.sha256(seed).hexdigest(),
        "review_salt_commitment_sha256": hashlib.sha256(salt).hexdigest(),
        "parent_sha256": FROZEN_PARENT_SHA256,
        "protected_inventory_sha256": _sha(args.protected_inventory),
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_config_sha256": _sha(args.tokenizer / "tokenizer_config.json"),
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
        "files_sha256": files,
        "limitations": [
            "Selection only, never release benchmark.",
            "Chinese rows require qualified bilingual review for transfer claims.",
            "Approximate near overlap cannot prove semantic independence.",
        ],
    }
    _write(
        args.output_dir / "manifest.json",
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
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--max-row-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "groups": manifest["counts"]["groups"],
                "rows": manifest["counts"]["rows"],
                "manifest_sha256": _sha(args.output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
