"""Build controlled Decision 2.0 pilot arms without using benchmark items.

Only a caller-supplied legacy TRAIN pool enters the generated training JSONL.
SELECT and CAL are either separate audit-only inputs or held out by source
group from that pool. Neither is copied into a training arm.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import difflib
import functools
import hashlib
import json
import random
import re
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "decision2-pilot-data/1"
GENERATOR = "textual-oracle-v1"
FAMILIES = (
    "pilot_string_composition",
    "pilot_narrative_reading",
    "pilot_open_world_abstention",
)
SOURCE = "decision2_programmatic_original_v1"
BENCHMARK_FAMILIES = {
    "attribute_gate",
    "rule_precedence",
    "set_reconciliation",
    "transition_table",
    "constraint_competition",
    "exception_stack",
    "evidence_join",
    "resource_ledger",
}
ROOT = Path(__file__).resolve().parents[2]


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("state", "instructions", "options", "task_type")}


def input_sha256(row: dict[str, Any]) -> str:
    return sha_bytes(canonical(input_payload(row)).encode("utf-8"))


def rng_for(seed: str, family: str, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{family}\0{index}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _common_row(
    seed: str,
    family: str,
    index: int,
    state: str,
    instructions: str,
    options: list[dict[str, str]],
    answer_key: str,
    task_type: str,
    template: str,
    audit: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    suffix = sha_bytes(f"{seed}\0{family}\0{index}".encode())[:20]
    row = {
        "id": f"d2p_{suffix}",
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": next(
            i for i, option in enumerate(options) if option["key"] == answer_key
        ),
        "task_type": task_type,
        "family": family,
        "group_id": f"d2pg_{suffix}",
        "language": language,
        "split": "train",
        "source": SOURCE,
        "evaluation_role": "train",
        "render_template": template,
        "audit_metadata": {"generator": GENERATOR, "case_index": index, **audit},
    }
    row["input_sha256"] = input_sha256(row)
    validate_train_row(row)
    return row


def _choice_options(
    rng: random.Random, values: list[str], answer: str
) -> tuple[list[dict[str, str]], str]:
    if len(set(values)) != len(values) or answer not in values:
        raise ValueError("Choice candidates must be distinct and include the answer")
    rng.shuffle(values)
    options = [
        {"key": f"K{i + 1}", "description": value} for i, value in enumerate(values)
    ]
    return options, options[values.index(answer)]["key"]


def _apply(code: str, op: str) -> str:
    if op == "reverse":
        return code[::-1]
    if op == "rotate_left":
        return code[1:] + code[0]
    if op == "swap_ends":
        return code[-1] + code[1:-1] + code[0]
    if op == "replace_middle_z":
        return code[:2] + "Z" + code[3:]
    raise ValueError(op)


def make_composition(seed: str, index: int) -> dict[str, Any]:
    rng = rng_for(seed, FAMILIES[0], index)
    language = "zh" if index % 5 == 0 else "en"
    operations = ["reverse", "rotate_left", "swap_ends", "replace_middle_z"]
    for _ in range(100):
        start = "".join(rng.sample("ABCDEFGHJKLMNPQRSTUVWXY", 5))
        ops = rng.sample(operations, 3 if index % 3 == 0 else 2)
        steps = [start]
        for op in ops:
            steps.append(_apply(steps[-1], op))
        final = steps[-1]
        wrong_order = start
        for op in reversed(ops):
            wrong_order = _apply(wrong_order, op)
        candidates = [start, *steps[1:-1], wrong_order]
        if final not in candidates and len(set(candidates)) >= 2:
            break
    else:
        raise RuntimeError("Could not construct a distinct composition case")
    while len(set(candidates)) < 3:
        candidates.append("".join(rng.sample("ABCDEFGHJKLMNPQRSTUVWXY", 5)))
    alternatives = list(dict.fromkeys(candidates))[:3]
    names_en = {
        "reverse": "reverse the entire code",
        "rotate_left": "move the first symbol to the end",
        "swap_ends": "swap the first and last symbols",
        "replace_middle_z": "replace the third symbol with Z",
    }
    names_zh = {
        "reverse": "将整个代码倒序",
        "rotate_left": "把首字符移到末尾",
        "swap_ends": "交换首尾字符",
        "replace_middle_z": "把第三个字符替换成 Z",
    }
    if language == "zh":
        state = (
            "标签机从代码 "
            + start
            + " 开始。按下列顺序执行，不能跳步：\n"
            + "\n".join(f"{i}. {names_zh[op]}。" for i, op in enumerate(ops, 1))
        )
        instructions = "完成全部操作后，打印出的五字符代码是什么？"
    else:
        state = (
            "A badge printer starts with code "
            + start
            + ". Apply the actions in the listed order:\n"
            + "\n".join(f"{i}. {names_en[op]}." for i, op in enumerate(ops, 1))
        )
        instructions = "Which five-symbol code is printed after every action?"
    options, key = _choice_options(rng, [final, *alternatives], final)
    return _common_row(
        seed,
        FAMILIES[0],
        index,
        state,
        instructions,
        options,
        key,
        "choice",
        "ordered_character_actions_v1",
        {"start": start, "operations": ops, "oracle_result": final},
        language,
    )


def make_reading(seed: str, index: int) -> dict[str, Any]:
    rng = rng_for(seed, FAMILIES[1], index)
    language = "zh" if index % 5 == 1 else "en"
    people = rng.sample(
        ["Ada", "Bela", "Cora", "Dina", "Ema", "Faye", "Gina", "Hana", "Iris"], 3
    )
    first, second, third = people
    items = rng.sample(
        [
            "amber compass",
            "blue atlas",
            "copper bell",
            "silver spool",
            "green journal",
            "violet lens",
            "red lantern",
            "brass key",
        ],
        3,
    )
    handoff, desk_item, shelf_item = items
    if language == "zh":
        state = (
            f"档案室的便条写道：{first} 把 {handoff} 交给 {second}。{second} 随后把它放进木盒。"
            f"另外，{third} 把 {desk_item} 留在桌上；{third} 把 {shelf_item} 交给 {first}，"
            f"{first} 把这件物品放在架上。"
        )
    else:
        state = (
            f"The archive note says that {first} handed the {handoff} to {second}. "
            f"{second} then placed it in the wooden box. Separately, {third} left the {desk_item} "
            f"on the desk; {third} gave the {shelf_item} to {first}, who put that item on the shelf."
        )
    if index % 2 == 0:
        instructions = (
            "木盒里放的是哪件物品？"
            if language == "zh"
            else "Which item was placed in the wooden box?"
        )
        options, key = _choice_options(
            rng, [handoff, desk_item, shelf_item, "not mentioned"], handoff
        )
        task_type, answer = "choice", handoff
    else:
        asserted = handoff if index % 4 == 1 else desk_item
        truth = asserted == handoff
        instructions = (
            f"{second} 把 {asserted} 放进木盒了吗？"
            if language == "zh"
            else f"Did {second} place the {asserted} in the wooden box?"
        )
        options = [
            {"key": key, "description": "True" if key == "true" else "False"}
            for key in rng.sample(["false", "true"], 2)
        ]
        key = "true" if truth else "false"
        task_type, answer = "noul", truth
    return _common_row(
        seed,
        FAMILIES[1],
        index,
        state,
        instructions,
        options,
        key,
        task_type,
        "archive_handoff_prose_v1",
        {
            "people": people,
            "items": items,
            "wooden_box_item": handoff,
            "oracle_result": answer,
        },
        language,
    )


def make_abstention(seed: str, index: int) -> dict[str, Any]:
    rng = rng_for(seed, FAMILIES[2], index)
    language = "zh" if index % 5 == 2 else "en"
    codes = [
        "ART-" + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXY23456789", k=5))
        for _ in range(3)
    ]
    while len(set(codes)) != 3:
        codes = [
            "ART-" + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXY23456789", k=5))
            for _ in range(3)
        ]
    materials = rng.sample(["linen", "brass", "clay"], 3)
    finishes = rng.sample(["matte", "polished", "satin"], 3)
    target = index % 3
    field = "material" if index % 4 < 2 else "finish"
    is_unknown = index % 2 == 0
    lines = []
    for j, code in enumerate(codes):
        attrs = {"material": materials[j], "finish": finishes[j]}
        if j == target and is_unknown:
            del attrs[field]
        rendered = "; ".join(f"{key}: {value}" for key, value in attrs.items())
        lines.append(f"{code} — {rendered}.")
    if language == "zh":
        state = "馆藏卡片只记录了以下内容：\n" + "\n".join(lines)
        instructions = f"卡片上写的 {codes[target]} 的 {field} 是什么？若未记录该属性，选择 not stated。"
    else:
        state = "Only these details appear on the collection cards:\n" + "\n".join(
            lines
        )
        instructions = f"What {field} does the card state for {codes[target]}? Select not stated if that attribute is absent."
    values = (
        ["linen", "brass", "clay"]
        if field == "material"
        else ["matte", "polished", "satin"]
    )
    answer = (
        "not stated"
        if is_unknown
        else (materials[target] if field == "material" else finishes[target])
    )
    options, key = _choice_options(rng, [*values, "not stated"], answer)
    return _common_row(
        seed,
        FAMILIES[2],
        index,
        state,
        instructions,
        options,
        key,
        "choice",
        "partial_collection_cards_v1",
        {
            "cards": codes,
            "queried_card": codes[target],
            "queried_field": field,
            "field_is_recorded": not is_unknown,
            "oracle_result": answer,
        },
        language,
    )


GENERATORS = dict(zip(FAMILIES, (make_composition, make_reading, make_abstention)))


def validate_train_row(row: dict[str, Any]) -> None:
    required = (
        "id",
        "state",
        "instructions",
        "options",
        "label",
        "task_type",
        "family",
        "group_id",
        "language",
        "split",
        "source",
        "evaluation_role",
        "render_template",
        "audit_metadata",
        "input_sha256",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{row.get('id', '<no id>')}: missing fields {missing}")
    if row["split"] != "train" or row["evaluation_role"] != "train":
        raise ValueError(f"{row['id']}: non-train item in training output")
    if not all(
        isinstance(row[key], str) and row[key].strip()
        for key in ("id", "family", "group_id", "source", "language", "render_template")
    ):
        raise ValueError(f"{row['id']}: empty required text field")
    if (
        not isinstance(row["instructions"], (str, dict, list))
        or not row["instructions"]
    ):
        raise ValueError(f"{row['id']}: instructions must be nonempty text or JSON")
    if not isinstance(row["state"], (dict, list, str)):
        raise ValueError(f"{row['id']}: state must be JSON text/object/list")
    if row["task_type"] not in {"choice", "noul", "score"}:
        raise ValueError(f"{row['id']}: unsupported task type")
    opts = row["options"]
    if (
        not isinstance(opts, list)
        or not 2 <= len(opts) <= 255
        or any(
            not isinstance(opt, dict)
            or not isinstance(opt.get("key"), str)
            or not opt["key"]
            or "description" not in opt
            for opt in opts
        )
    ):
        raise ValueError(f"{row['id']}: invalid options")
    try:
        for option in opts:
            canonical(option["description"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{row['id']}: option description must be finite JSON"
        ) from exc
    if len({opt["key"] for opt in opts}) != len(opts):
        raise ValueError(f"{row['id']}: duplicate option keys")
    if (
        not isinstance(row["label"], int)
        or isinstance(row["label"], bool)
        or not 0 <= row["label"] < len(opts)
    ):
        raise ValueError(f"{row['id']}: invalid label")
    if row["task_type"] == "noul" and {opt["key"] for opt in opts} != {"true", "false"}:
        raise ValueError(f"{row['id']}: noul requires false/true options")
    if row["task_type"] == "score" and {opt["key"] for opt in opts} != {
        str(i) for i in range(len(opts))
    }:
        raise ValueError(f"{row['id']}: score keys must enumerate levels 0..K-1")
    if not isinstance(row["audit_metadata"], dict) or row[
        "input_sha256"
    ] != input_sha256(row):
        raise ValueError(f"{row['id']}: invalid audit metadata or input hash")
    if "target_probs" in row or "teacher_probs" in row:
        raise ValueError(f"{row['id']}: pilot arm must contain hard gold only")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path.name}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{line_number}: expected object")
                rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: empty JSONL")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: duplicate or missing IDs")
    return rows


def read_legacy_train(path: Path) -> list[dict[str, Any]]:
    if "benchmark" in path.parts or path.name.endswith(
        (".gold.jsonl", ".prompts.jsonl")
    ):
        raise ValueError("Benchmark files cannot be legacy training input")
    rows = _read_jsonl(path)
    normalized = []
    for old in rows:
        if (
            old.get("schema_version") == "typed-decision-bench/1"
            or old.get("split") not in (None, "train")
            or old.get("evaluation_role") not in (None, "train")
        ):
            raise ValueError(f"{old['id']}: non-train row in legacy input")
        if old.get("family") in BENCHMARK_FAMILIES:
            raise ValueError(f"{old['id']}: benchmark family cannot enter training")
        if not old.get("group_id"):
            raise ValueError(f"{old['id']}: source group_id required for leakage audit")
        row = dict(old)
        normalization: dict[str, Any] = {}
        for soft_field in ("target_probs", "teacher_probs"):
            if soft_field in row:
                if row[soft_field] is not None:
                    raise ValueError(
                        f"{old['id']}: soft target requires separate replay protocol"
                    )
                del row[soft_field]
        if "task_type" not in row:
            row["task_type"] = "choice"
        row["split"] = row["evaluation_role"] = "train"
        original_source = row.get("source")
        if isinstance(original_source, dict):
            lineage = (
                original_source.get("dataset")
                or original_source.get("generator")
                or original_source.get("type")
                or "unknown"
            )
            if not isinstance(lineage, str):
                raise ValueError(f"{old['id']}: non-text source lineage")
            row["source"] = "legacy:" + lineage
            normalization["structured_source_to_id"] = True
        elif not original_source:
            row["source"] = "legacy_source_unverified"
            normalization["missing_source_defaulted"] = True
        elif not isinstance(original_source, str):
            raise ValueError(f"{old['id']}: source must be text or object")
        row["language"] = row.get("language") or "unspecified"
        if not row.get("render_template"):
            normalization["missing_render_template_defaulted"] = True
        row["render_template"] = row.get("render_template") or "legacy_unspecified"
        audit = row.get("audit_metadata") or {}
        if not isinstance(audit, dict):
            raise ValueError(f"{old['id']}: audit_metadata must be object")
        audit = dict(audit)
        if normalization:
            if isinstance(original_source, dict):
                audit["original_source"] = original_source
            audit["legacy_normalization"] = normalization
        digest = input_sha256(row)
        if row.get("input_sha256") and row["input_sha256"] != digest:
            audit["original_input_sha256"] = row["input_sha256"]
        row["audit_metadata"] = audit
        row["input_sha256"] = digest
        validate_train_row(row)
        normalized.append(row)
    return normalized


def read_reference(path: Path, expected: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    allowed_splits = (
        {"select", "dev"} if expected == "select" else {"cal", "calibration"}
    )
    allowed_roles = {"select"} if expected == "select" else {"cal", "calibrate"}
    for row in rows:
        if row.get("split") not in (None, *allowed_splits) or row.get(
            "evaluation_role"
        ) not in (None, *allowed_roles):
            raise ValueError(f"{row['id']}: wrong {expected} split/role")
        if not row.get("group_id"):
            raise ValueError(f"{row['id']}: group_id required for {expected} audit")
        for key in ("state", "instructions", "options"):
            if key not in row:
                raise ValueError(
                    f"{row['id']}: flattened {expected} reference required; missing {key}"
                )
        if "task_type" not in row:
            row["task_type"] = "choice"
    return rows


def sample_legacy(
    rows: list[dict[str, Any]], count: int, seed: str
) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"Requested {count} legacy rows, only {len(rows)} available")
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    ordered = sorted(
        grouped, key=lambda group: sha_bytes(f"{seed}\0legacy\0{group}".encode())
    )
    selected, remaining = [], count
    for group in ordered:
        if len(grouped[group]) <= remaining:
            selected.extend(grouped[group])
            remaining -= len(grouped[group])
            if remaining == 0:
                break
    if remaining:
        raise ValueError(
            f"Cannot select exactly {count} legacy rows without splitting source groups"
        )
    return selected


def derive_holdouts(
    rows: list[dict[str, Any]], select_count: int, cal_count: int, seed: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reserve complete source groups before any training-arm sampling."""
    if select_count <= 0 or cal_count <= 0:
        raise ValueError("Derived SELECT/CAL counts must be positive")
    selected = sample_legacy(rows, select_count, seed + "\0select")
    selected_groups = {row["group_id"] for row in selected}
    remaining = [row for row in rows if row["group_id"] not in selected_groups]
    calibrated = sample_legacy(remaining, cal_count, seed + "\0cal")
    calibrated_groups = {row["group_id"] for row in calibrated}
    train_pool = [row for row in remaining if row["group_id"] not in calibrated_groups]

    def reference_rows(
        source_rows: list[dict[str, Any]], split: str, role: str
    ) -> list[dict[str, Any]]:
        result = []
        for source_row in source_rows:
            row = dict(source_row)
            row["audit_metadata"] = {
                **source_row["audit_metadata"],
                "holdout_partition": "group_reservation_from_legacy_pool_v1",
            }
            row["split"], row["evaluation_role"] = split, role
            result.append(row)
        return result

    return (
        train_pool,
        reference_rows(selected, "select", "select"),
        reference_rows(calibrated, "cal", "calibrate"),
    )


def _near_text(row: dict[str, Any]) -> str:
    payload = input_payload(row)
    payload["options"] = sorted(payload["options"], key=canonical)
    return re.sub(r"\s+", " ", canonical(payload).casefold()).strip()


@functools.lru_cache(maxsize=100_000)
def _simhash(value: str) -> int:
    words = re.findall(r"[^\W_]+|\S", value, flags=re.UNICODE)
    features = {" ".join(words[i : i + 3]) for i in range(max(1, len(words) - 2))}
    vector = [0] * 64
    for feature in features:
        bits = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:8], "big")
        for bit in range(64):
            vector[bit] += 1 if bits & (1 << bit) else -1
    return sum(1 << bit for bit, weight in enumerate(vector) if weight >= 0)


def near_duplicates(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    threshold: float = 0.94,
    collect_left_ids: bool = False,
) -> dict[str, Any]:
    """Approximate candidate search followed by exact text-ratio verification.

    Eight 8-bit SimHash bands are an index, not a mathematical completeness
    guarantee. The manifest records this limitation and exact-hash overlap
    is checked separately.
    """
    right_text = [_near_text(row) for row in right]
    right_bits = [_simhash(value) for value in right_text]
    bands: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for index, bits in enumerate(right_bits):
        for band in range(8):
            bands[(band, (bits >> (band * 8)) & 255)].append(index)
    matches = []
    count = 0
    matched_left_ids = set()
    for row in left:
        value = _near_text(row)
        bits = _simhash(value)
        candidates = set()
        for band in range(8):
            candidates.update(bands.get((band, (bits >> (band * 8)) & 255), ()))
        for index in candidates:
            other = right_text[index]
            if abs(len(value) - len(other)) > 0.08 * max(len(value), len(other)):
                continue
            if (bits ^ right_bits[index]).bit_count() > 8:
                continue
            ratio = difflib.SequenceMatcher(None, value, other).ratio()
            if ratio >= threshold:
                count += 1
                if collect_left_ids:
                    matched_left_ids.add(row["id"])
                if len(matches) < 20:
                    matches.append(
                        {
                            "left_id": row["id"],
                            "right_id": right[index]["id"],
                            "similarity": round(ratio, 5),
                        }
                    )
    result = {
        "count": count,
        "examples": matches,
        "method": "8-band 64-bit SimHash candidates; SequenceMatcher>=0.94; approximate",
    }
    if collect_left_ids:
        result["left_ids"] = sorted(matched_left_ids)
    return result


def overlap_audit(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    fields = {
        "id": lambda row: row["id"],
        "group_id": lambda row: row["group_id"],
        "input_sha256": input_sha256,
    }
    result = {}
    for name, getter in fields.items():
        shared = set(map(getter, left)) & set(map(getter, right))
        result[name] = {"count": len(shared), "examples": sorted(shared)[:20]}
    result["near_duplicate"] = near_duplicates(left, right)
    return result


def audit_has_exact_overlap(audit: dict[str, Any]) -> bool:
    return any(audit[key]["count"] for key in ("id", "group_id", "input_sha256"))


def quarantine_holdout_neighbors(
    train_pool: list[dict[str, Any]], holdouts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove complete train source groups touching holdout inputs or near matches."""
    holdout_ids = {row["id"] for row in holdouts}
    holdout_groups = {row["group_id"] for row in holdouts}
    holdout_inputs = {input_sha256(row) for row in holdouts}
    near = near_duplicates(train_pool, holdouts, collect_left_ids=True)
    near_ids = set(near.pop("left_ids"))
    exact_id_rows = {row["id"] for row in train_pool if row["id"] in holdout_ids}
    exact_group_rows = {
        row["id"] for row in train_pool if row["group_id"] in holdout_groups
    }
    exact_input_rows = {
        row["id"] for row in train_pool if row["input_sha256"] in holdout_inputs
    }
    exact_ids = exact_id_rows | exact_group_rows | exact_input_rows
    flagged_ids = near_ids | exact_ids
    excluded_groups = {
        row["group_id"] for row in train_pool if row["id"] in flagged_ids
    }
    excluded = [row for row in train_pool if row["group_id"] in excluded_groups]
    clean = [row for row in train_pool if row["group_id"] not in excluded_groups]
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in excluded:
        by_group[row["group_id"]].append(row)
    details = []
    for group, group_rows in sorted(by_group.items()):
        ids = {row["id"] for row in group_rows}
        reasons = [
            name
            for name, flagged in (
                ("near_input", near_ids),
                ("exact_id", exact_id_rows),
                ("exact_group", exact_group_rows),
                ("exact_input", exact_input_rows),
            )
            if ids & flagged
        ]
        details.append(
            {
                "group_id": group,
                "rows": len(group_rows),
                "reasons": reasons,
                "family_counts": dict(
                    sorted(
                        collections.Counter(row["family"] for row in group_rows).items()
                    )
                ),
            }
        )
    receipt = {
        "method": "whole-source-group exclusion after exact and approximate near-input search",
        "near_candidate_pairs": near["count"],
        "near_candidate_rows": len(near_ids),
        "exact_candidate_rows": len(exact_ids),
        "exact_id_rows": len(exact_id_rows),
        "exact_group_rows": len(exact_group_rows),
        "exact_input_rows": len(exact_input_rows),
        "excluded_groups": len(excluded_groups),
        "excluded_rows": len(excluded),
        "excluded_group_ids_sha256": sha_bytes(
            canonical(sorted(excluded_groups)).encode()
        ),
        "excluded_family_counts": dict(
            sorted(collections.Counter(row["family"] for row in excluded).items())
        ),
        "excluded_group_details": details,
    }
    return clean, receipt


def exclude_overlong_groups(
    rows: list[dict[str, Any]], max_tokens: int, length_for: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every output partition within the same encoder length ceiling."""
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    excluded_groups = set()
    details = []
    for group, group_rows in sorted(grouped.items()):
        lengths = [length_for(row) for row in group_rows]
        excessive = sum(length > max_tokens for length in lengths)
        if excessive:
            excluded_groups.add(group)
            details.append(
                {
                    "group_id": group,
                    "rows": len(group_rows),
                    "overlong_rows": excessive,
                    "maximum_tokens": max(lengths),
                    "reason": "over_max_row_tokens",
                }
            )
    excluded = [row for row in rows if row["group_id"] in excluded_groups]
    clean = [row for row in rows if row["group_id"] not in excluded_groups]
    receipt = {
        "limit": max_tokens,
        "excluded_groups": len(excluded_groups),
        "excluded_rows": len(excluded),
        "excluded_group_ids_sha256": sha_bytes(
            canonical(sorted(excluded_groups)).encode()
        ),
        "excluded_family_counts": dict(
            sorted(collections.Counter(row["family"] for row in excluded).items())
        ),
        "excluded_group_details": details,
    }
    return clean, receipt


def train_consistency_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_input: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_input[row["input_sha256"]].append(row)
    repeated = [group for group in by_input.values() if len(group) > 1]
    conflicts = [
        group
        for group in repeated
        if len({group_row["options"][group_row["label"]]["key"] for group_row in group})
        > 1
    ]
    return {
        "repeated_input_groups": len(repeated),
        "repeated_input_rows": sum(len(group) for group in repeated),
        "conflicting_gold_groups": len(conflicts),
        "conflicting_gold_examples": [
            [row["id"] for row in group[:5]] for group in conflicts[:20]
        ],
    }


def legacy_normalization_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        changes = row["audit_metadata"].get("legacy_normalization", {})
        for key, value in changes.items():
            counts[key] += value if isinstance(value, int) else int(bool(value))
        counts["original_input_sha256_preserved"] += (
            "original_input_sha256" in row["audit_metadata"]
        )
    return dict(sorted(counts.items()))


def matched_legacy_subset(
    base_rows: list[dict[str, Any]],
    keep_count: int,
    new_rows: list[dict[str, Any]],
    length_for: Any,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace complete singleton source groups with similar total token cost.

    The baseline is fixed first. Removing only singleton groups preserves all
    multi-row parent components. This explicitly biases the replaced old rows
    toward the new examples' lengths, so the manifest records their mix.
    """
    remove_count = len(base_rows) - keep_count
    if remove_count <= 0 or remove_count != len(new_rows):
        raise ValueError(
            "Token-matched arm must replace a positive, equal number of legacy rows"
        )
    group_sizes = collections.Counter(row["group_id"] for row in base_rows)
    candidates = [row for row in base_rows if group_sizes[row["group_id"]] == 1]
    if len(candidates) < remove_count:
        raise ValueError(
            "Too few singleton legacy groups for row-and-token matched replacement"
        )
    target = sum(length_for(row) for row in new_rows)
    mean = target / remove_count
    candidates.sort(
        key=lambda row: (
            abs(length_for(row) - mean),
            sha_bytes(f"{seed}\0replace\0{row['id']}".encode()),
        )
    )
    chosen = candidates[:remove_count]
    remaining = candidates[remove_count:]
    removed_tokens = sum(length_for(row) for row in chosen)
    # Local exact-token swaps minimize the difference without changing row
    # count or crossing a multi-row group boundary.
    for _ in range(64):
        remaining.sort(key=lambda row: (length_for(row), row["id"]))
        remaining_lengths = [length_for(row) for row in remaining]
        best = None
        best_error = abs(removed_tokens - target)
        for chosen_index, old_row in enumerate(chosen):
            desired = target - (removed_tokens - length_for(old_row))
            offset = bisect.bisect_left(remaining_lengths, desired)
            for remaining_index in (offset - 1, offset):
                if 0 <= remaining_index < len(remaining):
                    new_total = (
                        removed_tokens
                        - length_for(old_row)
                        + remaining_lengths[remaining_index]
                    )
                    error = abs(new_total - target)
                    if error < best_error:
                        best_error = error
                        best = (chosen_index, remaining_index, new_total)
        if best is None:
            break
        chosen_index, remaining_index, removed_tokens = best
        chosen[chosen_index], remaining[remaining_index] = (
            remaining[remaining_index],
            chosen[chosen_index],
        )
        if removed_tokens == target:
            break
    removed_ids = {row["id"] for row in chosen}
    kept = [row for row in base_rows if row["id"] not in removed_ids]
    if len(kept) != keep_count:
        raise AssertionError("Matched replacement lost the requested row count")
    receipt = {
        "strategy": "singleton-source-group-token-match-v1",
        "removed_rows": len(chosen),
        "removed_groups": len({row["group_id"] for row in chosen}),
        "removed_tokens": removed_tokens,
        "new_tokens": target,
        "replacement_token_delta": target - removed_tokens,
        "removed_family_counts": dict(
            sorted(collections.Counter(row["family"] for row in chosen).items())
        ),
        "removed_task_type_counts": dict(
            sorted(collections.Counter(row["task_type"] for row in chosen).items())
        ),
        "removed_language_counts": dict(
            sorted(collections.Counter(row["language"] for row in chosen).items())
        ),
        "removed_id_sha256": sha_bytes(canonical(sorted(removed_ids)).encode()),
    }
    return kept, receipt


def stratified_legacy_subset(
    base_rows: list[dict[str, Any]],
    keep_count: int,
    new_rows: list[dict[str, Any]],
    length_for: Any,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove whole groups in proportion to legacy family/type/language strata.

    This deliberately does not optimize token length. It gives a complementary
    comparison to token matching and exposes the resulting token imbalance.
    """
    remove_count = len(base_rows) - keep_count
    if remove_count <= 0 or remove_count != len(new_rows):
        raise ValueError(
            "Stratified arm must replace a positive, equal number of legacy rows"
        )
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in base_rows:
        by_group[row["group_id"]].append(row)
    by_stratum: dict[tuple[str, str, str], list[tuple[str, list[dict[str, Any]]]]] = (
        collections.defaultdict(list)
    )
    stratum_rows: collections.Counter[tuple[str, str, str]] = collections.Counter()
    for group_id, rows in by_group.items():
        strata = {(row["family"], row["task_type"], row["language"]) for row in rows}
        if len(strata) != 1:
            raise ValueError(
                f"Cannot stratify mixed family/type/language source group {group_id}"
            )
        stratum = next(iter(strata))
        by_stratum[stratum].append((group_id, rows))
        stratum_rows[stratum] += len(rows)
    total = len(base_rows)
    quotas = {
        stratum: remove_count * count // total
        for stratum, count in stratum_rows.items()
    }
    unassigned = remove_count - sum(quotas.values())
    ranked = sorted(
        stratum_rows,
        key=lambda key: (
            -(remove_count * stratum_rows[key] % total),
            sha_bytes(f"{seed}\0quota\0{canonical(key)}".encode()),
        ),
    )
    for stratum in ranked[:unassigned]:
        quotas[stratum] += 1
    selected_groups: set[str] = set()
    actual: collections.Counter[tuple[str, str, str]] = collections.Counter()
    for stratum in sorted(by_stratum):
        groups = sorted(
            by_stratum[stratum],
            key=lambda item: sha_bytes(f"{seed}\0stratum\0{item[0]}".encode()),
        )
        # Bounded subset sum picks complete groups up to the target quota.
        reachable: dict[int, tuple[str, ...]] = {0: ()}
        for group_id, rows in groups:
            size = len(rows)
            for subtotal, chosen in sorted(reachable.items(), reverse=True):
                new_total = subtotal + size
                if new_total <= quotas[stratum] and new_total not in reachable:
                    reachable[new_total] = (*chosen, group_id)
        best = max(reachable)
        selected_groups.update(reachable[best])
        actual[stratum] = best
    remaining = remove_count - sum(actual.values())
    if remaining:
        # Quota gaps can arise from coarse groups; repair with complete groups.
        # The adjustment favors strata below their quota and smaller groups.
        candidates = [
            (stratum, group_id, rows)
            for stratum, groups in by_stratum.items()
            for group_id, rows in groups
            if group_id not in selected_groups
        ]
        while remaining:
            fit = [item for item in candidates if len(item[2]) <= remaining]
            if not fit:
                raise ValueError(
                    "Cannot reach exact row count with complete source groups"
                )
            fit.sort(
                key=lambda item: (
                    -(quotas[item[0]] - actual[item[0]]),
                    len(item[2]),
                    sha_bytes(f"{seed}\0repair\0{item[1]}".encode()),
                )
            )
            stratum, group_id, rows = fit[0]
            selected_groups.add(group_id)
            actual[stratum] += len(rows)
            remaining -= len(rows)
            candidates.remove(fit[0])
    removed = [row for row in base_rows if row["group_id"] in selected_groups]
    kept = [row for row in base_rows if row["group_id"] not in selected_groups]
    if len(removed) != remove_count or len(kept) != keep_count:
        raise AssertionError("Stratified replacement lost the requested row count")
    removed_tokens = sum(length_for(row) for row in removed)
    new_tokens = sum(length_for(row) for row in new_rows)
    return kept, {
        "strategy": "whole-source-group-family-task-language-stratified-v1",
        "removed_rows": len(removed),
        "removed_groups": len(selected_groups),
        "removed_tokens": removed_tokens,
        "new_tokens": new_tokens,
        "replacement_token_delta": new_tokens - removed_tokens,
        "removed_family_counts": dict(
            sorted(collections.Counter(row["family"] for row in removed).items())
        ),
        "removed_task_type_counts": dict(
            sorted(collections.Counter(row["task_type"] for row in removed).items())
        ),
        "removed_language_counts": dict(
            sorted(collections.Counter(row["language"] for row in removed).items())
        ),
        "stratum_quotas": {"|".join(key): quotas[key] for key in sorted(quotas)},
        "stratum_removed": {"|".join(key): actual[key] for key in sorted(quotas)},
        "removed_id_sha256": sha_bytes(
            canonical(sorted(row["id"] for row in removed)).encode()
        ),
        "length_selection_bias": "none; exact token budget is not targeted",
    }


def read_source_license_evidence(
    path: Path | None,
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        raise ValueError("Source license evidence needs a sources mapping")
    for source, record in records.items():
        if not isinstance(source, str) or not source or not isinstance(record, dict):
            raise ValueError("Invalid source license evidence entry")
        for key in ("license", "attribution", "evidence"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{source}: source evidence requires nonempty {key}")
    return records, {"file": path.name, "sha256": sha_file(path)}


def count_tokens(row: dict[str, Any], tokenizer: Any) -> int:
    def payload(value: Any) -> str:
        return value if isinstance(value, str) else canonical(value)

    prefix = f"Context:\n{payload(row['state'])}\n\nTask type: {row['task_type']}\nQuestion:\n{payload(row['instructions'])}\nOptions:"
    suffix = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"
    pieces = (
        [prefix]
        + [
            "\n<option>\n"
            + canonical(
                {"key": option["key"], "description": option.get("description")}
            )
            + "\n</option>"
            for option in row["options"]
        ]
        + [suffix]
    )
    return sum(
        len(tokenizer.encode(piece, add_special_tokens=False)) for piece in pieces
    )


def parse_arm(spec: str) -> tuple[str, tuple[int, int, int, int]]:
    try:
        name, *raw_counts = spec.split(":")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or len(raw_counts) != 4:
            raise ValueError
        counts = tuple(int(value) for value in raw_counts)
        if any(value < 0 for value in counts) or sum(counts) == 0:
            raise ValueError
        return name, counts  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(
            "Arm format: name:legacy:composition:reading:abstention, nonnegative integers"
        ) from exc


def _atomic_write(path: Path, raw: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
    temporary.replace(path)


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(canonical(row) + "\n" for row in rows)).encode("utf-8")


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    legacy_path = args.legacy_train.resolve()
    derive = getattr(args, "derive_holdouts", False)
    if derive:
        if args.select or args.cal:
            raise ValueError(
                "Use either --derive-holdouts or external --select and --cal"
            )
        if args.select_count is None or args.cal_count is None:
            raise ValueError(
                "--derive-holdouts requires --select-count and --cal-count"
            )
    else:
        if not args.select or not args.cal:
            raise ValueError("External references require both --select and --cal")
        if args.select_count is not None or args.cal_count is not None:
            raise ValueError("--select-count and --cal-count require --derive-holdouts")
        reference_paths = [args.select.resolve(), args.cal.resolve()]
        if len({legacy_path, *reference_paths}) != 3:
            raise ValueError("Legacy TRAIN, SELECT and CAL must be different files")
    output_dir = args.output_dir.resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        relative = output_dir.relative_to(ROOT)
        if not relative.parts or relative.parts[0] not in {"runs", ".private"}:
            raise ValueError(
                "Within this repository, write generated data only under ignored runs/ or .private/"
            )
    arm_specs = [parse_arm(value) for value in args.arm]
    if len({name for name, _ in arm_specs}) != len(arm_specs):
        raise ValueError("Duplicate arm names")
    for name in ("max_row_tokens", "max_arm_tokens"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.match_arm_tokens_percent is not None and args.match_arm_tokens_percent < 0:
        raise ValueError("--match-arm-tokens-percent must be nonnegative")
    if (
        args.max_row_tokens
        or args.max_arm_tokens
        or args.match_arm_tokens_percent is not None
    ):
        if not args.tokenizer:
            raise ValueError("Exact token limits require --tokenizer")
    if args.tokenizer and not args.tokenizer_revision:
        raise ValueError("--tokenizer-revision is required with --tokenizer")
    token_replacement = getattr(args, "token_match_replacements", False)
    stratified_replacement = getattr(args, "stratified_replacements", False)
    if token_replacement and stratified_replacement:
        raise ValueError("Choose one replacement strategy")
    if token_replacement or stratified_replacement:
        if not args.tokenizer:
            raise ValueError("Replacement strategies require --tokenizer")
        if arm_specs[0][1][0] != max(counts[0] for _, counts in arm_specs) or any(
            arm_specs[0][1][1:]
        ):
            raise ValueError(
                "First arm must be the largest legacy-only baseline for matched replacement"
            )
    source_evidence, source_evidence_file = read_source_license_evidence(
        getattr(args, "source_license_evidence", None)
    )

    def attribution_for(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        sources = sorted({row["source"] for row in rows})
        if source_evidence is not None:
            missing = sorted(set(sources) - set(source_evidence))
            if missing:
                raise ValueError(f"Source license evidence missing {missing}")
            return {source: source_evidence[source] for source in sources}
        return {
            source: {
                "license": "UNVERIFIED_BEFORE_PUBLICATION",
                "attribution": "Fill from upstream source record before release",
            }
            for source in sources
        }

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer), local_files_only=True, trust_remote_code=False
        )
    token_cache: dict[str, int] = {}

    def length_for(row: dict[str, Any]) -> int:
        if row["id"] not in token_cache:
            token_cache[row["id"]] = count_tokens(row, tokenizer)
        return token_cache[row["id"]]

    legacy = read_legacy_train(legacy_path)
    legacy_source_rows = len(legacy)
    source_sha = sha_file(legacy_path)
    holdout_files: dict[str, bytes] = {}
    holdout_manifest = None
    quarantine_receipt = None
    overlong_receipt = None
    if derive:
        if tokenizer and args.max_row_tokens:
            legacy, overlong_receipt = exclude_overlong_groups(
                legacy, args.max_row_tokens, length_for
            )
        legacy, select, cal = derive_holdouts(
            legacy, args.select_count, args.cal_count, args.seed
        )
        legacy, quarantine_receipt = quarantine_holdout_neighbors(
            legacy, [*select, *cal]
        )
        holdout_files = {
            "select.jsonl": jsonl_bytes(select),
            "cal.jsonl": jsonl_bytes(cal),
        }
        input_info = {
            "legacy_pool": {"sha256": source_sha, "rows": legacy_source_rows},
            "select_audit_only": {
                "sha256": sha_bytes(holdout_files["select.jsonl"]),
                "rows": len(select),
                "derived_from_legacy_pool": True,
            },
            "cal_audit_only": {
                "sha256": sha_bytes(holdout_files["cal.jsonl"]),
                "rows": len(cal),
                "derived_from_legacy_pool": True,
            },
        }
    else:
        select = read_reference(reference_paths[0], "select")
        cal = read_reference(reference_paths[1], "cal")
        input_info = {"legacy_train": {"sha256": source_sha, "rows": len(legacy)}}
        for role, path, rows in zip(
            ("select_audit_only", "cal_audit_only"), reference_paths, (select, cal)
        ):
            input_info[role] = {
                "sha256": sha_file(path),
                "rows": len(rows),
                "missing_split_rows": sum(row.get("split") is None for row in rows),
                "missing_evaluation_role_rows": sum(
                    row.get("evaluation_role") is None for row in rows
                ),
            }
    select_cal_audit = overlap_audit(select, cal)
    if audit_has_exact_overlap(select_cal_audit):
        raise ValueError("SELECT/CAL exact ID, group or input overlap")
    if (
        select_cal_audit["near_duplicate"]["count"]
        and args.near_duplicate_policy == "error"
    ):
        raise ValueError(
            "SELECT/CAL near duplicates; inspect references or use explicit report policy"
        )
    if derive:
        holdout_manifest = {
            "schema_version": SCHEMA,
            "partition": "group_reservation_from_legacy_pool_v1",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "legacy_pool_sha256": source_sha,
            "seed_sha256": sha_bytes(args.seed.encode()),
            "generator_code_sha256": sha_file(Path(__file__)),
            "train_pool_rows": len(legacy),
            "train_pool_groups": len({row["group_id"] for row in legacy}),
            "train_pool_quarantine": quarantine_receipt,
            "pre_partition_overlong_exclusions": overlong_receipt,
            "outputs": {
                name: {
                    "sha256": sha_bytes(raw),
                    "bytes": len(raw),
                    "rows": len(rows),
                    "groups": len({row["group_id"] for row in rows}),
                    "family_counts": dict(
                        sorted(
                            collections.Counter(row["family"] for row in rows).items()
                        )
                    ),
                    "task_type_counts": dict(
                        sorted(
                            collections.Counter(
                                row["task_type"] for row in rows
                            ).items()
                        )
                    ),
                    "language_counts": dict(
                        sorted(
                            collections.Counter(row["language"] for row in rows).items()
                        )
                    ),
                }
                for name, raw, rows in (
                    ("select.jsonl", holdout_files["select.jsonl"], select),
                    ("cal.jsonl", holdout_files["cal.jsonl"], cal),
                )
            },
            "source_attribution": attribution_for([*select, *cal]),
            "source_license_evidence_file": source_evidence_file,
            "select_cal_audit": select_cal_audit,
            "independence_scope": "group-held-out pilot monitoring only; source pool may have trained older models",
            "near_duplicate_policy": args.near_duplicate_policy,
        }
    maximum = [max(counts[i] for _, counts in arm_specs) for i in range(4)]
    selected_legacy = sample_legacy(legacy, maximum[0], args.seed)
    synthetic = {
        family: [
            GENERATORS[family](args.seed, index) for index in range(maximum[i + 1])
        ]
        for i, family in enumerate(FAMILIES)
    }
    if derive and tokenizer and holdout_manifest:
        for name, rows in (("select.jsonl", select), ("cal.jsonl", cal)):
            lengths = [count_tokens(row, tokenizer) for row in rows]
            for row, length in zip(rows, lengths):
                if args.max_row_tokens and length > args.max_row_tokens:
                    raise ValueError(
                        f"{name}: {row['id']} has {length} tokens, limit {args.max_row_tokens}"
                    )
            holdout_manifest["outputs"][name]["token_audit"] = {
                "method": "decoder-v2 segmented exact tokenizer",
                "tokenizer_revision": args.tokenizer_revision,
                "total": sum(lengths),
                "minimum": min(lengths),
                "maximum": max(lengths),
            }
    prepared = []
    for arm_index, (name, counts) in enumerate(arm_specs):
        new_rows = [
            row
            for i, family in enumerate(FAMILIES)
            for row in synthetic[family][: counts[i + 1]]
        ]
        if token_replacement and arm_index > 0:
            old_rows, selection = matched_legacy_subset(
                selected_legacy, counts[0], new_rows, length_for, args.seed
            )
        elif stratified_replacement and arm_index > 0:
            old_rows, selection = stratified_legacy_subset(
                selected_legacy, counts[0], new_rows, length_for, args.seed
            )
        else:
            old_rows = sample_legacy(selected_legacy, counts[0], args.seed)
            selection = {
                "strategy": "source-group-hash-sample-v1",
                "legacy_rows": len(old_rows),
            }
        rows = old_rows + new_rows
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate IDs")
        rng_for(args.seed, name, 0).shuffle(rows)
        for row in rows:
            validate_train_row(row)
        within_train = train_consistency_audit(rows)
        if within_train["conflicting_gold_groups"]:
            raise ValueError(f"{name}: identical inputs with conflicting labels")
        checks = {
            "train_select": overlap_audit(rows, select),
            "train_cal": overlap_audit(rows, cal),
            "select_cal": select_cal_audit,
        }
        for pair, result in checks.items():
            if audit_has_exact_overlap(result):
                raise ValueError(f"{name}: {pair} exact ID, group or input overlap")
            if (
                result["near_duplicate"]["count"]
                and args.near_duplicate_policy == "error"
            ):
                raise ValueError(
                    f"{name}: {pair} near duplicates; inspect or choose explicit report policy"
                )
        lengths = []
        if tokenizer:
            for row in rows:
                length = length_for(row)
                if args.max_row_tokens and length > args.max_row_tokens:
                    raise ValueError(
                        f"{name}: {row['id']} has {length} tokens, limit {args.max_row_tokens}"
                    )
                lengths.append(length)
            if args.max_arm_tokens and sum(lengths) > args.max_arm_tokens:
                raise ValueError(
                    f"{name}: {sum(lengths)} tokens exceeds limit {args.max_arm_tokens}"
                )
        payload = jsonl_bytes(rows)
        manifest = {
            "schema_version": SCHEMA,
            "generator_version": GENERATOR,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "arm": name,
            "arm_counts_requested": dict(zip(("legacy", *FAMILIES), counts)),
            "legacy_selection": selection,
            "seed_sha256": sha_bytes(args.seed.encode()),
            "generator_code_sha256": sha_file(Path(__file__)),
            "output": {
                "file": f"{name}.train.jsonl",
                "sha256": sha_bytes(payload),
                "bytes": len(payload),
                "rows": len(rows),
            },
            "inputs": input_info,
            "counts": {
                "family": dict(
                    sorted(collections.Counter(row["family"] for row in rows).items())
                ),
                "task_type": dict(
                    sorted(
                        collections.Counter(row["task_type"] for row in rows).items()
                    )
                ),
                "language": dict(
                    sorted(collections.Counter(row["language"] for row in rows).items())
                ),
                "source": dict(
                    sorted(collections.Counter(row["source"] for row in rows).items())
                ),
            },
            "source_attribution": attribution_for(rows),
            "source_license_evidence_file": source_evidence_file,
            "token_audit": {
                "method": (
                    "decoder-v2 segmented exact tokenizer"
                    if tokenizer
                    else "not measured"
                ),
                "tokenizer": args.tokenizer.name if args.tokenizer else None,
                "tokenizer_revision": args.tokenizer_revision if tokenizer else None,
                "total": sum(lengths) if tokenizer else None,
                "maximum": max(lengths) if lengths else None,
                "minimum": min(lengths) if lengths else None,
            },
            "leakage_audit": checks,
            "within_train_audit": within_train,
            "legacy_normalization": legacy_normalization_counts(rows),
            "near_duplicate_policy": args.near_duplicate_policy,
            "holdout_provenance": (
                "derived_from_legacy_pool" if derive else "external_audit_only"
            ),
            "limitations": [
                "Near-duplicate search is approximate and is not a proof of split independence.",
                "Programmatic text tasks do not establish real-world judgment quality.",
                "Legacy source rights and attribution require verification before publication.",
            ],
        }
        prepared.append((name, payload, manifest))
    if tokenizer and args.match_arm_tokens_percent is not None and len(prepared) > 1:
        reference_total = prepared[0][2]["token_audit"]["total"]
        for name, _, manifest in prepared[1:]:
            difference = (
                abs(manifest["token_audit"]["total"] - reference_total)
                / reference_total
            )
            if difference > args.match_arm_tokens_percent / 100:
                raise ValueError(
                    f"{name}: token total differs from first arm by {difference:.2%}"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = (
        [
            f"{name}.{suffix}"
            for name, _, _ in prepared
            for suffix in ("train.jsonl", "manifest.json")
        ]
        + list(holdout_files)
        + (["holdouts.manifest.json"] if holdout_manifest else [])
    )
    for name in output_names:
        file = output_dir / name
        if file.exists() and not args.overwrite:
            raise FileExistsError(f"{file}: use --overwrite explicitly")
    for name, raw in holdout_files.items():
        _atomic_write(output_dir / name, raw)
    if holdout_manifest:
        _atomic_write(
            output_dir / "holdouts.manifest.json",
            (
                json.dumps(holdout_manifest, ensure_ascii=False, indent=2) + "\n"
            ).encode(),
        )
    for name, payload, manifest in prepared:
        _atomic_write(output_dir / f"{name}.train.jsonl", payload)
        _atomic_write(
            output_dir / f"{name}.manifest.json",
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
        )
    return [manifest for _, _, manifest in prepared]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-train",
        type=Path,
        required=True,
        help="Existing 24k train JSONL; never a benchmark file",
    )
    parser.add_argument(
        "--select", type=Path, help="Flattened held-out SELECT JSONL, audit only"
    )
    parser.add_argument(
        "--cal", type=Path, help="Flattened held-out CAL JSONL, audit only"
    )
    parser.add_argument(
        "--derive-holdouts",
        action="store_true",
        help="Reserve SELECT/CAL groups from legacy pool",
    )
    parser.add_argument(
        "--select-count", type=int, help="Exact SELECT row count, with whole groups"
    )
    parser.add_argument(
        "--cal-count", type=int, help="Exact CAL row count, with whole groups"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="name:legacy:composition:reading:abstention; repeat for matched arms",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Local tokenizer for exact segmented input-token count",
    )
    parser.add_argument(
        "--tokenizer-revision",
        help="Exact tokenizer/model revision recorded in manifest",
    )
    parser.add_argument("--max-row-tokens", type=int)
    parser.add_argument("--max-arm-tokens", type=int)
    parser.add_argument("--match-arm-tokens-percent", type=float)
    parser.add_argument(
        "--token-match-replacements",
        action="store_true",
        help="Keep row count and replace singleton legacy groups with matched token cost",
    )
    parser.add_argument(
        "--stratified-replacements",
        action="store_true",
        help="Replace whole legacy groups in family/type/language proportions; report token imbalance",
    )
    parser.add_argument(
        "--source-license-evidence",
        type=Path,
        help="Private JSON source-to-license evidence; require coverage of every output source",
    )
    parser.add_argument(
        "--near-duplicate-policy", choices=("error", "report"), default="error"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for manifest in build(args):
        print(
            canonical(
                {
                    "arm": manifest["arm"],
                    "rows": manifest["output"]["rows"],
                    "sha256": manifest["output"]["sha256"],
                    "total_tokens": manifest["token_audit"]["total"],
                    "near_duplicates": {
                        pair: value["near_duplicate"]["count"]
                        for pair, value in manifest["leakage_audit"].items()
                    },
                }
            )
        )


if __name__ == "__main__":
    main()
