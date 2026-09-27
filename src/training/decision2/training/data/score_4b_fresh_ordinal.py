"""Build a fresh, source-disjoint three-level Score data *candidate*.

This CPU-only builder never loads benchmark answers or model predictions. Its
TRAIN rows and independent SELECT3 diagnostic remain HOLD until blind review.
The opaque seed and all produced rows stay in a private task directory.
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
import stat
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition, validate_row

VERSION = "decision2-4b-fresh-ordinal-candidate/1"
PARENT_SHA = {
    "train": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
PARENT_TOKENS = 4_194_465
TRAIN_GROUPS_PER_MECHANISM = 24
SELECT_GROUPS_PER_MECHANISM = 16

# The two SELECT mechanisms, source settings and rendering families do not
# occur in TRAIN. This is still a synthetic mechanism transfer diagnostic.
MECHANISMS = {
    "cold_chain_handoff": (
        "train",
        "peak temperature",
        "C",
        "upper",
        "handoff delay",
        "min",
        "upper",
    ),
    "dune_sample_release": (
        "train",
        "sample salinity",
        "ppt",
        "range",
        "signed witnesses",
        "people",
        "lower",
    ),
    "theatre_rigging": (
        "train",
        "lift load",
        "kg",
        "upper",
        "endorsed operators",
        "people",
        "lower",
    ),
    "seed_bank_storage": (
        "train",
        "germination reading",
        "percent",
        "lower",
        "warm storage",
        "hours",
        "upper",
    ),
    "harbour_survey": (
        "select",
        "channel depth",
        "m",
        "lower",
        "qualified crew",
        "people",
        "lower",
    ),
    "planetarium_event": (
        "select",
        "cloud estimate",
        "percent",
        "upper",
        "staffed stations",
        "stations",
        "lower",
    ),
}
SCENES = {
    "cold_chain_handoff": ("a refrigerated specimen transfer", "冷藏样本交接"),
    "dune_sample_release": ("a shoreline sample dispatch", "海岸样本转运"),
    "theatre_rigging": ("a touring stage setup", "巡演舞台搭建"),
    "seed_bank_storage": ("a seed-bank accession", "种子库入库"),
    "harbour_survey": ("a harbour survey departure", "港口测量出航"),
    "planetarium_event": ("a public observing session", "天文馆公众观测"),
}
CHINESE_FIELDS = {
    "peak temperature": "最高温度",
    "handoff delay": "交接耗时",
    "sample salinity": "样本盐度",
    "signed witnesses": "签字见证人数",
    "lift load": "吊装重量",
    "endorsed operators": "持证操作人数",
    "germination reading": "发芽率",
    "warm storage": "常温存放时长",
    "channel depth": "航道水深",
    "qualified crew": "合格船员人数",
    "cloud estimate": "云量预估",
    "staffed stations": "值守望远镜台数",
}
CHINESE_UNITS = {
    "C": "摄氏度",
    "min": "分钟",
    "ppt": "千分比",
    "people": "人",
    "kg": "千克",
    "percent": "%",
    "hours": "小时",
    "m": "米",
    "stations": "台",
}
LABEL_OPTIONS = {
    "en": (
        "Neither required check is met",
        "Exactly one required check is met",
        "Both required checks are met",
    ),
    "zh": ("两项必要核查均不满足", "仅有一项必要核查满足", "两项必要核查均满足"),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rng(seed: bytes, domain: str) -> random.Random:
    return random.Random(
        int(hmac.new(seed, domain.encode(), hashlib.sha256).hexdigest()[:16], 16)
    )


VALUE_RANGES = {
    "peak temperature": (2, 8),
    "handoff delay": (20, 90),
    "sample salinity": (20, 32),
    "signed witnesses": (2, 5),
    "lift load": (180, 760),
    "endorsed operators": (2, 6),
    "germination reading": (65, 94),
    "warm storage": (8, 60),
    "channel depth": (8, 32),
    "qualified crew": (3, 12),
    "cloud estimate": (15, 70),
    "staffed stations": (2, 8),
}


def _limits(
    rng: random.Random, mode: str, name: str
) -> tuple[tuple[int, int], int, int]:
    """Return a fixed rule plus one passing and one failing observation."""
    floor, ceiling = VALUE_RANGES[name]
    if mode == "range":
        low = rng.randint(floor + 2, ceiling - 7)
        high = low + rng.randint(4, 7)
        good = rng.choice((low, high, rng.randint(low, high)))
        bad = low - rng.randint(1, 4) if rng.randrange(2) else high + rng.randint(1, 4)
        return (low, high), good, bad
    threshold = rng.randint(floor + 1, ceiling - 1)
    gap = rng.randint(1, min(6, threshold - floor, ceiling - threshold))
    if mode == "upper":
        return (
            (threshold, threshold),
            rng.choice((threshold, threshold - gap)),
            threshold + gap,
        )
    if mode == "lower":
        return (
            (threshold, threshold),
            rng.choice((threshold, threshold + gap)),
            threshold - gap,
        )
    raise ValueError(mode)


def _criterion(value: int, limits: tuple[int, int], mode: str) -> bool:
    if mode == "upper":
        return value <= limits[1]
    if mode == "lower":
        return value >= limits[0]
    if mode == "range":
        return limits[0] <= value <= limits[1]
    raise ValueError(mode)


def _rule(name: str, unit: str, mode: str, limits: tuple[int, int], lang: str) -> str:
    if lang == "en":
        if mode == "range":
            return f"{name} must lie within {limits[0]}–{limits[1]} {unit}, including both ends."
        direction = "at most" if mode == "upper" else "at least"
        return f"{name} must be {direction} {limits[0]} {unit}."
    if mode == "range":
        return f"{name}须在 {limits[0]}–{limits[1]} {unit} 之间，含两个端点。"
    direction = "不超过" if mode == "upper" else "不少于"
    return f"{name}须{direction} {limits[0]} {unit}。"


def _render(
    mechanism: str,
    language: str,
    case: str,
    values: tuple[int, int],
    limits: tuple[tuple[int, int], tuple[int, int]],
    archive: tuple[int, int],
    *,
    select: bool,
    rng: random.Random,
) -> dict[str, Any]:
    _, name_a, unit_a, mode_a, name_b, unit_b, mode_b = MECHANISMS[mechanism]
    if language == "zh":
        name_a, name_b = CHINESE_FIELDS[name_a], CHINESE_FIELDS[name_b]
        unit_a, unit_b = CHINESE_UNITS[unit_a], CHINESE_UNITS[unit_b]
    scene = SCENES[mechanism][language == "zh"]
    if language == "en":
        current = (
            f"The current field sheet for {scene}, case {case}, records "
            f"{name_a} at {values[0]} {unit_a} and {name_b} at {values[1]} {unit_b}."
        )
        previous = (
            f"An earlier draft for the same case listed {name_a} at {archive[0]} {unit_a} "
            f"and {name_b} at {archive[1]} {unit_b}; the current field sheet replaces it."
        )
        policy = "Current release rule: " + " ".join(
            (
                _rule(name_a, unit_a, mode_a, limits[0], language),
                _rule(name_b, unit_b, mode_b, limits[1], language),
            )
        )
    else:
        current = (
            f"{scene}，案件 {case} 的现行现场记录载明：{name_a}为 {values[0]} {unit_a}，"
            f"{name_b}为 {values[1]} {unit_b}。"
        )
        previous = (
            f"同案旧草稿写着{name_a}为 {archive[0]} {unit_a}、{name_b}为 {archive[1]} {unit_b}；"
            "现行现场记录已取代这份草稿。"
        )
        policy = "现行放行规则：" + " ".join(
            (
                _rule(name_a, unit_a, mode_a, limits[0], language),
                _rule(name_b, unit_b, mode_b, limits[1], language),
            )
        )
    docs = [
        {"kind": "current", "text": current},
        {"kind": "archive", "text": previous},
        {"kind": "policy", "text": policy},
    ]
    rng.shuffle(docs)
    return {
        "case": case,
        "record_layout": "bulletin" if select else "field_sheet",
        "documents": docs,
    }


def _rendered_oracle(mechanism: str, state: dict[str, Any], language: str) -> int:
    """Read the displayed documents, separately from construction facts."""
    _, name_a, unit_a, mode_a, name_b, unit_b, mode_b = MECHANISMS[mechanism]
    if language == "zh":
        name_a, name_b = CHINESE_FIELDS[name_a], CHINESE_FIELDS[name_b]
        unit_a, unit_b = CHINESE_UNITS[unit_a], CHINESE_UNITS[unit_b]
    docs = {doc["kind"]: doc["text"] for doc in state["documents"]}
    if set(docs) != {"current", "archive", "policy"}:
        raise ValueError("Document identity missing or duplicated")
    values = []
    limits = []
    for name, unit, mode in ((name_a, unit_a, mode_a), (name_b, unit_b, mode_b)):
        if language == "en":
            match = re.search(
                rf"{re.escape(name)} at (\d+) {re.escape(unit)}", docs["current"]
            )
            if mode == "range":
                rule = re.search(
                    rf"{re.escape(name)} must lie within (\d+)–(\d+) {re.escape(unit)}",
                    docs["policy"],
                )
            else:
                phrase = "at most" if mode == "upper" else "at least"
                rule = re.search(
                    rf"{re.escape(name)} must be {phrase} (\d+) {re.escape(unit)}",
                    docs["policy"],
                )
        else:
            match = re.search(
                rf"{re.escape(name)}为 (\d+) {re.escape(unit)}", docs["current"]
            )
            if mode == "range":
                rule = re.search(
                    rf"{re.escape(name)}须在 (\d+)–(\d+) {re.escape(unit)}",
                    docs["policy"],
                )
            else:
                phrase = "不超过" if mode == "upper" else "不少于"
                rule = re.search(
                    rf"{re.escape(name)}须{phrase} (\d+) {re.escape(unit)}",
                    docs["policy"],
                )
        if match is None or rule is None:
            raise ValueError("Displayed evidence is not parseable")
        values.append(int(match[1]))
        limits.append(
            tuple(map(int, rule.groups()))
            if mode == "range"
            else (int(rule[1]), int(rule[1]))
        )
    return sum(
        _criterion(value, limit, mode)
        for value, limit, mode in zip(
            (values[0], values[1]), (limits[0], limits[1]), (mode_a, mode_b)
        )
    )


def generate(seed: bytes) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(seed) != 32:
        raise ValueError("A private 32-byte seed is required")
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "select": []}
    for mechanism, spec in MECHANISMS.items():
        split, name_a, _, mode_a, name_b, _, mode_b = spec
        n_groups = (
            TRAIN_GROUPS_PER_MECHANISM
            if split == "train"
            else SELECT_GROUPS_PER_MECHANISM
        )
        for index in range(n_groups):
            rng = _rng(seed, f"{VERSION}/{split}/{mechanism}/{index}")
            language = "zh" if index % 4 == 3 else "en"
            case = (
                hmac.new(
                    seed,
                    f"case/{VERSION}/{split}/{mechanism}/{index}".encode(),
                    hashlib.sha256,
                )
                .hexdigest()[:12]
                .upper()
            )
            limit_a, good_a, bad_a = _limits(rng, mode_a, name_a)
            limit_b, good_b, bad_b = _limits(rng, mode_b, name_b)
            archive = (good_a, bad_b) if index % 2 else (bad_a, good_b)
            first = 0 if index % 2 == 0 else 1
            settings = ((False, False), (first == 0, first == 1), (True, True))
            group = (
                f"d2-4b-s3-{_sha((mechanism + str(index) + _sha(seed)).encode())[:20]}"
            )
            options = [
                {"key": str(level), "description": description}
                for level, description in enumerate(LABEL_OPTIONS[language])
            ]
            instructions = (
                "Use only the current field sheet and the current release rule. "
                "Count how many of the two required checks are met, then choose 0, 1 or 2."
                if language == "en"
                else "只按现行现场记录和现行放行规则判断，数出两项必要核查中有几项满足，再选择 0、1 或 2。"
            )
            for level, flags in enumerate(settings):
                values = (
                    good_a if flags[0] else bad_a,
                    good_b if flags[1] else bad_b,
                )
                if (
                    sum(
                        (
                            _criterion(values[0], limit_a, mode_a),
                            _criterion(values[1], limit_b, mode_b),
                        )
                    )
                    != level
                ):
                    raise AssertionError("Construction oracle mismatch")
                state = _render(
                    mechanism,
                    language,
                    case,
                    values,
                    (limit_a, limit_b),
                    archive,
                    select=split == "select",
                    rng=_rng(seed, f"{group}/{level}/render"),
                )
                if _rendered_oracle(mechanism, state, language) != level:
                    raise AssertionError("Independent displayed-text oracle mismatch")
                row = {
                    "id": f"{group}-{level}",
                    "group_id": group,
                    "state": state,
                    "instructions": instructions,
                    "options": options,
                    "label": level,
                    "task_type": "score",
                    "family": f"score_{mechanism}",
                    "language": language,
                    "split": split,
                    "source": "decision2_original_4b_score_fresh_v1",
                    "evaluation_role": split,
                    "render_template": f"score_4b_fresh_{split}_{mechanism}_{language}",
                    "audit_metadata": {
                        "generation": VERSION,
                        "seed_sha256": _sha(seed),
                        "source_group_ordinal": index,
                        "variant": level,
                    },
                }
                row["input_sha256"] = pilot.input_sha256(row)
                validate_row(row, split)
                outputs[split].append(row)
    for split, rows in outputs.items():
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            groups[row["group_id"]].append(row)
        for triplet in groups.values():
            if len(triplet) != 3 or {r["label"] for r in triplet} != {0, 1, 2}:
                raise AssertionError("Incomplete ordinal triplet")
            # No one criterion may encode all three labels in a group.
            for slot in (0, 1):
                reports = [
                    next(
                        doc["text"]
                        for doc in r["state"]["documents"]
                        if doc["kind"] == "current"
                    )
                    for r in triplet
                ]
                _, na, ua, _, nb, ub, _ = MECHANISMS[triplet[0]["family"][6:]]
                if triplet[0]["language"] == "zh":
                    na, nb = CHINESE_FIELDS[na], CHINESE_FIELDS[nb]
                    ua, ub = CHINESE_UNITS[ua], CHINESE_UNITS[ub]
                name, unit = (na, ua) if slot == 0 else (nb, ub)
                pattern = (
                    rf"{re.escape(name)}为 (\d+) {re.escape(unit)}"
                    if triplet[0]["language"] == "zh"
                    else rf"{re.escape(name)} at (\d+) {re.escape(unit)}"
                )
                matches = [re.search(pattern, report) for report in reports]
                if any(match is None for match in matches):
                    raise AssertionError("Rendered source field is not parseable")
                features = [match[1] for match in matches if match is not None]
                if len(set(features)) != 2:
                    raise AssertionError(
                        "Single criterion has a three-way value shortcut"
                    )
        rows.sort(
            key=lambda r: (_sha(f"{VERSION}/{split}/{r['id']}".encode()), r["id"])
        )
    if len(outputs["train"]) != 288 or len(outputs["select"]) != 96:
        raise AssertionError("Unexpected fixed data size")
    check_partition_isolation(outputs)
    return outputs["train"], outputs["select"]


def _private_seed(path: Path) -> bytes:
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError("Seed file must have mode 0600")
    value = path.read_bytes()
    if len(value) != 32:
        raise ValueError("Seed must contain exactly 32 bytes")
    return value


def _write(path: Path, body: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)


def _load_goldfree_inventory(
    path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Gold-free protected inventory must be a nonempty list")
    references: dict[str, list[dict[str, Any]]] = {}
    receipts = []
    for item in inventory:
        role = item["role"]
        source_path = Path(item["path"])
        if role in references or not isinstance(role, str):
            raise ValueError("Duplicate or invalid protected role")
        if source_path.name not in {
            "prompts.jsonl",
            "packet.jsonl",
            "originals.jsonl",
            "variants.jsonl",
        } and not source_path.name.endswith(".prompts.jsonl"):
            raise ValueError("Protected input must be a gold-free prompt file")
        actual = pilot.sha_file(source_path)
        if actual != item["sha256"]:
            raise ValueError(f"Protected prompt hash mismatch: {role}")
        rows = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            allowed = [
                {"id", "state", "questions"},
                {
                    "family",
                    "group_id",
                    "id",
                    "instructions",
                    "language",
                    "options",
                    "state",
                },
                {
                    "family",
                    "group_id",
                    "review_id",
                    "instructions",
                    "language",
                    "options",
                    "state",
                },
            ]
            if set(obj) not in allowed or not isinstance(obj["state"], (dict, str)):
                raise ValueError(f"Protected input contains non-prompt fields: {role}")
            rows.append(
                {
                    "id": obj.get("id", obj.get("review_id")),
                    "group_id": obj.get("group_id"),
                    "input_sha256": None,
                    "state": obj["state"],
                    "instructions": obj.get(
                        "instructions", pilot.canonical(obj.get("questions"))
                    ),
                    "options": obj.get("options", []),
                    "task_type": "context",
                }
            )
        references[role] = rows
        receipts.append({"role": role, "sha256": actual, "rows": len(rows)})
    if not {
        "typed_dev",
        "css_pilot",
        "typed_final_goldfree",
        "css15_goldfree",
        "jevbench_public231",
    } <= set(references):
        raise ValueError("Core protected prompt roles are missing")
    return references, sorted(receipts, key=lambda x: x["role"])


ELIGIBLE_CHOICE_FAMILIES = (
    "stage4_arithmetic",
    "stage4_scope",
    "stage4_registers",
    "stage4_relations",
    "stage4_dense_table",
    "stage4_automaton",
)


def _replacement_feasibility(
    parent: list[dict[str, Any]], candidate: list[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    """Count a matched TRAIN-only row swap without using labels or model scores."""
    old_by_family: dict[str, list[tuple[dict[str, Any], int]]] = (
        collections.defaultdict(list)
    )
    for row in parent:
        if row["task_type"] == "choice" and row["family"] in ELIGIBLE_CHOICE_FAMILIES:
            if row["source"] != "legacy:stage4-general-composition-v2":
                raise ValueError("Eligible Choice row has unexpected original source")
            old_by_family[row["family"]].append(
                (row, pilot.count_tokens(row, tokenizer))
            )
    pool_counts = {name: len(old_by_family[name]) for name in ELIGIBLE_CHOICE_FAMILIES}
    if [pool_counts[name] for name in ELIGIBLE_CHOICE_FAMILIES] != [
        224,
        197,
        178,
        168,
        110,
        99,
    ]:
        raise ValueError("Eligible synthetic Choice source roster changed")
    if any(
        len({r["group_id"] for r, _ in old_by_family[name]}) != pool_counts[name]
        for name in ELIGIBLE_CHOICE_FAMILIES
    ):
        raise ValueError("Choice swap would split an original source group")
    quotas = (66, 58, 53, 50, 32, 29)
    needed = sum(pilot.count_tokens(row, tokenizer) for row in candidate)
    target_per_row = needed / len(candidate)
    selected: dict[str, tuple[dict[str, Any], int]] = {}
    remainder: dict[str, list[tuple[dict[str, Any], int]]] = {}
    for name, quota in zip(ELIGIBLE_CHOICE_FAMILIES, quotas):
        ordered = sorted(
            old_by_family[name],
            key=lambda item: (
                abs(item[1] - target_per_row),
                _sha(item[0]["id"].encode()),
            ),
        )
        for row, tokens in ordered[:quota]:
            selected[row["id"]] = (row, tokens)
        remainder[name] = ordered[quota:]
    removed_tokens = sum(tokens for _, tokens in selected.values())
    # A one-for-one swap is allowed only within the frozen source family quota.
    for _ in range(64):
        old_error = abs(removed_tokens - needed)
        best = None
        for name in ELIGIBLE_CHOICE_FAMILIES:
            selected_name = [
                entry for entry in selected.values() if entry[0]["family"] == name
            ]
            for old_row, old_tokens in selected_name:
                for new_row, new_tokens in remainder[name]:
                    error = abs(removed_tokens - old_tokens + new_tokens - needed)
                    candidate_swap = (
                        error,
                        old_row["id"],
                        new_row["id"],
                        old_tokens,
                        new_tokens,
                    )
                    if error < old_error and (best is None or candidate_swap < best):
                        best = candidate_swap
        if best is None:
            break
        _, old_id, new_id, old_tokens, new_tokens = best
        name = selected[old_id][0]["family"]
        old_entry = selected.pop(old_id)
        new_entry = next(item for item in remainder[name] if item[0]["id"] == new_id)
        remainder[name].remove(new_entry)
        remainder[name].append(old_entry)
        selected[new_id] = new_entry
        removed_tokens += new_tokens - old_tokens
    delta = needed - removed_tokens
    feasible = len(selected) == len(candidate) and abs(delta) * 200 <= PARENT_TOKENS
    receipt = {
        "eligible_family_counts": pool_counts,
        "replacement_quotas": dict(zip(ELIGIBLE_CHOICE_FAMILIES, quotas)),
        "selected_rows": len(selected),
        "candidate_tokens": needed,
        "removed_tokens": removed_tokens,
        "delta_tokens": delta,
        "relative_change": delta / PARENT_TOKENS,
        "within_0_5pct": feasible,
        "chosen_id_roster_sha256": _sha("\n".join(sorted(selected)).encode()),
    }
    if not feasible:
        return None, receipt
    additions = iter(candidate)
    merged = [next(additions) if row["id"] in selected else row for row in parent]
    if next(additions, None) is not None or len(merged) != len(parent):
        raise AssertionError("Token-matched row replacement was incomplete")
    if [
        (r["id"], r["task_type"])
        for r in merged
        if r["task_type"] != "score" and r["id"] not in {x["id"] for x in candidate}
    ] != [
        (r["id"], r["task_type"])
        for r in parent
        if r["id"] not in selected and r["task_type"] != "score"
    ]:
        raise AssertionError("Unselected Choice or Noul records changed")
    return merged, receipt


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for role, path in (
        ("train", args.parent_train),
        ("select", args.parent_select),
        ("cal", args.parent_cal),
    ):
        if pilot.sha_file(path) != PARENT_SHA[role]:
            raise ValueError(f"Frozen parent {role} bytes changed")
    parent = load_partition(args.parent_train, "train")
    select = load_partition(args.parent_select, "select")
    cal = load_partition(args.parent_cal, "cal")
    if tuple(map(len, (parent, select, cal))) != (7455, 700, 700):
        raise ValueError("Parent partition size changed")
    seed = _private_seed(args.seed_file)
    train_rows, select3_rows = generate(seed)
    references, protected_receipts = _load_goldfree_inventory(args.protected_inventory)
    overlap = {}
    for role, rows in (
        ("parent_train", parent),
        ("parent_select", select),
        ("parent_cal", cal),
        *sorted(references.items()),
    ):
        for candidate_role, candidate in (
            ("train", train_rows),
            ("select3", select3_rows),
        ):
            exact = targeted.context_overlap(candidate, rows, approximate=True)
            near_full = pilot.near_duplicates(candidate, rows)
            if near_full["count"]:
                raise ValueError(f"Near-complete {candidate_role} overlap with {role}")
            overlap[f"{candidate_role}/{role}"] = {
                "exact": {
                    key: value for key, value in exact.items() if key != "near_context"
                },
                "near_context": exact["near_context"]["count"],
                "near_full": near_full["count"],
            }
    if pilot.near_duplicates(train_rows, select3_rows)["count"]:
        raise ValueError("TRAIN and SELECT3 contain near-complete prompts")
    for role, rows in (("train", train_rows), ("select3", select3_rows)):
        lengths = [pilot.count_tokens(row, args.tokenizer) for row in rows]
        if max(lengths) > 8192:
            raise ValueError(f"{role} input exceeds the native 8192-token cap")
        overlap[role + "/token_summary"] = {
            "rows": len(lengths),
            "total": sum(lengths),
            "min": min(lengths),
            "max": max(lengths),
        }
    parent_tokens = sum(pilot.count_tokens(row, args.tokenizer) for row in parent)
    if parent_tokens != PARENT_TOKENS:
        raise ValueError("Pinned Qwen3.5-4B native TRAIN token budget changed")
    merged, replacement = _replacement_feasibility(parent, train_rows, args.tokenizer)
    if merged is not None:
        check_partition_isolation({"train": merged, "select": select, "cal": cal})
    # Do not spend SELECT3 on checkpoint selection. It is evaluated once on
    # each SELECT-selected BEST after an independently admitted training arm.
    from training.data import build_pilot

    blind = []
    key = []
    for row in select3_rows:
        alias = hmac.new(
            seed, f"blind/{row['id']}".encode(), hashlib.sha256
        ).hexdigest()[:24]
        group_alias = hmac.new(
            seed, f"blind-group/{row['group_id']}".encode(), hashlib.sha256
        ).hexdigest()[:24]
        blind.append(
            {
                "review_id": alias,
                "group_id": group_alias,
                "family": row["family"],
                "language": row["language"],
                "state": row["state"],
                "instructions": row["instructions"],
                "options": row["options"],
            }
        )
        key.append(
            {
                "review_id": alias,
                "row_id": row["id"],
                "group_id": row["group_id"],
                "label": row["label"],
            }
        )
    outputs = {
        "train-candidate.jsonl": build_pilot.jsonl_bytes(train_rows),
        "select3-candidate.jsonl": build_pilot.jsonl_bytes(select3_rows),
        "select3-blind.jsonl": build_pilot.jsonl_bytes(blind),
        "select3-sealed-key.jsonl": build_pilot.jsonl_bytes(key),
    }
    if merged is not None:
        outputs["train-proposed-unapproved.jsonl"] = build_pilot.jsonl_bytes(merged)
    args.output_dir.mkdir(parents=True, mode=0o700)
    for name, body in outputs.items():
        _write(args.output_dir / name, body)
    manifest = {
        "version": VERSION,
        "status": "HOLD_PENDING_INDEPENDENT_BLIND_REVIEW",
        "training_approved": False,
        "model_gate_approved": False,
        "seed_sha256": _sha(seed),
        "tokenizer_revision": args.tokenizer_revision,
        "parent_train_sha256": PARENT_SHA["train"],
        "parent_tokens": parent_tokens,
        "candidate_token_summary": {
            key: value
            for key, value in overlap.items()
            if key.endswith("/token_summary")
        },
        "replacement_feasibility": replacement,
        "overlap": {
            key: value
            for key, value in overlap.items()
            if not key.endswith("/token_summary")
        },
        "protected_sources": protected_receipts,
        "output_sha256": {name: _sha(body) for name, body in outputs.items()},
        "limitations": [
            "Synthetic source cases and surface variations do not establish real-task transfer.",
            "Train and SELECT3 have separate sources and templates but share a two-criterion ordinal abstraction.",
            "No independent blind answerability, bilingual, triplet or shortcut review has passed.",
            "No eligible replacement roster or matched 7455-row/466-update token budget has been frozen.",
        ],
    }
    _write(
        args.output_dir / "manifest.json",
        (
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-train", type=Path, required=True)
    parser.add_argument("--parent-select", type=Path, required=True)
    parser.add_argument("--parent-cal", type=Path, required=True)
    parser.add_argument("--protected-inventory", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--seed-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    args.tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    report = build(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "train": report["candidate_token_summary"]["train/token_summary"],
                "select3": report["candidate_token_summary"]["select3/token_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
