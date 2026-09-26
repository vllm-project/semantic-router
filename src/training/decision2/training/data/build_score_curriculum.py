"""Build a TRAIN-only, oracle checked three-level Score curriculum.

No benchmark generator or held-out gold is imported. A protected inventory
contains only prompt files; matching source groups are quarantined in full.
The output is a research candidate, not an approved training release.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

SEED = "decision20-score-curriculum-v1-20260927"
SOURCE = "decision2_internal_score_curriculum_v1"
FAMILIES = ("obligation_review", "weighted_points", "route_depth", "timely_streak")
GROUPS_PER_FAMILY = 80
EXPECTED_ROWS = len(FAMILIES) * GROUPS_PER_FAMILY * 3
FROZEN = {
    "base": "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    "base_manifest": "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
    "select": "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    "cal": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
}
REQUIRED_PROTECTED_ROLES = {"typed_dev", "css_pilot", "css15_goldfree"}
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
        statuses = [x["assessment"] for x in state["reviews"] if x["scope"] == "core"]
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
            (x for x in state["days"] if 1 <= x["day"] <= 5), key=lambda x: x["day"]
        )
        if [x["day"] for x in records] != [1, 2, 3, 4, 5]:
            raise ValueError("timely_streak requires exactly days 1 through 5")
        best = streak = 0
        for record in records:
            streak = streak + 1 if record["on_time"] else 0
            best = max(best, streak)
        return 0 if best <= 1 else (1 if best <= 3 else 2)
    raise ValueError(f"Unknown Score family {family}")


def _obligation_states(
    rng: random.Random, case: str, language: str
) -> list[dict[str, Any]]:
    words = (
        ["identity", "access", "evidence", "acknowledgement"]
        if language == "en"
        else ["身份", "访问", "证明", "确认"]
    )
    core = [{"name": word, "scope": "core", "assessment": "accepted"} for word in words]
    optional = {
        "name": "archive" if language == "en" else "归档",
        "scope": "informational",
        "assessment": "rejected",
    }
    order = list(range(len(core) + 1))
    rng.shuffle(order)
    states = []
    for level in range(3):
        reviews = [dict(item) for item in [*core, optional]]
        if level == 0:
            reviews[0]["assessment"] = "rejected"
            reviews[1]["assessment"] = "unresolved"
        elif level == 1:
            reviews[1]["assessment"] = "unresolved"
        states.append({"case": case, "reviews": [reviews[i] for i in order]})
    return states


def _weighted_states(
    rng: random.Random, case: str, language: str
) -> list[dict[str, Any]]:
    words = (
        ["clarity", "coverage", "relevance", "traceability"]
        if language == "en"
        else ["清晰度", "覆盖度", "相关性", "可追溯性"]
    )
    weights = [1, 2, 2, 3]
    rng.shuffle(weights)
    lower, upper = rng.randint(8, 10), rng.randint(15, 17)
    states = []
    for level in range(3):
        for _ in range(1024):
            marks = [rng.randint(0, 3) for _ in range(4)]
            points = sum(w * m for w, m in zip(weights, marks))
            actual = 0 if points < lower else (1 if points < upper else 2)
            if actual == level:
                break
        else:
            raise AssertionError("Could not sample weighted score bucket")
        signals = [
            {"name": name, "weight": weight, "mark": mark}
            for name, weight, mark in zip(words, weights, marks)
        ]
        rng.shuffle(signals)
        states.append(
            {"case": case, "signals": signals, "lower": lower, "upper": upper}
        )
    return states


def _route_states(rng: random.Random, case: str) -> list[dict[str, Any]]:
    nodes = [f"N{rng.randrange(1000, 9999)}" for _ in range(7)]
    if len(set(nodes)) != len(nodes):
        return _route_states(rng, case)
    start, via1, via2, finish, decoy1, decoy2, decoy3 = nodes
    fixed = [
        {"from": start, "to": via1},
        {"from": via1, "to": via2},
        {"from": decoy1, "to": decoy2},
        {"from": decoy2, "to": decoy3},
    ]
    states = []
    for level in range(3):
        links = [dict(item) for item in fixed]
        if level >= 1:
            links.append({"from": via2, "to": finish})
        if level == 2:
            links.append({"from": via1, "to": finish})
        rng.shuffle(links)
        states.append({"case": case, "start": start, "finish": finish, "links": links})
    return states


def _streak_states(rng: random.Random, case: str) -> list[dict[str, Any]]:
    schedules = (
        (False, True, False, True, False),
        (False, True, True, True, False),
        (False, True, True, True, True),
    )
    states = []
    for schedule in schedules:
        days = [{"day": 0, "on_time": True}]
        days.extend(
            {"day": day, "on_time": value} for day, value in enumerate(schedule, 1)
        )
        rng.shuffle(days)
        states.append({"case": case, "days": days})
    return states


def _question(family: str, language: str) -> tuple[str, list[dict[str, str]]]:
    if family == "obligation_review":
        instructions = (
            "Rate core reviews only: any rejected core review gives level 0; otherwise an unresolved core review gives level 1; all accepted core reviews give level 2. Informational reviews do not count."
            if language == "en"
            else "只评估核心审核：任一核心审核被拒绝为 0 档；否则有未解决的核心审核为 1 档；全部核心审核已接受为 2 档。信息性审核不计入。"
        )
        labels = (
            (
                "A core review is rejected",
                "No core rejection, but one is unresolved",
                "All core reviews are accepted",
            )
            if language == "en"
            else ("核心审核被拒绝", "无核心拒绝但仍有未解决项", "全部核心审核已接受")
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
                "Total within the middle band",
                "Total at or above upper cutoff",
            )
            if language == "en"
            else ("总分低于下界", "总分位于中档", "总分达到或高于上界")
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
            "Consider numbered days 1 through 5 only, in day order. Find the longest consecutive streak marked on_time. Give level 0 for a streak of at most one day, level 1 for two or three days, and level 2 for four or five days. Ignore day 0."
            if language == "en"
            else "仅按日期顺序查看第 1 至 5 天，找出按时完成的最长连续天数。最长为 0 或 1 天是 0 档，2 或 3 天是 1 档，4 或 5 天是 2 档。忽略第 0 天。"
        )
        labels = (
            (
                "Longest on-time streak is at most one",
                "Longest streak is two or three",
                "Longest streak is four or five",
            )
            if language == "en"
            else ("最长按时连续天数至多一天", "最长连续两到三天", "最长连续四到五天")
        )
    else:
        raise ValueError(family)
    return instructions, [
        {"key": str(index), "description": value} for index, value in enumerate(labels)
    ]


def generate() -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        for index in range(GROUPS_PER_FAMILY):
            rng = _rng(family, index)
            language = "zh" if index % 4 == 0 else "en"
            case = _case(rng, language)
            if family == "obligation_review":
                states = _obligation_states(rng, case, language)
            elif family == "weighted_points":
                states = _weighted_states(rng, case, language)
            elif family == "route_depth":
                states = _route_states(rng, case)
            else:
                states = _streak_states(rng, case)
            instructions, options = _question(family, language)
            stem = _sha(f"{SEED}\0{family}\0{index}")[:20]
            for level, state in enumerate(states):
                if oracle(family, state) != level:
                    raise AssertionError(
                        f"Oracle disagrees with construction: {family}/{index}/{level}"
                    )
                row = {
                    "id": f"d2sc_{stem}_{level}",
                    "state": state,
                    "instructions": instructions,
                    "options": options,
                    "label": level,
                    "task_type": "score",
                    "family": f"score_{family}",
                    "group_id": f"d2scg_{stem}",
                    "language": language,
                    "split": "train",
                    "source": SOURCE,
                    "evaluation_role": "train",
                    "render_template": f"score_curriculum_{family}_v1",
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
        if "gold" in file.name.lower() or not file.name.endswith(".prompts.jsonl"):
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
        "schema_version": "decision20-score-curriculum/1",
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
