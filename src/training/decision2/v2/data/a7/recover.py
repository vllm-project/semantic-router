"""Recover the held Stage1/Stage2 Noul and Score rows as sub-arm A7r (rule 7e).

Rule 7b held every Stage1/Stage2 Noul and Score row whose option keys are
opaque (`result_<n>` in construction order, or random words). Rule 7e
(`records/a7-prereg-amendment-3-2026-09-28.md`) maps them to the 2.0 key
contract from the option descriptions alone, never from the label:

- Noul: the two descriptions must be one of the whitelisted (affirmative,
  negative) pairs below; the affirmative option is keyed `true`, the negative
  `false`; option order and label index are unchanged.
- Score: every description must state its level and condition range
  ("Level k: between a and b conditions hold, inclusive." or the Chinese
  template); levels must be exactly 0..K-1 with contiguous ranges from 0. The
  options are reordered by level and keyed "0".."K-1"; the label follows its
  option. The gold range must contain the number of true conditions in the
  state (an independent recomputation); disagreements are dropped and counted.

Rows keep the 2.0 contract of `build_a7.normalize_row`. They are deduplicated
against the frozen A7 files and among themselves, never cross SELECT/CAL or a
v1 held-out slice, and a row whose connected component (the same union-find as
`build_a7.build`, over all source rows) already has frozen A7 rows inherits
their partition; other components follow the AHO hash rule. Writes
prelim/A7r.{train,aho}.jsonl and a count-only build manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, validate_row
from v2.data.a7.build_a7 import (
    Components,
    Excluded,
    _external_sets,
    _json_bytes,
    _write_new,
    input_hash,
    is_aho,
    keyed,
    normalize_row,
    read_pinned,
    rekeyed_hash,
    state_key,
)
from v2.data.freeze import canonical_jsonl, parse_jsonl

VERSION = "a7-rec10-v1"
SUB_ARM = "A7r"
RECOVERED_SOURCES = ("stage1", "stage2")
NOUL_PAIRS = {
    frozenset(
        {"The action is valid.", "The action is not valid."}
    ): "The action is valid.",
    frozenset({"操作有效", "操作无效"}): "操作有效",
    frozenset(
        {
            "The final set contains the target.",
            "The final set does not contain the target.",
        }
    ): "The final set contains the target.",
    frozenset({"最终集合包含目标", "最终集合不包含目标"}): "最终集合包含目标",
    frozenset(
        {"The rule holds; approve.", "The rule does not hold; deny."}
    ): "The rule holds; approve.",
    frozenset({"满足规则，允许", "不满足规则，拒绝"}): "满足规则，允许",
}
LEVEL = re.compile(
    r"^(?:Level (\d+): between (\d+) and (\d+) conditions hold, inclusive\."
    r"|等级 (\d+)：成立的条件数为 (\d+) 到 (\d+)（含边界）。)$"
)


def _level(description: Any) -> tuple[int, int, int] | None:
    match = LEVEL.match(description) if isinstance(description, str) else None
    if match is None:
        return None
    groups = [g for g in match.groups() if g is not None]
    return int(groups[0]), int(groups[1]), int(groups[2])


def true_count(state: Any) -> int | None:
    try:
        values = json.loads(state) if isinstance(state, str) else state
    except json.JSONDecodeError:
        return None
    if not isinstance(values, dict) or not all(
        isinstance(v, bool) for v in values.values()
    ):
        return None
    return sum(values.values())


def map_keys(row: Mapping[str, Any]) -> dict[str, Any]:
    """Rule 7e: 1.0 row with opaque Noul/Score keys -> row with contract keys."""
    options = row["options"]
    original = [option["key"] for option in options]
    if row["task_type"] == "noul":
        descriptions = [option.get("description") for option in options]
        affirmative = NOUL_PAIRS.get(
            frozenset(d for d in descriptions if isinstance(d, str))
        )
        if len(options) != 2 or affirmative is None or len(set(descriptions)) != 2:
            raise Excluded("rule7e_unmapped_noul")
        new_options = [
            dict(
                option, key="true" if option["description"] == affirmative else "false"
            )
            for option in options
        ]
        return {
            **row,
            "options": new_options,
            "_a7_rule7e": {"original_keys": original},
        }
    if row["task_type"] == "score":
        levels = [_level(option.get("description")) for option in options]
        if any(level is None for level in levels):
            raise Excluded("rule7e_unmapped_score")
        order = sorted(range(len(options)), key=lambda index: levels[index][0])
        ranked = [levels[index] for index in order]
        if [level for level, _, _ in ranked] != list(range(len(options))):
            raise Excluded("rule7e_levels_not_0_to_k")
        expected_low = 0
        for _, low, high in ranked:
            if low != expected_low or high < low:
                raise Excluded("rule7e_ranges_not_contiguous")
            expected_low = high + 1
        count = true_count(row["state"])
        gold_low, gold_high = levels[row["label"]][1:]
        if count is None:
            raise Excluded("rule7e_state_unparsed")
        if not gold_low <= count <= gold_high:
            raise Excluded("rule7e_oracle_disagrees")
        new_options = [
            dict(options[index], key=str(rank)) for rank, index in enumerate(order)
        ]
        return {
            **row,
            "options": new_options,
            "label": order.index(row["label"]),
            "_a7_rule7e": {"original_keys": original, "original_order": order},
        }
    raise Excluded("rule7e_not_applicable")


def recover_row(row: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    original_hash = input_hash(row)
    mapped = map_keys(row)
    rule = mapped.pop("_a7_rule7e")
    out = normalize_row(mapped, source["name"], source["sha256"])
    origin = out["audit_metadata"]["a7"]
    if origin["sub_arm"] != "A7o":
        raise Excluded("rule7e_unexpected_sub_arm")
    origin.update(
        version=VERSION,
        sub_arm=SUB_ARM,
        rule_7e=rule,
        original_input_sha256=original_hash,
    )
    validate_row(out, "train")
    return out


def _frozen(
    final_dirs: Sequence[Path],
) -> tuple[dict[str, str], dict[str, str], list[tuple[str, str, str | None]]]:
    """Frozen A7 rows: input hash -> part, group id -> part, (group, input, state) links."""
    inputs: dict[str, str] = {}
    groups: dict[str, str] = {}
    links: list[tuple[str, str, str | None]] = []
    for directory in final_dirs:
        for path in sorted(directory.glob("*.jsonl")):
            part = path.name.split(".")[-2]
            for row in parse_jsonl(path.read_bytes(), str(path)):
                inputs[row["input_sha256"]] = part
                groups[row["group_id"]] = part
                links.append(
                    (row["group_id"], row["input_sha256"], state_key(row["state"]))
                )
    return inputs, groups, links


def recover(spec: Mapping[str, Any], final_dirs: Sequence[Path]) -> dict[str, Any]:
    frozen_inputs, frozen_groups, frozen_links = _frozen(final_dirs)
    link_inputs: dict[str, list[str]] = collections.defaultdict(list)
    link_states: dict[str, list[str]] = collections.defaultdict(list)
    for group, digest_, key in frozen_links:
        link_inputs[digest_].append(group)
        if key is not None:
            link_states[key].append(group)
    excluded: collections.Counter[tuple[str, ...]] = collections.Counter()
    kept: dict[str, dict[str, Any]] = {}
    labels: dict[str, set[str]] = collections.defaultdict(set)
    duplicates: collections.Counter[str] = collections.Counter()
    loaded = {}
    for entry in spec["sources"]:
        rows = read_pinned(entry)
        loaded[entry["name"]] = len(rows)
        for row in rows:
            group = row.get("group_id")
            if isinstance(group, str) and all(field in row for field in INPUT_FIELDS):
                link_inputs[rekeyed_hash(row)].append(group)
                key = state_key(row["state"])
                if key is not None:
                    link_states[key].append(group)
            if entry["name"] not in RECOVERED_SOURCES or row.get("task_type") not in (
                "noul",
                "score",
            ):
                continue
            try:
                normalize_row(row, entry["name"], entry["sha256"])
                continue
            except Excluded as reason:
                if reason.reason not in ("opaque_noul_keys", "opaque_score_keys"):
                    continue
            try:
                out = recover_row(row, entry)
            except Excluded as reason:
                excluded[
                    (entry["name"], reason.reason, row["family"], row["task_type"])
                ] += 1
                continue
            digest_ = out["input_sha256"]
            labels[digest_].add(canonical(out["options"][out["label"]]))
            if digest_ in frozen_inputs:
                duplicates["already_in_frozen_a7"] += 1
            elif digest_ in kept:
                duplicates["within_recovered"] += 1
            else:
                kept[digest_] = out
    conflicts = {digest_ for digest_, values in labels.items() if len(values) > 1}
    for digest_ in conflicts:
        if kept.pop(digest_, None) is not None:
            duplicates["label_conflict_dropped"] += 1

    holdout_groups, holdout_inputs, holdout_states = _external_sets(
        spec.get("isolation_holdout", [])
    )
    train_groups, train_inputs, train_states = _external_sets(
        spec.get("isolation_train", [])
    )
    isolation_dropped: collections.Counter[str] = collections.Counter()
    for digest_ in list(kept):
        row = kept[digest_]
        if (
            digest_ in holdout_inputs
            or row["group_id"] in holdout_groups
            or state_key(row["state"]) in holdout_states
        ):
            isolation_dropped[row["family"]] += 1
            del kept[digest_]

    components = Components()
    components.link(link_inputs)
    components.link(link_states)
    inherited: dict[str, set[str]] = collections.defaultdict(set)
    for group, part in frozen_groups.items():
        inherited[components.find(group)].add(part)
    touched = set()
    root_groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in kept.values():
        root = components.find(row["group_id"])
        root_groups[root].add(row["group_id"])
        if (
            row["input_sha256"] in train_inputs
            or row["group_id"] in train_groups
            or state_key(row["state"]) in train_states
        ):
            touched.add(root)
    parts: dict[str, list[dict[str, Any]]] = {"train": [], "aho": []}
    assignment: collections.Counter[str] = collections.Counter()
    for row in kept.values():
        root = components.find(row["group_id"])
        frozen_parts = inherited.get(root, set())
        if len(frozen_parts) > 1:
            raise ValueError(f"component {root} spans frozen train and aho")
        if frozen_parts:
            part = next(iter(frozen_parts))
            assignment[f"inherited_{part}"] += 1
        elif root not in touched and is_aho(min(root_groups[root])):
            part = "aho"
            assignment["hash_aho"] += 1
        else:
            part = "train"
            assignment["hash_or_touch_train"] += 1
        if part == "aho":
            row = dict(row, split="select", evaluation_role="select")
            validate_row(row, "select")
        parts[part].append(row)
    manifest = {
        "schema": "decision2.v2.a7.recover.v1",
        "version": VERSION,
        "sub_arm": SUB_ARM,
        "sources": [
            {
                key: entry[key]
                for key in ("name", "path", "sha256", "rows")
                if key in entry
            }
            for entry in spec["sources"]
        ],
        "rows_loaded": loaded,
        "excluded": keyed(excluded),
        "duplicates": dict(sorted(duplicates.items())),
        "isolation_dropped": dict(sorted(isolation_dropped.items())),
        "partition_assignment": dict(sorted(assignment.items())),
        "parts": {
            part: {
                "rows": len(rows),
                "by_family_type": keyed(
                    collections.Counter((r["family"], r["task_type"]) for r in rows)
                ),
            }
            for part, rows in parts.items()
        },
    }
    return {"parts": parts, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--frozen-final", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit")
    args = parser.parse_args(argv)
    if (args.out_dir / "prelim").exists():
        parser.error(f"refusing to reuse {args.out_dir / 'prelim'}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = recover(spec, args.frozen_final)
    (args.out_dir / "prelim").mkdir(parents=True, mode=0o700)
    (args.out_dir / "views").mkdir(mode=0o700, exist_ok=True)
    outputs = {}
    for part, rows in result["parts"].items():
        if rows:
            data = canonical_jsonl(rows)
            _write_new(args.out_dir / "prelim" / f"{SUB_ARM}.{part}.jsonl", data)
            outputs[f"prelim/{SUB_ARM}.{part}.jsonl"] = hashlib.sha256(data).hexdigest()
    manifest = dict(result["manifest"], build_commit=args.commit, outputs=outputs)
    _write_new(args.out_dir / "build-manifest.json", _json_bytes(manifest))
    print(json.dumps({part: len(rows) for part, rows in result["parts"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
