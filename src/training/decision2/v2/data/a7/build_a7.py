"""Build data arm A7 (own Decision 1.0 decoder-family corpora) as sub-arms.

Reads the SHA-256-pinned source files named in a spec, normalizes each row to
the Decision 2.0 training contract, keeps one row per input hash, assigns it
to an origin sub-arm, applies the preregistered construction-defect rules and
splits whole connected components into TRAIN and AHO. Writes pre-admission
sub-arm files, recipe views (A7 ids per 1.0 mixture) and a count-only build
manifest. Protected panels are never opened here; the overlap, shortcut and
native-budget screens run on these files and `v2.data.a7.admit` applies them.

Rules: `records/a7-prereg-2026-09-28.md` (version a7-dec10-v1).
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row
from v2.data.freeze import canonical_jsonl
from v2.data.textnorm import compact, normalize

VERSION = "a7-dec10-v1"
SCHEMA = "decision2.v2.a7.build.v1"
SUB_ARMS = ("A7h", "A7m", "A7g", "A7i", "A7p", "A7o")
GENERATED_SUB_ARMS = frozenset({"A7g", "A7p", "A7o"})
SOURCE_KEYS = {
    "cosmos_qa": "dec10:cosmos_qa_train",
    "snli": "dec10:snli_train",
    "squad2_answerability": "dec10:squad2_train",
    "nyu-mll/multi_nli": "dec10:multinli_nonfiction_train",
    "PolyAI-LDN/banking77 upstream train; CC-BY-4.0": "dec10:banking77_train",
    "CLINC150 upstream train; CC-BY-3.0": "dec10:clinc150_train",
    "stage4-general-composition-v2": "dec10:generated_stage4_v2",
    "objective_generator": "dec10:generated_stage1_3",
    "objective_generator_stage2": "dec10:generated_stage1_3",
    "objective_generator_stage3_arithmetic": "dec10:generated_stage1_3",
    "stage3-logic-evidence-v1": "dec10:generated_stage1_3",
    "stage3-src/transitions.py": "dec10:generated_stage1_3",
    "synthetic:stage3:binding_membership:v1": "dec10:generated_stage1_3",
}
READING_KEYS = frozenset(
    {"dec10:cosmos_qa_train", "dec10:snli_train", "dec10:squad2_train"}
)
INTENT_KEYS = frozenset({"dec10:banking77_train", "dec10:clinc150_train"})
LEGACY_NUMERIC_FAMILIES = frozenset(
    {
        "arithmetic",
        "stage3_arithmetic_difference",
        "stage3_arithmetic_distance",
        "stage3_arithmetic_net",
        "stage3_arithmetic_product",
        "stage3_arithmetic_sum",
        "stage3_transition_registers",
        "stage3_transition_scalar",
    }
)
REPLAY_PREFIX = "stage4_replay_"
MIN_STATE_COMPACT = 20
AUDIT_EXTRA_FIELDS = (
    "upstream_label",
    "stage3_origin",
    "render_schema",
    "evaluation_role",
    "split",
)


class Excluded(Exception):
    """A row the preregistered rules keep out of A7, with its reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_pinned(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{entry['name']}: SHA-256 differs from the spec")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{entry['name']}:{number}: blank line")
            rows.append(json.loads(line))
    if "rows" in entry and len(rows) != entry["rows"]:
        raise ValueError(f"{entry['name']}: expected {entry['rows']} rows")
    return rows


def lineage(source: Any) -> str:
    """The upstream dataset or generator a 1.0 row came from."""
    if isinstance(source, str) and source.startswith("{"):
        source = ast.literal_eval(source)
    if isinstance(source, str) and source:
        return source
    if not isinstance(source, Mapping):
        raise Excluded("unrecognized_source")
    if source.get("type") == "stage3_replay":
        return lineage(source.get("original_source"))
    for field in ("dataset", "generator"):
        if isinstance(source.get(field), str) and source[field]:
            return source[field]
    raise Excluded("unrecognized_source")


def source_key(row: Mapping[str, Any]) -> str:
    key = SOURCE_KEYS.get(lineage(row.get("source")))
    if key is None:
        raise Excluded("unrecognized_source")
    return key


def sub_arm_of(source_name: str, key: str) -> str:
    if source_name == "natural8k":
        if key not in READING_KEYS:
            raise Excluded("unexpected_natural_source")
        return "A7h"
    if key == "dec10:multinli_nonfiction_train":
        return "A7m"
    if key in INTENT_KEYS:
        return "A7i"
    if key == "dec10:generated_stage4_v2":
        return "A7g"
    if key == "dec10:generated_stage1_3":
        return "A7p" if source_name == "stage4v2" else "A7o"
    raise Excluded("unexpected_source_for_file")


def base_family(family: str) -> str:
    return family[len(REPLAY_PREFIX) :] if family.startswith(REPLAY_PREFIX) else family


def check_rules(row: Mapping[str, Any], sub_arm: str) -> None:
    """Preregistered construction-defect rules 7a and 7b."""
    task_type = row.get("task_type")
    keys = [option.get("key") for option in row.get("options") or []]
    if (
        sub_arm in GENERATED_SUB_ARMS
        and task_type == "choice"
        and base_family(str(row.get("family"))) in LEGACY_NUMERIC_FAMILIES
    ):
        raise Excluded("legacy_numeric_choice")
    if task_type == "noul" and set(keys) != {"false", "true"}:
        raise Excluded("opaque_noul_keys")
    if task_type == "score" and set(keys) != {str(i) for i in range(len(keys))}:
        raise Excluded("opaque_score_keys")


def normalize_row(
    row: Mapping[str, Any], source_name: str, source_sha256: str
) -> dict[str, Any]:
    """1.0 row -> validated 2.0 training row (raises Excluded)."""
    key = source_key(row)
    sub_arm = sub_arm_of(source_name, key)
    check_rules(row, sub_arm)
    template = row.get("render_template")
    origin: dict[str, Any] = {
        "version": VERSION,
        "sub_arm": sub_arm,
        "source_file": source_name,
        "source_file_sha256": source_sha256,
        "original_id": row["id"],
        "original_source": row.get("source"),
    }
    extra = {field: row[field] for field in AUDIT_EXTRA_FIELDS if field in row}
    if extra:
        origin["original_fields"] = extra
    out = {
        "id": f"a7:{source_name}:{row['id']}",
        "state": row["state"],
        "instructions": row["instructions"],
        "options": row["options"],
        "label": row["label"],
        "task_type": row["task_type"],
        "family": row["family"],
        "group_id": row["group_id"],
        "language": row["language"],
        "split": "train",
        "source": key,
        "evaluation_role": "train",
        "render_template": (
            template if isinstance(template, str) and template else "dec10_unspecified"
        ),
        "audit_metadata": {"a7": origin},
    }
    if isinstance(row.get("audit_metadata"), dict):
        out["audit_metadata"]["original"] = row["audit_metadata"]
    out["input_sha256"] = digest({field: out[field] for field in INPUT_FIELDS})
    if row.get("input_sha256") not in (None, out["input_sha256"]):
        origin["original_input_sha256"] = row["input_sha256"]
    try:
        validate_row(out, "train")
    except (TypeError, ValueError) as exc:
        raise Excluded("contract_invalid") from exc
    return out


def state_key(state: Any) -> str | None:
    text = state if isinstance(state, str) else canonical(state)
    if len(compact(text)) < MIN_STATE_COMPACT:
        return None
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def input_hash(row: Mapping[str, Any]) -> str:
    return digest({field: row[field] for field in INPUT_FIELDS})


class Components:
    """Union-find over group ids."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent == item:
            return item
        root = self.find(parent)
        self.parent[item] = root
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def link(self, members: dict[str, list[str]]) -> None:
        for groups in members.values():
            first = groups[0]
            for other in groups[1:]:
                self.union(first, other)


def is_aho(component_key: str) -> bool:
    return int(hashlib.sha256(component_key.encode("utf-8")).hexdigest(), 16) % 10 == 0


def keyed(counts: Mapping[tuple[Any, ...], int]) -> dict[str, int]:
    """Counter over tuples -> {"a|b|c": count}, sorted by key."""
    flat: collections.Counter[str] = collections.Counter()
    for key, count in counts.items():
        flat["|".join(map(str, key))] += count
    return dict(sorted(flat.items()))


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    ).encode("utf-8")


def _external_sets(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    groups: set[str] = set()
    inputs: set[str] = set()
    states: set[str] = set()
    for entry in entries:
        for row in read_pinned(entry):
            if isinstance(row.get("group_id"), str):
                groups.add(row["group_id"])
            if all(field in row for field in INPUT_FIELDS):
                inputs.add(input_hash(row))
                key = state_key(row["state"])
                if key is not None:
                    states.add(key)
    return groups, inputs, states


def build(spec: Mapping[str, Any], commit: str | None = None) -> dict[str, Any]:
    """Return {"sub_arms": {name: {"train": rows, "aho": rows}}, "views": ..., "manifest": ...}."""
    if spec.get("version") != VERSION:
        raise ValueError(f"spec version must be {VERSION}")
    names = [entry["name"] for entry in spec["sources"]]
    if len(set(names)) != len(names):
        raise ValueError("source names must be unique")
    excluded = collections.Counter()
    loaded = collections.Counter()
    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    link_inputs: dict[str, list[str]] = collections.defaultdict(list)
    link_states: dict[str, list[str]] = collections.defaultdict(list)
    for order, entry in enumerate(spec["sources"]):
        rows = read_pinned(entry)
        loaded[entry["name"]] = len(rows)
        seen_ids: set[str] = set()
        for number, row in enumerate(rows):
            if row.get("id") in seen_ids:
                raise ValueError(f"{entry['name']}: duplicate id {row.get('id')}")
            seen_ids.add(row["id"])
            group = row.get("group_id")
            if isinstance(group, str) and all(field in row for field in INPUT_FIELDS):
                link_inputs[input_hash(row)].append(group)
                key = state_key(row["state"])
                if key is not None:
                    link_states[key].append(group)
            try:
                out = normalize_row(row, entry["name"], entry["sha256"])
            except Excluded as reason:
                excluded[
                    (
                        entry["name"],
                        reason.reason,
                        row.get("family"),
                        row.get("task_type"),
                    )
                ] += 1
                continue
            sub_arm = out["audit_metadata"]["a7"]["sub_arm"]
            candidates.append(((SUB_ARMS.index(sub_arm), order, number), out))
    candidates.sort(key=lambda item: item[0])
    kept: dict[str, dict[str, Any]] = {}
    labels: dict[str, set[Any]] = collections.defaultdict(set)
    duplicates = collections.Counter()
    for _, row in candidates:
        digest_ = row["input_sha256"]
        labels[digest_].add(canonical(row["options"][row["label"]]))
        if digest_ in kept:
            duplicates[
                (
                    kept[digest_]["audit_metadata"]["a7"]["source_file"],
                    row["audit_metadata"]["a7"]["source_file"],
                )
            ] += 1
            continue
        kept[digest_] = row
    conflicts = {digest_ for digest_, values in labels.items() if len(values) > 1}
    conflict_rows = collections.Counter()
    for digest_ in conflicts:
        row = kept.pop(digest_)
        conflict_rows[
            (row["audit_metadata"]["a7"]["sub_arm"], row["family"], row["task_type"])
        ] += 1

    holdout_groups, holdout_inputs, _ = _external_sets(
        spec.get("isolation_holdout", [])
    )
    train_groups, train_inputs, train_states = _external_sets(
        spec.get("isolation_train", [])
    )
    isolation_dropped = collections.Counter()
    for digest_ in list(kept):
        row = kept[digest_]
        if digest_ in holdout_inputs or row["group_id"] in holdout_groups:
            isolation_dropped[
                (row["audit_metadata"]["a7"]["sub_arm"], row["family"])
            ] += 1
            del kept[digest_]

    components = Components()
    components.link(link_inputs)
    components.link(link_states)
    touched_roots = set()
    for row in kept.values():
        if (
            row["input_sha256"] in train_inputs
            or row["group_id"] in train_groups
            or state_key(row["state"]) in train_states
        ):
            touched_roots.add(components.find(row["group_id"]))
    root_groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in kept.values():
        root_groups[components.find(row["group_id"])].add(row["group_id"])
    aho_roots = {
        root
        for root, groups in root_groups.items()
        if root not in touched_roots and is_aho(min(groups))
    }
    sub_arms: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"train": [], "aho": []} for name in SUB_ARMS
    }
    for row in kept.values():
        name = row["audit_metadata"]["a7"]["sub_arm"]
        if components.find(row["group_id"]) in aho_roots:
            held = dict(row, split="select", evaluation_role="select")
            validate_row(held, "select")
            sub_arms[name]["aho"].append(held)
        else:
            sub_arms[name]["train"].append(row)

    by_input = {digest_: row for digest_, row in kept.items()}
    aho_ids = {row["id"] for parts in sub_arms.values() for row in parts["aho"]}
    views = {}
    for entry in spec.get("views", []):
        members = []
        uncovered = collections.Counter()
        for row in read_pinned(entry):
            digest_ = input_hash(row)
            match = by_input.get(digest_)
            if match is not None:
                members.append(
                    {
                        "id": match["id"],
                        "part": "aho" if match["id"] in aho_ids else "train",
                        "sub_arm": match["audit_metadata"]["a7"]["sub_arm"],
                    }
                )
            elif digest_ in conflicts:
                uncovered["label_conflict"] += 1
            elif digest_ in link_inputs:
                uncovered["excluded_by_rule_or_isolation"] += 1
            else:
                uncovered["not_in_a7_sources"] += 1
        views[entry["name"]] = {
            "view": entry["name"],
            "source_file_sha256": entry["sha256"],
            "rows": entry.get("rows"),
            "used_by": entry.get("used_by", []),
            "members": sorted(members, key=lambda item: item["id"]),
            "covered": len(members),
            "uncovered": dict(sorted(uncovered.items())),
        }

    component_sizes = sorted(
        (len(groups) for groups in root_groups.values()), reverse=True
    )
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "build_commit": commit,
        "sources": [
            {
                key: entry[key]
                for key in ("name", "path", "sha256", "rows")
                if key in entry
            }
            for entry in spec["sources"]
        ],
        "isolation_inputs": {
            kind: [
                {key: entry[key] for key in ("name", "sha256") if key in entry}
                for entry in spec.get(kind, [])
            ]
            for kind in ("isolation_train", "isolation_holdout")
        },
        "rows_loaded": dict(loaded),
        "excluded": keyed(excluded),
        "duplicates_dropped": {
            f"{kept_file}<-{dropped_file}": count
            for (kept_file, dropped_file), count in sorted(duplicates.items())
        },
        "label_conflicts_dropped": keyed(conflict_rows),
        "isolation_dropped": keyed(isolation_dropped),
        "components": {
            "count": len(root_groups),
            "largest_group_counts": component_sizes[:10],
            "never_aho_because_touching_train": len(touched_roots & set(root_groups)),
            "aho_components": len(aho_roots),
            "rule": "groups linked by shared input_sha256 or normalized state "
            f"(>= {MIN_STATE_COMPACT} compact chars); AHO iff "
            "int(sha256(min group id), 16) % 10 == 0 and the component "
            "touches no isolation_train file",
        },
        "sub_arms": {
            name: {
                part: {
                    "rows": len(rows),
                    "by_task_type": dict(
                        sorted(
                            collections.Counter(
                                row["task_type"] for row in rows
                            ).items()
                        )
                    ),
                }
                for part, rows in parts.items()
            }
            for name, parts in sub_arms.items()
        },
        "views": {
            name: {
                key: view[key] for key in ("covered", "uncovered", "rows", "used_by")
            }
            for name, view in views.items()
        },
    }
    return {"sub_arms": sub_arms, "views": views, "manifest": manifest}


def write_outputs(result: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    """Write prelim/<sub>.{train,aho}.jsonl, views/<view>.prelim.json and the manifest."""
    (out_dir / "prelim").mkdir(parents=True, mode=0o700, exist_ok=False)
    (out_dir / "views").mkdir(mode=0o700)
    hashes: dict[str, str] = {}
    for name, parts in result["sub_arms"].items():
        for part, rows in parts.items():
            if not rows:
                continue
            data = canonical_jsonl(rows)
            path = out_dir / "prelim" / f"{name}.{part}.jsonl"
            _write_new(path, data)
            hashes[f"prelim/{path.name}"] = hashlib.sha256(data).hexdigest()
    for name, view in result["views"].items():
        data = _json_bytes(view)
        path = out_dir / "views" / f"{name}.prelim.json"
        _write_new(path, data)
        hashes[f"views/{path.name}"] = hashlib.sha256(data).hexdigest()
    manifest = dict(result["manifest"], outputs=hashes)
    _write_new(out_dir / "build-manifest.json", _json_bytes(manifest))
    return hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit", help="exact mirrored source commit")
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        parser.error(f"refusing to reuse {args.out_dir}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = build(spec, commit=args.commit)
    hashes = write_outputs(result, args.out_dir)
    summary = {
        name: {part: info["rows"] for part, info in parts.items()}
        for name, parts in result["manifest"]["sub_arms"].items()
    }
    print(json.dumps({"sub_arms": summary, "files": len(hashes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
