"""Build a decoder-track TRAIN mixture from a frozen JSON spec.

Every input file is verified against its dataset registry (SHA-256) before
use. Components are processed in spec order: rows of excluded families are
dropped, construction-order ``result_<n>`` Choice keys are renumbered by
position (A7 rule 7d) where a component asks for it, rows whose canonical
input already appeared earlier are dropped (``input_sha256``), an optional
view restricts a component to listed TRAIN ids, and a token budget (absolute
or a ratio of an earlier component's tokens) takes whole groups, stratified by
source × task type × language in a seed-keyed hash order. Tokens are native
``encode`` lengths under the given tokenizer (no truncation).

Spec: {"name", "seed", "exclude_families": [...], "exclude_group_ids"?:
{"file": "<source>:<path>", "sha256": <pinned>}, "dedupe_original_hashes"?:
bool, "components": [{"name", "files": ["<source>:<path under the snapshot
root>", ...], "view"?: "<source>:<path>", "ids"?: {"file": "<source>:<path>",
"sha256": <pinned>, "pool": <pool>}, "rekey"?: bool, "budget_tokens"?: int,
"budget_ratio"?: {"of": <component> or [<component>, ...], "ratio": float},
"budget_fraction"?: float of the component's own pool}]}; sources map to
snapshot roots and registries on the command line (registry "-" = none: every
file read from that source must carry a pinned SHA-256 in the spec).

``ids`` restricts a component to the rows a published recipe lists for one of
its pools (id lists joined to arm files, as the data track's recipes are
distributed); every listed id must be present in the component's files.
``dedupe_original_hashes`` also treats rows as duplicates when their
pre-renumbering input hashes (``audit_metadata.option_key_renumbering`` or
``audit_metadata.a7``) match any hash seen earlier. ``exclude_group_ids`` drops
every row of the listed groups (a ``{"group_ids": [...]}`` file, for example the
evaluation-panel exclusions of a data rescreen) before deduplication.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, digest, file_sha256, load_partition

from .build_template_s import group_rows, select_groups

CONSTRUCTION_ORDER_KEY = re.compile(r"result_[0-9]+")
_TOKENIZER: Any = None


def rekey(row: dict[str, Any]) -> dict[str, Any]:
    """A7 rule 7d: renumber construction-order ``result_<n>`` Choice keys."""
    keys = [option["key"] for option in row["options"]]
    if row["task_type"] != "choice" or not all(
        CONSTRUCTION_ORDER_KEY.fullmatch(key) for key in keys
    ):
        return row
    positional = [f"result_{index}" for index in range(len(keys))]
    if positional == keys:
        return row
    out = dict(
        row,
        options=[dict(o, key=k) for o, k in zip(row["options"], positional)],
        audit_metadata=dict(row["audit_metadata"], dec_rule_7d={"original_keys": keys}),
    )
    out["input_sha256"] = digest({field: out[field] for field in INPUT_FIELDS})
    return out


def _init_tokenizer(path: str) -> None:
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(path, local_files_only=True)


def _length(row: dict[str, Any]) -> int:
    from training.model.decision_model import encode

    return len(encode(row, _TOKENIZER, 8192)["ids"])


def token_lengths(
    rows: list[dict[str, Any]], tokenizer: Path, workers: int
) -> list[int]:
    if workers <= 1:
        _init_tokenizer(str(tokenizer))
        return [_length(row) for row in rows]
    with ProcessPoolExecutor(
        workers, initializer=_init_tokenizer, initargs=(str(tokenizer),)
    ) as pool:
        return list(pool.map(_length, rows, chunksize=256))


class Registry:
    """Snapshot root plus its registry; paths are relative to the root."""

    def __init__(self, root: Path, registry: str, prefix: str):
        self.root = root
        self.prefix = prefix
        self.registry_path = None if registry == "-" else root / registry
        self.files = (
            json.loads(self.registry_path.read_text())["files"]
            if self.registry_path is not None
            else {}
        )

    def verified(self, relative: str, pinned: str | None = None) -> Path:
        path = self.root / relative
        actual = file_sha256(path)
        if pinned is not None:
            if actual != pinned:
                raise ValueError(f"{relative}: sha256 {actual} != pinned {pinned}")
            return path
        if self.registry_path is None:
            raise ValueError(f"{relative}: source has no registry and no pinned sha256")
        key = relative.removeprefix(self.prefix)
        entry = self.files.get(key)
        if entry is None:
            raise ValueError(f"{relative} is not listed in {self.registry_path}")
        if actual != entry["sha256"]:
            raise ValueError(
                f"{relative}: sha256 {actual} != registry {entry['sha256']}"
            )
        return path


def row_hashes(row: dict[str, Any], original: bool) -> set[str]:
    """The row's input hash, plus its pre-renumbering hashes when ``original``."""
    hashes = {row["input_sha256"]}
    if original:
        meta = row.get("audit_metadata") or {}
        for block in ("option_key_renumbering", "a7"):
            value = (meta.get(block) or {}).get("original_input_sha256")
            if isinstance(value, str):
                hashes.add(value)
    return hashes


def recipe_ids(path: Path, pool: str) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry["pool"] == pool:
                if entry["id"] in ids:
                    raise ValueError(f"{path}: repeated recipe id {entry['id']}")
                ids.add(entry["id"])
    if not ids:
        raise ValueError(f"{path}: recipe lists no row for pool {pool}")
    return ids


def build(
    spec: dict[str, Any],
    registries: dict[str, Registry],
    tokenizer: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded = set(spec.get("exclude_families", []))
    original = bool(spec.get("dedupe_original_hashes"))
    seen_inputs: set[str] = set()
    seen_ids: set[str] = set()
    mixture: list[dict[str, Any]] = []
    components: dict[str, dict[str, Any]] = {}
    inputs: dict[str, str] = {}
    excluded_groups: set[str] = set()
    if "exclude_group_ids" in spec:
        reference = spec["exclude_group_ids"]
        source, relative = reference["file"].split(":", 1)
        path = registries[source].verified(relative, reference["sha256"])
        inputs[reference["file"]] = file_sha256(path)
        excluded_groups = set(json.loads(path.read_text())["group_ids"])
    for component in spec["components"]:
        name = component["name"]
        rows: list[dict[str, Any]] = []
        for reference in component["files"]:
            source, relative = reference.split(":", 1)
            path = registries[source].verified(relative)
            inputs[reference] = file_sha256(path)
            rows.extend(load_partition(path, "train"))
        stats: Counter = Counter(loaded=len(rows))
        if "ids" in component:
            spec_ids = component["ids"]
            source, relative = spec_ids["file"].split(":", 1)
            ids_path = registries[source].verified(relative, spec_ids["sha256"])
            inputs[spec_ids["file"]] = file_sha256(ids_path)
            wanted = recipe_ids(ids_path, spec_ids["pool"])
            rows = [row for row in rows if row["id"] in wanted]
            found = {row["id"] for row in rows}
            if len(found) != len(rows) or found != wanted:
                raise ValueError(
                    f"{name}: recipe pool {spec_ids['pool']} lists {len(wanted)} ids, "
                    f"files hold {len(found)} of them ({len(rows)} rows)"
                )
            stats["in_recipe"] = len(rows)
        if "view" in component:
            source, relative = component["view"].split(":", 1)
            view_path = registries[source].verified(relative)
            inputs[component["view"]] = file_sha256(view_path)
            members = {
                m["id"]
                for m in json.loads(view_path.read_text())["members"]
                if m["part"] == "train"
            }
            rows = [row for row in rows if row["id"] in members]
            stats["in_view"] = len(rows)
        kept: list[dict[str, Any]] = []
        for row in rows:
            if row["group_id"] in excluded_groups:
                stats["excluded_group"] += 1
                continue
            if row["family"] in excluded:
                stats["excluded_family"] += 1
                continue
            if component.get("rekey"):
                renumbered = rekey(row)
                stats["rekeyed"] += renumbered is not row
                row = renumbered
            hashes = row_hashes(row, original)
            if hashes & seen_inputs or row["id"] in seen_ids:
                stats["duplicate_input"] += 1
                continue
            seen_inputs.update(hashes)
            seen_ids.add(row["id"])
            kept.append(row)
        lengths = token_lengths(kept, tokenizer, workers)
        by_id = dict(zip((row["id"] for row in kept), lengths))
        pool_tokens = sum(lengths)
        budget = component.get("budget_tokens")
        if "budget_ratio" in component:
            ratio = component["budget_ratio"]
            of = [ratio["of"]] if isinstance(ratio["of"], str) else ratio["of"]
            budget = round(sum(components[c]["tokens"] for c in of) * ratio["ratio"])
        if "budget_fraction" in component:
            budget = round(pool_tokens * component["budget_fraction"])
        if budget is not None and budget < pool_tokens:
            groups = group_rows(kept)
            group_tokens = {
                g: sum(by_id[r["id"]] for r in members) for g, members in groups.items()
            }
            chosen = select_groups(
                groups, group_tokens, budget, f"{spec['seed']}\0{name}"
            )
            chosen_ids = {r["id"] for g in chosen for r in groups[g]}
            for row in kept:
                if row["id"] not in chosen_ids:
                    seen_inputs.difference_update(row_hashes(row, original))
                    seen_ids.discard(row["id"])
            kept = [row for row in kept if row["id"] in chosen_ids]
        tokens = sum(by_id[row["id"]] for row in kept)
        components[name] = {
            **dict(stats),
            "rows": len(kept),
            "tokens": tokens,
            "pool_tokens": pool_tokens,
            "budget_tokens": budget,
            "rows_by_type": dict(Counter(r["task_type"] for r in kept)),
            "rows_by_language": dict(Counter(r["language"] for r in kept)),
            "sources": dict(Counter(r["source"] for r in kept).most_common(20)),
            "max_tokens": max((by_id[r["id"]] for r in kept), default=0),
        }
        mixture.extend(kept)
    manifest = {
        "schema_version": "dec-mixture/1",
        "spec": spec,
        "inputs_sha256": inputs,
        "registries_sha256": {
            source: file_sha256(registry.registry_path)
            for source, registry in registries.items()
            if registry.registry_path is not None
        },
        "components": components,
        "rows": len(mixture),
        "tokens": sum(c["tokens"] for c in components.values()),
        "rows_by_type": dict(Counter(r["task_type"] for r in mixture)),
    }
    return mixture, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="name=<snapshot root>,<registry path under root>,<registry key prefix>",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    registries = {}
    for item in args.source:
        name, rest = item.split("=", 1)
        root, registry, prefix = rest.split(",")
        registries[name] = Registry(Path(root), registry, prefix)
    spec = json.loads(args.spec.read_text())
    mixture, manifest = build(spec, registries, args.tokenizer, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        for row in mixture:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    pending.replace(args.output)
    load_partition(args.output, "train")
    manifest["output_sha256"] = file_sha256(args.output)
    manifest["spec_sha256"] = file_sha256(args.spec)
    manifest["tokenizer_files_sha256"] = {
        name: file_sha256(args.tokenizer / name)
        for name in ("tokenizer.json", "tokenizer_config.json")
        if (args.tokenizer / name).is_file()
    }
    args.output.with_name(args.output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("rows", "tokens", "rows_by_type")}))


if __name__ == "__main__":
    main()
