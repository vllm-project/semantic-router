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

Spec: {"name", "seed", "exclude_families": [...], "components": [{"name",
"files": ["<source>:<path under the snapshot root>", ...], "view"?:
"<source>:<path>", "rekey"?: bool, "budget_tokens"?: int, "budget_ratio"?:
{"of": <component>, "ratio": float}}]}; sources map to snapshot roots and
registries on the command line.
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
        self.registry_path = root / registry
        self.files = json.loads(self.registry_path.read_text())["files"]

    def verified(self, relative: str) -> Path:
        key = relative.removeprefix(self.prefix)
        entry = self.files.get(key)
        if entry is None:
            raise ValueError(f"{relative} is not listed in {self.registry_path}")
        path = self.root / relative
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"{relative}: sha256 {actual} != registry {entry['sha256']}"
            )
        return path


def build(
    spec: dict[str, Any],
    registries: dict[str, Registry],
    tokenizer: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded = set(spec.get("exclude_families", []))
    seen_inputs: set[str] = set()
    seen_ids: set[str] = set()
    mixture: list[dict[str, Any]] = []
    components: dict[str, dict[str, Any]] = {}
    inputs: dict[str, str] = {}
    for component in spec["components"]:
        name = component["name"]
        rows: list[dict[str, Any]] = []
        for reference in component["files"]:
            source, relative = reference.split(":", 1)
            path = registries[source].verified(relative)
            inputs[reference] = file_sha256(path)
            rows.extend(load_partition(path, "train"))
        stats: Counter = Counter(loaded=len(rows))
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
            if row["family"] in excluded:
                stats["excluded_family"] += 1
                continue
            if component.get("rekey"):
                renumbered = rekey(row)
                stats["rekeyed"] += renumbered is not row
                row = renumbered
            if row["input_sha256"] in seen_inputs or row["id"] in seen_ids:
                stats["duplicate_input"] += 1
                continue
            seen_inputs.add(row["input_sha256"])
            seen_ids.add(row["id"])
            kept.append(row)
        lengths = token_lengths(kept, tokenizer, workers)
        by_id = dict(zip((row["id"] for row in kept), lengths))
        pool_tokens = sum(lengths)
        budget = component.get("budget_tokens")
        if "budget_ratio" in component:
            ratio = component["budget_ratio"]
            budget = round(components[ratio["of"]]["tokens"] * ratio["ratio"])
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
                    seen_inputs.discard(row["input_sha256"])
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
