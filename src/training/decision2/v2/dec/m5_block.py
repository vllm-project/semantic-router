"""Decoder Milestone 5: the MLX-DEV holdout and the N5B multilingual-block recomposition.

Inputs are N4XF's mixture (``train.jsonl`` + build manifest; components are
contiguous in spec order) and a ``build_mixture`` superset build of the five
multilingual pools (A7k, A7q, A7s, H5, H8 at full XL-r2 pool size), so every
candidate row is rendered exactly as ``build_mixture`` renders it; output rows
are copied as raw lines from those two files.

**Block** = the non-English rows of the five pools; each belongs to one cell
(``CELLS``). **MLX-DEV** takes XL-r2 groups absent from N4XF (non-English rows
only, whole groups), in sha256(``dec-m5-mlxdev-v1\\0<cell>\\0<group_id>``)
order: Noul cells up to N groups per language, the other cells until their row
target. A candidate group is dropped if any of its normalized input segments
(string fields of state / instructions / option descriptions split on
newlines; NFKC, lower-case, whitespace-collapsed; >= 20 characters) occurs in
any N4XF row, or if a row's input hash does. ``--template-max-groups N``
(amendment parameter; unset = the literal rule) ignores segments that occur in
more than N distinct N4XF groups (instruction / template lines).

**N5B**: N4XF rows outside the block are unchanged; block cells are
re-composed to ``QUOTAS`` of N4XF's block tokens. A shrinking cell keeps a
language-stratified subset of N4XF's whole groups in
sha256(``dec-m5-block-v1\\0<cell>\\0<group_id>``) order; a growing cell keeps all
N4XF rows and adds N4XF-unseen XL-r2 groups in that order, stratified by the
cell's XL-r2 language token mix (a group's language is its first row's), excluding MLX-DEV groups, groups with a row
sharing a (non-ignored) segment with MLX-DEV, r2-excluded groups and rows whose
input hash already occurs in N4XF. Unused A7k quota goes to A7q. Within a
stratum groups are added until its share is reached (the last may overshoot,
as ``build_template_s.select_groups``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from training.model.data import file_sha256

from .build_mixture import row_hashes

BLOCK_COMPONENTS = ("A7k", "A7q", "A7s", "H5", "H8")
QUOTAS = {
    "noul": 0.45,
    "a7q": 0.20,
    "a7k": 0.06,
    "a7s": 0.09,
    "h8-choice": 0.09,
    "h8-score": 0.05,
    "h5-choice": 0.06,
}
QUOTA_TOLERANCE = 0.05
BUDGET_TOKENS = 29_407_326
BUDGET_TOLERANCE = 0.01
MLX_PREFIX = "dec-m5-mlxdev-v1"
BLOCK_PREFIX = "dec-m5-block-v1"
MIN_SEGMENT = 20
# cell -> (component, task_type, source prefix or None, kind, amount)
MLX_CELLS: dict[str, tuple[str, str, str | None, str, int]] = {
    "h5-noul": ("H5", "noul", None, "groups_per_language", 100),
    "h8-miracl-noul": ("H8", "noul", "miracl", "groups_per_language", 80),
    "h8-choice": ("H8", "choice", "jglue_jcommonsenseqa", "rows", 400),
    "h5-choice": ("H5", "choice", "mtop", "rows", 400),
    "a7q": ("A7q", "score", None, "rows", 600),
    "a7k": ("A7k", "score", None, "rows", 400),
    "a7s": ("A7s", "score", None, "rows", 400),
    "h8-score": ("H8", "score", "sentimix_spanglish", "rows", 300),
}
# Growing cells are processed in this order so unused A7k quota can move to A7q.
CELL_ORDER = ("noul", "a7k", "a7q", "a7s", "h8-choice", "h8-score", "h5-choice")


def block_cell(component: str, row: dict[str, Any]) -> str | None:
    if component not in BLOCK_COMPONENTS or row["language"] == "en":
        return None
    kind = row["task_type"]
    if component in ("H5", "H8") and kind == "noul":
        return "noul"
    if component in ("A7k", "A7q", "A7s") and kind == "score":
        return component.lower()
    if component == "H8" and kind in ("choice", "score"):
        return f"h8-{kind}"
    if component == "H5" and kind == "choice":
        return "h5-choice"
    raise ValueError(f"{row['id']}: no block cell for {component} {kind}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def segments(row: dict[str, Any]) -> set[bytes]:
    """Hashes of the row's normalized input segments (>= MIN_SEGMENT characters)."""
    fields = [row["state"], row["instructions"]]
    fields += [option["description"] for option in row["options"]]
    out = set()
    for text in _strings(fields):
        for part in text.split("\n"):
            part = normalize(part)
            if len(part) >= MIN_SEGMENT:
                out.add(hashlib.sha256(part.encode("utf-8")).digest()[:16])
    return out


def hash_order(prefix: str, cell: str, group_ids: Iterable[str]) -> list[str]:
    return sorted(
        group_ids,
        key=lambda g: hashlib.sha256(f"{prefix}\0{cell}\0{g}".encode()).hexdigest(),
    )


def component_slices(
    rows: list[Any], manifest: dict[str, Any]
) -> list[tuple[str, list[Any]]]:
    """Split a mixture's rows into its components (contiguous, manifest order)."""
    order = manifest.get("component_order") or [
        c["name"] for c in manifest["spec"]["components"]
    ]
    out, start = [], 0
    for name in order:
        n = manifest["components"][name]["rows"]
        out.append((name, rows[start : start + n]))
        start += n
    if start != len(rows):
        raise ValueError(f"manifest lists {start} rows, file holds {len(rows)}")
    return out


def read_lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        return [(line, json.loads(line)) for line in stream]


def take_by_language(
    groups: dict[str, list[dict[str, Any]]],
    order: list[str],
    tokens: dict[str, int],
    shares: dict[str, float],
) -> list[str]:
    """Hash-ordered whole groups per language until each language's share."""
    taken: Counter = Counter()
    chosen = []
    for group_id in order:
        language = groups[group_id][0]["language"]
        if taken[language] >= shares.get(language, 0.0):
            continue
        chosen.append(group_id)
        taken[language] += tokens[group_id]
    return chosen


class Block:
    """All selections; ``lengths`` maps row id -> native tokens."""

    def __init__(
        self,
        n4xf: list[tuple[str, dict[str, Any]]],
        n4xf_manifest: dict[str, Any],
        superset: list[tuple[str, dict[str, Any]]],
        superset_manifest: dict[str, Any],
        excluded_groups: set[str],
        lengths: Callable[[list[dict[str, Any]]], dict[str, int]],
        template_max_groups: int | None = None,
    ):
        self.n4xf = n4xf
        self.n4xf_manifest = n4xf_manifest
        self.excluded_groups = excluded_groups
        self.template_max_groups = template_max_groups
        self.n4xf_components = component_slices(n4xf, n4xf_manifest)
        self.superset_components = dict(component_slices(superset, superset_manifest))
        self.n4xf_groups = {row["group_id"] for _, row in n4xf}
        self.n4xf_hashes: set[str] = set()
        group_segments: dict[str, set[bytes]] = defaultdict(set)
        for _, row in n4xf:
            self.n4xf_hashes |= row_hashes(row, True)
            group_segments[row["group_id"]] |= segments(row)
        frequency: Counter = Counter()
        for segs in group_segments.values():
            frequency.update(segs)
        self.ignored = (
            set()
            if template_max_groups is None
            else {s for s, n in frequency.items() if n > template_max_groups}
        )
        self.n4xf_segments = set(frequency) - self.ignored
        self.block_rows: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        self.block_component: dict[str, str] = {}
        for name, items in self.n4xf_components:
            for line, row in items:
                cell = block_cell(name, row)
                if cell is not None:
                    self.block_rows[cell].append((line, row))
                    self.block_component[row["id"]] = name
        self.stats: dict[str, Any] = {}
        self._segments: dict[str, set[bytes]] = {}
        # Candidate groups: XL-r2 rows of the five pools, non-English, N4XF-unseen, whole.
        self.candidates: dict[str, dict[str, list[tuple[str, dict[str, Any]]]]] = {}
        self.pool_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        reasons: dict[str, Counter] = defaultdict(Counter)
        for name in BLOCK_COMPONENTS:
            by_group: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
            for line, row in self.superset_components.get(name, []):
                by_group[row["group_id"]].append((line, row))
            for group_id, items in by_group.items():
                cells = {block_cell(name, row) for _, row in items}
                for _, row in items:
                    cell = block_cell(name, row)
                    if cell is not None:
                        self.pool_rows[cell].append(row)
                if group_id in self.n4xf_groups:
                    continue
                key = f"{name}:{'/'.join(sorted(map(str, cells)))}"
                if None in cells or len(cells) != 1:
                    reasons[key]["mixed_or_english_group"] += 1
                    continue
                if group_id in excluded_groups:
                    reasons[key]["r2_excluded_group"] += 1
                    continue
                if any(row_hashes(row, True) & self.n4xf_hashes for _, row in items):
                    reasons[key]["input_hash_in_n4xf"] += 1
                    continue
                self.candidates.setdefault(name, {})[group_id] = items
        self.stats["candidate_exclusions"] = {k: dict(v) for k, v in reasons.items()}
        needed = [
            row
            for items in self.candidates.values()
            for g in items.values()
            for _, row in g
        ]
        needed += [row for items in self.block_rows.values() for _, row in items]
        needed += [row for rows in self.pool_rows.values() for row in rows]
        unique = {row["id"]: row for row in needed}
        self.lengths = lengths(list(unique.values()))

    def segs(self, row: dict[str, Any]) -> set[bytes]:
        cached = self._segments.get(row["id"])
        if cached is None:
            cached = self._segments[row["id"]] = segments(row)
        return cached

    def group_tokens(self, items: list[tuple[str, dict[str, Any]]]) -> int:
        return sum(self.lengths[row["id"]] for _, row in items)

    # ---------------------------------------------------------------- MLX-DEV
    def mlx_dev(self) -> dict[str, list[str]]:
        selection: dict[str, list[str]] = {}
        report: dict[str, Any] = {}
        for cell, (component, kind, source, mode, amount) in MLX_CELLS.items():
            pool = {
                g: items
                for g, items in self.candidates.get(component, {}).items()
                if all(
                    row["task_type"] == kind
                    and (source is None or row["source"].startswith(source))
                    for _, row in items
                )
            }
            conflicts = 0
            chosen: list[str] = []
            per_language: Counter = Counter()
            rows = 0
            for group_id in hash_order(MLX_PREFIX, cell, pool):
                items = pool[group_id]
                language = items[0][1]["language"]
                if mode == "groups_per_language":
                    if per_language[language] >= amount:
                        continue
                elif rows >= amount:
                    break
                if any(self.segs(row) & self.n4xf_segments for _, row in items):
                    conflicts += 1
                    continue
                chosen.append(group_id)
                per_language[language] += 1
                rows += len(items)
            selection[cell] = chosen
            members = [row for g in chosen for _, row in pool[g]]
            report[cell] = {
                "component": component,
                "candidate_groups": len(pool),
                "segment_conflicts_skipped": conflicts,
                "groups": len(chosen),
                "rows": len(members),
                "tokens": sum(self.lengths[r["id"]] for r in members),
                "groups_by_language": dict(sorted(per_language.items())),
                "rows_by_language": dict(
                    sorted(Counter(r["language"] for r in members).items())
                ),
                "rows_by_source": dict(
                    sorted(Counter(r["source"] for r in members).items())
                ),
                "gold_by_key": dict(
                    sorted(
                        Counter(
                            r["options"][r["label"]]["key"] for r in members
                        ).items()
                    )
                ),
            }
        self.mlx_selection = selection
        self.mlx_groups = {g for groups in selection.values() for g in groups}
        self.mlx_segments: set[bytes] = set()
        for cell, groups in selection.items():
            component = MLX_CELLS[cell][0]
            for g in groups:
                for _, row in self.candidates[component][g]:
                    self.mlx_segments |= self.segs(row) - self.ignored
        self.stats["mlx_dev"] = report
        return selection

    def mlx_rows(self) -> list[tuple[str, str, dict[str, Any]]]:
        out = []
        for cell, groups in self.mlx_selection.items():
            component = MLX_CELLS[cell][0]
            for g in groups:
                out += [
                    (cell, line, row) for line, row in self.candidates[component][g]
                ]
        return out

    # -------------------------------------------------------------------- N5B
    def compose(self) -> dict[str, Any]:
        block_tokens = sum(
            self.lengths[row["id"]]
            for items in self.block_rows.values()
            for _, row in items
        )
        targets = {cell: round(QUOTAS[cell] * block_tokens) for cell in QUOTAS}
        kept: dict[str, set[str]] = {}
        added: dict[str, list[str]] = {}
        cells: dict[str, Any] = {}
        mlx_conflict_groups = 0
        for cell in CELL_ORDER:
            items = self.block_rows[cell]
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for _, row in items:
                groups[row["group_id"]].append(row)
            tokens = {
                g: sum(self.lengths[r["id"]] for r in rs) for g, rs in groups.items()
            }
            have = sum(tokens.values())
            target = targets[cell]
            info: dict[str, Any] = {
                "target_tokens": target,
                "n4xf_rows": len(items),
                "n4xf_tokens": have,
            }
            if have > target:
                by_language: Counter = Counter()
                for g, rs in groups.items():
                    by_language[rs[0]["language"]] += tokens[g]
                shares = {lang: target * t / have for lang, t in by_language.items()}
                chosen = take_by_language(
                    groups, hash_order(BLOCK_PREFIX, cell, groups), tokens, shares
                )
                kept[cell] = {r["id"] for g in chosen for r in groups[g]}
                added[cell] = []
                info["mode"] = "shrink"
            else:
                kept[cell] = {r["id"] for _, r in items}
                need = target - have
                # A group's stratum is its first row's language (build_template_s.stratum).
                pool_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in self.pool_rows[cell]:
                    pool_groups[row["group_id"]].append(row)
                pool_by_language: Counter = Counter()
                for rs in pool_groups.values():
                    pool_by_language[rs[0]["language"]] += sum(
                        self.lengths[r["id"]] for r in rs
                    )
                total = sum(pool_by_language.values())
                shares = {
                    lang: need * t / total
                    for lang, t in pool_by_language.items()
                    if total
                }
                component_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
                for component in BLOCK_COMPONENTS:
                    for g, rs in self.candidates.get(component, {}).items():
                        if (
                            block_cell(component, rs[0][1]) == cell
                            and g not in self.mlx_groups
                        ):
                            if any(self.segs(r) & self.mlx_segments for _, r in rs):
                                mlx_conflict_groups += 1
                                continue
                            component_groups[g] = rs
                cand = {g: [r for _, r in rs] for g, rs in component_groups.items()}
                cand_tokens = {
                    g: self.group_tokens(rs) for g, rs in component_groups.items()
                }
                chosen = take_by_language(
                    cand, hash_order(BLOCK_PREFIX, cell, cand), cand_tokens, shares
                )
                added[cell] = chosen
                info["mode"] = "grow"
                info["candidate_groups"] = len(cand)
                info["candidate_tokens"] = sum(cand_tokens.values())
                info["xl_r2_language_tokens"] = dict(sorted(pool_by_language.items()))
                if cell == "a7k":
                    achieved = have + sum(cand_tokens[g] for g in chosen)
                    if achieved < target:
                        moved = target - achieved
                        targets["a7q"] += moved
                        info["remainder_to_a7q"] = moved
            self._finish_cell(cell, info, kept, added)
            cells[cell] = info
        self.kept, self.added = kept, added
        self.stats["mlx_conflict_groups_excluded_from_additions"] = mlx_conflict_groups
        self.stats["block"] = {"n4xf_block_tokens": block_tokens, "cells": cells}
        return cells

    def _finish_cell(
        self, cell: str, info: dict[str, Any], kept: dict, added: dict
    ) -> None:
        rows = [r for _, r in self.block_rows[cell] if r["id"] in kept[cell]]
        for g in added[cell]:
            component = self._component_of_group(g)
            rows += [r for _, r in self.candidates[component][g]]
        tokens = sum(self.lengths[r["id"]] for r in rows)
        info.update(
            {
                "kept_n4xf_rows": len(kept[cell]),
                "added_groups": len(added[cell]),
                "rows": len(rows),
                "tokens": tokens,
                "rel_dev_from_target": tokens / info["target_tokens"] - 1,
                "rows_by_language": dict(
                    sorted(Counter(r["language"] for r in rows).items())
                ),
                "tokens_by_language": dict(
                    sorted(
                        Counter(
                            {
                                lang: sum(
                                    self.lengths[r["id"]]
                                    for r in rows
                                    if r["language"] == lang
                                )
                                for lang in {r["language"] for r in rows}
                            }
                        ).items()
                    )
                ),
                "rows_by_source": dict(
                    sorted(Counter(r["source"] for r in rows).items())
                ),
            }
        )

    def _component_of_group(self, group_id: str) -> str:
        for component, groups in self.candidates.items():
            if group_id in groups:
                return component
        raise KeyError(group_id)

    def mixture(self) -> tuple[list[str], dict[str, Any]]:
        keep_ids = set().union(*self.kept.values())
        added_by_component: dict[str, list[str]] = defaultdict(list)
        for cell in CELL_ORDER:
            for g in self.added[cell]:
                component = self._component_of_group(g)
                added_by_component[component] += [
                    line for line, _ in self.candidates[component][g]
                ]
        lines: list[str] = []
        components: dict[str, Any] = {}
        for name, items in self.n4xf_components:
            base = self.n4xf_manifest["components"][name]
            out = [
                (line, row)
                for line, row in items
                if row["id"] not in self.block_component or row["id"] in keep_ids
            ]
            removed = [
                row
                for line, row in items
                if row["id"] in self.block_component and row["id"] not in keep_ids
            ]
            extra = [json.loads(line) for line in added_by_component.get(name, [])]
            tokens = (
                base["tokens"]
                - sum(self.lengths[r["id"]] for r in removed)
                + sum(self.lengths[r["id"]] for r in extra)
            )
            lines += [line for line, _ in out] + added_by_component.get(name, [])
            rows = [row for _, row in out] + extra
            components[name] = {
                "rows": len(rows),
                "tokens": tokens,
                "n4xf_rows": base["rows"],
                "n4xf_tokens": base["tokens"],
                "removed_n4xf_rows": len(removed),
                "added_rows": len(extra),
                "rows_by_type": dict(Counter(r["task_type"] for r in rows)),
                "rows_by_language": dict(Counter(r["language"] for r in rows)),
            }
        all_rows = [json.loads(line) for line in lines]
        ids = [r["id"] for r in all_rows]
        if len(set(ids)) != len(ids):
            raise ValueError("N5B repeats a row id")
        tokens = sum(c["tokens"] for c in components.values())
        manifest = {
            "schema_version": "dec-m5-block/1",
            "component_order": [name for name, _ in self.n4xf_components],
            "components": components,
            "rows": len(all_rows),
            "tokens": tokens,
            "tokens_vs_budget": tokens / BUDGET_TOKENS - 1,
            "rows_by_type": dict(Counter(r["task_type"] for r in all_rows)),
        }
        return lines, manifest

    def checks(self, lines: list[str], manifest: dict[str, Any]) -> dict[str, Any]:
        rows = [json.loads(line) for line in lines]
        block_total = sum(c["tokens"] for c in self.stats["block"]["cells"].values())
        n4_lines = {row["id"]: line for line, row in self.n4xf}
        outside = [
            (line, row)
            for line, row in zip(lines, rows)
            if row["id"] in n4_lines and row["id"] not in self.block_component
        ]
        n4_outside = sum(
            1 for _, row in self.n4xf if row["id"] not in self.block_component
        )
        superset_lines = {
            row["id"]: line
            for items in self.superset_components.values()
            for line, row in items
        }
        rendered = [
            superset_lines.get(row["id"]) == line
            for items in self.block_rows.values()
            for line, row in items
        ]
        quotas = {
            cell: {
                "share": info["tokens"] / self.stats["block"]["n4xf_block_tokens"],
                "quota": QUOTAS[cell],
                "rel_dev": info["tokens"]
                / (QUOTAS[cell] * self.stats["block"]["n4xf_block_tokens"])
                - 1,
            }
            for cell, info in self.stats["block"]["cells"].items()
        }
        mlx_ids = {row["id"] for _, _, row in self.mlx_rows()}
        return {
            "budget_ok": abs(manifest["tokens_vs_budget"]) <= BUDGET_TOLERANCE,
            "outside_block_identical": len(outside) == n4_outside
            and all(n4_lines[row["id"]] == line for line, row in outside),
            "n4xf_block_rows_rendered_identically_in_superset": sum(rendered),
            "n4xf_block_rows": len(rendered),
            "block_tokens": block_total,
            "quotas": quotas,
            "quotas_ok": all(
                abs(q["rel_dev"]) <= QUOTA_TOLERANCE for q in quotas.values()
            ),
            "mlx_groups_in_n5b": sum(
                row["group_id"] in self.mlx_groups for row in rows
            ),
            "mlx_ids_in_n5b": sum(row["id"] in mlx_ids for row in rows),
            "mlx_groups_in_n4xf": len(self.mlx_groups & self.n4xf_groups),
            "mlx_segment_conflicts_n5b_rows": sum(
                bool(segments(row) & self.mlx_segments) for row in rows
            ),
            "r2_excluded_rows_n5b": sum(
                row["group_id"] in self.excluded_groups for row in rows
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n4xf", type=Path, required=True, help="N4XF train.jsonl (manifest beside it)"
    )
    parser.add_argument(
        "--superset", type=Path, required=True, help="superset build train.jsonl"
    )
    parser.add_argument("--excluded-groups", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--template-max-groups", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    if (out / "train.jsonl").exists():
        raise FileExistsError(out / "train.jsonl")
    from .build_mixture import token_lengths

    def lengths(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(
            zip(
                (r["id"] for r in rows),
                token_lengths(rows, args.tokenizer, args.workers),
            )
        )

    manifest_of = lambda p: json.loads(
        p.with_name(p.name + ".manifest.json").read_text()
    )  # noqa: E731
    block = Block(
        read_lines(args.n4xf),
        manifest_of(args.n4xf),
        read_lines(args.superset),
        manifest_of(args.superset),
        set(json.loads(args.excluded_groups.read_text())["group_ids"]),
        lengths,
        args.template_max_groups,
    )
    for name in ("A7k", "A7s"):
        n4 = [
            row for n, items in block.n4xf_components if n == name for _, row in items
        ]
        got = sum(block.lengths[r["id"]] for r in n4)
        if got != block.n4xf_manifest["components"][name]["tokens"]:
            raise ValueError(
                f"{name}: token lengths {got} differ from the N4XF manifest"
            )
    selection = block.mlx_dev()
    block.compose()
    lines, manifest = block.mixture()
    checks = block.checks(lines, manifest)
    out.mkdir(parents=True, exist_ok=True)
    pending = out / "train.jsonl.pending"
    with pending.open("x", encoding="utf-8") as stream:
        stream.writelines(lines)
    pending.replace(out / "train.jsonl")
    (out / "mlxdev.selection.json").write_text(
        json.dumps({"prefix": MLX_PREFIX, "cells": selection}, indent=1, sort_keys=True)
        + "\n"
    )
    manifest.update(
        {
            "inputs_sha256": {
                "n4xf": file_sha256(args.n4xf),
                "n4xf_manifest": file_sha256(
                    args.n4xf.with_name(args.n4xf.name + ".manifest.json")
                ),
                "superset": file_sha256(args.superset),
                "superset_manifest": file_sha256(
                    args.superset.with_name(args.superset.name + ".manifest.json")
                ),
                "excluded_groups": file_sha256(args.excluded_groups),
            },
            "tokenizer_files_sha256": {
                name: file_sha256(args.tokenizer / name)
                for name in ("tokenizer.json", "tokenizer_config.json")
                if (args.tokenizer / name).is_file()
            },
            "rules": {
                "block_prefix": BLOCK_PREFIX,
                "mlx_prefix": MLX_PREFIX,
                "quotas": QUOTAS,
                "mlx_cells": MLX_CELLS,
                "min_segment_chars": MIN_SEGMENT,
                "template_max_groups": args.template_max_groups,
                "ignored_template_segments": len(block.ignored),
            },
            "stats": block.stats,
            "checks": checks,
            "mlxdev_selection_sha256": file_sha256(out / "mlxdev.selection.json"),
            "output_sha256": file_sha256(out / "train.jsonl"),
        }
    )
    (out / "train.jsonl.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"rows": manifest["rows"], "tokens": manifest["tokens"], "checks": checks},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
