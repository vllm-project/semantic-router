"""Milestone 6 seed mixtures: disjoint, token-budgeted seeds of one XL r2 recipe.

    python3 -m v2.06b.m6_partition build --spec S.json [--spec ...] --output-dir DIR [--workers N]
    python3 -m v2.06b.m6_partition verify --spec S.json [--spec ...] --output-dir DIR \
        [--teacher FAMILY=PATH=SHA256 ...] [--rights-clean DIR] [--converter-bundle DIR] --report V.json

Seed mixture k of family F (`recipe`) is BASE + QKS_k + REST_{F,k}:

- BASE: every row of the base pool (A0s-strict), identical in every seed.
- QKS_k: per QKS pool (A7q, A7k, A7s), the rows in both `recipe` and `shared_with`, cut into
  `seeds` consecutive token parts: groups in sha256(salt, pool, group) order; seed 1 takes
  groups until the next would pass total / seeds (that group starts seed 2), and so on; the
  last seed takes the rest. Identical for every family with the same two recipes.
- REST: every other pool p of F has the per-seed target R_k * N_p / N_rest, with
  R_k = budget - BASE - QKS_k and N_p the pool's admissible tokens. One walk over the
  pool's groups in the same order fills seed 1 until the next group would pass its target,
  then seed 2, then seed 3, then stops; the rest of the pool is unused.

Groups are whole (group_id within a pool). A group with a row over `max_row_tokens` is
dropped before partitioning; a row repeating an earlier input of its seed mixture is
dropped. Tokens are `mixture.UNIT`. Every seed must land within `budget_tolerance` of the
budget. `build` computes all seeds of a spec's partition and writes the spec's own seed
(`<name>.train.jsonl`, `<name>.report.json`, both write-once); `verify` checks a set of
built seeds (disjointness, shared BASE and QKS, coverage, trainer isolation, teacher
coverage) and writes a counts-only report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from . import mixture as mix
from .common import digest, file_sha256, write_json

TEMPLATE = "M6"
CountMany = Callable[[list[dict[str, Any]]], list[int]]


def partition_order(groups: list[str], salt: str, pool: str) -> list[str]:
    return sorted(
        groups,
        key=lambda g: (hashlib.sha256(f"{salt}\0{pool}\0{g}".encode()).hexdigest(), g),
    )


def equal_parts(sizes: list[int], parts: int) -> list[int]:
    """Part index per group (in order): each part until the next group would pass total / parts."""
    total = sum(sizes)
    out, part, used = [], 0, 0
    for size in sizes:
        if part < parts - 1 and parts * (used + size) > total:
            part, used = part + 1, 0
        out.append(part)
        used += size
    return out


def sequential_fill(sizes: list[int], targets: list[Fraction]) -> list[int | None]:
    """Part index per group (in order), or None once every target is full."""
    out: list[int | None] = []
    part, used = 0, 0
    for size in sizes:
        while part < len(targets) and used + size > targets[part]:
            part, used = part + 1, 0
        if part == len(targets):
            out.extend([None] * (len(sizes) - len(out)))
            break
        out.append(part)
        used += size
    return out


def read_recipe(entry: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(
            f"{entry['name']}: recipe ids file differs from its frozen hash"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != entry["rows"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{entry['name']}: recipe row count differs or ids repeat")
    return rows


class Inputs:
    """Hash-pinned pools and recipes, loaded once, with token counts cached per row."""

    def __init__(self, count_many: CountMany):
        self.count_many = count_many
        self.pools: dict[str, dict[str, dict[str, Any]]] = {}
        self.pool_sha: dict[str, str] = {}
        self.owner: dict[str, str] = {}
        self.recipes: dict[str, list[dict[str, Any]]] = {}
        self.tokens: dict[tuple[str, str], int] = {}

    def pool(self, name: str, entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if name in self.pools:
            if self.pool_sha[name] != entry["sha256"]:
                raise ValueError(f"{name}: two specs pin different pool files")
            return self.pools[name]
        rows = mix.arm_rows({"arm": name, **entry})
        for row in rows:
            if row["id"] in self.owner:
                raise ValueError(
                    f"{row['id']}: in pools {self.owner[row['id']]} and {name}"
                )
            self.owner[row["id"]] = name
        self.pools[name] = {row["id"]: row for row in rows}
        self.pool_sha[name] = entry["sha256"]
        return self.pools[name]

    def recipe(self, spec: dict[str, Any], key: str) -> list[dict[str, Any]]:
        """Recipe entries, each resolved to exactly one pool row with the same metadata."""
        entry = spec[key]
        if entry["sha256"] not in self.recipes:
            rows = read_recipe(entry)
            for e in rows:
                if e["pool"] not in spec["pools"]:
                    raise ValueError(f"{entry['name']}: pool {e['pool']} is not pinned")
                row = self.pool(e["pool"], spec["pools"][e["pool"]]).get(e["id"])
                if row is None or self.owner[e["id"]] != e["pool"]:
                    raise ValueError(f"{e['id']}: not a row of pool {e['pool']}")
                if any(row[f] != e[f] for f in ("task_type", "language", "source")):
                    raise ValueError(f"{e['id']}: recipe metadata differs from its row")
            self.recipes[entry["sha256"]] = rows
        return self.recipes[entry["sha256"]]

    def count(self, pairs: list[tuple[str, str]]) -> None:
        todo = sorted({p for p in pairs if p not in self.tokens})
        counts = self.count_many([self.pools[pool][rid] for pool, rid in todo])
        self.tokens.update(zip(todo, counts))


def pool_groups(
    inputs: Inputs, pool: str, ids: list[str], cap: int
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Admissible groups (id lists, pool-file order) of these rows and the over-cap drops."""
    wanted = set(ids)
    groups: dict[str, list[str]] = defaultdict(list)
    for rid, row in inputs.pools[pool].items():
        if rid in wanted:
            groups[row["group_id"]].append(rid)
    kept, drops = {}, Counter()
    for group, members in groups.items():
        if max(inputs.tokens[(pool, r)] for r in members) > cap:
            drops["groups"] += 1
            drops["rows"] += len(members)
            drops["tokens"] += sum(inputs.tokens[(pool, r)] for r in members)
            continue
        kept[group] = members
    return kept, dict(drops)


def size(inputs: Inputs, pool: str, members: list[str]) -> int:
    return sum(inputs.tokens[(pool, r)] for r in members)


def partition(inputs: Inputs, spec: dict[str, Any]) -> dict[str, Any]:
    """Every seed's (pool, ids) selection of one family, plus the per-pool accounting."""
    if spec["template"] != TEMPLATE or spec["unit"] != mix.UNIT:
        raise ValueError("Template M6 in the Qwen3 native unit expected")
    seeds, cap, salt = spec["seeds"], spec["max_row_tokens"], spec["partition_salt"]
    base_pool, qks_pools = spec["base_pool"], spec["qks_pools"]
    recipe = inputs.recipe(spec, "recipe")
    shared = inputs.recipe(spec, "shared_with")
    by_pool: dict[str, list[str]] = defaultdict(list)
    for e in recipe:
        by_pool[e["pool"]].append(e["id"])
    shared_ids = {e["id"] for e in shared}
    inputs.count([(e["pool"], e["id"]) for e in recipe])

    base_ids = list(inputs.pools[base_pool])
    if sorted(base_ids) != sorted(by_pool[base_pool]) or not base_ids:
        raise ValueError("The recipe's base rows are not the whole base pool")
    if any(inputs.tokens[(base_pool, r)] > cap for r in base_ids):
        raise ValueError(f"A base row exceeds {cap} tokens")
    base_tokens = size(inputs, base_pool, base_ids)

    selection: list[dict[str, list[str]]] = [{} for _ in range(seeds)]
    qks_report, drops = {}, {}
    qks_tokens = [0] * seeds
    for pool in qks_pools:
        ids = [r for r in by_pool[pool] if r in shared_ids]
        groups, drops[pool] = pool_groups(inputs, pool, ids, cap)
        order = partition_order(list(groups), salt, pool)
        sizes = [size(inputs, pool, groups[g]) for g in order]
        parts = equal_parts(sizes, seeds)
        per_seed_tokens = [0] * seeds
        for group, s, n in zip(order, parts, sizes):
            selection[s].setdefault(pool, []).extend(groups[group])
            per_seed_tokens[s] += n
        for s in range(seeds):
            qks_tokens[s] += per_seed_tokens[s]
        qks_report[pool] = {
            "recipe_rows": len(by_pool[pool]),
            "shared_rows": len(ids),
            "admissible_groups": len(groups),
            "admissible_tokens": sum(sizes),
            "part_tokens": per_seed_tokens,
            "part_rows": [len(selection[s].get(pool, [])) for s in range(seeds)],
            "part_groups": [parts.count(s) for s in range(seeds)],
        }

    rest_pools = sorted(p for p in by_pool if p != base_pool and p not in qks_pools)
    admissible = {}
    for pool in rest_pools:
        admissible[pool], drops[pool] = pool_groups(inputs, pool, by_pool[pool], cap)
    available = {
        p: sum(size(inputs, p, m) for m in admissible[p].values()) for p in rest_pools
    }
    n_rest = sum(available.values())
    budget = spec["budget_tokens"]
    rest_budget = [budget - base_tokens - qks_tokens[s] for s in range(seeds)]
    if any(r <= 0 for r in rest_budget) or sum(rest_budget) > n_rest:
        raise ValueError("The rest budget is not positive or exceeds the rest pools")
    rest_report = {}
    for pool in rest_pools:
        groups = admissible[pool]
        order = partition_order(list(groups), salt, pool)
        sizes = [size(inputs, pool, groups[g]) for g in order]
        targets = [
            Fraction(rest_budget[s] * available[pool], n_rest) for s in range(seeds)
        ]
        parts = sequential_fill(sizes, targets)
        per_seed_tokens = [0] * seeds
        for group, s, n in zip(order, parts, sizes):
            if s is None:
                continue
            selection[s].setdefault(pool, []).extend(groups[group])
            per_seed_tokens[s] += n
        rest_report[pool] = {
            "recipe_rows": len(by_pool[pool]),
            "recipe_tokens": size(inputs, pool, by_pool[pool]),
            "admissible_groups": len(groups),
            "admissible_tokens": available[pool],
            "target_tokens": [round(float(t), 1) for t in targets],
            "part_tokens": per_seed_tokens,
            "part_rows": [len(selection[s].get(pool, [])) for s in range(seeds)],
            "part_groups": [parts.count(s) for s in range(seeds)],
            "unused_groups": parts.count(None),
        }
    return {
        "base_ids": base_ids,
        "base_tokens": base_tokens,
        "qks_tokens": qks_tokens,
        "rest_budget_tokens": rest_budget,
        "rest_pool_tokens": n_rest,
        "selection": selection,
        "qks": qks_report,
        "rest": rest_report,
        "over_cap_dropped": {p: d for p, d in sorted(drops.items()) if d},
    }


def build(
    inputs: Inputs, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    """Rows, token counts and report of the spec's own seed."""
    plan = partition(inputs, spec)
    k = spec["seed"] - 1
    base_pool = spec["base_pool"]
    parts = [(base_pool, plan["base_ids"])]
    parts += [(p, plan["selection"][k].get(p, [])) for p in spec["qks_pools"]]
    parts += [(p, plan["selection"][k].get(p, [])) for p in sorted(plan["rest"])]
    rows, tokens, pools = [], [], []
    seen: set[str] = set()
    repeats: Counter[str] = Counter()
    for pool, ids in parts:
        for rid in ids:
            row = inputs.pools[pool][rid]
            if row["input_sha256"] in seen:
                repeats[pool] += 1
                continue
            seen.add(row["input_sha256"])
            rows.append(row)
            tokens.append(inputs.tokens[(pool, rid)])
            pools.append(pool)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Seed mixture repeats an id")
    total = sum(tokens)
    deviation = (total - spec["budget_tokens"]) / spec["budget_tokens"]
    pool_tokens: Counter[str] = Counter()
    pool_rows: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    for row, n, pool in zip(rows, tokens, pools):
        pool_tokens[pool] += n
        pool_rows[pool] += 1
        language_tokens[row["language"]] += n
    group_keys = sorted(
        {f"{p}\0{r['group_id']}" for r, p in zip(rows, pools) if p != base_pool}
    )
    report = {
        "template": TEMPLATE,
        "unit": mix.UNIT,
        "name": spec["name"],
        "family": spec["family"],
        "seed": spec["seed"],
        "seeds": spec["seeds"],
        "partition_salt": spec["partition_salt"],
        "max_row_tokens": spec["max_row_tokens"],
        "budget_tokens": spec["budget_tokens"],
        "budget_tolerance": spec["budget_tolerance"],
        "recipe": {k2: spec["recipe"][k2] for k2 in ("name", "sha256", "rows")},
        "shared_with": {
            k2: spec["shared_with"][k2] for k2 in ("name", "sha256", "rows")
        },
        "pool_sha256": {p: inputs.pool_sha[p] for p in sorted(pool_rows)},
        "base": {
            "pool": base_pool,
            "rows": len(plan["base_ids"]),
            "tokens": plan["base_tokens"],
        },
        "qks_tokens": plan["qks_tokens"][k],
        "qks_part_tokens": plan["qks_tokens"],
        "rest_budget_tokens": plan["rest_budget_tokens"][k],
        "rest_pool_tokens": plan["rest_pool_tokens"],
        "rest_tokens": sum(
            n
            for n, p in zip(tokens, pools)
            if p != base_pool and p not in spec["qks_pools"]
        ),
        "qks": plan["qks"],
        "rest": plan["rest"],
        "dropped": {
            "over_cap_groups": plan["over_cap_dropped"],
            "repeats_earlier_input": dict(sorted(repeats.items())),
        },
        "pool_rows": dict(sorted(pool_rows.items())),
        "pool_tokens": dict(sorted(pool_tokens.items())),
        "language_tokens": dict(sorted(language_tokens.items())),
        "english_token_share": round(language_tokens["en"] / total, 4),
        "train_rows": len(rows),
        "train_tokens": total,
        "budget_deviation": round(deviation, 6),
        "train_ids_sha256": digest(sorted(ids)),
        "nonbase_groups": len(group_keys),
        "nonbase_group_keys_sha256": digest(group_keys),
        "profile": mix.profile(rows, tokens),
    }
    if abs(deviation) > spec["budget_tolerance"]:
        raise ValueError(
            f"{spec['name']}: {total} tokens is outside the budget tolerance"
        )
    return rows, tokens, report


_COUNT: Any = None


def _init_worker(tokenizer: str) -> None:
    global _COUNT
    _COUNT = mix.token_counter(tokenizer)


def _count_chunk(rows: list[dict[str, Any]]) -> list[int]:
    return [_COUNT(row) for row in rows]


def parallel_counter(tokenizer: str, workers: int) -> CountMany:
    def count_many(rows: list[dict[str, Any]]) -> list[int]:
        if not rows:
            return []
        if workers <= 1:
            _init_worker(tokenizer)
            return _count_chunk(rows)
        chunks = [rows[i : i + 256] for i in range(0, len(rows), 256)]
        with multiprocessing.get_context("fork").Pool(
            workers, _init_worker, (tokenizer,)
        ) as pool:
            return [n for part in pool.imap(_count_chunk, chunks) for n in part]

    return count_many


def disjoint(sets: list[set[str]]) -> bool:
    return all(
        not (sets[i] & sets[j])
        for i in range(len(sets))
        for j in range(i + 1, len(sets))
    )


def verify(
    specs: list[dict[str, Any]],
    built: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]],
    inputs: Inputs,
    teachers: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Cross-seed checks of built seed mixtures (counts only)."""
    checks: dict[str, Any] = {}
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        families[spec["family"]].append(spec)
    base_sets, qks_sets, per_seed = {}, defaultdict(dict), {}
    for spec in specs:
        rows, report = built[spec["name"]]
        ids = [r["id"] for r in rows]
        pool_of = {e["id"]: e["pool"] for e in inputs.recipe(spec, "recipe")}
        tokens = [inputs.tokens[(pool_of[r], r)] for r in ids]
        entry = {
            "rows": len(rows),
            "tokens": sum(tokens),
            "ids_match_report": digest(sorted(ids)) == report["train_ids_sha256"],
            "tokens_match_report": sum(tokens) == report["train_tokens"],
            "within_tolerance": abs(sum(tokens) - spec["budget_tokens"])
            <= spec["budget_tolerance"] * spec["budget_tokens"],
        }
        base_sets[spec["name"]] = {r for r in ids if pool_of[r] == spec["base_pool"]}
        qks_sets[spec["seed"]][spec["family"]] = {
            r for r in ids if pool_of[r] in spec["qks_pools"]
        }
        teacher = teachers.get(spec["family"])
        if teacher is not None:
            missing: Counter[str] = Counter()
            for row in rows:
                t = teacher.get(row["id"])
                if (
                    t is None
                    or t["input_sha256"] != row["input_sha256"]
                    or set(t["teacher_probs"]) != {o["key"] for o in row["options"]}
                ):
                    missing[pool_of[row["id"]]] += 1
            entry["teacher_covered"] = len(rows) - sum(missing.values())
            entry["teacher_missing_by_pool"] = dict(sorted(missing.items()))
        per_seed[spec["name"]] = entry
    checks["per_seed"] = per_seed
    base_pool_ids = set().union(*base_sets.values())
    checks["base_identical"] = all(s == base_pool_ids for s in base_sets.values())
    checks["base_rows"] = len(base_pool_ids)
    checks["qks_identical_across_families"] = {
        str(k): len({frozenset(v) for v in fams.values()}) == 1 and len(fams) > 1
        for k, fams in sorted(qks_sets.items())
    }
    checks["families"] = {}
    for family, members in sorted(families.items()):
        members = sorted(members, key=lambda s: s["seed"])
        spec0 = members[0]
        pool_of = {e["id"]: e["pool"] for e in inputs.recipe(spec0, "recipe")}
        nonbase = []
        groups = []
        for spec in members:
            rows, _ = built[spec["name"]]
            nb = [r for r in rows if pool_of[r["id"]] != spec0["base_pool"]]
            nonbase.append({r["id"] for r in nb})
            groups.append({f"{pool_of[r['id']]}\0{r['group_id']}" for r in nb})
        union = set().union(*nonbase) | base_sets[spec0["name"]]
        recipe = inputs.recipe(spec0, "recipe")
        by_pool_rows: Counter[str] = Counter(e["pool"] for e in recipe)
        by_pool_tokens: Counter[str] = Counter()
        used_rows: Counter[str] = Counter()
        used_tokens: Counter[str] = Counter()
        for e in recipe:
            n = inputs.tokens[(e["pool"], e["id"])]
            by_pool_tokens[e["pool"]] += n
            if e["id"] in union:
                used_rows[e["pool"]] += 1
                used_tokens[e["pool"]] += n
        checks["families"][family] = {
            "seeds": [s["name"] for s in members],
            "nonbase_ids_disjoint": disjoint(nonbase),
            "nonbase_groups_disjoint": disjoint(groups),
            "group_ids_disjoint_across_pools": disjoint(
                [{g.split("\0", 1)[1] for g in gs} for gs in groups]
            ),
            "union_rows": len(union),
            "recipe_rows": len(recipe),
            "union_row_share": round(len(union) / len(recipe), 4),
            "union_token_share": round(
                sum(used_tokens.values()) / sum(by_pool_tokens.values()), 4
            ),
            "recipe_tokens": sum(by_pool_tokens.values()),
            "union_by_pool": {
                p: {
                    "rows": used_rows[p],
                    "of_rows": by_pool_rows[p],
                    "tokens": used_tokens[p],
                    "of_tokens": by_pool_tokens[p],
                }
                for p in sorted(by_pool_rows)
            },
        }
    checks["pass"] = (
        checks["base_identical"]
        and all(checks["qks_identical_across_families"].values())
        and all(
            e["ids_match_report"] and e["tokens_match_report"] and e["within_tolerance"]
            for e in per_seed.values()
        )
        and all(
            f["nonbase_ids_disjoint"] and f["nonbase_groups_disjoint"]
            for f in checks["families"].values()
        )
    )
    return checks


def load_teacher(item: str) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
    family, rest = item.split("=", 1)
    path, expected = rest.rsplit("=", 1)
    if file_sha256(path) != expected:
        raise ValueError(f"{family}: teacher file differs from its frozen hash")
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["id"] in entries:
            raise ValueError(f"{family}: repeated teacher id")
        entries[entry["id"]] = entry
    return family, entries, {"sha256": expected, "entries": len(entries)}


def main() -> None:
    from training.model.data import canonical, load_partition

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--spec", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--teacher", action="append", default=[], help="FAMILY=PATH=SHA256"
    )
    parser.add_argument("--rights-clean", type=Path)
    parser.add_argument("--converter-bundle", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    specs = [json.loads(p.read_text()) for p in args.spec]
    if len({s["tokenizer"] for s in specs}) != 1:
        raise ValueError("Specs name different tokenizers")
    inputs = Inputs(parallel_counter(specs[0]["tokenizer"], args.workers))
    if args.command == "build":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, spec in zip(args.spec, specs):
            output = args.output_dir / f"{spec['name']}.train.jsonl"
            report_path = args.output_dir / f"{spec['name']}.report.json"
            if output.exists() or report_path.exists():
                raise FileExistsError("refusing to overwrite a materialized mixture")
            rows, _, report = build(inputs, spec)
            pending = output.with_name(output.name + ".pending")
            with pending.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(canonical(row) + "\n")
            pending.replace(output)
            report["spec_sha256"] = file_sha256(path)
            report["rows_file_sha256"] = file_sha256(output)
            write_json(report_path, report, exclusive=True)
            print(
                json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "name",
                            "train_rows",
                            "train_tokens",
                            "budget_deviation",
                            "train_ids_sha256",
                            "rows_file_sha256",
                        )
                    }
                ),
                flush=True,
            )
        return
    if args.report is None:
        raise ValueError("verify needs --report")
    built = {}
    for path, spec in zip(args.spec, specs):
        report = json.loads(
            (args.output_dir / f"{spec['name']}.report.json").read_text()
        )
        rows_path = args.output_dir / f"{spec['name']}.train.jsonl"
        if file_sha256(rows_path) != report["rows_file_sha256"]:
            raise ValueError(f"{spec['name']}: rows file differs from its report")
        if report["spec_sha256"] != file_sha256(path):
            raise ValueError(f"{spec['name']}: built from a different spec")
        built[spec["name"]] = (load_partition(rows_path, "train"), report)
        inputs.recipe(spec, "recipe")
        inputs.recipe(spec, "shared_with")
        inputs.count([(e["pool"], e["id"]) for e in inputs.recipe(spec, "recipe")])
    teachers, teacher_files = {}, {}
    for item in args.teacher:
        family, entries, meta = load_teacher(item)
        teachers[family], teacher_files[family] = entries, meta
    result = verify(specs, built, inputs, teachers)
    result["teachers"] = teacher_files
    result["mixtures"] = {
        name: {
            "sha256": report["rows_file_sha256"],
            "spec_sha256": report["spec_sha256"],
        }
        for name, (_, report) in built.items()
    }
    if args.rights_clean is not None:
        from training.model.data import check_partition_isolation

        from .common import load_rights_clean

        splits = load_rights_clean(args.rights_clean)
        for name, (rows, _) in built.items():
            check_partition_isolation(
                {"train": rows, "select": splits["select"], "cal": splits["cal"]}
            )
        result["trainer_isolation_vs_select_cal"] = "pass"
    if args.converter_bundle is not None:
        from .common import native_records

        union = {r["id"]: r for rows, _ in built.values() for r in rows}
        records = native_records(list(union.values()), args.converter_bundle)
        result["native_conversion"] = {
            "rows": len(records),
            "task_types": dict(
                sorted(Counter(r["question"]["type"].lower() for r in records).items())
            ),
        }
    write_json(args.report, result, exclusive=True)
    print(json.dumps({"pass": result["pass"], "report": str(args.report)}))


if __name__ == "__main__":
    main()
