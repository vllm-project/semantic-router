"""Decoder M8 top-up data (prereg dec-m8-prereg-2026-09-30.md, "Data" and "Teacher targets").

One deterministic CPU pass (node B):

- base: N4XF's mixture minus every row of the quarantined groups (the M7 4B base);
- slice: whole base groups chosen by `build_template_s.select_groups` (strata source x task type x language,
  seed-keyed hash order, each stratum's share of the budget), kept in base order;
- pools: each row's recipe pool from the N4XF spec's id files (first component wins, as in the builder);
- S: the frozen human-rated rule (`specs/m8-human-rated-rule.json`: the 9B M6 soft-target rule plus a pool alias);
- C teacher: N4XF's own-Lux records of the slice rows, copied unchanged (rows without one stay gold-only);
- prompts: each slice row as one gold-free prompt {id, state, questions: {q: {type, instructions, criteria}}} that
  `question_to_row` renders to the same prompt text and option keys as the row (checked for every row); a row that
  does not convert is left out of the prompts and reported;
- shards: the prompts split by position modulo --shards, for the node-A label jobs.

Outputs (a new directory): slice/train.jsonl, slice/train.ids.jsonl ({id, pool, class}), teacher/C/teacher.jsonl,
label/prompts.jsonl, label/shard-<k>.prompts.jsonl, manifest.json.

usage (launch.sh --cpu): python3 v2/dec/ops/m8/m8_data.py --base B --base-sha S --spec SPEC --spec-sha S
    --root mx=DIR --root xl=DIR --quarantine Q --lux-teacher T --lux-teacher-sha S --rule R --tokenizer TOK
    --budget 6200000 --seed dec-m8-topup-v1 --isolation select=F --isolation cal=F --panel-prompts NAME=F ...
    --shards 3 --workers 40 --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "v2" / "9b"))

from lux9b.m3_data import c1_keys, denied_hits  # noqa: E402
from training.model.data import (  # noqa: E402
    canonical,
    check_partition_isolation,
    file_sha256,
    load_partition,
)
from training.model.decision_model import segments  # noqa: E402
from training.model.infer import question_to_row  # noqa: E402
from v2.common import eval_only  # noqa: E402
from v2.dec.build_template_s import group_rows, select_groups  # noqa: E402

SCHEMA = "dec-m8-data/1"
QUESTION = "q"
C1_REGISTRY = CODE / "v2/eval/records/sealed-c1-source-registry-2026-09-28.json"
Row = dict[str, Any]


def read_jsonl(path: Path) -> list[Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return file_sha256(path)


def verified(path: Path, sha: str) -> Path:
    got = file_sha256(path)
    if got != sha:
        raise ValueError(f"{path}: sha256 {got} differs from {sha}")
    return path


def human_rated(row: Row, pool: str, rule_doc: dict[str, Any]) -> bool:
    """The 9B M6 rule (`lux9b.m6_data.human_rated`) with the 4B pool alias applied first."""
    rule = rule_doc["soft_target_rule"]
    pool = rule_doc.get("pool_alias", {}).get(pool, pool)
    if pool not in rule["pools"]:
        raise ValueError(f"pool {pool} is not named by the soft-target rule")
    sources = rule["human_sources"].get(pool, [])
    if row["source"] == rule["replay_source"]:
        return row["source"] in sources and "upstream_label" in row
    return "*" in sources or row["source"] in sources


def pool_map(
    spec: dict[str, Any], roots: dict[str, Path]
) -> tuple[dict[str, str], dict[str, str]]:
    """id -> recipe pool (component name) from the spec's id files, first component wins."""
    pools: dict[str, str] = {}
    inputs: dict[str, str] = {}
    cache: dict[str, list[dict[str, Any]]] = {}
    for component in spec["components"]:
        ids = component.get("ids")
        if ids is None:
            raise ValueError(f"{component['name']}: component without an id file")
        source, relative = ids["file"].split(":", 1)
        path = roots[source] / relative
        if ids["file"] not in cache:
            eval_only.check_file(path)
            inputs[ids["file"]] = file_sha256(verified(path, ids["sha256"]))
            cache[ids["file"]] = read_jsonl(path)
        for entry in cache[ids["file"]]:
            if entry["pool"] == ids["pool"]:
                pools.setdefault(entry["id"], component["name"])
    return pools, inputs


def to_prompt(row: Row) -> dict[str, Any]:
    """One gold-free prompt whose single question renders exactly as the TRAIN row."""
    kind = row["task_type"]
    keys = [option["key"] for option in row["options"]]
    if kind == "score":
        if keys != [str(index) for index in range(len(keys))]:
            raise ValueError("score option keys are not 0..n-1 in order")
        criteria: Any = [option["description"] for option in row["options"]]
    else:
        criteria = {option["key"]: option["description"] for option in row["options"]}
        if len(criteria) != len(keys):
            raise ValueError("repeated option key")
    prompt = {
        "id": row["id"],
        "state": row["state"],
        "questions": {
            QUESTION: {
                "type": kind,
                "instructions": row["instructions"],
                "criteria": criteria,
            }
        },
    }
    back = question_to_row(prompt, QUESTION, prompt["questions"][QUESTION])
    if [option["key"] for option in back["options"]] != keys:
        raise ValueError("option keys differ after conversion")
    if segments(back) != segments(row):
        raise ValueError("rendered prompt differs after conversion")
    # The collector reads the prompt back from JSON: the round trip must keep the rendering.
    reread = json.loads(json.dumps(prompt, ensure_ascii=False))
    if segments(
        question_to_row(reread, QUESTION, reread["questions"][QUESTION])
    ) != segments(row):
        raise ValueError("rendered prompt differs after a JSON round trip")
    return prompt


def state_digest(state: Any) -> str:
    return hashlib.sha256(canonical(state).encode("utf-8")).hexdigest()


def summary(rows: list[Row], length: dict[str, int]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "tokens": sum(length[r["id"]] for r in rows),
        "rows_by_type": dict(sorted(Counter(r["task_type"] for r in rows).items())),
        "rows_by_language": dict(Counter(r["language"] for r in rows).most_common(12)),
    }


def build(
    base: list[Row],
    length: dict[str, int],
    pools: dict[str, str],
    rule: dict[str, Any],
    lux: dict[str, dict[str, Any]],
    *,
    quarantine: set[str],
    budget: int,
    seed: str,
    shards: int,
) -> dict[str, Any]:
    kept = [r for r in base if r["group_id"] not in quarantine]
    groups = group_rows(kept)
    gtok = {g: sum(length[r["id"]] for r in rows) for g, rows in groups.items()}
    chosen = set(select_groups(groups, gtok, budget, seed))
    slice_rows = [r for r in kept if r["group_id"] in chosen]
    missing = sorted({r["id"] for r in slice_rows} - set(pools))
    if missing:
        raise ValueError(
            f"{len(missing)} slice rows have no recipe pool, e.g. {missing[:3]}"
        )
    ids, prompts, failed = [], [], {}
    for row in slice_rows:
        pool = pools[row["id"]]
        cls = "human" if human_rated(row, pool, rule) else "typed"
        ids.append({"id": row["id"], "pool": pool, "class": cls})
        try:
            prompts.append(to_prompt(row))
        except ValueError as exc:
            failed[row["id"]] = str(exc)
    teacher_c = [lux[r["id"]] for r in slice_rows if r["id"] in lux]
    by_id = {r["id"]: r for r in slice_rows}
    for record in teacher_c:
        if not {"id", "input_sha256", "teacher_probs"} <= set(record):
            raise ValueError(f"{record.get('id')}: malformed own-Lux record")
        row = by_id[record["id"]]
        if record["input_sha256"] != row["input_sha256"]:
            raise ValueError(f"{row['id']}: own-Lux input hash differs")
        if set(record["teacher_probs"]) != {o["key"] for o in row["options"]}:
            raise ValueError(f"{row['id']}: own-Lux keys differ from options")
    cls_of = {e["id"]: e["class"] for e in ids}
    pool_of = {e["id"]: e["pool"] for e in ids}
    shard_prompts = [prompts[k::shards] for k in range(shards)]
    return {
        "kept": kept,
        "slice": slice_rows,
        "ids": ids,
        "prompts": prompts,
        "shards": shard_prompts,
        "failed": failed,
        "teacher_c": teacher_c,
        "stats": {
            "base": summary(kept, length),
            "quarantined_rows": len(base) - len(kept),
            "slice": summary(slice_rows, length),
            "slice_groups": len(chosen),
            "slice_by_class": {
                cls: summary([r for r in slice_rows if cls_of[r["id"]] == cls], length)
                for cls in ("human", "typed")
            },
            "slice_by_pool": {
                pool: {
                    "rows": n,
                    "tokens": sum(
                        length[r["id"]] for r in slice_rows if pool_of[r["id"]] == pool
                    ),
                    "human_rows": sum(
                        1
                        for r in slice_rows
                        if pool_of[r["id"]] == pool and cls_of[r["id"]] == "human"
                    ),
                }
                for pool, n in sorted(Counter(pool_of.values()).items())
            },
            "prompts": len(prompts),
            "prompt_conversion_failures": len(failed),
            "teacher_c_rows": len(teacher_c),
            "teacher_c_gold_only_rows": len(slice_rows) - len(teacher_c),
            "gold_only_by_pool": dict(
                sorted(
                    Counter(
                        pool_of[r["id"]] for r in slice_rows if r["id"] not in lux
                    ).items()
                )
            ),
            "shard_prompts": [len(s) for s in shard_prompts],
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--base-sha", required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--spec-sha", required=True)
    p.add_argument(
        "--root",
        action="append",
        required=True,
        help="NAME=DIR for the spec's id files",
    )
    p.add_argument("--quarantine", type=Path, required=True)
    p.add_argument("--lux-teacher", type=Path, required=True)
    p.add_argument("--lux-teacher-sha", required=True)
    p.add_argument("--rule", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--seed", required=True)
    p.add_argument(
        "--isolation", action="append", default=[], help="select=F / cal=F partitions"
    )
    p.add_argument(
        "--panel-prompts",
        action="append",
        default=[],
        help="NAME=gold-free prompt file",
    )
    p.add_argument("--shards", type=int, default=3)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    # Panel prompts and SELECT / CAL are read only for the isolation checks; every training input is guarded.
    eval_only.guard(
        {k: v for k, v in vars(a).items() if k not in ("isolation", "panel_prompts")}
    )
    if a.output.exists():
        raise FileExistsError(a.output)
    roots = {k: Path(v) for k, v in (r.split("=", 1) for r in a.root)}
    base = load_partition(verified(a.base, a.base_sha), "train")
    spec = json.loads(verified(a.spec, a.spec_sha).read_text())
    eval_only.guard(spec)
    pools, id_inputs = pool_map(spec, roots)
    rule = json.loads(a.rule.read_text())
    quarantine = set(json.loads(a.quarantine.read_text())["group_ids"])
    lux = {r["id"]: r for r in read_jsonl(verified(a.lux_teacher, a.lux_teacher_sha))}
    from v2.dec.build_mixture import token_lengths

    length = dict(
        zip((r["id"] for r in base), token_lengths(base, a.tokenizer, a.workers))
    )
    out = build(
        base,
        length,
        pools,
        rule,
        lux,
        quarantine=quarantine,
        budget=a.budget,
        seed=a.seed,
        shards=a.shards,
    )
    eval_only.check_rows(out["slice"])
    partitions = {"train": out["slice"]}
    for item in a.isolation:
        name, path = item.split("=", 1)
        partitions[name] = load_partition(Path(path), name)
    check_partition_isolation(partitions)
    slice_states = {state_digest(r["state"]) for r in out["slice"]}
    panel_hits = {}
    for item in a.panel_prompts:
        name, path = item.split("=", 1)
        states = {state_digest(r["state"]) for r in read_jsonl(Path(path))}
        panel_hits[name] = {
            "prompts_file_sha256": file_sha256(Path(path)),
            "shared_states": len(states & slice_states),
        }
    if any(v["shared_states"] for v in panel_hits.values()):
        raise ValueError(f"slice shares states with a development panel: {panel_hits}")
    keys = c1_keys(json.loads(C1_REGISTRY.read_text()))
    sources = sorted({r["source"] for r in out["slice"]})
    families = sorted({r["family"] for r in out["slice"]})
    c1_sources = {s: h for s in sources if (h := denied_hits(s, keys, []))}
    c1_families = {f: h for f in families if (h := denied_hits(f, keys, []))}
    if c1_sources:
        raise ValueError(f"C1 registry source hits: {sorted(c1_sources)}")
    d = a.output
    files = {
        "slice/train.jsonl": write_jsonl(d / "slice/train.jsonl", out["slice"]),
        "slice/train.ids.jsonl": write_jsonl(d / "slice/train.ids.jsonl", out["ids"]),
        "teacher/C/teacher.jsonl": write_jsonl(
            d / "teacher/C/teacher.jsonl", out["teacher_c"]
        ),
        "label/prompts.jsonl": write_jsonl(d / "label/prompts.jsonl", out["prompts"]),
    }
    for k, shard in enumerate(out["shards"]):
        files[f"label/shard-{k}.prompts.jsonl"] = write_jsonl(
            d / f"label/shard-{k}.prompts.jsonl", shard
        )
    if out["failed"]:
        files["label/conversion-failures.jsonl"] = write_jsonl(
            d / "label/conversion-failures.jsonl",
            [{"id": k, "reason": v} for k, v in sorted(out["failed"].items())],
        )
    manifest = {
        "schema": SCHEMA,
        "prereg": "dec-m8-prereg-2026-09-30.md",
        "inputs": {
            "base": {"path": str(a.base), "sha256": a.base_sha},
            "spec": {"path": str(a.spec), "sha256": a.spec_sha},
            "id_files": id_inputs,
            "quarantine": {
                "path": str(a.quarantine),
                "sha256": file_sha256(a.quarantine),
                "groups": sorted(quarantine),
            },
            "own_lux_teacher": {
                "path": str(a.lux_teacher),
                "sha256": a.lux_teacher_sha,
            },
            "rule": {"path": str(a.rule), "sha256": file_sha256(a.rule)},
            "tokenizer": str(a.tokenizer),
        },
        "budget_tokens": a.budget,
        "seed": a.seed,
        "shards": a.shards,
        "isolation": sorted(partitions),
        "panel_state_overlap": panel_hits,
        "c1_registry": {
            "sha256": file_sha256(C1_REGISTRY),
            "keys": len(keys),
            "source_hits": c1_sources,
            "family_name_hits_reported_only": c1_families,
        },
        "files_sha256": files,
        **out["stats"],
    }
    (d / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "slice": out["stats"]["slice"],
                "by_class": {
                    k: v["tokens"] for k, v in out["stats"]["slice_by_class"].items()
                },
                "prompt_failures": len(out["failed"]),
                "teacher_c_rows": len(out["teacher_c"]),
                "train_sha256": files["slice/train.jsonl"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
