"""Count M6 training rows per source over the six seed mixtures (node A, read-only).

Run on node A with `python3 -c`; prints one JSON document of counts, identifiers and
SHA-256 values only (never row text). Rows are keyed by `id`; a row that appears in
several seed files (BASE, shared QKS thirds) counts once in the union.
"""

import collections
import hashlib
import json

D = "/data/dev2/runs/06b/m6/data/"
FILES = [f"m6-{f}-s{k}.train.jsonl" for f in ("cx", "mx") for k in (1, 2, 3)]
RECIPES = {
    "cx-xl-r2-a7v1-full": "inputs/recipes/cx-xl-r2-a7v1-full.ids.jsonl",
    "cx-xl-r2-nogap-full": "inputs/recipes/cx-xl-r2-nogap-full.ids.jsonl",
}
TEACHERS = {"m6-cx": "m6-cx.lux1.jsonl", "m6-mx": "m6-mx.lux1.jsonl"}
OTHER = ["inputs/INPUTS.json", "m6-verify.json"]
SPLIT = {"CLINC150": "clinc150", "PolyAI-LDN/banking77": "banking77"}


def sha256(rel):
    h = hashlib.sha256()
    with open(D + rel, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def origin_label(am):
    o = am.get("original_source")
    if o is None and isinstance(am.get("a7"), dict):
        o = am["a7"].get("original_source")
    if isinstance(o, dict):
        inner = o.get("original_source")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, dict):
            o = inner
        return o.get("dataset") or o.get("generator")
    return o if isinstance(o, str) else None


def part_of(source, label):
    if source != "legacy:stage3_replay":
        return source
    for prefix, name in SPLIT.items():
        if label and label.startswith(prefix):
            return f"{source}#{name}"
    return f"{source}#project-generated"


checks = collections.Counter()
recipes, recipe_pool, recipe_source = {}, {}, {}
for name, rel in RECIPES.items():
    n = 0
    for line in open(D + rel):
        r = json.loads(line)
        n += 1
        i = r["id"]
        if i in recipe_pool and (recipe_pool[i], recipe_source[i]) != (
            r["pool"],
            r["source"],
        ):
            checks["recipe_pool_or_source_conflict"] += 1
        recipe_pool[i], recipe_source[i] = r["pool"], r["source"]
    recipes[name] = {"file": rel, "sha256": sha256(rel), "rows": n}

rows, files = {}, []
per_file = collections.defaultdict(collections.Counter)
fam_ids = {"m6-cx": set(), "m6-mx": set()}
labels = collections.defaultdict(collections.Counter)
for fn in FILES:
    fam, n, ids = fn[:5], 0, set()
    for line in open(D + fn):
        r = json.loads(line)
        n += 1
        i = r["id"]
        ids.add(i)
        fam_ids[fam].add(i)
        label = origin_label(r["audit_metadata"])
        part = part_of(r["source"], label)
        per_file[part][fn] += 1
        if i in rows:
            if (rows[i]["input_sha256"], rows[i]["part"]) != (r["input_sha256"], part):
                checks["same_id_differs_across_files"] += 1
            continue
        labels[r["source"]][label or "<none>"] += 1
        pool = recipe_pool.get(i)
        if pool is None:
            checks["id_not_in_either_recipe"] += 1
        elif recipe_source[i] != r["source"]:
            checks["source_differs_from_recipe"] += 1
        rows[i] = {
            "part": part,
            "pool": pool,
            "task": r["task_type"],
            "lang": r["language"],
            "family": r["family"],
            "group": r["group_id"],
            "input_sha256": r["input_sha256"],
        }
    files.append(
        {"file": fn, "sha256": sha256(fn), "rows": n, "distinct_ids": len(ids)}
    )

teachers, teacher = {}, {}
for fam, rel in TEACHERS.items():
    n, ids = 0, set()
    for line in open(D + rel):
        r = json.loads(line)
        n += 1
        i = r["id"]
        ids.add(i)
        probs = hashlib.sha256(
            json.dumps(r["teacher_probs"], sort_keys=True).encode()
        ).hexdigest()
        key = (r["input_sha256"], probs)
        if i in teacher:
            checks["teacher_ids_in_both_files"] += 1
            if teacher[i] != key:
                checks["teacher_shared_id_differs"] += 1
        else:
            teacher[i] = key
    teachers[fam] = {
        "file": rel,
        "sha256": sha256(rel),
        "entries": n,
        "distinct_ids": len(ids),
        "ids_equal_family_union": ids == fam_ids[fam],
    }
checks["train_union_rows_without_teacher"] = sum(1 for i in rows if i not in teacher)
checks["teacher_ids_outside_train_union"] = sum(1 for i in teacher if i not in rows)
checks["teacher_input_sha256_mismatch"] = sum(
    1 for i, v in rows.items() if i in teacher and teacher[i][0] != v["input_sha256"]
)

parts = {}
pools = collections.defaultdict(lambda: {"union": 0, "m6-cx": 0, "m6-mx": 0})
for i, v in rows.items():
    p = parts.setdefault(
        v["part"],
        {
            "rows_union": 0,
            "rows_family_union": {"m6-cx": 0, "m6-mx": 0},
            "pools": collections.Counter(),
            "task_types": collections.Counter(),
            "languages": collections.Counter(),
            "row_families": collections.Counter(),
        },
    )
    p["rows_union"] += 1
    p["pools"][v["pool"]] += 1
    p["task_types"][v["task"]] += 1
    p["languages"][v["lang"]] += 1
    p["row_families"][v["family"]] += 1
    pools[v["pool"]]["union"] += 1
    for fam, ids in fam_ids.items():
        if i in ids:
            p["rows_family_union"][fam] += 1
            pools[v["pool"]][fam] += 1
for name, p in parts.items():
    p["rows_per_file"] = {fn: per_file[name][fn] for fn in FILES}
    p["rows_per_file_sum"] = sum(per_file[name].values())
    for k in ("pools", "task_types", "languages", "row_families"):
        p[k] = dict(sorted(p[k].items()))

print(
    json.dumps(
        {
            "schema": "dev2-m6-union-counts/1",
            "data_dir": D.rstrip("/"),
            "files": files,
            "recipes": recipes,
            "teachers": teachers,
            "other_inputs": {rel: sha256(rel) for rel in OTHER},
            "union": {
                "distinct_ids": len(rows),
                "distinct_input_sha256": len(
                    {v["input_sha256"] for v in rows.values()}
                ),
                "distinct_group_ids": len({v["group"] for v in rows.values()}),
                "rows_per_file_sum": sum(f["rows"] for f in files),
                "family_union_ids": {fam: len(ids) for fam, ids in fam_ids.items()},
                "ids_in_both_families": len(fam_ids["m6-cx"] & fam_ids["m6-mx"]),
                "task_types": dict(
                    sorted(
                        collections.Counter(v["task"] for v in rows.values()).items()
                    )
                ),
                "languages": dict(
                    sorted(
                        collections.Counter(v["lang"] for v in rows.values()).items()
                    )
                ),
            },
            "pools": dict(sorted(pools.items())),
            "parts": dict(sorted(parts.items())),
            "origin_labels": {
                s: dict(sorted(c.items())) for s, c in sorted(labels.items())
            },
            "checks": dict(sorted(checks.items())),
        },
        indent=1,
        sort_keys=False,
    )
)
