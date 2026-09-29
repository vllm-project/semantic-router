"""Data-lock checks for decoder M5 on node B (prereg preflights 1-3), host python from the mirror.

Phase n5n: N4XF mixture identity, excluded groups, own-Nox label checks (whole-batch re-label bitwise,
M3 overlap argmax >= 0.97), label rows == N4XF block rows, N5N teacher coverage (all rows but H7; Nox on
exactly the block rows). Phase block: N5B budget / quotas / outside-block identity / MLX-DEV disjointness
(groups, ids and segments under the amended template threshold) for N4XF and N5B, the added-row label
checks, N5B / N5BN teacher coverage, and a per-arm status (N5B, N5BN, MLX-DEV). Writes
/data/dev2/runs/dec/m5/lock-<phase>.json; exit 1 on any failure.

usage: PYTHONPATH=<mirror>/src/training/decision2 python3 m5-lockcheck.py n5n
       PYTHONPATH=<mirror>/src/training/decision2 python3 m5-lockcheck.py block <template-max-groups>
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from v2.dec.m5_block import (
    BUDGET_TOKENS,
    QUOTAS,
    QUOTA_TOLERANCE,
    component_slices,
    read_lines,
    segments,
)
from v2.dec.m5_labels import is_block

from training.model.data import file_sha256

M = Path("/data/dev2/runs/dec/m5")
N4 = Path("/data/dev2/runs/dec/m4/data/m4-xl-full-29m/train.jsonl")
N4_SHA = "c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60"
LUX_SHA = "e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c"
LUXT = Path("/data/dev2/runs/dec/m4/teacher/m4-xl-full-29m/lux-teacher.jsonl")
EXCL = Path(sys.argv[0]).resolve().parents[2] / "specs" / "m4-r2-excluded-groups.json"
phase = sys.argv[1]


def load(path):
    man = json.loads(path.with_name(path.name + ".manifest.json").read_text())
    return man, [
        (name, line, row)
        for name, items in component_slices(read_lines(path), man)
        for line, row in items
    ]


def jload(path):
    return json.loads(Path(path).read_text())


def coverage(train, teacher, first_ids):
    """Rows lacking a target must be exactly the H7 rows; ``first_ids`` rows must take source 0."""
    targets = {}
    for line in open(teacher, encoding="utf-8"):
        r = json.loads(line)
        targets[r["id"]] = r["input_sha256"]
    missing = [(c, r["id"]) for c, _, r in train if r["id"] not in targets]
    bad_hash = sum(
        targets.get(r["id"], r["input_sha256"]) != r["input_sha256"]
        for _, _, r in train
    )
    man = jload(str(teacher) + ".manifest.json")
    out = {
        "teacher_sha256": file_sha256(teacher),
        "manifest_output_sha256": man["output_sha256"],
        "rows": len(train),
        "covered": len(train) - len(missing),
        "missing_by_component": dict(Counter(c for c, _ in missing)),
        "h7_rows": sum(c == "H7" for c, _, _ in train),
        "input_hash_mismatch": bad_hash,
        "used_by_source": [
            (Path(s["file"]).name, s.get("used", 0)) for s in man["sources"]
        ],
        "overlap": man["overlap"],
        "train_label_agreement": {
            k: {t: round(v["accuracy"], 4) for t, v in a.items()}
            for k, a in man["train_label_agreement"].items()
        },
    }
    out["ok"] = (
        set(out["missing_by_component"]) <= {"H7"}
        and out["missing_by_component"].get("H7", 0) == out["h7_rows"]
        and bad_hash == 0
        and man["output_sha256"] == out["teacher_sha256"]
    )
    if first_ids is not None:
        used0 = man["sources"][0].get("used", 0)
        out["nox_rows_expected"] = len(first_ids)
        out["nox_rows_used"] = used0
        out["ok"] = out["ok"] and used0 == len(first_ids)
    return out


result = {"phase": phase}
fails = []
excluded = set(jload(EXCL)["group_ids"])
n4_man, n4 = load(N4)
n4_block = {r["id"] for c, _, r in n4 if is_block(c, r)}
if phase == "n5n":
    sha = file_sha256(N4)
    result["mixture"] = {
        "train": str(N4),
        "train_sha256": sha,
        "rows": len(n4),
        "tokens": n4_man["tokens"],
        "tokens_vs_budget": n4_man["tokens"] / BUDGET_TOKENS - 1,
        "block_rows": len(n4_block),
        "excluded_group_rows": sum(r["group_id"] in excluded for _, _, r in n4),
    }
    if sha != N4_SHA or result["mixture"]["excluded_group_rows"]:
        fails.append("mixture")
    if file_sha256(LUXT) != LUX_SHA:
        fails.append("n4xf lux teacher hash")
    rows_man = jload(M / "teacher/nox-rows-n5n/rows.jsonl.manifest.json")
    labels = M / "teacher/nox-n5n/labels.jsonl"
    label_ids = {json.loads(x)["id"] for x in open(labels)}
    lab_man = jload(str(labels) + ".manifest.json")
    result["labels"] = {
        "rows_file_sha256": rows_man["output_sha256"],
        "rows": rows_man["rows"],
        "labels_sha256": file_sha256(labels),
        "labels_rows": len(label_ids),
        "equals_block": label_ids == n4_block,
        "teacher": [
            lab_man["teacher_repo"],
            lab_man["teacher_revision"],
            lab_man["teacher_kind"],
        ],
        "temperatures": lab_man["teacher_temperatures"],
        "max_length": lab_man["max_length"],
        "seconds": lab_man["seconds"],
        "gold_agreement": {
            k: round(v["accuracy"], 4)
            for k, v in lab_man["train_label_agreement"].items()
        },
        "check_sample": jload(M / "teacher/nox-check-n5n/check.jsonl.manifest.json"),
        "bitwise": jload(M / "teacher/checks-n5n-bitwise/bitwise.json"),
        "m3_overlap": jload(M / "teacher/checks-n5n-overlap/overlap.json"),
    }
    if not result["labels"]["equals_block"]:
        fails.append("labels != block rows")
    if not result["labels"]["bitwise"]["pass"]:
        fails.append("bitwise re-label")
    if not result["labels"]["m3_overlap"]["pass"]:
        fails.append("M3 overlap agreement")
    result["teacher_n5n"] = coverage(n4, M / "teacher/n5n/teacher.jsonl", n4_block)
    if not result["teacher_n5n"]["ok"]:
        fails.append("N5N coverage")
    result["hw1"] = (M / "data/hw1.sha256").read_text().split("\n")
    result["diag_n4xf_sha256"] = file_sha256(M / "teacher/diag-n4xf/diag.json")
else:
    tmg = int(sys.argv[2])
    n5b_path = M / "data/n5b/train.jsonl"
    n5b_man, n5b = load(n5b_path)
    n4_lines = {r["id"]: line for c, line, r in n4}
    outside_n4 = [r["id"] for c, _, r in n4 if not is_block(c, r)]
    outside_n5b = [r["id"] for c, _, r in n5b if not is_block(c, r)]
    identical = outside_n4 == outside_n5b and all(
        n4_lines[r["id"]] == line for c, line, r in n5b if not is_block(c, r)
    )
    block_n5b = {r["id"] for c, _, r in n5b if is_block(c, r)}
    # Independent quota recount from the manifest's cell table
    cells = n5b_man["stats"]["block"]["cells"]
    btot = n5b_man["stats"]["block"]["n4xf_block_tokens"]
    quotas = {
        c: {
            "tokens": v["tokens"],
            "share": v["tokens"] / btot,
            "rel_dev": v["tokens"] / (QUOTAS[c] * btot) - 1,
        }
        for c, v in cells.items()
    }
    # MLX-DEV disjointness, recomputed from the panel
    panel = [json.loads(x) for x in open(M / "mlxdev/build/panel.jsonl")]
    mlx_groups = {r["group_id"] for r in panel}
    mlx_ids = {r["id"] for r in panel}
    freq = Counter()
    by_group = defaultdict(set)
    for _, _, r in n4:
        by_group[r["group_id"]] |= segments(r)
    for s in by_group.values():
        freq.update(s)
    ignored = {s for s, n in freq.items() if n > tmg}
    mlx_segs = set()
    for r in panel:
        mlx_segs |= segments(r) - ignored

    def disjoint(rows):
        return {
            "mlx_groups": sum(r["group_id"] in mlx_groups for _, _, r in rows),
            "mlx_ids": sum(r["id"] in mlx_ids for _, _, r in rows),
            "segment_conflict_rows": sum(
                bool(segments(r) & mlx_segs) for _, _, r in rows
            ),
            "excluded_group_rows": sum(r["group_id"] in excluded for _, _, r in rows),
        }

    result["n5b"] = {
        "train_sha256": file_sha256(n5b_path),
        "manifest_output_sha256": n5b_man["output_sha256"],
        "rows": len(n5b),
        "tokens": n5b_man["tokens"],
        "tokens_vs_budget": n5b_man["tokens"] / BUDGET_TOKENS - 1,
        "rows_by_type": n5b_man["rows_by_type"],
        "outside_block_identical_to_n4xf": identical,
        "block_rows": len(block_n5b),
        "block_tokens_n4xf": btot,
        "quotas": quotas,
        "builder_checks": n5b_man["checks"],
        "mlx_disjoint": disjoint(n5b),
    }
    result["n4xf_mlx_disjoint"] = disjoint(n4)
    for key, d in (
        ("n5b", result["n5b"]["mlx_disjoint"]),
        ("n4xf", result["n4xf_mlx_disjoint"]),
    ):
        if any(d.values()):
            fails.append(f"{key} MLX-DEV / excluded-group overlap")
    if abs(result["n5b"]["tokens_vs_budget"]) > 0.01:
        fails.append("N5B budget")
    if not identical:
        fails.append("N5B outside-block rows")
    if any(abs(q["rel_dev"]) > QUOTA_TOLERANCE for q in quotas.values()):
        fails.append("N5B quotas")
    result["mlxdev"] = jload(M / "mlxdev/build/panel.jsonl.manifest.json")
    result["mlxdev"]["template_max_groups"] = tmg
    result["mlxdev"]["ignored_template_segments"] = len(ignored)
    labels = M / "teacher/nox-n5b-add/labels.jsonl"
    add_ids = {json.loads(x)["id"] for x in open(labels)}
    result["labels_added"] = {
        "labels_sha256": file_sha256(labels),
        "rows": len(add_ids),
        "equals_added_block_rows": add_ids == block_n5b - n4_block,
        "bitwise": jload(M / "teacher/checks-n5b-add-bitwise/bitwise.json"),
        "m3_overlap": jload(M / "teacher/checks-n5b-add-overlap/overlap.json"),
    }
    if (
        not result["labels_added"]["equals_added_block_rows"]
        or not result["labels_added"]["bitwise"]["pass"]
    ):
        fails.append("added-row labels")
    ov = result["labels_added"]["m3_overlap"]
    if ov["overlap_rows"] and not ov["pass"]:
        fails.append("added-row M3 overlap")
    for key in ("n5b", "n5bn"):
        teacher = M / f"teacher/{key}/teacher.jsonl"
        if teacher.is_file():
            result[f"teacher_{key}"] = coverage(n5b, teacher, None)
        else:
            receipt = M / f"teacher/{key}.stderr.log"
            error = (
                receipt.read_text().strip().splitlines()[-1:]
                if receipt.is_file()
                else []
            )
            result[f"teacher_{key}"] = {
                "ok": False,
                "missing_file": str(teacher),
                "error": error,
            }
    missing = M / "teacher/missing-n5b/uncovered.jsonl.manifest.json"
    if missing.is_file():
        result["teacher_n5b"]["uncovered_rows"] = jload(missing)
    if result["teacher_n5bn"].get("teacher_sha256"):
        # N5BN: Nox on every block row (the two label files are sources 0 and 1)
        man = jload(M / "teacher/n5bn/teacher.jsonl.manifest.json")
        used_nox = man["sources"][0].get("used", 0) + man["sources"][1].get("used", 0)
        result["teacher_n5bn"]["nox_rows_used"] = used_nox
        result["teacher_n5bn"]["nox_rows_expected"] = len(block_n5b)
        if used_nox != len(block_n5b):
            result["teacher_n5bn"]["ok"] = False
    for key in ("teacher_n5b", "teacher_n5bn"):
        if not result[key]["ok"]:
            fails.append(key)
    result["diag_n5b_sha256"] = file_sha256(M / "teacher/diag-n5b-sources/diag.json")
result["fails"] = fails
result["status"] = "PASS" if not fails else "FAIL"
if phase != "n5n":
    # Per arm: the shared build / MLX-DEV checks stop everything; a teacher or added-label failure stops its arm.
    own = {
        "N5B": {"teacher_n5b"},
        "N5BN": {"teacher_n5bn", "added-row labels", "added-row M3 overlap"},
        "MLX-DEV": set(),
    }
    shared = [f for f in fails if not any(f in v for v in own.values())]
    result["arm_status"] = {
        arm: "PASS" if not shared and not (set(fails) & mine) else "FAIL"
        for arm, mine in own.items()
    }
out = M / f"lock-{phase}.json"
out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print(
    json.dumps(
        {
            "status": result["status"],
            "fails": fails,
            "arm_status": result.get("arm_status"),
            "file": str(out),
            "sha256": file_sha256(out),
        }
    )
)
sys.exit(0 if not fails else 1)
