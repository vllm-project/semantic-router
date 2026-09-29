"""Data-lock checks for decoder M4 on node B (host python): token budget, excluded-group rows,
A0s / retention identity with M3 (components sliced in spec build order), teacher coverage by source.

usage: python3 m4-lockcheck.py <excluded-groups json>
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

M = Path("/data/dev2/runs/dec/m4")
M3 = Path("/data/dev2/runs/dec/m3/data/m3-v2m-ret/train.jsonl")
TOTAL = 29249047
EXCL = set(json.load(open(sys.argv[1]))["group_ids"]) if len(sys.argv) > 1 else set()
mixes = [
    m
    for m in ("m4-v2m-ret-r2", "m4-v2m-ret-r2-q20", "m4-xl-a7v1-29m", "m4-xl-full-29m")
    if (M / "data" / m / "train.jsonl.manifest.json").is_file()
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


m3rows = rows(M3)
man3 = json.load(open(str(M3) + ".manifest.json"))
out = {}
for mix in mixes:
    d = M / "data" / mix
    man = json.load(open(d / "train.jsonl.manifest.json"))
    train = rows(d / "train.jsonl")
    ids = {r["id"] for r in train}
    info = {
        "rows": man["rows"],
        "tokens": man["tokens"],
        "tokens_vs_budget": round(man["tokens"] / TOTAL - 1, 5),
        "train_sha256": man["output_sha256"],
        "spec_sha256": man["spec_sha256"],
        "rows_by_type": man["rows_by_type"],
        "components": {
            k: [
                v["rows"],
                v["tokens"],
                v.get("excluded_group", 0),
                v.get("duplicate_input", 0),
            ]
            for k, v in man["components"].items()
        },
        "excluded_group_rows_left": sum(r["group_id"] in EXCL for r in train),
        "langs_top": Counter(r["language"] for r in train).most_common(8),
    }
    tf = M / "teacher" / mix / "lux-teacher.jsonl"
    if tf.is_file():
        tm = json.load(open(str(tf) + ".manifest.json"))
        info["teacher"] = {
            "sha256": tm["output_sha256"],
            "covered": tm["covered"],
            "rows": tm["rows"],
            "missing_by_pool": tm["missing_by_pool"],
            "overlap": tm["overlap"],
            "used_by_source": [
                (Path(s["file"]).name, s.get("used", 0)) for s in tm["sources"]
            ],
            "agreement": {
                k: {t: round(v["accuracy"], 4) for t, v in a.items()}
                for k, a in tm["train_label_agreement"].items()
            },
        }
    out[mix] = info


# A0s and retention identity vs M3 (by component, using the manifests' row order: components are contiguous)
def comp_ids(train, man):
    ids, i = {}, 0
    for name in [c["name"] for c in man["spec"]["components"]]:
        n = man["components"][name]["rows"]
        ids[name] = {r["id"] for r in train[i : i + n]}
        i += n
    assert i == len(train), (i, len(train))
    return ids


c3 = comp_ids(m3rows, man3)
for mix in mixes:
    d = M / "data" / mix
    man = json.load(open(d / "train.jsonl.manifest.json"))
    c = comp_ids(rows(d / "train.jsonl"), man)
    out[mix]["A0s_identical_to_m3"] = c.get("A0s") == c3["A0s"]
    if "A7-stage4v2-ret" in c:
        out[mix]["retention_identical_to_m3"] = (
            c["A7-stage4v2-ret"] == c3["A7-stage4v2-ret"]
        )
    if mix == "m4-v2m-ret-r2":
        m3ids = {r["id"] for r in m3rows}
        now = set().union(*c.values())
        out[mix]["m3_minus_this"] = len(m3ids - now)
        out[mix]["this_minus_m3"] = len(now - m3ids)
gap = M / "teacher" / "ret-gap-rows" / "rows.jsonl.manifest.json"
if gap.is_file():
    g = json.load(open(gap))
    out["ret-gap-rows"] = {
        "uncovered": g["uncovered"],
        "uncovered_by_type": g["uncovered_by_type"],
        "check_sample": g["check_sample"],
        "sha256": g["output_sha256"],
    }
lab = M / "teacher" / "ret-gap-lux" / "labels.jsonl"
if lab.is_file():
    out["ret-gap-lux"] = {"sha256": sha(lab), "rows": sum(1 for _ in open(lab))}
print(json.dumps(out, indent=1))
