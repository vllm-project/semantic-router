"""Shape of training / typed corpora vs public-231 hard (node A, stdlib; aggregates only).

For each A7 arm file (sampled), typed FINAL prompts and public-231 hard prompts:
state length distribution (<1k / 1k-4k / >4k chars), dict-state share, and the
share of states carrying a quoted, person-attributed conclusion (same regex as
gap_cue.py). Evaluation panels are read goldfree; no rows are copied anywhere.
"""

import glob
import json
import os
import random
import re
import sys

ARGS = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
SNAP = ARGS.get("snap")
N = ARGS.get("n", 4000)
CUE = re.compile(
    r"(note|comment|remark|draft|recommend\w*|message|e-mail|email|chat|says|said|"
    r"screener|planner|approver|supervisor|reviewer|clerk|analyst|manager|lead)"
    r"[^\"\u201c\n]{0,80}?[:,]\s*[\"\u201c]",
    re.I,
)


def shape(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    lens, dic, cue = [], 0, 0
    for r in rows:
        st = r.get("state")
        if st is None and isinstance(r.get("segments"), (list, dict)):
            st = r["segments"]
        s = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
        dic += not isinstance(st, str)
        lens.append(len(s))
        cue += bool(CUE.search(s))
    lens.sort()
    return {
        "n": n,
        "p50_chars": lens[n // 2],
        "p90_chars": lens[int(n * 0.9)],
        "share_gt4k": round(sum(x > 4000 for x in lens) / n, 4),
        "share_1k_4k": round(sum(1000 < x <= 4000 for x in lens) / n, 4),
        "share_dict_state": round(dic / n, 4),
        "share_quoted_human_conclusion": round(cue / n, 4),
    }


def sample_jsonl(path, n):
    rnd = random.Random(0)
    res, seen = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            if len(res) < n:
                res.append(line)
            else:
                j = rnd.randrange(seen)
                if j < n:
                    res[j] = line
    return [json.loads(x) for x in res], seen


out = {}
for path in sorted(
    glob.glob(os.path.join(SNAP, "v2/a7/arms/*/**/*.jsonl"), recursive=True)
):
    rel = os.path.relpath(path, SNAP)
    if "train" not in os.path.basename(path).lower() and not ARGS.get("all_files"):
        continue
    rows, total = sample_jsonl(path, N)
    s = shape(rows)
    s["rows_total"] = total
    out[rel] = s

tf = [
    json.loads(x)
    for x in open("/data/dev2/private/panels/goldfree/typed-final.prompts.jsonl")
]
out["typed-final (goldfree)"] = shape(tf)
pub = [
    json.loads(x)
    for x in open("/data/dev2/private/panels/goldfree/public231.prompts.jsonl")
]
out["public231 hard (goldfree)"] = shape([p for p in pub if p["id"].startswith("hard")])
out["public231 all (goldfree)"] = shape(pub)
json.dump(out, sys.stdout, indent=1, sort_keys=True)
