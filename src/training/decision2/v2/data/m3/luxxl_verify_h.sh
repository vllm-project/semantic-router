#!/usr/bin/env bash
# Independent re-download check of own-Lux wave h-w1 at its HF revision (node A, CPU, read-only on inputs).
set -euo pipefail
export HF_HUB_CACHE=/data/dev2/hf-cache
REPO=llm-semantic-router/decision-2.0-training-data
REV=75e557f170979bdbc428b6ea698a2047e2d2a5cd
R2=100536133e192c54ec57c2599a5e4706f6d334ff
W=/data/dev2/runs/data/m3b/lux-xl
GAP=/data/dev2/runs/data/m3b/gap/c2/final
V=$W/verify-h-w1
umask 077
mkdir "$V"
D=$V/dl
hf download "$REPO" m3/teachers/lux1/xl/h-w1.targets.jsonl m3/teachers/lux1/xl/h-w1.attestation.jsonl \
  m3/teachers/lux1/xl/h-w1.report.json m3/teachers/lux1/xl/coverage-r2.json m3/teachers/lux1/xl/README.md \
  --repo-type dataset --revision "$REV" --local-dir "$D" > "$V/download.log" 2>&1
hf download "$REPO" m3/mixtures/xl-r2/mx-xl-full-r2.lux1.missing.jsonl m3/mixtures/xl-r2/mx-xl-short-r2.lux1.missing.jsonl \
  --repo-type dataset --revision "$R2" --local-dir "$D" >> "$V/download.log" 2>&1
PY=$(head -1 "$(command -v hf)" | sed 's/^#!//')
"$PY" - "$D" "$W/upload-h-w1" "$GAP" "$REPO" "$REV" <<'PYEOF'
import hashlib, json, math, sys
from huggingface_hub import HfApi
d, up, gap, repo, rev = sys.argv[1:]
x = f"{d}/m3/teachers/lux1/xl"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
out = {"revision": rev}
files = ["h-w1.targets.jsonl", "h-w1.attestation.jsonl", "h-w1.report.json", "coverage-r2.json", "README.md"]
out["sha256_equal_to_upload"] = {f: sha(f"{x}/{f}") == sha(f"{up}/{f}") for f in files}
out["sha256"] = {f: sha(f"{x}/{f}") for f in files}
ids = lambda p: [json.loads(l)["id"] for l in open(p) if l.strip()]
full = ids(f"{d}/m3/mixtures/xl-r2/mx-xl-full-r2.lux1.missing.jsonl")
short = ids(f"{d}/m3/mixtures/xl-r2/mx-xl-short-r2.lux1.missing.jsonl")
train = {}
for arm in ("h7", "h8"):
    for l in open(f"{gap}/{arm}.train.jsonl"):
        r = json.loads(l); train[r["id"]] = r["input_sha256"]
targets = [json.loads(l) for l in open(f"{x}/h-w1.targets.jsonl")]
tid = [t["id"] for t in targets]
out["targets"] = len(targets)
out["targets_sorted_unique"] = tid == sorted(set(tid))
out["targets_equal_r2_missing_union"] = set(tid) == set(full) | set(short)
out["full_r2_missing_covered"] = f"{len(set(full) & set(tid))} / {len(full)}"
out["short_r2_missing_covered"] = f"{len(set(short) & set(tid))} / {len(short)}"
out["input_sha256_matches_train_row"] = sum(train.get(t["id"]) == t["input_sha256"] for t in targets)
def probs_ok(p):
    vals = list(p.values()) if isinstance(p, dict) else list(p)
    return all(v >= 0 for v in vals) and abs(sum(vals) - 1) < 1e-4
out["teacher_probs_normalized"] = sum(probs_ok(t["teacher_probs"]) for t in targets)
att = [json.loads(l) for l in open(f"{x}/h-w1.attestation.jsonl")]
ok = lambda a: (a["target"] is True and a["model_id"] == "llm-semantic-router/Decision-1.0-Lux-9B"
    and a["model_revision"] == "bd45a30aee8c84032791c245c70f86dee5389cc8" and a["revision_attested"] is True
    and a["runtime_matches_validated"] is True and a["node"] == "node B" and a["gpu"] == 7
    and a["image_id"].startswith("sha256:ce895822") and a["backend"] == "lux")
out["attestation_lines"] = len(att)
out["attestation_ok"] = sum(ok(a) for a in att)
out["attestation_ids_equal_targets"] = [a["id"] for a in att] == tid if "id" in att[0] else "no id field"
out["attestation_input_sha256_equal_targets"] = [a["input_sha256"] for a in att] == [t["input_sha256"] for t in targets]
tree = {f.path: f for f in HfApi().list_repo_tree(repo, path_in_repo="m3/teachers/lux1/xl", repo_type="dataset", revision=rev)}
names = sorted(p.rsplit("/", 1)[1] for p in tree)
out["tree_files"] = len(names)
out["tree_waves"] = sorted({n.split(".")[0] for n in names if n.endswith(".targets.jsonl")})
info = HfApi().dataset_info(repo, revision=rev)
out["private"] = info.private
print(json.dumps(out, indent=1, sort_keys=True))
PYEOF
