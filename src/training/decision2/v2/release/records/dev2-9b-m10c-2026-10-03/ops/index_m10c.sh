#!/usr/bin/env bash
# Decision-2.0-Lux-9B Index-first successor of KIB4-a40 (9B M10 amendment 7): the card's private Index input on node A, built
# by python -m v2.release.card_index (schema dev2-card-index/1: board-served parameter counts, the audited footnote).
#   base       per other tier, the newest released Index input under /data/dev2/private/release/ whose point for that
#              tier carries exactly the weights of the tier's current Hub main, so the card plots every Decision 2.0
#              point at its current weights. Releases chain their inputs (each replaces its own tier's point); the 4B
#              and 27B releases of 2026-10-02 ran side by side, so no single input carries both, and each tier's point
#              comes from the newest input that carries it. Board points and footnote must be equal in every source.
#   runs       those family points, one run per scored weights; plus this candidate's kit run (kit index.json
#              of the IX1 run on exactly these weights: balanced skill, area skills x 100, its SHA-256)
#   board      the public board snapshot of 2026-09-28 (Space index a5a4aa0a)
#   manifests  MODEL_MANIFEST.json of every other tier's current Hub main; 9B = the candidate's restaged package (its
#              identity and loaded count are the shipped package's)
# Outputs (private, mode 700): $PRIV/{manifests/,base.json,runs.json,decision-index-card.json}; prints hashes only.
# Usage (node A): bash <mirror>/v2/release/records/dev2-9b-m10c-2026-10-03/ops/index_m10c.sh CAND
set -euo pipefail
ARM=${1:?ARM}
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
PRIV=/data/dev2/private/release/m10c/$ARM
BOARD=/data/dev2/private/eval/index021/space/index-v0.2.1-7cdcea3d.json
BOARD_SHA=a5a4aa0a2cce1520
HFPY=/data/dev2/tools/hf-cli/bin/python
export PYTHONPATH=$S HF_HUB_CACHE=/data/dev2/hf-cache
umask 077
[[ -f "$PRIV/kit-index.json" && -f "$PRIV/package-manifest.json" && -f "$PRIV/receipt.json" ]] ||
  { echo "run inputs_m10c.sh $ARM first" >&2; exit 1; }
[[ ! -e "$PRIV/decision-index-card.json" ]] || { echo "$PRIV/decision-index-card.json exists" >&2; exit 1; }
[[ "$(sha256sum < "$BOARD" | cut -c1-16)" == "$BOARD_SHA" ]] || { echo "board snapshot is not $BOARD_SHA" >&2; exit 1; }
mkdir -p "$PRIV/manifests/9B"
cp "$PRIV/package-manifest.json" "$PRIV/manifests/9B/MODEL_MANIFEST.json"
"$HFPY" - "$PRIV/manifests" <<'PY'
import json, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
out = Path(sys.argv[1])
api = HfApi()
mains = {}
for tier, codename in (("0.6B", "Kai"), ("0.8B", "Eos"), ("2B", "Sol"), ("4B", "Nox"), ("27B", "Vega")):
    repo = f"vllm-sr/Decision-2.0-{codename}-{tier}"
    sha = api.model_info(repo).sha
    path = hf_hub_download(repo, "MODEL_MANIFEST.json", revision=sha)
    (out / tier).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, out / tier / "MODEL_MANIFEST.json")
    mains[tier] = {"repo": repo, "revision": sha,
                   "identity": json.load(open(path))["identity"]["model_sha256"]}
(out / "mains.json").write_text(json.dumps(mains, indent=1, sort_keys=True) + "\n")
print(json.dumps({t: [v["revision"][:8], v["identity"][:12]] for t, v in mains.items()}))
PY
python3 - "$PRIV" <<'PY'
import glob, hashlib, json, os, sys
from pathlib import Path
from v2.release import card_index
priv = Path(sys.argv[1])
mains = json.loads((priv / "manifests/mains.json").read_text())
inputs = []
for path in glob.glob("/data/dev2/private/release/**/decision-index-card.json", recursive=True):
    if Path(path).resolve().is_relative_to(priv.resolve()):
        continue
    try:
        inputs.append((os.path.getmtime(path), path, card_index.load(Path(path))))
    except Exception:
        continue
sources, points = {}, {}
for tier, m in mains.items():
    have = [(t, path, b) for t, path, b in inputs
            for p in b["family"] if p["tier"] == tier and p["model_sha256"] == m["identity"]]
    if not have:
        raise SystemExit(f"no released Index input carries the current {tier} weights")
    _, path, b = max(have, key=lambda x: x[0])
    sources[tier] = {"path": path, "sha256": b["sha256"]}
    points[tier] = next(p for p in b["family"] if p["tier"] == tier)
bases = {s["path"]: card_index.load(Path(s["path"])) for s in sources.values()}
first = next(iter(bases.values()))
for b in bases.values():
    assert (b["decision1"], b["entrants"], b["footnote"], b["edition"]) == (
        first["decision1"], first["entrants"], first["footnote"], first["edition"]), "sources differ in board points"
(priv / "base.json").write_text(json.dumps({"per_tier": sources}, sort_keys=True) + "\n")
base = first
receipt = json.loads((priv / "receipt.json").read_text())
manifest = json.loads((priv / "package-manifest.json").read_text())
identity = manifest["identity"]["model_sha256"]
assert receipt["model_source"]["model_sha256"] == identity, "the Index run scored other weights"
kit = json.loads((priv / "kit-index.json").read_text())
assert kit["edition"] == base["edition"], kit["edition"]
runs = {
    f"released-{p['tier']}": {
        "model_sha256": p["model_sha256"], "edition": base["edition"], "balanced_skill": p["balanced_skill"],
        "areas": p["areas"], "index_sha256": p["kit_index_sha256"],
    }
    for p in points.values()
}
runs["candidate-9B"] = {
    "model_sha256": identity, "edition": kit["edition"], "balanced_skill": kit["scores"]["balanced_skill"],
    "areas": {a["id"]: 100 * a["skill"] for a in kit["areas"]},
    "index_sha256": hashlib.sha256((priv / "kit-index.json").read_bytes()).hexdigest(),
}
(priv / "runs.json").write_text(json.dumps(runs, sort_keys=True) + "\n")  # card_index reads one JSON line
print(json.dumps({"sources": {t: s["sha256"][:12] for t, s in sorted(sources.items())}, "runs": len(runs)}))
PY
python3 -m v2.release.card_index --runs "$PRIV/runs.json" --board "$BOARD" --snapshot 2026-09-28 \
  --manifests "$PRIV/manifests" --out "$PRIV/decision-index-card.json"
python3 - "$PRIV" <<'PY'
import json, sys
from pathlib import Path
from v2.release import card_index
priv = Path(sys.argv[1])
srcs = json.loads((priv / "base.json").read_text())["per_tier"]
new = card_index.load(priv / "decision-index-card.json")
same = {}
for tier, s in srcs.items():
    same[tier] = next(p for p in card_index.load(Path(s["path"]))["family"] if p["tier"] == tier)
old = card_index.load(Path(next(iter(srcs.values()))["path"]))
changed = [p["tier"] for p in new["family"] if p != same.get(p["tier"])]
assert not set(changed) - {"9B"}, f"family points other than 9B differ from their sources: {changed}"
assert new["decision1"] == old["decision1"] and new["entrants"] == old["entrants"], "board points changed"
assert new["footnote"] == old["footnote"]
print(json.dumps({"changed_family_points": changed, "sha256": new["sha256"]}))
PY
chmod 600 "$PRIV"/*.json "$PRIV/manifests"/*/MODEL_MANIFEST.json
