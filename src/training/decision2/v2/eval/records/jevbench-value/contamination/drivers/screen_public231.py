"""Fresh public-231-only overlap screen of released-model training files (CPU).

Usage (stdin driver): ssh NODE "cd /tmp && PYTHONPATH=$S python3 - '<json>'" < screen_public231.py
  json: {"out": DIR, "workers": N, "files": [{"label", "path", "sha256"}]}

1. Builds a one-role protected inventory from the gold-free public-231 prompts
   (id / state / questions; answer-free, hash-pinned) and runs
   `python3 -m v2.data.overlap --protected-inventory` (rules E / S / L / N) per file.
2. Supplemental, unsampled: word 8-gram and 13-gram containment of every public
   item (state + question texts) in each file's state / instructions / options
   leaves, plus exact normalized-leaf equality. Grams present in >= 5 public items
   are template text and are reported separately.
Per-item and per-row details stay in DIR (0700); stdout carries aggregates only.
"""

import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from v2.data.textnorm import compact, normalize, text_leaves, word_tokens

PUBLIC231 = "/data/dev2/private/panels/goldfree/public231.prompts.jsonl"
PUBLIC231_SHA = "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd"
ROLE = "jevbench_public231"
TEMPLATE_ITEMS = 5


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question_texts(questions):
    return text_leaves(questions, decode_json=True)


def ngrams(tokens, n):
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def tier_of(item_id):
    for tier in ("easy", "standard", "hard"):
        if item_id.startswith(tier):
            return tier
    return "unknown"


def main():
    args = json.loads(sys.argv[1])
    out = Path(args["out"])
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    if sha_file(PUBLIC231) != PUBLIC231_SHA:
        raise SystemExit("public231 prompts hash mismatch")
    items = [
        json.loads(line) for line in open(PUBLIC231, encoding="utf-8") if line.strip()
    ]
    inv_dir = out / "inventory"
    inv_dir.mkdir(exist_ok=True)
    inv_file = inv_dir / f"{ROLE}.jsonl"
    if not inv_file.exists():
        with open(inv_file, "w", encoding="utf-8") as stream:
            for row in items:
                stream.write(
                    json.dumps(
                        {k: row[k] for k in ("id", "state", "questions")},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.chmod(inv_file, 0o600)
    manifest = inv_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            [{"role": ROLE, "path": str(inv_file), "sha256": sha_file(inv_file)}]
        )
    )

    item_grams = {}
    item_leaves = {}
    for row in items:
        leaves = text_leaves(row["state"], decode_json=True) + question_texts(
            row["questions"]
        )
        grams = {8: set(), 13: set()}
        for leaf in leaves:
            toks = word_tokens(leaf)
            for n in grams:
                grams[n] |= ngrams(toks, n)
        item_grams[row["id"]] = grams
        item_leaves[row["id"]] = {normalize(l) for l in leaves if len(compact(l)) >= 20}
    gram_items = {n: collections.defaultdict(set) for n in (8, 13)}
    for iid, grams in item_grams.items():
        for n, gs in grams.items():
            for g in gs:
                gram_items[n][g].add(iid)
    template = {
        n: {g for g, s in gram_items[n].items() if len(s) >= TEMPLATE_ITEMS}
        for n in (8, 13)
    }
    leaf_items = collections.defaultdict(set)
    for iid, ls in item_leaves.items():
        for l in ls:
            leaf_items[l].add(iid)

    report = {
        "public231_sha256": PUBLIC231_SHA,
        "items": len(items),
        "template_grams": {str(n): len(template[n]) for n in (8, 13)},
        "files": [],
    }
    for spec in args["files"]:
        label, path = spec["label"], spec["path"]
        entry = {"label": label, "sha256_expected": spec["sha256"]}
        actual = sha_file(path)
        entry["sha256_ok"] = actual.startswith(spec["sha256"])
        entry["sha256"] = actual
        if not entry["sha256_ok"]:
            report["files"].append(entry)
            continue
        priv = out / f"{label}.overlap.private.json"
        pub = out / f"{label}.overlap.public.json"
        for p in (priv, pub):
            if p.exists():
                p.unlink()
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "v2.data.overlap",
                "--candidates",
                path,
                "--protected-inventory",
                str(manifest),
                "--private-receipt",
                str(priv),
                "--public-receipt",
                str(pub),
                "--workers",
                str(args.get("workers", 16)),
            ],
            capture_output=True,
            text=True,
        )
        entry["overlap_rc"] = proc.returncode
        if proc.returncode:
            entry["overlap_stderr_tail"] = proc.stderr[-600:]
        else:
            p = json.loads(pub.read_text())
            entry["overlap"] = {
                "candidate_rows": p["candidates"]["rows"],
                "candidate_groups": p["candidates"]["groups"],
                "by_method": p["flagged"]["by_method"],
                "quarantine": p["flagged"]["quarantine"],
                "boilerplate": p["flagged"]["boilerplate"],
            }
            q = json.loads(priv.read_text())
            hit_items = set()
            for rec in q["groups"].values():
                for method, roles in rec.get("by_method", {}).items():
                    for ids in roles.values():
                        hit_items.update(ids if isinstance(ids, list) else [])
            entry["overlap"]["distinct_public_items_hit"] = len(hit_items)

        hits = {n: collections.defaultdict(set) for n in (8, 13)}
        leaf_hits = collections.defaultdict(set)
        row_hits = []
        rows = 0
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                leaves = []
                for field in ("state", "instructions", "options"):
                    if field in row:
                        leaves += text_leaves(row[field], decode_json=True)
                row_items = collections.Counter()
                for leaf in leaves:
                    nl = normalize(leaf)
                    if nl in leaf_items:
                        for iid in leaf_items[nl]:
                            leaf_hits[iid].add(row.get("id"))
                    toks = word_tokens(leaf)
                    for n in (8, 13):
                        for g in ngrams(toks, n):
                            if g in gram_items[n] and g not in template[n]:
                                for iid in gram_items[n][g]:
                                    hits[n][iid].add(g)
                                    if n == 13:
                                        row_items[iid] += 1
                if row_items:
                    iid, c = row_items.most_common(1)[0]
                    row_hits.append(
                        {
                            "row": row.get("id"),
                            "group": row.get("group_id"),
                            "source": row.get("source"),
                            "item": iid,
                            "grams13": c,
                        }
                    )
        cont = {}
        for iid, grams in item_grams.items():
            denom = len(grams[8] - template[8])
            cont[iid] = len(hits[8].get(iid, ())) / denom if denom else 0.0
        by_tier = collections.defaultdict(lambda: collections.Counter())
        for iid in item_grams:
            t = tier_of(iid)
            by_tier[t]["items"] += 1
            if hits[13].get(iid):
                by_tier[t]["any13"] += 1
            if cont[iid] >= 0.5:
                by_tier[t]["cont8_ge_0.5"] += 1
            if leaf_hits.get(iid):
                by_tier[t]["exact_leaf"] += 1
        vals = sorted(cont.values(), reverse=True)
        entry["ngram"] = {
            "rows": rows,
            "items_exact_leaf": sum(1 for i in item_grams if leaf_hits.get(i)),
            "items_any_13gram": sum(1 for i in item_grams if hits[13].get(i)),
            "items_any_8gram": sum(1 for i in item_grams if hits[8].get(i)),
            "items_cont8_ge_0.25": sum(v >= 0.25 for v in vals),
            "items_cont8_ge_0.5": sum(v >= 0.5 for v in vals),
            "max_cont8": round(vals[0], 4) if vals else 0.0,
            "top5_cont8": [round(v, 4) for v in vals[:5]],
            "rows_with_13gram_hit": len(row_hits),
            "rows_with_ge3_13grams": sum(r["grams13"] >= 3 for r in row_hits),
            "by_tier": {t: dict(c) for t, c in by_tier.items()},
            "sources_of_13gram_rows": dict(
                collections.Counter(r["source"] for r in row_hits).most_common(10)
            ),
        }
        detail = out / f"{label}.ngram.private.json"
        detail.write_text(
            json.dumps(
                {
                    "containment8": cont,
                    "exact_leaf": {
                        i: sorted(map(str, v)) for i, v in leaf_hits.items()
                    },
                    "rows_13gram": row_hits,
                }
            )
        )
        os.chmod(detail, 0o600)
        report["files"].append(entry)
    (out / "aggregate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


main()
