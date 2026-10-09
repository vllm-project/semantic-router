"""Check built proxy items against the public suite (exact + 13-gram) and freeze the final proxy.

Rule (pre-registered before any anchor run): a proxy case is removed when any of its requests has an
exact non-boilerplate state, question+options or option-set match with a public-suite item, or contains
at least ``MIN_COVERAGE`` of one public item's non-boilerplate 13-grams. For benchmarks whose candidates
come from a shared public corpus (ToolRet/BRIGHT documents, HoVer Wikipedia evidence, ESCI product text,
the fixed ACOS/POP909/PhishNChips question inventories) only the item-specific part is checked.

Outputs ``<final>/{s,o}/*.jsonl.gz``, ``<final>/manifest.json`` (counts, sha256, overlap stats) and
``<final>/protected-items.jsonl.gz`` (every final item, for ws-data decontamination).

    python -m d25.vega.eval.proxy.overlap --build /data/d25/vega/proxy/build --final /data/d25/vega/proxy/final/pv1
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from d25.vega.eval.proxy.common import (
    PROXY_VERSION,
    read_jsonl,
    sha256_file,
    write_jsonl,
)

MIN_COVERAGE = 0.5
SUITE = [
    "/data/d25/shared/index-suite-0.3/selected-rows.jsonl.gz",
    "/data/d25/shared/index-suite-0.3/added-rows.jsonl.gz",
    "/data/d25/shared/index-suite-0.3/gsm8k-rows.jsonl.gz",
]
STATE_ONLY = {2: "query", 36: "query", 61: "claim", 37: "search_query"}
QUESTION_ONLY = {38, 22, 56, 9}


def probe_rows(n, row):
    """Rows in ws-data's training-row shape (one per question) restricted to the item-specific content."""
    st = row["state"]
    if n in STATE_ONLY:
        yield {
            "id": row["_evaluation"]["run_id"] + "#state",
            "state": st.get(STATE_ONLY[n]) if isinstance(st, dict) else st,
            "question": {"type": "noul", "instructions": ""},
        }
        return
    if n in QUESTION_ONLY:
        state = (
            st.get("review")
            if n == 38
            else (
                st
                if n == 56
                else (
                    json.dumps(st.get("context_notes"))
                    if n == 22
                    else st.get("request") if n == 9 else st
                )
            )
        )
        yield {
            "id": row["_evaluation"]["run_id"] + "#state",
            "state": state,
            "question": {"type": "noul", "instructions": ""},
        }
        return
    for k, q in row["questions"].items():
        yield {"id": f"{row['_evaluation']['run_id']}#{k}", "state": st, "question": q}


def main(argv=None):
    from d25.vega.data import decontam as D

    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/data/d25/vega/proxy/build")
    ap.add_argument("--final", default=f"/data/d25/vega/proxy/final/{PROXY_VERSION}")
    ap.add_argument("--index", default="/data/d25/vega/proxy/decontam-index-0.3")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args(argv)
    index = Path(a.index)
    if not (index / "meta.json").exists():
        D.build_index(SUITE, index, a.workers)
    (index / "rule.json").write_text(
        json.dumps(
            {
                "min_hits": None,
                "min_coverage": MIN_COVERAGE,
                "calibrated": False,
                "note": "ws-proxy item-overlap rule",
            }
        )
    )
    build, final = Path(a.build), Path(a.final)
    manifest = {
        "version": PROXY_VERSION,
        "rule": {
            "min_coverage": MIN_COVERAGE,
            "exact": ["state", "question+options", "option set"],
            "state_only": STATE_ONLY,
            "question_only": sorted(QUESTION_ONLY),
        },
        "s": {},
        "o": {},
    }
    protected = []
    for part in ("s", "o"):
        for path in sorted((build / part).glob("*.jsonl.gz")):
            rows = list(read_jsonl(path))
            n = rows[0]["_evaluation"]["catalog_id"]
            probes, owner = [], {}
            for r in rows:
                for p in probe_rows(n, r):
                    probes.append(p)
                    owner[p["id"]] = r["_evaluation"]["group_id"]
            flags = list(D.check_rows(index, probes, a.workers))
            reasons = collections.Counter(x for f in flags for x in f["reasons"])
            bad_groups = {owner[f["id"]] for f in flags if f["drop"]}
            anyhit = sum(f["hits"] > 0 for f in flags)
            keep = [r for r in rows if r["_evaluation"]["group_id"] not in bad_groups]
            out = final / part / path.name
            write_jsonl(out, keep)
            groups = {r["_evaluation"]["group_id"] for r in rows}
            info = {
                "catalog_id": n,
                "requests_built": len(rows),
                "cases_built": len(groups),
                "cases_removed": len(bad_groups),
                "requests": len(keep),
                "cases": len({r["_evaluation"]["group_id"] for r in keep}),
                "questions": sum(len(r["questions"]) for r in keep),
                "probes": len(probes),
                "probes_with_any_13gram_hit": anyhit,
                "reasons": dict(reasons),
                "sha256": sha256_file(out, gunzip=True),
            }
            manifest[part][path.name.split(".")[0]] = info
            print(
                json.dumps(
                    {
                        "part": part,
                        "file": path.name,
                        **{k: v for k, v in info.items() if k != "sha256"},
                    }
                ),
                flush=True,
            )
            for r in keep:
                protected.append(
                    {
                        "proxy": PROXY_VERSION,
                        "part": part,
                        "catalog_id": n,
                        "id": r["_evaluation"]["run_id"],
                        "state": r["state"],
                        "questions": r["questions"],
                    }
                )
    write_jsonl(final / "protected-items.jsonl.gz", protected)
    manifest["protected_items_sha256"] = sha256_file(
        final / "protected-items.jsonl.gz", gunzip=True
    )
    (final / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {
                "event": "frozen",
                "final": str(final),
                "s_requests": sum(v["requests"] for v in manifest["s"].values()),
                "s_questions": sum(v["questions"] for v in manifest["s"].values()),
                "o_questions": sum(v["questions"] for v in manifest["o"].values()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
