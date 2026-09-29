"""Reliability / validity statistics for JevBench public 231 (CPU only; numpy + scipy).

Reads OUT/matrix.json (written by 01_collect.py on node A) and the canonical runs' REPORT.json
and mlx-diag score files. Writes per-item statistics to OUT/item_stats.json (node A only) and
prints aggregate-only JSON (no item ids, predictions or gold) on stdout.
Usage (node A):
  docker run --rm -i --network none -v /data/dev2:/data/dev2 decision20-train-fast:host2 \
    python3 - '{"out": "/data/dev2/private/eval/jevbench-value/stats"}' < 02_analyze.py
"""

import itertools, json, math, os, sys
import numpy as np
from scipy import stats

args = json.loads(sys.argv[1])
OUT = args["out"]
SEED = 20260927
RUNS = "/data/dev2/runs"

# key, label, canonical run, size tier, lineage (own1 | rel2 | cand2 | peer), card model, C1, mlx override
M = [
    (
        "kai1",
        "Decision 1.0 Kai",
        "eval/m1-adopt/kai1",
        "0.6B",
        "own1",
        True,
        17.77,
        None,
    ),
    ("lex1", "Decision 1.0 Lex", "eval/m1/r3-lex", "0.6B", "own1", True, 20.82, None),
    ("eos1", "Decision 1.0 Eos", "eval/m1/r4-eos1", "0.8B", "own1", True, 37.94, None),
    ("sol1", "Decision 1.0 Sol", "eval/m1-adopt/sol1", "2B", "own1", True, None, None),
    ("nox1", "Decision 1.0 Nox", "eval/m1-adopt/nox1", "4B", "own1", True, None, None),
    (
        "lux1",
        "Decision 1.0 Lux",
        "eval/m1/d1-lux1-autotune-cache",
        "9B",
        "own1",
        True,
        None,
        "eval/m2/mlx/x-lux1",
    ),
    (
        "dev2-0.6b",
        "DEV2.0-0.6B (t-a7)",
        "06b/m4/formal/m4-t-a7-soup",
        "0.6B",
        "rel2",
        True,
        33.21,
        None,
    ),
    (
        "dev2-0.8b",
        "DEV2.0-0.8B (E8F)",
        "dec/formal/m2/m2-E8F-soup-nodeA",
        "0.8B",
        "rel2",
        True,
        40.24,
        None,
    ),
    (
        "dev2-2b",
        "DEV2.0-2B (S2T)",
        "dec/formal/m3/m3-S2T-soup-nodeA",
        "2B",
        "rel2",
        True,
        None,
        None,
    ),
    (
        "dev2-4b",
        "DEV2.0-4B (N4XF)",
        "dec/formal/m4/m4-N4XF-soup-nodeA",
        "4B",
        "rel2",
        True,
        None,
        None,
    ),
    (
        "dev2-8b",
        "DEV2.0-8B (K-a13)",
        "9b/formal-m4/K-a13-16k",
        "9B",
        "rel2",
        True,
        None,
        None,
    ),
    (
        "dev2-27b",
        "DEV2.0-27B F1 (M3-A)",
        "nodeB/M3-A-soup_formal",
        "27B",
        "rel2",
        True,
        None,
        None,
    ),
    (
        "c06-m4-v2",
        "0.6B cand m4-v2",
        "06b/m4/formal/m4-v2-soup",
        "0.6B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c06-m5-x",
        "0.6B cand m5-x",
        "06b/m5/formal/m5-x-soup",
        "0.6B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c06-m5-z",
        "0.6B cand m5-z",
        "06b/m5/formal/m5-z-soup",
        "0.6B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c06-m6-mxcx",
        "0.6B cand m6-mxcx",
        "06b/m6/formal/m6-mxcx-soup",
        "0.6B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c06-m6-mxcxa",
        "0.6B cand m6-mxcxa",
        "06b/m6/formal/m6-mxcxa-soup",
        "0.6B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c06-m7-mx",
        "0.6B cand m7-mx",
        "06b/m7/formal/m7-mx-soup",
        "0.6B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c06-m7-mxcx",
        "0.6B cand m7-mxcx",
        "06b/m7/formal/m7-mxcx-soup",
        "0.6B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c06-old",
        "old DEV2.0-0.6B (deleted ckpt)",
        "eval/m1-adopt/dev20-06b",
        "0.6B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c08-B8F-s1",
        "0.8B cand B8F-s1",
        "dec/formal/m2/m2-B8F-s1-nodeA",
        "0.8B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c08-B8F-s2",
        "0.8B cand B8F-s2",
        "dec/formal/m2/m2-B8F-s2-nodeA",
        "0.8B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c08-E8F-s1",
        "0.8B cand E8F-s1",
        "dec/formal/m2/m2-E8F-s1-nodeA",
        "0.8B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c08-E8F-s2",
        "0.8B cand E8F-s2",
        "dec/formal/m2/m2-E8F-s2-nodeA",
        "0.8B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c08-E8F-s3",
        "0.8B cand E8F-s3",
        "dec/formal/m2/m2-E8F-s3-nodeA",
        "0.8B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c08-E8V",
        "0.8B cand E8V soup",
        "dec/formal/m3/m3-E8V-soup-nodeA",
        "0.8B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-X2",
        "4B cand E1-X2",
        "dec/formal/e1-x2-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-X4K-s1",
        "4B cand X4K-s1",
        "dec/formal/m2/m2-X4K-s1-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-X4K-s2",
        "4B cand X4K-s2",
        "dec/formal/m2/m2-X4K-s2-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-X4R-s1",
        "4B cand X4R-s1",
        "dec/formal/m2/m2-X4R-s1-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-X4R-s2",
        "4B cand X4R-s2",
        "dec/formal/m2/m2-X4R-s2-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-N4J",
        "4B cand N4J",
        "dec/formal/m3/m3-N4J-soup-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-N4L",
        "4B cand N4L",
        "dec/formal/m3/m3-N4L-soup-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-N4LKr",
        "4B cand N4LKr",
        "dec/formal/m3/m3-N4LKr-soup-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-N4T",
        "4B cand N4T",
        "dec/formal/m3/m3-N4T-soup-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-N4LX",
        "4B cand N4LX",
        "dec/formal/m4/m4-N4LX-soup-nodeA",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-N5B",
        "4B cand N5B",
        "dec/formal/m5/m5-N5B-soup",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-N5BN",
        "4B cand N5BN",
        "dec/formal/m5/m5-N5BN-soup",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c4-N5N",
        "4B cand N5N",
        "dec/formal/m5/m5-N5N-soup",
        "4B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c9-B-s1",
        "9B cand B-s1",
        "9b/formal-m3/B-s1-16k",
        "9B",
        "cand2",
        True,
        None,
        None,
    ),
    ("c9-DW", "9B cand DW", "9b/formal-m3/DW-16k", "9B", "cand2", True, None, None),
    (
        "c9-KN-a12",
        "9B cand KN-a12",
        "9b/formal-m4/KN-a12-16k",
        "9B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c9-U-a13",
        "9B cand U-a13",
        "9b/formal-m4/U-a13-16k",
        "9B",
        "cand2",
        True,
        None,
        None,
    ),
    ("c9-L2", "9B cand L2", "9b/formal/l2-8k", "9B", "cand2", True, None, None),
    (
        "c27-C0",
        "27B cand C0",
        "nodeB/formal-C0_same-panel",
        "27B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c27-C1",
        "27B cand M2-C1",
        "nodeB/m2-C1-formal",
        "27B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c27-K1",
        "27B cand M2-K1",
        "nodeB/m2-K1-formal",
        "27B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c27-S1",
        "27B cand M2-S1",
        "nodeB/m2-S1-formal",
        "27B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "c27-F2",
        "27B F2 (M3-S)",
        "nodeB/M3-S-soup_formal",
        "27B",
        "cand2",
        True,
        None,
        None,
    ),
    (
        "gliner25",
        "GLiNER2.5-Decide",
        "eval/m1/p1-gliner25",
        "0.6B",
        "peer",
        True,
        22.82,
        None,
    ),
    (
        "bosun06",
        "Bosun v3.1 0.6B",
        "eval/m1-adopt/bosun",
        "0.6B",
        "peer",
        True,
        35.03,
        None,
    ),
    (
        "bosun17b",
        "Bosun v3.1 1.7B",
        "eval/m1/p3-bosun17b",
        "2B",
        "peer",
        True,
        None,
        None,
    ),
    ("jpt08b", "JPT-0.8B", "eval/m1/p2-jpt08b", "0.8B", "peer", False, 39.01, None),
    ("jpt4b", "JPT-4B", "eval/m1/p4-jpt4b", "4B", "peer", False, None, None),
    ("jpt9b", "JPT-9B", "eval/m1-adopt/jpt9b", "9B", "peer", False, None, None),
    (
        "decider2b",
        "Decider 2B",
        "eval/m1-adopt/decider2b",
        "2B",
        "peer",
        True,
        None,
        None,
    ),
    (
        "decider4b",
        "Decider 4B",
        "eval/m1-adopt/decider4b",
        "4B",
        "peer",
        True,
        None,
        None,
    ),
    (
        "intern08b",
        "Intern-Decision 0.8B",
        "eval/m2/q2b-intern08b",
        "0.8B",
        "peer",
        True,
        34.58,
        None,
    ),
    ("kev08b", "Kev 0.8B", "eval/m2/q1-kev08b", "0.8B", "peer", True, 39.06, None),
    (
        "thisthat12",
        "This-That 1.2",
        "eval/m2/q4-thisthat12",
        "2B",
        "peer",
        True,
        None,
        None,
    ),
    ("jet62", "Jet v6.2", "eval/m2/q5b-jet62", "4B", "peer", True, None, None),
    ("nimble2", "Nimble 9B v2", "eval/m2/q6-nimble2", "9B", "peer", True, None, None),
    (
        "hopperg",
        "Hopper (G) 1.2",
        "eval/m2/q8-hopperg",
        "4B",
        "peer",
        False,
        None,
        None,
    ),
    (
        "autojev27",
        "AutoJev-27B",
        "eval/m4/nodeB-kernel/autojev27",
        "27B",
        "peer",
        True,
        None,
        "eval/m5/mlx-diag-27b/autojev27",
    ),
    (
        "eikos27b",
        "Eikos-27B",
        "eval/m4/nodeB-kernel/eikos27b",
        "27B",
        "peer",
        True,
        None,
        None,
    ),
    (
        "jebadiah27b",
        "Jebadiah-27B",
        "eval/m4/nodeB-kernel/jebadiah27b",
        "27B",
        "peer",
        True,
        None,
        None,
    ),
]

RETEST = [  # (group, kind, run a, run b)
    (
        "Lux1",
        "same weights",
        "eval/m1/d1-lux1-autotune-cache",
        "eval/m1/r1-lux1-repeat",
    ),
    (
        "Lux1",
        "same weights",
        "eval/m1/d1-lux1-autotune-cache",
        "eval/m1/d2-lux1-frozen-cache",
    ),
    (
        "Lux1",
        "same weights (node B r4)",
        "eval/m1/d1-lux1-autotune-cache",
        "eval/m1-adopt/lux1",
    ),
    (
        "Lux1",
        "same weights (node-B kernel)",
        "eval/m1/d1-lux1-autotune-cache",
        "eval/m4/nodeB-kernel/lux1",
    ),
    (
        "Lux1",
        "renderer (adopted vs shared 16K)",
        "eval/m1/d1-lux1-autotune-cache",
        "9b/formal-m3/lux1-16k-shared",
    ),
    (
        "AutoJev",
        "node A vs node B",
        "eval/m1-adopt/autojev27",
        "eval/m2/n2-autojev27-nodeB",
    ),
    (
        "AutoJev",
        "node B vs node-B copy",
        "eval/m2/n2-autojev27-nodeB",
        "nodeB/m2-peer-autojev27-nodeB",
    ),
    (
        "AutoJev",
        "node B vs 27B-track peer run",
        "eval/m2/n2-autojev27-nodeB",
        "nodeB/formal-peer-autojev27",
    ),
    (
        "AutoJev",
        "node A vs node-B kernel",
        "eval/m1-adopt/autojev27",
        "eval/m4/nodeB-kernel/autojev27",
    ),
    (
        "AutoJev",
        "kernel vs 27B-track kernel",
        "eval/m4/nodeB-kernel/autojev27",
        "nodeB/m2-peer-autojev27-nodeB-kernel",
    ),
    (
        "Eikos",
        "old node B vs kernel",
        "eval/m2/q3-eikos27b-nodeB",
        "eval/m4/nodeB-kernel/eikos27b",
    ),
    (
        "Eikos",
        "kernel vs 27B-track kernel",
        "eval/m4/nodeB-kernel/eikos27b",
        "nodeB/m3-peer-eikos27-nodeB-kernel",
    ),
    (
        "Jebadiah",
        "old node B vs kernel",
        "eval/m2/q7-jebadiah27b-nodeB",
        "eval/m4/nodeB-kernel/jebadiah27b",
    ),
    (
        "Jebadiah",
        "kernel vs 27B-track kernel",
        "eval/m4/nodeB-kernel/jebadiah27b",
        "nodeB/m3-peer-jebadiah-nodeB-kernel",
    ),
    ("Kai1", "repeat", "eval/m1-adopt/kai1", "eval/m1/r2-kai1-repeat"),
    (
        "DEV2.0-0.6B",
        "release rescore (CAL698)",
        "06b/m4/formal/m4-t-a7-soup",
        "release/dev2-0p6b-cal698-rescore",
    ),
    (
        "DEV2.0-0.6B",
        "M6 control re-run",
        "06b/m4/formal/m4-t-a7-soup",
        "06b/m6/formal/m6-control-released",
    ),
    (
        "DEV2.0-0.8B",
        "release t1-derived",
        "dec/formal/m2/m2-E8F-soup-nodeA",
        "release/dev2-0p8b-t1-derived",
    ),
    (
        "DEV2.0-2B",
        "release t1-derived",
        "dec/formal/m3/m3-S2T-soup-nodeA",
        "release/dev2-2b-t1-derived",
    ),
    (
        "DEV2.0-4B",
        "release t1-derived",
        "dec/formal/m4/m4-N4XF-soup-nodeA",
        "release/dev2-4b-t1-derived",
    ),
    (
        "DEV2.0-4B",
        "node-B reference re-run",
        "dec/formal/m4/m4-N4XF-soup-nodeA",
        "dec/formal/m5/m5-ref-N4XF-soup",
    ),
    (
        "DEV2.0-8B",
        "release t1-derived",
        "9b/formal-m4/K-a13-16k",
        "release/dev2-8b-t1-derived",
    ),
    (
        "27B C0",
        "same-panel vs 8K formal",
        "nodeB/formal-C0_same-panel",
        "nodeB/m2-F0-formal",
    ),
    ("Nox1", "context 8K vs 16K", "dec/formal/m2/nox1-8k", "dec/formal/m3/nox1-16k"),
    ("Nox1", "adopted vs 16K", "eval/m1-adopt/nox1", "dec/formal/m3/nox1-16k"),
    ("Sol1", "context 8K vs 16K", "dec/formal/m2/sol1-8k", "dec/formal/m3/sol1-16k"),
    ("Sol1", "adopted vs 16K", "eval/m1-adopt/sol1", "dec/formal/m3/sol1-16k"),
    ("Eos1", "context 8K vs 16K", "dec/formal/m2/eos1-8k", "dec/formal/m2/eos1-16k"),
    ("Eos1", "adopted vs 16K", "eval/m1/r4-eos1", "dec/formal/m2/eos1-16k"),
    ("Lux1", "context 8K vs 16K", "9b/formal/lux1-8k", "9b/formal-m3/lux1-16k-shared"),
    ("Kai1", "context 1K vs 8K", "eval/m1-adopt/kai1", "06b/m2/formal/kai1-native-8k"),
]

FOCUS = [  # (label, run a, run b)
    (
        "DEV2.0-8B vs Lux1 same-renderer 16K",
        "9b/formal-m4/K-a13-16k",
        "9b/formal-m3/lux1-16k-shared",
    ),
    (
        "DEV2.0-8B vs Lux1 adopted (d1)",
        "9b/formal-m4/K-a13-16k",
        "eval/m1/d1-lux1-autotune-cache",
    ),
    (
        "DEV2.0-4B vs Nox1 adopted",
        "dec/formal/m4/m4-N4XF-soup-nodeA",
        "eval/m1-adopt/nox1",
    ),
    (
        "DEV2.0-4B vs Nox1 16K",
        "dec/formal/m4/m4-N4XF-soup-nodeA",
        "dec/formal/m3/nox1-16k",
    ),
    (
        "DEV2.0-4B vs Decider 4B",
        "dec/formal/m4/m4-N4XF-soup-nodeA",
        "eval/m1-adopt/decider4b",
    ),
    ("DEV2.0-4B vs JPT-4B", "dec/formal/m4/m4-N4XF-soup-nodeA", "eval/m1/p4-jpt4b"),
    ("DEV2.0-4B vs Jet v6.2", "dec/formal/m4/m4-N4XF-soup-nodeA", "eval/m2/q5b-jet62"),
    (
        "DEV2.0-4B vs Hopper (G)",
        "dec/formal/m4/m4-N4XF-soup-nodeA",
        "eval/m2/q8-hopperg",
    ),
    (
        "DEV2.0-27B F1 vs AutoJev-27B (kernel)",
        "nodeB/M3-A-soup_formal",
        "eval/m4/nodeB-kernel/autojev27",
    ),
    (
        "DEV2.0-27B F1 vs Eikos-27B (kernel)",
        "nodeB/M3-A-soup_formal",
        "eval/m4/nodeB-kernel/eikos27b",
    ),
    (
        "DEV2.0-27B F1 vs Jebadiah-27B (kernel)",
        "nodeB/M3-A-soup_formal",
        "eval/m4/nodeB-kernel/jebadiah27b",
    ),
]

mat = json.load(open(os.path.join(OUT, "matrix.json")))
ids = mat["item_ids"]
gold = mat["gold"]
tier_of = np.array([gold[i]["tier"] for i in ids])
type_of = np.array([gold[i]["task_type"] for i in ids])
TIERS = ["easy", "standard", "hard"]
tier_idx = {t: np.where(tier_of == t)[0] for t in TIERS}


def vec(run):
    c = mat["runs"][run]["correct"]
    return np.array([c[i] for i in ids], dtype=np.int8)


def run_dir(run):
    return (
        os.path.join(OUT, run) if run.startswith("nodeB/") else os.path.join(RUNS, run)
    )


def load_report(run):
    r = json.load(open(os.path.join(run_dir(run), "REPORT.json")))
    bt = r["panels"]["typed-final"]["by_type"]
    return {
        "v3": r["v3"]["score"],
        "T": r["v3"]["T"],
        "H": r["v3"]["H"],
        "choice": bt["choice"]["accuracy"],
        "noul": bt["noul"]["accuracy"],
        "score": bt["score"]["accuracy"],
        "params_loaded": (r.get("parameters") or {}).get("loaded"),
        "report_tier": (r.get("model") or {}).get("tier"),
    }


def load_mlx(run, override):
    cands = [override] if override else []
    cands += [run, run + "-mlx", run.replace("-nodeA", "") + "-mlx"]
    alias = {
        "eval/m1-adopt/kai1": "eval/m2/mlx/x-kai",
        "eval/m1/r3-lex": "eval/m2/mlx/x-lex",
        "eval/m1/r4-eos1": "eval/m2/mlx/x-eos1",
        "eval/m1-adopt/sol1": "eval/m2/mlx/x-sol1",
        "eval/m1-adopt/nox1": "eval/m2/mlx/x-nox1",
        "eval/m1/p1-gliner25": "eval/m2/mlx/x-gliner25",
        "eval/m1-adopt/bosun": "eval/m2/mlx/x-bosun06",
        "eval/m1/p3-bosun17b": "eval/m2/mlx/x-bosun17b",
        "eval/m1/p2-jpt08b": "eval/m2/mlx/x-jpt08b",
        "eval/m1/p4-jpt4b": "eval/m2/mlx/x-jpt4b",
        "eval/m1-adopt/jpt9b": "eval/m2/mlx/x-jpt9b",
        "eval/m1-adopt/decider2b": "eval/m2/mlx/x-decider2b",
        "eval/m1-adopt/decider4b": "eval/m2/mlx/x-decider4b",
        "eval/m4/nodeB-kernel/eikos27b": "eval/m5/mlx-diag-27b/eikos27b",
        "eval/m4/nodeB-kernel/jebadiah27b": "eval/m5/mlx-diag-27b/jebadiah27b",
        "nodeB/M3-A-soup_formal": "27b/m3-f2/mlx-diag/M3-A-soup",
        "nodeB/M3-S-soup_formal": "27b/m3-f2/mlx-diag/M3-S-soup",
    }
    if run in alias:
        cands.insert(0, alias[run])
    for c in cands:
        p = os.path.join(RUNS, c, "mlx-diag.score.json")
        if os.path.isfile(p):
            return json.load(open(p))["type_macro_accuracy"], c
    return None, None


PARAMS_FALLBACK = {  # canonical REPORT has no loaded count: same weights' earlier run, or the base architecture
    "autojev27": ("run", "eval/m1-adopt/autojev27"),
    "eikos27b": ("run", "eval/m2/q3-eikos27b-nodeB"),
    "jebadiah27b": ("run", "eval/m2/q7-jebadiah27b-nodeB"),
    "c4-X2": ("run", "eval/m1-adopt/nox1"),
}
models = []
for key, label, run, size, lin, card, c1, mlxo in M:
    rep = load_report(run)
    if not rep["params_loaded"] and key in PARAMS_FALLBACK:
        rep["params_loaded"] = load_report(PARAMS_FALLBACK[key][1])["params_loaded"]
        rep["params_source"] = "fallback " + PARAMS_FALLBACK[key][1]
    mlx, mlxsrc = load_mlx(run, mlxo)
    v = vec(run)
    models.append(
        dict(
            key=key,
            label=label,
            run=run,
            size=size,
            lineage=lin,
            card=card,
            c1=c1,
            mlx=mlx,
            mlx_source=mlxsrc,
            vec=v,
            total=int(v.sum()),
            tiers={t: int(v[tier_idx[t]].sum()) for t in TIERS},
            **rep
        )
    )
X = np.stack([m["vec"] for m in models]).astype(float)  # models x items
K = [m["key"] for m in models]
kidx = {k: i for i, k in enumerate(K)}
nmod, nitem = X.shape
dup = [
    (a["key"], b["key"])
    for a, b in itertools.combinations(models, 2)
    if np.array_equal(a["vec"], b["vec"])
]


def f(x, d=4):
    return (
        None
        if x is None or (isinstance(x, float) and not math.isfinite(x))
        else round(float(x), d)
    )


# ---------- 2a stratified bootstrap of totals ----------
rng = np.random.default_rng(SEED)
B = 5000
draw = np.concatenate(
    [rng.choice(tier_idx[t], size=(B, len(tier_idx[t])), replace=True) for t in TIERS],
    axis=1,
)  # B x 231
boot_tot = np.stack([X[:, draw[b]].sum(axis=1) for b in range(B)], axis=1)  # models x B
lo, hi = np.percentile(boot_tot, [2.5, 97.5], axis=1)
halfw = (hi - lo) / 2
for i, m in enumerate(models):
    m["ci95"] = [float(lo[i]), float(hi[i])]
    m["ci_halfwidth"] = float(halfw[i])
    m["boot_se"] = float(boot_tot[i].std(ddof=1))


# ---------- 2b item stats ----------
def item_stats(Xs):
    p = Xs.mean(axis=0)
    tot = Xs.sum(axis=1)
    rpb = np.full(Xs.shape[1], np.nan)
    for j in range(Xs.shape[1]):
        if 0 < p[j] < 1:
            rest = tot - Xs[:, j]
            if rest.std() > 0:
                rpb[j] = np.corrcoef(Xs[:, j], rest)[0, 1]
    out = {}
    for t in TIERS + ["all"]:
        ix = np.arange(Xs.shape[1]) if t == "all" else tier_idx[t]
        pt, rt = p[ix], rpb[ix]
        ok = rt[~np.isnan(rt)]
        out[t] = {
            "items": int(len(ix)),
            "mean_p": f(pt.mean()),
            "all_right": int((pt == 1).sum()),
            "all_wrong": int((pt == 0).sum()),
            "informative_0_1": int(((pt > 0) & (pt < 1)).sum()),
            "p_between_.1_.9": int(((pt >= 0.1) & (pt <= 0.9)).sum()),
            "rpb_median": f(np.median(ok)) if len(ok) else None,
            "rpb_negative": int((ok < 0).sum()),
            "rpb_ge_.2": int((ok >= 0.2).sum()),
        }
    return out, p, rpb


sets = {
    "all_distinct": [m["key"] for m in models],
    "families_ii": [
        m["key"] for m in models if m["lineage"] in ("own1", "rel2", "peer")
    ],
}
sets["families_iii_card_only"] = [
    k for k in sets["families_ii"] if models[kidx[k]]["card"]
]
sets["all_distinct_card_only"] = [m["key"] for m in models if m["card"]]
for sz in ["0.6B", "0.8B", "2B", "4B", "9B", "27B"]:
    sets["pool_" + sz] = [m["key"] for m in models if m["size"] == sz]
    sets["own2_" + sz] = [
        m["key"]
        for m in models
        if m["size"] == sz and m["lineage"] in ("rel2", "cand2")
    ]
sets["own2_all"] = [m["key"] for m in models if m["lineage"] in ("rel2", "cand2")]

item_out = {}
item_private = {}
for sname in ["all_distinct", "families_ii"]:
    rows = [kidx[k] for k in sets[sname]]
    st, p, rpb = item_stats(X[rows])
    item_out[sname] = st
    item_private[sname] = {
        ids[j]: {"p": f(p[j]), "rpb": f(rpb[j])} for j in range(nitem)
    }
easy_scores = sorted(m["tiers"]["easy"] for m in models)
with open(os.path.join(OUT, "item_stats.json"), "w") as fh:
    json.dump(item_private, fh)
os.chmod(os.path.join(OUT, "item_stats.json"), 0o600)

# ---------- 2c split-half + KR-20 ----------
strata = {}
for j in range(nitem):
    strata.setdefault((tier_of[j], type_of[j]), []).append(j)
rng2 = np.random.default_rng(SEED + 1)
splits = []
for _ in range(1000):
    a = np.zeros(nitem, bool)
    for js in strata.values():
        js = np.array(js)
        perm = rng2.permutation(js)
        nh = len(js) // 2 + (rng2.random() < 0.5 if len(js) % 2 else 0)
        a[perm[:nh]] = True
    splits.append(a)


def kr20(Xs):
    k = Xs.shape[1]
    p = Xs.mean(axis=0)
    var = Xs.sum(axis=1).var(ddof=0)
    return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / var)) if var > 0 else None


def split_half(Xs):
    rs, ss = [], []
    for a in splits:
        h1, h2 = Xs[:, a].sum(axis=1), Xs[:, ~a].sum(axis=1)
        if h1.std() == 0 or h2.std() == 0:
            continue
        rs.append(np.corrcoef(h1, h2)[0, 1])
        ss.append(stats.spearmanr(h1, h2).statistic)
    if not rs:
        return None
    r = float(np.mean(rs))
    return {
        "pearson_mean": f(r),
        "spearman_mean": f(np.mean(ss)),
        "pearson_p05_p95": [f(np.percentile(rs, 5)), f(np.percentile(rs, 95))],
        "spearman_brown": f(2 * r / (1 + r)) if r > -1 else None,
    }


def between_sd(Xs):
    return float(Xs.sum(axis=1).std(ddof=1))


rel_pools = {}
for sname, keys in sets.items():
    rows = [kidx[k] for k in keys]
    if len(rows) < 3:
        continue
    Xs = X[rows]
    tot = Xs.sum(axis=1)
    ent = {
        "n_models": len(rows),
        "total_range": [int(tot.min()), int(tot.max())],
        "between_model_sd": f(between_sd(Xs), 2),
        "median_boot_se": f(np.median([models[r]["boot_se"] for r in rows]), 2),
        "kr20_all": f(kr20(Xs)),
        "split_half_all": split_half(Xs),
    }
    ent["signal_to_noise_var_ratio"] = (
        f(ent["between_model_sd"] ** 2 / ent["median_boot_se"] ** 2, 2)
        if ent["median_boot_se"]
        else None
    )
    for t in TIERS:
        Xt = Xs[:, tier_idx[t]]
        ent["kr20_" + t] = f(kr20(Xt))
        tt = Xt.sum(axis=1)
        ent["range_" + t] = [int(tt.min()), int(tt.max())]
    rel_pools[sname] = ent

# ---------- paired helpers ----------
Z_A, Z_B = stats.norm.ppf(0.975), stats.norm.ppf(0.8)


def reject_region(n, alpha=0.05):
    if n == 0:
        return np.zeros(1, bool)
    b = np.arange(n + 1)
    pv = np.minimum(
        1, 2 * np.minimum(stats.binom.cdf(b, n, 0.5), stats.binom.sf(b - 1, n, 0.5))
    )
    return pv <= alpha


def exact_power(n, D):
    if n == 0:
        return 0.0
    p1 = min(1.0, max(0.0, (n + D) / (2 * n)))
    rr = reject_region(n)
    return float(stats.binom.pmf(np.arange(n + 1), n, p1)[rr].sum())


def exact_mdd(n):
    for D in range(0, n + 1):
        if exact_power(n, D) >= 0.8:
            return D
    return None


def n_needed(q, delta):
    """Smallest panel size N with exact McNemar power >= .8 for discordance rate q and per-item gap delta."""
    if q <= 0 or delta <= 0 or delta > q:
        return None
    approx = (Z_A * math.sqrt(q) + Z_B * math.sqrt(q - delta**2)) ** 2 / delta**2
    for N in range(max(10, int(approx * 0.7)), int(approx * 2) + 50):
        n, D = round(q * N), round(delta * N)
        if n > 0 and exact_power(n, D) >= 0.8:
            return {"connor_approx": int(math.ceil(approx)), "exact": N}
    return {"connor_approx": int(math.ceil(approx)), "exact": None}


def paired(va, vb):
    b = int(((va == 1) & (vb == 0)).sum())
    c = int(((va == 0) & (vb == 1)).sum())
    n = b + c
    p = float(stats.binomtest(b, n, 0.5).pvalue) if n else 1.0
    d = X_row_boot(va) - X_row_boot(vb)
    lo_, hi_ = np.percentile(d, [2.5, 97.5])
    out = {
        "a_total": int(va.sum()),
        "b_total": int(vb.sum()),
        "diff": int(va.sum() - vb.sum()),
        "b_a_only": b,
        "c_b_only": c,
        "discordant": n,
        "mcnemar_exact_p": f(p),
        "boot_ci95_diff": [float(lo_), float(hi_)],
        "mdd_normal_items": f(math.sqrt(n) * (Z_A + Z_B), 1),
        "mdd_exact_items": exact_mdd(n),
        "power_true_gap_5": f(exact_power(n, 5), 3),
    }
    q = n / nitem
    out["panel_n_for_80pct_power_gap5_of231"] = n_needed(q, 5 / nitem)
    return out


def X_row_boot(v):
    return v.astype(float)[draw].sum(axis=1)


# ---------- 2d retest ----------
retest = []
for g, kind, ra, rb in RETEST:
    va, vb = vec(ra), vec(rb)
    b = int(((va == 1) & (vb == 0)).sum())
    c = int(((va == 0) & (vb == 1)).sum())
    pa = mat["runs"][ra]["predicted"]
    pb = mat["runs"][rb]["predicted"]
    pred_diff = sum(
        1
        for i in ids
        if json.dumps(pa[i], sort_keys=True) != json.dumps(pb[i], sort_keys=True)
    )
    retest.append(
        {
            "group": g,
            "kind": kind,
            "a": ra,
            "b": rb,
            "a_total": int(va.sum()),
            "b_total": int(vb.sum()),
            "flips": b + c,
            "a_only": b,
            "b_only": c,
            "identical_vector": bool(np.array_equal(va, vb)),
            "predicted_label_differs_items": pred_diff,
        }
    )

# ---------- 2e all same-tier pairs ----------
pair_rows = []
for sz in ["0.6B", "0.8B", "2B", "4B", "9B", "27B"]:
    ks = sets["pool_" + sz]
    for a, b in itertools.combinations(ks, 2):
        ma, mb = models[kidx[a]], models[kidx[b]]
        pr = paired(ma["vec"], mb["vec"])
        pr.update(
            {
                "size": sz,
                "a": a,
                "b": b,
                "sibling_2p0": ma["lineage"] in ("rel2", "cand2")
                and mb["lineage"] in ("rel2", "cand2"),
            }
        )
        pair_rows.append(pr)


def summarize_pairs(rows):
    if not rows:
        return None
    nd = np.array([r["discordant"] for r in rows])
    md = np.array(
        [
            np.nan if r["mdd_exact_items"] is None else r["mdd_exact_items"]
            for r in rows
        ],
        float,
    )
    mn = np.array([r["mdd_normal_items"] for r in rows], float)
    sig = sum(1 for r in rows if r["mcnemar_exact_p"] < 0.05)
    return {
        "pairs": len(rows),
        "median_discordant": f(np.median(nd), 1),
        "iqr_discordant": [f(np.percentile(nd, 25), 1), f(np.percentile(nd, 75), 1)],
        "median_mdd_normal_items": f(np.median(mn), 1),
        "iqr_mdd_normal_items": [
            f(np.percentile(mn, 25), 1),
            f(np.percentile(mn, 75), 1),
        ],
        "median_mdd_exact_items_reachable": (
            f(np.nanmedian(md), 1) if np.isfinite(md).any() else None
        ),
        "pairs_exact_80pct_unreachable": int(np.isnan(md).sum()),
        "significant_p05": sig,
        "median_abs_diff": f(np.median([abs(r["diff"]) for r in rows]), 1),
    }


pair_summary = {}
for sz in ["0.6B", "0.8B", "2B", "4B", "9B", "27B", "all"]:
    rows = [r for r in pair_rows if sz == "all" or r["size"] == sz]
    pair_summary[sz] = {
        "all_pairs": summarize_pairs(rows),
        "non_sibling_pairs": summarize_pairs([r for r in rows if not r["sibling_2p0"]]),
        "sibling_2p0_pairs": summarize_pairs([r for r in rows if r["sibling_2p0"]]),
    }
focus = []
for label, ra, rb in FOCUS:
    pr = paired(vec(ra), vec(rb))
    pr.update({"label": label, "a": ra, "b": rb})
    focus.append(pr)


# ---------- 3 validity ----------
def metric(m, name):
    if name == "public_total":
        return m["total"]
    if name == "tier_macro":
        return float(np.mean([m["tiers"][t] / len(tier_idx[t]) for t in TIERS]))
    if name == "standard_only":
        return m["tiers"]["standard"]
    if name == "hard_only":
        return m["tiers"]["hard"]
    if name == "easy_only":
        return m["tiers"]["easy"]
    if name == "log10_params":
        return math.log10(m["params_loaded"]) if m["params_loaded"] else None
    return m.get(name)


PUB = ["public_total", "tier_macro", "standard_only", "hard_only"]
CRIT = ["v3", "T", "H", "choice", "noul", "score", "log10_params", "c1", "mlx"]
rng3 = np.random.default_rng(SEED + 2)


def corr_ci(x, y, nb=2000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 4 or x.std() == 0 or y.std() == 0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    rho = float(stats.spearmanr(x, y).statistic)
    rb, sb = [], []
    for _ in range(nb):
        ix = rng3.integers(0, n, n)
        xs, ys = x[ix], y[ix]
        if xs.std() == 0 or ys.std() == 0:
            continue
        rb.append(np.corrcoef(xs, ys)[0, 1])
        sb.append(stats.spearmanr(xs, ys).statistic)
    sb = [s for s in sb if np.isfinite(s)]
    return {
        "n": n,
        "pearson": f(r, 3),
        "pearson_ci": [f(np.percentile(rb, 2.5), 3), f(np.percentile(rb, 97.5), 3)],
        "spearman": f(rho, 3),
        "spearman_ci": [f(np.percentile(sb, 2.5), 3), f(np.percentile(sb, 97.5), 3)],
        "pearson_p": f(stats.pearsonr(x, y).pvalue, 4),
        "spearman_p": f(stats.spearmanr(x, y).pvalue, 4),
    }


def resid(y, Z):
    Z = np.column_stack([np.ones(len(y))] + [np.asarray(z, float) for z in Z])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta, beta


def partial(x, y, z, rank=False):
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    if rank:
        x, y, z = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    rx, _ = resid(x, [z])
    ry, _ = resid(y, [z])
    if rx.std() == 0 or ry.std() == 0:
        return None
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    t = r * math.sqrt((n - 3) / max(1e-12, 1 - r * r))
    return {"r": f(r, 3), "p": f(2 * stats.t.sf(abs(t), n - 3), 4), "n": n}


def within_tier(ms, a, b):
    xs, ys = [], []
    for sz in set(m["size"] for m in ms):
        grp = [
            m
            for m in ms
            if m["size"] == sz and metric(m, a) is not None and metric(m, b) is not None
        ]
        if len(grp) < 2:
            continue
        xa = np.array([metric(m, a) for m in grp], float)
        ya = np.array([metric(m, b) for m in grp], float)
        xs += list(xa - xa.mean())
        ys += list(ya - ya.mean())
    if len(xs) < 4 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return {
        "n": len(xs),
        "pearson": f(np.corrcoef(xs, ys)[0, 1], 3),
        "spearman": f(stats.spearmanr(xs, ys).statistic, 3),
    }


validity = {}
for sname in [
    "all_distinct",
    "families_ii",
    "families_iii_card_only",
    "all_distinct_card_only",
]:
    ms = [models[kidx[k]] for k in sets[sname]]
    ent = {}
    for pm in PUB:
        ent[pm] = {}
        for cr in CRIT:
            sub = [m for m in ms if metric(m, cr) is not None]
            x = [metric(m, pm) for m in sub]
            y = [metric(m, cr) for m in sub]
            e = {"raw": corr_ci(x, y)}
            if cr not in ("log10_params",) and len(sub) >= 5:
                sp = [m for m in sub if metric(m, "log10_params") is not None]
                xp = [metric(m, pm) for m in sp]
                yp = [metric(m, cr) for m in sp]
                lp = [metric(m, "log10_params") for m in sp]
                e["partial_given_log10_params_pearson"] = partial(xp, yp, lp)
                e["partial_given_log10_params_spearman"] = partial(
                    xp, yp, lp, rank=True
                )
                e["within_size_tier_demeaned"] = within_tier(sub, pm, cr)
            ent[pm][cr] = e
    validity[sname] = ent

# ---------- 4 information beyond v3 ----------
beyond = {}
for sname in ["all_distinct", "families_ii", "families_iii_card_only"]:
    ms = [models[kidx[k]] for k in sets[sname]]
    y = np.array([m["total"] for m in ms], float)
    v3 = np.array([m["v3"] for m in ms])
    T = np.array([m["T"] for m in ms])
    H = np.array([m["H"] for m in ms])
    r1, b1 = resid(y, [v3])
    sd1 = math.sqrt((r1**2).sum() / (len(y) - 2))
    r2, b2 = resid(y, [T, H])
    ent = {
        "n": len(ms),
        "fit_v3": {
            "intercept": f(b1[0], 2),
            "slope": f(b1[1], 3),
            "r2": f(1 - (r1**2).sum() / ((y - y.mean()) ** 2).sum(), 3),
            "resid_sd": f(sd1, 2),
        },
        "fit_T_H": {"r2": f(1 - (r2**2).sum() / ((y - y.mean()) ** 2).sum(), 3)},
    }
    zs = lambda a: (a - a.mean()) / a.std(ddof=1)
    _, bz = resid(zs(y), [zs(T), zs(H)])
    ent["fit_T_H"]["std_beta_T"] = f(bz[1], 3)
    ent["fit_T_H"]["std_beta_H"] = f(bz[2], 3)
    comp = {}
    for pm in ["public_total", "easy_only", "standard_only", "hard_only"]:
        yy = np.array([metric(m, pm) for m in ms], float)
        comp[pm] = (
            {
                "r_T": f(np.corrcoef(yy, T)[0, 1], 3),
                "r_H": f(np.corrcoef(yy, H)[0, 1], 3),
                "partial_T_given_H": partial(yy, T, H),
                "partial_H_given_T": partial(yy, H, T),
            }
            if yy.std() > 0
            else None
        )
    ent["component_tracking"] = comp
    rows = []
    for m, r in zip(ms, r1):
        z = r / sd1
        m.setdefault("resid", {})[sname] = f(r, 2)
        rows.append(
            {
                "key": m["key"],
                "label": m["label"],
                "size": m["size"],
                "lineage": m["lineage"],
                "public": m["total"],
                "v3": f(m["v3"], 2),
                "resid_items": f(r, 2),
                "z": f(z, 2),
                "outlier_2sd": bool(abs(z) > 2),
            }
        )
    ent["residuals"] = sorted(rows, key=lambda r: -r["z"])
    # mlx residual vs public residual
    sub = [(m, r) for m, r in zip(ms, r1) if m["mlx"] is not None]
    if len(sub) >= 5:
        mv = np.array([m["v3"] for m, _ in sub])
        my = np.array([m["mlx"] for m, _ in sub])
        rm, _ = resid(my, [mv])
        ent["public_resid_vs_mlx_resid"] = corr_ci([r for _, r in sub], rm)
        ent["partial_public_mlx_given_v3"] = partial(
            [m["total"] for m, _ in sub], my, mv
        )
    beyond[sname] = ent

# C1 block (n = 10)
c1m = [m for m in models if m["c1"] is not None]
yc = np.array([m["c1"] for m in c1m])
vc = np.array([m["v3"] for m in c1m])
pc = np.array([m["total"] for m in c1m], float)
rc1, _ = resid(yc, [vc])
rp1, _ = resid(pc, [vc])


def loo(y, Zs):
    n = len(y)
    press = 0.0
    for i in range(n):
        mask = np.arange(n) != i
        Z = np.column_stack([np.ones(n)] + list(Zs))
        beta, *_ = np.linalg.lstsq(Z[mask], y[mask], rcond=None)
        press += (y[i] - Z[i] @ beta) ** 2
    return {
        "loo_rmse": f(math.sqrt(press / n), 3),
        "loo_q2": f(1 - press / ((y - y.mean()) ** 2).sum(), 3),
    }


c1_block = {
    "n": len(c1m),
    "models": [
        {"key": m["key"], "c1": m["c1"], "public": m["total"], "v3": f(m["v3"], 2)}
        for m in c1m
    ],
    "public_vs_c1": corr_ci(pc, yc),
    "v3_vs_c1": corr_ci(vc, yc),
    "public_resid_vs_c1_resid_within_c1_set": corr_ci(rp1, rc1),
    "partial_public_c1_given_v3": partial(pc, yc, vc),
    "partial_public_c1_given_v3_spearman": partial(pc, yc, vc, rank=True),
    "loo_c1_from_v3": loo(yc, [vc]),
    "loo_c1_from_public": loo(yc, [pc]),
    "loo_c1_from_v3_plus_public": loo(yc, [vc, pc]),
    "loo_c1_from_T_H": loo(
        yc, [np.array([m["T"] for m in c1m]), np.array([m["H"] for m in c1m])]
    ),
    "by_event": {},
}
cv = [m for m in c1m if m["key"] not in ("kai1", "lex1", "gliner25")]
c1_block["sensitivity_excluding_invalid_heavy_kai_lex_gliner"] = {
    "n": len(cv),
    "public_vs_c1": corr_ci([m["total"] for m in cv], [m["c1"] for m in cv]),
    "v3_vs_c1": corr_ci([m["v3"] for m in cv], [m["c1"] for m in cv]),
    "partial_public_c1_given_v3": partial(
        [m["total"] for m in cv], [m["c1"] for m in cv], [m["v3"] for m in cv]
    ),
    "loo_c1_from_v3": loo(
        np.array([m["c1"] for m in cv]), [np.array([m["v3"] for m in cv])]
    ),
    "loo_c1_from_v3_plus_public": loo(
        np.array([m["c1"] for m in cv]),
        [np.array([m["v3"] for m in cv]), np.array([m["total"] for m in cv], float)],
    ),
}
for ev, sz in [("event1_0.8B", "0.8B"), ("event2_0.6B", "0.6B")]:
    g = [m for m in c1m if m["size"] == sz]
    c1_block["by_event"][ev] = {
        "n": len(g),
        "spearman_public_c1": f(
            stats.spearmanr([m["total"] for m in g], [m["c1"] for m in g]).statistic, 3
        ),
        "spearman_v3_c1": f(
            stats.spearmanr([m["v3"] for m in g], [m["c1"] for m in g]).statistic, 3
        ),
    }

# ---------- output ----------
model_rows = [
    {k: (v if k != "vec" else None) for k, v in m.items() if k != "vec"} for m in models
]
for r in model_rows:
    r["v3"] = f(r["v3"], 3)
    r["T"] = f(r["T"], 4)
    r["H"] = f(r["H"], 4)
print(
    json.dumps(
        {
            "seed": SEED,
            "bootstrap_draws": B,
            "split_half_splits": 1000,
            "n_distinct_models": nmod,
            "models_without_params_loaded": [
                m["key"] for m in models if not m["params_loaded"]
            ],
            "identical_vectors_among_distinct_models": dup,
            "models": model_rows,
            "sets": sets,
            "reliability": {
                "bootstrap_ci_halfwidth": {
                    "median": f(np.median(halfw), 2),
                    "min": f(halfw.min(), 2),
                    "max": f(halfw.max(), 2),
                    "median_se": f(np.median([m["boot_se"] for m in models]), 2),
                },
                "item_stats": item_out,
                "easy_tier_score_distribution": {
                    str(s): easy_scores.count(s) for s in sorted(set(easy_scores))
                },
                "pools": rel_pools,
                "retest": retest,
                "pair_summary": pair_summary,
                "focus_pairs": focus,
                "same_tier_pairs": [{k: v for k, v in r.items()} for r in pair_rows],
            },
            "validity": validity,
            "beyond_v3": beyond,
            "c1": c1_block,
        },
        default=lambda o: o.item() if hasattr(o, "item") else str(o),
    )
)
