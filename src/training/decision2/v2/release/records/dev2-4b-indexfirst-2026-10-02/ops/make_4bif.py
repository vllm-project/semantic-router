"""Release spec and decision of the Decision-2.0-Nox-4B Index-first successor (worker 5e7b8132).

User rule 2026-10-02 09:55 UTC+8 (COORDINATION "INDEX-FIRST"): the only quality gate is the frozen candidate's private
Jev Decision Index delta vs the current release, significantly positive (paired bootstrap over rows within
benchmarks through the board's area weights, 2,000 replicates, 95% lower bound > 0; ``gate.py`` IF1, which binds the
two IX1 run receipts to exactly this package's weights and the current revision's). Integrity checks: R3 (no type
collapsed on the formal typed panel), IF3 (the row-level Index contamination audit of the training file), and the
release items 2-7 (package parity, download hashes, examples, output consistency, card, Transformers remote code with
the Hub smoke under 5.17 / 5.18). v3, human transfer, mlx-diag, the tier gates, overlap exposure and public 231 are
references (``1_successor_references``); C1 is not run. Among the 4B candidates the one with the largest lower bound
ships (``CHOICE`` below, from the private bootstraps; values stay private).

The spec derives from the round-3 product spec of the current revision (specs/dev2-4b-card3.json, ``main``
54b084f9): roster, peers, runtime (BF16-resident, forward token budget, Transformers remote code; runtime_source =
automap_source of the current revision), remote code, licence and the product card carry over; the weights, the
scored run, the gate evidence and the card's Index input and assets (the default generator: banner concept A) are
the candidate's.

Run on node A from the exact mirror holding this file (host python3; every hashed file is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> --cand CAND --out DIR [--check]
It writes DIR/dev2-4b-4bif-CAND.json and DIR/Decision-2.0-Nox-4B.decision.4bif-CAND.json; the committed copies live
in v2/release/specs and this record directory, and --check compares them byte for byte.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from v2.release import gate

ROOT = Path(__file__).resolve().parents[5]
RECORD = Path(__file__).resolve().parents[1]
BASE_SPEC = ROOT / "v2/release/specs/dev2-4b-card3.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Nox-4B"
REPO = f"llm-semantic-router/{NAME}"
REL = "/data/dev2/runs/release"
FR = "/data/dev2/runs/dec/formal"
DECISIONS = f"{REL}/decisions"
CURRENT_RUN = f"{REL}/dev2-4b-lh-t1-derived"
CURRENT_MLX = f"{REL}/dev2-4b-lh-t1-derived-mlx"
CURRENT = {
    "revision": "54b084f98e1713480f7d5b83b11e37171cba2270",
    "gate_sha256": "6f03954ab404287e68f93a91fdfe094e15a899d73f7c5e7833fc5b5574511f53",
    "decision_sha256": "6f8eb8e9d5338964017f3287af4a2aafa3540fa83f978581a3751e0f21b4dd2d",
    "manifest_sha256": "dd0ee28d812265121b10944c80cc8a3b1673797ccbcedd0ff36ccda6865ad3ed",
    "weights_identity": "6a555335e077fd26952fd58df06a7dcca1f5ef317194f2e3e4d7664d5c071cd7",
}
TRAIN = "d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5"
LH_FAMILIAR_ROWS = 114
LH_ARM = (
    "4b-LHA10SD = the released LH recipe (rank-128 LoRA on Qwen/Qwen3.5-4B-Base @1001bb4d, merged, scoring head) "
    "on the released 4B mixture plus 10% IB1-r3 / IB2 breadth rows, with typed-row self-distillation from LH "
    "(decoder M13)"
)
CANDS = {
    "a75": {
        "index_name": "DEV2.0-4B-LHA10SD-a75-bf16",
        "run": f"{FR}/m16/m16-4b-LHA10SD-a75",
        "fp32": "96d09b160fd0f3b8673620ef37c1d226cab5afa9d1634f156c4151206a2ccc73",
        "vendor": "5b246b11096adbb8df73b6ba34f96b7373f7c95c",
        "what": "the decoder M16 interpolation W = 0.25 x LH + 0.75 x 4b-LHA10SD (per tensor in FP32); "
        + LH_ARM,
        "exposure": f"{FR}/m16/exposure-4b-LHA10SD-a75.json",
    },
    "a50": {
        "index_name": "DEV2.0-4B-LHA10SD-a50-bf16",
        "run": f"{FR}/m16/m16-4b-LHA10SD-a50",
        "fp32": "84f462f2515526601d2726b9f78cda75135d5c66ba0cee14fc4b25d30450a464",
        "vendor": "5b246b11096adbb8df73b6ba34f96b7373f7c95c",
        "what": "the decoder M16 interpolation W = 0.5 x LH + 0.5 x 4b-LHA10SD (per tensor in FP32); "
        + LH_ARM,
        "exposure": f"{FR}/m16/exposure-4b-LHA10SD-a50.json",
    },
    "UP": {
        "index_name": "DEV2.0-4B-LHA10UP-bf16",
        "run": f"{FR}/4bif/4bif-4b-LHA10UP",
        "fp32": "9e80e7654965f539b78fe91c6428253ccf565bf29a9550d1f4c94e911b77007e",
        "vendor": "87a991fe50d76211d80fd80fde24a197c5cad280",
        "what": "the decoder M14 arm 4b-LHA10UP (the M13 recipe with the IB rows upweighted; soup of two seeds)",
        "exposure": f"{FR}/m16/exposure-4b-LHA10SD-a75.json",
    },
    "SDB": {
        "index_name": "DEV2.0-4B-LHA10SD-bf16",
        "run": f"{FR}/m13/m13-4b-LHA10SD",
        "fp32": "255021e0a3f2af49d41ffbd3425d541e6d7749b8cd95a8e8a7b764a81e8a037d",
        "vendor": "c2610143be2ff302c080a5cdfe093e41d173f8da",
        "what": "the decoder M13 arm " + LH_ARM + "; soup of two seeds",
        "exposure": f"{FR}/m13/exposure-4b-LHA10SD.json",
    },
    "S10": {
        "index_name": "DEV2.0-4B-LHS10SD-bf16",
        "run": f"{FR}/m17/m17-4b-LHS10SD",
        "fp32": "537553da252d79e9fd4aca7731e74367cf15899788b80c912c3d4fc07a3f28ea",
        "vendor": "d80933b6c9c26e7b77ceb1342f261a284bdc92eb",
        "what": "the decoder M17 arm 4b-LHS10SD (the released LH recipe at LH's token count with English released "
        "rows of 10% of the tokens swapped for IB1-r3 / IB2 breadth rows, every non-English row kept, typed-row "
        "self-distillation from LH; soup of two seeds)",
        "exposure": f"{FR}/m17/exposure-4b-LHS10SD.json",
        "train": "72fa2d844fbf94be890858b9b66af0e26e12a62011929bf1eb0de1ebe93ef025",
    },
    "S17": {
        "index_name": "DEV2.0-4B-LHS17SD-bf16",
        "run": f"{FR}/m17/m17-4b-LHS17SD",
        "fp32": "8995bd9d58a3aab7c83baee47adfb1ce0e4782f51097b98a3d0a2247ddf14079",
        "vendor": "d80933b6c9c26e7b77ceb1342f261a284bdc92eb",
        "what": "the decoder M17 arm 4b-LHS17SD (the released LH recipe at LH's token count with English released "
        "rows of 17% of the tokens swapped for IB1-r3 / IB2 breadth rows, every non-English row kept, typed-row "
        "self-distillation from LH; soup of two seeds)",
        "exposure": f"{FR}/m17/exposure-4b-LHS17SD.json",
        "train": "14bce13ce926b354e214581e7cf4718d03f80c975b36dbce6731fbc6517e25a0",
    },
}
# The release choice: M17 4b-LHS17SD, the largest private Index lower bound vs the current release among the finished
# BF16 Index runs (M13 4b-LHA10SD, M17 4b-LHS10SD / 4b-LHS17SD); directive of 2026-10-02 12:25 UTC+8: release the
# highest-scoring candidate, never a lower one once a higher one is measured. The M13 choice was never published.
CHOICE = "S17"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's Index-first rule of 2026-10-02 09:55 UTC+8 "
    "(release gate = a significantly positive private Index delta vs the current release; integrity checks; "
    "references not blocking; the largest Index-gain lower bound) and the user directive relayed 2026-10-02 12:25 "
    "UTC+8 (release the highest-scoring candidate directly; never a lower candidate once a higher one is measured; "
    "compare every finished BF16 Index run: M13 4b-LHA10SD, M17 4b-LHS10SD and 4b-LHS17SD), applied to the 4B tier "
    "by the 4B Index-first release worker 5e7b8132"
)
PREPARED_BY = "Decision 2.0 release engineering, 4B Index-first worker 5e7b8132 (worktree vllm-sr-dev2-4b-indexfirst)"


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def paths(cand: str) -> dict:
    c = CANDS[cand]
    inputs = f"{REL}/inputs/dev2-4b-4bif-{cand}"
    private = f"/data/dev2/private/release/4bif/{cand}"
    return {
        **c,
        "in": inputs,
        "private": private,
        "mlx": f"{c['run']}-mlx",
        "gates": f"{inputs}/gates",
        "current_gate": f"{inputs}/current/card3-4b-gate.json",
        "current_decision": f"{inputs}/current/{NAME}.decision.card3.json",
    }


def spec(cand: str) -> dict:
    p = paths(cand)
    old = load(BASE_SPEC)
    assert (
        old["repo_id"] == REPO
        and old["expected_identity"]["model_sha256"] == CURRENT["weights_identity"]
    )
    s = copy.deepcopy(old)
    bf16 = load(f"{p['in']}/bf16/bf16-copy.json")
    bf16_receipt = sha(f"{p['in']}/bf16/bf16-copy.json")
    identity = bf16["model_sha256"]
    assert bf16["source_model_sha256"] == p["fp32"]
    receipt = load(f"{p['private']}/receipt.json")
    assert (
        receipt["model_source"]["model_sha256"] == identity
    ), "the Index run scored other weights"
    seal = load(f"{p['run']}/SEAL.json")
    formal_cache = f"{p['in']}/formal-cache"
    mlx_cache = f"{p['in']}/mlx-cache"
    s["_release"] = {
        "successor": f"{NAME} Index-first successor: {p['what']}; stored as its v2.release.bf16_copy.",
        "gate": "successor profile with index_first (user rule 2026-10-02 09:55): IF1 = the private Index paired "
        "bootstrap of these weights minus the current revision's, 95% lower bound > 0, bound to both IX1 run "
        "receipts; R3 = no type collapsed on the formal typed panel; IF3 = the row-level Index contamination "
        "audit of the training file; items 2-7 as for every release. v3, human transfer, mlx-diag, tier gates, "
        "exposure and public 231 are references; C1 is not run. The Index files stay in node A's private tree.",
        "scored": f"T = 1: the sealed formal run {Path(p['run']).name} (image dbe5f32b, the 4B formal library's "
        "master cache), kept at T = 1 by the 23:15 rule, scored on node A by m6-score.sh.",
        "storage": f"v2.release.bf16_copy of the frozen FP32 weights (identity {p['fp32'][:8]} -> {identity[:8]}; "
        f"receipt {bf16_receipt[:8]}), the copy the Index run {p['index_name']} scored.",
        "runtime": "vendor_source = the formal run's runner mirror (training/model checked equal to the scored "
        "adapter sources at build time); runtime_source = automap_source = the current revision's (BF16-resident, "
        "Transformers remote code, forward token budget), unchanged.",
        "card": "the product card of the current revision with this candidate's reports, an Index input built by "
        "python -m v2.release.card_index (board-served parameter counts, the audited footnote) with the 4B point "
        "from the Index run on exactly these weights, and assets rendered by the default v2.release.card_assets "
        "(banner concept A); card.speed keeps the current revision's bench receipt (same architecture, runtime and "
        "shapes).",
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-4b-card3.json",
            "sha256": sha(BASE_SPEC),
        },
        "previous": old["_release"],
    }
    s["checkpoint"] = f"{p['in']}/bf16/checkpoint"
    s["expected_identity"] = {"model_sha256": identity}
    s["bf16_copy"] = {
        "receipt": f"{p['in']}/bf16/bf16-copy.json",
        "sha256": sha(f"{p['in']}/bf16/bf16-copy.json"),
    }
    s["vendor_source"] = (
        f"/data/dev2/src/{p['vendor']}-src_training_decision2/src/training/decision2"
    )
    s["gate_receipt"] = f"{DECISIONS}/{NAME}.decision.4bif-{cand}.json"
    s["runtime_equivalence"] = (
        "decision2/qwen.py loads this full checkpoint with the training/model sources vendored from the formal run's "
        f"own runner mirror ({p['vendor'][:9]}), whose SHA-256 equal the scored adapter sources (checked at build "
        "time), and applies the per-item batching, BF16-backbone / FP32-head execution, raw probabilities "
        "(temperature 1; no calibration file) and answer normalization of v2.dec.infer_dec. The scored checkpoint "
        f"({p['fp32'][:8]}) stored every tensor in FP32; this package (v2.release.bf16_copy) stores its 248 Linear "
        "projection matrices in BF16 exactly as BF16 autocast rounds them and every other tensor bit for bit in "
        "FP32. The runtime, the Transformers remote code and the forward token budget are the current revision's. "
        "Checked on one GPU against the formal run's predictions of every scored prompt (typed-final 1,600, css15 "
        "6,547, public231 231) and of the mlx-diag diagnostic (2,275) by release.sh --parity, with copies of the "
        "formal and mlx-diag runs' persisted Triton autotune caches, and AutoModel against the native runtime on "
        "every scored prompt."
    )
    s["scored"] = {
        "label": f"post-key same-panel run {Path(p['run']).name} at T = 1 (the formal run kept T = 1 under the "
        "23:15 rule; sealed)",
        "report_sha256": sha(f"{p['run']}/REPORT.json"),
        "seal_sha256": sha(f"{p['run']}/SEAL.json"),
        "predictions_sha256": {
            **{
                panel: seal["panels"][panel]["predictions_sha256"]
                for panel in ("typed-final", "css15", "public231")
            },
            "mlx-diag": sha(f"{p['mlx']}/output/mlx-diag.predictions.jsonl"),
        },
        "paired_sha256": sha(f"{p['run']}/PAIRED-vs-adopted-1.0.json"),
        "native_manifest": f"{p['run']}/output/typed-final.predictions.jsonl.manifest.json",
    }
    card = s["card"]
    card["paired"] = f"{p['gates']}/paired-vs-adopted-1.0.json"
    card["paired_peers"] = {
        "decider4b": f"{p['run']}/PAIRED-vs-decider4b.json",
        "jet62": f"{p['run']}/PAIRED-vs-jet62.json",
    }
    reports = card["reports"]
    assert reports[0]["role"] == "candidate" and reports[0]["label"] == NAME
    reports[0]["report"] = f"{p['run']}/REPORT.json"
    reports[0]["mlx"] = f"{p['mlx']}/mlx-diag.score.json"
    card["index"] = {
        "path": f"{p['private']}/decision-index-card.json",
        "sha256": sha(f"{p['private']}/decision-index-card.json"),
    }
    card["assets"] = {
        "dir": f"{p['private']}/4b",
        "receipt_sha256": sha(f"{p['private']}/4b/card-assets.json"),
    }
    s["frozen_autotune_cache"] = {
        "formal": f"{formal_cache} (the formal run's persisted cache, manifest "
        f"{sha(formal_cache + '.sha256')[:8]}, relayed from the collecting node)",
        "mlx-diag": f"{mlx_cache} (the mlx-diag run's persisted cache, manifest "
        f"{sha(mlx_cache + '.sha256')[:8]}, relayed from the collecting node)",
    }
    s["gate_profile"] = {
        "name": "successor",
        "run": p["run"],
        "current": {
            "revision": CURRENT["revision"],
            "gate": p["current_gate"],
            "decision": p["current_decision"],
            "run": CURRENT_RUN,
            "mlx_predictions": f"{CURRENT_MLX}/output/mlx-diag.predictions.jsonl",
        },
        "types": f"{p['gates']}/types.json",
        "paired": f"{p['gates']}/paired-vs-nox-4b.json",
        "mlx_paired": f"{p['gates']}/mlx-paired-vs-nox-4b.json",
        "public231": f"{p['gates']}/public231-vs-nox-4b.json",
        "exposure": [p["exposure"]],
        "tier": {
            "reference": "decider4b",
            "v3_share": 0.9,
            "paired": f"{p['gates']}/paired-vs-decider4b.json",
        },
        "index_first": {
            "bootstrap": f"{p['private']}/boot-full-vs-lh.json",
            "receipt": f"{p['private']}/receipt.json",
            "base_receipt": f"{p['private']}/base-receipt.json",
            "audit": f"{p['private']}/audit.json",
        },
    }
    return s


def audit_disclosure(p: dict, sets: dict) -> str:
    if "train" not in p:
        return (
            f"Index contamination audit of the training rows ({TRAIN[:8]}): the IB rows add no item or "
            "familiar-text rows beyond the current revision's own training rows"
        )
    own = sets[p["run"].rsplit("/", 1)[1].removeprefix("m17-")]
    return (
        f"Index contamination audit of the training rows ({p['train'][:8]}, {own['training_lines']} lines): "
        f"{own['item_rows']} item rows and {own['duplicate_rows']} familiar-text rows (the current revision's own "
        f"training rows: 0 and {LH_FAMILIAR_ROWS})"
    )


def decision(s: dict, cand: str) -> dict:
    p = paths(cand)
    profile = gate.gate_profile(s)
    items = gate.successor_items(s, profile)
    failed = [k for k, v in items.items() if not v["passed"]]
    if failed:
        raise SystemExit(f"Index-first items fail: {failed}")
    paired = load(profile["paired"])
    low, high = gate._low_high(paired["ci95"])
    h = paired["axis_ci95"]["H"]["delta"]
    v3 = load(f"{p['run']}/REPORT.json")["v3"]["score"]
    current_v3 = load(f"{CURRENT_RUN}/REPORT.json")["v3"]["score"]
    public = load(profile["public231"])
    mlx = load(profile["mlx_paired"])["overall"]
    own = load(s["card"]["paired"])
    audit = load(profile["index_first"]["audit"])
    sets = audit["training_sets"]
    return {
        "schema": gate.DECISION_SCHEMA,
        "status": "final",
        "decision": "release",
        "model_name": NAME,
        "repo_id": REPO,
        "name_basis": s.get("name_basis"),
        "name_base_model": s.get("name_base_model"),
        "identity": s["expected_identity"],
        "report_sha256": s["scored"]["report_sha256"],
        "paired_sha256": sha(s["card"]["paired"]),
        "gate_profile": "successor",
        "rule": gate.INDEX_FIRST,
        "current_revision": CURRENT["revision"],
        "evidence_sha256": gate.evidence_sha256(profile),
        "calibration": "none (temperature 1; every Decision 2.0 model keeps T = 1): the formal run kept T = 1 under "
        "the 23:15 rule; no calibration file is shipped",
        "action": f"New main revision of the private repository {REPO}: {p['what']}, stored as its BF16 copy "
        f"(qwen-full, T = 1, 16,384 tokens), replaces the LH weights ({CURRENT['weights_identity'][:8]}, revision "
        f"{CURRENT['revision'][:8]}) with the product card of the current revision (default generator, banner "
        "concept A) and this candidate's Index input; then the superseded weight blobs are purged with "
        "rewrite_history=False (hf_headroom.sh first). The repository stays private and in the private collection.",
        "rationale": "Index-first rule (user 2026-10-02 09:55 UTC+8). IF1: the private Index paired bootstrap of "
        "these exact weights minus the current revision's has a 95% lower bound > 0 (2,000 replicates; values in "
        "private files only; evidence_sha256.index_first_bootstrap); the largest lower bound among the finished "
        "BF16 Index runs of M13 4b-LHA10SD and M17 4b-LHS10SD / 4b-LHS17SD (same panel, seed and replicates; "
        "directive 2026-10-02 12:25 UTC+8). R3: types "
        + ", ".join(
            f"{k} {v}" for k, v in gate._verdicts(load(profile["types"])).items()
        )
        + ". IF3: row-level Index audit, planted control "
        f"{audit['planted_control']['found']} / {audit['planted_control']['planted']}; "
        + "; ".join(f"{n} {v['item_rows']} item rows" for n, v in sorted(sets.items()))
        + ". References (not release blockers): post-key v3 "
        f"{v3:.3f} vs {current_v3:.3f}, {signed(paired['point']['delta']['score'])} [{signed(low)}, {signed(high)}]; "
        f"human transfer {signed(h['low'], 3)} to {signed(h['high'], 3)}; card-eligible mlx-diag "
        f"{signed(mlx['delta'], 4)} [{signed(mlx['ci95']['low'], 4)}, {signed(mlx['ci95']['high'], 4)}]; vs "
        f"adopted Nox 1.0 {signed(own['point']['delta']['score'])} [{signed(gate._low_high(own['ci95'])[0])}, "
        f"{signed(gate._low_high(own['ci95'])[1])}]; public 231 {public['left_correct']} vs "
        f"{public['right_correct']} (McNemar p {public['mcnemar_exact_p']:.3f}); C1 post-key not run.",
        "approved_package": {
            "identity": s["expected_identity"]["model_sha256"],
            "fp32_identity": p["fp32"],
            "loaded_parameters": 4208383488,
            "profile": "qwen-full (BF16 storage of the Linear projections)",
            "temperature": 1,
            "max_input_tokens": 16384,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": "apache-2.0: own LoRA (merged), head, runtime and card; base Qwen/Qwen3.5-4B-Base "
        "@1001bb4d Apache-2.0 (LICENSE = Apache License 2.0, Copyright 2026 Alibaba Cloud); tokenizer files of "
        "Decision 1.0 Nox-4B @cde2a68d (Qwen3.5-4B vocabulary, Apache-2.0); the IB1-r3 / IB2 training rows are "
        "licence-clean (release-safe data records); the package keeps the Apache-2.0 LICENSE (product card: no "
        "NOTICE, attributions or evaluation/ files).",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": [
            "successor items 1 and 4 of the former rule are not met (v3 "
            + ("significantly below" if high < 0 else "not significantly above")
            + " the current revision; card-eligible mlx-diag "
            + ("below it" if mlx["ci95"]["high"] < 0 else "not above it")
            + "); under the Index-first rule they are references",
            "the training rows include the IB1-r3 / IB2 families matched to Index benchmark families (HoVer, "
            "When2Call, iSarcasmEval, GSM8K) and the BPoMP format; the internal records report the transfer-only "
            "Index delta without those benchmarks (private values)",
            audit_disclosure(p, sets),
            "card.speed is the current revision's bench receipt (same architecture, runtime and shapes)",
        ],
        "prepared_by": PREPARED_BY,
        "decided_by": DECIDED_BY,
        "decided_utc": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", choices=sorted(CANDS), required=True)
    ap.add_argument("--decided-utc", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--check", action="store_true", help="compare with the committed copies"
    )
    args = ap.parse_args()
    if CHOICE != args.cand:
        raise SystemExit(f"{args.cand} is not the recorded choice ({CHOICE})")
    p = paths(args.cand)
    for path, value in (
        (p["current_gate"], CURRENT["gate_sha256"]),
        (p["current_decision"], CURRENT["decision_sha256"]),
    ):
        if sha(path) != value:
            raise SystemExit(f"{path} is not the current revision's file")
    s = spec(args.cand)
    spec_text = json.dumps(s, indent=2, ensure_ascii=False) + "\n"
    d = decision(s, args.cand)
    d["decided_utc"] = args.decided_utc
    decision_text = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    outputs = {
        f"dev2-4b-4bif-{args.cand}.json": (spec_text, SPECS),
        f"{NAME}.decision.4bif-{args.cand}.json": (decision_text, RECORD),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    problems = []
    for name, (text, committed) in outputs.items():
        (args.out / name).write_text(text, encoding="utf-8")
        if args.check and (committed / name).read_text(encoding="utf-8") != text:
            problems.append(f"{committed / name} differs from the derivation")
        print(name, hashlib.sha256(text.encode()).hexdigest())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
