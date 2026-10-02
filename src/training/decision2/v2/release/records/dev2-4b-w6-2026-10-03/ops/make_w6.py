"""Release spec and decision of a Decision-2.0-Nox-4B wave-6 cross-arm soup (4B owner, decoder M17b, the only Nox-4B
publisher).

Rule (user 2026-10-03 01:27, COORDINATION 01:36): keep optimizing toward same-size first place with progressive
releases under the same rules as before. A wave-6 candidate (prereg v2/dec/records/dec-m17b-wave6-prereg-2026-10-03.md)
is released on top of the current main d55528d1 (M17 4b-SDMLxALL) when it passes the Index-first gate: IF1 = the
private Index paired bootstrap of these weights minus the current revision's (DEV2.0-4B-SDMLxALL-bf16) has a 95% lower
bound > 0. Integrity checks: R3 (no type collapsed on the formal typed panel), IF3 (one row-level Index contamination
audit of the distinct TRAIN files of wave 6's members, m17b-ops.sh audit6) and the release items 2-7. v3, human
transfer, mlx-diag, the tier gates and public 231 are references; overlap exposure and C1 are not run.

The spec derives from the current revision's spec (specs/dev2-4b-xall.json, ``main`` d55528d1): roster, peers, runtime
(the phase A runtime), remote code, licence and the product card carry over; the weights, the scored run, the gate
evidence and the card's Index input and assets (the default generator) are the candidate's.

Run on node A from the exact mirror holding this file (host python3; every hashed file is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> --cand KEY --decided-utc UTC --out DIR [--check]
It writes DIR/dev2-4b-<key>.json and DIR/Decision-2.0-Nox-4B.decision.<key>.json (key = KEY lower-case); the
committed copies live in v2/release/specs and this record directory, and --check compares them byte for byte.
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
BASE_SPEC = ROOT / "v2/release/specs/dev2-4b-xall.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Nox-4B"
REPO = f"vllm-sr/{NAME}"
REL = "/data/dev2/runs/release"
FR = "/data/dev2/runs/dec/formal"
DECISIONS = f"{REL}/decisions"
CURRENT_RUN = f"{FR}/m17/m17-4b-SDMLxALL"
CURRENT_MLX = f"{FR}/m17/m17-4b-SDMLxALL-mlx"
CURRENT = {
    "revision": "d55528d1635fc474061ec59e31a7c722d3e7ab95",
    "gate_sha256": "854b31f7fb81946887929f885f4a3e72d055e64347c2dcfe33a05df481e8b0bd",
    "decision_sha256": "95903cb6b4ffcef4ce1160605d6f7cabfdb6b7f0d162ca93cf33b38cafc878f9",
    "manifest_sha256": "c87715ba9a56cf8e95e97102f9518d9edea7e4282290462f5682d4cef9764133",
    "weights_identity": "48e8c023d0f4b18461fa5f2a64c7113f25839f3cf4d8023893e7631deb0417cb",
}
CURRENT_RECORD = ROOT / "v2/release/records/dev2-4b-xall-2026-10-02"
VENDOR = "95cc6e548ceece6ca852eee729c99da68057e078"
# The phase A runtime plus the opt-in shared-context switch (9d90afd10 merged; COORDINATION 2026-10-03 02:23).
RUNTIME = "b50e8650b87ef5129ea6c26223d494fa86480d4d"
SEVEN = "the released M15 4b-LHA10SDML, 4b-LHS17SD, 4b-LHS17UP, 4b-LHS17IB4, 4b-LHS17IB4X, 4b-SDMLIB4 and 4b-LHS17ML"
NEW3 = "the arm factory's 4b-LHS17IB4-lrh (half LR), 4b-LHS17IB4ML and 4b-LHS23IB4"
RECIPE = (
    "each member a soup of rank-128 LoRA seeds on Qwen/Qwen3.5-4B-Base @1001bb4d (merged, scoring head, typed-row "
    "self-distillation; the released 4B mixture with IB1-r3 / IB2 / IB3-r2 / IB4 phase-1 breadth rows and copies of "
    "released non-English rows in different doses)"
)
CANDS = {
    "AFxALL3": {
        "fp32": "df95b2861f475052b9259dab60fd69d0e2c4d7f474929a0f2f92ecdae824b287",
        "members": 10,
        "what": "the wave-6 cross-arm soup 4b-AFxALL3: the uniform FP32 average of ten arm soups, "
        f"{SEVEN} at their most seeds (UP and ML and IB4X with four seeds, IB4 and SDMLIB4 with five), plus {NEW3}; "
        + RECIPE,
    },
    "AFxALL2": {
        "fp32": "70612d1c270447b3e07ddb94277fc74b7dbb0903b7f885ca35335ba730d27a0b",
        "members": 10,
        "what": "the wave-6 cross-arm soup 4b-AFxALL2: the uniform FP32 average of ten arm soups, "
        f"{SEVEN} (UP with four seeds, IB4 and SDMLIB4 with five), plus {NEW3}; "
        + RECIPE,
    },
    "XALLx": {
        "fp32": "7ae47bc20bd873e770dd7975b04467a5f4756c8e2c5e18b634095445636d1f8b",
        "members": 7,
        "what": "the wave-6 cross-arm soup 4b-XALLx: the uniform FP32 average of the released soup's seven arm "
        f"soups, {SEVEN}, each at its most seeds (UP and ML and IB4X with four seeds, IB4 and SDMLIB4 with five); "
        + RECIPE,
    },
    "XALLU2": {
        "fp32": "58fa05858a1ae5f79b5cea268d7b8fc427f1d23a47644303f617b8a56be75a58",
        "members": 7,
        "what": "the wave-6 cross-arm soup 4b-XALLU2: the FP32 average of the released soup's seven two-seed arm "
        f"soups, {SEVEN}, with 4b-LHS17UP at weight 2/8 and the others at 1/8; "
        + RECIPE,
    },
    "LHS17IB4-lrh": {
        "fp32": "0b40be9478c041e964ce340d3985369bd67b15260c1205b7cf2b7928004fe7f9",
        "members": 1,
        "what": "the arm factory's 4b-LHS17IB4-lrh: the uniform FP32 soup of two seeds (BEST checkpoints 1249 and "
        "1254) of a rank-128 LoRA arm on Qwen/Qwen3.5-4B-Base @1001bb4d (merged, scoring head, typed-row "
        "self-distillation) trained on M17's 4b-LHS17IB4 mixture (the released 4B mixture with IB1-r3 / IB2 swap "
        "rows plus the IB3-r2 / IB4 phase-1 families) at half the LoRA / head learning rate (5e-5)",
    },
    "AFxALL": {
        "fp32": "8f282287620dc5498c09d9bba2f8be0b86fa2f60749a6920e9a760b5fc7a5a93",
        "members": 12,
        "what": "the wave-6 cross-arm soup 4b-AFxALL: the uniform FP32 average of twelve arm soups, "
        f"{SEVEN}, 4b-LHS10SD and 4b-LHS23SD (UP with four seeds, IB4 and SDMLIB4 with five), plus {NEW3}; "
        + RECIPE,
    },
}
AUDIT_SETS = [
    "4b-LHA10SDML",
    "4b-LHS10SD",
    "4b-LHS17IB4",
    "4b-LHS17IB4ML",
    "4b-LHS17IB4X",
    "4b-LHS17ML",
    "4b-LHS17SD",
    "4b-LHS23IB4",
    "4b-LHS23SD",
    "4b-SDMLIB4",
]
FROZEN_CACHE = f"{REL}/triton/runtime-a-4b"
FROZEN_DIGEST = "98750b7b891ea53fd4e56cc3bb54027384ae47de42ce957fc60a9aa184b59e70"
DECIDED_BY = (
    "the user (2026-10-03 01:27 UTC+8, relayed by the coordinator, parent agent, Decision 2.0 program, COORDINATION "
    "01:36): ship the best candidates, then keep optimizing to same-size first place with progressive releases under "
    "the same rules (Index-first gate, integrity checks, fast-path post-checks, purge of superseded weights); applied "
    "to the 4B tier by the 4B owner (decoder M17b, ff70d16e), the only Nox-4B publisher"
)
PREPARED_BY = (
    "Decision 2.0 release engineering, the 4B owner (decoder M17b, worktree vllm-sr-dev2-dec-m17), the only Nox-4B "
    "publisher"
)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def paths(cand: str) -> dict:
    c = CANDS[cand]
    key = cand.lower()
    inputs = f"{REL}/inputs/dev2-4b-{key}"
    private = f"/data/dev2/private/release/4bif/{cand}"
    run = f"{FR}/m17/m17-4b-{cand}"
    return {
        **c,
        "cand": cand,
        "key": key,
        "index_name": f"AF-4b-{cand}-bf16",
        "run": run,
        "vendor": VENDOR,
        "in": inputs,
        "private": private,
        "mlx": f"{run}-mlx",
        "gates": f"{inputs}/gates",
        "current_gate": f"{inputs}/current/xall-4b-gate.json",
        "current_decision": f"{inputs}/current/{NAME}.decision.xall.json",
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
    assert bf16["source_model_sha256"] == p["fp32"], "the BF16 copy is not of this soup"
    receipt = load(f"{p['private']}/receipt.json")
    assert (
        receipt["model_source"]["model_sha256"] == identity
    ), "the Index run scored other weights"
    seal = load(f"{p['run']}/SEAL.json")
    s["_release"] = {
        "successor": f"{NAME} successor under the progressive-release rule (user 2026-10-03 01:27 UTC+8): "
        f"{p['what']}; stored as its v2.release.bf16_copy.",
        "gate": "successor profile with index_first: IF1 = the private Index paired bootstrap of these weights minus "
        "the current revision's, bound to both IX1 run receipts, 95% lower bound > 0; R3 = no type collapsed on the "
        "formal typed panel; IF3 = one row-level Index contamination audit of the distinct training files of wave "
        "6's members; items 2-7 as for every release. v3, human transfer, mlx-diag, tier gates and public 231 are "
        "references; overlap exposure and C1 are not run. The Index files stay in node A's private tree.",
        "scored": f"T = 1: the sealed formal run {Path(p['run']).name} (image dbe5f32b, the 4B formal library's "
        "master cache), kept at T = 1 by the 23:15 rule, collected on node F, scored on node A by m6-score.sh.",
        "storage": f"v2.release.bf16_copy of the frozen FP32 weights (identity {p['fp32'][:8]} -> {identity[:8]}; "
        f"receipt {bf16_receipt[:8]}), the copy the Index run {p['index_name']} scored.",
        "runtime": "vendor_source = the formal run's runner mirror (training/model checked equal to the scored "
        "adapter sources at build time); runtime_source = this branch's mirror "
        f"{RUNTIME[:9]}: the current revision's speed-up phase A runtime (HIP-graph replay, shared BF16 casts, the "
        "gfx942 fused Triton kernels) plus the opt-in shared-context switch (shared_ctx.py, off by default, the "
        "default path byte-identical; COORDINATION 2026-10-03 02:23), with the phase A frozen pre-warmed autotune "
        "cache for parity; automap_source = the current revision's, unchanged.",
        "card": "the product card of the current revision with this candidate's reports, an Index input built by "
        "python -m v2.release.card_index (board-served parameter counts, the audited footnote) with the 4B point "
        "from the Index run on exactly these weights and every other tier's point at its current Hub main, and "
        "assets rendered by the default v2.release.card_assets; card.speed keeps the current revision's bench "
        "receipt (same architecture, runtime and shapes).",
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-4b-xall.json",
            "sha256": sha(BASE_SPEC),
        },
        "previous": old["_release"],
    }
    s["checkpoint"] = f"{p['in']}/bf16/checkpoint"
    s["expected_identity"] = {"model_sha256": identity}
    s["bf16_copy"] = {
        "receipt": f"{p['in']}/bf16/bf16-copy.json",
        "sha256": bf16_receipt,
    }
    s["vendor_source"] = (
        f"/data/dev2/src/{p['vendor']}-src_training_decision2/src/training/decision2"
    )
    s["runtime_source"] = (
        f"/data/dev2/src/{RUNTIME}-src_training_decision2/src/training/decision2"
    )
    s["gate_receipt"] = f"{DECISIONS}/{NAME}.decision.{p['key']}.json"
    s["runtime_equivalence"] = (
        "decision2/qwen.py loads this full checkpoint with the training/model sources vendored from the formal run's "
        f"own runner mirror ({p['vendor'][:9]}), whose SHA-256 equal the scored adapter sources (checked at build "
        "time), and applies the per-item batching, BF16-backbone / FP32-head execution, raw probabilities "
        "(temperature 1; no calibration file) and answer normalization of v2.dec.infer_dec. The scored checkpoint "
        f"({p['fp32'][:8]}) stored every tensor in FP32; this package (v2.release.bf16_copy) stores its 248 Linear "
        "projection matrices in BF16 exactly as BF16 autocast rounds them and every other tensor bit for bit in "
        "FP32. The runtime is the current revision's speed-up phase A runtime plus the opt-in shared-context "
        "switch (off by default; the default path is unchanged); the Transformers remote code and the forward "
        "token budget are the current revision's. Checked on one GPU against the formal run's predictions of every "
        "scored prompt (typed-final 1,600, css15 6,547, public231 231) and of the mlx-diag diagnostic (2,275) by "
        "release.sh --parity, with a copy of the current revision's frozen pre-warmed autotune cache, and AutoModel "
        "against the native runtime on every scored prompt."
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
        "release": f"{FROZEN_CACHE} (the current revision's frozen pre-warmed cache, digest {FROZEN_DIGEST[:8]}; "
        "copied per run)",
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
        "tier": {
            "reference": "decider4b",
            "v3_share": 0.9,
            "paired": f"{p['gates']}/paired-vs-decider4b.json",
        },
        "index_first": {
            "bootstrap": f"{p['private']}/boot-full-vs-xall.json",
            "receipt": f"{p['private']}/receipt.json",
            "base_receipt": f"{p['private']}/base-receipt.json",
            "audit": f"{p['private']}/audit.json",
        },
    }
    return s


def decision(cand: str, s: dict) -> dict:
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
    assert sorted(sets) == sorted(AUDIT_SETS), sorted(sets)
    assert all(
        v["item_rows"] == 0 for v in sets.values()
    ), "an audited training file has Index item rows"
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
        "action": f"New main revision of the public repository {REPO}: {p['what']}, stored as its BF16 copy "
        f"(qwen-full, T = 1, 16,384 tokens), replaces the M17 4b-SDMLxALL weights ({CURRENT['weights_identity'][:8]}, "
        f"revision {CURRENT['revision'][:8]}) keeping its runtime, with the product card of the current revision "
        "(default generator) and this candidate's Index input; then the superseded weight blobs are purged with "
        "rewrite_history=False. The repository stays public and in the public Decision 2.0 collection.",
        "rationale": "Progressive release (user 2026-10-03 01:27 UTC+8): the best wave-6 candidate that passes the "
        "gate. IF1: the private Index paired bootstrap of these exact weights minus the current revision's (2,000 "
        "replicates, same panel and seed; values in private files only; evidence_sha256.index_first_bootstrap) has "
        "a 95% lower bound above 0. R3: types "
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
        f"{public['right_correct']} (McNemar p {public['mcnemar_exact_p']:.3f}); overlap exposure and C1 post-key "
        "not run.",
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
        "Decision 1.0 Nox-4B @cde2a68d (Qwen3.5-4B vocabulary, Apache-2.0); the IB1-r3 / IB2 / IB3-r2 / IB4 "
        "phase-1 training rows are licence-clean (release-safe data records) and the multilingual copies are "
        "released training rows; the package keeps the Apache-2.0 LICENSE (product card: no NOTICE, attributions "
        "or evaluation/ files).",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": [
            (
                "a two-seed soup of one arm (uniform; the arm factory's half-LR training run); its training rows "
                if p["members"] == 1
                else f"an FP32 weight average of {p['members']} arm soups "
                f"({'4b-LHS17UP listed twice' if cand == 'XALLU2' else 'uniform'}; no new training run); the "
                "training rows of its "
            )
            + "members include the IB1-r3 / IB2 / IB3-r2 / IB4 phase-1 families matched to Index benchmark families "
            "(HoVer, When2Call, iSarcasmEval, GSM8K and others) and the BPoMP format",
            *(
                f"Index contamination audit of the {n} training rows ({v['training_lines']} lines): "
                f"{v['item_rows']} item rows and {v['duplicate_rows']} familiar-text rows"
                for n, v in sorted(sets.items())
            ),
            "one audit covers the ten distinct training files of every wave-6 candidate's members; 4b-LHS17UP "
            "trains on the 4b-LHS17SD file and 4b-LHS17IB4-lrh on the 4b-LHS17IB4 file (same rows, other row weights "
            "or learning rate), and the factory's extra seeds use their arm's locked file",
            "card.speed is the current revision's (phase A) bench receipt (same architecture, runtime and shapes)",
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
    p = paths(args.cand)
    for path, value in (
        (p["current_gate"], CURRENT["gate_sha256"]),
        (p["current_decision"], CURRENT["decision_sha256"]),
    ):
        if sha(path) != value:
            raise SystemExit(f"{path} is not the recorded file")
    s = spec(args.cand)
    spec_text = json.dumps(s, indent=2, ensure_ascii=False) + "\n"
    d = decision(args.cand, s)
    d["decided_utc"] = args.decided_utc
    decision_text = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    outputs = {
        f"dev2-4b-{p['key']}.json": (spec_text, SPECS),
        f"{NAME}.decision.{p['key']}.json": (decision_text, RECORD),
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
