"""Release spec and decision of Decision-2.0-Nox-4B M15 4b-LHA10SDML (decoder M17, the single Nox-4B publisher).

User decision relayed by the coordinator 2026-10-02 15:45 UTC+8: release 4b-LHA10SDML to Decision-2.0-Nox-4B now,
once, over the current revision b285e7a1 (M17 4b-LHS17SD), as an override of the Index-first significance rule:
"the Index is equal within noise, transfer is significantly better". IF1's lower bound is not above 0; the gate
passes IF1 on the user override record (``user-override.json`` in this record, ``gate.USER_OVERRIDE_SCHEMA``),
which names these weights and the superseded revision and is bound by the decision. Integrity checks as for every
Index-first release: R3 (no type collapsed on the formal typed panel; "if a type collapses, stop and report"), IF3
(the row-level Index contamination audit of the M15 TRAIN) and the release items 2-7. v3, human transfer, mlx-diag,
the tier gates and public 231 are references; overlap exposure and C1 are not run.

The spec derives from the current revision's spec (specs/dev2-4b-4bif-S17.json, ``main`` b285e7a1): roster, peers,
runtime, remote code, licence and the product card carry over; the weights, the scored run, the gate evidence and the
card's Index input and assets (the default generator) are the candidate's.

Run on node A from the exact mirror holding this file (host python3; every hashed file is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> --decided-utc UTC --out DIR [--check]
It writes DIR/dev2-4b-sdml.json and DIR/Decision-2.0-Nox-4B.decision.sdml.json; the committed copies live in
v2/release/specs and this record directory, and --check compares them byte for byte.
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
BASE_SPEC = ROOT / "v2/release/specs/dev2-4b-4bif-S17.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Nox-4B"
REPO = f"llm-semantic-router/{NAME}"
REL = "/data/dev2/runs/release"
FR = "/data/dev2/runs/dec/formal"
DECISIONS = f"{REL}/decisions"
CURRENT_RUN = f"{FR}/m17/m17-4b-LHS17SD"
CURRENT_MLX = f"{FR}/m17/m17-4b-LHS17SD-mlx"
CURRENT = {
    "revision": "b285e7a17791a06058817680592484868dde9545",
    "gate_sha256": "54a8c2aba52aee4d6b035e57c22ed0c7c025a75d469a73c0036952710ea5d70c",
    "decision_sha256": "18d60dac94fe64ca06042d4dfc1d5dbda881fd5e4fbb699085db09e57a9ece63",
    "manifest_sha256": "35eba1817fc7f45183deba0cad0b12c0cad84177dd4937e4a6fd5c83019b0469",
    "weights_identity": "74ec8b2f838df1b8c26e94f83845f389e963da3a4a0576111a71cf8a617297d7",
}
CURRENT_RECORD = ROOT / "v2/release/records/dev2-4b-indexfirst-2026-10-02"
CAND = "SDML"
C = {
    "index_name": "IS-4b-LHA10SDML-bf16",
    "run": f"{FR}/m17/m17-4b-LHA10SDML",
    "fp32": "1b51567523426b896ae50afeefca4f133c2e3c220c349ebfb9b9c630600dd19d",
    "vendor": "8f6a04a2c7cd96775f203ce33c780d7a277488ff",
    "what": "the decoder M15 arm 4b-LHA10SDML (the released LH recipe, rank-128 LoRA on Qwen/Qwen3.5-4B-Base "
    "@1001bb4d, merged, scoring head, on the released 4B mixture plus 10% IB1-r3 / IB2 breadth rows and copies of "
    "released non-English rows, with typed-row self-distillation from LH; soup of two seeds)",
    "train": "fef6b036f33de6756dab083fd21ab462ec2975cfa63120f2452d3c9145d33dd4",
    "audit_set": "4b-LHA10SDML",
}
QUOTE = "the Index is equal within noise, transfer is significantly better"
DECIDED_BY = (
    "the user (decision relayed by the coordinator, parent agent, Decision 2.0 program, 2026-10-02 15:45 UTC+8): "
    "release 4b-LHA10SDML to Decision-2.0-Nox-4B now, once, as an override of the Index-first significance rule "
    f'("{QUOTE}"); the integrity checks stay; applied to the 4B tier by decoder M17, the single Nox-4B publisher'
)
PREPARED_BY = "Decision 2.0 release engineering, decoder M17 (worktree vllm-sr-dev2-dec-m17), the single Nox-4B publisher"


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def paths(cand: str = CAND) -> dict:
    assert cand == CAND, cand
    inputs = f"{REL}/inputs/dev2-4b-sdml"
    private = "/data/dev2/private/release/4bif/SDML"
    return {
        **C,
        "in": inputs,
        "private": private,
        "mlx": f"{C['run']}-mlx",
        "gates": f"{inputs}/gates",
        "override": f"{inputs}/user-override.json",
        "current_gate": f"{inputs}/current/4bif-S17-4b-gate.json",
        "current_decision": f"{inputs}/current/{NAME}.decision.4bif-S17.json",
    }


def spec() -> dict:
    p = paths()
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
    override = load(p["override"])
    assert (
        override["identity"] == identity
        and override["current_revision"] == CURRENT["revision"]
    )
    seal = load(f"{p['run']}/SEAL.json")
    formal_cache = f"{p['in']}/formal-cache"
    mlx_cache = f"{p['in']}/mlx-cache"
    s["_release"] = {
        "successor": f"{NAME} successor by user decision: {p['what']}; stored as its v2.release.bf16_copy.",
        "gate": "successor profile with index_first and a user override of IF1 (user decision 2026-10-02 15:45 "
        f'UTC+8: "{QUOTE}"): IF1 = the private Index paired bootstrap of these weights minus the current '
        "revision's, bound to both IX1 run receipts, its lower-bound verdict printed and passed on the user override "
        "record; R3 = no type collapsed on the formal typed panel; IF3 = the row-level Index contamination audit of "
        "the training file; items 2-7 as for every release. v3, human transfer, mlx-diag, tier gates and public 231 "
        "are references; overlap exposure and C1 are not run. The Index files stay in node A's private tree.",
        "scored": f"T = 1: the sealed formal run {Path(p['run']).name} (image dbe5f32b, the 4B formal library's "
        "master cache), kept at T = 1 by the 23:15 rule, scored on node A by m6-score.sh.",
        "storage": f"v2.release.bf16_copy of the frozen FP32 weights (identity {p['fp32'][:8]} -> {identity[:8]}; "
        f"receipt {bf16_receipt[:8]}), the copy the Index run {p['index_name']} scored.",
        "runtime": "vendor_source = the formal run's runner mirror (training/model checked equal to the scored "
        "adapter sources at build time); runtime_source = automap_source = the current revision's (BF16-resident, "
        "Transformers remote code, forward token budget), unchanged.",
        "card": "the product card of the current revision with this candidate's reports, an Index input built by "
        "python -m v2.release.card_index (board-served parameter counts, the audited footnote) with the 4B point "
        "from the Index run on exactly these weights, and assets rendered by the default v2.release.card_assets; "
        "card.speed keeps the current revision's bench receipt (same architecture, runtime and shapes).",
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-4b-4bif-S17.json",
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
    s["gate_receipt"] = f"{DECISIONS}/{NAME}.decision.sdml.json"
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
        "tier": {
            "reference": "decider4b",
            "v3_share": 0.9,
            "paired": f"{p['gates']}/paired-vs-decider4b.json",
        },
        "index_first": {
            "bootstrap": f"{p['private']}/boot-full-vs-s17.json",
            "receipt": f"{p['private']}/receipt.json",
            "base_receipt": f"{p['private']}/base-receipt.json",
            "audit": f"{p['private']}/audit.json",
            "user_override": p["override"],
        },
    }
    return s


def decision(s: dict) -> dict:
    p = paths()
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
    mine = sets[p["audit_set"]]
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
        f"(qwen-full, T = 1, 16,384 tokens), replaces the M17 4b-LHS17SD weights ({CURRENT['weights_identity'][:8]}, "
        f"revision {CURRENT['revision'][:8]}) with the product card of the current revision (default generator) and "
        "this candidate's Index input; then the superseded weight blobs are purged with rewrite_history=False "
        "(hf_headroom.sh first). The repository stays private and in the private collection.",
        "rationale": "User decision 2026-10-02 15:45 UTC+8, an override of the Index-first significance rule: "
        f'"{QUOTE}". IF1: the private Index paired bootstrap of these exact weights minus the current revision\'s '
        "(2,000 replicates, same panel and seed; values in private files only; "
        "evidence_sha256.index_first_bootstrap) has a 95% lower bound not above 0; it passes on the user override "
        "record (evidence_sha256.index_first_user_override). R3: types "
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
        "Decision 1.0 Nox-4B @cde2a68d (Qwen3.5-4B vocabulary, Apache-2.0); the IB1-r3 / IB2 training rows are "
        "licence-clean (release-safe data records) and the multilingual copies are released training rows; the "
        "package keeps the Apache-2.0 LICENSE (product card: no NOTICE, attributions or evaluation/ files).",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": [
            "released by user decision over the Index-first significance rule: the private Index delta vs the "
            "current revision is not significant (equal within noise); the transfer-only Index delta (without the "
            "IB-matched benchmarks) is significantly positive (private values)",
            "the training rows include the IB1-r3 / IB2 families matched to Index benchmark families (HoVer, "
            "When2Call, iSarcasmEval, GSM8K) and the BPoMP format",
            f"Index contamination audit of the training rows ({p['train'][:8]}, {mine['training_lines']} lines): "
            f"{mine['item_rows']} item rows and {mine['duplicate_rows']} familiar-text rows",
            "card.speed is the current revision's bench receipt (same architecture, runtime and shapes)",
        ],
        "prepared_by": PREPARED_BY,
        "decided_by": DECIDED_BY,
        "decided_utc": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decided-utc", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--check", action="store_true", help="compare with the committed copies"
    )
    args = ap.parse_args()
    p = paths()
    for path, value in (
        (p["current_gate"], CURRENT["gate_sha256"]),
        (p["current_decision"], CURRENT["decision_sha256"]),
        (p["override"], sha(RECORD / "user-override.json")),
    ):
        if sha(path) != value:
            raise SystemExit(f"{path} is not the recorded file")
    s = spec()
    spec_text = json.dumps(s, indent=2, ensure_ascii=False) + "\n"
    d = decision(s)
    d["decided_utc"] = args.decided_utc
    decision_text = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    outputs = {
        "dev2-4b-sdml.json": (spec_text, SPECS),
        f"{NAME}.decision.sdml.json": (decision_text, RECORD),
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
