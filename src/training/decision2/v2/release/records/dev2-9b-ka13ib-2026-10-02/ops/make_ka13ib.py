"""Release spec and decision of the Decision-2.0-Lux-9B successor K-a13IB (9B Milestone 9, the Index path).

Coordinator decision 2026-10-02 02:05 UTC+8 (the user did not answer; the recommended rule was adopted): for
frontier-targeted finalists item 1 becomes item 1' -- v3 not significantly below the current revision (paired 95% CI
upper bound > 0) and the private Index delta against it significantly positive (paired bootstrap 95% lower bound > 0);
items 2-8 unchanged. K-a13IB (the M9 stage-3 soup; records v2/9b/records/lux9b-m9-stage3-result-2026-10-01.md and
lux9b-m9-ix-ka13ib-receipt-2026-10-01.md) qualifies on 1' and 2-7 and ships as the next main of
llm-semantic-router/Decision-2.0-Lux-9B after item 8 (the C1 post-key guard), at T = 1, stored as its
v2.release.bf16_copy and served by the released runtime (forward token budget, Transformers remote code).

The spec derives from the product-card spec of the current revision (specs/dev2-9b-product.json, card round 2,
revision 586af779): roster, peers, runtime, remote code, licence and the product card carry over; the weights, the
scored run, the gate evidence and the card's Index input and assets are K-a13IB's. The Index input is the round-2
file with the 9B point replaced by an independent kit run on exactly these weights (K-a13IB-bf16) and the footnote of
the coordinator decision; it and the rendered assets stay in node A's private tree, pinned by SHA-256.

Run on node A from the exact mirror holding this file (host python3; every file it hashes is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> draft --out DIR
  PYTHONPATH=... python3 <this file> final --c1-summary <node A SUMMARY.json> --out DIR
It writes DIR/dev2-9b-ka13ib[.draft].json and DIR/Decision-2.0-Lux-9B.decision.ka13ib[.draft].json; the committed
copies live in v2/release/specs and this record directory, and --check compares them byte for byte.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from v2.release import gate

ROOT = Path(__file__).resolve().parents[4]
RECORD = Path(__file__).resolve().parents[1]
BASE_SPEC = ROOT / "v2/release/specs/dev2-9b-product.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Lux-9B"
REPO = f"llm-semantic-router/{NAME}"
REL = "/data/dev2/runs/release"
IN = f"{REL}/inputs/dev2-9b-ka13ib"
RUN = f"{REL}/dev2-9b-ka13ib-t1-derived"
MLX = f"{REL}/dev2-9b-ka13ib-t1-derived-mlx"
FORMAL = "/data/dev2/runs/9b/formal-m9/K-a13IB-16k"
CURRENT_RUN = f"{REL}/dev2-8b-t1-derived"
PRIVATE = "/data/dev2/private/release/ka13ib"
DECISIONS = f"{REL}/decisions"
VENDOR = "/data/dev2/src/787abdc54946ccdb05e52cc5c34ecb619dced237-src_training_decision2/src/training/decision2"
FP32_IDENTITY = "4701ba41c70636b5e215cf1f296bc920a9ee39960d4d28a2338e20b2b81e0d91"
CURRENT = {
    "revision": "586af77916ee508320421bda6c22f7f0305a7279",
    "gate": f"{IN}/current/card2-9b-gate.json",
    "decision": f"{IN}/current/{NAME}.decision.card2.json",
    "gate_sha256": "6343afa6cb7804002997616edbab03e1cff7eebd3c7dd333100a60f8b4e64476",
    "decision_sha256": "5e9f13ae73cd52d24155885e6666cd6aee2afa466b1e1906a634e1143e9a0a6e",
    "manifest_sha256": "cd710f2656e50cbc62c245c93c0b8700e28c91d533b321c743b4e7abb54f166c",
    "weights_identity": "b1ed5a71038b474902d2a6dfebeee0c6f7ce901097553b4621f6f0a23fb6da98",
}
EXPOSURE = [
    "/data/dev2/runs/9b/m6/exposure/x60.json",
    "/data/dev2/runs/9b/m9/exposure/ib1-ib2-train.json",
]
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: decision 2026-10-02 02:05 UTC+8 (the Index path for "
    "frontier-targeted finalists, items 1'-8; the user did not answer and the recommended rule was adopted; the user "
    "may override), shipping K-a13IB as the Decision-2.0-Lux-9B successor after item 8"
)
PREPARED_BY = "Decision 2.0 9B worker (worktree vllm-sr-dev2-9b-m9)"
DECIDED_UTC = "2026-10-01T18:05:00Z"


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def spec(summary_path: str | None) -> dict:
    old = load(BASE_SPEC)
    assert (
        old["repo_id"] == REPO
        and old["expected_identity"]["model_sha256"] == CURRENT["weights_identity"]
    )
    s = copy.deepcopy(old)
    bf16 = load(f"{IN}/bf16/bf16-copy.json")
    identity = bf16["model_sha256"]
    assert bf16["source_model_sha256"] == FP32_IDENTITY
    seal = load(f"{RUN}/SEAL.json")
    s["_release"] = {
        "successor": "Decision-2.0-Lux-9B successor K-a13IB (9B Milestone 9 stage-3 soup: the K-a13 recipe with the "
        "IB1-r3 / IB2 rows at matched tokens; [KIB soup, Lux 1.0, Lux 1.0] at alpha 1/3) under the Index path of the "
        "coordinator decision 2026-10-02 02:05 UTC+8: item 1' (v3 not significantly below the current revision and "
        "the private Index delta against it significantly positive) and items 2-8 of the successor rule.",
        "gate": f"successor profile with index_path: R1' and R2-R7 against the current revision {CURRENT['revision'][:8]} "
        "(its scored run dev2-8b-t1-derived), R5 = the tier gates (adopted Lux 1.0, v3 >= 0.9 x Nimble v2, human "
        "transfer vs Nimble v2), R6 = both exposure receipts (x60 and the IB1 / IB2 rows), R8 = the JevArena-C1 v1.2 "
        "post-key guard. The Index bootstrap file stays in node A's private tree (SHA-256 bound by the decision).",
        "scored": "T = 1: the sealed CAL698 formal run formal-m9/K-a13IB-16k (node A GPU6, image f83b1d10) returned "
        "to T = 1 offline with 0 answer changes on every panel and mlx-diag, then adopted, sealed and paired on node A "
        "(ops/prep.sh derive / paired).",
        "storage": f"v2.release.bf16_copy of the frozen FP32 soup (identity {FP32_IDENTITY[:8]} -> {identity[:8]}; "
        f"receipt {sha(f'{IN}/bf16/bf16-copy.json')[:8]}).",
        "runtime": "vendor_source = the formal run's runner mirror 787abdc54 (training/model checked equal to the "
        "scored adapter sources at build time); runtime_source = automap_source = 99432d1a7, the released runtime "
        "(BF16-resident, Transformers remote code, forward token budget), unchanged from the current revision.",
        "card": "the product card of the current revision (card round 2) with K-a13IB's reports; card.index = the "
        "round-2 Index input with the 9B point from an independent kit run on exactly these weights (K-a13IB-bf16) "
        "and the footnote of the coordinator decision; card.assets rendered by v2.release.card_assets in the "
        "round-2 environment (Python 3.12.13, matplotlib 3.11.2, Pillow 12.3.0, the same Inter fonts and logo); "
        "card.speed keeps the 400-request bench receipt of this runtime (same architecture, runtime and shapes).",
        "c1": (
            "final: post-key C1 summary bound as R8"
            if summary_path
            else "draft: item 8 pending"
        ),
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-9b-product.json",
            "sha256": sha(BASE_SPEC),
        },
        "previous": old["_release"],
    }
    s["checkpoint"] = f"{IN}/bf16/checkpoint"
    s["expected_identity"] = {"model_sha256": identity}
    s["bf16_copy"] = {
        "receipt": f"{IN}/bf16/bf16-copy.json",
        "sha256": sha(f"{IN}/bf16/bf16-copy.json"),
    }
    s["vendor_source"] = VENDOR
    s["gate_receipt"] = (
        f"{DECISIONS}/{NAME}.decision.ka13ib{'' if summary_path else '.draft'}.json"
    )
    s["runtime_equivalence"] = (
        "decision2/qwen.py loads this full checkpoint with the training/model sources vendored from the scored run's "
        "own runner mirror (787abdc54), whose SHA-256 equal the scored adapter sources (checked at build time), and "
        "applies the per-item batching, BF16-backbone / FP32-head execution, raw probabilities (temperature 1; no "
        "calibration file) and answer normalization of v2.dec.infer_dec, which for this non-residual checkpoint "
        f"wraps the same DecisionModel without extra readouts. The scored checkpoint ({FP32_IDENTITY[:8]}) stored "
        "every tensor in FP32; this package (v2.release.bf16_copy) stores its 248 Linear projection matrices in BF16 "
        "exactly as BF16 autocast rounds them and every other tensor bit for bit in FP32, and the runtime holds the "
        "backbone's BF16-exact Linear weights in BF16, the values BF16 autocast multiplies with. The runtime, the "
        "Transformers remote code (AutoModel with trust_remote_code) and the forward token budget are the current "
        "revision's. Checked on one GPU of the scoring node against the T = 1 predictions derived exactly from the "
        "sealed CAL698 predictions of every scored prompt (typed-final 1,600, css15 6,547, public231 231) and of the "
        "mlx-diag diagnostic (2,275) by release.sh --parity, with a copy of the persisted Triton autotune cache of the "
        "formal run, and AutoModel against the native runtime on every scored prompt."
    )
    s["scored"] = {
        "label": "post-key same-panel run K-a13IB-16k at T = 1 (predictions derived from the sealed CAL698 run by "
        "undoing its temperatures; adopted and sealed)",
        "report_sha256": sha(f"{RUN}/REPORT.json"),
        "seal_sha256": sha(f"{RUN}/SEAL.json"),
        "predictions_sha256": {
            **{
                p: seal["panels"][p]["predictions_sha256"]
                for p in ("typed-final", "css15", "public231")
            },
            "mlx-diag": sha(f"{MLX}/output/mlx-diag.predictions.jsonl"),
        },
        "paired_sha256": sha(f"{RUN}/PAIRED-vs-adopted-1.0.json"),
        "native_manifest": f"{FORMAL}/output/typed-final.predictions.jsonl.manifest.json",
    }
    card = s["card"]
    card["paired"] = f"{RUN}/PAIRED-vs-adopted-1.0.json"
    card["paired_peers"] = {"nimble2": f"{RUN}/PAIRED-vs-nimble2.json"}
    reports = card["reports"]
    assert reports[0]["role"] == "candidate" and reports[0]["label"] == NAME
    reports[0]["report"] = f"{RUN}/REPORT.json"
    reports[0]["mlx"] = f"{MLX}/mlx-diag.score.json"
    card["index"] = {
        "path": f"{PRIVATE}/decision-index-card.json",
        "sha256": sha(f"{PRIVATE}/decision-index-card.json"),
    }
    card["assets"] = {
        "dir": f"{PRIVATE}/9b",
        "receipt_sha256": sha(f"{PRIVATE}/9b/card-assets.json"),
    }
    s["gate_profile"] = {
        "name": "successor",
        "run": RUN,
        "current": {
            "revision": CURRENT["revision"],
            "gate": CURRENT["gate"],
            "decision": CURRENT["decision"],
            "run": CURRENT_RUN,
            "mlx_predictions": f"{IN}/current-mlx/output/mlx-diag.predictions.jsonl",
        },
        "paired": f"{IN}/gates/paired-vs-dev2-9b.json",
        "types": f"{IN}/gates/types.json",
        "mlx_paired": f"{IN}/gates/mlx-paired-vs-dev2-9b.json",
        "exposure": list(EXPOSURE),
        "public231": f"{IN}/gates/public231-vs-dev2-9b.json",
        "tier": {
            "reference": "nimble2",
            "v3_share": 0.9,
            "paired": f"{IN}/gates/paired-vs-nimble2.json",
        },
        "index_path": {"bootstrap": f"{PRIVATE}/paired-boot-vs-dev20-9b.json"},
    }
    if summary_path:
        s["gate_profile"]["c1_postkey"] = summary_path
    s["frozen_autotune_cache"] = {
        "formal": "/data/dev2/runs/9b/formal-m9/triton-cache (the formal run's persisted cache; the typed-final, "
        "css15, public231 and mlx-diag runs shared it)",
    }
    return s


def decision(s: dict, summary_path: str | None) -> dict:
    final = summary_path is not None
    profile = gate.gate_profile(s)
    items = gate.successor_items(s, profile)
    failed = [k for k, v in items.items() if not v["passed"]]
    if failed:
        raise SystemExit(f"successor items fail: {failed}")
    paired = load(profile["paired"])
    low, high = gate._low_high(paired["ci95"])
    h = paired["axis_ci95"]["H"]["delta"]
    v3 = load(f"{RUN}/REPORT.json")["v3"]["score"]
    current_v3 = load(f"{CURRENT_RUN}/REPORT.json")["v3"]["score"]
    public = load(profile["public231"])
    mlx = load(profile["mlx_paired"])["overall"]
    own = load(s["card"]["paired"])
    peer = load(profile["tier"]["paired"])
    item8 = ""
    if final:
        summary = load(summary_path)
        rule = summary["item8"]
        if rule["verdict"] != "PASS" or summary["role"] != "successor":
            raise SystemExit("item 8 did not pass: no release")
        item8 = (
            f"Item 8 (JevArena-C1 v1.2 post-key guard vs the registered 9B baseline): C1 {summary['c1']:.2f}, "
            f"{signed(rule['delta'])} [{signed(rule['ci95'][0])}, {signed(rule['ci95'][1])}], {rule['verdict']}; "
            "bound by evidence_sha256.c1_postkey. "
        )
    value = {
        "schema": gate.DECISION_SCHEMA,
        "status": "final" if final else "draft",
        "decision": "release",
        "model_name": NAME,
        "repo_id": REPO,
        "name_basis": s.get("name_basis"),
        "name_base_model": s.get("name_base_model"),
        "identity": s["expected_identity"],
        "report_sha256": s["scored"]["report_sha256"],
        "paired_sha256": sha(s["card"]["paired"]),
        "gate_profile": "successor",
        "current_revision": CURRENT["revision"],
        "evidence_sha256": gate.evidence_sha256(profile),
        "calibration": "none (temperature 1; every Decision 2.0 model keeps T = 1): the CAL698 per-type "
        f"temperatures of the formal run (calibration {sha('/data/dev2/runs/9b/formal-m9/K-a13IB-cal/calibration.json')[:8]}) "
        "are not shipped; undoing them changes 0 answers on every scored panel and mlx-diag",
        "action": f"New main revision of the private repository {REPO}: the 9B M9 successor K-a13IB (the K-a13 recipe "
        "with the IB1-r3 / IB2 rows; [KIB soup, Lux 1.0, Lux 1.0] at alpha 1/3; qwen-full, BF16 storage, T = 1, "
        f"16,384 tokens) replaces the K-a13 weights ({CURRENT['weights_identity'][:8]}, revision "
        f"{CURRENT['revision'][:8]}) with the product card of card round 2; then the superseded weight blobs are "
        "purged with rewrite_history=False (hf_headroom.sh first) and the C1 baseline registry takes this "
        "revision's post-key run. The repository stays private and in the private collection.",
        "rationale": "Index path (coordinator decision 2026-10-02 02:05 UTC+8). Item 1': post-key v3 "
        f"{v3:.3f} vs {current_v3:.3f}, {signed(paired['point']['delta']['score'])} [{signed(low)}, {signed(high)}] "
        "(not significantly below); private Index paired bootstrap 95% lower bound > 0 (values in private files "
        f"only; evidence_sha256.index_path). Items 2-7: human transfer {signed(h['low'], 3)} to "
        f"{signed(h['high'], 3)} (not below); types OK; card-eligible mlx-diag {signed(mlx['delta'], 4)} "
        f"[{signed(mlx['ci95']['low'], 4)}, {signed(mlx['ci95']['high'], 4)}]; tier gates (adopted Lux 1.0 "
        f"{signed(own['point']['delta']['score'])} [{signed(gate._low_high(own['ci95'])[0])}, "
        f"{signed(gate._low_high(own['ci95'])[1])}]; v3 >= 0.9 x Nimble v2; human transfer vs Nimble v2 upper "
        f"{signed(peer['axis_ci95']['H']['delta']['high'], 3)}); 0 exposed training groups in both exposure receipts; "
        f"public 231 {public['left_correct']} vs {public['right_correct']} (McNemar p {public['mcnemar_exact_p']:.3f}). "
        + (
            item8
            or "Draft for the frozen package that item 8 scores; the final decision adds the C1 summary."
        ),
        "approved_package": {
            "identity": s["expected_identity"]["model_sha256"],
            "fp32_identity": FP32_IDENTITY,
            "profile": "qwen-full (BF16 storage of the Linear projections)",
            "temperature": 1,
            "max_input_tokens": 16384,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": "apache-2.0: own weights; Decision 1.0 Lux-9B @bd45a30a (direct weight origin, "
        "interpolated) and the Qwen/Qwen3.5-9B @c2022362 text backbone and tokenizer are Apache-2.0; the package "
        "keeps the Apache-2.0 LICENSE (product card: no NOTICE, attributions or evaluation/ files).",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": [
            "item 1 of the successor rule (v3 significantly above the current revision) is not met; the Index path "
            "replaces it with item 1' for frontier-targeted finalists",
            "the training rows include the IB1-r3 / IB2 families matched to Index benchmark families (HoVer, "
            "When2Call, iSarcasmEval, GSM8K) and the BPoMP format; the internal records report the transfer-only "
            "Index delta without those benchmarks (private values)",
            "Index contamination audit of the training rows: 0 item duplicates; familiar text in ANLI, BANKING77 "
            "and HoVer, as in the current revision's own audit",
            "card.speed is the current revision's bench receipt (same architecture, runtime and shapes)",
        ],
        "prepared_by": PREPARED_BY,
    }
    if final:
        value["decided_by"] = DECIDED_BY
        value["decided_utc"] = DECIDED_UTC
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("draft", "final"))
    ap.add_argument("--c1-summary")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--check", action="store_true", help="compare with the committed copies"
    )
    args = ap.parse_args()
    final = args.mode == "final"
    if final != bool(args.c1_summary):
        raise SystemExit("final needs --c1-summary; draft takes none")
    for path, value in (
        (CURRENT["gate"], CURRENT["gate_sha256"]),
        (CURRENT["decision"], CURRENT["decision_sha256"]),
    ):
        if sha(path) != value:
            raise SystemExit(f"{path} is not the current revision's file")
    suffix = "" if final else ".draft"
    s = spec(args.c1_summary)
    spec_text = json.dumps(s, indent=2, ensure_ascii=False) + "\n"
    d = decision(s, args.c1_summary)
    decision_text = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    outputs = {
        f"dev2-9b-ka13ib{suffix}.json": (spec_text, SPECS),
        f"{NAME}.decision.ka13ib{suffix}.json": (decision_text, RECORD),
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
