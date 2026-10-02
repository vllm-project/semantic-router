"""Release spec and decision of the Decision-2.0-Lux-9B successor K-a12IB (Index sweep, Index-first rule).

User decision 2026-10-02 09:55 UTC+8 (COORDINATION): the only quality gate is the frozen candidate's private Jev
Decision Index delta vs the current release (paired bootstrap, 95% lower bound > 0); integrity checks: exact package
parity, Hub trust_remote_code under Transformers 5.17 / 5.18, a row-level Index contamination audit of the training
file and no decision type collapsed; JevArena v3, human transfer, mlx-diag, public 231 and C1 are references. The
Index sweep (COORDINATION 10:00) selects per tier the candidate with the largest lower bound. K-a12IB (9B M9 stage 4:
[KIB soup, Lux 1.0] at alpha 1/2; v2/9b/records/lux9b-m9-stage4-result-2026-10-01.md) ships as the next main of
llm-semantic-router/Decision-2.0-Lux-9B at T = 1, stored as its v2.release.bf16_copy and served by the released
runtime (forward token budget, Transformers remote code).

The spec derives from the current revision's spec (specs/dev2-9b-ka13ib.json, K-a13IB, revision 259a4550): roster,
peers, runtime, remote code, licence and the product card carry over; the weights, the scored run, the gate evidence
and the card's Index input and assets are K-a12IB's. The Index evidence is the IX1 run of exactly these weights
(IS-K-a12IB-bf16) minus the current revision's (K-a13IB-bf16); the bootstrap, both run receipts and the audit are
copied into node A's private tree and bound by SHA-256; no Index value is written here.

Run on node A from the exact mirror holding this file (host python3; every file it hashes is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> --out DIR [--check]
It writes DIR/dev2-9b-ka12ib.json and DIR/Decision-2.0-Lux-9B.decision.ka12ib.json; the committed copies live in
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
BASE_SPEC = ROOT / "v2/release/specs/dev2-9b-ka13ib.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Lux-9B"
REPO = f"llm-semantic-router/{NAME}"
REL = "/data/dev2/runs/release"
IN = f"{REL}/inputs/dev2-9b-ka12ib"
RUN = f"{REL}/dev2-9b-ka12ib-t1-derived"
MLX = f"{REL}/dev2-9b-ka12ib-t1-derived-mlx"
FORMAL = "/data/dev2/runs/9b/formal-m9/K-a12IB-16k"
CAL = "/data/dev2/runs/9b/formal-m9/K-a12IB-cal/calibration.json"
CURRENT_RUN = f"{REL}/dev2-9b-ka13ib-t1-derived"
CURRENT_MLX = f"{REL}/dev2-9b-ka13ib-t1-derived-mlx"
PRIVATE = "/data/dev2/private/release/ka12ib"
DECISIONS = f"{REL}/decisions"
FP32_IDENTITY = "68fed4cb24ecefb9499f4b532a4a34df621ab89138bc03f2086965f5dc0c35d0"
FORMAL_MIRROR = "0e5ec18b7dd67007863f990fee1771c7a4379514"
VENDOR = f"/data/dev2/src/{FORMAL_MIRROR}-src_training_decision2/src/training/decision2"
CURRENT = {
    "revision": "259a45502bfca2f585a59ef99079533106e0b136",
    "gate": f"{IN}/current/ka13ib-gate.json",
    "decision": f"{IN}/current/{NAME}.decision.ka13ib.json",
    "gate_sha256": "f3d79e248560812b7e83763190bc3da0732affdf9986b9e729a089221f626ee1",
    "decision_sha256": "99621f30280fed67579ca2e9d85b01e916485b867fb8b72eac26d035f826a56e",
    "manifest_sha256": "01d642a1195edb60ca18b7846bc76e8bc0df4fe50fec254b39e2761fc4d26ab5",
    "weights_identity": "b9a65d601bb6780c2c07998aecbfab322ede459c26fcb8ed262281ad08f909fb",
}
EXPOSURE = [
    "/data/dev2/runs/9b/m6/exposure/x60.json",
    "/data/dev2/runs/9b/m9/exposure/ib1-ib2-train.json",
]
INDEX = {
    "bootstrap": f"{PRIVATE}/paired-boot-full-vs-current.json",
    "receipt": f"{PRIVATE}/receipt-K-a12IB-bf16.json",
    "base_receipt": f"{PRIVATE}/receipt-K-a13IB-bf16.json",
    "audit": f"{PRIVATE}/audit-kib.json",
}
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's Index-first release rule (2026-10-02 09:55 UTC+8) "
    "and the Index-sweep assignment (COORDINATION 10:00): per tier, the frozen candidate with the largest Index-gain "
    "lower bound > 0 ships after the integrity checks"
)
PREPARED_BY = "Decision 2.0 Index-sweep worker (worktree vllm-sr-dev2-index-sweep)"
ORIGIN_SHARES = (
    "one third of that average plus two thirds of the original",
    "one half of that average plus one half of the original",
)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def spec() -> dict:
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
        "successor": "Decision-2.0-Lux-9B successor K-a12IB (9B Milestone 9 stage-4 point: [KIB soup, Lux 1.0] at "
        "alpha 1/2; the KIB soup is the K-a13 recipe with the IB1-r3 / IB2 rows at matched tokens) under the user's "
        "Index-first release rule (2026-10-02 09:55 UTC+8) and the Index sweep (COORDINATION 10:00).",
        "gate": f"successor profile with index_first: IF1 the private Index paired bootstrap of these weights minus the "
        f"current revision {CURRENT['revision'][:8]} (both IX1 run receipts bound), R3 no type collapsed on the scored "
        "run, IF3 the row-level Index audit of the training file; v3, human transfer, mlx-diag, the tier gates (Nimble "
        "v2), overlap exposure and public 231 are references. The Index files stay in node A's private tree "
        "(SHA-256 bound by the decision).",
        "scored": "T = 1: the sealed CAL698 formal run formal-m9/K-a12IB-16k (node A GPU3, image f83b1d10) returned "
        "to T = 1 offline with 0 answer changes on every panel and mlx-diag, then adopted, sealed and paired on node A "
        "(ops/prep.sh derive / paired).",
        "storage": f"v2.release.bf16_copy of the frozen FP32 soup (identity {FP32_IDENTITY[:8]} -> {identity[:8]}; "
        f"receipt {sha(f'{IN}/bf16/bf16-copy.json')[:8]}), the copy the Index sweep scored (IS-K-a12IB-bf16).",
        "runtime": "vendor_source = the formal run's runner mirror (training/model checked equal to the scored "
        "adapter sources at build time); runtime_source = automap_source = the current revision's (the released "
        "runtime: BF16-resident, Transformers remote code, forward token budget).",
        "card": "the current revision's product card (default generator: banner concept A, round-3 Index "
        "conventions) with K-a12IB's reports: card.index = the current revision's Index input with the 9B point "
        "replaced by the independent kit run on exactly these weights (IS-K-a12IB-bf16); card.assets rendered by "
        "v2.release.card_assets; card.speed keeps the bench receipt of this runtime (same architecture, runtime and "
        "shapes).",
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-9b-ka13ib.json",
            "sha256": sha(BASE_SPEC),
        },
        "previous": old["_release"],
    }
    assert old["origin"]["summary"].count(ORIGIN_SHARES[0]) == 1
    s["origin"] = {
        **old["origin"],
        "summary": old["origin"]["summary"].replace(*ORIGIN_SHARES),
    }
    s["checkpoint"] = f"{IN}/bf16/checkpoint"
    s["expected_identity"] = {"model_sha256": identity}
    s["bf16_copy"] = {
        "receipt": f"{IN}/bf16/bf16-copy.json",
        "sha256": sha(f"{IN}/bf16/bf16-copy.json"),
    }
    s["vendor_source"] = VENDOR
    s["gate_receipt"] = f"{DECISIONS}/{NAME}.decision.ka12ib.json"
    text = old["runtime_equivalence"]
    for before, after in (
        ("787abdc54", FORMAL_MIRROR[:9]),
        ("4701ba41", FP32_IDENTITY[:8]),
    ):
        assert before in text, before
        text = text.replace(before, after)
    s["runtime_equivalence"] = text
    s["scored"] = {
        "label": "post-key same-panel run K-a12IB-16k at T = 1 (predictions derived from the sealed CAL698 run by "
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
            "mlx_predictions": f"{CURRENT_MLX}/output/mlx-diag.predictions.jsonl",
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
        "index_first": dict(INDEX),
    }
    s["frozen_autotune_cache"] = {
        "formal": "/data/dev2/runs/9b/formal-m9/triton-cache (the formal run's persisted cache; the typed-final, "
        "css15, public231 and mlx-diag runs shared it)",
    }
    return s


def decision(s: dict) -> dict:
    profile = gate.gate_profile(s)
    items = gate.successor_items(s, profile)
    failed = [k for k, v in items.items() if not v["passed"]]
    if failed:
        raise SystemExit(f"Index-first items fail: {failed}")
    paired = load(profile["paired"])
    low, high = gate._low_high(paired["ci95"])
    h = paired["axis_ci95"]["H"]["delta"]
    v3 = load(f"{RUN}/REPORT.json")["v3"]["score"]
    current_v3 = load(f"{CURRENT_RUN}/REPORT.json")["v3"]["score"]
    public = load(profile["public231"])
    mlx = load(profile["mlx_paired"])["overall"]
    audit = load(INDEX["audit"])
    kib = next(iter(audit["training_sets"].values()))
    return {
        "schema": gate.DECISION_SCHEMA,
        "status": "final",
        "decision": "release",
        "rule": gate.INDEX_FIRST,
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
        f"temperatures of the formal run (calibration {sha(CAL)[:8]}) are not shipped; undoing them changes 0 "
        "answers on every scored panel and mlx-diag",
        "action": f"New main revision of the private repository {REPO}: the 9B M9 point K-a12IB ([KIB soup, Lux 1.0] "
        "at alpha 1/2; qwen-full, BF16 storage, T = 1, 16,384 tokens) replaces the K-a13IB weights "
        f"({CURRENT['weights_identity'][:8]}, revision {CURRENT['revision'][:8]}) with the current product card "
        "(banner concept A); then the superseded weight blobs are purged with rewrite_history=False "
        "(hf_headroom.sh first). The repository stays private and in the private collection.",
        "rationale": "Index-first rule (user 2026-10-02 09:55 UTC+8), Index sweep selection (COORDINATION 10:00). "
        "IF1: private Index paired bootstrap of exactly these weights minus the current revision's, 95% lower bound "
        "> 0 (values in private files only; evidence_sha256.index_first_bootstrap); the largest lower bound among the "
        "9B candidates measured (K-a12IB, L9IB). Integrity: types OK on the scored run (R3); row-level Index audit "
        f"of the training file ({kib['training_lines']} lines: {kib['item_rows']} item rows; planted control "
        f"{audit['planted_control']['found']} / {audit['planted_control']['planted']}); exact parity on every scored "
        "prompt and Transformers trust_remote_code under 5.17 / 5.18 are release.sh steps (gate items 3-7). "
        f"References (not gates): post-key v3 {v3:.3f} vs {current_v3:.3f}, "
        f"{signed(paired['point']['delta']['score'])} [{signed(low)}, {signed(high)}]; human transfer "
        f"{signed(h['low'], 3)} to {signed(h['high'], 3)}; card-eligible mlx-diag {signed(mlx['delta'], 4)} "
        f"[{signed(mlx['ci95']['low'], 4)}, {signed(mlx['ci95']['high'], 4)}]; public 231 {public['left_correct']} "
        f"vs {public['right_correct']} (McNemar p {public['mcnemar_exact_p']:.3f}); C1 not run (a reference under "
        "the rule).",
        "approved_package": {
            "identity": s["expected_identity"]["model_sha256"],
            "fp32_identity": FP32_IDENTITY,
            "profile": "qwen-full (BF16 storage of the Linear projections)",
            "temperature": 1,
            "max_input_tokens": 16384,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": s.get("licence", {}).get("decision")
        or "apache-2.0: own weights; Decision 1.0 Lux-9B @bd45a30a (direct weight origin, interpolated) and the "
        "Qwen/Qwen3.5-9B @c2022362 text backbone and tokenizer are Apache-2.0; the package keeps the Apache-2.0 LICENSE.",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": [
            "release by the Index-first rule: the JevArena v3, human-transfer, mlx-diag, public 231 and C1 readings "
            "are references, not gates (values in the rationale)",
            "the training rows include the IB1-r3 / IB2 families matched to Index benchmark families (HoVer, "
            "When2Call, iSarcasmEval, GSM8K) and the BPoMP format; the internal records report the transfer-only "
            "Index delta without those benchmarks (private values)",
            "card.speed is the current revision's bench receipt (same architecture, runtime and shapes)",
        ],
        "prepared_by": PREPARED_BY,
        "decided_by": DECIDED_BY,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--check", action="store_true", help="compare with the committed copies"
    )
    args = ap.parse_args()
    for path, value in (
        (CURRENT["gate"], CURRENT["gate_sha256"]),
        (CURRENT["decision"], CURRENT["decision_sha256"]),
    ):
        if sha(path) != value:
            raise SystemExit(f"{path} is not the current revision's file")
    s = spec()
    spec_text = json.dumps(s, indent=2, ensure_ascii=False) + "\n"
    d = decision(s)
    decision_text = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    outputs = {
        "dev2-9b-ka12ib.json": (spec_text, SPECS),
        f"{NAME}.decision.ka12ib.json": (decision_text, RECORD),
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
