"""Release spec and decision of the Decision-2.0-Lux-9B Index-first successor of KIB4-a40 (9B M10 amendment 7).

User rule 2026-10-02 09:55 UTC+8 (COORDINATION "INDEX-FIRST"): the only quality gate is the frozen candidate's private
Jev Decision Index delta vs the current release, significantly positive (paired bootstrap over rows within
benchmarks through the board's area weights, 2,000 replicates, 95% lower bound > 0; ``gate.py`` IF1, which binds the
two IX1 run receipts to exactly this package's weights and the current revision's). Integrity checks: R3 (no type
collapsed on the formal typed panel), IF3 (the row-level Index contamination audit of the training files), and the
release items 2-7. v3, human transfer, mlx-diag, the tier gates, overlap exposure and public 231 are references
(``1_successor_references``); C1 is not run. The highest measured passer ships (``CHOICE``; values stay private).

The spec derives from the spec of the current revision (specs/dev2-9b-ras.json, ``main`` 214ffa43: the KIB4-a40
weights of f3122c7c on the phase A runtime with the opt-in shared-context switch, a runtime-only revision of
2026-10-03): roster, peers, runtime, remote code, licence and the product card carry over; the weights, the scored
run, the gate evidence and the card's Index input and assets (the default generator, banner concept A) are the
candidate's. The scored run is the candidate's sealed CAL698 formal run (m9/formal.sh, runner mirror unchanged since
787abdc54) returned to T = 1 offline (ops/prep_m10c.sh derive), as for KIB4-a40. The Index base is M10-KIB4-a40-bf16,
the IX1 run of the current revision's exact weights.

Run on node A from the exact mirror holding this file (host python3; every hashed file is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> --cand CAND --decided-utc UTC --out DIR [--check]
It writes DIR/dev2-9b-m10c-CAND.json and DIR/Decision-2.0-Lux-9B.decision.m10c-CAND.json; the committed copies live
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
BASE_SPEC = ROOT / "v2/release/specs/dev2-9b-ras.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Lux-9B"
REPO = f"vllm-sr/{NAME}"
REL = "/data/dev2/runs/release"
FORMAL = "/data/dev2/runs/9b/formal-m9"
DECISIONS = f"{REL}/decisions"
CURRENT_RUN = f"{REL}/dev2-9b-m10-KIB4-a40-t1-derived"
CURRENT_MLX = f"{REL}/dev2-9b-m10-KIB4-a40-t1-derived-mlx"
CURRENT_INDEX = "M10-KIB4-a40-bf16"
# The Lux runtime-only switch revision 214ffa43 (record dev2-runtime-a-2026-10-02, 9b/switch; the 9B owner, 2026-10-03
# 02:24Z): the KIB4-a40 weights of f3122c7c, unchanged.
CURRENT = {
    "revision": "214ffa4322bc1bce3215c1bd5de6168402c76969",
    "gate_sha256": "277f24d1fd356f5ca169d09005d2089d8325e4d3905c4368a95d3b2403a9ea50",
    "decision_sha256": "6c18dd57d5bd2070ac9e4849ed1c76877847924e1dbd3c851e050a2e441fa328",
    "manifest_sha256": "6e8e7f7dbe98cc100ed2d478eec9aba3ae1bc1711ac2ffe323f60512684d913b",
    "weights_identity": "0ece5faa210173f443f353474b419fd913a253b06645da3e8370c2db0e1c3339",
}
VENDOR = "/data/dev2/src/787abdc54946ccdb05e52cc5c34ecb619dced237-src_training_decision2/src/training/decision2"
EXPOSURE = [
    "/data/dev2/runs/9b/m6/exposure/x60.json",
    "/data/dev2/runs/9b/m9/exposure/ib1-ib2-train.json",
]
RECIPE = (
    "full fine-tuning of Decision 1.0 Lux-9B on the K-a13IB recipe (x60 with own-Lux targets, IB rows gold only, "
    "60.18M native tokens per seed)"
)
ARMS = {
    "KIB4": "9B M10 arm KIB4 ("
    + RECIPE
    + " with IB1 sentfin replaced by the IB4 phase-1 families)",
    "KIB4P": "9B M10 arm KIB4 ("
    + RECIPE
    + " with IB1 sentfin replaced by the IB4 phase-1 families) with its third seed (amendment 6)",
    "KX": "9B M10 arm KX ("
    + RECIPE
    + " with IB1 sentfin replaced by the IB4 phase-1 families, plus the IB3-r2 maths rows and one copy of whole "
    "non-English x60 groups to x60's multilingual token share)",
    "KIBM": "9B M10 arm KIBM ("
    + RECIPE
    + " with IB1 sentfin replaced by the IB3-r2 maths rows)",
    "KSW": "9B M10 arm KSW ("
    + RECIPE
    + " with the x60 cut over English groups only, IB1 sentfin dropped and self-distillation from K-a13IB)",
    "KIB4H": "9B M10 arm KIB4H ("
    + RECIPE
    + " with IB1 sentfin replaced by the IB4 phase-1 families, at half learning rates: backbone 5e-6, head 5e-5; "
    "the seeds of KIB4 s1 / s2)",
    "KIB4-lrhh": "the arm factory's 9B arm KIB4-lrhh ("
    + RECIPE
    + " with IB1 sentfin replaced by the IB4 phase-1 families, at half learning rates: backbone 5e-6, head 5e-5; "
    "seeds 21 / 22)",
}
HLR4 = (
    "four half-learning-rate seeds of the KIB4 recipe ("
    + RECIPE
    + " with IB1 sentfin replaced by the IB4 phase-1 families, at backbone LR 5e-6 and head LR 5e-5): 9B M10 KIB4H "
    "s1 / s2 (the seeds of KIB4 s1 / s2) and the arm factory's KIB4-lrhh s1 / s2 (seeds 21 / 22)"
)
ALPHA = {
    "a33": "1/3",
    "a25": "1/4",
    "a40": "2/5",
    "a50": "1/2",
    "a60": "3/5",
    "a80": "4/5",
}
KIB4_FAMILY = (
    "nine KIB4-family seeds: M10 KIB4 s1-s3 and arm-factory KIB4 s4-s5 (the KIB4 recipe), KIB4W2 s1-s2 (IB4 phase-1 "
    "rows at twice the loss weight) and KIB4L2 s1-s2 (backbone LR 2e-5), all on KIB4's TRAIN"
)
# Candidates (amendment 7; filled in when measured): fp32 = the point's model SHA-256 from its soup build,
# audit_sets = the TRAIN files of its members, index_name = its IX1 run, index_node = the node holding that run.
CANDS: dict[str, dict] = {
    "X7-a40": {
        "what": "the 9B M10 cross-arm point X7-a40: W = 3/5 x Decision 1.0 Lux-9B + 2/15 x each of the uniform "
        "FP32 seed soups of "
        + ARMS["KIB4"]
        + ", "
        + ARMS["KX"]
        + " and "
        + ARMS["KSW"]
        + " (the uniform FP32 average of the points KIB4-a40, KX-a40 and KSW-a40)",
        "audit_sets": ["KIB4", "KX", "KSW"],
        "index_node": "b",
    },
    "X8-a40": {
        "what": "the 9B M10 cross-arm point X8-a40: W = 3/5 x Decision 1.0 Lux-9B + 1/5 x each of the uniform "
        "FP32 seed soups of "
        + ARMS["KIB4"]
        + " and "
        + ARMS["KSW"]
        + " (the uniform FP32 average of the points KIB4-a40 and KSW-a40)",
        "fp32": "310228cb3e188332695c34a533727c7fb342807ff4d55b457589bff3ce7e3cb9",
        "audit_sets": ["KIB4", "KSW"],
        "index_node": "b",
    },
    # Amendment 13 (half learning rates); every member seed trained on KIB4's TRAIN 2e72bcfd / teacher 377f8878.
    "KIB4H-a40": {
        "fp32": "5d22b60be9e0581e6946ebf9ca573ca797e666b2d9b2203703e058978ce8e1fa",
        "audit_sets": ["KIB4"],
        "index_node": "b",
    },
    "KIB4H-a80": {
        "fp32": "4dea07dc0ad38062bf88b4a5bca5d3b3db18a9c575bce433024f0cb3d0a97862",
        "audit_sets": ["KIB4"],
        "index_node": "b",
    },
    "KIB4-lrhh-a80": {
        "fp32": "699bfd70f5b1094c807ad6df6dc6fdeb9fd1e4c4db558aceca8a4d4bba07fc8e",
        "audit_sets": ["KIB4"],
        "index_node": "b",
    },
    "HLR4-a80": {
        "what": "the 9B M10 point HLR4-a80: W = 1/5 x Decision 1.0 Lux-9B + 4/5 x the uniform FP32 soup of "
        + HLR4,
        "fp32": "d99cc9f171ee2263791f4d3fd5eb850b12bcca6344710ab178190b87c44e4a4e",
        "audit_sets": ["KIB4"],
        "index_node": "b",
    },
    "HLR4-a60": {
        "what": "the 9B M10 point HLR4-a60: W = 2/5 x Decision 1.0 Lux-9B + 3/5 x the uniform FP32 soup of "
        + HLR4,
        "fp32": "0b056425ff705a81c5c734f6c72a64e950ddd29e44a581be914414d24ef94579",
        "audit_sets": ["KIB4"],
        "index_node": "b",
    },
}
# The release choice (the highest measured passer); set when chosen.
CHOICE = None
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's Index-first rule of 2026-10-02 09:55 UTC+8 "
    "(release gate = a significantly positive private Index delta vs the current release; integrity checks; "
    "references not blocking) and the directive of 2026-10-02 12:30 UTC+8 (release the highest measured candidate "
    "directly), applied to the 9B tier by the Lux-9B publisher (9B M10 worker)"
)
PREPARED_BY = (
    "Decision 2.0 9B M10 worker, Lux-9B publisher (worktree vllm-sr-dev2-9b-m10)"
)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def what(cand: str) -> str:
    c = CANDS[cand]
    if "what" in c:
        return c["what"]
    arm, point = cand.rsplit("-", 1)
    return (
        f"the 9B M10 point {cand}: W = (1 - {ALPHA[point]}) x Decision 1.0 Lux-9B + {ALPHA[point]} x the "
        f"uniform FP32 soup of the seeds of {ARMS[arm]}"
    )


def paths(cand: str) -> dict:
    c = CANDS[cand]
    inputs = f"{REL}/inputs/dev2-9b-m10c-{cand}"
    return {
        **c,
        "index_name": c.get("index_name", f"M10-{cand}-bf16"),
        "formal": f"{FORMAL}/M10-{cand}-16k",
        "formal_mlx": f"{FORMAL}/M10-{cand}-16k-mlx",
        "cal": f"{FORMAL}/M10-{cand}-cal/calibration.json",
        "run": f"{REL}/dev2-9b-m10c-{cand}-t1-derived",
        "mlx": f"{REL}/dev2-9b-m10c-{cand}-t1-derived-mlx",
        "in": inputs,
        "private": f"/data/dev2/private/release/m10c/{cand}",
        "gates": f"{inputs}/gates",
        "current_gate": f"{inputs}/current/ras-gate.json",
        "current_decision": f"{inputs}/current/{NAME}.decision.ras.json",
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
    s["_release"] = {
        "successor": f"{NAME} Index-first successor: {what(cand)}; stored as its v2.release.bf16_copy.",
        "gate": "successor profile with index_first (user rule 2026-10-02 09:55): IF1 = the private Index paired "
        "bootstrap of these weights minus the current revision's (M10-KIB4-a40-bf16), 95% lower bound > 0, bound to "
        "both IX1 run receipts; R3 = no type collapsed on the formal typed panel; IF3 = the row-level Index "
        "contamination audit of the training files; items 2-7 as for every release. v3, human transfer, "
        "mlx-diag, tier gates, exposure and public 231 are references; C1 is not run. The Index files stay in "
        "release node's private tree.",
        "scored": f"T = 1: the sealed CAL698 formal run {Path(p['formal']).name} (image f83b1d10, "
        "runner mirror equal to 787abdc54 for every scored source) returned to T = 1 offline with 0 answer "
        "changes on every panel and mlx-diag, then adopted, sealed and paired on the release node (ops/prep_m10.sh).",
        "storage": f"v2.release.bf16_copy of the frozen FP32 point (identity {p['fp32'][:8]} -> {identity[:8]}; "
        f"receipt {bf16_receipt[:8]}), the copy the Index run {p['index_name']} scored.",
        "runtime": "vendor_source = K-a13IB's formal runner mirror 787abdc54 (training/model checked equal to the "
        "scored adapter sources at build time); runtime_source = automap_source = the current revision's, "
        "unchanged: the phase A runtime (HIP-graph replay of each padded input shape and, on gfx942, fused Triton "
        "element-wise kernels; BF16-resident, Transformers remote code, forward token budget) plus the opt-in "
        "shared-context switch (shared_ctx.py, off by default, the default path byte-identical; runtime_source "
        "9cffe606c, whose runtime files equal integration ef27d8885's), with the phase A frozen pre-warmed "
        "autotune cache for parity.",
        "card": "the product card of the current revision with this candidate's reports, an Index input built by "
        "python -m v2.release.card_index (board-served parameter counts, the audited footnote) with the 9B point "
        "from the Index run on exactly these weights, and assets rendered by the default v2.release.card_assets "
        "(banner concept A); card.speed keeps the current revision's bench receipt (same architecture, runtime and "
        "shapes).",
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-9b-ras.json",
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
    s["vendor_source"] = VENDOR
    s["gate_receipt"] = f"{DECISIONS}/{NAME}.decision.m10c-{cand}.json"
    s["runtime_equivalence"] = (
        "decision2/qwen.py loads this full checkpoint with the training/model sources vendored from the scored run's "
        "runner mirror (787abdc54), whose SHA-256 equal the scored adapter sources (checked at build time), and "
        "applies the per-item batching, BF16-backbone / FP32-head execution, raw probabilities (temperature 1; no "
        "calibration file) and answer normalization of v2.dec.infer_dec. The scored checkpoint "
        f"({p['fp32'][:8]}) stored every tensor in FP32; this package (v2.release.bf16_copy) stores its 248 Linear "
        "projection matrices in BF16 exactly as BF16 autocast rounds them and every other tensor bit for bit in "
        "FP32. The runtime, the Transformers remote code and the forward token budget are the current revision's: "
        "the phase A runtime, which replays the backbone of each exact padded input shape as a HIP graph and, on "
        "MI300-class (gfx942) GPUs, fuses the element-wise ops of each decoder layer into Triton kernels that round "
        "exactly as the ops they replace, plus the opt-in shared-context switch (off by default; the default path "
        "runs exactly the phase A runtime). "
        "Checked on one GPU against the T = 1 predictions derived exactly from the sealed CAL698 predictions of "
        "every scored prompt (typed-final 1,600, css15 6,547, public231 231) and of the mlx-diag diagnostic "
        "(2,275) by release.sh --parity, with a copy of the current revision's frozen pre-warmed autotune cache "
        "(the formal run's persisted cache plus the fused kernels' configurations), and "
        "AutoModel against the native runtime on every scored prompt."
    )
    s["scored"] = {
        "label": f"post-key same-panel run {Path(p['formal']).name} at T = 1 (predictions derived from the sealed "
        "CAL698 run by undoing its temperatures; adopted and sealed)",
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
        "native_manifest": f"{p['formal']}/output/typed-final.predictions.jsonl.manifest.json",
    }
    card = s["card"]
    card["paired"] = f"{p['run']}/PAIRED-vs-adopted-1.0.json"
    card["paired_peers"] = {"nimble2": f"{p['run']}/PAIRED-vs-nimble2.json"}
    reports = card["reports"]
    assert reports[0]["role"] == "candidate" and reports[0]["label"] == NAME
    reports[0]["report"] = f"{p['run']}/REPORT.json"
    reports[0]["mlx"] = f"{p['mlx']}/mlx-diag.score.json"
    card["index"] = {
        "path": f"{p['private']}/decision-index-card.json",
        "sha256": sha(f"{p['private']}/decision-index-card.json"),
    }
    card["assets"] = {
        "dir": f"{p['private']}/9b",
        "receipt_sha256": sha(f"{p['private']}/9b/card-assets.json"),
    }
    s["frozen_autotune_cache"] = {
        "formal": f"{FORMAL}/triton-cache (the formal runs' persisted cache; the typed-final, css15, public231 and "
        "mlx-diag runs shared it)",
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
        "paired": f"{p['gates']}/paired-vs-lux-9b.json",
        "mlx_paired": f"{p['gates']}/mlx-paired-vs-lux-9b.json",
        "public231": f"{p['gates']}/public231-vs-lux-9b.json",
        "exposure": list(EXPOSURE),
        "tier": {
            "reference": "nimble2",
            "v3_share": 0.9,
            "paired": f"{p['gates']}/paired-vs-nimble2.json",
        },
        "index_first": {
            "bootstrap": f"{p['private']}/boot-full-vs-kib4a40.json",
            "receipt": f"{p['private']}/receipt.json",
            "base_receipt": f"{p['private']}/base-receipt.json",
            "audit": f"{p['private']}/audit.json",
        },
    }
    return s


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
    used = {k: v for k, v in sets.items() if k in p["audit_sets"]}
    assert set(used) == set(
        p["audit_sets"]
    ), "the audit lacks a training file of this candidate"
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
        "calibration": "none (temperature 1; every Decision 2.0 model keeps T = 1): the CAL698 per-type "
        f"temperatures of the formal run (calibration {sha(p['cal'])[:8]}) are not shipped; undoing them changes 0 "
        "answers on every scored panel and mlx-diag",
        "action": f"New main revision of the public repository {REPO}: {what(cand)}, stored as its BF16 copy "
        f"(qwen-full, T = 1, 16,384 tokens), replaces the KIB4-a40 weights ({CURRENT['weights_identity'][:8]}, "
        f"revision {CURRENT['revision'][:8]}) with the product card of the current revision (default generator, "
        "banner concept A) and this candidate's Index input; then the superseded weight blobs are purged with "
        "rewrite_history=False (hf_headroom.sh first). The repository stays public and in the public Decision 2.0 "
        "collection (release Hub visibility policy, USER 2026-10-03 01:11 UTC+8).",
        "rationale": "Index-first rule (user 2026-10-02 09:55 UTC+8). IF1: the private Index paired bootstrap of "
        "these exact weights minus the current revision's has a 95% lower bound > 0 (2,000 replicates; values in "
        "private files only; evidence_sha256.index_first_bootstrap); the highest measured passer of the 9B M10 "
        "continuation (amendments 7 and 13; directive 2026-10-02 12:30 UTC+8). R3: types "
        + ", ".join(
            f"{k} {v}" for k, v in gate._verdicts(load(profile["types"])).items()
        )
        + ". IF3: row-level Index audit, planted control "
        f"{audit['planted_control']['found']} / {audit['planted_control']['planted']}; "
        + "; ".join(f"{n} {v['item_rows']} item rows" for n, v in sorted(used.items()))
        + ". References (not release blockers): post-key v3 "
        f"{v3:.3f} vs {current_v3:.3f}, {signed(paired['point']['delta']['score'])} [{signed(low)}, {signed(high)}]; "
        f"human transfer {signed(h['low'], 3)} to {signed(h['high'], 3)}; card-eligible mlx-diag "
        f"{signed(mlx['delta'], 4)} [{signed(mlx['ci95']['low'], 4)}, {signed(mlx['ci95']['high'], 4)}]; vs "
        f"adopted Lux 1.0 {signed(own['point']['delta']['score'])} [{signed(gate._low_high(own['ci95'])[0])}, "
        f"{signed(gate._low_high(own['ci95'])[1])}]; public 231 {public['left_correct']} vs "
        f"{public['right_correct']} (McNemar p {public['mcnemar_exact_p']:.3f}); C1 post-key not run.",
        "approved_package": {
            "identity": s["expected_identity"]["model_sha256"],
            "fp32_identity": p["fp32"],
            "loaded_parameters": 7940895744,
            "profile": "qwen-full (BF16 storage of the Linear projections)",
            "temperature": 1,
            "max_input_tokens": 16384,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": "apache-2.0: own weights; Decision 1.0 Lux-9B @bd45a30a (direct weight origin, "
        "interpolated) and the Qwen/Qwen3.5-9B @c2022362 text backbone and tokenizer are Apache-2.0; the IB1-r3 / "
        "IB2 / IB3-r2 / IB4 phase-1 training rows are licence-clean (release-safe data records; IB4 phase 1 C1 "
        "recheck r3 PASS); the package keeps the Apache-2.0 LICENSE (product card: no NOTICE, attributions or "
        "evaluation/ files).",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": [
            "successor items 1 and 4 of the former rule are references under the Index-first rule (v3 "
            + (
                "significantly below"
                if high < 0
                else "not significantly above" if low <= 0 else "significantly above"
            )
            + " the current revision; card-eligible mlx-diag "
            + ("below it" if mlx["ci95"]["high"] < 0 else "not below it")
            + ")",
            "the training rows include IB families matched to Index benchmark families (HoVer, When2Call, "
            "iSarcasmEval (IB4 isarc2 is in-distribution), GSM8K, RAGTruth-style, FinEntity-style) and the BPoMP "
            "format; the internal records report the transfer-only Index delta without HoVer, When2Call, "
            "iSarcasmEval, GSM8K and BPoMP (private values)",
            "Index contamination audit of the training rows: "
            + "; ".join(
                f"{n} ({v['training_lines']} lines) {v['item_rows']} item rows, {v['duplicate_rows']} familiar-text rows"
                for n, v in sorted(used.items())
            ),
            "the overlap-exposure references cover the x60 and IB1 / IB2 rows; the IB3-r2 / IB4 rows are covered by "
            "their own release-safe records and by the row-level Index audit",
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
        f"dev2-9b-m10c-{args.cand}.json": (spec_text, SPECS),
        f"{NAME}.decision.m10c-{args.cand}.json": (decision_text, RECORD),
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
