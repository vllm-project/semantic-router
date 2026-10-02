"""Release specs and decisions of the 0.8B / 2B Index-first successors (M16 08b-RA-a75 / M13 2b-RASD).

User decision 2026-10-02 09:55 UTC+8 (COORDINATION): the release gate is the private Jev Decision Index delta of the
frozen candidate vs the current release (paired bootstrap, >= 2,000 replicates, 95% lower bound > 0) plus integrity
checks (exact package parity, Hub / trust_remote_code, the row-level contamination audit, no collapsed type); v3,
human transfer, mlx-diag, public 231 and C1 are references; per tier the largest Index-gain lower bound is chosen.
The winners are M16's interpolation 08b-RA-a75 (Decision-2.0-Eos-0.8B) and, by the coordinator's 13:05 UTC+8
decision, M13's soup 2b-RASD (Decision-2.0-Sol-2B; formal T = 1 collection on the M16 path, no mlx-diag run),
shipped at T = 1 as the v2.release.bf16_copy of the FP32 points and served by the released runtime.

Each spec derives from the tier's current-main card spec (0.8B round 3, 2B round 4): roster, peers, runtime, remote code, licence and
the product card carry over (banner concept A is the generator default); the weights, origin text, scored run, gate
evidence and the card's Index input and assets are the successor's. The Index input is built by
``python -m v2.release.card_index`` from the kit runs of every tier's current release weights with this tier's
point scored on exactly these weights; it and the assets stay in node A's private tree, pinned by SHA-256.

Run on node A from the exact mirror holding this file (host python3; every file it hashes is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> 0p8b|2b --out DIR [--check]
It writes DIR/dev2-<key>-ixf.json and DIR/<name>.decision.ixf.json; the committed copies live in v2/release/specs
and this record directory, and --check compares them byte for byte. Prints only file SHA-256s.
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
SPECS = ROOT / "v2/release/specs"
CARDS = {
    "card3": ROOT / "v2/release/records/dev2-card3-2026-10-02",
    "card4": ROOT / "v2/release/records/dev2-card4-2026-10-02",
}
REL = "/data/dev2/runs/release"
DECISIONS = f"{REL}/decisions"
VENDOR = "/data/dev2/src/5b246b11096adbb8df73b6ba34f96b7373f7c95c-src_training_decision2/src/training/decision2"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's Index-first release rule of 2026-10-02 09:55 "
    "UTC+8 (the private Index delta vs the current release significantly positive; integrity checks; v3, human "
    "transfer, mlx-diag, public 231 and C1 reported as references) and the per-tier choice by the largest "
    "Index-gain lower bound, applied to M16's Index-path candidates by the 0.8B / 2B continuation"
)
PREPARED_BY = "Decision 2.0 decoder track, 0.8B / 2B Index-path continuation (worktree vllm-sr-dev2-dec-08bfast)"
DECIDED_UTC = "2026-10-02T02:15:00Z"
TIERS = {
    "0p8b": {
        "tier": "0.8B",
        "name": "Decision-2.0-Eos-0.8B",
        "point": "08b-RA-a75",
        "alpha": "three quarters",
        "current": "9c7f3ea09a2b04a0647e5919af23c20ed982f246",
        "current_run": f"{REL}/dev2-0p8b-t1-derived",
        "own": "adopted-1.0",
        "own_label": "Decision 1.0 Eos",
        "references": ("intern", "kev"),
        "train_note": "the arm's TRAIN file (12bd63d8; the previous release's rows plus the IB1-r3 / IB2 rows)",
        "origin": (
            "Every weight of Decision 1.0 Eos was fine-tuned (nothing frozen, no adapter). The release interpolates two "
            "such fine-tunes per tensor: one quarter of the previous Decision 2.0 Eos 0.8B weights (the uniform average "
            "of three seeds) plus three quarters of the uniform average of two seeds of a further fine-tune from "
            "Decision 1.0 Eos. Decision 1.0 Eos is itself a text-only fine-tune of "
            "[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) at "
            "`2fc06364715b967f1860aea9cf38778875588b17` (Apache-2.0), whose text backbone and tokenizer this model "
            "inherits; the Qwen3.5 vision tower is not part of it."
        ),
    },
    "2b": {
        "tier": "2B",
        "name": "Decision-2.0-Sol-2B",
        "point": "2b-RASD",
        "card": "card4",
        "mlx": False,
        "what": "the M13 soup",
        "successor": "Decision-2.0-Sol-2B successor 2b-RASD (decoder M13: the uniform average of two seeds of a full "
        "fine-tune of Decision 1.0 Sol on M12's 2b-RA TRAIN rows with the previous Decision 2.0 Sol 2B's answer "
        "probabilities as soft targets, KL weight 1.0; formal collection on the M16 path) under the user's "
        "Index-first release rule of 2026-10-02 09:55 UTC+8 and the coordinator's 13:05 UTC+8 choice: the private "
        "Index delta vs the current release significantly positive, plus the integrity checks; the per-tier choice "
        "by the largest Index-gain lower bound.",
        "uncalibrated": "collected at T = 1 (no CAL698 fit)",
        "current": "8ed41433f5f20c73bf04fe7ef92f2d69c1145002",
        "current_run": f"{REL}/dev2-2b-t1-derived",
        "own": "same-limit-16k",
        "own_label": "Decision 1.0 Sol (16K)",
        "references": ("decider2b", "thisthat12"),
        "tier_gate": lambda g: {
            "reference": "decider2b",
            "v3_share": 0.9,
            "paired": f"{g}/paired-vs-decider2b.json",
        },
        "train_note": "the arm's TRAIN file (08140409; the previous release's rows plus the IB1-r3 / IB2 rows)",
        "origin": (
            "Every weight of Decision 1.0 Sol was fine-tuned (nothing frozen, no adapter). The release is the uniform "
            "average of two seeds of that fine-tune, trained with the previous Decision 2.0 Sol 2B's answer "
            "probabilities as soft targets (self-distillation). Decision 1.0 Sol is itself a "
            "text-only fine-tune of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) at "
            "`15852e8c16360a2fea060d615a32b45270f8a8fc` (Apache-2.0), whose text backbone and tokenizer this model "
            "inherits; the Qwen3.5 vision tower is not part of it."
        ),
    },
}


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def paths(key: str) -> dict[str, str]:
    t = TIERS[key]
    inputs = f"{REL}/inputs/dev2-{key}-ixf"
    private = f"/data/dev2/private/release/ixf-{key}"
    return {
        "in": inputs,
        "run": f"{REL}/dev2-{key}-ixf-t1",
        "mlx": f"{REL}/dev2-{key}-ixf-t1-mlx",
        "formal": f"/data/dev2/runs/dec/formal/m16/m16-{t['point']}",
        "private": private,
        "gates": f"{inputs}/gates",
        "current_gate": f"{inputs}/current/gate.json",
        "current_decision": f"{inputs}/current/{t['name']}.decision.{card_round(key)}.json",
    }


def card_round(key: str) -> str:
    return TIERS[key].get("card", "card3")


def has_mlx(key: str) -> bool:
    return TIERS[key].get("mlx", True)


def spec(key: str) -> dict:
    t, p = TIERS[key], paths(key)
    base = SPECS / f"dev2-{key}-{card_round(key)}.json"
    old = load(base)
    assert old["repo_id"] == f"llm-semantic-router/{t['name']}"
    s = copy.deepcopy(old)
    bf16 = load(f"{p['in']}/bf16/bf16-copy.json")
    identity = bf16["model_sha256"]
    fp32 = bf16["source_model_sha256"]
    seal = load(f"{p['run']}/SEAL.json")
    s["_release"] = {
        "successor": t.get("successor")
        or f"{t['name']} successor M16 {t['point']} (decoder M16: the per-tensor interpolation W = (1 - "
        f"alpha) x current release + alpha x arm, alpha {t['alpha']}) under the user's Index-first release rule "
        "of 2026-10-02 09:55 UTC+8: the private Index delta vs the current release significantly positive, plus "
        "the integrity checks; the per-tier choice by the largest Index-gain lower bound.",
        "gate": f"successor profile with the Index-first rule (index_first block, gate.py f3590ef2a) against the current revision {t['current'][:8]} (its scored run "
        f"{Path(t['current_run']).name}): IF1 the full-panel private Index bootstrap of exactly these weights minus the "
        "current release's (both IX1 receipts bound, one panel), R3 no type collapsed, IF3 the row-level contamination audit; "
        "v3, human transfer, mlx-diag, exposure and public 231 (and at 2B the tier gates) are bound and printed as "
        "references. The Index files stay in node A's private tree."
        + (
            ""
            if has_mlx(key)
            else " No mlx-diag run (COORDINATION 11:40: no reference-only evals)."
        ),
        "scored": f"T = 1: the sealed M16 formal run m16-{t['point']} (node B, image dbe5f32b), collected without "
        "calibration, adopted unchanged on node A (ops/prep.sh adopt / paired).",
        "storage": f"v2.release.bf16_copy of the frozen FP32 point (identity {fp32[:8]} -> {identity[:8]}; receipt "
        f"{sha(p['in'] + '/bf16/bf16-copy.json')[:8]}).",
        "runtime": "vendor_source = the formal run's runner mirror 5b246b110 (training/model checked equal to the "
        "scored adapter sources at build time); runtime_source = automap_source = 99432d1a7, the released runtime, "
        "unchanged from the current revision.",
        "card": "the product card of the current revision with the successor's reports, the default banner "
        "(concept A, card round 4) and the round-3 Index conventions: card.index is built by "
        "python -m v2.release.card_index from the kit runs of every tier's current release weights, this tier's "
        "point scored on exactly these weights; card.assets rendered by v2.release.card_assets (Python 3.12.13, "
        "matplotlib 3.11.2, Pillow 12.3.0, Inter, the same logo); card.speed keeps the bench receipt of this "
        "runtime (same architecture, runtime and shapes).",
        "replaces_spec": {
            "spec": f"v2/release/specs/dev2-{key}-{card_round(key)}.json",
            "sha256": sha(base),
        },
        "previous": old["_release"],
    }
    s["origin"] = {**old["origin"], "summary": t["origin"]}
    s["checkpoint"] = f"{p['in']}/bf16/checkpoint"
    s["expected_identity"] = {"model_sha256": identity}
    s["bf16_copy"] = {
        "receipt": f"{p['in']}/bf16/bf16-copy.json",
        "sha256": sha(f"{p['in']}/bf16/bf16-copy.json"),
    }
    s["vendor_source"] = VENDOR
    s["gate_receipt"] = f"{DECISIONS}/{t['name']}.decision.ixf.json"
    s["runtime_equivalence"] = (
        "decision2/qwen.py loads this full checkpoint with the training/model sources vendored from the scored run's "
        "own runner mirror (5b246b110), whose SHA-256 equal the scored adapter sources (checked at build time), and "
        "applies the per-item batching, BF16-backbone / FP32-head execution, raw probabilities (temperature 1; no "
        "calibration file) and answer normalization of v2.dec.infer_dec, which for this non-residual checkpoint "
        f"wraps the same DecisionModel without extra readouts. The scored checkpoint ({fp32[:8]}) stored every "
        "tensor in FP32; this package (v2.release.bf16_copy) stores its Linear projection matrices in BF16 exactly "
        "as BF16 autocast rounds them and every other tensor bit for bit in FP32, and the runtime holds the "
        "backbone's BF16-exact Linear weights in BF16, the values BF16 autocast multiplies with. The runtime, the "
        "Transformers remote code (AutoModel with trust_remote_code) and the forward token budget are the current "
        "revision's. Checked on one GPU, in the scored image with a copy of the persisted Triton autotune cache of "
        "the formal run, against the sealed formal predictions of every scored prompt (typed-final 1,600, css15 "
        "6,547, public231 231) by release.sh --parity before and after the download, and AutoModel against the "
        "native runtime on every scored prompt."
    )
    s["scored"] = {
        "label": f"post-key same-panel run m16-{t['point']} at T = 1 (the sealed M16 formal run, collected without "
        "calibration; adopted and sealed)",
        "report_sha256": sha(f"{p['run']}/REPORT.json"),
        "seal_sha256": sha(f"{p['run']}/SEAL.json"),
        "predictions_sha256": {
            **{
                q: seal["panels"][q]["predictions_sha256"]
                for q in ("typed-final", "css15", "public231")
            },
            **(
                {"mlx-diag": sha(f"{p['mlx']}/output/mlx-diag.predictions.jsonl")}
                if has_mlx(key)
                else {}
            ),
        },
        "paired_sha256": sha(f"{p['run']}/PAIRED-vs-{t['own']}.json"),
        "native_manifest": f"{p['formal']}/output/typed-final.predictions.jsonl.manifest.json",
    }
    card = s["card"]
    card["paired"] = f"{p['run']}/PAIRED-vs-{t['own']}.json"
    reports = card["reports"]
    assert reports[0]["role"] == "candidate" and reports[0]["label"] == t["name"]
    reports[0]["report"] = f"{p['run']}/REPORT.json"
    if has_mlx(key):
        reports[0]["mlx"] = f"{p['mlx']}/mlx-diag.score.json"
    else:
        reports[0].pop("mlx", None)
    card["index"] = {
        "path": f"{p['private']}/decision-index-card.json",
        "sha256": sha(f"{p['private']}/decision-index-card.json"),
    }
    card["assets"] = {
        "dir": f"{p['private']}/{key}",
        "receipt_sha256": sha(f"{p['private']}/{key}/card-assets.json"),
    }
    g = p["gates"]
    s["gate_profile"] = {
        "name": gate.SUCCESSOR,
        "run": p["run"],
        "current": {
            "revision": t["current"],
            "gate": p["current_gate"],
            "decision": p["current_decision"],
            "run": t["current_run"],
            **(
                {
                    "mlx_predictions": f"{p['in']}/current-mlx/output/mlx-diag.predictions.jsonl"
                }
                if has_mlx(key)
                else {}
            ),
        },
        "paired": f"{g}/paired-vs-dev2-{key}.json",
        "types": f"{g}/types.json",
        **(
            {
                "mlx_paired": f"{g}/mlx-paired-vs-current.json",
                "exposure": f"{g}/exposure.json",
            }
            if has_mlx(key)
            else {}
        ),
        "public231": f"{g}/public231-vs-current.json",
        **({"tier": t["tier_gate"](g)} if t.get("tier_gate") else {}),
        "index_first": {
            "bootstrap": f"{p['private']}/paired-boot-full-vs-current.json",
            "receipt": f"{p['private']}/ix1-receipt.json",
            "base_receipt": f"{p['private']}/ix1-reference-receipt.json",
            "audit": f"{p['private']}/contamination-audit.json",
        },
    }
    s["frozen_autotune_cache"] = {
        "formal": f"{p['formal']}-cache (typed-final, css15, public231; manifest {p['formal']}-cache.sha256)",
    }
    if has_mlx(key):
        s["frozen_autotune_cache"][
            "mlx"
        ] = f"{p['formal']}-mlx-cache (mlx-diag; manifest {p['formal']}-mlx-cache.sha256)"
    return s


def decision(key: str, s: dict) -> dict:
    t, p = TIERS[key], paths(key)
    profile = gate.gate_profile(s)
    items = gate.successor_items(s, profile)
    assert list(items) == list(gate.INDEX_FIRST_ITEMS), list(items)
    failed = [k for k, v in items.items() if not v["passed"]]
    if failed:
        raise SystemExit(f"Index-first items fail: {failed}")
    g = p["gates"]
    v3p = load(profile["paired"])
    low, high = gate._low_high(v3p["ci95"])
    h = v3p["axis_ci95"]["H"]["delta"]
    own = load(f"{g}/paired-vs-{t['own']}.json")
    own_low, own_high = gate._low_high(own["ci95"])
    public = load(profile["public231"])
    if has_mlx(key):
        mlx = load(profile["mlx_paired"])["overall"]
        exposure = load(profile["exposure"])
        mlx_text = (
            f"; card-eligible mlx-diag {signed(mlx['delta'], 4)} [{signed(mlx['ci95']['low'], 4)}, "
            f"{signed(mlx['ci95']['high'], 4)}]"
        )
        exposure_text = f"; {len(exposure.get('groups') or [])} exposed training groups"
    else:
        mlx_text = "; mlx-diag not run"
        exposure_text = ""
    v3 = load(f"{p['run']}/REPORT.json")["v3"]["score"]
    current_v3 = load(f"{t['current_run']}/REPORT.json")["v3"]["score"]
    peers = []
    for n in t["references"]:
        pp = load(f"{g}/paired-vs-{n}.json")
        pl, ph = gate._low_high(pp["ci95"])
        peers.append(
            f"{n} {signed(pp['point']['delta']['score'])} [{signed(pl)}, {signed(ph)}]"
        )
    current_decision = load(p["current_decision"])
    current_gate = load(p["current_gate"])
    return {
        "schema": gate.DECISION_SCHEMA,
        "status": "final",
        "decision": "release",
        "model_name": t["name"],
        "repo_id": s["repo_id"],
        "identity": s["expected_identity"],
        "report_sha256": s["scored"]["report_sha256"],
        "paired_sha256": sha(s["card"]["paired"]),
        "gate_profile": gate.SUCCESSOR,
        "rule": gate.INDEX_FIRST,
        "current_revision": t["current"],
        "evidence_sha256": gate.evidence_sha256(profile),
        "calibration": "none (temperature 1; every Decision 2.0 model keeps T = 1): the formal run was "
        f"{t.get('uncalibrated', 'collected without calibration (CAL698 rejected)')}",
        "action": f"New main revision of the private repository {s['repo_id']}: {t.get('what', 'the M16 interpolation')} {t['point']} "
        f"(qwen-full, BF16 storage, T = 1, 16,384 tokens) replaces the weights "
        f"{current_decision['identity']['model_sha256'][:8]} of revision {t['current'][:8]}, with the product card "
        "(banner concept A, round-3 Index conventions); then the superseded weight blobs are purged with "
        "rewrite_history=False (hf_headroom.sh first). The repository stays private and in the private collection.",
        "rationale": "Index-first rule (user decision 2026-10-02 09:55 UTC+8). Release gate: the private Index paired "
        "bootstrap of exactly these weights minus the current release (full panel, 2,000 replicates) has a 95% lower "
        "bound > 0 (values in private files only; evidence_sha256.index_bootstrap); the per-tier choice is the "
        "largest Index-gain lower bound among the measured candidates. Integrity: types "
        + ", ".join(
            f"{k} {v}" for k, v in gate._verdicts(load(profile["types"])).items()
        )
        + f"; row-level Index contamination audit of {t['train_note']} with the planted control complete; exact "
        "package parity and the Hub / trust_remote_code checks are release.sh steps (gate items 2-7). References "
        f"(not gating): post-key v3 {v3:.3f} vs {current_v3:.3f}, {signed(v3p['point']['delta']['score'])} "
        f"[{signed(low)}, {signed(high)}]; human transfer {signed(h['low'], 3)} to {signed(h['high'], 3)}; vs "
        f"{t['own_label']} {signed(own['point']['delta']['score'])} [{signed(own_low)}, {signed(own_high)}]; peers "
        + "; ".join(peers)
        + mlx_text
        + f"; public 231 {public['left_correct']} vs {public['right_correct']} "
        f"(McNemar p {public['mcnemar_exact_p']:.3f}){exposure_text}. "
        "C1 item 8 was not run (09:55: a reference, run only if already in progress).",
        "approved_package": {
            "identity": s["expected_identity"]["model_sha256"],
            "fp32_identity": load(f"{p['in']}/bf16/bf16-copy.json")[
                "source_model_sha256"
            ],
            "profile": "qwen-full (BF16 storage of the Linear projections)",
            "temperature": 1,
            "max_input_tokens": 16384,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": "apache-2.0, unchanged: own weights; the Decision 1.0 base and the Qwen3.5 text "
        "backbone and tokenizer are Apache-2.0 (the product card ships the Apache-2.0 LICENSE).",
        "supersedes": {
            "revision": t["current"],
            "released_as": f"{s['repo_id']}@{t['current']}",
            "final_sha256": sha(p["current_decision"]),
            "gate_sha256": sha(p["current_gate"]),
            "manifest_sha256": current_gate["manifest_sha256"],
            "weights_identity": current_decision["identity"]["model_sha256"],
        },
        "disclosures": [
            "items 1 and 6(b) of the successor rule (v3 significantly above the current revision) are not met; the "
            "Index-first rule makes v3 a reference",
            "the training rows include the IB1-r3 / IB2 families matched to Index benchmark families (HoVer, "
            "When2Call, iSarcasmEval, GSM8K) and the BPoMP format; the internal records report the transfer-only "
            "Index delta without those benchmarks (private values)",
            "card.speed is the current revision's bench receipt (same architecture, runtime and shapes)",
        ],
        "prepared_by": PREPARED_BY,
        "decided_by": DECIDED_BY,
        "decided_utc": DECIDED_UTC,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=sorted(TIERS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--check", action="store_true", help="compare with the committed copies"
    )
    args = ap.parse_args()
    t, p = TIERS[args.key], paths(args.key)
    cards = CARDS[card_round(args.key)]
    committed = {
        p["current_gate"]: cards / f"{args.key}/release/receipts/gate.json",
        p["current_decision"]: cards
        / f"{t['name']}.decision.{card_round(args.key)}.json",
    }
    for node_path, record in committed.items():
        if sha(node_path) != sha(record):
            raise SystemExit(
                f"{node_path} is not the current revision's committed file {record}"
            )
    s = spec(args.key)
    spec_text = json.dumps(s, indent=2, ensure_ascii=False) + "\n"
    d = decision(args.key, s)
    decision_text = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    outputs = {
        f"dev2-{args.key}-ixf.json": (spec_text, SPECS),
        f"{t['name']}.decision.ixf.json": (decision_text, RECORD),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    problems = []
    for name, (text, where) in outputs.items():
        (args.out / name).write_text(text, encoding="utf-8")
        if args.check and (where / name).read_text(encoding="utf-8") != text:
            problems.append(f"{where / name} differs from the derivation")
        print(name, hashlib.sha256(text.encode()).hexdigest())
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
