"""Release spec and decision of a Decision-2.0-Vega-27B successor over M6-IB (27B worker, continuations #5 and #6).

The current revision is the organization card-only revision of M6-IB (``main`` e60bd8e3 on vllm-sr, record
``dev2-org-2026-10-02``; M6-IB's weights, released at 781b2b24, record ``dev2-27b-indexfirst-2026-10-02``). Same rules as that
release (user rule 2026-10-02 09:55 UTC+8, progressive release 11:35): the gate is the frozen candidate's private Jev
Decision Index paired bootstrap minus the current revision's (2,000 replicates, 95% lower bound > 0; ``gate.py``
IF1 binds both IX1 run receipts to these weights and to M6-IB's), plus R3 (no type collapsed on the formal typed
panel), IF3 (the row-level Index audit of every training file of the candidate) and the release items 2-7. v3,
human transfer, mlx-diag, tier gates, exposure and public 231 are references, now against the current revision's
runs (M6-IB's formal run and node A mlx-diag collection); C1 is not run.

Candidates (ARMS): cross-arm averages of the two M6 arms on the same base (the 2B finding of COORDINATION 15:55: such
averages usually score above both parents), i.e. exact weighted LoRA soups of the four seeds of M6-IB (A20 + IB1-r3)
and M6-IB2 (A20 + IB1-r3 + IB2), rank 512.

The spec derives from the current revision's spec (specs/dev2-27b-org.json): roster, peers, runtime,
remote code, licence and the product card carry over; the weights (qwen-adapter: the frozen soup checkpoint itself),
the scored run, the gate evidence and the card's Index input and assets (the default generator, banner concept A)
are the candidate's.

Run on node A from the exact mirror holding this file (host python3; every hashed file is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> --arm ARM --decided-utc UTC --out DIR [--check]
It writes DIR/dev2-27b-27bx-ARM.json and DIR/Decision-2.0-Vega-27B.decision.27bx-ARM.json; the committed copies
live in v2/release/specs and this record directory, and --check compares them byte for byte.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from v2.release import gate, layout

ROOT = Path(__file__).resolve().parents[5]
RECORD = Path(__file__).resolve().parents[1]
BASE_SPEC = ROOT / "v2/release/specs/dev2-27b-org.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Vega-27B"
REPO = f"vllm-sr/{NAME}"
REL = "/data/dev2/runs/release"
DECISIONS = f"{REL}/decisions"
M6 = "/data/dev2/runs/27b/m6"
GATES = f"{M6}/gates"
CURRENT_ARM = "M6-IB"
CURRENT = {
    "revision": "e60bd8e346110b3348b6e5a25c07329d350c4b21",
    "gate": f"{REL}/dev2-org-27B-20261002T113747Z/receipts/gate.json",
    "decision": f"{DECISIONS}/{NAME}.decision.org.json",
    "gate_sha256": "e9591a86c2ba07b1496da1206e5277e68044dfd443e5205f241b82974b0aef6b",
    "decision_sha256": "08d19f0e7d482fef162efb2231624f4d52f03ad86f194cac4a2a7257faa03aea",
    "manifest_sha256": "23a040164046b0fcd4a88006cabc474d0e965a5f091806b027b933ce2a5b24e4",
    "weights_identity": "e50fb4c130483d1c889c226f726fcca1e134a3ae0292f20907586738c06c9680",
}
CURRENT_RUN = f"{M6}/{CURRENT_ARM}/formal"
CURRENT_MLX = f"{M6}/mlx-diag/{CURRENT_ARM}/output/mlx-diag.predictions.jsonl"
LOADED_BASE, LOADED_PER_RANK = 25629863936, 7295488
MIXES = (
    {  # every training file of the soup's members (every a20ib1 row is an a20ib12 row)
        "a20ib1": "a2ccf844c612dc61cfc76ab9c53e4f23fddd8311ae451b9c9cdd352dac42491a",
        "a20ib12": "16cb5bbbcc40623426c00163e920fa1a1149348e00a724aed4ab9b4a29f7200c",
    }
)
CROSS = (
    "the cross-arm average of the 27B M6 arms M6-IB (the released A20r recipe on the A20 mixture plus the IB1-r3 "
    "Index-breadth rows; the current revision) and M6-IB2 (the same recipe on A20 + IB1-r3 + IB2): the exact "
    "weighted LoRA soup of their four seeds (two per arm, each a rank-128 LoRA on the frozen Qwen3.8-27B text "
    "backbone with the native candidate head; rank 512, alpha 1024)"
)
ARMS = {
    "M6-IBxIB2-m50": {
        "weights": "1/4 per seed (each arm 1/2)",
        "vendor": "8351e7c4d90e21aa2166a39427febb8e325cc0cc",
    },
    "M6-IBxIB2-m67": {
        "weights": "1/6 per M6-IB seed and 1/3 per M6-IB2 seed (M6-IB2 2/3)",
        "vendor": "8351e7c4d90e21aa2166a39427febb8e325cc0cc",
    },
}
# the larger Index lower bound vs M6-IB of the two qualifying arms
CHOICE = "M6-IBxIB2-m50"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's Index-first rule of 2026-10-02 09:55 UTC+8 "
    "(release gate = a significantly positive private Index delta vs the current release; integrity checks; "
    "references not blocking) and the progressive-release directive of 11:35 UTC+8, applied to the 27B tier by the "
    "27B worker (continuations #5 and #6 of 2026-10-02: the cross-arm average of M6-IB and M6-IB2, released if its "
    "Index lower bound vs M6-IB is > 0; COORDINATION 21:20 UTC+8: release the larger lower bound on top of the "
    "organization card-only revision e60bd8e3)"
)
PREPARED_BY = (
    "Decision 2.0 27B worker, continuations #5 and #6 (worktree vllm-sr-dev2-27b)"
)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def paths(arm: str) -> dict:
    a = ARMS[arm]
    private = f"/data/dev2/private/release/27bif/{arm}"
    return {
        **a,
        "private": private,
        "run": f"{M6}/{arm}/formal",
        "checkpoint": f"{M6}/{arm}/checkpoint",
        "mlx": f"{M6}/mlx-diag/{arm}",
        "gates": f"{GATES}/{arm}",
        "exposure": f"{GATES}/overlap/exposure-m6-a20ib12.json",
        "mlx_paired": f"{GATES}/mlx/{arm}-vs-{CURRENT_ARM}.json",
    }


def loaded(soup: dict) -> int:
    return LOADED_BASE + LOADED_PER_RANK * soup["lora"]["rank"]


def spec(arm: str) -> dict:
    p = paths(arm)
    old = layout.current_ids(load(BASE_SPEC))
    assert (
        old["repo_id"] == REPO
        and old["expected_identity"]["model_sha256"] == CURRENT["weights_identity"]
    )
    s = copy.deepcopy(old)
    soup = load(f"{p['checkpoint']}/soup_manifest.json")
    identity = soup["output"]["model_sha256"]
    receipt = load(f"{p['private']}/receipt.json")
    assert (
        receipt["model_source"]["model_sha256"] == identity
    ), "the Index run scored other weights"
    seal = load(f"{p['run']}/SEAL.json")
    post = load(f"{p['run']}/triton-cache.post.json")
    lora = soup["lora"]
    assert lora["members"] == 4 and lora["member_rank"] == 128, lora
    what = f"{CROSS}, weights {p['weights']}, T = 1"
    s["_release"] = {
        "successor": f"{NAME} Index-first successor of M6-IB: {what}.",
        "gate": "successor profile with index_first (user rule 2026-10-02 09:55): IF1 = the private Index paired "
        "bootstrap of these weights minus the current revision's (M6-IB), 95% lower bound > 0, bound to both IX1 "
        "run receipts; R3 = no type collapsed on the formal typed panel; IF3 = the row-level Index contamination "
        "audit of both training files; items 2-7 as for every release. v3, human transfer, mlx-diag, tier gates, "
        "exposure and public 231 are references against the current revision's runs; C1 is not run. The Index "
        "files stay in the release node's private tree.",
        "scored": f"T = 1: the sealed formal run {arm}/formal (image dbe5f32b with its kernels), the same weights "
        "the IX1 Index run scored (restaged into the released runtime).",
        "storage": f"qwen-adapter: the frozen soup checkpoint itself (identity {identity[:8]}); the base is pinned, "
        "not redistributed.",
        "runtime": "vendor_source = the formal run's runner mirror (training/model checked equal to the scored "
        "adapter sources at build time); runtime_source = automap_source = the current revision's (BF16-resident "
        "backbone, Transformers remote code, forward token budget), unchanged.",
        "card": "the product card of the current revision with this candidate's reports, an Index input built by "
        "python -m v2.release.card_index (board-served parameter counts, the audited footnote) with the 27B point "
        "from the Index run on exactly these weights, and assets rendered on the release node by the default "
        "v2.release.card_assets (banner concept A) in the card render environment of the earlier rounds; "
        "card.speed keeps the current revision's bench receipt (same base, runtime and shapes).",
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-27b-org.json",
            "sha256": sha(BASE_SPEC),
        },
        "previous": old["_release"],
    }
    s["checkpoint"] = p["checkpoint"]
    s["expected_identity"] = {"model_sha256": identity}
    s["vendor_source"] = (
        f"/data/dev2/src/{p['vendor']}-src_training_decision2/src/training/decision2"
    )
    s["gate_receipt"] = f"{DECISIONS}/{NAME}.decision.27bx-{arm}.json"
    s["origin"]["summary"] = (
        f"A PEFT LoRA adapter (rank {lora['rank']}, alpha {lora['alpha']}, on all 496 text projections of the "
        "attention, gated-delta and MLP blocks) and a native candidate head trained on the frozen Qwen3.8-27B text "
        "backbone at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0), which is not redistributed. The adapter "
        f"is the exact weighted soup of {lora['members']} seeds of one rank-{lora['member_rank']} recipe trained on "
        "two mixtures (two seeds each; their LoRA factors stacked along the rank with each up-projection scaled by "
        f"its seed's weight, {p['weights']}, so the weighted mean of the seeds' weight updates is reproduced "
        "exactly), with the heads averaged by the same weights."
    )
    s["frozen_autotune_cache"] = {
        "path": f"{p['run']}/triton-cache",
        "tree_sha256": post["post_sha256"],
        "post_record": f"{p['run']}/triton-cache.post.json",
        "origin": f"the formal run's persisted cache: a fresh copy of F1's scored post-run cache (tree "
        f"{post['pre_sha256'][:8]}...) that added no autotune entry (group-file path rewrites only); the mlx-diag "
        "run copied the same tree; release runs copy it with python3 -m v2.27b.triton_cache copy --expect "
        "<tree_sha256>",
    }
    s["scored"] = {
        "label": f"post-key same-panel run {arm}/formal at T = 1 (kernel path, frozen autotune cache, 32,768-token "
        "limit)",
        "report_sha256": sha(f"{p['run']}/REPORT.json"),
        "seal_sha256": sha(f"{p['run']}/SEAL.json"),
        "predictions_sha256": {
            **{
                panel: seal["panels"][panel]["predictions_sha256"]
                for panel in ("typed-final", "css15", "public231")
            },
            "mlx-diag": sha(f"{p['mlx']}/output/mlx-diag.predictions.jsonl"),
        },
        "paired_sha256": sha(f"{p['gates']}/paired-vs-autojev27.json"),
        "native_manifest": f"{p['run']}/output/typed-final.predictions.jsonl.manifest.json",
    }
    card = s["card"]
    card["paired"] = f"{p['gates']}/paired-vs-autojev27.json"
    card["paired_peers"] = {
        "eikos27": f"{p['gates']}/paired-vs-eikos27b.json",
        "jebadiah27": f"{p['gates']}/paired-vs-jebadiah27b.json",
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
        "dir": f"{p['private']}/27b",
        "receipt_sha256": sha(f"{p['private']}/27b/card-assets.json"),
    }
    s["gate_profile"] = {
        "name": "successor",
        "run": p["run"],
        "current": {
            "revision": CURRENT["revision"],
            "gate": CURRENT["gate"],
            "decision": CURRENT["decision"],
            "run": CURRENT_RUN,
            "mlx_predictions": CURRENT_MLX,
        },
        "types": f"{p['gates']}/types.json",
        "paired": f"{p['gates']}/paired-vs-{CURRENT_ARM}.json",
        "mlx_paired": p["mlx_paired"],
        "public231": f"{p['gates']}/public231-vs-{CURRENT_ARM}.json",
        "exposure": [p["exposure"]],
        "tier": {
            "reference": "autojev27",
            "v3_share": 0.9,
            "paired": f"{p['gates']}/paired-vs-autojev27.json",
            "no_1_0": True,
        },
        "index_first": {
            "bootstrap": f"{p['private']}/boot-vs-current.json",
            "receipt": f"{p['private']}/receipt.json",
            "base_receipt": f"{p['private']}/base-receipt.json",
            "audit": f"{p['private']}/audit.json",
        },
    }
    return s


def decision(s: dict, arm: str) -> dict:
    p = paths(arm)
    soup = load(f"{p['checkpoint']}/soup_manifest.json")
    profile = gate.gate_profile(s)
    items = gate.successor_items(s, profile)
    failed = [k for k, v in items.items() if not v["passed"]]
    if failed:
        raise SystemExit(f"Index-first items fail: {failed}")
    adoption = load(f"{M6}/{arm}/ADOPTION.json")
    if adoption["adopt"]:
        raise SystemExit(
            "the formal run adopted CAL698; every Decision 2.0 model keeps T = 1"
        )
    paired = load(profile["paired"])
    low, high = gate._low_high(paired["ci95"])
    h = paired["axis_ci95"]["H"]["delta"]
    public = load(profile["public231"])
    mlx = load(profile["mlx_paired"])
    mlx_low, mlx_high = mlx["bootstrap"]["card_macro_ci95"]
    own = load(s["card"]["paired"])
    own_low, own_high = gate._low_high(own["ci95"])
    audit = load(profile["index_first"]["audit"])
    sets = audit["training_sets"]
    assert set(sets) == set(MIXES), sorted(sets)
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
        "calibration": "none (temperature 1; every Decision 2.0 model keeps T = 1): CAL698 was not adopted for "
        "this soup; no calibration file is shipped",
        "action": f"New main revision of the private repository {REPO}: {CROSS}, weights {p['weights']}; replaces "
        f"the M6-IB adapter ({CURRENT['weights_identity'][:8]}, revision {CURRENT['revision'][:8]}) with the "
        "product card of the current revision (default generator, banner concept A) and this candidate's Index "
        "input; then the superseded weight blobs are purged with rewrite_history=False (hf_headroom.sh first). The "
        "repository stays private and in the private collection.",
        "rationale": "Index-first rule (user 2026-10-02 09:55 UTC+8) and progressive release (11:35 UTC+8). IF1: "
        "the private Index paired bootstrap of these exact weights minus the current revision's (M6-IB) has a 95% "
        "lower bound > 0 (2,000 replicates, seed 20261002; values in private files only; "
        "evidence_sha256.index_first_bootstrap). R3: types "
        + ", ".join(
            f"{k} {v}" for k, v in gate._verdicts(load(profile["types"])).items()
        )
        + ". IF3: row-level Index audit, planted control "
        f"{audit['planted_control']['found']} / {audit['planted_control']['planted']}; "
        + "; ".join(f"{n} {v['item_rows']} item rows" for n, v in sorted(sets.items()))
        + ". References vs the current revision (not release blockers): post-key v3 "
        f"{paired['point']['left']['score']:.2f} vs {paired['point']['right']['score']:.2f}, "
        f"{signed(paired['point']['delta']['score'])} [{signed(low)}, {signed(high)}]; human transfer "
        f"{signed(h['low'], 3)} to {signed(h['high'], 3)}; card-eligible mlx-diag "
        f"{signed(mlx['delta']['card_macro'], 4)} [{signed(mlx_low, 4)}, {signed(mlx_high, 4)}]; public 231 "
        f"{public['left_correct']} vs {public['right_correct']} (McNemar p {public['mcnemar_exact_p']:.3f}); vs "
        f"AutoJev-27B {signed(own['point']['delta']['score'])} [{signed(own_low)}, {signed(own_high)}]; C1 "
        "post-key not run.",
        "approved_package": {
            "identity": s["expected_identity"]["model_sha256"],
            "loaded_parameters": loaded(soup),
            "profile": "qwen-adapter (unmerged PEFT LoRA and the native head on the pinned Qwen3.8-27B base)",
            "temperature": 1,
            "max_input_tokens": 32768,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": "apache-2.0: own LoRA adapter, head, runtime and card; base Qwen/Qwen3.8-27B @1d4bf0f2 "
        "Apache-2.0, pinned and not redistributed; the IB1-r3 and IB2 training rows are licence-clean (release-safe "
        "data records); the package keeps the Apache-2.0 LICENSE (product card: no NOTICE, attributions or "
        "evaluation/ files).",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": [
            "v3 against the current revision is a reference under the Index-first rule (v3 "
            + (
                "significantly below"
                if high < 0
                else (
                    "significantly above"
                    if low > 0
                    else "not significantly different from"
                )
            )
            + " the current revision)",
            "the training rows include the IB1-r3 and IB2 families matched to Index benchmark families and formats "
            "(IB2's hover and gsm2 are in-distribution for HoVer and GSM8K); the internal records report the "
            "transfer-only Index delta without those benchmarks (private values)",
            "Index contamination audit of both training files (every a20ib1 row is an a20ib12 row): "
            + "; ".join(
                f"{n} ({MIXES[n][:8]}) {v['training_lines']} lines, {v['item_rows']} item rows and "
                f"{v['duplicate_rows']} familiar-text rows"
                for n, v in sorted(sets.items())
            ),
            "card.speed is the current revision's bench receipt (same base, runtime and shapes; this adapter's "
            "rank is 512 instead of 256)",
        ],
        "prepared_by": PREPARED_BY,
        "decided_by": DECIDED_BY,
        "decided_utc": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--decided-utc", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--check", action="store_true", help="compare with the committed copies"
    )
    args = ap.parse_args()
    if CHOICE != args.arm:
        raise SystemExit(f"{args.arm} is not the recorded choice ({CHOICE})")
    for path, value in (
        (CURRENT["gate"], CURRENT["gate_sha256"]),
        (CURRENT["decision"], CURRENT["decision_sha256"]),
    ):
        if sha(path) != value:
            raise SystemExit(f"{path} is not the current revision's file")
    if (
        load(CURRENT["decision"])["identity"]["model_sha256"]
        != CURRENT["weights_identity"]
    ):
        raise SystemExit("the current decision names other weights")
    s = spec(args.arm)
    spec_text = json.dumps(s, indent=2, ensure_ascii=False) + "\n"
    d = decision(s, args.arm)
    d["decided_utc"] = args.decided_utc
    decision_text = json.dumps(d, indent=1, ensure_ascii=False) + "\n"
    outputs = {
        f"dev2-27b-27bx-{args.arm}.json": (spec_text, SPECS),
        f"{NAME}.decision.27bx-{args.arm}.json": (decision_text, RECORD),
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
