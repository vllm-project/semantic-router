"""Release spec and decision of the Decision-2.0-Vega-27B Index-first successor (27B M6 worker 355ad916).

User rule 2026-10-02 09:55 UTC+8 (COORDINATION "INDEX-FIRST"; prereg amendment 7 of the 27B M6 milestone) and the
progressive-release directive of 11:35: the only quality gate is the frozen candidate's private Jev Decision Index
delta vs the current release, significantly positive (paired bootstrap over rows within benchmarks through the
board's area weights, 2,000 replicates, 95% lower bound > 0; ``gate.py`` IF1, which binds the two IX1 run receipts
to exactly this package's weights and the current revision's). Integrity checks: R3 (no type collapsed on the formal
typed panel), IF3 (the row-level Index contamination audit of the training file), and the release items 2-7
(package parity, download hashes, examples, output consistency, card, Transformers remote code with the Hub smoke
under 5.17 / 5.18). v3, human transfer, mlx-diag, the tier gates, overlap exposure and public 231 are references
(``1_successor_references``); C1 is not run. M6-IB is the first 27B candidate whose Index run finished with a lower
bound > 0 (the 11:35 directive: release it now; later candidates must beat the then-current release).

The spec derives from the round-4 card spec of the current revision (specs/dev2-27b-card4.json, ``main``
b689ee66, banner concept A): roster, peers, runtime (BF16-resident backbone, forward token budget, Transformers
remote code; runtime_source = automap_source of the current revision), remote code, licence and the product card
carry over; the weights (qwen-adapter: the frozen soup checkpoint itself), the scored run, the gate evidence and the
card's Index input and assets (the default generator: banner concept A) are the candidate's.

Run on node A from the exact mirror holding this file (host python3; every hashed file is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> --arm M6-IB --decided-utc UTC --out DIR [--check]
It writes DIR/dev2-27b-27bif-ARM.json and DIR/Decision-2.0-Vega-27B.decision.27bif-ARM.json; the committed copies
live in v2/release/specs and this record directory, and --check compares them byte for byte.
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
BASE_SPEC = ROOT / "v2/release/specs/dev2-27b-card4.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Vega-27B"
REPO = f"llm-semantic-router/{NAME}"
REL = "/data/dev2/runs/release"
DECISIONS = f"{REL}/decisions"
M6 = "/data/dev2/runs/27b/m6"
GATES = f"{M6}/gates"
# The node A mirror (integration >= cd565a588) that holds the round-4 card record of the current revision.
CARD4 = (
    "/data/dev2/src/fc7b6c866a8452f48f503ba7381e6dc47a9633be-src_training_decision2/src/training/decision2/"
    "v2/release/records/dev2-card4-2026-10-02"
)
CURRENT = {
    "revision": "b689ee66b51a0c0c45faf42b63e5aa0eaf187bde",
    "gate": f"{CARD4}/27b/release/receipts/gate.json",
    "decision": f"{CARD4}/{NAME}.decision.card4.json",
    "gate_sha256": "8a3da2516d1b8898761986b8bfdac93c42d43b214d82b62dfc9338f198a3ce70",
    "decision_sha256": "1f533d2e3b7671635b8ee42094215b114b8e27f19bf3dc8d5277d01b294a9f42",
    "manifest_sha256": "6358ebfa1f081ff40726121ec10f5feb26ac72d39249cc1c48a918e099be1437",
    "weights_identity": "2e07451107a2ca9735064cc01f7e43e8522b4bae48f2d77bdf29f7ea2fe6362c",
}
CURRENT_RUN = "/data/dev2/runs/27b/M4-A20r-soup/formal"
CURRENT_MLX = (
    "/data/dev2/runs/27b/m4-mlx/M4-A20r-soup/output/mlx-diag.predictions.jsonl"
)
LOADED_PARAMETERS = 27497508864
ARMS = {
    "M6-IB": {
        "mix": "a20ib1",
        "train": "a2ccf844c612dc61cfc76ab9c53e4f23fddd8311ae451b9c9cdd352dac42491a",
        "vendor": "b74685ddb863f0abf9d303589edb2e8e89b85a02",
        "what": "the 27B M6 arm M6-IB = the released A20r recipe (rank-128 LoRA per seed on the frozen Qwen3.8-27B "
        "text backbone with the native candidate head) on the A20 mixture plus the IB1-r3 Index-breadth rows; the "
        "exact uniform soup of two seeds (rank 256, alpha 512), T = 1",
    },
}
CHOICE = "M6-IB"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's Index-first rule of 2026-10-02 09:55 UTC+8 "
    "(release gate = a significantly positive private Index delta vs the current release; integrity checks; "
    "references not blocking) and the progressive-release directive of 11:35 UTC+8 (release a frozen candidate as "
    "soon as it qualifies; later candidates must beat the then-current release), applied to the 27B tier by the "
    "27B M6 worker 355ad916 (COORDINATOR INTERRUPT 12:40 UTC+8: release M6-IB to Vega-27B now)"
)
PREPARED_BY = "Decision 2.0 27B M6 worker 355ad916 (worktree vllm-sr-dev2-27b)"


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
        "exposure": f"{GATES}/overlap/exposure-m6-{a['mix']}.json",
        "mlx_paired": f"{GATES}/mlx/{arm}-vs-A20r.json",
    }


def spec(arm: str) -> dict:
    p = paths(arm)
    old = load(BASE_SPEC)
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
    s["_release"] = {
        "successor": f"{NAME} Index-first successor: {p['what']}.",
        "gate": "successor profile with index_first (user rule 2026-10-02 09:55): IF1 = the private Index paired "
        "bootstrap of these weights minus the current revision's, 95% lower bound > 0, bound to both IX1 run "
        "receipts; R3 = no type collapsed on the formal typed panel; IF3 = the row-level Index contamination "
        "audit of the training file; items 2-7 as for every release. v3, human transfer, mlx-diag, tier gates, "
        "exposure and public 231 are references; C1 is not run. The Index files stay in node A's private tree.",
        "scored": f"T = 1: the sealed formal run {arm}/formal (image dbe5f32b with its kernels, CAL698 not "
        "adopted), the same weights the IX1 Index run scored (restaged into the released runtime).",
        "storage": f"qwen-adapter: the frozen soup checkpoint itself (identity {identity[:8]}); the base is pinned, "
        "not redistributed.",
        "runtime": "vendor_source = the formal run's runner mirror (training/model checked equal to the scored "
        "adapter sources at build time); runtime_source = automap_source = the current revision's (BF16-resident "
        "backbone, Transformers remote code, forward token budget), unchanged.",
        "card": "the product card of the current revision with this candidate's reports, an Index input built by "
        "python -m v2.release.card_index (board-served parameter counts, the audited footnote) with the 27B point "
        "from the Index run on exactly these weights, and assets rendered on node A by the default "
        "v2.release.card_assets (banner concept A) in the card render environment of the earlier rounds (ops/"
        "render_env.sh); card.speed keeps the current revision's bench receipt (same base, runtime and shapes).",
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-27b-card4.json",
            "sha256": sha(BASE_SPEC),
        },
        "previous": old["_release"],
    }
    s["checkpoint"] = p["checkpoint"]
    s["expected_identity"] = {"model_sha256": identity}
    s["vendor_source"] = (
        f"/data/dev2/src/{p['vendor']}-src_training_decision2/src/training/decision2"
    )
    s["gate_receipt"] = f"{DECISIONS}/{NAME}.decision.27bif-{arm}.json"
    s["origin"]["summary"] = (
        f"A PEFT LoRA adapter (rank {lora['rank']}, alpha {lora['alpha']}, on all 496 text projections of the "
        "attention, gated-delta and MLP blocks) and a native candidate head trained on the frozen Qwen3.8-27B text "
        "backbone at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0), which is not redistributed. The adapter "
        f"is the exact uniform soup of {lora['members']} seeds of one rank-{lora['member_rank']} recipe (their LoRA "
        "factors stacked along the rank with the up-projections halved, so the mean of the two seeds' weight "
        "updates is reproduced exactly), with the two heads averaged."
    )
    old_check = old["runtime_equivalence"].rsplit(" Checked on one GPU", 1)
    assert len(old_check) == 2, "the current runtime_equivalence text changed"
    s["runtime_equivalence"] = (
        old_check[0]
        + " Checked on one GPU of node A (same image and base bytes as the scoring run) against the "
        "scored T = 1 predictions of every formal prompt (typed-final 1,600, css15 6,547, public231 231) and of the "
        "mlx-diag diagnostic (2,275) by release.sh --parity, with the image's FLA and causal-conv1d kernels required "
        "and a fresh copy of the formal run's persisted Triton autotune cache (the cache the mlx-diag run also "
        "copied), and AutoModel against the native runtime on every scored prompt."
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
        "limit; CAL698 not adopted)",
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
        "paired": f"{p['gates']}/paired-vs-A20r.json",
        "mlx_paired": p["mlx_paired"],
        "public231": f"{p['gates']}/public231-vs-A20r.json",
        "exposure": [p["exposure"]],
        "tier": {
            "reference": "autojev27",
            "v3_share": 0.9,
            "paired": f"{p['gates']}/paired-vs-autojev27.json",
            "no_1_0": True,
        },
        "index_first": {
            "bootstrap": f"{p['private']}/boot-vs-a20r.json",
            "receipt": f"{p['private']}/receipt.json",
            "base_receipt": f"{p['private']}/base-receipt.json",
            "audit": f"{p['private']}/audit.json",
        },
    }
    return s


def decision(s: dict, arm: str) -> dict:
    p = paths(arm)
    profile = gate.gate_profile(s)
    items = gate.successor_items(s, profile)
    failed = [k for k, v in items.items() if not v["passed"]]
    if failed:
        raise SystemExit(f"Index-first items fail: {failed}")
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
    train = sets[p["mix"]]
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
        "this soup (it worsened two calibration metrics); no calibration file is shipped",
        "action": f"New main revision of the private repository {REPO}: {p['what']}; replaces the A20r adapter "
        f"({CURRENT['weights_identity'][:8]}, revision {CURRENT['revision'][:8]}) with the product card of the "
        "current revision (default generator, banner concept A) and this candidate's Index input; then the "
        "superseded weight blobs are purged with rewrite_history=False (hf_headroom.sh first). The repository "
        "stays private and in the private collection.",
        "rationale": "Index-first rule (user 2026-10-02 09:55 UTC+8) and progressive release (11:35 UTC+8). IF1: "
        "the private Index paired bootstrap of these exact weights minus the current revision's has a 95% lower "
        "bound > 0 (2,000 replicates, seed 20261002; values in private files only; "
        "evidence_sha256.index_first_bootstrap). R3: types "
        + ", ".join(
            f"{k} {v}" for k, v in gate._verdicts(load(profile["types"])).items()
        )
        + ". IF3: row-level Index audit, planted control "
        f"{audit['planted_control']['found']} / {audit['planted_control']['planted']}; "
        + "; ".join(f"{n} {v['item_rows']} item rows" for n, v in sorted(sets.items()))
        + ". References (not release blockers): post-key v3 "
        f"{paired['point']['left']['score']:.2f} vs {paired['point']['right']['score']:.2f}, "
        f"{signed(paired['point']['delta']['score'])} [{signed(low)}, {signed(high)}]; human transfer "
        f"{signed(h['low'], 3)} to {signed(h['high'], 3)}; card-eligible mlx-diag "
        f"{signed(mlx['delta']['card_macro'], 4)} [{signed(mlx_low, 4)}, {signed(mlx_high, 4)}]; vs AutoJev-27B "
        f"{signed(own['point']['delta']['score'])} [{signed(own_low)}, {signed(own_high)}]; public 231 "
        f"{public['left_correct']} vs {public['right_correct']} (McNemar p {public['mcnemar_exact_p']:.3f}); C1 "
        "post-key not run.",
        "approved_package": {
            "identity": s["expected_identity"]["model_sha256"],
            "loaded_parameters": LOADED_PARAMETERS,
            "profile": "qwen-adapter (unmerged PEFT LoRA and the native head on the pinned Qwen3.8-27B base)",
            "temperature": 1,
            "max_input_tokens": 32768,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": "apache-2.0: own LoRA adapter, head, runtime and card; base Qwen/Qwen3.8-27B @1d4bf0f2 "
        "Apache-2.0, pinned and not redistributed; the IB1-r3 training rows are licence-clean (release-safe data "
        "records); the package keeps the Apache-2.0 LICENSE (product card: no NOTICE, attributions or evaluation/ "
        "files).",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": [
            "the former successor item 1 (v3 significantly above the current revision) and the beats-AutoJev "
            "reading are not met (v3 "
            + ("significantly below" if high < 0 else "not significantly above")
            + " the current revision); under the Index-first rule they are references",
            "the training rows include the IB1-r3 families matched to Index benchmark families and formats; the "
            "internal records report the transfer-only Index delta without those benchmarks (private values)",
            f"Index contamination audit of the training rows ({p['train'][:8]}, {train['training_lines']} lines, "
            f"the one file both seeds trained on): {train['item_rows']} item rows and {train['duplicate_rows']} "
            "familiar-text rows",
            "card.speed is the current revision's bench receipt (same base, runtime and shapes; this adapter's "
            "rank is 256 instead of 64)",
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
        f"dev2-27b-27bif-{args.arm}.json": (spec_text, SPECS),
        f"{NAME}.decision.27bif-{args.arm}.json": (decision_text, RECORD),
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
