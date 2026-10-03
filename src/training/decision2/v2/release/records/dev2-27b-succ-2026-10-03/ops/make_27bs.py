"""Release spec and decision of a Decision-2.0-Vega-27B successor over M6-IBxIB2-m50 (27B continuation #7).

The current revision is the runtime-only revision 9b067a95 on vllm-sr (phase A fast path and the opt-in
shared-context switch, off by default; record ``dev2-runtime-a-2026-10-02``, spec ``specs/dev2-27b-ras.json``) of the
weights released at 5c85c127 (the cross-arm soup M6-IBxIB2-m50, record ``dev2-27b-xarm-2026-10-02``). Same rules as
that release (user rule 2026-10-02 09:55 UTC+8; progressive release; COORDINATION 2026-10-03 01:36: same rules for the
public repositories): the gate is the frozen candidate's private Jev Decision Index paired bootstrap minus the current
revision's (2,000 replicates, 95% lower bound > 0; ``gate.py`` IF1 binds both IX1 run receipts to these weights and to
M6-IBxIB2-m50's), plus R3 (no type collapsed on the formal typed panel), IF3 (the row-level Index audit of every
training file of the candidate) and the release items 2-7. v3, human transfer, mlx-diag, tier gates, exposure and
public 231 are references against the current revision's runs (M6-IBxIB2-m50's formal run and mlx-diag collection);
C1 is not run.

Candidates (ARMS): the preregistered M7 / M8 / M9 soups (27B records m8-prereg-2026-10-02.md amendments 1-2 and
m9-prereg-2026-10-03.md amendments 1-2): exact weighted LoRA soups of BEST checkpoints of rank-128 seeds on the frozen
Qwen3.8-27B text backbone, uniform within an arm.

The spec derives from the current revision's spec (specs/dev2-27b-ras.json): roster, peers, runtime (runtime_source
and automap_source: phase A and the switch), remote code, licence, card.speed and the product card carry over; the
weights (qwen-adapter: the frozen soup checkpoint itself), the scored run, the vendored training/model sources (the
formal run's runner mirror, read from its COLLECT.json), the gate evidence and the card's Index input and assets (the
default generator, banner concept A) are the candidate's.

Run on node A from the exact mirror holding this file (host python3; every hashed file is a node A path):
  PYTHONPATH=<mirror>/src/training/decision2 python3 <this file> --arm ARM --decided-utc UTC --out DIR [--check]
It writes DIR/dev2-27b-27bs-ARM.json and DIR/Decision-2.0-Vega-27B.decision.27bs-ARM.json; the committed copies
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
BASE_SPEC = ROOT / "v2/release/specs/dev2-27b-ras.json"
SPECS = ROOT / "v2/release/specs"
NAME = "Decision-2.0-Vega-27B"
REPO = f"vllm-sr/{NAME}"
REL = "/data/dev2/runs/release"
DECISIONS = f"{REL}/decisions"
M6 = "/data/dev2/runs/27b/m6"
GATES = f"{M6}/gates"
CURRENT_ARM = "M6-IBxIB2-m50"
CURRENT = {
    "revision": "9b067a95560284dac8c98ef4130fd5a2c5a92ff9",
    "gate": f"{REL}/dev2-ras-27B-20261002T213625Z/receipts/gate.json",
    "decision": f"{DECISIONS}/{NAME}.decision.ras.json",
    "gate_sha256": "8b14ba65635ce810e2129b34e3b8bf014715efc5cc92119a6b6741cfae4b98d0",
    "decision_sha256": "fb52e3f63b4ee70d103e9aba6406111933b54ee5572701ea8eaca183036ccf70",
    "manifest_sha256": "0e90656f0e5b2a11a461d4ceb8f3a996c07813b5b643b0548adbc90d2f435414",
    "weights_identity": "584697310502cd20e91728e69f724c028b47b95daf3790256dea3e434c48be82",
}
CURRENT_RUN = f"{M6}/{CURRENT_ARM}/formal"
CURRENT_MLX = f"{M6}/mlx-diag/{CURRENT_ARM}/output/mlx-diag.predictions.jsonl"
LOADED_BASE, LOADED_PER_RANK = 25629863936, 7295488
MIXES = {  # SHA-256 of every training file a candidate's members trained on (node B mixtures-m6/m7/m9-1)
    "a20ib1": "a2ccf844c612dc61cfc76ab9c53e4f23fddd8311ae451b9c9cdd352dac42491a",
    "a20ib12": "16cb5bbbcc40623426c00163e920fa1a1149348e00a724aed4ab9b4a29f7200c",
    "a20ib14": "c16185b10c787676688351e99ecc7b2fd4efcdbe0d6ee537dcb098faf479d201",
    "a20ib124": "6f4ba05ef5abe4e7bfabe3195883a7ac11de2f0e8feebfe323d1021411d1826d",
    "a20ib14ml": "f60267508858796b32591fa3b1e4db4eaba15642ee56aaffe18c7611da23b2f7",
    "a20ib124ml": "68cda2995c97c3739e2e02e4dc4d7c490bd8e147d011df4d5ac53a91e77b9b46",
    "a20ib1ml": "9bdba9848756ecf74fe25ac3f1b92f529f891b4f11628d5bee72fa78e77584f9",
    "a20ib12ml": "0a3620792f8b8fd643faf5450a118a507af4be0b21f2cfee499732f9f3bd1109",
}
RECIPE = {  # mixture -> what its arm trains on
    "a20ib1": "A20 + IB1-r3 (M6-IB's file, the released recipe)",
    "a20ib12": "A20 + IB1-r3 + IB2 (M6-IB2's file)",
    "a20ib14": "A20 + IB1-r3 without sentfin + IB4 phase 1",
    "a20ib124": "A20 + IB1-r3 without sentfin + IB2 + IB4 phase 1",
    "a20ib14ml": "A20 + IB1-r3 without sentfin + IB4 phase 1 + the ML block",
    "a20ib124ml": "A20 + IB1-r3 without sentfin + IB2 + IB4 phase 1 + the ML block",
    "a20ib1ml": "A20 + IB1-r3 + the ML block",
    "a20ib12ml": "A20 + IB1-r3 + IB2 + the ML block",
}
M50 = "M6-IB s1 / s2 and M6-IB2 s1 / s2 (the current weights' four seeds)"
HALF = "half learning rates (LoRA 1e-5, head 5e-5, backbone 5e-7)"
ARMS = {  # members: allowed member counts; every member is a rank-128 seed
    "X7-IBxIB2xIB14ML": {
        "members": (6,),
        "mixes": ("a20ib1", "a20ib12", "a20ib14ml"),
        "seeds": f"{M50}, M7-IB14ML s1 / s2",
        "weights": "1/6 per seed (each arm 1/3)",
    },
    "M7-IB14ML": {
        "members": (2,),
        "mixes": ("a20ib14ml",),
        "seeds": "M7-IB14ML s1 / s2",
        "weights": "1/2 per seed",
    },
    "X7-4ARM": {
        "members": (9,),
        "mixes": ("a20ib1", "a20ib12", "a20ib14ml", "a20ib124ml"),
        "seeds": f"{M50}, M7-IB14ML s1 / s2, M7-IB124ML s1 / s2 / s3",
        "weights": "each arm 1/4, its seeds uniform within it",
    },
    "M7-IB124ML": {
        "members": (3,),
        "mixes": ("a20ib124ml",),
        "seeds": "M7-IB124ML s1 / s2 / s3",
        "weights": "1/3 per seed",
    },
    "X8-IBxIB2-8": {
        "members": (8,),
        "mixes": ("a20ib1", "a20ib12"),
        "seeds": f"{M50}, M8-IB s3 / s4, M8-IB2 s3 / s4",
        "weights": "1/8 per seed (each arm 1/2)",
    },
    "X8-ML": {
        "members": (12, 14),
        "mixes": (
            "a20ib1",
            "a20ib12",
            "a20ib14ml",
            "a20ib124ml",
            "a20ib14",
            "a20ib124",
        ),
        "seeds": f"{M50}, M7-IB14ML s1 / s2 + M8-IB14ML s3, M7-IB124ML s1 / s2 / s3, M8-IB14 s4 / s5, "
        "M8-IB124 s4 / s5 (at most two per arm by SELECT700 if the rank passes 1,536)",
        "weights": "each arm 1/6, its seeds uniform within it",
    },
    "M8-IB14": {
        "members": (2,),
        "mixes": ("a20ib14",),
        "seeds": "M8-IB14 s4 / s5",
        "weights": "1/2 per seed",
    },
    "M8-IB124": {
        "members": (2,),
        "mixes": ("a20ib124",),
        "seeds": "M8-IB124 s4 / s5",
        "weights": "1/2 per seed",
    },
    "X9-LRH2": {
        "members": (2,),
        "mixes": ("a20ib1", "a20ib12"),
        "seeds": f"M9-IB-lrh s5 and M9-IB2-lrh s5 (the two released recipes at {HALF})",
        "weights": "1/2 per seed",
    },
    "X9-LRH2xM50": {
        "members": (6,),
        "mixes": ("a20ib1", "a20ib12"),
        "seeds": f"M9-IB-lrh s5 and M9-IB2-lrh s5 (the two released recipes at {HALF}) and {M50}",
        "weights": "1/4 per half-LR seed and 1/8 per seed of the current weights (each recipe-LR arm 1/4)",
    },
    "X9-LRH": {
        "members": (3, 4),
        "mixes": ("a20ib1", "a20ib12"),
        "seeds": f"M9-IB-lrh s5 / s6 and M9-IB2-lrh s5 / s6 (the two released recipes at {HALF})",
        "weights": "each arm 1/2, its seeds uniform within it",
    },
    "X9-LRHxM50": {
        "members": (7, 8),
        "mixes": ("a20ib1", "a20ib12"),
        "seeds": f"M9-IB-lrh s5 / s6 and M9-IB2-lrh s5 / s6 (the two released recipes at {HALF}) and {M50}",
        "weights": "each recipe-LR arm 1/4, its seeds uniform within it",
    },
    "X9-ML0": {
        "members": (6, 7, 8),
        "mixes": ("a20ib1", "a20ib12", "a20ib1ml", "a20ib12ml"),
        "seeds": f"{M50}, M9-IB1ML s5 (/ s6), M9-IB12ML s5 (/ s6)",
        "weights": "each arm 1/4, its seeds uniform within it",
    },
    "X9-IBxIB2-10": {
        "members": (10,),
        "mixes": ("a20ib1", "a20ib12"),
        "seeds": f"{M50}, M8-IB s3 / s4, M8-IB2 s3 / s4, M9-IB s5, M9-IB2 s5",
        "weights": "1/10 per seed (each arm 1/2)",
    },
}
# set to the candidate with the largest Index lower bound > 0 vs the current revision, before deriving
CHOICE = None
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's Index-first rule of 2026-10-02 09:55 UTC+8 "
    "(release gate = a significantly positive private Index delta vs the current release; integrity checks; "
    "references not blocking), the progressive-release directive and the user's 2026-10-03 01:27 UTC+8 instruction "
    "(keep optimizing to same-size SOTA with progressive releases under the same rules), applied to the 27B tier by "
    "the 27B owner (continuation #7: the preregistered M7 / M8 / M9 soups, released on top of 9b067a95 when the "
    "largest Index lower bound vs M6-IBxIB2-m50 is > 0)"
)
PREPARED_BY = "Decision 2.0 27B owner, continuation #7 (worktree vllm-sr-dev2-27b)"
IB4_MIXES = {"a20ib14", "a20ib124", "a20ib14ml", "a20ib124ml"}
ML_MIXES = {"a20ib14ml", "a20ib124ml", "a20ib1ml", "a20ib12ml"}
IB2_MIXES = {"a20ib12", "a20ib124", "a20ib124ml", "a20ib12ml"}


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def paths(arm: str) -> dict:
    a = ARMS[arm]
    private = f"/data/dev2/private/release/27bs/{arm}"
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


def what(arm: str) -> str:
    a = ARMS[arm]
    mixes = "; ".join(f"{m}: {RECIPE[m]}" for m in a["mixes"])
    return (
        f"the exact weighted LoRA soup {arm} of the 27B seeds {a['seeds']} (each a rank-128 LoRA on the frozen "
        f"Qwen3.8-27B text backbone with the native candidate head; training files {mixes})"
    )


def vendor(arm: str) -> str:
    commit = load(f"{paths(arm)['run']}/COLLECT.json")["source_mirror"]["commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise SystemExit(f"{arm}/formal/COLLECT.json names no runner mirror commit")
    return commit


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
    assert lora["members"] in p["members"] and lora["member_rank"] == 128, lora
    text = f"{what(arm)}, weights {p['weights']}, T = 1"
    s["_release"] = {
        "successor": f"{NAME} Index-first successor of M6-IBxIB2-m50: {text}.",
        "gate": "successor profile with index_first (user rule 2026-10-02 09:55): IF1 = the private Index paired "
        "bootstrap of these weights minus the current revision's (M6-IBxIB2-m50), 95% lower bound > 0, bound to "
        "both IX1 run receipts; R3 = no type collapsed on the formal typed panel; IF3 = the row-level Index "
        "contamination audit of every training file; items 2-7 as for every release. v3, human transfer, mlx-diag, "
        "tier gates, exposure and public 231 are references against the current revision's runs; C1 is not run. "
        "The Index files stay in the release node's private tree.",
        "scored": f"T = 1: the sealed formal run {arm}/formal (image dbe5f32b with its kernels), the same weights "
        "the IX1 Index run scored (restaged into the released runtime).",
        "storage": f"qwen-adapter: the frozen soup checkpoint itself (identity {identity[:8]}); the base is pinned, "
        "not redistributed.",
        "runtime": "vendor_source = the formal run's runner mirror (training/model checked equal to the scored "
        "adapter sources at build time); runtime_source = automap_source = the current revision's (the phase A "
        "fast path and the opt-in shared-context switch, off by default, on the BF16-resident backbone with "
        "Transformers remote code and the forward token budget), unchanged.",
        "card": "the product card of the current revision with this candidate's reports, an Index input built by "
        "python -m v2.release.card_index (board-served parameter counts, the audited footnote) with the 27B point "
        "from the Index run on exactly these weights and every other tier at its current main, and assets rendered "
        "on the release node by the default v2.release.card_assets (banner concept A) in the card render "
        "environment of the earlier rounds; card.speed keeps the current revision's phase A bench receipt (same "
        "base, runtime and shapes).",
        "replaces_spec": {
            "spec": "v2/release/specs/dev2-27b-ras.json",
            "sha256": sha(BASE_SPEC),
        },
        "previous": old["_release"],
    }
    s["checkpoint"] = p["checkpoint"]
    s["expected_identity"] = {"model_sha256": identity}
    s["vendor_source"] = (
        f"/data/dev2/src/{vendor(arm)}-src_training_decision2/src/training/decision2"
    )
    s["gate_receipt"] = f"{DECISIONS}/{NAME}.decision.27bs-{arm}.json"
    s["origin"]["summary"] = (
        f"A PEFT LoRA adapter (rank {lora['rank']}, alpha {lora['alpha']}, on all 496 text projections of the "
        "attention, gated-delta and MLP blocks) and a native candidate head trained on the frozen Qwen3.8-27B text "
        "backbone at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0), which is not redistributed. The adapter "
        f"is the exact weighted soup of {lora['members']} seeds of rank-{lora['member_rank']} recipes ({p['seeds']}; "
        "their LoRA factors stacked along the rank with each up-projection scaled by its seed's weight, "
        f"{p['weights']}, so the weighted mean of the seeds' weight updates is reproduced exactly), with the heads "
        "averaged by the same weights."
    )
    s["frozen_autotune_cache"] = {
        "path": f"{p['run']}/triton-cache",
        "tree_sha256": post["post_sha256"],
        "post_record": f"{p['run']}/triton-cache.post.json",
        "origin": f"the formal run's persisted cache: a fresh copy of F1's scored post-run cache (tree "
        f"{post['pre_sha256'][:8]}...); the mlx-diag run copied the same tree; release runs copy it with "
        "python3 -m v2.27b.triton_cache copy --expect <tree_sha256>",
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


def disclosures(arm: str, high: float, low: float, sets: dict) -> list[str]:
    mixes = set(ARMS[arm]["mixes"])
    families = "IB1-r3 families matched to Index benchmark families and formats"
    if mixes & IB2_MIXES:
        families += (
            " and IB2's (hover and gsm2 are in-distribution for HoVer and GSM8K)"
        )
    if mixes & IB4_MIXES:
        families += " and IB4 phase 1's (isarc2 is in-distribution for iSarcasmEval; sqa2 is RAGTruth-style)"
    out = [
        "v3 against the current revision is a reference under the Index-first rule (v3 "
        + (
            "significantly below"
            if high < 0
            else (
                "significantly above" if low > 0 else "not significantly different from"
            )
        )
        + " the current revision)",
        f"the training rows include the {families}; the internal records report the transfer-only Index delta "
        "without those benchmarks (private values)",
        "Index contamination audit of every training file: "
        + "; ".join(
            f"{n} ({MIXES[n][:8]}) {v['training_lines']} lines, {v['item_rows']} item rows and "
            f"{v['duplicate_rows']} familiar-text rows"
            for n, v in sorted(sets.items())
        ),
    ]
    if "M9-IB-lrh" in ARMS[arm]["seeds"]:
        out.append(
            f"the half-LR seeds train the released recipes at {HALF}; every other setting is unchanged"
        )
    if mixes - {"a20ib1", "a20ib12"}:
        out.append(
            "the overlap-exposure reference covers the A20, IB1-r3 and IB2 rows; the IB4 phase 1 rows are covered "
            "by their own release-safe records and the ML block only repeats A20 rows; every file is in the "
            "row-level Index audit"
        )
    out.append(
        "card.speed is the current revision's phase A bench receipt (same base, runtime and shapes; this adapter's "
        "rank differs)"
    )
    return out


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
    assert set(sets) == set(p["mixes"]), sorted(sets)
    mixes = set(p["mixes"])
    rows = "A20 rows, the IB1-r3"
    if mixes & IB2_MIXES:
        rows += ", IB2"
    if mixes & IB4_MIXES:
        rows += " and IB4 phase 1 (C1 recheck r3 PASS)"
    rows += " training rows"
    if mixes & ML_MIXES:
        rows += " and the ML block (copies of A20's own rows)"
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
        "action": f"New main revision of the public repository {REPO}: {what(arm)}, weights {p['weights']}; "
        f"replaces the M6-IBxIB2-m50 adapter ({CURRENT['weights_identity'][:8]}, revision "
        f"{CURRENT['revision'][:8]}) and keeps its runtime (the phase A fast path and the opt-in shared-context "
        "switch, off by default) and product card (default generator, banner concept A) with this candidate's "
        "Index input; then the superseded weight blobs are purged with rewrite_history=False (hf_headroom.sh "
        "first). The repository stays public and in the public Decision 2.0 collection.",
        "rationale": "Index-first rule (user 2026-10-02 09:55 UTC+8) and progressive release (user 2026-10-03 "
        "01:27 UTC+8). IF1: the private Index paired bootstrap of these exact weights minus the current revision's "
        "(M6-IBxIB2-m50) has a 95% lower bound > 0 (2,000 replicates, seed 20261002; values in private files only; "
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
        f"Apache-2.0, pinned and not redistributed; the {rows} are licence-clean (release-safe data records); the "
        "package keeps the Apache-2.0 LICENSE (product card: no NOTICE, attributions or evaluation/ files).",
        "supersedes": {
            "revision": CURRENT["revision"],
            "final_sha256": CURRENT["decision_sha256"],
            "gate_sha256": CURRENT["gate_sha256"],
            "manifest_sha256": CURRENT["manifest_sha256"],
            "weights_identity": CURRENT["weights_identity"],
        },
        "disclosures": disclosures(arm, high, low, sets),
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
        f"dev2-27b-27bs-{args.arm}.json": (spec_text, SPECS),
        f"{NAME}.decision.27bs-{args.arm}.json": (decision_text, RECORD),
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
