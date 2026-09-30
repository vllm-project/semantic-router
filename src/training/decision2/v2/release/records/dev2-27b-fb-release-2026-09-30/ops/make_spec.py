"""Specs and decisions for DEV2.0-27B = 27B M4b F-b (full fine-tune), stored as bf16z.

Coordinator notes 2026-09-30 06:10 (F-b release path) and 07:15 (lossless bf16z storage; CAL698 rebound to the
released copy). The card spec is derived from the current card-c1 spec (card peers, roster, banner, licence files);
the weights, recipe, scores, disclosures and C1 lines are F-b's.

  --stage draft : specs/dev2-27b-fb-draft.json (bf16z package, successor items R1-R7) and
                  specs/dev2-27b-fb-c1pkg.json (the same weights as a plain package: the C1 frozen package),
                  plus the draft decision
  --stage final : specs/dev2-27b-fb-release.json (items R1-R8) and the final decision; needs the C1 summary
                  copied to records/dev2-27b-fb-release-2026-09-30/c1/SUMMARY.json

Run from src/training/decision2:  python3 v2/release/records/dev2-27b-fb-release-2026-09-30/ops/make_spec.py --stage S [--check]
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

SPECS = Path("v2/release/specs")
OUT = Path("v2/release/records/dev2-27b-fb-release-2026-09-30")
CARD_PASS = Path("v2/release/records/dev2-c1-card-pass-2026-09-29")
IN = "/data/dev2/runs/release/inputs/dev2-27b-fb"
M4B = "/data/dev2/runs/27b/m4b"
MIRROR = "/data/dev2/src/fcd616036495dd34efda0a37bff3b5127b3d7f73-src_training_decision2/src/training/decision2"
SCORED_MIRROR = "/data/dev2/src/ee13316871f8f3e56d8a49edc493b251d45db158-src_training_decision2/src/training/decision2"
DECISIONS = "/data/dev2/runs/release/decisions"
CURRENT = "c0dba600087f582a1033830a2276d22e96e6324c"
BF16_IDENTITY = "48ab177b2794b991a48c94a99ff53aa0db70505104e86c4a43ed031a6ba67784"
IDENTITY = "d3460f3d4ebd4d04b07d3e4ec310a1a58d1da550302b512e8726db8190ae2e71"
SCORED_IDENTITY = "5bcfa3d4bd1f83c5f4f999f9b8b68a01bb2bef69020c85661a2a937d9ecb02f9"
LOADED = "25,629,863,936"
PREPARED_BY = "Decision 2.0 release engineering, ~27B release worker (worktree vllm-sr-dev2-release-27b)"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: F-b approved as the DEV2.0-27B successor in the cross-track "
    "note of 2026-09-30 06:10 UTC+8 (items 1-7 pass; item 4 on the card-eligible Choice + Noul reading), released in "
    "bf16z storage with CAL698 rebound to the released copy (coordinator 2026-09-30 07:15), item 8 by the C1 "
    "post-key guard; under the user's full-autonomy mandate"
)
FILES = {
    # node paths -> sha256 (gathered on node B 2026-09-29 ~23:50Z)
    "paired": (
        f"{M4B}/F-b/gates/paired-vs-f1.json",
        "6194dc230a3b4f57c00faec9c867efecf9022934b3a3039e264526388a9189d1",
    ),
    "types": (
        f"{M4B}/F-b/gates/types-cand.json",
        "3e1d3f100c88e97b7dffc8ae9cebc6527521126cdf23423244b02d4164989d3e",
    ),
    "mlx_paired": (
        "/data/dev2/runs/27b/m4b-mlx-pair/card/F-b-vs-F1.json",
        "49901c1b15fa166161174bb8fb4031b3181a61aa96e9b84bdf7cfddb15863d68",
    ),
    "exposure": (
        f"{M4B}/F-b/overlap/exposure.json",
        "dbdc9393242820d8907a4c2aaf307a8d66fa667de7b6e69142d08907a1642e27",
    ),
    "public231": (
        f"{M4B}/F-b/gates/public231-vs-F1.json",
        "c8a64a231341d7b8c98b14fd1366400c25d2580501194f5766ae123d3c215040",
    ),
    "tier_paired": (
        f"{M4B}/F-b/gates/paired-vs-autojev27.json",
        "6d56dd95ca1afe245515359df8d5d196ce4cd7753eb27144abcb33b978ff383b",
    ),
}
C1_SEALED = (
    "**Independent sealed confirmation (previous revision)** — JevArena-C1 v1.2 (2,840 human-labeled items from 8 "
    "sources published after the relevant cutoffs, never used for training or development; scored once): the previous "
    "DEV2.0-27B weights (release package `5683c6f0`, a LoRA adapter, identity `b7fd44e3`) score **57.33** vs AutoJev-27B "
    "58.17 (−0.84, 95% CI [−2.50, +0.69]) and Eikos-27B 59.37 (−2.04, 95% CI [−3.66, −0.44]). The current weights were "
    "not part of any sealed event."
)
STORAGE_NOTE = (
    "**Storage:** the weights are stored losslessly compressed (`bf16z`): each tensor's bytes are split into byte planes "
    "(the sign / exponent and mantissa bytes of BF16, all four bytes of FP32) and each plane is zstd-compressed. The "
    "loader restores the exact BF16 / FP32 safetensors files, checks every restored file's SHA-256 and the model "
    "identity, and caches them in `$DECISION2_CACHE` (default `~/.cache/decision2`; about 54 GB) on first load. "
    "`Decision2.from_pretrained` does this automatically; `python -m decision2.bf16z restore --package .` (run in the "
    "download) prints the restored plain checkpoint directory for other tooling (`backbone/` then loads with "
    "Transformers)."
)
LOADING_SNIPPET = (
    "**Loading:** `import sys; from huggingface_hub import snapshot_download; path = "
    'snapshot_download("llm-semantic-router/DEV2.0-27B"); sys.path.insert(0, path); from decision2 import Decision2; '
    "model = Decision2.from_pretrained(path)`: this downloads the compressed package (about 35 GB), restores the exact "
    "weights once and checks them before loading. No base download is needed: the package holds the full fine-tuned "
    "text backbone."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipts() -> dict:
    return {
        name: OUT / "inputs" / name
        for name in (
            "bf16-copy.json",
            "bf16z.json",
            "bf16z-verify.json",
            "calibration.public.json",
        )
    }


def c1_postkey_line(summary: dict | None) -> str:
    if summary is None:
        return ""
    rule = summary["item8"]
    low, high = rule["ci95"]
    return (
        f"**JevArena-C1 v1.2, post-key (not an independent validation):** {summary['c1']:.2f} on this revision vs 57.33 "
        f"for the previous revision ({rule['delta']:+.2f}, 95% CI [{low:+.2f}, {high:+.2f}]; 2,840 items, paired by "
        "source group). The independent sealed confirmation above measured the previous revision."
    )


def card_text(old: dict, summary: dict | None) -> dict:
    text = copy.deepcopy(old["card"]["text"])
    text["architecture"] = (
        "All 25.62B parameters of the 64-layer Qwen3.8-27B text backbone (48 gated-delta and 16 attention layers) are "
        "fine-tuned; the model reads the state, the question and every supplied candidate once, and a shared candidate "
        "head scores the candidates against a global query and returns probabilities without generating text."
    )
    text["runtime_note"] = (
        "The download includes a small local runtime (`decision2/`) for one CUDA or ROCm GPU; it restores the "
        "losslessly compressed weights once and does not start a hosted endpoint."
    )
    confirmation = C1_SEALED
    line = c1_postkey_line(summary)
    if line:
        confirmation += "\n\n" + line
    text["confirmation"] = confirmation
    text["comparator_note"] = (
        "The three 27B peers were rerun on the same kernel image as this model. Eikos-27B is shown from its BF16 weights "
        "(`caiovicentino1/Eikos-27B`), the sibling of the FP8 build listed on the Decision Index; its card declares MIT "
        "for the authors' contributions on the Apache-2.0 base. AutoJev-27B declares Apache-2.0 weights and MIT code, "
        "Jebadiah-27B Apache-2.0. Invalid answers count as failures: none for this model; 13, 4 and 153 transfer answers "
        "for AutoJev-27B, Eikos-27B and Jebadiah-27B, and 36 public-231 answers for Jebadiah-27B."
    )
    text["training"] = [
        "**Recipe:** a full-parameter fine-tune of Qwen3.8-27B (all 25,624,600,064 text-backbone parameters) plus a new "
        "candidate head, on exactly the previous revision's data. FSDP over three GPUs with FP32 weights, gradients and "
        "AdamW states and BF16 compute; one pass of cross-entropy plus 0.5 × Brier over the 10.0M-token mixture below "
        "(inputs up to 4,096 tokens, gold labels only), learning rates backbone 1e-5 / head 1e-4 with 10% linear warmup "
        "and cosine decay, seeds 20260926 and 20260928; the exact uniform FP32 soup of the two seeds' final checkpoints "
        "(both were their seed's best on the selection set).",
        old["card"]["text"]["training"][1],
        "**Capacity, not data:** trained on the same rows and tokens as the previous revision (a rank-16 LoRA soup), "
        "full fine-tuning raised typed-reasoning accuracy by 12.0 points (95% CI +9.4 to +14.6; constraint competition "
        ".570 → .980, exception stack .788 → .828, resource ledger .793 → .823) with human transfer level (−0.7 points, "
        "95% CI −7.3 to +5.0).",
        old["card"]["text"]["training"][3],
    ]
    text["limitations"] = [
        "**Level with AutoJev-27B.** Post-key JevArena v3 is level with AutoJev-27B (−0.45, 95% CI [−3.68, +3.98]): typed "
        "accuracy is slightly higher (T 0.9075 vs 0.8869, +0.021, 95% CI [−0.001, +0.042]) and human transfer slightly "
        "lower (H 0.566 vs 0.587, −0.020, 95% CI [−0.069, +0.046]); neither difference is significant. Against the other "
        "27B models: Eikos-27B +2.39 (95% CI [−0.88, +7.28], not significant) and Jebadiah-27B +6.21 (95% CI [+2.81, "
        "+10.19]).",
        "**Against the previous revision:** JevArena v3 +4.47 (95% CI [+0.19, +8.13]); human transfer level (−0.007, 95% "
        "CI [−0.073, +0.050]).",
        "**JevBench public 231:** 197 correct (48 / 71 / 78), level with the previous revision's 198 (−1, 95% CI [−9, "
        "+7]) and AutoJev-27B's 201 (−4, 95% CI [−13, +5]), above Jebadiah-27B's 176 (+21, 95% CI [+9, +32]), but "
        "significantly below Eikos-27B's 212 (−15, 95% CI [−24, −6]; hard tier 78 vs 92), mostly in two hard-tier skills "
        "that neither JevArena nor this model's training covers: applying long policy documents with amendments and "
        "precedence (9 vs 15 of the 19 such items) and checking a quoted person's plausible but wrong conclusion against "
        "the evidence instead of adopting it.",
        "**JevArena-C1:** the independent sealed comparison measured the previous revision: level with AutoJev-27B and "
        "significantly below Eikos-27B (mainly on Choice). The post-key C1 line above measures this revision against "
        "the previous one only.",
        "**Multilingual.** On the card-eligible parts of the mlx-diag diagnostic (Choice and Noul in seven languages) "
        "the macro accuracy is 81.2% vs 82.4% for the previous revision (−1.2 points, 95% CI [−2.9, +0.4], not "
        "significant): non-English Choice rose (81.2% vs 79.1%, +2.1, 95% CI [+0.5, +3.7]) while non-English Noul "
        "(paraphrase pairs) fell (80.3% vs 85.5%, −5.2, 95% CI [−8.3, −2.0]); the weakest language is Korean (74% vs "
        "79%, 100 items). The 27B peers' non-English Noul is 88.7% (AutoJev-27B), 85.0% (Eikos-27B) and 85.3% "
        "(Jebadiah-27B). An internal multilingual Score diagnostic, not shown here, also declined.",
        "**Score levels.** On typed Score it uses all five levels with a spread close to the gold labels (29 / 80 / 100 "
        "/ 73 / 118 answers vs 35 / 79 / 83 / 75 / 128).",
        "**Calibration.** Per-type temperatures fitted on the clean 698-item calibration partition (Choice 0.522, Noul "
        "0.605, Score 0.05) are applied (`calibration.json`): they improved every development calibration measure. "
        "Typed Brier 0.061 and ECE 0.021.",
        "**Input limit.** 32,768 tokens for the complete state, question and candidates; longer inputs return an "
        "over-budget error instead of being truncated. Training inputs were at most 4,096 tokens.",
        "**Seed soup.** Only the uniform soup of two seeds was evaluated on JevArena v3 and is released.",
        "**Memory and disk.** One GPU with at least 120 GB of memory (the evaluation peaked at 107.6 GB allocated, 116.5 "
        "GB reserved); about 35 GB of download plus about 54 GB for the restored weights. CPU inference was not verified.",
        "**Evaluation familiarity.** A screen of the program's training pools against the reported evaluation panels "
        "found no exposure for this model's training data.",
    ]
    text["details"] = [
        "**Weights:** the exact uniform soup of two full-parameter fine-tunes of Qwen3.8-27B at "
        "`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (seeds 20260926 and 20260928), stored with the 496 Linear projection "
        "matrices in BF16 exactly as the runtime's BF16 autocast rounds them and every other tensor (embedding table, "
        "norms, gated-delta parameters, convolution filters, decision head) in FP32; answers are identical to the "
        "evaluated FP32 checkpoint on every scored prompt.",
        STORAGE_NOTE,
        LOADING_SNIPPET,
    ]
    return text


def build(stage: str) -> dict[Path, str]:
    old = json.loads((SPECS / "dev2-27b-card-c1.json").read_text(encoding="utf-8"))
    r = receipts()
    summary_path = OUT / "c1" / "SUMMARY.json"
    summary = json.loads(summary_path.read_text()) if stage == "final" else None
    spec = copy.deepcopy(old)
    for key in ("base", "frozen_autotune_cache"):
        spec.pop(key, None)
    decision_path = (
        f"{DECISIONS}/DEV2.0-27B.decision.fb{'' if stage == 'final' else '-draft'}.json"
    )
    spec.update(
        {
            "profile": "qwen-full",
            "checkpoint": f"{IN}/bf16z-public",
            "expected_identity": {"model_sha256": IDENTITY},
            "bf16z": {
                "receipt": f"{IN}/bf16z.json",
                "sha256": sha(r["bf16z.json"]),
                "verify": f"{IN}/bf16z-verify.json",
                "verify_sha256": sha(r["bf16z-verify.json"]),
            },
            "calibration": {
                "path": f"{IN}/cal698-public/calibration.public.json",
                "sha256": sha(r["calibration.public.json"]),
            },
            "vendor_source": SCORED_MIRROR,
            "runtime_source": MIRROR,
            "gate_receipt": decision_path,
        }
    )
    spec["origin"] = {
        "repo_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "relation": "finetune",
        "summary": (
            "A full-parameter fine-tune of the Qwen3.8-27B text backbone at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` "
            "(Apache-2.0) with a native candidate head: the exact uniform FP32 soup of two seeds of one recipe, stored "
            "with the Linear projection matrices rounded to BF16 as the runtime's autocast rounds them, and "
            "losslessly compressed (bf16z)."
        ),
    }
    spec["runtime_requirements"] = {
        k: v
        for k, v in old["runtime_requirements"].items()
        if k not in ("peft", "huggingface_hub")
    } | {"numpy": "2.3.5", "zstandard": "0.25.0 (bf16z restore)"}
    spec["runtime_equivalence"] = (
        "decision2/qwen.py restores the bf16z weight files (every restored file's SHA-256 recorded in "
        "MODEL_MANIFEST.json storage, and the restored checkpoint's identity checked against the scored identity chain), "
        "then loads the full checkpoint with the training/model sources vendored from the scored run's own source "
        "mirror (ee1331687), whose SHA-256 equal the scored adapter sources (checked at build time), and applies the "
        "per-item batching, BF16-backbone / FP32-head execution, the CAL698 per-type temperatures of calibration.json "
        "and the answer normalization of training.model.infer as run by v2.27b.typed_collect_kernel. The scored "
        f"checkpoint ({SCORED_IDENTITY[:8]}) stored every tensor in FP32; this package (v2.release.bf16_copy, identity "
        f"{BF16_IDENTITY[:8]}, then decision_config.json's soup-member provenance paths trimmed to track-relative form, "
        f"identity {IDENTITY[:8]}) stores its 496 Linear projection matrices in BF16 exactly as BF16 autocast rounds them "
        "and every other tensor bit for bit in FP32; the CAL698 fit repeated on it reproduced all 698 calibration logits "
        "exactly. Checked on one GPU of the scoring node against the scored predictions of every formal prompt "
        "(typed-final 1,600, css15 6,547, public231 231) and of the mlx-diag diagnostic (2,275) by release.sh --parity, "
        "each with a fresh copy of the persisted Triton autotune cache of the run that scored it."
    )
    spec["scored"] = {
        "label": "post-key same-panel run m4b/F-b/formal with CAL698 temperatures (kernel path, frozen autotune cache, 32,768-token limit)",
        "report_sha256": "6895fee43e7ae1bca09ce72ad68a08ca6da2424ba5bd5ad6f6dd1ea440842d18",
        "seal_sha256": "03f5b49f66c8182daf22ebf43aa1057f816c23174f2a745631bb71da36ba6b56",
        "predictions_sha256": {
            "typed-final": "c838e9cf5d23adca5760019f6a61f6a286b6d894772a113ab9ad39518422e851",
            "css15": "8e3528d8daa6c405ecec1e4e4fe97054414b7af7f062b83a21c2b4df1abc01ed",
            "public231": "3f9a9e7de7e0aa683dc60d546619b1143685eb5b53236359670ce033a9f9dc14",
            "mlx-diag": "81fba7701bac3452ce380f3e9c51c39af4290f7e77b9320dc22e47c779e874d9",
        },
        "paired_sha256": FILES["tier_paired"][1],
        "native_manifest": f"{M4B}/F-b/formal/output/typed-final.predictions.jsonl.manifest.json",
    }
    lic = copy.deepcopy(old["licence"])
    lic["components"] = [
        {
            "component": "DEV2.0-27B weights (full fine-tune of Qwen3.8-27B), decision head, package runtime, card and artwork",
            "licence": "apache-2.0",
            "source": "llm-semantic-router/DEV2.0-27B",
        },
        {
            "component": "Qwen3.8-27B (direct weight origin of the fine-tuned backbone and tokenizer)",
            "licence": "apache-2.0",
            "source": "Qwen/Qwen3.8-27B@1d4bf0f2",
        },
    ]
    lic["attributions"] = [
        "[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: the "
        "starting weights of this full fine-tune and its tokenizer (Apache-2.0, `LICENSES/Qwen3.8-27B-LICENSE.txt`).",
        old["licence"]["attributions"][1].replace(
            "Training data of this adapter", "Training data of this model"
        ),
        *old["licence"]["attributions"][2:],
    ]
    spec["licence"] = lic
    card = spec["card"]
    card["reports"][0].update(
        {"report": f"{M4B}/F-b/formal/REPORT.json", "mlx": f"{IN}/mlx-diag.score.json"}
    )
    card["paired"] = FILES["tier_paired"][0]
    card["paired_peers"] = {
        "eikos27": f"{M4B}/F-b/gates/paired-vs-eikos27b.json",
        "jebadiah27": f"{M4B}/F-b/gates/paired-vs-jebadiah27b.json",
    }
    card["calibration_text"] = (
        "per-type temperatures fitted on the clean 698-item calibration partition (Choice 0.522, Noul 0.605, Score "
        "0.05; `calibration.json`), adopted because they improved every development calibration measure."
    )
    card["requirements_text"] = (
        "Tested on one AMD MI325X GPU (ROCm 7.2) with Python 3.12, PyTorch 2.12 and Transformers 5.17 (BF16 backbone, "
        "FP32 head), plus numpy and zstandard to restore the compressed weights. The evaluation peaked at 107.6 GB of "
        "allocated GPU memory (116.5 GB reserved); no evaluated input reached the 32,768-token limit. GPU answers match "
        "the evaluated predictions with the flash-linear-attention 0.5.2 and causal-conv1d 1.7.0 kernels and a "
        "persisted Triton autotune cache; without those kernels Transformers uses its reference implementation of the "
        "same operations, which is slower and differs slightly in the last digits."
    )
    card["text"] = card_text(old, summary)
    names = ["R1-R7"] + (["R8"] if summary else [])
    profile = {
        "name": "successor",
        "run": f"{M4B}/F-b/formal",
        "current": {
            "revision": CURRENT,
            "gate": f"{MIRROR}/v2/release/records/dev2-c1-card-pass-2026-09-29/27b/receipts/gate.json",
            "decision": f"{MIRROR}/v2/release/records/dev2-c1-card-pass-2026-09-29/DEV2.0-27B.decision.json",
            "run": "/data/dev2/runs/27b/M3-A-soup/formal",
            "mlx_predictions": "/data/dev2/runs/27b/m3-f2/mlx-diag/M3-A-soup/output/mlx-diag.predictions.jsonl",
        },
        **{
            k: FILES[k][0]
            for k in ("paired", "types", "mlx_paired", "exposure", "public231")
        },
        "tier": {
            "reference": "autojev27",
            "v3_share": 0.9,
            "paired": FILES["tier_paired"][0],
            "no_own_1_0": True,
        },
    }
    if summary:
        profile["c1_postkey"] = f"{IN}/c1/SUMMARY.json"
    spec["gate_profile"] = profile
    spec["_release"] = {
        "candidate": "27B M4b F-b (A1-soup: full-parameter fine-tune, gold, two-seed FP32 soup); coordinator 2026-09-30 06:10 (successor, items 1-7) and 07:15 (bf16z storage, CAL698 rebound).",
        "gate": f"successor profile ({', '.join(names)}) against the current revision {CURRENT[:8]}; no Decision 1.0 at this size, so the tier gates and the card compare with AutoJev-27B.",
        "storage": "checkpoint = the bf16z compression of the v2.release.bf16_copy of the scored FP32 soup, with decision_config.json's two soup-member paths trimmed to track-relative form (identity chain 5bcfa3d4 -> 48ab177b -> d3460f3d; receipts inputs/bf16-copy.json and inputs/redaction.json); the builder derives the identity from the restored files' SHA-256 (compression and verify receipts pinned).",
        "previous": old["_release"],
    }
    evidence = {
        "paired": FILES["paired"][1],
        "types": FILES["types"][1],
        "mlx_paired": FILES["mlx_paired"][1],
        "exposure": FILES["exposure"][1],
        "public231": FILES["public231"][1],
        "tier_paired": FILES["tier_paired"][1],
        "current_gate": sha(CARD_PASS / "27b" / "receipts" / "gate.json"),
        "current_decision": sha(CARD_PASS / "DEV2.0-27B.decision.json"),
    }
    if summary:
        evidence["c1_postkey"] = sha(summary_path)
    decision = {
        "schema": "dev2-release-decision/1",
        "status": "final" if summary else "draft",
        "decision": "release",
        "model_name": "DEV2.0-27B",
        "repo_id": "llm-semantic-router/DEV2.0-27B",
        "identity": {"model_sha256": IDENTITY},
        "report_sha256": spec["scored"]["report_sha256"],
        "paired_sha256": FILES["tier_paired"][1],
        "gate_profile": "successor",
        "current_revision": CURRENT,
        "evidence_sha256": dict(sorted(evidence.items())),
        "prepared_by": PREPARED_BY,
        **(
            {"decided_by": DECIDED_BY, "decided_utc": "2026-09-29T23:15:00Z"}
            if summary
            else {}
        ),
        "rationale": (
            "F-b is the successor of DEV2.0-27B (coordinator 06:10): post-key v3 71.684, +4.47 [+0.19, +8.13] vs the "
            "current revision's scored run; human transfer −0.007 [−0.073, +0.050]; no type collapsed; card-eligible "
            "mlx-diag −0.012 [−0.029, +0.004]; v3 ≥ 90% of AutoJev-27B and H vs AutoJev-27B [−0.069, +0.046]; no overlap "
            "exposure; public 231 197 vs 198 (OK)"
            + (
                f"; C1 post-key {summary['c1']:.2f} vs 57.33 ({summary['item8']['delta']:+.2f}, "
                f"[{summary['item8']['ci95'][0]:+.2f}, {summary['item8']['ci95'][1]:+.2f}], {summary['item8']['verdict']})"
                if summary
                else "; item 8 (C1 post-key) pending"
            )
            + ". Released as qwen-full with the BF16 storage copy (exact parity) losslessly compressed as bf16z, and the "
            "CAL698 temperatures adopted under the 23:15 rule, refitted on the copy with identical logits."
        ),
        "calibration": "CAL698 adopted (23:15 rule: typed-DEV Brier .035 -> .024, ECE .091 -> .023; CSS-pilot Brier .624 -> .591, ECE .112 -> .081); temperatures Choice 0.522, Noul 0.605, Score 0.05, refitted on the BF16 copy (identity 48ab177b; logits identical to the scored fit)",
        "supersedes": {
            "final_sha256": evidence["current_decision"],
            "released_as": f"llm-semantic-router/DEV2.0-27B@{CURRENT}",
            "released_gate_sha256": evidence["current_gate"],
        },
        "revision_binding": "This decision names no revision; receipts/gate.json of the release.sh --upload --collect --already-collected run binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
    }
    out = {}
    if stage == "draft":
        out[SPECS / "dev2-27b-fb-draft.json"] = spec
        plain = copy.deepcopy(spec)
        plain.pop("bf16z")
        plain["checkpoint"] = f"{IN}/bf16-public"
        plain["runtime_equivalence"] = spec["runtime_equivalence"].replace(
            "decision2/qwen.py restores the bf16z weight files (every restored file's SHA-256 recorded in "
            "MODEL_MANIFEST.json storage, and the restored checkpoint's identity checked against the scored identity "
            "chain), then loads",
            "decision2/qwen.py loads",
        )
        plain["_release"] = dict(
            spec["_release"],
            c1_package="The same weights as a plain (uncompressed) qwen-full package: the frozen package of the C1 post-key guard (item 8); never uploaded.",
        )
        out[SPECS / "dev2-27b-fb-c1pkg.json"] = plain
        out[OUT / "DEV2.0-27B.decision.draft.json"] = decision
    else:
        out[SPECS / "dev2-27b-fb-release.json"] = spec
        out[OUT / "DEV2.0-27B.decision.json"] = decision
    return {
        path: json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        for path, value in out.items()
    }


def main() -> int:
    stage = sys.argv[sys.argv.index("--stage") + 1]
    check = "--check" in sys.argv
    problems = []
    for path, text in build(stage).items():
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                problems.append(f"{path} differs from the derivation")
        else:
            path.write_text(text, encoding="utf-8")
        print(path, hashlib.sha256(text.encode()).hexdigest())
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
