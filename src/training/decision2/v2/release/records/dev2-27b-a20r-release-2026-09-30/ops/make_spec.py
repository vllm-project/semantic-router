"""Release spec and decisions of the DEV2.0-27B successor (27B Milestone 4, M4-A20r soup).

Coordinator note 2026-09-30 07:20 UTC+8: A20r is THE 27B successor; F-b is superseded. It is released as the new
`main` of llm-semantic-router/DEV2.0-27B (profile qwen-adapter, rank 64, T = 1) under the successor gate profile at a
size without Decision 1.0 (items R1-R7 vs the current revision c0dba600, plus R8 = the C1 post-key guard). The spec
is derived from the current revision's spec (specs/dev2-27b-card-c1.json): peers, roster, banner, base, licence
files and runtime requirements carry over; the model, scored run, evidence and card text are A20r's (27B record
v2/27b/records/m4-results-2026-09-29.md; node A copies of node B's files, hash-equal).

Run from src/training/decision2:
  python3 v2/release/records/dev2-27b-a20r-release-2026-09-30/ops/make_spec.py draft
  python3 v2/release/records/dev2-27b-a20r-release-2026-09-30/ops/make_spec.py final --c1-summary SUMMARY.json

draft: the spec before item 8 (C1 line pending, no R8) and its draft decision, for the frozen package that C1
scores. final: the C1 post-key line and R8 from the collected SUMMARY.json (a local copy of the node A file), and
the final decision. Decisions are written next to this script and installed on node A under
/data/dev2/runs/release/decisions/.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent
SPECS = Path("v2/release/specs")
BASE_SPEC = SPECS / "dev2-27b-card-c1.json"
OUT_SPEC = SPECS / "dev2-27b-a20r-release.json"
DRAFT_SPEC = SPECS / "dev2-27b-a20r-release.draft.json"
DECISIONS = Path("/data/dev2/runs/release/decisions")
R = "/data/dev2/runs/27b"
RUN = f"{R}/M4-A20r-soup/formal"
MIRROR_RUNTIME = "/data/dev2/src/c68de36a5245c4e14eccc7b3d43279b75affbc51-src_training_decision2/src/training/decision2"
MIRROR_SCORED = "/data/dev2/src/ff660322a76b04c4774fdb6aaa31e5af2a45743e-src_training_decision2/src/training/decision2"
CARD_PASS = f"{MIRROR_RUNTIME}/v2/release/records/dev2-c1-card-pass-2026-09-29"
CURRENT_REVISION = "c0dba600087f582a1033830a2276d22e96e6324c"
IDENTITY = "2e07451107a2ca9735064cc01f7e43e8522b4bae48f2d77bdf29f7ea2fe6362c"
LOADED = 26096775168
SHA = {
    "report": "e9a7c4700bd97bb78efa1c28117aa2a8a7a92fc2b3a29ead6d4f7e9c7f49b600",
    "seal": "9d60e61171da3ea477ab59e1f5303f0bcbc560a5964e8a7ebe13e61855911611",
    "typed-final": "ea573b2d50be50efd3048ee427309232b1f9b998b98d9997644bd82a51442d0e",
    "css15": "a5ed828a9b6a839b765643362a08a95386312bf4fdaf4c33e2f6d97a28337fbb",
    "public231": "6746b61a58603418d43ce22109d08b79d9cf3bab63aa14acca6cd42c5dcf131d",
    "mlx-diag": "1ab3487d9b7b9368450b003697692891ed6947cad18c2876b9626c96fb0d87e4",
    "paired_autojev": "09ff4b0f872b60fe665124440b111afe03575d44a458c8743446a0ca7d0200c6",
    "types": "de356f39813028200989985b8184c91e912799d4cf22e8f763722ef16e2b2e3f",
    "cache": "f474e2e9dbdf3a2900465bce74256326ba2756250861b45359742685982cb5f7",
}
# gate.evidence_sha256 of the profile, from the node A files (checked again by the builder and at seal).
EVIDENCE = {
    "current_decision": "fd1b5ef0643532407b27a4f7f8d6748e8f8f9fa27b627f5b285b8ddb41d3e008",
    "current_gate": "e6965d75e1f7cac175522f19fbabedd333ad2e8136f8e69a0eb9fb7b200b4a3f",
    "exposure": "084781a6a0b47fb1598b33432f4d43be8cee8af2af63b44417bc9f087e69ede4",
    "mlx_paired": "f1351c070025d3d18c1eb49c59ea897796c03242881268e2e41438c3b45e1ea5",
    "paired": "9e77de58c9a8caa96201514a4864d9f3838c7cb0e95a5e2ef954d8f266ec5f37",
    "public231": "bf20dc42932f6e1aeeea05b29458b8bb57b12dde9622c8a35300528b0e9f9632",
    "tier_paired": SHA["paired_autojev"],
    "types": SHA["types"],
}
PREPARED_BY = "Decision 2.0 release engineering, 27B successor worker (worktree vllm-sr-dev2-release)"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: cross-track note of 2026-09-30 07:20 UTC+8 (A20r is THE "
    "27B successor; release it as DEV2.0-27B main after the C1 post-key guard, item 8), under the user's "
    "full-autonomy mandate"
)

SEALED_C1 = (
    "**Independent sealed confirmation** \u2014 JevArena-C1 v1.2 (2,840 human-labeled items from 8 sources published "
    "after the relevant cutoffs, never used for training or development; scored once) measured the previous "
    "revision (release package `5683c6f0`, weights identity `b7fd44e3`): 57.33 vs AutoJev-27B 58.17 (\u22120.84, "
    "95% CI [\u22122.50, +0.69]) and Eikos-27B 59.37 (\u22122.04, 95% CI [\u22123.66, \u22120.44]); it has not been "
    "repeated for this revision. No Decision 1.0 model exists at this size."
)


def signed(x: float) -> str:
    return f"{x:+.2f}".replace("-", "\u2212")


def postkey_line(summary: dict | None) -> str:
    if summary is None:
        return (
            "**JevArena-C1 v1.2, post-key (not an independent validation):** pending for this revision (successor "
            "rule item 8)."
        )
    rule = summary["item8"]
    lo, hi = rule["ci95"]
    return (
        f"**JevArena-C1 v1.2, post-key (not an independent validation):** {summary['c1']:.2f} on this revision "
        f"vs {summary['c1'] - rule['delta']:.2f} for the previous revision ({signed(rule['delta'])}, 95% CI "
        f"[{signed(lo)}, {signed(hi)}]; {summary['items']:,} items, paired by source group). The independent "
        "sealed confirmation above measured the previous revision."
    )


TRAINING = [
    "**Recipe:** a rank-32 LoRA (alpha 64, dropout 0.05) on all 496 text projections of the frozen Qwen3.8-27B base "
    "plus a new candidate head, one pass of cross-entropy plus 0.5 \u00d7 Brier over 25.0M tokens of decision rows "
    "(inputs up to 4,096 tokens), learning rates LoRA 2e-5 / head 1e-4, seeds 20260926 and 20260928; each seed's "
    "checkpoint chosen on a held-out selection set (updates 3,561 and 2,230 of 3,561), then the exact uniform soup "
    "of the two adapters (one rank-64 adapter whose weight update is the mean of the two) and heads.",
    "**Data (56,969 rows, 25.0M tokens):** the strict Decision 2.0 base (6,416 rows: program-generated decision "
    "tasks plus the human labels of GoEmotions, SNLI, CLINC150 and BANKING77; the Cosmos QA and SQuAD 2.0 "
    "answerability families removed), the A6 Score arms (4,651 rows: project-generated Score tasks with 2 to 10 "
    "levels in English and Chinese, and the graded human ratings of KLUE STS, JGLUE JSTS, IBM ArgQ-30k and SAF in "
    "Korean, Japanese, English and German), and a 20.0M-token slice of the A7 typed curricula (45,902 rows, 80% of "
    "the training tokens, an equal token quota for each of 39 families): program-generated Decision 1.0 Stage 1\u20134 "
    "decision tasks (36,105 rows) plus the BANKING77 (5,115 rows) and CLINC150 (4,682 rows) intent labels. By tokens "
    "the data is 60% English, 38% Chinese and about 2% Korean, Japanese and German together; 66% Choice, 19% Noul "
    "and 14% Score. The human sources are CC BY (3.0 or 4.0) or CC BY-SA (3.0 or 4.0).",
    "**Where the A7 dose comes from:** the A7 typed curricula are the project's own Decision 1.0 decoder-family "
    "training corpora (the shared Choice and Score training data of the Sol, Nox, Eos, Lux and Kai 1.0 models), "
    "re-audited into separable parts. The program-generated tasks carry program-oracle labels that the 1.0 builders "
    "verified independently; BANKING77 (CC BY 4.0) and CLINC150 (CC BY 3.0) keep their publishers' labels. Whole "
    "groups that touched a protected evaluation inventory were set aside, and no formal evaluation panel, "
    "JevArena-C1 source or mlx-diag test split is in the training data. The previous revision trained on a 5.0M-token "
    "slice of the same curricula (14,755 rows); this one adds 31,147 rows.",
    "**What the A7 dose did:** against a control trained on the same number of tokens that repeats the previous "
    "revision's data, the new A7 content raised typed-reasoning accuracy on unseen formal families by 4.6 points "
    "(95% CI +3.1 to +6.1) and the JevArena v3 composite by 2.4 points (95% CI +0.1 to +5.9); it cannot be separated "
    "from the Choice-heavier type mix (66% Choice tokens against 54%). At that dose, rank 32 per seed instead of "
    "rank 8 added 5.8 typed points (95% CI +4.0 to +7.6); its v3 gain (+1.9, 95% CI \u22120.9 to +5.0) is not "
    "significant on its own. More updates on the previous revision's data alone added nothing (v3 +0.8, 95% CI "
    "\u22121.9 to +2.0).",
    "**Not used:** outputs of Jev or of any third-party decision model; every row trains on its gold label. Datasets "
    "keep their own licences ([attributions](ATTRIBUTIONS.md)).",
]

LIMITATIONS = [
    "**Level with AutoJev-27B.** Post-key v3 72.36 vs AutoJev-27B 72.13 (+0.23; 95% CI lower bound \u22121.60): not a "
    "significant difference, so the higher point estimate is not a lead. Typed accuracy is level (T 0.896 vs 0.887) "
    "and so is human transfer (H 0.584 vs 0.587; \u22120.002, 95% CI [\u22120.030, +0.066], not significant). Typed "
    "Choice is lower (753 vs 800 of 800 answers correct) and constraint competition is 0.882 vs 1.000; exception "
    "stack is level (0.843) and resource ledger is higher (0.860 vs 0.705).",
    "**Against the other 27B models** JevArena v3 is significantly higher than Eikos-27B (+3.07, 95% CI [+0.20, "
    "+7.75]) and Jebadiah-27B (+6.89, 95% CI [+4.33, +10.54]); human transfer is level with both (\u22120.003, 95% CI "
    "[\u22120.047, +0.071], and +0.006, 95% CI [\u22120.031, +0.065]).",
    "**Against the previous revision** (`c0dba600`, the rank-16 M3-A soup): post-key v3 +5.15 (95% CI [+2.19, "
    "+8.02]), typed accuracy +0.109 (95% CI [+0.085, +0.133]), human transfer +0.011 (95% CI [\u22120.037, +0.056], "
    "level), public 231 +5 (95% CI [\u22121, +11], level) and card-eligible mlx-diag +1.4 points (95% CI +0.3 to "
    "+2.5). Transfer macro-F1 is lower on TempoWiC (0.713 vs 0.769) and Reddit humor (0.569 vs 0.631).",
    "**JevBench public 231:** 203 correct (48 / 72 / 83): level with AutoJev-27B's 201 (+2, 95% CI [\u22126, +10]) "
    "and above Jebadiah-27B's 176 (+27, 95% CI [+16, +38]), but significantly below Eikos-27B's 212 (\u22129, 95% CI "
    "[\u221216, \u22122]; hard tier 83 vs 92), mostly in two hard-tier skills that neither JevArena nor this model's "
    "training covers: applying long policy documents with amendments and precedence (11 vs 15 of the 19 such items) "
    "and checking a quoted person's plausible but wrong conclusion against the evidence instead of adopting it (33 "
    "vs 38 of 46).",
    "**Score levels.** On typed Score it uses all five levels (51 / 66 / 67 / 72 / 150 answers at levels 0\u20134 "
    "against 35 / 79 / 83 / 75 / 128 in the gold labels): the top level is still over-predicted, less than in the "
    "previous revision (182), and levels 1 and 2 are under-predicted.",
    "**Transfer tasks.** Wiki politeness, TempoWiC and Reddit humor are lower than all three 27B models shown "
    "(macro-F1 0.490 against 0.529\u20130.591, 0.713 against 0.736\u20130.798 and 0.569 against 0.607\u20130.645), and "
    "MRF by less than 0.01; flute and media ideology trail AutoJev-27B and Eikos-27B, and persuasion trails "
    "AutoJev-27B and Jebadiah-27B.",
    "**Long inputs** (over 4,000 characters): transfer accuracy 0.578, level with AutoJev-27B (0.576) and Eikos-27B "
    "(0.583); public 231 answers 26 of 37 (AutoJev-27B 28, Eikos-27B 29).",
    "**Multilingual.** On the mlx-diag diagnostic (public test splits in seven languages, English instructions over "
    "target-language states), non-English Choice is level with English (81.0% vs 81.7%) and non-English Noul "
    "(paraphrase pairs) is 4.0 points lower (86.0% vs 90.0%). The weakest are Korean Noul (80%, 100 items) and "
    "Spanish Choice (77.8%, 126 items). The three 27B peers ran the same diagnostic (Choice and Noul parts; no "
    "paired test): non-English Choice is 80.6% for AutoJev-27B, 78.4% for Eikos-27B and 77.6% for Jebadiah-27B "
    "(81.0% here), and non-English Noul is 88.7%, 85.0% and 85.3% (86.0% here); Korean Noul is 85%, 84% and 80% "
    "(80% here). The diagnostic's Score part, not shown because its source data are non-commercial, is weakest "
    "outside English.",
    "**Calibration.** The package returns raw probabilities (temperature 1); per-type temperatures fitted on the "
    "calibration partition were rejected because they worsened calibration on development data.",
    "**Input limit.** 32,768 tokens for the complete state, question and candidates; longer inputs return an "
    "over-budget error instead of being truncated. Training inputs were at most 4,096 tokens.",
    "**Seed soup.** Only the uniform soup of two seeds was evaluated on JevArena v3 and is released; a retrained seed "
    "of the same recipe can land below it.",
    "**Base download and memory.** The base weights are not in this repository; loading needs the pinned "
    "Qwen3.8-27B revision (about 55 GB of files) and one GPU with at least 122 GB of memory (the evaluation's peak "
    "on an 18,261-token input: 110 GB allocated, 121 GB reserved; longer inputs were not measured). CPU inference "
    "was not verified.",
    "**Evaluation familiarity.** A later screen of the program's training pools found topical-phrase overlap (no "
    "exact duplicates) between some training groups and 84 items of the reported evaluation panels. This model's "
    "training data contains none of those groups. Rescored without the 84 items, post-key v3 stays 72.36, the "
    "comparison bar (90% of AutoJev-27B) moves from 64.92 to 64.89, and none of this card's v3, human-transfer or "
    "public-231 comparisons change.",
]

DETAILS = [
    "**Weights:** the exact soup of two rank-32 LoRA adapters of the same recipe and start (seeds 20260926 and "
    "20260928; selected updates 3,561 and 2,230), stored as one standard PEFT rank-64 adapter (alpha 128), plus the "
    "averaged decision head.",
    "**Loading:** `Decision2.from_pretrained` downloads the 28 pinned files of Qwen/Qwen3.8-27B at "
    "`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` into the Hugging Face cache (or takes `base_path=` for a local copy "
    "of that revision) and checks each file's SHA-256 before loading the adapter on it.",
]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec(
    summary: dict | None, c1_node_path: str | None, c1_disclosure: str | None
) -> dict:
    old = json.loads(BASE_SPEC.read_text())
    s = copy.deepcopy(old)
    final = summary is not None
    s["_release"] = {
        "candidate": "27B Milestone 4 successor M4-A20r soup (coordinator 2026-09-30 07:20 UTC+8: A20r is THE 27B "
        "successor; F-b is superseded). 27B record v2/27b/records/m4-results-2026-09-29.md (commit 632f4005d).",
        "name": "DEV2.0-27B: named after its base model Qwen/Qwen3.8-27B (name_basis base); it loads 26,096,775,168 "
        "parameters (text backbone 25,624,600,064 + rank-64 LoRA 466,911,232 + head 5,263,872; no vision tower, MTP "
        "or LM head).",
        "gate": "successor profile at a size without Decision 1.0 (tier.no_1_0): R1-R7 of the 16:05 / 17:15 "
        "successor rule against the current revision c0dba600 (scored run M3-A-soup/formal), R5 = the no-1.0 tier "
        "gates vs AutoJev-27B, and R8 = the JevArena-C1 v1.2 post-key guard (item 8).",
        "inputs": "node A copies of the 27B track's node-B files at the same paths (hash-equal lists); the "
        "checkpoint came through llm-semantic-router/dev2-27b-staging@90a70b4a and was re-hashed on node A.",
        "runtime": "vendor_source = the scored run's source mirror ff660322a (training/model byte-identical to the "
        "formal run's); runtime_source pinned to mirror c68de36a5 for both the frozen package that C1 scores and "
        "the release, so the two builds differ only in card files.",
        "c1": (
            "final: post-key C1 line from the collected SUMMARY.json"
            if final
            else "draft: C1 line pending"
        ),
    }
    s["checkpoint"] = f"{R}/M4-A20r-soup/soup/checkpoint"
    s["expected_identity"] = {"model_sha256": IDENTITY}
    s["calibration"] = None
    s["vendor_source"] = MIRROR_SCORED
    s["runtime_source"] = MIRROR_RUNTIME
    s["gate_receipt"] = str(
        DECISIONS
        / (
            "DEV2.0-27B.decision.a20r.json"
            if final
            else "DEV2.0-27B.decision.a20r.draft.json"
        )
    )
    profile = {
        "name": "successor",
        "run": RUN,
        "current": {
            "revision": CURRENT_REVISION,
            "gate": f"{CARD_PASS}/27b/receipts/gate.json",
            "decision": f"{CARD_PASS}/DEV2.0-27B.decision.json",
            "run": f"{R}/M3-A-soup/formal",
            "mlx_predictions": f"{R}/m3-f2/mlx-diag/M3-A-soup/output/mlx-diag.predictions.jsonl",
        },
        "paired": f"{R}/m4-gates/M4-A20r-soup/paired-vs-F1.json",
        "types": f"{R}/m4-gates/M4-A20r-soup/types.json",
        "mlx_paired": f"{R}/m4-mlx/mlx-paired-M4-A20r-soup-vs-DEV2.0-27B.json",
        "exposure": f"{R}/m4-overlap/exposure-m4-a20.json",
        "public231": f"{R}/m4-gates/public231-M4-A20r-vs-DEV2.0-27B.json",
        "tier": {
            "reference": "autojev27",
            "v3_share": 0.9,
            "paired": f"{R}/m4-gates/M4-A20r-soup/paired-vs-autojev27.json",
            "no_1_0": True,
        },
    }
    if final:
        profile["c1_postkey"] = c1_node_path
    s["gate_profile"] = profile
    s["frozen_autotune_cache"] = {
        "path": f"{RUN}/triton-cache",
        "tree_sha256": SHA["cache"],
        "post_record": f"{RUN}/triton-cache.post.json",
        "origin": "the formal run's persisted cache: a fresh copy of F1's scored post-run cache (tree 03b172f1...) "
        "that added no autotune entry (group-file path rewrites only, frozen check passed); release runs copy it "
        "with python3 -m v2.27b.triton_cache copy --expect <tree_sha256>",
    }
    s["origin"]["summary"] = (
        "A PEFT LoRA adapter (rank 64, alpha 128, on all 496 text projections of the attention, gated-delta and MLP "
        "blocks) and a native candidate head trained on the frozen Qwen3.8-27B text backbone at "
        "`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0), which is not redistributed. The adapter is the "
        "exact uniform soup of two seeds of one rank-32 recipe (their LoRA factors stacked along the rank with the "
        "up-projections halved, so the mean of the two seeds' weight updates is reproduced exactly), with the two "
        "heads averaged."
    )
    s["runtime_equivalence"] = old["runtime_equivalence"].replace(
        "Checked on one GPU of the scoring node against",
        "Checked on one GPU of another node of the same type (same image and base bytes as the scoring run) against",
    )
    s["scored"] = {
        "label": "post-key same-panel run M4-A20r-soup/formal at T = 1 (kernel path, frozen autotune cache, "
        "32,768-token limit)",
        "report_sha256": SHA["report"],
        "seal_sha256": SHA["seal"],
        "predictions_sha256": {
            k: SHA[k] for k in ("typed-final", "css15", "public231", "mlx-diag")
        },
        "paired_sha256": SHA["paired_autojev"],
        "native_manifest": f"{RUN}/output/typed-final.predictions.jsonl.manifest.json",
    }
    attributions = s["licence"]["attributions"]
    attributions[1] = attributions[1].replace(
        "A7 typed curricula: the CLINC150 and BANKING77 intent labels",
        "A7 typed curricula (the project's own Decision 1.0 decoder-family training corpora; a 20.0M-token slice): "
        "the CLINC150 and BANKING77 intent labels",
    )
    card = s["card"]
    card["paired"] = f"{R}/m4-gates/M4-A20r-soup/paired-vs-autojev27.json"
    card["paired_peers"] = {
        "eikos27": f"{R}/m4-gates/M4-A20r-soup/paired-vs-eikos27b.json",
        "jebadiah27": f"{R}/m4-gates/M4-A20r-soup/paired-vs-jebadiah27b.json",
    }
    card["calibration_text"] = (
        "raw model probabilities (temperature 1). Per-type temperatures fitted on the clean 698-item calibration "
        "partition (Choice 0.639, Noul 0.545, Score 0.05) were evaluated and not adopted because they worsened "
        "calibration on out-of-distribution development data."
    )
    card["requirements_text"] = old["card"]["requirements_text"].replace(
        "The evaluation peaked at 108 GB of allocated GPU memory (120 GB reserved)",
        "The evaluation peaked at 110 GB of allocated GPU memory (121 GB reserved)",
    )
    assert card["requirements_text"] != old["card"]["requirements_text"]
    text = card["text"]
    text["architecture"] = text["architecture"].replace(
        "A rank-16 LoRA adapter", "A rank-64 LoRA adapter"
    )
    assert text["architecture"] != old["card"]["text"]["architecture"]
    text["confirmation"] = SEALED_C1 + "\n\n" + postkey_line(summary)
    text["training"] = TRAINING
    limits = list(LIMITATIONS)
    if c1_disclosure:
        limits.insert(4, c1_disclosure)
    text["limitations"] = limits
    text["details"] = DETAILS
    reports = card["reports"]
    assert reports[0]["role"] == "candidate"
    reports[0]["report"] = f"{RUN}/REPORT.json"
    reports[0]["mlx"] = f"{R}/m4-mlx/M4-A20r-soup/mlx-diag.score.json"
    return s


def decision(s: dict, summary: dict | None, summary_sha: str | None) -> dict:
    final = summary is not None
    evidence = dict(EVIDENCE)
    if final:
        evidence["c1_postkey"] = summary_sha
    value = {
        "schema": "dev2-release-decision/1",
        "status": "final" if final else "draft",
        "decision": "release",
        "model_name": "DEV2.0-27B",
        "repo_id": "llm-semantic-router/DEV2.0-27B",
        "name_basis": "base",
        "name_base_model": "Qwen/Qwen3.8-27B",
        "identity": {"model_sha256": IDENTITY},
        "report_sha256": SHA["report"],
        "paired_sha256": SHA["paired_autojev"],
        "gate_profile": "successor",
        "current_revision": CURRENT_REVISION,
        "evidence_sha256": dict(sorted(evidence.items())),
        "calibration": "none (temperature 1); the CAL698 per-type temperatures (Choice 0.639, Noul 0.545, Score "
        "0.05) were rejected under the 23:15 rule (typed-DEV Brier .065 -> .073 and ECE .049 -> .069; CSS-pilot ECE "
        ".088 -> .128; only CSS-pilot Brier improved, .464 -> .452)",
        "prepared_by": PREPARED_BY,
        "action": "New main revision of the private repository llm-semantic-router/DEV2.0-27B: the 27B Milestone 4 "
        "successor M4-A20r soup (rank-64 qwen-adapter on Qwen/Qwen3.8-27B@1d4bf0f2, T = 1, 32,768 tokens, "
        "26,096,775,168 loaded parameters) replaces the M3-A soup (c0dba600); then the superseded adapter blobs are "
        "purged with rewrite_history=False and the C1 baseline registry takes this revision's post-key run.",
        "rationale": "Coordinator decision 2026-09-30 07:20 UTC+8: A20r is THE 27B successor (post-key v3 72.36, "
        "+5.15 [+2.19, +8.02] vs the current revision; all seven 16:05 successor items pass; F-b is superseded: "
        "higher v3 lower bound, better human transfer, mlx-diag and public 231, and a storage-light adapter). "
        "Not a significant difference from AutoJev-27B (+0.23; lower bound -1.60): no 'beats AutoJev' claim. "
        + (
            f"Item 8 (JevArena-C1 v1.2 post-key guard vs the registered 57.33): C1 {summary['c1']:.2f}, "
            f"{summary['item8']['delta']:+.2f} [{summary['item8']['ci95'][0]:+.2f}, "
            f"{summary['item8']['ci95'][1]:+.2f}], {summary['item8']['verdict']}; bound by "
            "evidence_sha256.c1_postkey."
            if final
            else "Draft for the frozen package that item 8 scores; the final decision adds the C1 summary."
        ),
        "approved_package": {
            "identity": IDENTITY,
            "loaded_parameters": LOADED,
            "base": "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0: 28 pinned files by SHA-256",
            "profile": "qwen-adapter (rank 64, alpha 128)",
            "temperature": 1,
            "max_input_tokens": 32768,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": "apache-2.0 (adapter, head, runtime and card ours; base Qwen3.8-27B Apache-2.0; "
        "training-data licences credited in ATTRIBUTIONS.md)",
        "supersedes": {
            "revision": CURRENT_REVISION,
            "final_sha256": "fd1b5ef0643532407b27a4f7f8d6748e8f8f9fa27b627f5b285b8ddb41d3e008",
            "gate_sha256": "e6965d75e1f7cac175522f19fbabedd333ad2e8136f8e69a0eb9fb7b200b4a3f",
            "weights_identity": "b7fd44e3ad73d24faa85c2ba06ba61d4494aa4be2a2025c580ba02bedae6e781",
        },
        "disclosures": [
            "post-key v3 72.36 vs AutoJev-27B 72.13 (+0.23; 95% CI lower bound -1.60): not a significant difference",
            "public 231 203 vs Eikos-27B 212 (-9, 95% CI -16, -2; gates public231 REGRESSION vs that peer); the two "
            "hard-tier skills (long policy 11 vs 15 of 19; quoted conclusions 33 vs 38 of 46)",
            "typed Choice 753 vs AutoJev-27B 800; constraint competition .882 vs 1.000",
            "human transfer .584 vs AutoJev-27B .587 (-0.002, 95% CI -0.030, +0.066; not significant)",
            "card-eligible mlx-diag +.014 (95% CI +.003, +.025) vs the previous revision; the XNLI Score part stays "
            "internal (-.015, 95% CI -.032, +.001, reported only)",
            "transfer tasks below all three peers (wiki politeness, TempoWiC, Reddit humor, MRF) and lower than the "
            "previous revision on TempoWiC and Reddit humor",
        ],
    }
    if final:
        value["decided_by"] = DECIDED_BY
        value["decided_utc"] = "2026-09-29T23:20:00Z"
        del value["prepared_by"]
        value["prepared_by_release_engineering"] = PREPARED_BY
    return value


def main() -> int:
    mode = sys.argv[1]
    if mode == "draft":
        s = spec(None, None, None)
        DRAFT_SPEC.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        d = decision(s, None, None)
        out = RECORD / "DEV2.0-27B.decision.a20r.draft.json"
    elif mode == "final":
        arg = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        path = Path(arg["--c1-summary"])
        summary = json.loads(path.read_text())
        if summary["item8"]["verdict"] != "PASS" or summary["role"] != "successor":
            raise SystemExit("item 8 did not pass: no release")
        s = spec(summary, arg["--c1-node-path"], arg.get("--c1-disclosure"))
        OUT_SPEC.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        d = decision(s, summary, sha_file(path))
        out = RECORD / "DEV2.0-27B.decision.a20r.json"
    else:
        raise SystemExit(
            "usage: make_spec.py draft | final --c1-summary SUMMARY.json --c1-node-path PATH "
            "[--c1-disclosure TEXT]"
        )
    out.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "spec": str(OUT_SPEC if mode == "final" else DRAFT_SPEC),
                "decision": str(out),
                "decision_sha256": sha_file(out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
