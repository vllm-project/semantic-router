"""Release spec and decisions of the DEV2.0-4B successor m10-4b-LH (decoder Milestone 10).

User request 2026-10-01 (release worker b5f60b33): m10-4b-LH, a rank-128 LoRA on Qwen3.5-4B-Base merged into the
base with the candidate head, trained on the released 4B mixture, is the next main of llm-semantic-router/DEV2.0-4B
under the successor gate profile (R1-R7 vs the current revision, R8 = the C1 post-key guard), at T = 1
(COORDINATION 2026-10-01 13:40), stored as its v2.release.bf16_copy and served by the BF16-resident runtime. The spec
derives from the current revision's spec (specs/dev2-4b-bf16r.json): peers, roster, banner, runtime requirements and
the licence files carry over; the weights, lineage, scored run, evidence and card text are LH's (decoder records
v2/dec/records/dec-m10-results-2026-10-01.md and dec-m10-handoff-2026-10-01.md; node A inputs from ops/prep_lh.sh).

Run from src/training/decision2:
  python3 v2/release/records/dev2-4b-m10lh-2026-10-01/ops/make_lh.py draft [--runtime-line TEXT]
  python3 v2/release/records/dev2-4b-m10lh-2026-10-01/ops/make_lh.py final --c1-summary SUMMARY.json \
      --c1-node-path PATH [--runtime-line TEXT] [--current current.json] [--runtime-source DIR]
draft: the spec before item 8 (C1 line pending, no R8) and its draft decision, for the frozen package C1 scores.
final: the C1 post-key line and R8 from the collected SUMMARY.json (a local copy of the node A file) and the final
decision. --current names the current revision (default: the BF16-resident revision 4f560ae5); a JSON file with
revision, gate, decision, decision_sha256, gate_sha256 and manifest_sha256 replaces it when another revision of
the repository lands first.
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
BASE_SPEC = SPECS / "dev2-4b-bf16r.json"
OUT_SPEC = SPECS / "dev2-4b-m10lh.json"
DRAFT_SPEC = SPECS / "dev2-4b-m10lh.draft.json"
DECISIONS = Path("/data/dev2/runs/release/decisions")
REL = "/data/dev2/runs/release"
IN = f"{REL}/inputs/dev2-4b-lh"
RUN = f"{REL}/dev2-4b-lh-t1-derived"
MLX = f"{REL}/dev2-4b-lh-t1-derived-mlx"
FORMAL = "/data/dev2/runs/dec/formal/m10/m10-4b-LH"
MIRROR = "/data/dev2/src/{}-src_training_decision2/src/training/decision2"
VENDOR = MIRROR.format("88aacb9fbcf03618bf8b18e58e2f5e1fcdc1c83d")
RUNTIME = MIRROR.format("5dc962b003cc15b12a3568fe3c8ac8c2931b03a0")
RECORDS_MIRROR = MIRROR.format("df3f242f89ca69855aa2a2f6ecb89cb857af1e5d")
BF16R = f"{RECORDS_MIRROR}/v2/release/records/dev2-bf16-resident-2026-10-01"
CURRENT = {
    "revision": "4f560ae5d26d378cd8db0a93a06c1603ea76b635",
    "gate": f"{BF16R}/4b/release/receipts/gate.json",
    "decision": f"{BF16R}/DEV2.0-4B.decision.bf16r.json",
    "decision_sha256": "b33fb92af1610a9f0ef5e317c6462b5bd07415b5e86608af56dd0e5e585edae1",
    "gate_sha256": "573936ec2b0bb023e48c9d00ed4c58f8017422a70763aa826809228c0801efea",
    "manifest_sha256": "bbf9456946a0f3a9d742c38fffcca243618cf78989457f7390cede98013910e2",
}
CURRENT_RUN = f"{REL}/dev2-4b-t1-derived"
IDENTITY = "6a555335e077fd26952fd58df06a7dcca1f5ef317194f2e3e4d7664d5c071cd7"
FP32_IDENTITY = "5fa2a6998ede6290e1c741c71b44d145b7564b622b0c8e4561ef8f7f69c4c489"
LOADED = 4208383488
BASE = ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b")
SHA = {
    "report": "375e1ab553009361d5e67d1751e9f6638eed17c3db87d12b36e5ea11f169aa88",
    "seal": "f42e00cbc5be0a821ee40f20787ef22d6071bc466641d50a0f862bcb7ad935d3",
    "typed-final": "c148d77a60c822f9e0b2fc97754d664d588ff1116686a0ffb1b376bd71aadcda",
    "css15": "56cf468fab45b2e7aa5ef12d18c80c3d27dd59ec60e66a88e612582c9d4deae5",
    "public231": "aebe679df3847372153941f4ef5d907b1604f160849030e6f751d6cf9b79bd87",
    "mlx-diag": "94124564bcc3c377ff08438f9429fdab26be00e67174a77674e31c9b1a0d8992",
    "paired_nox1": "9fec07bb780f1b7beb064af572050c3a1a105859e933e9a655f39c6ae8f12b8d",
    "bf16_copy": "7d8f21f5adb70eb331f3a6b47396b569da546613927b16890486cda81c7247a1",
}
EVIDENCE = {
    "exposure": "d42e69cafc4cf1d16958d19dd16fec6e1a7ced706ea62459fa9f065cb09c3891",
    "mlx_paired": "f264b9a418a7e78ec4a4579c8faf4cc4b79d7589f58bf16a1ed74747c0cb3a99",
    "paired": "e1fd9d1e8b94d6a4db9db90d598d9b02a0ad126398a724ea0bd6697479a766d6",
    "public231": "47c9381f21c6705fef130e6e0a2f2d373131103e90bb9fed7f746164ef9c9ace",
    "tier_paired": "b9df43c8cab4e0a6d974724a1b602ca1fbb149c2e855219e355b00d6327e0fff",
    "types": "c586b71f0199fcc0bbc2fb318669a197a8828290a7dc1f90f3ffffb40923c678",
}
PREPARED_BY = "Decision 2.0 release engineering, 4B successor worker (worktree vllm-sr-dev2-release)"
DECIDED_BY = (
    "coordinator (parent agent), Decision 2.0 program: the user's release job of 2026-10-01 (ship m10-4b-LH as the "
    "next DEV2.0-4B revision after successor item 8), under the user's full-autonomy mandate"
)

SEALED_C1 = (
    "**Independent sealed confirmation** \u2014 JevArena-C1 v1.2 (2,840 human-labeled items from 8 sources published "
    "after the relevant cutoffs, never used for training or development; scored once) measured the previous weights "
    "(revision `452f1332`, weights identity `11b5ca1c`): 48.38 vs Decision 1.0 Nox 49.70 (\u22121.32, 95% CI "
    "[\u22123.00, +0.35]), Decider 4B 49.65 (\u22121.27, 95% CI [\u22123.05, +0.49]) and Jet v6.2 50.44 (\u22122.06, "
    "95% CI [\u22123.86, \u22120.18]). It has not been repeated for this revision."
)


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace("-", "\u2212")


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
        f"vs {summary['c1'] - rule['delta']:.2f} for the previous weights ({signed(rule['delta'])}, 95% CI "
        f"[{signed(lo)}, {signed(hi)}]; {summary['items']:,} items, paired by source group). The independent "
        "sealed confirmation above measured the previous weights."
    )


ARCHITECTURE = (
    "The 32-layer Qwen3.5 text backbone of Qwen3.5-4B-Base (24 gated-delta and 8 attention layers), adapted with a "
    "rank-128 LoRA that is merged into its weights, reads the state, the question and every supplied candidate "
    "once; a shared candidate head scores the candidates against a global query and returns probabilities without "
    "generating text."
)

TRAINING = [
    "**Recipe:** a rank-128 LoRA (alpha 256, dropout 0.05) on all 248 text projections of the attention, "
    "gated-delta and MLP blocks of [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) at "
    "`1001bb4d`, plus a new candidate head, trained for one epoch (LoRA and head learning rate 1e-4; AdamW, weight "
    "decay 0.01, 5% warmup then cosine; cross-entropy plus 0.5 \u00d7 Brier on the gold labels plus 1.0 \u00d7 KL "
    "divergence to Decision 1.0 Lux's answer probabilities on every row except the 3,759 gold-only long-evidence and "
    "multilingual rows) on 58,739 decision rows (29.4M tokens, inputs up to 8,192 tokens) with seeds 20260926, "
    "20260927 and 20260928. Each seed's checkpoint was chosen on a held-out selection set (updates 787, 777 and "
    "785), its LoRA was merged into the base weights, and the release is the uniform weight average of the three "
    "merged models with their heads.",
    "**Same data and teacher as the previous revision:** the training rows are the previous revision's 58,742 "
    "rows in the same order without one quarantined group of 3 HotpotQA rows; no data source was added.",
    "**Teacher:** Decision 1.0 Lux-9B only (our own model, [`llm-semantic-router/Decision-1.0-Lux-9B`]"
    "(https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) at `bd45a30a`); its answer probabilities, "
    "computed on a runtime without the causal-conv1d kernel, are the soft targets on 54,980 rows. The 3,759 "
    "long-evidence and multilingual rows (HoVer, Natural Questions, TyDi QA, MIRACL, JCommonsenseQA, SentiMix "
    "Spanglish) train on their gold labels only. No outputs of Jev or of any third-party decision model were used.",
    "**Own Decision 1.0 training corpora (17,663 rows, 25 languages):** program-generated English and Chinese "
    "decision tasks labeled by program oracles (8,038 rows; no model-written states or labels), plus the human "
    "labels of CLINC150 (CC BY 3.0), BANKING77 (CC BY 4.0), MultiNLI non-fiction genres (OANC terms) and SNLI (CC "
    "BY-SA 4.0), and, rebuilt from their upstream files with human labels only, OpenAssistant OASST1 reply ratings "
    "in 23 languages (Apache-2.0), KLUE STS and JGLUE JSTS similarity (CC BY-SA 4.0), and SentiMix Hinglish and "
    "AfriSenti Swahili sentiment (CC BY 4.0).",
    None,  # Decision 2.0 data: the base spec's item with the new row count
    "**Not used:** outputs of Jev or of any third-party decision model; Decision 1.0 Nox's weights (earlier "
    "revisions of this repository were fine-tunes of it, so its Cosmos QA and SQuAD 2.0 answerability training no "
    "longer reaches this model through its weights); and the MASSIVE, PAWS-X and XNLI sources behind the "
    "multilingual diagnostic. Datasets keep their own licences ([attributions](ATTRIBUTIONS.md)).",
]

LIMITATIONS = [
    "**Against the previous revision** (`4f560ae5`, the full fine-tune of Decision 1.0 Nox): post-key JevArena v3 "
    "67.34 vs 63.15 (+4.19, 95% CI [+0.10, +9.88]). The gain is typed reasoning (T 0.814 vs 0.688, +0.126, 95% CI "
    "[+0.098, +0.153]; Choice 702 vs 582 of 800, Noul 743 vs 734 of 800, Score 258 vs 185 of 400). Human transfer is "
    "lower but not significantly (H 0.557 vs 0.580, \u22120.023, 95% CI [\u22120.082, +0.074]): 8 of the 15 transfer "
    "tasks are lower (macro-F1: ibc \u22120.100, talklife \u22120.050, conv_go_awry \u22120.042, reddit_humor "
    "\u22120.040, persuasion \u22120.027, wiki_politeness \u22120.023, tropes \u22120.017, raop \u22120.014) and 7 "
    "higher (wiki_corpus +0.102, flute +0.048, media_ideology +0.040, mrf +0.026, indian_english_dialect +0.020, "
    "emotion +0.009, tempowic +0.004). JevBench public 231 is level (172 vs 171; +1, 95% CI [\u22127, +9]; hard tier "
    "57 vs 56), and so are inputs over 4,000 characters (public 231: 15 vs 16 of 37; human-transfer panel: 43.6% vs "
    "41.1%).",
    "**Against the 4B peers** post-key JevArena v3 is significantly higher than Decider 4B (+5.46, 95% CI [+0.28, "
    "+8.11]) and Jet v6.2 (+6.97, 95% CI [+0.98, +11.13]), from typed reasoning (T +0.125, 95% CI [+0.095, +0.153] "
    "and +0.136, 95% CI [+0.104, +0.167]); human transfer is level with both (+0.001, 95% CI [\u22120.078, +0.043], "
    "and +0.019, 95% CI [\u22120.074, +0.088]). Typed Choice is still below both peers (0.877 vs 0.938 and 0.900). "
    "JevBench public 231 is significantly below Decider 4B (172 vs 192; \u221220 items, 95% CI [\u221230, \u221210]; "
    "hard tier 57 vs 73) and level with Jet v6.2 (174; \u22122, 95% CI [\u221211, +7]). Long inputs are weaker than "
    "both peers (inputs over 4,000 characters: public 231 15 of 37 vs 25 and 19; human-transfer panel 43.6% vs "
    "52.3% and 45.4%).",
    "**Against Decision 1.0 Nox** (adopted run): post-key v3 +10.87 (95% CI [+4.99, +14.67]); human transfer is "
    "higher but not significantly (+0.038, 95% CI [\u22120.055, +0.100]), with ibc (\u22120.096), talklife "
    "(\u22120.051), mrf (\u22120.020), wiki_politeness (\u22120.013) and conv_go_awry (\u22120.002) lower; public 231 "
    "is level (172 vs 173; \u22121, 95% CI [\u221210, +8]).",
    "**Score levels.** It rarely predicts the lowest typed Score level (7 of 400 predictions against 35 in the gold "
    "labels; recall 0.11 on level 0, as in the previous revision); levels 1\u20134 are recalled at 0.70, 0.55, 0.79 "
    "and 0.73.",
    "**Multilingual.** On the mlx-diag diagnostic, non-English Choice is level with English (78.4% vs 78.6%) and "
    "above the previous revision (75.1%), and non-English Noul (paraphrase pairs) is 76.3%, 10.7 points below "
    "English (87.0%), above the previous revision (72.7%) and below Decision 1.0 Nox (80.0%). The weakest are "
    "Korean Noul (67%, 100 items) and Japanese Noul (73%). Across the card-eligible Choice and Noul parts it is "
    "+3.7 points above the previous revision (95% CI +2.5 to +4.9). The diagnostic's Score part is not shown "
    "because its source data are non-commercial.",
    "**Calibration.** The package returns raw probabilities (temperature 1). At temperature 1 the typed-panel "
    "calibration error (ECE 0.111) is higher than the peers' (0.056 and 0.077) and Decision 1.0 Nox's (0.092), "
    "while its Brier score (0.116) is the lowest of the five models shown.",
    "**Seed soup.** The three seeds score 0.895, 0.890 and 0.905 on the held-out selection set; only their uniform "
    "average was evaluated on JevArena v3 and is released. A retrained seed of the same recipe can land below it.",
    "**Evaluation familiarity.** A later screen of the program's training pools found topical-phrase overlap (no "
    "exact duplicates) between some training groups and 84 items of the reported evaluation panels. This model's "
    "training data contains none of those groups, and without the 84 items its JevArena v3 margin over the previous "
    "revision stays significant (95% CI [+0.11, +9.75]).",
    "**CPU.** CPU inference was not verified for this release: where the GPU-only causal-conv1d package is "
    "installed, Transformers routes the convolution to that kernel and a CPU run fails.",
]

DETAILS = [
    "**Weights:** the uniform average of three rank-128 LoRA fine-tunes of Qwen3.5-4B-Base that share the recipe "
    "and differ in seed, each merged into the base weights first (selected updates 787, 777 and 785), plus the "
    "averaged candidate head. Earlier revisions of this repository were full fine-tunes of Decision 1.0 Nox; the "
    "tokenizer files are unchanged (Decision 1.0 Nox's, the Qwen3.5-4B vocabulary).",
    "**Storage:** the 248 Linear projection matrices are stored in BF16 exactly as BF16 autocast rounds them and "
    "every other tensor in FP32 (the evaluated FP32 weights, identity `5fa2a699`, give the same answers).",
    "**Comparator licences:** Decider 4B's model card declares Apache-2.0 and its repository has no LICENSE file; "
    "Jet v6.2 ships an Apache-2.0 LICENSE file and was measured with its bundled runtime on ROCm, a platform its card "
    "does not list.",
]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec(
    summary: dict | None,
    c1_node_path: str | None,
    runtime_line: str | None,
    current: dict,
    runtime_source: str,
) -> dict:
    old = json.loads(BASE_SPEC.read_text())
    s = copy.deepcopy(old)
    final = summary is not None
    s["_release"] = {
        "candidate": "decoder Milestone 10 successor m10-4b-LH (user release job 2026-10-01): rank-128 LoRA on "
        "Qwen3.5-4B-Base (merged) with the candidate head, uniform soup of three seeds; decoder records "
        "v2/dec/records/dec-m10-results-2026-10-01.md and dec-m10-handoff-2026-10-01.md.",
        "gate": f"successor profile: R1-R7 against the current revision {current['revision'][:8]} (its scored run "
        "dev2-4b-t1-derived), R5 = the tier gates (own Decision 1.0 Nox adopted run, v3 >= 0.9 x Decider 4B, "
        "human transfer vs Decider 4B), and R8 = the JevArena-C1 v1.2 post-key guard (item 8).",
        "scored": "T = 1 (COORDINATION 2026-10-01 13:40): the sealed CAL698 formal run m10-4b-LH (node F GPU3, "
        "image dbe5f32b) returned to T = 1 offline with 0 answer changes, byte-equal to the decoder's own T = 1 "
        "derivation, then adopted, sealed and paired on node A (ops/prep_lh.sh derive).",
        "storage": f"v2.release.bf16_copy of the decoder's frozen FP32 checkpoint (identity {FP32_IDENTITY[:8]} -> "
        f"{IDENTITY[:8]}; receipt {SHA['bf16_copy'][:8]}).",
        "runtime": "vendor_source = the formal run's source mirror 88aacb9fb (training/model byte-identical); "
        f"runtime_source = {Path(runtime_source).parents[2].name[:9]} (BF16-resident runtime).",
        "c1": (
            "final: post-key C1 line from the collected SUMMARY.json"
            if final
            else "draft: C1 line pending"
        ),
        "previous": old["_release"],
    }
    s["checkpoint"] = f"{IN}/bf16/checkpoint"
    s["expected_identity"] = {"model_sha256": IDENTITY}
    s["bf16_copy"] = {
        "receipt": f"{IN}/bf16/bf16-copy.json",
        "sha256": SHA["bf16_copy"],
    }
    s["calibration"] = None
    s["vendor_source"] = VENDOR
    s["runtime_source"] = runtime_source
    s["gate_receipt"] = str(
        DECISIONS
        / (
            "DEV2.0-4B.decision.m10lh.json"
            if final
            else "DEV2.0-4B.decision.m10lh.draft.json"
        )
    )
    s["origin"] = {
        "repo_id": BASE[0],
        "revision": BASE[1],
        "relation": "finetune",
        "summary": "A rank-128 LoRA on all 248 text projections of Qwen3.5-4B-Base (Apache-2.0), merged into the "
        "base weights, with a new candidate head; the release is the uniform weight average of three seeds of that "
        "merged fine-tune. The Qwen3.5 vision tower is not part of it. The tokenizer files are Decision 1.0 Nox's "
        "(the Qwen3.5-4B vocabulary), unchanged from earlier revisions.",
    }
    s["runtime_equivalence"] = (
        "decision2/qwen.py loads this full checkpoint with the vendored training/model sources whose SHA-256 equal "
        "the scored adapter sources (checked at build time) and applies the per-item batching, BF16-backbone / "
        "FP32-head execution, raw probabilities (temperature 1; no calibration file) and answer normalization of "
        "v2.dec.infer_dec, which for this merged full checkpoint wraps the same DecisionModel without extra "
        f"readouts. The scored checkpoint ({FP32_IDENTITY[:8]}) stored every tensor in FP32; this package "
        f"(v2.release.bf16_copy, receipt {SHA['bf16_copy'][:8]}) stores its 248 Linear projection matrices in BF16 "
        "exactly as BF16 autocast rounds them and every other tensor bit for bit in FP32, and the runtime holds the "
        "backbone's BF16-exact Linear weights in BF16, the values BF16 autocast multiplies with. Checked with 0 "
        "answer changes on one GPU of another node of the same type (same image, kernels and frozen autotune "
        "caches as the scoring run) against the T = 1 predictions derived exactly from the sealed CAL698 "
        "predictions of every scored prompt (typed-final 1,600, css15 6,547, public231 231) and of the mlx-diag "
        "diagnostic (2,275) by release.sh --parity, each panel with a copy of the persisted Triton autotune cache "
        "of the run that scored it."
    )
    s["scored"] = {
        "label": "post-key same-panel run m10-4b-LH at T = 1 (predictions derived from the sealed CAL698 run by "
        "undoing its temperatures; adopted and sealed)",
        "report_sha256": SHA["report"],
        "seal_sha256": SHA["seal"],
        "predictions_sha256": {
            k: SHA[k] for k in ("typed-final", "css15", "public231", "mlx-diag")
        },
        "paired_sha256": SHA["paired_nox1"],
        "native_manifest": f"{FORMAL}/output/typed-final.predictions.jsonl.manifest.json",
    }
    lic = s["licence"]
    lic["components"] = [
        {
            "component": "DEV2.0-4B weights, decision head, package runtime, card and artwork",
            "licence": "apache-2.0",
            "source": "llm-semantic-router/DEV2.0-4B",
        },
        {
            "component": "Qwen3.5-4B-Base text backbone (direct weight origin; rank-128 LoRA merged)",
            "licence": "apache-2.0",
            "source": f"{BASE[0]}@{BASE[1][:8]}",
        },
        {
            "component": "tokenizer files of Decision 1.0 Nox-4B (the Qwen3.5-4B vocabulary)",
            "licence": "apache-2.0",
            "source": "llm-semantic-router/Decision-1.0-Nox-4B@cde2a68d",
        },
        {
            "component": "Qwen3.5-4B tokenizer and chat template (upstream of Decision 1.0 Nox's tokenizer files)",
            "licence": "apache-2.0",
            "source": "Qwen/Qwen3.5-4B@851bf6e8",
        },
    ]
    lic["files"] = lic["files"] + [
        {
            "source": f"/data/dev2/hf-cache/models--Qwen--Qwen3.5-4B-Base/snapshots/{BASE[1]}/LICENSE",
            "path": "LICENSES/Qwen3.5-4B-Base-LICENSE.txt",
            "sha256": "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
        }
    ]
    att = lic["attributions"]
    assert att[0].startswith("[Decision 1.0 Nox-4B]") and att[1].startswith(
        "[Qwen3.5-4B]"
    )
    assert att[2].startswith("[Decision 1.0 Lux-9B]") and "54,983" in att[2]
    lic["attributions"] = [
        f"[Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) at `{BASE[1]}`: direct weight origin; a "
        "rank-128 LoRA was trained on it and merged into its text backbone (Apache-2.0, "
        "`LICENSES/Qwen3.5-4B-Base-LICENSE.txt`).",
        "[Decision 1.0 Nox-4B](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B) at "
        "`cde2a68dbaa557ea65dc458104d410a0802ee259`: tokenizer files (Apache-2.0, `LICENSE`); none of its weights "
        "are included.",
        "[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`: "
        "upstream vocabulary of the tokenizer files and the chat template (Apache-2.0, "
        "`LICENSES/Qwen3.5-4B-LICENSE.txt`).",
        att[2].replace("54,983", "54,980"),
        *att[3:],
    ]
    card = s["card"]
    card["paired"] = f"{IN}/gates/paired-vs-adopted-1.0.json"
    card["paired_peers"] = {
        "decider4b": f"{RUN}/PAIRED-vs-decider4b.json",
        "jet62": f"{RUN}/PAIRED-vs-jet62.json",
    }
    card["calibration_text"] = (
        "raw model probabilities (temperature 1). Per-type temperatures fitted on the clean 698-item calibration "
        "partition (Choice 0.621, Noul 0.479, Score 0.209) lowered every development calibration measure and change "
        "no answer, but are not shipped: every Decision 2.0 model keeps temperature 1."
    )
    text = card["text"]
    text["architecture"] = ARCHITECTURE
    text["confirmation"] = SEALED_C1 + "\n\n" + postkey_line(summary)
    old_d2 = old["card"]["text"]["training"][3]
    assert old_d2.startswith("**Decision 2.0 data (41,079 rows")
    training = list(TRAINING)
    training[training.index(None)] = old_d2.replace("41,079 rows", "41,076 rows")
    text["training"] = training
    text["limitations"] = list(LIMITATIONS)
    text["details"] = DETAILS + ([runtime_line] if runtime_line else [])
    reports = card["reports"]
    assert reports[0]["role"] == "candidate"
    reports[0]["report"] = f"{RUN}/REPORT.json"
    reports[0]["mlx"] = f"{MLX}/mlx-diag.score.json"
    s["gate_profile"] = {
        "name": "successor",
        "run": RUN,
        "current": {
            "revision": current["revision"],
            "gate": current["gate"],
            "decision": current["decision"],
            "run": CURRENT_RUN,
            "mlx_predictions": f"{IN}/current-mlx/output/mlx-diag.predictions.jsonl",
        },
        "paired": f"{IN}/gates/paired-vs-dev2-4b.json",
        "types": f"{IN}/gates/types.json",
        "mlx_paired": f"{IN}/gates/mlx-paired-vs-dev2-4b.json",
        "exposure": "/data/dev2/runs/dec/m10/exposure/exposure-m10-4b-base.json",
        "public231": f"{IN}/gates/public231-vs-dev2-4b.json",
        "tier": {
            "reference": "decider4b",
            "v3_share": 0.9,
            "paired": f"{IN}/gates/paired-vs-decider4b.json",
        },
    }
    if final:
        s["gate_profile"]["c1_postkey"] = c1_node_path
    s["frozen_autotune_cache"] = {
        "formal": "/data/dev2/runs/dec/formal/m10/m10-4b-LH-cache-frozen (1,654 files; the formal run's persisted "
        "cache, manifest 569e86f5)",
        "mlx-diag": f"{IN}/mlx-cache-frozen (1,654 files; the mlx-diag run's persisted cache, manifest 0bfda305, "
        "relayed from node F)",
    }
    return s


def decision(
    s: dict, summary: dict | None, summary_sha: str | None, current: dict
) -> dict:
    final = summary is not None
    evidence = dict(EVIDENCE)
    evidence["current_decision"] = current["decision_sha256"]
    evidence["current_gate"] = current["gate_sha256"]
    if final:
        evidence["c1_postkey"] = summary_sha
    value = {
        "schema": "dev2-release-decision/1",
        "status": "final" if final else "draft",
        "decision": "release",
        "model_name": "DEV2.0-4B",
        "repo_id": "llm-semantic-router/DEV2.0-4B",
        "identity": {"model_sha256": IDENTITY},
        "report_sha256": SHA["report"],
        "paired_sha256": SHA["paired_nox1"],
        "gate_profile": "successor",
        "current_revision": current["revision"],
        "evidence_sha256": dict(sorted(evidence.items())),
        "calibration": "none (temperature 1; COORDINATION 2026-10-01 13:40 keep T = 1 everywhere): the CAL698 "
        "per-type temperatures 8d88e163 (Choice 0.621, Noul 0.479, Score 0.209), adopted by the decoder under the "
        "23:15 rule, are not shipped; undoing them changes 0 answers on every scored panel and mlx-diag",
        "prepared_by": PREPARED_BY,
        "action": "New main revision of the private repository llm-semantic-router/DEV2.0-4B: the decoder M10 "
        "successor m10-4b-LH (rank-128 LoRA on Qwen/Qwen3.5-4B-Base@1001bb4d merged, candidate head, uniform soup "
        "of three seeds; qwen-full, BF16 storage, T = 1, 16,384 tokens, 4,208,383,488 loaded parameters) replaces "
        f"the N4XF soup ({current['revision'][:8]}); then the superseded weight blobs are purged with "
        "rewrite_history=False and the C1 baseline registry takes this revision's post-key run.",
        "rationale": "Successor rule items 1-7 pass against the current revision (decoder record "
        "dec-m10-results-2026-10-01; recomputed on the T = 1 run): post-key v3 67.34 vs 63.15, +4.19 [+0.10, "
        "+9.88]; human transfer -0.023 [-0.082, +0.074] (not below); types OK; card-eligible mlx-diag +0.037 "
        "[+0.025, +0.049]; tier gates (vs adopted Nox 1.0 +10.87 [+4.99, +14.67]; v3 67.34 >= 55.69 = 0.9 x "
        "Decider 4B; human transfer vs Decider 4B +0.001 [-0.078, +0.043]); 0 exposed training groups; public 231 "
        "172 vs 171 (McNemar p 1.0). "
        + (
            f"Item 8 (JevArena-C1 v1.2 post-key guard vs the registered 48.38): C1 {summary['c1']:.2f}, "
            f"{summary['item8']['delta']:+.2f} [{summary['item8']['ci95'][0]:+.2f}, "
            f"{summary['item8']['ci95'][1]:+.2f}], {summary['item8']['verdict']}; bound by "
            "evidence_sha256.c1_postkey."
            if final
            else "Draft for the frozen package that item 8 scores; the final decision adds the C1 summary."
        ),
        "approved_package": {
            "identity": IDENTITY,
            "fp32_identity": FP32_IDENTITY,
            "loaded_parameters": LOADED,
            "profile": "qwen-full (BF16 storage of the Linear projections)",
            "temperature": 1,
            "max_input_tokens": 16384,
            "revision_binding": "receipts/gate.json of the release.sh --upload --collect --already-collected run "
            "binds this file's SHA-256 to the new Hub revision and package manifest it verified.",
        },
        "licence_decision": "apache-2.0: own LoRA (merged), head, runtime and card; base Qwen/Qwen3.5-4B-Base "
        "@1001bb4d Apache-2.0 (card metadata license apache-2.0, LICENSE = Apache License 2.0, Copyright 2026 "
        "Alibaba Cloud, sha256 50cbab8a); tokenizer files of Decision 1.0 Nox-4B @cde2a68d and the Qwen3.5-4B "
        "@851bf6e8 vocabulary / chat template Apache-2.0; teacher Decision 1.0 Lux-9B (own, Apache-2.0, no weights "
        "included); training-data licences (CC BY, CC BY-SA, MIT, Apache-2.0, CC0, OANC) credited in "
        "ATTRIBUTIONS.md; Decider 4B (Apache-2.0, card metadata only) and Jet v6.2 (Apache-2.0 LICENSE) shown; the "
        "XNLI-based mlx-diag Score part stays off the card.",
        "supersedes": {
            "revision": current["revision"],
            "final_sha256": current["decision_sha256"],
            "gate_sha256": current["gate_sha256"],
            "manifest_sha256": current["manifest_sha256"],
            "weights_identity": "8b0a8bc7f504faa69904329225a11f36cc7352abb9bd64f778d7b777f91c5b79",
        },
        "disclosures": [
            "lineage: weights from Qwen3.5-4B-Base through a merged rank-128 LoRA, not from Decision 1.0 Nox; same "
            "training mixture (minus 3 quarantined HotpotQA rows) and own-Lux teacher",
            "human transfer .557 vs .580 for the previous revision (-0.023 [-0.082, +0.074], not significant); 8 of "
            "15 CSS15 tasks lower, ibc -0.100 and talklife -0.050 the largest",
            "public 231 172 vs Decider 4B 192 (-20 [-30, -10]; McNemar p .0002); level with Jet v6.2 and the "
            "previous revision",
            "typed Choice .877 below Decider 4B .938 and Jet v6.2 .900; Score level 0 rarely predicted (7 of 400)",
            "typed ECE at T = 1 .111 above the peers' .056 / .077",
            "mlx-diag non-English Noul 76.3% below Decision 1.0 Nox 80.0%; Korean Noul 67%",
        ],
    }
    if final:
        value["decided_by"] = DECIDED_BY
        value["prepared_by_release_engineering"] = PREPARED_BY
        del value["prepared_by"]
    return value


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    arg = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    current = (
        json.loads(Path(arg["--current"]).read_text())
        if "--current" in arg
        else CURRENT
    )
    runtime_source = arg.get("--runtime-source", RUNTIME)
    line = arg.get("--runtime-line")
    if mode == "draft":
        s = spec(None, None, line, current, runtime_source)
        DRAFT_SPEC.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        d = decision(s, None, None, current)
        out = RECORD / "DEV2.0-4B.decision.m10lh.draft.json"
    elif mode == "final":
        path = Path(arg["--c1-summary"])
        summary = json.loads(path.read_text())
        if summary["item8"]["verdict"] != "PASS" or summary["role"] != "successor":
            raise SystemExit("item 8 did not pass: no release")
        s = spec(summary, arg["--c1-node-path"], line, current, runtime_source)
        OUT_SPEC.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        d = decision(s, summary, sha_file(path), current)
        out = RECORD / "DEV2.0-4B.decision.m10lh.json"
    else:
        raise SystemExit(__doc__)
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
