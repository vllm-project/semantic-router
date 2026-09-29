"""Write specs/dev2-0p6b-m8-release.json from the current release spec and the BF16 parity staging spec.

Run from src/training/decision2: python3 v2/release/records/dev2-0p6b-m8-release-2026-09-29/ops/make_spec.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
base = json.load(open("v2/release/specs/dev2-0p6b-release.json"))
stg = json.load(open("v2/release/specs/dev2-0p6b-m8-bf16-parity-staging.json"))
s = dict(base)
B = "/data/dev2/runs/release/inputs/dev2-0p6b-m8-bf16"
M8 = "/data/dev2/runs/06b/m8/formal/m8-s5-b05"
G = M8 + ".gates"
MIR = (
    "/data/dev2/src/95a683e506e723e7f7810402d3152a2192a7f864-src_training_decision2"
    "/src/training/decision2/v2/release/records/dev2-0p6b-release-2026-09-28"
)
s["checkpoint"] = B + "/checkpoint"
for k in ("expected_identity", "score_bias", "bf16_copy", "scored"):
    s[k] = stg[k]
s["calibration"] = None
s["gate_receipt"] = "/data/dev2/runs/release/decisions/DEV2.0-0.6B.m8.decision.json"
s["runtime_equivalence"] = (
    "decision2/qwen.py loads this full checkpoint with the vendored training/model sources whose SHA-256 equal "
    "the scored adapter sources (checked at build time) and applies the per-state batching, BF16-backbone / "
    "FP32-head execution, raw probabilities (temperature 1; no calibration file), the fixed five-level Score "
    "logit offsets of score_bias.json and the answer normalization of training.model.infer (adapter "
    "dev2-06b-causal-8k-sb). The scored checkpoint (01fae750) stored every tensor in FP32; this package "
    "(v2.release.bf16_copy, receipt 2a991bfb) stores its 196 Linear projection matrices in BF16 exactly as BF16 "
    "autocast rounds them and every other tensor bit for bit in FP32, with the same offsets rebound to its hash. "
    "Checked on one GPU of the scoring node against the sealed predictions of every scored prompt (typed-final "
    "1,600, css15 6,547, public231 231) and of the mlx-diag diagnostic (2,275) by release.sh --parity at "
    "tolerance 0: 0 answer changes, maximum probability drift 0.0."
)
s["origin"] = dict(base["origin"])
s["origin"]["summary"] = (
    "Every weight of the official Qwen3-0.6B-Base (Apache-2.0) was fine-tuned (nothing frozen, no adapter) "
    "together with a new candidate head trained from random initialization; the release is the uniform average "
    "of six such fine-tunes (two training recipes × three seeds), plus fixed per-level offsets on five-level Score "
    "logits. It contains no Decision 1.0 Kai or Lex weights: Decision 1.0 Kai is the comparison model, and our own "
    "Decision 1.0 Lux-9B supplied soft training targets only."
)
lic = json.loads(json.dumps(base["licence"]))
lic["attributions"][1] = (
    "[Decision 1.0 Lux-9B](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) (Apache-2.0), our own "
    "model: its probabilities on all 232,754 training prompts were soft training targets. None of its weights are "
    "included."
)
lic["attributions"][2] = (
    (HERE / "credits/training-attribution.txt").read_text(encoding="utf-8").strip()
)
lic["components"][0][
    "component"
] = "DEV2.0-0.6B weights, Score offsets, decision head, package runtime, card and artwork"
s["licence"] = lic
c = json.loads(json.dumps(base["card"]))
c["paired"] = M8 + "/PAIRED-vs-kai1.json"
c["reports"][0] = {
    "key": "candidate",
    "role": "candidate",
    "report": M8 + "/REPORT.json",
    "mlx": M8 + "-mlx/mlx-diag.score.json",
    "label": "DEV2.0-0.6B",
}
c["calibration_text"] = (
    "Raw probabilities at T = 1. Five-level Score answers first add the fixed per-level logit offsets in "
    "`score_bias.json` (see Limitations). Per-type temperatures fitted on the clean 698-item calibration "
    "partition (Choice 0.584, Noul 0.554, Score 0.450) were evaluated and not adopted because they worsened "
    "calibration on out-of-distribution development data."
)
t = c["text"]
t["confirmation"] = (
    "**Independent sealed confirmation:** The sealed JevArena-C1 comparison measured the previous revision "
    "(`99c4e799`); it has not been repeated for this revision."
)
t["details"] = [
    "**Weights:** The uniform average of six full fine-tunes of Qwen3-0.6B-Base (two training recipes × three "
    "seeds; 30M tokens per seed, one epoch), with a new candidate head. Decision 1.0 Lux-9B probabilities were soft "
    "training targets on every row; none of its weights are included.",
    "**This revision** replaces `99c4e799` (a different three-seed average): post-key JevArena v3 48.64 against "
    "43.54, a paired difference of +5.09 (95% interval [+2.65, +9.30]). Only five-level Score answers depend on the "
    "offsets; Choice, Noul and other Score scales come from the weights alone.",
    "**Storage:** the backbone's Linear projection matrices are stored in BF16 exactly as BF16 autocast rounds them, "
    "every other tensor in FP32; on every evaluated prompt the answers and probabilities equal those of the "
    "evaluated FP32 weights.",
]
t["training"] = [
    "**Recipe:** each fine-tune is one epoch on its own 30M-token mixture (logical batch 16; backbone learning rate "
    "1e-5 with 10% warm-up and cosine decay, head 2e-4) with cross-entropy plus 0.5 × Brier and a KL term (weight "
    "1.0) toward soft targets from our own [Decision 1.0 Lux-9B](https://huggingface.co/llm-semantic-router/"
    "Decision-1.0-Lux-9B) on every row. No other teacher, and no outputs of Jev or of any third-party decision model.",
    "**Data:** together the six mixtures hold 232,754 distinct decision rows: Choice 88,676, Noul 75,701 and Score "
    "68,377; English 130,105, Chinese 46,699, Hinglish 12,269, Spanish 8,791, Japanese 6,537, Korean 5,884 and 26 "
    "other languages.",
    "**Program-generated tasks (89,771 rows):** Decision 1.0 stage 1–4 generators and Decision 2.0 programmatic and "
    "verifiable generators, labeled by program oracles (no third-party text).",
    "**Human-labeled and public data (142,983 rows)** from 37 public datasets that keep their own licences "
    "([attributions](ATTRIBUTIONS.md)). The training mixtures contain no MASSIVE, PAWS-X or XNLI rows (the sources "
    "of mlx-diag), and every group the evaluation-overlap rescreen excluded was removed before training.",
    "**Score offsets:** fitted afterwards on 400 fresh draws of the benchmark's own typed Score generator (see "
    "Limitations); no weight was changed.",
]
t["limitations"] = [
    "**Score offsets:** Five-level Score answers add fixed per-level offsets to the model's logits before the "
    "softmax (`score_bias.json`: +0.039, +0.203, +0.079, −0.152, −0.170 for levels 0–4). Choice, Noul and other "
    "Score scales are unchanged. The offsets were fitted on 400 fresh draws of the benchmark's own typed Score "
    "generator (`resource_ledger`, the family behind the typed panel's Score items). These draws are disjoint from "
    "the panel's items up to event ids and row order, but about three quarters share an answer structure with them. "
    "The offsets were chosen on a separate 400-item check half and on held-out human 5-level ratings. Without them "
    "the same weights give the top level on 94.75% of the typed Score items.",
    "**Typed Score accuracy stays near always-majority:** On the typed Score family the model is right on 34.3% of "
    "items, against 32.0% for always answering the most common level (difference +2.3 points, 95% CI −1.5 to +6.3). "
    "The offsets fix level usage (62.0% of answers now on the top level), not Score skill on this family, and level "
    "3 is never predicted there.",
    "**Human 5-level ratings:** On 1,809 held-out human-rated 5-level items the offsets change accuracy by −1.1 "
    "points (34.1% → 33.0%; 95% CI −2.8 to +0.7).",
    "**Against the previous revision:** The human-transfer axis is .459 against .480 (−.021, 95% CI −.064 to "
    "+.061). CSS15 losses are wiki politeness −3.4, Indian English dialect −3.4 and FLUTE −1.6 accuracy points. "
    "Public-231 easy is 47 against 48. Typed-panel calibration is weaker at T = 1 (ECE .155 against .118; Brier "
    ".321 against .309).",
    "**Multilingual (mlx-diag, card-eligible parts):** Choice: 75.4% English, 70.2% other languages (weakest "
    "Arabic, 62.7%). Noul: 53.0% English, 52.8% other languages, near chance (weakest Spanish, 50.0%). The Choice + "
    "Noul macro is +0.4 points against the previous revision (95% CI −0.6 to +1.4).",
    "**Public 231:** public-only rerun (about a third of the official Intelligence inputs), not the official "
    "JevBench score; easy tier at ceiling; totals within about 10 items are not distinguishable. It answers 152 "
    "(easy 47, standard 61, hard 44); paired differences: previous revision +10 items (95% interval [0, +20], "
    "p .087), Bosun v3.1 0.6B +19 ([+7, +31]), GLiNER2.5-Decide +36 ([+23, +49]), Decision 1.0 Kai +38 ([+23, "
    "+53]; +25 [+10, +41] against Kai at 8,192 tokens) and Decision 1.0 Lex +39 ([+24, +55]). No peer is "
    "significantly ahead on it.",
    "**Same-limit comparison:** the table shows Decision 1.0 Kai at its published 1,024-token input limit. Run at "
    "this model's 8,192-token limit, Kai scores 35.97 on JevArena v3 and 127/231 on JevBench public, and the paired "
    "v3 difference is +12.67 (95% interval [+9.74, +17.42]).",
    "**Overlap:** None of the evaluation-overlap rescreen's excluded training groups is in the training mixtures.",
]
s["card"] = c
s["gate_profile"] = {
    "name": "successor",
    "run": M8,
    "current": {
        "revision": "99c4e799392afa73241915aeb16fefa5ef3518d7",
        "gate": MIR + "/final/receipts/gate.json",
        "decision": MIR + "/DEV2.0-0.6B.decision.json",
        "run": "/data/dev2/runs/06b/m4/formal/m4-t-a7-soup",
        "mlx_predictions": "/data/dev2/runs/06b/m4/formal/m4-t-a7-soup-mlx/output/mlx-diag.predictions.jsonl",
    },
    "paired": G + "/paired-vs-released.json",
    "types": G + "/types.json",
    "mlx_paired": G + "/mlx-paired.json",
    "exposure": "/data/dev2/runs/06b/m8/overlap/exposure-m6-mxcx-soup.json",
    "public231": "/data/dev2/runs/release/dev2-0p6b-m8-public231-20260929T132851Z/public231-vs-previous.json",
    "tier": {
        "reference": "gliner25",
        "v3_share": 0.9,
        "paired": G + "/paired-vs-gliner25.json",
    },
}
out = Path("v2/release/specs/dev2-0p6b-m8-release.json")
out.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)
