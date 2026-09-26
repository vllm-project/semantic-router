#!/usr/bin/env python3
"""Fixed-policy controls for the modality routing task (AR / DIFFUSION / BOTH).

These are deliberately simple, untrained policies. They exist to set a floor:
if a finetuned encoder does not clear them by a meaningful margin, it has not
earned its per-request latency and memory in the router.

Three controls are reported:

  lexical         hand-written keyword / shape rules, no training
  majority        always predict the most frequent training class
  prior_random    sample a label from the training class distribution

Reported per control: overall accuracy plus per-class recall and precision.
Accuracy alone hides a class that is never predicted, which is exactly the
failure that would show up in routing.

Rules were designed against the TRAIN split only. Evaluate on test once.

Usage:
    python3 fixed_policy_controls.py --split train
    python3 fixed_policy_controls.py --split test --json results.json
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

LABELS = ["AR", "DIFFUSION", "BOTH"]

# --------------------------------------------------------------------------
# Lexical rules
#
# Ordering matters. A visual verb on its own is a DIFFUSION cue; a visual verb
# alongside a text verb is a BOTH cue. Getting that order wrong makes the
# lexical control steal DIFFUSION examples.
# --------------------------------------------------------------------------

import re

# Rule 1 - the two dominant BOTH templates in the dataset.
BOTH_TEMPLATE = re.compile(
    r"consists of both text and multiple images"
    r"|in this task, you are given a high-level goal",
    re.I,
)

# A request for written output. Stems, because "explanations" and "learning"
# are the same signal as "explain" and "learn".
TEXT_VERB = re.compile(
    r"\b("
    r"explain\w*|explanat\w*|describ\w*|teach\w*|learn\w*|understand\w*|"
    r"summari[sz]\w*|compar\w*|"
    r"walk me through|guide me through|help me|"
    r"tell me|writ(e|ing)|list|"
    r"what (is|are|does)|how (do|does|to)|how \w+ works?|why"
    r")\b",
    re.I,
)

# A request for visual output.
#
# Expanded after inspecting misses on the TRAIN split. These are the obvious
# surface cues only. The residual misses carry no visual signal at all (see
# README) and are deliberately left uncaught: chasing them would mean fitting
# the label rather than reading the request.
VISUAL_CUE = re.compile(
    r"\b("
    r"illustrate|illustrated|illustration of|"
    r"draw|sketch|"
    r"infographic|"
    r"visual guide|visual representation|visually|with visuals?|"
    r"both text and (pictures|images)|"
    r"show me (a |the |each |what )?(step|steps|difference|diagram|picture|image)|"
    r"show me what .{0,40} looks like|"
    r"generate (an image|a diagram|a chart)|"
    r"create (an infographic|a chart|a diagram|a visual)|"
    r"include a (diagram|chart|picture|image)|"
    r"(colou?r)-coded (chart|diagram|table)|"
    r"side-by-side images?|"
    r"with (a )?(diagrams?|charts?|infographics?)"
    r")\b",
    re.I,
)

# Image-generation prompt vocabulary. Strong DIFFUSION signal.
DIFFUSION_STYLE = re.compile(
    r"\b("
    r"\d+k|highly detailed|ultra[- ]detailed|intricate details|"
    r"artstation|trending on|deviantart|"
    r"octane render|unreal engine|ray trac(ing|ed)|"
    r"concept art|digital painting|matte painting|oil painting|"
    r"cinematic lighting|volumetric lighting|rim light|studio lighting|"
    r"photorealistic|hyperrealistic|photoreal|"
    r"bokeh|sharp focus|depth of field|wide angle|"
    r"(art|painting|poster|portrait|illustration) by|"
    r"character design|splash art|key visual"
    r")\b",
    re.I,
)


def looks_like_prompt(text: str) -> bool:
    """Comma-separated tag soup with no sentence structure."""
    if "?" in text:
        return False
    if "|" in text:
        return True
    words = text.split()
    if not words:
        return False
    commas = text.count(",")
    return commas >= 3 and commas / len(words) > 0.08


def predict_lexical(text: str) -> str:
    # 1. explicit BOTH templates
    if BOTH_TEMPLATE.search(text):
        return "BOTH"

    has_text = bool(TEXT_VERB.search(text))
    has_visual = bool(VISUAL_CUE.search(text))

    # 2. asks for prose AND a visual
    if has_text and has_visual:
        return "BOTH"

    # 3. image-prompt vocabulary or shape
    if DIFFUSION_STYLE.search(text) or looks_like_prompt(text):
        return "DIFFUSION"

    # 4. visual request with no prose request
    if has_visual:
        return "DIFFUSION"

    # 5. default
    return "AR"


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(rows, predict):
    correct = 0
    stats = {label: {"tp": 0, "gold": 0, "pred": 0} for label in LABELS}
    confusion = Counter()

    for row in rows:
        gold = row["label_name"]
        pred = predict(row["text"])
        stats[gold]["gold"] += 1
        stats[pred]["pred"] += 1
        confusion[(gold, pred)] += 1
        if gold == pred:
            correct += 1
            stats[gold]["tp"] += 1

    per_class = {}
    for label in LABELS:
        s = stats[label]
        per_class[label] = {
            "recall": s["tp"] / s["gold"] if s["gold"] else None,
            "precision": s["tp"] / s["pred"] if s["pred"] else None,
            "support": s["gold"],
            "predicted": s["pred"],
        }

    return {
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "per_class": per_class,
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
    }


def fmt(value):
    return "  n/a " if value is None else f"{value:.4f}"


def report(name, result, note=None):
    print(f"\n{name}")
    print("-" * len(name))
    if note:
        print(f"  {note}")
    acc = result["accuracy"]
    print(f"  accuracy: {result['correct']}/{result['total']} = {acc:.4f}")
    print(f"  {'class':<12}{'recall':>8}{'precision':>11}{'support':>9}{'pred':>7}")
    for label in LABELS:
        c = result["per_class"][label]
        print(
            f"  {label:<12}{fmt(c['recall']):>8}{fmt(c['precision']):>11}"
            f"{c['support']:>9}{c['predicted']:>7}"
        )


# --------------------------------------------------------------------------


def load(path):
    return [json.loads(line) for line in Path(path).open()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        default="exported_modality_routing_dataset",
        help="Directory holding train/validation/test .jsonl files",
    )
    ap.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
        help="Split to evaluate. Rules were designed on train; measure test once.",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Trials for the prior-matched random control",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", help="Write results to this path as JSON")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    eval_rows = load(data_dir / f"{args.split}.jsonl")
    train_rows = load(data_dir / "train.jsonl")

    # Priors come from train, never from the split being measured.
    train_counts = Counter(r["label_name"] for r in train_rows)
    n_train = sum(train_counts.values())
    priors = {label: train_counts[label] / n_train for label in LABELS}

    top = max(train_counts.values())
    tied = [label for label in LABELS if train_counts[label] == top]
    majority = tied[0]

    print(f"split:      {args.split}.jsonl  ({len(eval_rows)} rows)")
    print(f"priors:     from train.jsonl ({n_train} rows)")
    dist = "  ".join(f"{label} {priors[label]:.3f}" for label in LABELS)
    print(f"            {dist}")
    eval_counts = Counter(r["label_name"] for r in eval_rows)
    dist = "  ".join(f"{label} {eval_counts[label]}" for label in LABELS)
    print(f"eval dist:  {dist}")

    results = {
        "split": args.split,
        "n_eval": len(eval_rows),
        "train_priors": priors,
        "controls": {},
    }

    # --- lexical ---
    lex = score(eval_rows, predict_lexical)
    results["controls"]["lexical"] = lex
    report("lexical", lex)

    # --- majority ---
    note = None
    if len(tied) > 1:
        note = f"tie between {', '.join(tied)}; broke to {majority} by label order"
    maj = score(eval_rows, lambda _t: majority)
    results["controls"]["majority"] = maj
    report(f"majority ({majority})", maj, note)

    # --- prior-matched random ---
    rng = random.Random(args.seed)
    weights = [priors[label] for label in LABELS]
    accs = []
    for _ in range(args.trials):
        hits = sum(
            1
            for r in eval_rows
            if rng.choices(LABELS, weights=weights, k=1)[0] == r["label_name"]
        )
        accs.append(hits / len(eval_rows))
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    analytic = sum(
        priors[label] * eval_counts[label] / len(eval_rows) for label in LABELS
    )

    results["controls"]["prior_random"] = {
        "accuracy_mean": mean,
        "accuracy_std": var**0.5,
        "accuracy_analytic": analytic,
        "trials": args.trials,
        "seed": args.seed,
        "per_class_recall": priors,  # recall for class i is P(predict i) = prior_i
    }

    print("\nprior-matched random")
    print("--------------------")
    print(f"  trials: {args.trials}  seed: {args.seed}")
    print(f"  accuracy: {mean:.4f} +/- {var**0.5:.4f}  (analytic {analytic:.4f})")
    print(f"  {'class':<12}{'recall':>8}")
    for label in LABELS:
        print(f"  {label:<12}{priors[label]:>8.4f}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
