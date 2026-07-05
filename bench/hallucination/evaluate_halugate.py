"""Standalone eval of the halugate detector pipeline on HaluEval — no router needed.

Runs the public sentinel + detector (the NLI explainer is private, but it only
filters false positives, so sentinel+detector upper-bounds the full pipeline).
Uses the paper serialization (arXiv:2603.23508 eq.3): Context [SEP] Query [SEP] Response.

    python3 -m bench.hallucination.evaluate_halugate --max-samples 1000 --sweep
"""

import argparse
import json

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

SENTINEL = "llm-semantic-router/halugate-sentinel"
DETECTOR = "llm-semantic-router/modernbert-base-32k-haldetect-combined"


def answer_token_probs(tok, model, device, context, question, answer):
    """Hallucination probability for each answer token."""
    prefix = f"{context} [SEP] {question} [SEP] "
    enc = tok(
        prefix + answer,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits[0], dim=-1)[:, 1].tolist()
    return [
        p for p, (s, e) in zip(probs, offsets, strict=False) if e > s >= len(prefix)
    ]


def flagged(probs, threshold, min_span):
    run = best = 0
    for p in probs:
        run = run + 1 if p > threshold else 0
        best = max(best, run)
    return best >= min_span


def metrics(rows, threshold, min_span):
    tp = fp = fn = tn = 0
    for faithful, needs_check, probs in rows:
        hit = needs_check and flagged(probs, threshold, min_span)
        if not faithful:
            tp, fn = tp + hit, fn + (not hit)
        else:
            fp, tn = fp + hit, tn + (not hit)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "min_span": min_span,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
        "accuracy": round((tp + tn) / len(rows), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=0.82, help="production default")
    ap.add_argument("--min-span", type=int, default=2, help="production default")
    ap.add_argument("--sweep", action="store_true", help="grid over thresholds/spans")
    ap.add_argument("--no-sentinel", action="store_true", help="detector only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_tok = AutoTokenizer.from_pretrained(DETECTOR)
    d_model = (
        AutoModelForTokenClassification.from_pretrained(DETECTOR).to(device).eval()
    )
    if not args.no_sentinel:
        s_tok = AutoTokenizer.from_pretrained(SENTINEL)
        s_model = (
            AutoModelForSequenceClassification.from_pretrained(SENTINEL)
            .to(device)
            .eval()
        )

    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    rows, skipped = [], 0
    for i, item in enumerate(ds):
        if i >= args.max_samples:
            break
        needs_check = True
        if not args.no_sentinel:
            enc = s_tok(
                item["question"], return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                label = int(s_model(**enc).logits[0].argmax())
            needs_check = s_model.config.id2label[label] == "FACT_CHECK_NEEDED"
            skipped += not needs_check
        probs = answer_token_probs(
            d_tok, d_model, device, item["knowledge"], item["question"], item["answer"]
        )
        rows.append((item["hallucination"] == "no", needs_check, probs))

    print(f"sentinel skipped {skipped}/{len(rows)}")
    if args.sweep:
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.82, 0.9):
            for span in (1, 2, 3):
                print(json.dumps(metrics(rows, thr, span)))
    else:
        print(json.dumps(metrics(rows, args.threshold, args.min_span), indent=2))


if __name__ == "__main__":
    main()
