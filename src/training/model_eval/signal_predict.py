#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Prediction Script
========================

Tags each task with a signal label by directly running any HuggingFace
text-classification or token-classification model — no router, no config
YAML required.

Complements signal_eval.py:
  - signal_eval.py: measures signal accuracy vs ground truth via
    router /v1/eval API (requires router online + config YAML). Use when
    you already have a running SR instance and want to validate signal
    quality end-to-end.
  - signal_predict.py: produces signal labels for downstream
    join/statistics via direct classifier inference (no router needed).
    Use when you are in the pre-config phase (config not generated yet,
    router not running) and just need per-question labels.

The candle binding (Rust) used by SR internally is not directly callable
from Python; it requires the router process. This script uses transformers
+ torch instead, which is portable and sufficient for small classifiers
like mmbert32k (760 questions in minutes on CPU).

Decoupled from any specific model: the model class (sequence vs token
classification) and the label list are read from the model's config
(id2label), so any HF text-classification checkpoint works.

Input:  tasks jsonl, one {task_id, prompt_user, ...} per line.
Output: JSON dict {task_id: {<label-field>: label, confidence: score}}.

How to read the output
----------------------

Terminal output (stdout) is the primary readout. Focus on:

  1. Loaded-model banner. Example:
       loaded model: models/mmbert32k-intent-classifier-merged (sequence_cls, 14 labels) on cpu
       760 tasks -> classifier

       Confirms the model class detected (sequence_cls or token_cls), the
       label count (read from config.id2label), and the device.

  2. Per-question JSON file (--out). Format:
       {
         "task_id_1": {"domain": "math", "confidence": 0.92},
         "task_id_2": {"domain": "philosophy", "confidence": 0.85},
         ...
       }

       The field name is controlled by --label-field (default "label").
       Use --label-field domain for backward compatibility with
       join_three_way.py, which reads sigs[id]["domain"].

  3. Label distribution at the end of stdout. Example:
       domain distribution:
                math      184
         philosophy      120
                law       95
                ...

       A highly skewed distribution (one label dominating) may indicate
       classifier mis-calibration — investigate before using the
       predictions for routing policy.

Pass the --out JSON to downstream consumers:
  - join_three_way.py --signals <out.json>  (offline config-effect simulation)
  - signal_eval.py --custom-ground-truth <out.json>  (if you have ground
    truth and want to measure classifier accuracy; requires the file to
    follow signal_eval's expected_ground_truth format)

Usage:
    # Default: mmbert32k intent classifier (domain signal)
    python src/training/model_eval/signal_predict.py \\
        --tasks tasks/eval_office.jsonl \\
        --out results/dryrun_office_test_raw.json \\
        --model-id models/mmbert32k-intent-classifier-merged \\
        --label-field domain --device cpu

    # Any HF text-classification model (e.g., jailbreak detector)
    python src/training/model_eval/signal_predict.py \\
        --tasks tasks/eval_office.jsonl \\
        --out results/jailbreak_preds.json \\
        --model-id models/mmbert32k-jailbreak-detector-merged \\
        --label-field jailbreak --device cpu

    # PII detection (token classification)
    python src/training/model_eval/signal_predict.py \\
        --tasks tasks/eval_office.jsonl \\
        --out results/pii_preds.json \\
        --model-id models/mmbert32k-pii-detector-merged \\
        --label-field pii --device cpu
"""
import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

# AutoModel avoids hard-coding ModernBert — HF picks the right class from
# the checkpoint's config.architectures field. For token classification we
# fall back to AutoModelForTokenClassification.
_AUTO_CLS_BY_PROBLEM_TYPE = {
    "single_label_classification": AutoModelForSequenceClassification,
    "multi_label_classification": AutoModelForSequenceClassification,
}


def load_classifier(model_id: str, device: str):
    """Load any HF sequence/token classification model.

    The model class is chosen from config.problem_type (or
    config.architectures) — no hard-coding of ModernBert. The label list
    comes from config.id2label.
    """
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    problem_type = getattr(config, "problem_type", None)
    is_token_cls = False
    if problem_type == "token_classification":
        is_token_cls = True
    else:
        # Infer from architectures name as a fallback
        arch = getattr(config, "architectures", []) or []
        if any("TokenClassification" in a for a in arch):
            is_token_cls = True

    if is_token_cls:
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
        )
    model.to(device).eval()

    id2label = {
        int(k): str(v) for k, v in (getattr(config, "id2label", {}) or {}).items()
    }
    return model, tokenizer, id2label, is_token_cls


def predict_batch_sequence_cls(
    texts, model, tokenizer, id2label, device, batch_size=16
):
    """Predict (label, confidence) for text classification."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
        for p, c in zip(preds.cpu().tolist(), confs.cpu().tolist()):
            label = id2label.get(p, str(p))
            results.append((label, c))
    return results


def predict_batch_token_cls(texts, model, tokenizer, id2label, device, batch_size=16):
    """Predict per-token labels for token classification.

    Returns one aggregated label per text (the highest-frequency non-O
    label, or "O" if all tokens are O). Confidence is the mean probability
    of the chosen label across tokens where it appears.

    Padding and special tokens are filtered via ``attention_mask`` so
    predictions depend only on real tokens — a short text returns the same
    label whether evaluated alone or batched with a longer text.
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            is_split_into_words=False,
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
            attn = inputs.attention_mask.cpu().tolist()
        for idx in range(len(batch)):
            # Keep only real-token positions (attention_mask == 1) so
            # padding does not influence the aggregated label.
            mask = attn[idx]
            real_ids = [p for p, m in zip(pred_ids[idx], mask) if m == 1]
            real_probs = [p for p, m in zip(probs[idx].cpu().tolist(), mask) if m == 1]
            if not real_ids:
                results.append(("O", 0.0))
                continue
            counts = Counter(id2label.get(p, str(p)) for p in real_ids)
            non_o = {k: v for k, v in counts.items() if k not in ("O", "0")}
            if non_o:
                label = max(non_o, key=non_o.get)
            else:
                label = counts.most_common(1)[0][0]
            conf = sum(max(p) for p in real_probs) / max(len(real_probs), 1)
            results.append((label, round(conf, 4)))
    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tasks",
        required=True,
        help="Input tasks jsonl (one {task_id, prompt_user} per line)",
    )
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument(
        "--model-id",
        required=True,
        help="HF id or local path of a text/token classification model",
    )
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--label-field",
        default="label",
        help="Field name for the predicted label in output JSON "
        '(default: "label"; use "domain" for backward compat with '
        "join_three_way.py)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model, tokenizer, id2label, is_token_cls = load_classifier(
        args.model_id, args.device
    )
    logger.info(
        f"loaded model: {args.model_id} "
        f"({'token_cls' if is_token_cls else 'sequence_cls'}, "
        f"{len(id2label)} labels) on {args.device}"
    )

    with open(args.tasks, encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f]
    texts = [t["prompt_user"] for t in tasks]
    logger.info(f"{len(tasks)} tasks -> classifier")

    t0 = time.time()
    if is_token_cls:
        preds = predict_batch_token_cls(
            texts,
            model,
            tokenizer,
            id2label,
            args.device,
            args.batch_size,
        )
    else:
        preds = predict_batch_sequence_cls(
            texts,
            model,
            tokenizer,
            id2label,
            args.device,
            args.batch_size,
        )
    elapsed = time.time() - t0
    per_q = elapsed * 1000 / max(len(tasks), 1)

    sigs = {}
    for t, (label, conf) in zip(tasks, preds):
        sigs[t["task_id"]] = {
            args.label_field: label,
            "confidence": round(conf, 4) if isinstance(conf, float) else conf,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sigs, f, ensure_ascii=False, indent=2)

    dist = Counter(s[args.label_field] for s in sigs.values())
    logger.info(
        f"done: {len(sigs)} sigs in {elapsed:.1f}s ({per_q:.1f}ms/q) -> {args.out}"
    )
    print(f"{args.label_field} distribution:")
    for d, n in dist.most_common():
        print(f"  {d:20s} {n:4d}")


if __name__ == "__main__":
    main()
