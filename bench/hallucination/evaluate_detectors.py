"""Standalone detector benchmark — no router/Envoy needed.

Compares hallucination detectors (LettuceDetect v2 Qwen generative, mmBERT
encoder + taxonomy head, ...) on the same datasets and example-level metrics
as evaluate.py, plus character-level span P/R/F1 and taxonomy accuracy when
the dataset carries gold spans/categories.

Usage:
    pip install lettucedetect
    python3 -m bench.hallucination.evaluate_detectors \
        --detector llm:KRLabsOrg/lettucedect-v2-qwen-2b \
        --detector transformer:KRLabsOrg/lettucedect-v2-mmbert-base:KRLabsOrg/lettucedect-v2-taxonomy-head \
        --dataset halueval --max-samples 50
"""

import argparse
import json
import time
from pathlib import Path

from .datasets import get_dataset

try:
    from lettucedetect.models.inference import HallucinationDetector

    HAS_LETTUCEDETECT = True
except ImportError:
    HAS_LETTUCEDETECT = False

RESULTS_DIR = Path(__file__).parent / "results"


def build_detector(spec: str, base_url: str | None = None):
    """spec: 'llm:<model>' or 'transformer:<model_path>[:<taxonomy_head>]'"""
    if not HAS_LETTUCEDETECT:
        raise ImportError("lettucedetect not installed. Run: pip install lettucedetect")

    method, _, rest = spec.partition(":")
    if method == "llm":
        kwargs = {"base_url": base_url, "api_key": "none"} if base_url else {}
        return HallucinationDetector(method="llm", model=rest, **kwargs)
    model_path, _, taxonomy = rest.partition(":")
    kwargs = {"taxonomy_head": taxonomy} if taxonomy else {}
    return HallucinationDetector(method="transformer", model_path=model_path, **kwargs)


def char_set(spans, answer):
    """Character indices covered by spans; recovers offsets from text if absent."""
    chars = set()
    for s in spans:
        start, end = s.get("start"), s.get("end")
        if start is None and s.get("text"):
            start = answer.find(s["text"])
            end = start + len(s["text"]) if start >= 0 else None
        if start is not None and end is not None and start >= 0:
            chars.update(range(start, end))
    return chars


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def evaluate(detector, samples):
    ex = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    char_tp = char_fp = char_fn = 0
    tax_total = tax_cat_ok = tax_sub_ok = 0
    latencies, rows = [], []

    for sample in samples:
        answer = sample.llm_response or sample.gold_answer
        t0 = time.perf_counter()
        try:
            preds = detector.predict(
                context=[sample.context],
                question=sample.question,
                answer=answer,
                output_format="spans",
            )
        except Exception as e:
            print(f"  {sample.id}: predict failed: {e}")
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

        flagged = bool(preds)
        if sample.is_faithful is not None:
            key = (
                ("tp" if flagged else "fn")
                if not sample.is_faithful
                else ("fp" if flagged else "tn")
            )
            ex[key] += 1

        gold_spans = sample.hallucination_spans or []
        if gold_spans:
            pred_chars = char_set(preds, answer)
            gold_chars = char_set(gold_spans, answer)
            char_tp += len(pred_chars & gold_chars)
            char_fp += len(pred_chars - gold_chars)
            char_fn += len(gold_chars - pred_chars)

            for g in gold_spans:
                if not g.get("category"):
                    continue
                g_chars = char_set([g], answer)
                # ponytail: max-overlap span matching; alignment-based if ties matter
                best = max(
                    preds,
                    key=lambda p: len(char_set([p], answer) & g_chars),
                    default=None,
                )
                if best is None or not char_set([best], answer) & g_chars:
                    continue
                tax_total += 1
                tax_cat_ok += best.get("category") == g["category"]
                tax_sub_ok += best.get("subcategory") == g.get("subcategory")

        rows.append({"id": sample.id, "flagged": flagged, "spans": preds})

    ep, er, ef1 = prf(ex["tp"], ex["fp"], ex["fn"])
    total_ex = sum(ex.values())
    cp, cr, cf1 = prf(char_tp, char_fp, char_fn)
    latencies.sort()
    metrics = {
        "samples": len(rows),
        "example_level": {
            **ex,
            "precision": ep,
            "recall": er,
            "f1": ef1,
            "accuracy": (ex["tp"] + ex["tn"]) / total_ex if total_ex else None,
        },
        "char_level": (
            {"precision": cp, "recall": cr, "f1": cf1}
            if char_tp + char_fp + char_fn
            else None
        ),
        "taxonomy": (
            {
                "matched_spans": tax_total,
                "category_accuracy": tax_cat_ok / tax_total,
                "subcategory_accuracy": tax_sub_ok / tax_total,
            }
            if tax_total
            else None
        ),
        "latency_ms": (
            {
                "avg": sum(latencies) / len(latencies),
                "p50": latencies[len(latencies) // 2],
                "p99": latencies[int(len(latencies) * 0.99)],
            }
            if latencies
            else None
        ),
    }
    return metrics, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--detector",
        action="append",
        required=True,
        help="llm:<model> or transformer:<path>[:<taxonomy_head>]",
    )
    ap.add_argument("--dataset", default="halueval")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint for llm detectors (e.g. vLLM)",
    )
    args = ap.parse_args()

    samples = get_dataset(args.dataset).load(args.max_samples)
    RESULTS_DIR.mkdir(exist_ok=True)

    all_metrics = {}
    for spec in args.detector:
        print(f"\n=== {spec} ===")
        metrics, rows = evaluate(build_detector(spec, args.base_url), samples)
        all_metrics[spec] = metrics
        print(json.dumps(metrics, indent=2))
        safe = spec.replace("/", "_").replace(":", "-")
        out = RESULTS_DIR / f"detectors_{Path(args.dataset).stem}_{safe}.json"
        out.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2))
        print(f"Saved {out}")

    if len(all_metrics) > 1:
        print("\n=== Comparison ===")
        for spec, m in all_metrics.items():
            e, c = m["example_level"], m["char_level"]
            lat = m["latency_ms"]
            print(
                (
                    f"{spec}\n  example F1={e['f1']:.3f} acc={e['accuracy']}"
                    f"  char F1={c['f1']:.3f}"
                    if c
                    else f"{spec}\n  example F1={e['f1']:.3f}"
                ),
            )
            if lat:
                print(f"  latency avg={lat['avg']:.0f}ms p99={lat['p99']:.0f}ms")


def _self_test():
    answer = "Paris is in Germany and has 2M people."
    gold = [{"start": 9, "end": 19, "text": "in Germany", "category": "contradiction"}]
    pred = [{"text": "in Germany", "category": "contradiction"}]
    assert char_set(pred, answer) == char_set(gold, answer) == set(range(9, 19))
    assert prf(10, 0, 0) == (1.0, 1.0, 1.0)
    assert prf(0, 5, 5) == (0.0, 0.0, 0.0)
    print("self-test OK")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
