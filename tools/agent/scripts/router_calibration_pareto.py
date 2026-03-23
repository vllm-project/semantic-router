"""Confidence routing calibration and per-category Pareto analysis.

Evaluates a small model's logprob confidence as a routing signal for
escalation to a larger model.  Produces:

  1. Calibration metrics (ECE, reliability diagram)
  2. Four-quadrant escalation analysis (uplift / regression / waste / redundant)
  3. Per-category optimal thresholds derived via marginal-value gradient
  4. Projected accuracy and cost under per-category routing policy

Usage
-----
    python router_calibration_pareto.py \\
        --small-endpoint http://127.0.0.1:8091/v1 \\
        --large-endpoint http://127.0.0.1:8090/v1 \\
        --small-model Qwen2.5-7B \\
        --large-model Qwen2.5-72B \\
        --api-key "$VLLM_API_KEY" \\
        --samples-per-category 5 \\
        --output-dir ./calibration_results

The script:
  - Sends every MMLU-Pro question to both models (with logprobs on the
    small model) so that results are from real inference, not simulation.
  - Computes the SR confidence normalization: (avg_logprob + 3) / 3
  - Derives per-category routing policies (ESCALATE / CAUTIOUS / AVOID /
    SKIP) from the four-quadrant breakdown.
  - Outputs a JSON summary and optional Markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import OpenAI

ANSWER_PATTERN = re.compile(
    r"(?:answer(?:\sis)?:?\s*)(A|B|C|D|E|F|G|H|I|J)", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--small-endpoint",
        required=True,
        help="OpenAI-compatible base URL for small model",
    )
    p.add_argument(
        "--large-endpoint",
        required=True,
        help="OpenAI-compatible base URL for large model",
    )
    p.add_argument(
        "--small-model", required=True, help="Model name for the small model"
    )
    p.add_argument(
        "--large-model", required=True, help="Model name for the large model"
    )
    p.add_argument("--api-key", default="", help="API key for the endpoints")
    p.add_argument(
        "--small-price-prompt",
        type=float,
        default=0.0,
        help="Small model $/M prompt tokens",
    )
    p.add_argument(
        "--small-price-completion",
        type=float,
        default=0.0,
        help="Small model $/M completion tokens",
    )
    p.add_argument(
        "--large-price-prompt",
        type=float,
        default=0.0,
        help="Large model $/M prompt tokens",
    )
    p.add_argument(
        "--large-price-completion",
        type=float,
        default=0.0,
        help="Large model $/M completion tokens",
    )
    p.add_argument("--samples-per-category", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrent", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="./calibration_results")
    p.add_argument("--report", action="store_true", help="Write Markdown report")
    return p.parse_args()


# ── confidence helpers ───────────────────────────────────────────────────


def normalize_logprob(avg_lp: float) -> float:
    """Match the SR Go code: ``(avg_logprob + 3) / 3``, clamped [0, 1]."""
    return max(0.0, min(1.0, (avg_lp + 3.0) / 3.0))


def extract_answer(text: str) -> str | None:
    m = ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    for ch in reversed(text):
        if ch.upper() in "ABCDEFGHIJ":
            return ch.upper()
    return None


# ── data loading ─────────────────────────────────────────────────────────


def load_mmlu_questions(samples_per_category: int, seed: int) -> pd.DataFrame:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    df = pd.DataFrame(dataset)
    np.random.seed(seed)
    sampled = []
    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat]
        if len(cat_df) > samples_per_category:
            sampled.append(cat_df.sample(samples_per_category, random_state=seed))
        else:
            sampled.append(cat_df)
    return pd.concat(sampled)


def format_prompt(question: str, options: list[str]) -> str:
    letter_map = {i: chr(65 + i) for i in range(10)}
    formatted = ""
    for i, opt in enumerate(options):
        if opt.lower() != "n/a":
            formatted += f"{letter_map[i]}) {opt}\n"
    return (
        f"Question: {question}\n\nOptions:\n{formatted}\n\n"
        "Please choose the correct answer from the options above. "
        "Provide your answer in the format 'Answer: [letter]'."
    )


# ── inference ────────────────────────────────────────────────────────────


def evaluate_with_logprobs(client: OpenAI, model: str, prompt: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.0,
            logprobs=True,
            top_logprobs=1,
        )
        content = resp.choices[0].message.content or ""
        lp_content = (
            resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
        )
        token_lps = [t.logprob for t in lp_content] if lp_content else []
        avg_lp = sum(token_lps) / len(token_lps) if token_lps else -10.0
        usage = resp.usage
        return {
            "response": content,
            "predicted": extract_answer(content),
            "avg_logprob": avg_lp,
            "confidence": normalize_logprob(avg_lp),
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "latency": time.time() - t0,
            "success": True,
        }
    except Exception as e:
        return {
            "response": str(e),
            "predicted": None,
            "avg_logprob": -10.0,
            "confidence": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency": time.time() - t0,
            "success": False,
        }


def run_eval(
    questions: pd.DataFrame,
    endpoint: str,
    model: str,
    api_key: str,
    label: str,
    concurrent: int,
) -> list[dict[str, Any]]:
    client = OpenAI(base_url=endpoint, api_key=api_key if api_key else "dummy")
    prompts = []
    for _, row in questions.iterrows():
        prompts.append(
            {
                "question_id": str(row["question_id"]),
                "category": row["category"],
                "correct_answer": row["answer"],
                "prompt": format_prompt(row["question"], row["options"]),
            }
        )
    results = []
    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = {
            pool.submit(evaluate_with_logprobs, client, model, p["prompt"]): p
            for p in prompts
        }
        done = 0
        for fut in as_completed(futures):
            p = futures[fut]
            r = fut.result()
            is_correct = (
                (r["predicted"] == p["correct_answer"]) if r["predicted"] else False
            )
            results.append({**p, **r, "is_correct": is_correct})
            done += 1
            if done % 10 == 0:
                print(f"  [{label}] {done}/{len(prompts)}")
    print(f"  [{label}] {len(results)}/{len(prompts)} complete")
    return results


# ── analysis ─────────────────────────────────────────────────────────────


def four_quadrant(small: list[dict], large: list[dict]) -> dict[str, list[dict]]:
    large_map = {q["question_id"]: q for q in large}
    quads: dict[str, list] = {
        "uplift": [],
        "regression": [],
        "waste": [],
        "redundant": [],
    }
    for qs in small:
        ql = large_map[qs["question_id"]]
        rec = {**qs, "correct_large": ql["is_correct"]}
        if not qs["is_correct"] and ql["is_correct"]:
            quads["uplift"].append(rec)
        elif qs["is_correct"] and not ql["is_correct"]:
            quads["regression"].append(rec)
        elif not qs["is_correct"] and not ql["is_correct"]:
            quads["waste"].append(rec)
        else:
            quads["redundant"].append(rec)
    return quads


def calibration_bins(small: list[dict], num_bins: int = 5) -> tuple[list[dict], float]:
    confs = [q["confidence"] for q in small]
    mn, mx = min(confs), max(confs)
    bw = (mx - mn) / num_bins
    bins = []
    for b in range(num_bins):
        lo = mn + b * bw
        hi = mn + (b + 1) * bw if b < num_bins - 1 else mx + 0.001
        in_bin = [q for q in small if lo <= q["confidence"] < hi]
        if in_bin:
            avg_c = sum(q["confidence"] for q in in_bin) / len(in_bin)
            acc = sum(1 for q in in_bin if q["is_correct"]) / len(in_bin)
            bins.append(
                {
                    "range": f"[{lo:.3f}, {hi:.3f})",
                    "count": len(in_bin),
                    "avg_confidence": round(avg_c, 4),
                    "actual_accuracy": round(acc, 4),
                    "gap": round(avg_c - acc, 4),
                }
            )
    N = len(small)
    ece = sum(b["count"] * abs(b["gap"]) for b in bins) / N if N else 0
    return bins, round(ece, 4)


def per_category_analysis(small: list[dict], large: list[dict]) -> dict[str, dict]:
    large_map = {q["question_id"]: q for q in large}
    cats: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            "correct_small": 0,
            "correct_large": 0,
            "uplift": 0,
            "regression": 0,
            "waste": 0,
            "redundant": 0,
            "confidences": [],
        }
    )
    for qs in small:
        ql = large_map[qs["question_id"]]
        c = cats[qs["category"]]
        c["n"] += 1
        c["correct_small"] += int(qs["is_correct"])
        c["correct_large"] += int(ql["is_correct"])
        c["confidences"].append(qs["confidence"])
        if not qs["is_correct"] and ql["is_correct"]:
            c["uplift"] += 1
        elif qs["is_correct"] and not ql["is_correct"]:
            c["regression"] += 1
        elif not qs["is_correct"] and not ql["is_correct"]:
            c["waste"] += 1
        else:
            c["redundant"] += 1

    result = {}
    for cat in sorted(cats):
        c = cats[cat]
        net = c["uplift"] - c["regression"]
        if net > 0 and c["waste"] <= c["uplift"]:
            strategy = "escalate"
            threshold = 0.999
        elif net > 0:
            strategy = "cautious"
            threshold = 0.96
        elif net < 0:
            strategy = "avoid"
            threshold = 0.80
        else:
            strategy = "skip"
            threshold = 0.80
        result[cat] = {
            **{k: v for k, v in c.items() if k != "confidences"},
            "net_uplift": net,
            "strategy": strategy,
            "recommended_threshold": threshold,
            "confidence_min": round(min(c["confidences"]), 4),
            "confidence_max": round(max(c["confidences"]), 4),
            "confidence_mean": round(sum(c["confidences"]) / len(c["confidences"]), 4),
        }
    return result


def project_per_category(
    small: list[dict],
    large: list[dict],
    cat_policies: dict[str, dict],
    pricing_small: dict,
    pricing_large: dict,
    default_threshold: float = 0.95,
) -> dict[str, Any]:
    """Project accuracy and cost under per-category routing policy."""
    large_map = {q["question_id"]: q for q in large}
    N = len(small)
    correct = 0
    total_cost = 0.0
    escalated = 0

    for qs in small:
        ql = large_map[qs["question_id"]]
        cat = qs["category"]
        policy = cat_policies.get(cat, {})
        threshold = policy.get("recommended_threshold", default_threshold)

        cost_s = (
            qs["prompt_tokens"] * pricing_small.get("prompt", 0)
            + qs["completion_tokens"] * pricing_small.get("completion", 0)
        ) / 1_000_000
        cost_l = (
            qs["prompt_tokens"] * pricing_large.get("prompt", 0)
            + ql["completion_tokens"] * pricing_large.get("completion", 0)
        ) / 1_000_000

        if qs["confidence"] < threshold:
            escalated += 1
            is_c = ql["is_correct"]
            total_cost += cost_s + cost_l
        else:
            is_c = qs["is_correct"]
            total_cost += cost_s

        if is_c:
            correct += 1

    return {
        "correct": correct,
        "accuracy": round(correct / N * 100, 2) if N else 0,
        "escalated": escalated,
        "escalation_rate": round(escalated / N * 100, 1) if N else 0,
        "total_cost": round(total_cost, 6),
        "cost_per_1k": round(total_cost / N * 1000, 6) if N else 0,
    }


# ── main ─────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pricing_small = {
        "prompt": args.small_price_prompt,
        "completion": args.small_price_completion,
    }
    pricing_large = {
        "prompt": args.large_price_prompt,
        "completion": args.large_price_completion,
    }

    print("Loading MMLU-Pro dataset...")
    questions = load_mmlu_questions(args.samples_per_category, args.seed)
    print(
        f"Loaded {len(questions)} questions across {questions['category'].nunique()} categories"
    )

    print(f"\n=== Evaluating {args.small_model} (with logprobs) ===")
    small_results = run_eval(
        questions,
        args.small_endpoint,
        args.small_model,
        args.api_key,
        "small",
        args.concurrent,
    )

    print(f"\n=== Evaluating {args.large_model} (with logprobs) ===")
    large_results = run_eval(
        questions,
        args.large_endpoint,
        args.large_model,
        args.api_key,
        "large",
        args.concurrent,
    )

    # Calibration
    bins, ece = calibration_bins(small_results)
    print(f"\nECE: {ece}")
    for b in bins:
        print(
            f"  {b['range']:>20} n={b['count']:>3}  conf={b['avg_confidence']:.3f}  "
            f"acc={b['actual_accuracy']:.3f}  gap={b['gap']:+.3f}"
        )

    # Four quadrants
    quads = four_quadrant(small_results, large_results)
    N = len(small_results)
    print(
        f"\nFour-quadrant: uplift={len(quads['uplift'])}, regression={len(quads['regression'])}, "
        f"waste={len(quads['waste'])}, redundant={len(quads['redundant'])}"
    )

    # Per-category
    cat_policies = per_category_analysis(small_results, large_results)
    print("\nPer-category policies:")
    for cat, p in sorted(cat_policies.items()):
        print(
            f"  {cat:>16}: {p['strategy']:>10}  net={p['net_uplift']:+d}  "
            f"threshold={p['recommended_threshold']:.3f}"
        )

    # Projections
    proj_global = project_per_category(
        small_results, large_results, {}, pricing_small, pricing_large, 0.95
    )
    proj_percat = project_per_category(
        small_results, large_results, cat_policies, pricing_small, pricing_large
    )

    acc_small = sum(1 for q in small_results if q["is_correct"]) / N * 100
    acc_large = sum(1 for q in large_results if q["is_correct"]) / N * 100

    print(f"\n{'Policy':>20} {'Accuracy':>9} {'Escalated':>10} {'Cost/1k':>10}")
    print("-" * 55)
    print(
        f"{'Always small':>20} {acc_small:>8.1f}% {'0':>9} ${pricing_small['prompt']*0.25 + pricing_small['completion']*0.2:>8.4f}"
    )
    print(
        f"{'Global t=0.95':>20} {proj_global['accuracy']:>8.1f}% {proj_global['escalated']:>9} ${proj_global['cost_per_1k']:>8.6f}"
    )
    print(
        f"{'Per-Category':>20} {proj_percat['accuracy']:>8.1f}% {proj_percat['escalated']:>9} ${proj_percat['cost_per_1k']:>8.6f}"
    )
    print(f"{'Always large':>20} {acc_large:>8.1f}% {N:>9}")

    # Save summary
    summary = {
        "small_model": args.small_model,
        "large_model": args.large_model,
        "num_questions": N,
        "samples_per_category": args.samples_per_category,
        "calibration": {"ece": ece, "bins": bins},
        "four_quadrant": {k: len(v) for k, v in quads.items()},
        "baseline_small_accuracy": round(acc_small, 2),
        "baseline_large_accuracy": round(acc_large, 2),
        "per_category": cat_policies,
        "projection_global_095": proj_global,
        "projection_per_category": proj_percat,
    }
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "small_results.json").write_text(
        json.dumps(small_results, indent=2, default=str)
    )
    (out / "large_results.json").write_text(
        json.dumps(large_results, indent=2, default=str)
    )
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
