"""
Multi-Dataset Reasoning Benchmark

A comprehensive evaluation framework for comparing semantic router performance
against direct vLLM inference across various reasoning datasets.

Features:
- Dataset-agnostic architecture supporting MMLU, ARC, GPQA, TruthfulQA, CommonsenseQA, HellaSwag
- Optimized token limits per dataset complexity
- Multiple reasoning modes (NR, XC, NR_REASONING)
- Structured response parsing with robust answer extraction
- Comprehensive metrics and visualization
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm

from .dataset_factory import DatasetFactory, list_available_datasets
from .dataset_interface import DatasetInfo, Question, questions_to_dataframe

# Robust answer extraction patterns for structured response parsing
ANSWER_PATTERN_PRIMARY = re.compile(r"(?:answer\s*:?\s*)([A-Z])", re.IGNORECASE)
ANSWER_PATTERN_FINAL = re.compile(r"(?:final\s*answer\s*:?\s*)([A-Z])", re.IGNORECASE)
ANSWER_PATTERN_CONCLUSION = re.compile(
    r"(?:therefore|thus|so).*?([A-Z])", re.IGNORECASE
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Dataset Reasoning Benchmark: Comprehensive evaluation framework for semantic router vs direct vLLM"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        help="Dataset to evaluate on. Use --list-datasets to see available options.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )

    # Semantic router configuration
    parser.add_argument(
        "--router-endpoint",
        type=str,
        default=os.environ.get("ROUTER_ENDPOINT", "http://127.0.0.1:8801/v1"),
        help="Semantic router endpoint URL",
    )
    parser.add_argument(
        "--router-api-key",
        type=str,
        default=os.environ.get(
            "ROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "1234")
        ),
        help="API key for router endpoint",
    )
    parser.add_argument(
        "--router-models",
        type=str,
        nargs="+",
        default=["auto"],
        help="Router models to evaluate (default: auto).",
    )

    # Direct vLLM configuration
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default=os.environ.get("VLLM_ENDPOINT", ""),
        help="Direct vLLM endpoint URL",
    )
    parser.add_argument(
        "--vllm-api-key",
        type=str,
        default=os.environ.get("VLLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="API key for vLLM endpoint",
    )
    parser.add_argument(
        "--vllm-models",
        type=str,
        nargs="+",
        default=[],
        help="Direct vLLM models to evaluate (leave empty to fetch from endpoint).",
    )

    # vLLM reasoning modes
    parser.add_argument(
        "--vllm-exec-modes",
        type=str,
        nargs="+",
        default=["NR", "XC"],
        help="vLLM reasoning modes: NR (neutral), XC (chain-of-thought), NR_REASONING (reasoning-enabled)",
    )
    parser.add_argument(
        "--run-router",
        action="store_true",
        help="Evaluate semantic router performance",
    )
    parser.add_argument(
        "--run-vllm",
        action="store_true",
        help="Evaluate direct vLLM performance across multiple reasoning modes",
    )

    # Dataset filtering options
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="List of categories to evaluate. If not provided, all available categories will be used.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5,
        help="Number of questions to sample per category. If not provided, all questions will be used.",
    )

    # Execution options
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent requests to make",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/reasonbench",
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (default: dataset-optimal)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for text generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--ar-extra-body",
        type=str,
        default="",
        help=(
            'JSON string passed as extra_body for AR mode (e.g., \'{"reasoning":{"effort":"medium"}}\'). '
            "If empty, AR modes are disabled."
        ),
    )
    return parser.parse_args()


def get_dataset_optimal_tokens(dataset_info):
    """
    Determine optimal token limit based on dataset complexity and reasoning requirements.

    Token limits are optimized for structured response generation while maintaining
    efficiency across different reasoning complexity levels.
    """
    dataset_name = dataset_info.name.lower()
    difficulty = dataset_info.difficulty_level.lower()

    # Optimized token limits per dataset
    dataset_tokens = {
        "gpqa": 500,  # Graduate-level scientific reasoning
        "truthfulqa": 250,  # Misconception analysis
        "hellaswag": 250,  # Natural continuation reasoning
        "arc": 220,  # Elementary/middle school science
        "commonsenseqa": 300,  # Common sense reasoning
        "mmlu": 150 if difficulty == "undergraduate" else 200,  # Academic knowledge
    }

    # Find matching dataset
    for dataset_key, tokens in dataset_tokens.items():
        if dataset_key in dataset_name:
            return tokens

    # Default based on difficulty level
    difficulty_tokens = {"graduate": 300, "hard": 300, "moderate": 200, "easy": 150}

    return difficulty_tokens.get(difficulty, 200)


def get_available_models(endpoint: str, api_key: str = "") -> List[str]:
    """Get available models from an endpoint."""
    client = OpenAI(base_url=endpoint, api_key=api_key or None)
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        print(f"Error communicating with endpoint to list models: {e}")
        return []


def extract_answer(response: Any) -> Optional[str]:
    """Extract answer from model response."""
    # Normalize non-string responses into a string to be robust to providers
    # that return structured content (e.g., lists of parts or dicts).
    if response is None:
        return None

    if not isinstance(response, str):
        try:
            # Handle list-of-parts shapes
            if isinstance(response, list):
                parts: List[str] = []
                for part in response:
                    if isinstance(part, dict):
                        if "text" in part and isinstance(part["text"], str):
                            parts.append(part["text"])
                        elif "content" in part and isinstance(part["content"], str):
                            parts.append(part["content"])
                        else:
                            parts.append(str(part))
                    else:
                        parts.append(str(part))
                response = "\n".join(parts)
            # Handle dict shapes
            elif isinstance(response, dict):
                for key in ("content", "text", "reasoning_content"):
                    val = response.get(key) if isinstance(response, dict) else None
                    if isinstance(val, str) and val:
                        response = val
                        break
                else:
                    # Fallback to JSON stringification
                    response = json.dumps(response, ensure_ascii=False)
            else:
                response = str(response)
        except Exception:
            response = str(response)

    # Try multiple extraction patterns in order of preference
    patterns = [ANSWER_PATTERN_PRIMARY, ANSWER_PATTERN_FINAL, ANSWER_PATTERN_CONCLUSION]

    for pattern in patterns:
        match = pattern.search(response)
        if match:
            return match.group(1).upper()

    # Fallback 1: Look for standalone letters at end of response
    lines = response.strip().split("\n")
    for line in reversed(lines[-3:]):  # Check last 3 lines
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return line.upper()

    # Fallback 2: Find last letter in entire response
    for char in reversed(response):
        if char.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return char.upper()

    return None


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool, Optional[int], Optional[int], Optional[int]]:
    """Call model with given parameters."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body if extra_body else None,
        )
        # For reasoning models, content might be in reasoning_content instead of content
        message = response.choices[0].message
        text = message.content or getattr(message, "reasoning_content", None) or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        return text, True, prompt_tokens, completion_tokens, total_tokens
    except Exception as e:
        print(f"Model call failed: {e}")
        return "ERROR", False, None, None, None


def build_extra_body_for_model(
    model_name: str, reasoning: Optional[bool]
) -> Optional[Dict[str, Any]]:
    """Return an extra_body dict to toggle reasoning for a given model.

    - DeepSeek v3.1: {"chat_template_kwargs": {"thinking": true/false}}
    - GPT-OSS: {"reasoning_effort": "low|medium|high"} when ON; if not provided, then low
    """
    # reasoning: True -> ON, False -> OFF, None -> base (default behavior)

    lower = model_name.lower()
    if (("ds" in lower) or ("deepseek" in lower)) and (
        "v31" in lower or "v3.1" in lower or "v3" in lower
    ):
        if reasoning is True:
            return {"chat_template_kwargs": {"thinking": True}}
        elif reasoning is False:
            return {"chat_template_kwargs": {"thinking": False}}
        else:  # reasoning is None (base mode)
            # Base: do not set thinking for DeepSeek - let it use default behavior
            return None

    # Qwen3 family
    if "qwen3" in lower:
        if reasoning is True:
            return {"chat_template_kwargs": {"enable_thinking": True}}
        if reasoning is False:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    # GPT OSS family
    if "gpt-oss" in lower or "openai/gpt-oss" in lower or "gpt_oss" in lower:
        if reasoning is True:
            return {"reasoning_effort": "high"}
        elif reasoning is False:
            return {"reasoning_effort": "low"}
        else:  # reasoning is None (base mode)
            # Base: do not set reasoning_effort - let it use default behavior
            return None

    return None


def process_question_single(
    client: OpenAI,
    model: str,
    question: Question,
    dataset: Any,  # DatasetInterface
    prompt_mode: str,
    max_tokens: int,
    temperature: float,
    ar_extra_body: Optional[Dict[str, Any]] = None,
    mode_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single question with the model."""
    # Format prompt based on mode
    if prompt_mode == "XC":
        prompt = dataset.format_prompt(question, "explicit_cot")
        extra_body = None
    elif prompt_mode == "AR":
        prompt = dataset.format_prompt(question, "plain")
        extra_body = ar_extra_body
    else:  # NR or Router-Transparent
        prompt = dataset.format_prompt(question, "plain")
        extra_body = None

    start_time = time.time()
    response_text, success, prompt_tokens, completion_tokens, total_tokens = call_model(
        client, model, prompt, max_tokens, temperature, extra_body=extra_body
    )
    end_time = time.time()

    predicted_answer = extract_answer(response_text) if success else None

    # Compare predicted answer with correct answer (handle both letter and index formats)
    if predicted_answer and predicted_answer in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if isinstance(question.correct_answer, str):
            # Dataset stores answer as letter (e.g., MMLU: "F")
            is_correct = predicted_answer == question.correct_answer
        elif isinstance(question.correct_answer, int):
            # Dataset stores answer as index (e.g., CommonsenseQA: 1, ARC: 0)
            predicted_idx = ord(predicted_answer) - ord("A")
            is_correct = predicted_idx == question.correct_answer
        else:
            is_correct = False
    else:
        is_correct = False

    return {
        "mode": prompt_mode,
        "mode_label": mode_label or prompt_mode,
        "question_id": question.question_id,
        "category": question.category,
        "question": question.question,
        "options": question.options,
        "correct_answer": question.correct_answer,
        "model_response": response_text,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "response_time": end_time - start_time,
        "success": success,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def evaluate_model_router_transparent(
    questions: List[Question],
    dataset: Any,  # DatasetInterface
    model: str,
    endpoint: str,
    api_key: str,
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
) -> pd.DataFrame:
    """Evaluate model in router-transparent mode."""
    client = OpenAI(base_url=endpoint, api_key=api_key or None)
    print(f"Using model: {model}, endpoint: {endpoint}")

    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        for question in questions:
            futures.append(
                executor.submit(
                    process_question_single,
                    client,
                    model,
                    question,
                    dataset,
                    "NR",
                    max_tokens,
                    temperature,
                    None,
                    mode_label="Router_NR",
                )
            )

        for future in tqdm(
            futures, total=len(futures), desc=f"Evaluating {model} (Router-Transparent)"
        ):
            results.append(future.result())

    return pd.DataFrame(results)


def evaluate_model_vllm_multimode(
    questions: List[Question],
    dataset: Any,  # DatasetInterface
    model: str,
    endpoint: str,
    api_key: str,
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
    exec_modes: List[str],
) -> pd.DataFrame:
    """Run vLLM with 2-3 realistic reasoning scenarios.

    The scenarios represent real-world router decision patterns:
    1. NR - Plain prompt, no reasoning toggle (fast baseline) - ALWAYS included
    2. XC - CoT prompt, no reasoning toggle (prompt-based reasoning) - ONLY if dataset has CoT
    3. NR_REASONING - Plain prompt, reasoning toggle ON (model-based reasoning) - ALWAYS included
    """
    client = OpenAI(base_url=endpoint, api_key=api_key or "dummy-key")
    print(f"Using vLLM model: {model}, endpoint: {endpoint}")

    # Check if dataset has actual CoT content by examining sample questions
    has_cot_content = any(
        q.cot_content is not None and q.cot_content.strip() for q in questions[:10]
    )

    if has_cot_content:
        print(f"  Dataset has CoT content - using 3 modes: NR, XC, NR_REASONING")
    else:
        print(
            f"  Dataset lacks CoT content - using 2 modes: NR, NR_REASONING (skipping XC)"
        )

    results: List[Dict[str, Any]] = []

    # Define mode variants based on model type and CoT availability
    model_lower = model.lower()
    is_deepseek_or_qwen = (
        (("ds" in model_lower) or ("deepseek" in model_lower))
        and ("v31" in model_lower or "v3.1" in model_lower or "v3" in model_lower)
    ) or ("qwen3" in model_lower)

    # Base modes (always included)
    if is_deepseek_or_qwen:
        mode_variants: List[Tuple[str, str, Optional[bool]]] = [
            ("VLLM_NR", "NR", False),  # Plain prompt, reasoning OFF (baseline)
            (
                "VLLM_NR_REASONING",
                "NR",
                True,
            ),  # Plain prompt, reasoning ON (model reasoning)
        ]
    else:
        mode_variants: List[Tuple[str, str, Optional[bool]]] = [
            ("VLLM_NR", "NR", None),  # Plain prompt, no toggle (baseline)
            (
                "VLLM_NR_REASONING",
                "NR",
                True,
            ),  # Plain prompt, reasoning toggle ON (model reasoning)
        ]

    # Add XC mode only if dataset has CoT content
    if has_cot_content:
        if is_deepseek_or_qwen:
            mode_variants.insert(
                1, ("VLLM_XC", "XC", False)
            )  # Insert between NR and NR_REASONING
        else:
            mode_variants.insert(
                1, ("VLLM_XC", "XC", None)
            )  # Insert between NR and NR_REASONING

    def run_variants(q: Question) -> List[Dict[str, Any]]:
        local_records: List[Dict[str, Any]] = []
        for label, prompt_mode, reasoning_flag in mode_variants:
            extra_body = build_extra_body_for_model(model, reasoning_flag)
            # Debug: print extra_body for first question to verify configuration
            if q == questions[0]:
                print(
                    f"  {label}: reasoning_flag={reasoning_flag}, extra_body={extra_body}"
                )
            rec = process_question_single(
                client,
                model,
                q,
                dataset,
                prompt_mode,
                max_tokens,
                temperature,
                ar_extra_body=extra_body,
                mode_label=label,
            )
            local_records.append(rec)
        return local_records

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(run_variants, q) for q in questions]
        for future in tqdm(
            futures, total=len(futures), desc=f"Evaluating {model} (vLLM modes)"
        ):
            results.extend(future.result())

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze results and compute metrics."""
    valid = results_df[results_df["success"]]
    overall_acc = valid["is_correct"].mean() if not valid.empty else 0.0

    category_metrics: Dict[str, Dict[str, Any]] = {}
    for category in valid["category"].unique():
        sub = valid[valid["category"] == category]
        category_metrics[category] = {
            "accuracy": float(sub["is_correct"].mean()) if not sub.empty else 0.0,
            "avg_response_time": (
                float(sub["response_time"].mean()) if not sub.empty else 0.0
            ),
            "avg_prompt_tokens": (
                float(sub["prompt_tokens"].dropna().mean())
                if not sub["prompt_tokens"].dropna().empty
                else None
            ),
            "avg_completion_tokens": (
                float(sub["completion_tokens"].dropna().mean())
                if not sub["completion_tokens"].dropna().empty
                else None
            ),
            "avg_total_tokens": (
                float(sub["total_tokens"].dropna().mean())
                if not sub["total_tokens"].dropna().empty
                else None
            ),
        }

    avg_latency = valid["response_time"].mean() if not valid.empty else 0.0
    avg_prompt_tokens = (
        valid["prompt_tokens"].dropna().mean() if not valid.empty else None
    )
    avg_completion_tokens = (
        valid["completion_tokens"].dropna().mean() if not valid.empty else None
    )
    avg_total_tokens = (
        valid["total_tokens"].dropna().mean() if not valid.empty else None
    )

    # Optional: metrics by mode_label
    by_mode: Dict[str, Dict[str, Any]] = {}
    if "mode_label" in valid.columns:
        for label in valid["mode_label"].unique():
            sub = valid[valid["mode_label"] == label]
            by_mode[label] = {
                "accuracy": float(sub["is_correct"].mean()) if not sub.empty else 0.0,
                "avg_response_time": (
                    float(sub["response_time"].mean()) if not sub.empty else 0.0
                ),
                "avg_prompt_tokens": (
                    float(sub["prompt_tokens"].dropna().mean())
                    if not sub["prompt_tokens"].dropna().empty
                    else None
                ),
                "avg_completion_tokens": (
                    float(sub["completion_tokens"].dropna().mean())
                    if not sub["completion_tokens"].dropna().empty
                    else None
                ),
                "avg_total_tokens": (
                    float(sub["total_tokens"].dropna().mean())
                    if not sub["total_tokens"].dropna().empty
                    else None
                ),
            }

    return {
        "overall_accuracy": float(overall_acc),
        "category_metrics": category_metrics,
        "avg_response_time": float(avg_latency) if avg_latency is not None else 0.0,
        "avg_prompt_tokens": (
            float(avg_prompt_tokens) if avg_prompt_tokens is not None else None
        ),
        "avg_completion_tokens": (
            float(avg_completion_tokens) if avg_completion_tokens is not None else None
        ),
        "avg_total_tokens": (
            float(avg_total_tokens) if avg_total_tokens is not None else None
        ),
        "total_questions": int(len(results_df)),
        "successful_queries": int(len(valid)),
        "failed_queries": int(len(results_df) - len(valid)),
        "by_mode": by_mode,
    }


def save_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    model: str,
    dataset_name: str,
    output_dir: str,
):
    """Save results to files."""
    model_name = model.replace("/", "_")
    model_dir = os.path.join(output_dir, f"{dataset_name}_{model_name}")
    os.makedirs(model_dir, exist_ok=True)

    results_df.to_csv(os.path.join(model_dir, "detailed_results.csv"), index=False)

    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "model": model,
                "dataset": dataset_name,
                **analysis,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 50)
    print(f"Model: {model} | Dataset: {dataset_name}")
    print(f"Overall Accuracy: {analysis['overall_accuracy']:.4f}")
    print(f"Total Questions: {analysis['total_questions']}")
    print(f"Successful Queries: {analysis['successful_queries']}")
    print(f"Failed Queries: {analysis['failed_queries']}")
    print(
        f"Avg Latency: {analysis['avg_response_time']:.2f}s | Avg Total Tokens: {analysis['avg_total_tokens']}"
    )
    print("=" * 50 + "\n")

    if "category_metrics" in analysis:
        print("Category Metrics (acc | latency | total_tokens):")
        printable = []
        for category, met in analysis["category_metrics"].items():
            printable.append((category, met.get("accuracy", 0.0)))
        for category, acc in sorted(printable, key=lambda x: x[1], reverse=True):
            m = analysis["category_metrics"][category]
            print(
                f"  {category}: acc={m['accuracy']:.4f}, latency={m['avg_response_time']:.2f}s, tokens={m['avg_total_tokens']}"
            )
        print()


def main():
    args = parse_args()

    # Handle dataset listing
    if args.list_datasets:
        list_available_datasets()
        return

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = DatasetFactory.create_dataset(args.dataset)
        questions, dataset_info = dataset.load_dataset(
            categories=args.categories,
            samples_per_category=args.samples_per_category,
            seed=args.seed,
        )
        print(
            f"Dataset loaded: {len(questions)} questions across {len(dataset_info.categories)} categories"
        )
        print(f"Categories: {', '.join(dataset_info.categories)}")

        # Check for empty dataset
        if len(questions) == 0:
            print(f"❌ No questions loaded from dataset '{args.dataset}'")
            print("This could be due to:")
            print("  - Dataset requiring authentication (gated dataset)")
            print("  - Network connectivity issues")
            print("  - Invalid dataset name or configuration")
            print("\nTry a different dataset:")
            list_available_datasets()
            return

    except Exception as e:
        print(f"Error loading dataset '{args.dataset}': {e}")
        print("\nAvailable datasets:")
        list_available_datasets()
        return

    # Resolve endpoints and models
    router_endpoint = (
        args.router_endpoint
        or os.environ.get("ROUTER_ENDPOINT")
        or "http://127.0.0.1:8801/v1"
    )
    router_api_key = (
        args.router_api_key
        or os.environ.get("ROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "1234"
    )

    vllm_endpoint = args.vllm_endpoint or os.environ.get("VLLM_ENDPOINT", "")
    vllm_api_key = (
        args.vllm_api_key
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )

    router_models = args.router_models
    if router_models and len(router_models) == 1 and "," in router_models[0]:
        router_models = router_models[0].split(",")
    if not router_models or (len(router_models) == 1 and router_models[0] == "auto"):
        print("Fetching available models from router endpoint...")
        fetched_models = get_available_models(router_endpoint, router_api_key)
        if fetched_models:
            router_models = fetched_models
        else:
            print("No models returned from endpoint, using 'auto' as fallback")
            router_models = ["auto"]

    vllm_models = args.vllm_models
    if vllm_models and len(vllm_models) == 1 and "," in vllm_models[0]:
        vllm_models = vllm_models[0].split(",")
    if not vllm_models and vllm_endpoint:
        print("Fetching available models from vLLM endpoint...")
        vllm_models = get_available_models(vllm_endpoint, vllm_api_key)

    print(f"Router models: {router_models}")
    print(f"vLLM models: {vllm_models}")

    # Determine optimal token limit for this dataset
    if args.max_tokens:
        optimal_tokens = args.max_tokens
        print(f"Using user-specified max_tokens: {optimal_tokens}")
    else:
        optimal_tokens = get_dataset_optimal_tokens(dataset_info)
        print(
            f"Using dataset-optimal max_tokens: {optimal_tokens} (for {dataset_info.name})"
        )

    # Router evaluation (NR-only)
    if args.run_router and router_endpoint and router_models:
        for model in router_models:
            print(f"\nEvaluating router model: {model}")
            rt_df = evaluate_model_router_transparent(
                questions=questions,
                dataset=dataset,
                model=model,
                endpoint=router_endpoint,
                api_key=router_api_key,
                concurrent_requests=args.concurrent_requests,
                max_tokens=optimal_tokens,
                temperature=args.temperature,
            )
            analysis = analyze_results(rt_df)
            save_results(
                results_df=rt_df,
                analysis=analysis,
                model=f"router::{model}",
                dataset_name=dataset_info.name,
                output_dir=args.output_dir,
            )

    # Direct vLLM evaluation (NR/XC with reasoning ON/OFF)
    if args.run_vllm and vllm_endpoint and vllm_models:
        for model in vllm_models:
            print(f"\nEvaluating vLLM model: {model}")
            vdf = evaluate_model_vllm_multimode(
                questions=questions,
                dataset=dataset,
                model=model,
                endpoint=vllm_endpoint,
                api_key=vllm_api_key,
                concurrent_requests=args.concurrent_requests,
                max_tokens=optimal_tokens,
                temperature=args.temperature,
                exec_modes=args.vllm_exec_modes,
            )
            analysis = analyze_results(vdf)
            save_results(
                results_df=vdf,
                analysis=analysis,
                model=f"vllm::{model}",
                dataset_name=dataset_info.name,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
