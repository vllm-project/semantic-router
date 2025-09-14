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
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# This benchmark supports two usage patterns:
# 1) Router-transparent: send a single neutral prompt; router/model decides reasoning.
# 2) vLLM 3-case evaluation: run realistic scenarios that match router decision patterns:
#    - NR: Plain prompt, no reasoning toggle (baseline/fast)
#    - XC: CoT prompt, no reasoning toggle (prompt-based reasoning)
#    - NR_REASONING: Plain prompt, reasoning toggle ON (model-based reasoning)


ANSWER_PATTERN = re.compile(r"(?:answer(?:\sis)?:?\s*)([A-J])", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReasonBench: evaluate router vs direct vLLM on MMLU-Pro with detailed metrics"
    )
    # Router endpoint (NR-only evaluation; router decides reasoning internally)
    parser.add_argument(
        "--router-endpoint",
        type=str,
        default=os.environ.get("ROUTER_ENDPOINT", "http://localhost:8000/v1"),
        help="Router endpoint URL (OpenAI-compatible)",
    )
    parser.add_argument(
        "--router-api-key",
        type=str,
        default=os.environ.get("ROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="API key for router endpoint",
    )
    parser.add_argument(
        "--router-models",
        type=str,
        nargs="+",
        default=["auto"],
        help="Router models to evaluate (default: auto).",
    )

    # Direct vLLM endpoint (multi-mode evaluation)
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default=os.environ.get("VLLM_ENDPOINT", ""),
        help="Direct vLLM endpoint URL (OpenAI-compatible)",
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

    # VLLM execution modes (prompt styles)
    parser.add_argument(
        "--vllm-exec-modes",
        type=str,
        nargs="+",
        default=["NR", "XC"],
        help="DEPRECATED: vLLM now runs 3 fixed realistic modes: NR (plain), XC (CoT), NR_REASONING (plain+toggle)",
    )
    parser.add_argument(
        "--run-router",
        action="store_true",
        help="Run router NR-only evaluation",
    )
    parser.add_argument(
        "--run-vllm",
        action="store_true",
        help="Run vLLM direct evaluation (NR/XC with reasoning ON/OFF)",
    )

    # Policy reporting options (removed)
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
        default=2048,
        help="Maximum number of tokens to generate",
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


def get_available_models(endpoint: str, api_key: str = "") -> List[str]:
    client = OpenAI(base_url=endpoint, api_key=api_key or None)
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception as e:
        print(f"Error communicating with endpoint to list models: {e}")
        return []


def load_mmlu_pro_dataset(
    categories: Optional[List[str]] = None,
    samples_per_category: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    df = pd.DataFrame(dataset)

    all_categories = sorted(df["category"].unique().tolist())

    if categories:
        df = df[df["category"].isin(categories)]
        if df.empty:
            valid_categories = ", ".join(all_categories)
            raise ValueError(
                f"No data found for specified categories. Valid categories are: {valid_categories}"
            )

    if samples_per_category:
        random.seed(seed)
        np.random.seed(seed)
        sampled_dfs = []
        for category in df["category"].unique():
            category_df = df[df["category"] == category]
            if len(category_df) > samples_per_category:
                sampled_df = category_df.sample(samples_per_category, random_state=seed)
                sampled_dfs.append(sampled_df)
            else:
                sampled_dfs.append(category_df)
        df = pd.concat(sampled_dfs)

    return df, all_categories


def format_plain_prompt(question: str, options: List[str]) -> str:
    letter_mapping = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
    }
    formatted_options = ""
    for i, option in enumerate(options):
        if option.lower() != "n/a":
            formatted_options += f"{letter_mapping[i]}) {option}\n"

    prompt = (
        f"Question: {question}\n\nOptions:\n{formatted_options}\n\n"
        "Please choose the correct answer from the options above. Provide your answer in the format 'Answer: [letter]'."
    )
    return prompt


def format_cot_prompt(question: str, options: List[str]) -> str:
    letter_mapping = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
    }
    formatted_options = ""
    for i, option in enumerate(options):
        if option.lower() != "n/a":
            formatted_options += f"{letter_mapping[i]}) {option}\n"

    prompt = (
        f"Question: {question}\n\nOptions:\n{formatted_options}\n\n"
        "Please solve this step-by-step, then provide your final answer in the format 'Answer: [letter]'."
    )
    return prompt


def format_explicit_cot_prompt(
    question: str, options: List[str], cot_content: Optional[str]
) -> str:
    """Use dataset CoT content explicitly alongside the question/options.

    Note: This prompt includes provided CoT content to test explicit CoT behavior.
    """
    letter_mapping = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
    }
    formatted_options = ""
    for i, option in enumerate(options):
        if option.lower() != "n/a":
            formatted_options += f"{letter_mapping[i]}) {option}\n"

    cot_section = f"\nExplanation: {cot_content}\n" if cot_content else "\n"
    prompt = (
        f"Question: {question}\n\nOptions:\n{formatted_options}"
        f"{cot_section}\nUse the explanation if helpful and provide your final answer in the format 'Answer: [letter]'."
    )
    return prompt


def extract_answer(response: Any) -> Optional[str]:
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

    match = ANSWER_PATTERN.search(response)
    if match:
        return match.group(1).upper()
    for char in reversed(response):
        if char.upper() in "ABCDEFGHIJ":
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
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body if extra_body else None,
        )
        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        return text, True, prompt_tokens, completion_tokens, total_tokens
    except Exception as e:
        print(f"❌ Model call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Model: {model}")
        print(f"   Endpoint: {getattr(client, '_base_url', 'unknown')}")
        print(f"   API key set: {'Yes' if getattr(client, 'api_key', None) else 'No'}")
        if hasattr(e, "response"):
            print(f"   HTTP status: {getattr(e.response, 'status_code', 'unknown')}")
            print(f"   Response text: {getattr(e.response, 'text', 'unknown')}")
        import traceback

        print(f"   Full traceback: {traceback.format_exc()}")
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
    question_data: Dict[str, Any],
    prompt_mode: str,
    max_tokens: int,
    temperature: float,
    ar_extra_body: Optional[Dict[str, Any]] = None,
    mode_label: Optional[str] = None,
) -> Dict[str, Any]:
    question = question_data["question"]
    options = question_data["options"]
    correct_answer = question_data["answer"]
    cot_content = (
        question_data.get("cot_content") if isinstance(question_data, dict) else None
    )

    if prompt_mode == "XC":
        # Prefer explicit CoT content from dataset when available
        prompt = format_explicit_cot_prompt(question, options, cot_content)
        extra_body = None
    elif prompt_mode == "AR":
        prompt = format_plain_prompt(question, options)
        extra_body = ar_extra_body
    else:  # NR or Router-Transparent
        prompt = format_plain_prompt(question, options)
        extra_body = None

    start_time = time.time()
    response_text, success, prompt_tokens, completion_tokens, total_tokens = call_model(
        client, model, prompt, max_tokens, temperature, extra_body=extra_body
    )
    end_time = time.time()

    predicted_answer = extract_answer(response_text) if success else None
    is_correct = (predicted_answer == correct_answer) if predicted_answer else False

    return {
        "mode": prompt_mode,
        "mode_label": mode_label or prompt_mode,
        "question_id": question_data["question_id"],
        "category": question_data["category"],
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
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
    df: pd.DataFrame,
    model: str,
    endpoint: str,
    api_key: str,
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
) -> pd.DataFrame:
    """
    Evaluate router in transparent mode - send plain prompts and let router decide reasoning.

    This represents the 'auto' mode where the router internally decides whether to use
    reasoning or not based on the question complexity.
    """
    client = OpenAI(base_url=endpoint, api_key=api_key or None)
    print(f"Using model: {model}, endpoint: {endpoint}")
    print(
        f"API key provided: {'Yes' if api_key else 'No'} (length: {len(api_key) if api_key else 0})"
    )

    results: List[Dict[str, Any]] = []
    questions_data = df.to_dict("records")

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        for question_data in questions_data:
            futures.append(
                executor.submit(
                    process_question_single,
                    client,
                    model,
                    question_data,
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
    df: pd.DataFrame,
    model: str,
    endpoint: str,
    api_key: str,
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
    exec_modes: List[str],
) -> pd.DataFrame:
    """Run vLLM with 3 realistic reasoning scenarios.

    The 3 scenarios represent real-world router decision patterns:
    1. NR - Plain prompt, no reasoning toggle (fast baseline)
    2. XC - CoT prompt, no reasoning toggle (prompt-based reasoning)
    3. NR_REASONING - Plain prompt, reasoning toggle ON (model-based reasoning)
    """
    client = OpenAI(base_url=endpoint, api_key=api_key or "dummy-key")
    print(f"Using vLLM model: {model}, endpoint: {endpoint}")

    results: List[Dict[str, Any]] = []
    questions_data = df.to_dict("records")

    # Define 3 realistic mode variants: (label, prompt_mode, reasoning_flag)
    # For DeepSeek and Qwen3 models, explicitly set reasoning flags for all modes
    model_lower = model.lower()
    is_deepseek_or_qwen = (
        (("ds" in model_lower) or ("deepseek" in model_lower))
        and ("v31" in model_lower or "v3.1" in model_lower or "v3" in model_lower)
    ) or ("qwen3" in model_lower)

    if is_deepseek_or_qwen:
        mode_variants: List[Tuple[str, str, Optional[bool]]] = [
            ("VLLM_NR", "NR", False),  # Plain prompt, reasoning OFF (baseline)
            ("VLLM_XC", "XC", False),  # CoT prompt, reasoning OFF (prompt reasoning)
            (
                "VLLM_NR_REASONING",
                "NR",
                True,
            ),  # Plain prompt, reasoning ON (model reasoning)
        ]
    else:
        mode_variants: List[Tuple[str, str, Optional[bool]]] = [
            ("VLLM_NR", "NR", None),  # Plain prompt, no toggle (baseline)
            ("VLLM_XC", "XC", None),  # CoT prompt, no toggle (prompt reasoning)
            (
                "VLLM_NR_REASONING",
                "NR",
                True,
            ),  # Plain prompt, toggle ON (model reasoning)
        ]

    def run_variants(q: Dict[str, Any]) -> List[Dict[str, Any]]:
        local_records: List[Dict[str, Any]] = []
        for label, prompt_mode, reasoning_flag in mode_variants:
            extra_body = build_extra_body_for_model(model, reasoning_flag)
            # Debug: print extra_body for first question to verify configuration
            if q == questions_data[0]:
                print(
                    f"  {label}: reasoning_flag={reasoning_flag}, extra_body={extra_body}"
                )
            rec = process_question_single(
                client,
                model,
                q,
                prompt_mode,
                max_tokens,
                temperature,
                ar_extra_body=extra_body,
                mode_label=label,
            )
            local_records.append(rec)
        return local_records

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(run_variants, q) for q in questions_data]
        for future in tqdm(
            futures, total=len(futures), desc=f"Evaluating {model} (vLLM modes)"
        ):
            results.extend(future.result())

    return pd.DataFrame(results)


def evaluate_model_policies(
    df: pd.DataFrame,
    model: str,
    endpoint: str,
    api_key: str,
    concurrent_requests: int,
    max_tokens: int,
    temperature: float,
    exec_modes: List[str],
    ar_extra_body: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    client = OpenAI(base_url=endpoint, api_key=api_key or None)
    print(f"Using model: {model}, endpoint: {endpoint}")

    # Run NR/XC/AR for each question to enable Oracle and policy simulation
    per_call_records: List[Dict[str, Any]] = []
    questions = df.to_dict("records")

    def run_all_modes(q: Dict[str, Any]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        # NR
        records.append(
            process_question_single(
                client, model, q, "NR", max_tokens, temperature, None
            )
        )
        # XC
        records.append(
            process_question_single(
                client, model, q, "XC", max_tokens, temperature, None
            )
        )
        # AR (optional)
        if ar_extra_body is not None:
            records.append(
                process_question_single(
                    client, model, q, "AR", max_tokens, temperature, ar_extra_body
                )
            )
        return records

    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(run_all_modes, q) for q in questions]
        for future in tqdm(
            futures, total=len(futures), desc=f"Evaluating {model} (policies)"
        ):
            per_call_records.extend(future.result())

    calls_df = pd.DataFrame(per_call_records)

    # Organize records by question_id
    grouped: Dict[str, pd.DataFrame] = {
        qid: sub for qid, sub in calls_df.groupby("question_id")
    }

    def choose_oracle(sub: pd.DataFrame) -> pd.Series:
        # Pick best correctness; tie-break by lower total_tokens then lower latency
        sub = sub.copy()
        sub["correct_int"] = sub["is_correct"].astype(int)
        # Replace NaNs for tie-breakers with large values
        sub["total_tokens_fill"] = sub["total_tokens"].fillna(1e9)
        sub["latency_fill"] = sub["response_time"].fillna(1e9)
        sub = sub.sort_values(
            by=["correct_int", "total_tokens_fill", "latency_fill"],
            ascending=[False, True, True],
        )
        return sub.iloc[0]

    def pick_by_policy(sub: pd.DataFrame, policy: str) -> pd.Series:
        if policy == "Always-NR":
            return sub[sub["mode"] == "NR"].iloc[0]
        if policy == "Always-XC":
            return sub[sub["mode"] == "XC"].iloc[0]
        if policy == "Always-AR":
            return sub[sub["mode"] == "AR"].iloc[0]
        if policy == "Oracle":
            return choose_oracle(sub)
        # Router-Transparent: just NR call here; handled elsewhere in router-transparent path
        return sub[sub["mode"] == "NR"].iloc[0]

    # Compute per-policy selections and metrics
    policies_to_compute = [m for m in exec_modes if m != "Router-Transparent"]
    policy_rows: Dict[str, List[pd.Series]] = {name: [] for name in policies_to_compute}

    for qid, sub in grouped.items():
        available_modes = set(sub["mode"].tolist())
        for policy in list(policy_rows.keys()):
            if "AR" in policy and "AR" not in available_modes:
                # Skip AR policies if AR not available
                continue
            try:
                chosen = pick_by_policy(sub, policy)
                policy_rows[policy].append(chosen)
            except Exception:
                continue

    policy_metrics: Dict[str, Dict[str, Any]] = {}
    for policy, rows in policy_rows.items():
        if not rows:
            continue
        dfp = pd.DataFrame(rows)
        metrics = analyze_results(dfp)
        # Regret vs Oracle
        oracle_rows = [choose_oracle(sub) for _, sub in grouped.items()]
        oracle_df = pd.DataFrame(oracle_rows)
        metrics["regret_accuracy"] = (
            (oracle_df["is_correct"].mean() - dfp["is_correct"].mean())
            if not dfp.empty
            else 0.0
        )
        metrics["regret_tokens"] = (
            (
                oracle_df["total_tokens"].dropna().mean()
                - dfp["total_tokens"].dropna().mean()
            )
            if not dfp["total_tokens"].dropna().empty
            and not oracle_df["total_tokens"].dropna().empty
            else None
        )

        policy_metrics[policy] = metrics

    return calls_df, policy_metrics


def analyze_results(results_df: pd.DataFrame) -> Dict[str, Any]:
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

    # Per-category metrics by mode and ranges across modes
    category_by_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}
    category_ranges: Dict[str, Dict[str, Dict[str, float]]] = {}
    if (
        "mode_label" in valid.columns
        and "category" in valid.columns
        and not valid.empty
    ):
        grouped = (
            valid.groupby(["category", "mode_label"]).agg(
                accuracy=("is_correct", "mean"),
                avg_response_time=("response_time", "mean"),
                avg_prompt_tokens=("prompt_tokens", "mean"),
                avg_completion_tokens=("completion_tokens", "mean"),
                avg_total_tokens=("total_tokens", "mean"),
            )
        ).reset_index()

        # Build nested dict {category: {mode_label: metrics}}
        for cat in grouped["category"].unique():
            cat_df = grouped[grouped["category"] == cat]
            mode_dict: Dict[str, Dict[str, Any]] = {}
            for _, row in cat_df.iterrows():
                mode_label = str(row["mode_label"])  # ensure JSON-safe key
                mode_dict[mode_label] = {
                    "accuracy": (
                        float(row["accuracy"]) if pd.notna(row["accuracy"]) else 0.0
                    ),
                    "avg_response_time": (
                        float(row["avg_response_time"])
                        if pd.notna(row["avg_response_time"])
                        else 0.0
                    ),
                    "avg_prompt_tokens": (
                        float(row["avg_prompt_tokens"])
                        if pd.notna(row["avg_prompt_tokens"])
                        else None
                    ),
                    "avg_completion_tokens": (
                        float(row["avg_completion_tokens"])
                        if pd.notna(row["avg_completion_tokens"])
                        else None
                    ),
                    "avg_total_tokens": (
                        float(row["avg_total_tokens"])
                        if pd.notna(row["avg_total_tokens"])
                        else None
                    ),
                }
            category_by_mode[cat] = mode_dict

            # Compute ranges (min/max across modes) for selected metrics
            def _mm(values: List[float]) -> Dict[str, float]:
                if not values:
                    return {"min": 0.0, "max": 0.0}
                return {
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }

            acc_vals = [
                v.get("accuracy", 0.0)
                for v in mode_dict.values()
                if v.get("accuracy") is not None
            ]
            lat_vals = [
                v.get("avg_response_time", 0.0)
                for v in mode_dict.values()
                if v.get("avg_response_time") is not None
            ]
            tok_vals = [
                v.get("avg_total_tokens")
                for v in mode_dict.values()
                if v.get("avg_total_tokens") is not None
            ]
            category_ranges[cat] = {
                "accuracy": _mm(acc_vals),
                "avg_response_time": _mm(lat_vals),
                "avg_total_tokens": _mm(tok_vals),
            }

    # Overall ranges across modes (not per-category)
    mode_ranges: Dict[str, Dict[str, float]] = {}
    if by_mode:

        def _mm(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"min": 0.0, "max": 0.0}
            return {"min": float(np.nanmin(values)), "max": float(np.nanmax(values))}

        acc_vals = [
            m["accuracy"] for m in by_mode.values() if m.get("accuracy") is not None
        ]
        lat_vals = [
            m["avg_response_time"]
            for m in by_mode.values()
            if m.get("avg_response_time") is not None
        ]
        tok_vals = [
            m.get("avg_total_tokens")
            for m in by_mode.values()
            if m.get("avg_total_tokens") is not None
        ]
        mode_ranges = {
            "accuracy": _mm(acc_vals),
            "avg_response_time": _mm(lat_vals),
            "avg_total_tokens": _mm(tok_vals),
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
        "category_by_mode": category_by_mode,
        "category_ranges": category_ranges,
        "mode_ranges": mode_ranges,
    }


def save_results(
    results_df: pd.DataFrame,
    analysis: Dict[str, Any],
    model: str,
    output_dir: str,
):
    model_name = model.replace("/", "_")
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    results_df.to_csv(os.path.join(model_dir, "detailed_results.csv"), index=False)

    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "model": model,
                **analysis,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 50)
    print(f"Model: {model}")
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

    # Charts
    try:
        os.makedirs(model_dir, exist_ok=True)
        # Per-category accuracy by mode (if multi-mode)
        dfp = results_df[results_df["success"]].copy()
        agg_cols = [
            ("accuracy", "is_correct", "mean"),
            ("avg_latency", "response_time", "mean"),
            ("avg_total_tokens", "total_tokens", "mean"),
        ]
        if not dfp.empty and "mode_label" in dfp.columns:
            grouped = dfp.groupby(["category", "mode_label"]).agg(
                {c: f for _, c, f in agg_cols}
            )
            grouped.columns = [a for a, _, _ in agg_cols]
            grouped = grouped.reset_index()
            # Accuracy chart
            plt.figure(figsize=(14, 7))
            sns.barplot(data=grouped, x="category", y="accuracy", hue="mode_label")
            plt.title(f"Per-category accuracy by mode: {model}")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "category_accuracy_by_mode.png"))
            plt.close()

            # Latency chart
            plt.figure(figsize=(14, 7))
            sns.barplot(data=grouped, x="category", y="avg_latency", hue="mode_label")
            plt.title(f"Per-category latency by mode: {model}")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "category_latency_by_mode.png"))
            plt.close()

            # Tokens chart
            plt.figure(figsize=(14, 7))
            sns.barplot(
                data=grouped, x="category", y="avg_total_tokens", hue="mode_label"
            )
            plt.title(f"Per-category total tokens by mode: {model}")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, "category_tokens_by_mode.png"))
            plt.close()
    except Exception as e:
        print(f"Plot generation failed: {e}")


def save_policy_results(
    calls_df: pd.DataFrame,
    policy_metrics: Dict[str, Dict[str, Any]],
    model: str,
    output_dir: str,
):
    model_name = model.replace("/", "_")
    model_dir = os.path.join(output_dir, f"{model_name}_policies")
    os.makedirs(model_dir, exist_ok=True)

    # Save raw calls
    calls_df.to_csv(os.path.join(model_dir, "per_call_results.csv"), index=False)

    with open(os.path.join(model_dir, "policy_metrics.json"), "w") as f:
        json.dump(policy_metrics, f, indent=2)

    print("\nPolicy Metrics:")
    for name, metrics in policy_metrics.items():
        print(
            f"- {name}: acc={metrics['overall_accuracy']:.4f}, avg_tokens={metrics['avg_total_tokens']}, avg_latency={metrics['avg_response_time']:.2f}s"
        )


def main():
    args = parse_args()

    # Resolve router endpoint/key
    router_endpoint = (
        args.router_endpoint
        or os.environ.get("ROUTER_ENDPOINT")
        or "http://localhost:8000/v1"
    )
    router_api_key = (
        args.router_api_key
        or os.environ.get("ROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )

    # Resolve vLLM endpoint/key
    vllm_endpoint = args.vllm_endpoint or os.environ.get("VLLM_ENDPOINT", "")
    vllm_api_key = (
        args.vllm_api_key
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    router_models = args.router_models
    if router_models and len(router_models) == 1 and "," in router_models[0]:
        router_models = router_models[0].split(",")
    if not router_models:
        print("Fetching available models from router endpoint...")
        router_models = get_available_models(router_endpoint, router_api_key)

    vllm_models = args.vllm_models
    if vllm_models and len(vllm_models) == 1 and "," in vllm_models[0]:
        vllm_models = vllm_models[0].split(",")
    if not vllm_models and vllm_endpoint:
        print("Fetching available models from vLLM endpoint...")
        vllm_models = get_available_models(vllm_endpoint, vllm_api_key)

    print(f"Router models: {router_models}")
    print(f"vLLM models: {vllm_models}")
    print("Loading MMLU-Pro dataset...")
    df, all_categories = load_mmlu_pro_dataset(
        categories=args.categories,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )

    if args.categories is None:
        print(f"Available categories: {all_categories}")

    print(f"Dataset loaded: {len(df)} questions")

    # Prepare AR extra_body if provided
    ar_extra_body = None
    if args.ar_extra_body:
        try:
            ar_extra_body = json.loads(args.ar_extra_body)
        except Exception as e:
            print(f"Failed to parse --ar-extra-body JSON: {e}")
            ar_extra_body = None

    # Router evaluation (NR-only)
    if args.run_router and router_endpoint and router_models:
        for model in router_models:
            print(f"\nEvaluating router model: {model}")
            rt_df = evaluate_model_router_transparent(
                df=df,
                model=model,
                endpoint=router_endpoint,
                api_key=router_api_key,
                concurrent_requests=args.concurrent_requests,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            analysis = analyze_results(rt_df)
            save_results(
                results_df=rt_df,
                analysis=analysis,
                model=f"router::{model}",
                output_dir=args.output_dir,
            )

    # Direct vLLM evaluation (NR/XC with reasoning ON/OFF)
    if args.run_vllm and vllm_endpoint and vllm_models:
        for model in vllm_models:
            print(f"\nEvaluating vLLM model: {model}")
            vdf = evaluate_model_vllm_multimode(
                df=df,
                model=model,
                endpoint=vllm_endpoint,
                api_key=vllm_api_key,
                concurrent_requests=args.concurrent_requests,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                exec_modes=args.vllm_exec_modes,
            )
            analysis = analyze_results(vdf)
            save_results(
                results_df=vdf,
                analysis=analysis,
                model=f"vllm::{model}",
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
