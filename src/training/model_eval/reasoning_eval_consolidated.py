#!/usr/bin/env python3
"""
Statistical Reasoning Mode Evaluation for LLM Router Configuration

A model evaluation framework that determines optimal reasoning mode
configurations for large language model routers using statistical
analysis and evidence-based decision making.

This module implements a multi-criteria statistical framework that evaluates
the effectiveness of reasoning modes across different task categories, generating
optimized router configurations based on empirical evidence rather than
arbitrary thresholds.

Statistical Methodology:
    - McNemar's test for paired binary outcomes
    - Fisher's exact test for unpaired comparisons
    - Cohen's h effect size analysis for practical significance
    - Bayesian evidence estimation using Beta-Binomial conjugate priors
    - Multi-pathway significance testing framework

Key Features:
    - Robust statistical inference for small sample sizes
    - Evidence-based decision making with transparent reasoning
    - Comprehensive effect size and practical significance analysis
    - Bayesian probability estimation for intuitive interpretation
    - Production-ready configuration generation

Usage:
    python reasoning_eval_consolidated.py \\
        --endpoint http://localhost:8000/v1 \\
        --samples-per-category 25 \\
        --output-config optimized_config.yaml

Outputs:
    - YAML configuration file with optimized reasoning decisions
    - CSV file containing detailed evaluation results
    - JSON file with comprehensive statistical analysis
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from openai import OpenAI
from scipy import stats
from tqdm import tqdm

# Regular expression pattern for extracting multiple choice answers from model responses
ANSWER_PATTERN = re.compile(r"(?:answer(?:\sis)?:?\s*)([A-J])", re.IGNORECASE)

# Statistical significance thresholds
ALPHA_STRICT = 0.05  # Traditional significance level
ALPHA_RELAXED = 0.10  # Relaxed significance level for borderline cases
EFFECT_SIZE_MEDIUM = 0.2  # Cohen's h threshold for medium effect size
BAYESIAN_THRESHOLD = 0.8  # Bayesian probability threshold for strong evidence
MIN_IMPROVEMENT = 0.10  # Minimum improvement threshold for Bayesian pathway

# Model API configuration
MAX_TOKENS_REASONING = (
    6144  # Maximum tokens for reasoning mode (allows full reasoning chains)
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the reasoning evaluation framework.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Statistical reasoning mode evaluation for LLM router configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # API Configuration
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="API endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (auto-detects OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API (sets endpoint to https://api.openai.com/v1)",
    )

    # Model Configuration
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="Models to evaluate (auto-discover if not specified)",
    )

    # Evaluation Parameters
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5,
        help="Number of questions per category (default: %(default)s)",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=4,
        help="Number of concurrent API requests (default: %(default)s)",
    )

    # Output Configuration
    parser.add_argument(
        "--output-config",
        type=str,
        default="config.yaml",
        help="Output configuration file path (default: %(default)s)",
    )
    parser.add_argument(
        "--config-template",
        type=str,
        default="",
        help="Path to configuration template YAML file",
    )

    # Statistical Parameters
    parser.add_argument(
        "--significance-level",
        type=float,
        default=ALPHA_STRICT,
        help=f"Statistical significance level (default: {ALPHA_STRICT})",
    )

    return parser.parse_args()


def get_models(endpoint: str, api_key: str) -> List[str]:
    """
    Discover available models from the specified API endpoint.

    Args:
        endpoint: API endpoint URL
        api_key: API authentication key

    Returns:
        List of available model identifiers

    Raises:
        None: Errors are logged and empty list is returned
    """
    try:
        # Validate OpenAI API requirements
        if "api.openai.com" in endpoint and not api_key:
            print(
                "ERROR: OpenAI API requires authentication. Please set OPENAI_API_KEY environment variable or use --api-key"
            )
            return []

        # Initialize API client
        client = OpenAI(base_url=endpoint, api_key=api_key or "dummy")
        models_response = client.models.list()
        model_list = [model.id for model in models_response.data]

        # Apply OpenAI-specific filtering for reasoning-capable models
        if "api.openai.com" in endpoint:
            reasoning_capable = ["gpt-5"]
            filtered_models = [
                model
                for model in model_list
                if any(capability in model.lower() for capability in reasoning_capable)
            ]

            if filtered_models:
                print(
                    f"Discovered {len(filtered_models)} reasoning-capable models: {filtered_models}"
                )
                return filtered_models
            else:
                print(
                    f"WARNING: No reasoning-capable models found. Available models: {model_list}"
                )
                return model_list

        print(f"Discovered {len(model_list)} models from endpoint")
        return model_list

    except Exception as e:
        print(f"ERROR: Failed to discover models from {endpoint}: {e}")
        return []


def build_reasoning_params(model: str, reasoning: bool) -> Optional[Dict[str, Any]]:
    """
    Construct model-specific reasoning parameters for API requests.

    Different model families use different parameter structures to enable
    reasoning mode. This function maps model identifiers to their appropriate
    reasoning parameter format.

    Args:
        model: Model identifier string
        reasoning: Whether to enable reasoning mode

    Returns:
        Dictionary of reasoning parameters, or None if no parameters needed

    Note:
        Parameter structures are based on model family conventions:
        - DeepSeek models: chat_template_kwargs.thinking
        - Qwen3 models: chat_template_kwargs.enable_thinking
        - GPT-OSS models: reasoning_effort parameter
    """
    model_lower = model.lower()

    # vLLM-hosted model families with specific reasoning implementations
    if "deepseek" in model_lower and re.search(r"v3(?:[._]?1(?:\.\d+)?)?", model_lower):
        # DeepSeek v3.x reasoning via chat template kwargs
        return {"chat_template_kwargs": {"thinking": reasoning}}
    elif "qwen3" in model_lower:
        # Qwen3 reasoning via chat template kwargs
        return {"chat_template_kwargs": {"enable_thinking": reasoning}}
    elif "gpt-oss" in model_lower:
        # GPT-OSS reasoning via effort parameter
        return {"reasoning_effort": "high" if reasoning else "low"}

    # Model does not support reasoning parameters
    return None


def format_prompt(question: str, options: List[str]) -> str:
    """Format MMLU-Pro prompt."""
    letters = "ABCDEFGHIJ"
    formatted = "\n".join(
        f"{letters[i]}) {opt}" for i, opt in enumerate(options) if opt.lower() != "n/a"
    )
    return f"Question: {question}\n\nOptions:\n{formatted}\n\nProvide your answer in the format 'Answer: [letter]'."


def extract_answer(response: str) -> Optional[str]:
    """Extract answer from model response."""
    if not response:
        return None
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
    extra_body: Optional[Dict] = None,
    debug_print: bool = False,
) -> Dict[str, Any]:
    """Call model and return result."""
    try:
        start = time.time()

        # Build request parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS_REASONING,
            "temperature": 0.0,
        }

        # Add reasoning parameters based on model type
        if extra_body:
            if "reasoning" in extra_body:
                # OpenAI reasoning parameter goes at top level
                params["reasoning"] = extra_body["reasoning"]
                if debug_print:
                    print(
                        f"🔧 OpenAI reasoning param: reasoning={extra_body['reasoning']}"
                    )
            else:
                # All vLLM parameters (including chat_template_kwargs) go in extra_body
                # The Python OpenAI client doesn't support chat_template_kwargs at top level
                params["extra_body"] = extra_body
                if debug_print:
                    print(f"🔧 vLLM extra_body: {extra_body}")
        elif debug_print:
            print(f"🔧 No reasoning params for {model}")

        response = client.chat.completions.create(**params)

        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)

        return {
            "response": text,
            "success": True,
            "time": time.time() - start,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": (
                getattr(usage, "completion_tokens", None) if usage else None
            ),
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
    except Exception as e:
        return {
            "response": str(e),
            "success": False,
            "time": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }


def evaluate_model(
    model: str, endpoint: str, api_key: str, df: pd.DataFrame, concurrent: int
) -> pd.DataFrame:
    """Evaluate model with NR and NR_REASONING modes."""
    client = OpenAI(base_url=endpoint, api_key=api_key or "dummy")
    print(f"Evaluating {model} with {len(df)} questions...")

    # Special handling for o1 models (always use reasoning, no separate modes)
    if "o1" in model.lower():
        print("Note: o1 models always use reasoning. Running single mode evaluation.")
        modes = [("NR_REASONING", True)]  # Only one mode for o1
    else:
        modes = [("NR", False), ("NR_REASONING", True)]

    # Create all tasks (question × mode combinations)
    tasks = []
    for _, row in df.iterrows():
        prompt = format_prompt(row["question"], row["options"])
        for mode, reasoning in modes:
            extra_body = build_reasoning_params(model, reasoning)
            tasks.append((row, prompt, mode, extra_body))

    print(f"Total tasks: {len(tasks)} ({len(df)} questions × {len(modes)} modes)")

    # Track debug printing with a counter
    debug_counter = {"count": 0}

    def process_task(task):
        row, prompt, mode, extra_body = task
        # Enable debug printing for first 2 tasks only
        debug_print = debug_counter["count"] < 2
        if debug_print:
            debug_counter["count"] += 1
        result = call_model(client, model, prompt, extra_body, debug_print=debug_print)
        predicted = extract_answer(result["response"]) if result["success"] else None

        return {
            "question_id": row["question_id"],
            "category": row["category"],
            "correct_answer": row["answer"],
            "predicted_answer": predicted,
            "is_correct": predicted == row["answer"] if predicted else False,
            "mode": mode,
            "success": result["success"],
            "response_time": result["time"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        }

    # Process tasks concurrently
    results = []
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        try:
            for future in tqdm(futures, desc=f"Evaluating {model}"):
                results.append(future.result())
        except KeyboardInterrupt:
            print("\n⚠️  Evaluation interrupted by user. Cancelling remaining tasks...")
            # Cancel remaining futures
            for future in futures:
                future.cancel()
            # Collect results from completed futures
            for future in futures:
                if future.done() and not future.cancelled():
                    try:
                        results.append(future.result())
                    except Exception:
                        pass  # Skip failed results
            if not results:
                print("❌ No results to save.")
                raise
            print(f"✅ Saved {len(results)} partial results.")

    return pd.DataFrame(results)


def analyze_reasoning_statistical(results_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze reasoning effectiveness using enhanced statistical methodology.

    This function implements a multi-criteria statistical framework that goes beyond
    simple p-value thresholds to make evidence-based decisions about reasoning mode
    effectiveness across different categories.

    STATISTICAL METHODOLOGY:
    =======================

    1. SIGNIFICANCE TESTING:
       - McNemar's test for paired data (same questions, different modes)
       - Fisher's exact test for unpaired data as fallback
       - Appropriate for small sample sizes with exact p-values

    2. EFFECT SIZE CALCULATION:
       - Cohen's h for proportion differences
       - Formula: h = 2 * (arcsin(√p₂) - arcsin(√p₁))
       - Provides standardized measure of practical significance

    3. BAYESIAN ANALYSIS:
       - Beta-binomial conjugate prior model
       - Uniform priors: Beta(1,1) for both conditions
       - Monte Carlo estimation of P(reasoning > baseline)

    4. MULTI-CRITERIA DECISION:
       Reasoning is enabled if ANY of these criteria are met:
       a) Traditional significance: p < 0.05
       b) Borderline significance: p < 0.10 AND |Cohen's h| > 0.2
       c) Bayesian evidence: P(reasoning > baseline) > 80% AND improvement > 10%

    Args:
        results_df (pd.DataFrame): Evaluation results with columns:
            - category: Question category
            - mode: 'NR' or 'NR_REASONING'
            - is_correct: Boolean correctness
            - total_tokens: Token usage
            - success: Whether evaluation succeeded

    Returns:
        Dict[str, Dict]: Statistical analysis results for each category containing:
            - use_reasoning: Boolean decision
            - reasoning_effort: 'low', 'medium', or 'high'
            - reason: Detailed explanation of decision
            - nr_accuracy: Baseline accuracy
            - nr_reasoning_accuracy: Reasoning mode accuracy
            - improvement: Absolute improvement
            - token_overhead: Token usage multiplier
            - p_value: Statistical significance
            - cohens_h: Effect size
            - bayesian_prob: Probability reasoning is better
            - sample_size: Number of questions evaluated
            - traditional_significant: p < 0.05
            - borderline_significant: p < 0.10 AND medium effect
            - bayesian_significant: Strong Bayesian evidence

    INTERPRETATION GUIDE:
    ====================
    - p_value < 0.05: Strong statistical evidence
    - 0.05 ≤ p_value < 0.10 with |h| > 0.2: Moderate evidence with meaningful effect
    - bayesian_prob > 80%: Strong probabilistic evidence
    - |cohens_h| > 0.2: Practically meaningful difference
    - improvement > 5%: Substantial practical benefit

    Example:
        >>> df = pd.read_csv('evaluation_results.csv')
        >>> decisions = analyze_reasoning_statistical(df)
        >>> enabled_categories = [cat for cat, dec in decisions.items()
        ...                      if dec['use_reasoning']]
    """
    decisions = {}

    for category in results_df["category"].unique():
        cat_df = results_df[
            (results_df["category"] == category) & (results_df["success"])
        ]

        nr_df = cat_df[cat_df["mode"] == "NR"]
        nr_reasoning_df = cat_df[cat_df["mode"] == "NR_REASONING"]

        if nr_df.empty or nr_reasoning_df.empty:
            decisions[category] = {
                "use_reasoning": False,
                "reasoning_effort": "low",
                "reason": "No data",
                "nr_accuracy": 0.0,
                "nr_reasoning_accuracy": 0.0,
                "improvement": 0.0,
                "token_overhead": 0.0,
                "p_value": 1.0,
                "sample_size": 0,
                "statistically_significant": False,
            }
            continue

        # Get binary results for statistical testing
        nr_results = nr_df["is_correct"].values
        nr_reasoning_results = nr_reasoning_df["is_correct"].values

        nr_acc = nr_results.mean()
        nr_reasoning_acc = nr_reasoning_results.mean()
        nr_tokens = nr_df["total_tokens"].dropna().mean() or 0
        nr_reasoning_tokens = nr_reasoning_df["total_tokens"].dropna().mean() or 0

        improvement = nr_reasoning_acc - nr_acc
        overhead = nr_reasoning_tokens / nr_tokens if nr_tokens > 0 else float("inf")

        # STEP 1: STATISTICAL SIGNIFICANCE TESTING
        # ========================================
        # We use different tests depending on whether we have paired data
        # (same questions evaluated in both modes) or unpaired data.

        if len(nr_results) == len(nr_reasoning_results):
            # PAIRED DATA: McNemar's Test
            # ---------------------------
            # McNemar's test is appropriate for paired binary data where the same
            # subjects (questions) are tested under two different conditions (modes).
            # It focuses on the discordant pairs - cases where the two modes disagree.

            # Create 2x2 contingency table for paired responses:
            #                    Reasoning Mode
            #                 Correct  Incorrect
            # NR Mode Correct    a        b      <- b = nr_only_correct
            #      Incorrect     c        d      <- c = reasoning_only_correct

            both_correct = np.sum((nr_results == 1) & (nr_reasoning_results == 1))  # a
            nr_only_correct = np.sum(
                (nr_results == 1) & (nr_reasoning_results == 0)
            )  # b
            reasoning_only_correct = np.sum(
                (nr_results == 0) & (nr_reasoning_results == 1)
            )  # c
            both_wrong = np.sum((nr_results == 0) & (nr_reasoning_results == 0))  # d

            # McNemar's test statistic focuses on discordant pairs (b + c)
            # Under null hypothesis: P(b) = P(c), so (b-c)² / (b+c) ~ χ²(1)
            discordant_pairs = nr_only_correct + reasoning_only_correct

            if discordant_pairs > 0:
                # Apply continuity correction for small samples: |b-c| - 1
                # This makes the test more conservative for small sample sizes
                mcnemar_stat = (
                    abs(nr_only_correct - reasoning_only_correct) - 1
                ) ** 2 / discordant_pairs
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
            else:
                # No discordant pairs means identical performance
                p_value = 1.0
        else:
            # UNPAIRED DATA: Fisher's Exact Test
            # ----------------------------------
            # When sample sizes differ, we can't pair the data, so we use
            # Fisher's exact test for 2x2 contingency tables. This test
            # provides exact p-values regardless of sample size.

            nr_correct = np.sum(nr_results)
            nr_total = len(nr_results)
            reasoning_correct = np.sum(nr_reasoning_results)
            reasoning_total = len(nr_reasoning_results)

            # Create 2x2 contingency table:
            #              Correct  Incorrect  Total
            # NR Mode        a        b         n1
            # Reasoning      c        d         n2
            contingency_table = [
                [nr_correct, nr_total - nr_correct],
                [reasoning_correct, reasoning_total - reasoning_correct],
            ]

            # Fisher's exact test computes exact p-value using hypergeometric distribution
            _, p_value = stats.fisher_exact(contingency_table, alternative="two-sided")

        # STEP 2: EFFECT SIZE CALCULATION
        # ===============================
        # Cohen's h measures the effect size for differences between proportions.
        # It's based on the arcsine transformation, which stabilizes variance
        # and makes the metric more interpretable across different base rates.

        # Formula: h = 2 * (arcsin(√p₂) - arcsin(√p₁))
        # Interpretation: |h| < 0.2 (small), 0.2-0.5 (medium), >0.5 (large)
        if nr_acc > 0 and nr_reasoning_acc > 0:
            cohens_h = 2 * (
                np.arcsin(np.sqrt(nr_reasoning_acc)) - np.arcsin(np.sqrt(nr_acc))
            )
        else:
            cohens_h = 0

        # STEP 3: BAYESIAN ANALYSIS
        # =========================
        # We use a Beta-Binomial conjugate prior model to estimate the probability
        # that reasoning mode is actually better than baseline. This provides an
        # intuitive probabilistic interpretation of the evidence.

        from scipy.stats import beta

        # Set up Beta posteriors with uniform priors Beta(1,1)
        # Posterior: Beta(successes + 1, failures + 1)
        nr_successes = np.sum(nr_results)
        nr_failures = len(nr_results) - nr_successes
        reasoning_successes = np.sum(nr_reasoning_results)
        reasoning_failures = len(nr_reasoning_results) - reasoning_successes

        nr_posterior = beta(nr_successes + 1, nr_failures + 1)
        reasoning_posterior = beta(reasoning_successes + 1, reasoning_failures + 1)

        # Monte Carlo estimation of P(reasoning_accuracy > baseline_accuracy)
        # This gives us the probability that reasoning mode is actually better
        np.random.seed(42)  # For reproducible results
        n_samples = 10000
        nr_samples = nr_posterior.rvs(n_samples)
        reasoning_samples = reasoning_posterior.rvs(n_samples)
        bayesian_prob = np.mean(reasoning_samples > nr_samples)

        # STEP 4: MULTI-CRITERIA DECISION FRAMEWORK
        # =========================================
        # We use three pathways to significance, allowing for different types
        # of evidence to support the decision to enable reasoning mode.

        # Apply multi-criteria decision framework
        traditional_significant = p_value < ALPHA_STRICT
        borderline_significant = (p_value < ALPHA_RELAXED) and (
            abs(cohens_h) > EFFECT_SIZE_MEDIUM
        )
        bayesian_significant = (bayesian_prob > BAYESIAN_THRESHOLD) and (
            improvement > MIN_IMPROVEMENT
        )

        # Aggregate significance across all pathways
        statistically_significant = (
            traditional_significant or borderline_significant or bayesian_significant
        )

        # Final decision requires both significance and positive improvement
        use_reasoning = statistically_significant and improvement > 0

        # Determine reasoning effort level based on improvement magnitude
        if improvement >= 0.15:
            effort = "high"
        elif improvement >= 0.10:
            effort = "medium"
        else:
            effort = "low"

        # Generate detailed reasoning explanation
        significance_criteria = []
        if traditional_significant:
            significance_criteria.append(f"Traditional significance (p={p_value:.3f})")
        if borderline_significant:
            significance_criteria.append(
                f"Borderline significance with medium effect (p={p_value:.3f}, h={cohens_h:.2f})"
            )
        if bayesian_significant:
            significance_criteria.append(
                f"Strong Bayesian evidence (P={bayesian_prob:.1%}, Δ={improvement:.1%})"
            )

        if not statistically_significant:
            reason = f"Insufficient evidence: p={p_value:.3f}, h={cohens_h:.2f}, P(better)={bayesian_prob:.1%}"
        elif improvement <= 0:
            reason = f"Significant but detrimental effect: {improvement:.1%} ({'; '.join(significance_criteria)})"
        else:
            reason = f"Evidence-based improvement: {improvement:.1%} ({'; '.join(significance_criteria)})"

        decisions[category] = {
            "use_reasoning": bool(use_reasoning),
            "reasoning_effort": effort,
            "reason": reason,
            "nr_accuracy": float(nr_acc),
            "nr_reasoning_accuracy": float(nr_reasoning_acc),
            "improvement": float(improvement),
            "token_overhead": float(overhead),
            "p_value": float(p_value),
            "sample_size": len(nr_results),
            "statistically_significant": bool(statistically_significant),
            "cohens_h": float(cohens_h),
            "bayesian_prob": float(bayesian_prob),
            "traditional_significant": bool(traditional_significant),
            "borderline_significant": bool(borderline_significant),
            "bayesian_significant": bool(bayesian_significant),
        }

    return decisions


def load_config_template(template_path: str = "") -> Dict:
    """Load configuration template from YAML file."""
    if not template_path:
        template_path = os.path.join(os.path.dirname(__file__), "config_template.yaml")

    try:
        with open(template_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"✅ Loaded config template from: {template_path}")
            return config
    except FileNotFoundError:
        print(f"⚠️  Warning: Template file not found at {template_path}")
        return None


def generate_config(
    results_df: pd.DataFrame,
    reasoning_decisions: Dict,
    model: str,
    template_path: str = "",
) -> Dict:
    """
    Generate router configuration from template with complete model and endpoint setup.

    Args:
        results_df: Evaluation results DataFrame
        reasoning_decisions: Statistical analysis decisions for each category
        model: Primary model identifier
        template_path: Path to configuration template file

    Returns:
        Complete router configuration dictionary
    """
    # Load template
    config = load_config_template(template_path)
    if config is None:
        print("❌ Cannot generate config without template. Exiting.")
        return None

    # Set model-specific values
    config["default_model"] = model

    # Configure vLLM endpoints
    if "vllm_endpoints" in config and config["vllm_endpoints"]:
        # Update the first endpoint with the actual model
        config["vllm_endpoints"][0]["models"] = [model]

    # Configure model-specific settings
    if "model_config" not in config:
        config["model_config"] = {}

    # Determine reasoning family based on model name
    model_lower = model.lower()
    if "qwen3" in model_lower:
        reasoning_family = "qwen3"
    elif "deepseek" in model_lower and re.search(
        r"v3(?:[._]?1(?:\.\d+)?)?", model_lower
    ):
        reasoning_family = "deepseek"
    elif "gpt-oss" in model_lower:
        reasoning_family = "gpt-oss"
    elif "gpt" in model_lower:
        reasoning_family = "gpt"
    else:
        reasoning_family = "gpt"  # Default fallback

    # Add model configuration
    config["model_config"][model] = {
        "reasoning_family": reasoning_family,
        "preferred_endpoints": ["endpoint1"],
        "pii_policy": {"allow_by_default": True},
    }

    # Add categories with reasoning decisions
    config["categories"] = []
    for category, decision in reasoning_decisions.items():
        # Get best accuracy for model scoring (use reasoning accuracy if enabled, otherwise baseline)
        cat_df = results_df[
            (results_df["category"] == category) & (results_df["success"])
        ]
        if decision["use_reasoning"]:
            best_acc = decision["nr_reasoning_accuracy"]
        else:
            best_acc = decision["nr_accuracy"]

        config["categories"].append(
            {
                "name": category,
                "use_reasoning": decision["use_reasoning"],
                "reasoning_description": f"Data-driven decision: {decision['reason']}",
                "reasoning_effort": decision["reasoning_effort"],
                "model_scores": [{"model": model, "score": float(best_acc)}],
            }
        )

    return config


def main():
    args = parse_args()

    # Handle OpenAI API setup
    if args.use_openai:
        args.endpoint = "https://api.openai.com/v1"
        print("Using OpenAI API endpoint")

    # Auto-detect API key from environment if not provided
    if not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY", "")
        if args.api_key:
            print("Using API key from OPENAI_API_KEY environment variable")

    # Validate API key for OpenAI
    if "api.openai.com" in args.endpoint and not args.api_key:
        print("❌ OpenAI API requires an API key. Please:")
        print("   1. Set OPENAI_API_KEY environment variable, or")
        print("   2. Use --api-key parameter")
        return

    print(f"Endpoint: {args.endpoint}")

    # Get models
    if not args.models:
        print("Auto-discovering models...")
        args.models = get_models(args.endpoint, args.api_key)
        if not args.models:
            print("No models found!")
            return
        print(f"Found models: {args.models}")
    else:
        print(f"Using specified models: {args.models}")

    # Load dataset
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    df = pd.DataFrame(dataset)

    # Sample questions per category
    if args.samples_per_category:
        sampled = []
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            sample_size = min(args.samples_per_category, len(cat_df))
            sampled.append(cat_df.sample(sample_size, random_state=42))
        df = pd.concat(sampled)

    print(
        f"Evaluating {len(df)} questions across {df['category'].nunique()} categories"
    )

    # Evaluate each model
    all_results = []
    try:
        for model in args.models:
            results_df = evaluate_model(
                model, args.endpoint, args.api_key, df, args.concurrent_requests
            )
            results_df["model"] = model
            all_results.append(results_df)
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user.")
        if not all_results:
            print("❌ No complete model evaluations to save.")
            return
        print(f"✅ Proceeding with {len(all_results)} completed model evaluations.")

    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Analyze reasoning effectiveness using statistical significance
    print("\nAnalyzing reasoning effectiveness using statistical significance...")
    reasoning_decisions = analyze_reasoning_statistical(combined_df)

    # Print analysis
    reasoning_enabled = sum(
        1 for d in reasoning_decisions.values() if d["use_reasoning"]
    )
    print(
        f"\nReasoning Analysis ({reasoning_enabled}/{len(reasoning_decisions)} categories enabled):"
    )
    for category, decision in reasoning_decisions.items():
        status = "✓ ENABLED" if decision["use_reasoning"] else "✗ DISABLED"
        acc_change = (
            f"{decision['nr_accuracy']:.1%} → {decision['nr_reasoning_accuracy']:.1%}"
        )
        tokens = f"{decision['token_overhead']:.1f}x tokens"
        p_val = f"p={decision['p_value']:.3f}"
        print(f"  {category}: {status}")
        print(f"    Accuracy: {acc_change} ({decision['improvement']:+.1%})")
        print(
            f"    Statistical: {p_val} ({'significant' if decision['statistically_significant'] else 'not significant'})"
        )
        print(
            f"    Effect size: h={decision['cohens_h']:.2f}, Bayesian prob={decision['bayesian_prob']:.1%}"
        )
        print(f"    Cost: {tokens}")
        print(f"    Reason: {decision['reason']}")
        print()

    # Generate config (use first model as default)
    config = generate_config(
        combined_df, reasoning_decisions, args.models[0], args.config_template
    )

    # Save config
    os.makedirs(
        (
            os.path.dirname(args.output_config)
            if os.path.dirname(args.output_config)
            else "."
        ),
        exist_ok=True,
    )
    with open(args.output_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Save detailed results
    results_file = args.output_config.replace(".yaml", "_results.csv")
    combined_df.to_csv(results_file, index=False)

    # Save reasoning analysis
    analysis_file = args.output_config.replace(".yaml", "_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(reasoning_decisions, f, indent=2)

    print(f"\n✅ Config saved to: {args.output_config}")
    print(f"📊 Results saved to: {results_file}")
    print(f"📈 Analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
