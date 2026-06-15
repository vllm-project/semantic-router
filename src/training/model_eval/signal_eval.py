"""
Signal Evaluation Script
========================

Evaluates the accuracy of router's signal extraction (domain, fact_check, user_feedback).
Uses the eval API to get signal outputs and compares them with ground truth labels.

Usage:
    python src/training/model_eval/signal_eval.py \
        --dimension domain \
        --endpoint http://localhost:8080/v1/eval \
        --max_samples 100 \
        --output results/signal_eval_domain.json
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from datasets import load_dataset
from tqdm import tqdm

# Import constants from mom_collection_eval
try:
    from .constants import MODEL_REGISTRY
except ImportError:
    from constants import MODEL_REGISTRY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("SignalEval")

# Dataset registry - each dataset has a unique ID
# To add new datasets (e.g., multilingual), just add new entries here
DATASET_REGISTRY = {
    # Domain classification datasets
    "mmlu-pro-en": {
        "dimension": "domain",
        "name": "MMLU-Pro (English)",
        "description": "14 STEM and humanities categories in English",
        "hf_dataset": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",  # Field in matched_signals
        "label_mapping": None,  # No mapping needed, labels match signal names
    },
    # MMLU-ProX multilingual datasets (29 languages)
    # Each language is a separate config with validation/test splits
    "mmlu-prox-af": {
        "dimension": "domain",
        "name": "MMLU-ProX (Afrikaans)",
        "description": "Multilingual MMLU-ProX in Afrikaans",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "af",  # Language config
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-ar": {
        "dimension": "domain",
        "name": "MMLU-ProX (Arabic)",
        "description": "Multilingual MMLU-ProX in Arabic",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "ar",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-bn": {
        "dimension": "domain",
        "name": "MMLU-ProX (Bengali)",
        "description": "Multilingual MMLU-ProX in Bengali",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "bn",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-cs": {
        "dimension": "domain",
        "name": "MMLU-ProX (Czech)",
        "description": "Multilingual MMLU-ProX in Czech",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "cs",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-zh": {
        "dimension": "domain",
        "name": "MMLU-ProX (Chinese)",
        "description": "Multilingual MMLU-ProX in Chinese",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "zh",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-de": {
        "dimension": "domain",
        "name": "MMLU-ProX (German)",
        "description": "Multilingual MMLU-ProX in German",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "de",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-en": {
        "dimension": "domain",
        "name": "MMLU-ProX (English)",
        "description": "MMLU-Pro in English (ProX version)",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "en",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-es": {
        "dimension": "domain",
        "name": "MMLU-ProX (Spanish)",
        "description": "Multilingual MMLU-ProX in Spanish",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "es",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-fr": {
        "dimension": "domain",
        "name": "MMLU-ProX (French)",
        "description": "Multilingual MMLU-ProX in French",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "fr",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-hi": {
        "dimension": "domain",
        "name": "MMLU-ProX (Hindi)",
        "description": "Multilingual MMLU-ProX in Hindi",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "hi",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-hu": {
        "dimension": "domain",
        "name": "MMLU-ProX (Hungarian)",
        "description": "Multilingual MMLU-ProX in Hungarian",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "hu",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-id": {
        "dimension": "domain",
        "name": "MMLU-ProX (Indonesian)",
        "description": "Multilingual MMLU-ProX in Indonesian",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "id",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-it": {
        "dimension": "domain",
        "name": "MMLU-ProX (Italian)",
        "description": "Multilingual MMLU-ProX in Italian",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "it",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-ja": {
        "dimension": "domain",
        "name": "MMLU-ProX (Japanese)",
        "description": "Multilingual MMLU-ProX in Japanese",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "ja",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-ko": {
        "dimension": "domain",
        "name": "MMLU-ProX (Korean)",
        "description": "Multilingual MMLU-ProX in Korean",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "ko",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-mr": {
        "dimension": "domain",
        "name": "MMLU-ProX (Marathi)",
        "description": "Multilingual MMLU-ProX in Marathi",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "mr",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-ne": {
        "dimension": "domain",
        "name": "MMLU-ProX (Nepali)",
        "description": "Multilingual MMLU-ProX in Nepali",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "ne",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-pt": {
        "dimension": "domain",
        "name": "MMLU-ProX (Portuguese)",
        "description": "Multilingual MMLU-ProX in Portuguese",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "pt",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-ru": {
        "dimension": "domain",
        "name": "MMLU-ProX (Russian)",
        "description": "Multilingual MMLU-ProX in Russian",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "ru",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-sr": {
        "dimension": "domain",
        "name": "MMLU-ProX (Serbian)",
        "description": "Multilingual MMLU-ProX in Serbian",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "sr",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-sw": {
        "dimension": "domain",
        "name": "MMLU-ProX (Swahili)",
        "description": "Multilingual MMLU-ProX in Swahili",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "sw",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-te": {
        "dimension": "domain",
        "name": "MMLU-ProX (Telugu)",
        "description": "Multilingual MMLU-ProX in Telugu",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "te",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-uk": {
        "dimension": "domain",
        "name": "MMLU-ProX (Ukrainian)",
        "description": "Multilingual MMLU-ProX in Ukrainian",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "uk",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-ur": {
        "dimension": "domain",
        "name": "MMLU-ProX (Urdu)",
        "description": "Multilingual MMLU-ProX in Urdu",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "ur",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-vi": {
        "dimension": "domain",
        "name": "MMLU-ProX (Vietnamese)",
        "description": "Multilingual MMLU-ProX in Vietnamese",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "vi",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-wo": {
        "dimension": "domain",
        "name": "MMLU-ProX (Wolof)",
        "description": "Multilingual MMLU-ProX in Wolof",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "wo",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-yo": {
        "dimension": "domain",
        "name": "MMLU-ProX (Yoruba)",
        "description": "Multilingual MMLU-ProX in Yoruba",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "yo",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-th": {
        "dimension": "domain",
        "name": "MMLU-ProX (Thai)",
        "description": "Multilingual MMLU-ProX in Thai",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "th",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    "mmlu-prox-zu": {
        "dimension": "domain",
        "name": "MMLU-ProX (Zulu)",
        "description": "Multilingual MMLU-ProX in Zulu",
        "hf_dataset": "li-lab/MMLU-ProX",
        "hf_config": "zu",
        "split": "test",
        "text_col": "question",
        "label_col": "category",
        "signal_field": "domains",
        "label_mapping": None,
    },
    # Fact check datasets
    "fact-check-en": {
        "dimension": "fact_check",
        "name": "Fact Check (English)",
        "description": "Binary fact-check classification",
        "hf_dataset": "llm-semantic-router/fact-check-classification-dataset",
        "split": "test",
        "text_col": "text",
        "label_col": "label_id",
        "signal_field": "fact_check",  # Field in matched_signals
        "label_mapping": {
            0: "no_fact_check_needed",  # label_id 0 = NO_FACT_CHECK_NEEDED
            1: "needs_fact_check",  # label_id 1 = FACT_CHECK_NEEDED
        },
    },
    # User feedback datasets
    "feedback-en": {
        "dimension": "user_feedback",
        "name": "User Feedback (English)",
        "description": "4-class user feedback detection",
        "hf_dataset": "llm-semantic-router/feedback-detector-dataset",
        "split": "validation",
        "text_col": "text",
        "label_col": "label_name",  # Actual column name in dataset
        "signal_field": "user_feedback",  # Field in matched_signals
        "label_mapping": {
            "SAT": "satisfied",
            "NEED_CLARIFICATION": "need_clarification",
            "WANT_DIFFERENT": "want_different",
            "WRONG_ANSWER": "wrong_answer",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Signal Evaluation Script")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset ID to evaluate (e.g., mmlu-pro-en, fact-check-en, feedback-en)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8080/v1/eval",
        help="Eval API endpoint URL",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1 for sequential)",
    )
    return parser.parse_args()


def load_dataset_by_id(dataset_id: str, max_samples: Optional[int] = None):
    """Load dataset by dataset ID from registry."""
    config = DATASET_REGISTRY[dataset_id]
    dimension = config["dimension"]
    logger.info(
        f"Loading dataset '{dataset_id}': {config['hf_dataset']} (split: {config['split']})"
    )

    try:
        # Load dataset with optional config parameter
        if config.get("hf_config"):
            ds = load_dataset(
                config["hf_dataset"], config["hf_config"], split=config["split"]
            )
            logger.info(f"Using config: {config['hf_config']}")
        else:
            ds = load_dataset(config["hf_dataset"], split=config["split"])

        # Filter for domain dimension (only keep MMLU-Pro categories)
        if dimension == "domain":
            valid_categories = [
                "biology",
                "business",
                "chemistry",
                "computer science",
                "economics",
                "engineering",
                "health",
                "history",
                "law",
                "math",
                "philosophy",
                "physics",
                "psychology",
                "other",
            ]
            ds = ds.filter(lambda x: x[config["label_col"]] in valid_categories)

        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))

        logger.info(f"Loaded {len(ds)} samples")
        return ds, config
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def call_eval_api(query: str, endpoint: str, timeout: int) -> Optional[Dict]:
    """Call the eval API and return the response."""
    try:
        response = requests.post(
            endpoint,
            json={"text": query},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"API call failed: {e}")
        return None


def extract_signal_output(api_response: Dict, signal_field: str) -> Optional[str]:
    """Extract signal output from API response."""
    try:
        matched_signals = api_response.get("decision_result", {}).get(
            "matched_signals", {}
        )
        signal_list = matched_signals.get(signal_field, [])

        if not signal_list or len(signal_list) == 0:
            return None

        # Return the first element (signal only returns one result)
        return signal_list[0]
    except Exception as e:
        logger.warning(f"Failed to extract signal: {e}")
        return None


def map_label(label, label_mapping: Optional[Dict]) -> str:
    """Map dataset label to signal name.

    Args:
        label: The label from the dataset (can be str, int, or other types)
        label_mapping: Optional mapping from dataset labels to signal names

    Returns:
        The mapped signal name as a string
    """
    if label_mapping is None:
        return str(label)

    # Try to get the mapping, return the label as string if not found
    mapped = label_mapping.get(label, label)
    return str(mapped)


def evaluate_single_sample(
    sample: Dict,
    config: Dict,
    endpoint: str,
    timeout: int,
) -> Dict:
    """Evaluate a single sample and return the result."""
    query = sample[config["text_col"]]
    expected_label = sample[config["label_col"]]
    expected_signal = map_label(expected_label, config["label_mapping"])

    # Call eval API
    api_response = call_eval_api(query, endpoint, timeout)

    if api_response is None:
        # API call failed, skip this sample
        return {
            "query": query[:200],  # Truncate long queries
            "expected": expected_signal,
            "actual": None,
            "status": "skip",
            "reason": "API call failed",
        }

    # Extract signal output
    actual_signal = extract_signal_output(api_response, config["signal_field"])

    if actual_signal is None:
        # No signal detected, skip this sample
        return {
            "query": query[:200],
            "expected": expected_signal,
            "actual": None,
            "status": "skip",
            "reason": "No signal detected",
        }

    # Compare expected and actual
    is_correct = actual_signal == expected_signal
    status = "correct" if is_correct else "incorrect"

    return {
        "query": query[:200],
        "expected": expected_signal,
        "actual": actual_signal,
        "status": status,
    }


def evaluate_dataset(
    dataset_id: str,
    endpoint: str,
    max_samples: Optional[int],
    timeout: int,
    concurrent: int = 1,
) -> Dict:
    """Evaluate a single dataset."""
    dataset, config = load_dataset_by_id(dataset_id, max_samples)
    dimension = config["dimension"]

    results = {
        "dataset_id": dataset_id,
        "dimension": dimension,
        "total_samples": len(dataset),
        "correct": 0,
        "incorrect": 0,
        "skipped": 0,
        "accuracy": 0.0,
        "details": [],
    }

    logger.info(
        f"Starting evaluation for dataset: {dataset_id} (dimension: {dimension})"
    )
    logger.info(f"Concurrent requests: {concurrent}")

    if concurrent <= 1:
        # Sequential execution
        for sample in tqdm(dataset, desc=f"Evaluating {config['name']}"):
            detail = evaluate_single_sample(sample, config, endpoint, timeout)
            results["details"].append(detail)

            if detail["status"] == "correct":
                results["correct"] += 1
            elif detail["status"] == "incorrect":
                results["incorrect"] += 1
            else:
                results["skipped"] += 1
    else:
        # Concurrent execution
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    evaluate_single_sample, sample, config, endpoint, timeout
                ): idx
                for idx, sample in enumerate(dataset)
            }

            # Process results as they complete
            for future in tqdm(
                as_completed(futures),
                total=len(dataset),
                desc=f"Evaluating {config['name']}",
            ):
                detail = future.result()
                results["details"].append(detail)

                if detail["status"] == "correct":
                    results["correct"] += 1
                elif detail["status"] == "incorrect":
                    results["incorrect"] += 1
                else:
                    results["skipped"] += 1

    # Calculate accuracy
    total_evaluated = results["correct"] + results["incorrect"]
    if total_evaluated > 0:
        results["accuracy"] = results["correct"] / total_evaluated

    logger.info(
        f"Evaluation complete: {results['correct']}/{total_evaluated} correct "
        f"(accuracy: {results['accuracy']:.4f}), {results['skipped']} skipped"
    )

    return results


def main():
    args = parse_args()

    # Get dataset config
    dataset_config = DATASET_REGISTRY[args.dataset]
    dimension = dataset_config["dimension"]

    logger.info(f"Signal Evaluation - Dataset: {args.dataset}")
    logger.info(f"Dimension: {dimension}")
    logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Max samples: {args.max_samples or 'all'}")
    logger.info(f"Concurrent requests: {args.concurrent}")

    # Run evaluation
    start_time = time.time()
    results = evaluate_dataset(
        dataset_id=args.dataset,
        endpoint=args.endpoint,
        max_samples=args.max_samples,
        timeout=args.timeout,
        concurrent=args.concurrent,
    )
    elapsed_time = time.time() - start_time

    # Add metadata
    results["metadata"] = {
        "dataset_id": args.dataset,
        "dataset_name": dataset_config["name"],
        "description": dataset_config["description"],
        "hf_dataset": dataset_config["hf_dataset"],
        "dimension": dimension,
        "endpoint": args.endpoint,
        "max_samples": args.max_samples,
        "concurrent": args.concurrent,
        "elapsed_time_seconds": elapsed_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Signal Evaluation Summary - {dataset_config['name']}")
    print(f"Dimension: {dimension}")
    print("=" * 60)
    print(f"Total samples: {results['total_samples']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
