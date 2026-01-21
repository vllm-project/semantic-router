#!/usr/bin/env python3
"""
Dataset Label Verification and Correction Script

Uses GPT-OSS-120B (via vLLM) to verify and correct labels in the
feedback-detector-dataset. The original dataset was labeled by a weaker
model, so this script uses a stronger model to audit and fix labels.

Usage:
    python verify_dataset_labels.py --sample 100  # Quick test with 100 samples
    python verify_dataset_labels.py --all         # Process entire dataset
    python verify_dataset_labels.py --correct     # Auto-correct labels
"""

import os
import json
import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Label definitions
LABELS = ["SAT", "NEED_CLARIFICATION", "WRONG_ANSWER", "WANT_DIFFERENT"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# vLLM API configuration
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "openai/gpt-oss-120b")


@dataclass
class VerificationResult:
    """Result of label verification for a single example."""

    index: int
    text: str
    original_label: str
    predicted_label: str
    confidence: str  # "high", "medium", "low"
    reasoning: str
    is_correct: bool
    suggested_correction: Optional[str] = None


@dataclass
class VerificationStats:
    """Statistics for the verification process."""

    total: int = 0
    correct: int = 0
    incorrect: int = 0
    uncertain: int = 0
    errors: int = 0

    # Per-label stats
    label_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "uncertain": self.uncertain,
            "errors": self.errors,
            "accuracy": self.accuracy(),
            "label_stats": self.label_stats,
        }


def create_verification_prompt(text: str, original_label: str) -> str:
    """Create a prompt for GPT-OSS-120B to verify the label."""
    # Keep prompt very concise for Responses API
    text_snippet = text[:150].replace('"', "'")
    return f"""Classify "{text_snippet}" into SAT, NEED_CLARIFICATION, WRONG_ANSWER, or WANT_DIFFERENT.
Current label: {original_label}
Output JSON: {{"label":"...","correct":true/false}}"""


def extract_json_from_response(content: str) -> Optional[Dict]:
    """
    Extract JSON from model response, handling various formats:
    - Pure JSON
    - Markdown code blocks
    - GPT-OSS format with analysis...assistantfinal{JSON}
    """
    content = content.strip()

    # Try 1: Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown code blocks
    if "```" in content:
        try:
            # Find content between ``` markers
            parts = content.split("```")
            for part in parts[1::2]:  # Get odd indices (inside code blocks)
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                return json.loads(part)
        except (json.JSONDecodeError, IndexError):
            pass

    # Try 3: GPT-OSS format - extract JSON after "assistantfinal" or similar markers
    markers = ["assistantfinal", "assistant_final", "final:", "output:"]
    for marker in markers:
        if marker in content.lower():
            idx = content.lower().find(marker) + len(marker)
            remaining = content[idx:].strip()
            # Find the JSON object
            if "{" in remaining:
                json_start = remaining.find("{")
                # Find matching closing brace
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(remaining[json_start:], json_start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                try:
                    return json.loads(remaining[json_start:json_end])
                except json.JSONDecodeError:
                    pass

    # Try 4: Find any JSON object in the content
    if "{" in content and "}" in content:
        json_start = content.find("{")
        # Find the last matching closing brace
        brace_count = 0
        json_end = json_start
        for i, char in enumerate(content[json_start:], json_start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        try:
            return json.loads(content[json_start:json_end])
        except json.JSONDecodeError:
            pass

    return None


def call_vllm_api(prompt: str, max_retries: int = 3) -> Optional[Dict]:
    """Call the vLLM Chat Completions API with include_reasoning=false to skip CoT."""
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,  # Need enough tokens for reasoning + JSON output
        "temperature": 0.0,
        # Skip CoT reasoning output for GPT-OSS
        "include_reasoning": False,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                VLLM_API_URL, json=payload, headers=headers, timeout=60
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # With include_reasoning=false, content should be clean JSON
            parsed = extract_json_from_response(content)
            if parsed:
                return parsed

            # Only log on last attempt
            if attempt == max_retries - 1:
                logger.debug(
                    f"Could not extract JSON after {max_retries} attempts. Content: {content[:200]}"
                )

        except requests.exceptions.RequestException as e:
            logger.warning(f"API error on attempt {attempt + 1}: {e}")
            time.sleep(2**attempt)  # Exponential backoff
        except Exception as e:
            logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")

    return None


def verify_single_example(
    index: int, text: str, original_label: str
) -> VerificationResult:
    """Verify a single example using GPT-OSS-120B."""
    prompt = create_verification_prompt(text, original_label)

    result = call_vllm_api(prompt)

    if result is None:
        # API call failed
        return VerificationResult(
            index=index,
            text=text[:200],
            original_label=original_label,
            predicted_label="ERROR",
            confidence="low",
            reasoning="API call failed",
            is_correct=True,  # Assume correct if we can't verify
            suggested_correction=None,
        )

    # Handle simplified JSON response format
    predicted = result.get("label") or result.get("predicted_label", "UNKNOWN")
    is_correct = result.get("correct", result.get("is_correct", True))

    # Determine suggested correction
    suggested = None
    if not is_correct and predicted != original_label:
        suggested = predicted

    return VerificationResult(
        index=index,
        text=text[:200],
        original_label=original_label,
        predicted_label=predicted,
        confidence=result.get(
            "confidence", "high"
        ),  # Default to high since model is confident
        reasoning=result.get("reasoning", ""),
        is_correct=is_correct,
        suggested_correction=suggested,
    )


def verify_dataset(
    dataset: Dataset,
    num_samples: Optional[int] = None,
    num_workers: int = 4,
    progress: bool = True,
) -> Tuple[List[VerificationResult], VerificationStats]:
    """
    Verify labels in the dataset using GPT-OSS-120B.

    Args:
        dataset: HuggingFace Dataset to verify
        num_samples: Number of samples to verify (None = all)
        num_workers: Number of parallel workers
        progress: Show progress bar

    Returns:
        Tuple of (results list, statistics)
    """
    results = []
    stats = VerificationStats()

    # Initialize label stats
    for label in LABELS:
        stats.label_stats[label] = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "confused_with": {},
        }

    # Prepare examples
    if num_samples is not None:
        indices = list(range(min(num_samples, len(dataset))))
    else:
        indices = list(range(len(dataset)))

    logger.info(f"Verifying {len(indices)} examples...")

    # Process examples
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for idx in indices:
            example = dataset[idx]
            text = example.get("text", "")
            label = example.get(
                "label_name", ID2LABEL.get(example.get("label", 0), "SAT")
            )

            future = executor.submit(verify_single_example, idx, text, label)
            futures[future] = idx

        # Collect results with progress bar
        iterator = as_completed(futures)
        if progress:
            iterator = tqdm(iterator, total=len(futures), desc="Verifying")

        for future in iterator:
            try:
                result = future.result()
                results.append(result)

                # Update stats
                stats.total += 1

                if result.predicted_label == "ERROR":
                    stats.errors += 1
                    continue

                original = result.original_label
                predicted = result.predicted_label

                # Update label stats
                if original in stats.label_stats:
                    stats.label_stats[original]["total"] += 1

                if result.is_correct:
                    stats.correct += 1
                    if original in stats.label_stats:
                        stats.label_stats[original]["correct"] += 1
                elif result.confidence == "low":
                    stats.uncertain += 1
                else:
                    stats.incorrect += 1
                    if original in stats.label_stats:
                        stats.label_stats[original]["incorrect"] += 1
                        # Track confusion
                        if (
                            predicted
                            not in stats.label_stats[original]["confused_with"]
                        ):
                            stats.label_stats[original]["confused_with"][predicted] = 0
                        stats.label_stats[original]["confused_with"][predicted] += 1

            except Exception as e:
                logger.error(f"Error processing result: {e}")
                stats.errors += 1

    return results, stats


def correct_dataset(
    dataset: Dataset,
    results: List[VerificationResult],
    confidence_threshold: str = "medium",
) -> Dataset:
    """
    Apply corrections to the dataset based on verification results.

    Args:
        dataset: Original dataset
        results: Verification results
        confidence_threshold: Minimum confidence to apply correction ("high" or "medium")

    Returns:
        Corrected dataset
    """
    confidence_levels = {"high": 2, "medium": 1, "low": 0}
    threshold = confidence_levels.get(confidence_threshold, 1)

    # Create correction map
    corrections = {}
    for result in results:
        if not result.is_correct and result.suggested_correction:
            conf_level = confidence_levels.get(result.confidence, 0)
            if conf_level >= threshold:
                corrections[result.index] = result.suggested_correction

    logger.info(f"Applying {len(corrections)} corrections to dataset...")

    # Apply corrections
    def apply_correction(example, idx):
        if idx in corrections:
            new_label = corrections[idx]
            example["label_name"] = new_label
            example["label"] = LABEL2ID.get(new_label, example["label"])
            example["corrected"] = True
            example["original_label"] = example.get("label_name", "")
        return example

    corrected = dataset.map(
        apply_correction, with_indices=True, desc="Applying corrections"
    )

    return corrected


def print_report(stats: VerificationStats, results: List[VerificationResult]):
    """Print a detailed verification report."""
    print("\n" + "=" * 70)
    print("DATASET LABEL VERIFICATION REPORT")
    print("=" * 70)

    print(f"\n📊 Overall Statistics:")
    print(f"   Total examples verified: {stats.total}")
    print(
        f"   Correct labels: {stats.correct} ({stats.correct/max(1,stats.total)*100:.1f}%)"
    )
    print(
        f"   Incorrect labels: {stats.incorrect} ({stats.incorrect/max(1,stats.total)*100:.1f}%)"
    )
    print(
        f"   Uncertain: {stats.uncertain} ({stats.uncertain/max(1,stats.total)*100:.1f}%)"
    )
    print(f"   Errors: {stats.errors}")

    print(f"\n📈 Per-Label Statistics:")
    for label, label_stats in stats.label_stats.items():
        total = label_stats["total"]
        if total == 0:
            continue
        correct = label_stats["correct"]
        incorrect = label_stats["incorrect"]

        print(f"\n   {label}:")
        print(f"      Total: {total}")
        print(f"      Correct: {correct} ({correct/total*100:.1f}%)")
        print(f"      Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")

        if label_stats["confused_with"]:
            print(f"      Confused with:")
            for confused_label, count in sorted(
                label_stats["confused_with"].items(), key=lambda x: -x[1]
            ):
                print(f"         → {confused_label}: {count}")

    # Show some examples of incorrect labels
    incorrect_examples = [
        r for r in results if not r.is_correct and r.confidence != "low"
    ]
    if incorrect_examples:
        print(f"\n❌ Sample Incorrect Labels (showing up to 10):")
        for i, result in enumerate(incorrect_examples[:10], 1):
            print(f"\n   Example {i}:")
            print(f"      Text: {result.text[:100]}...")
            print(f"      Original: {result.original_label}")
            print(f"      Should be: {result.suggested_correction}")
            print(f"      Confidence: {result.confidence}")
            print(f"      Reasoning: {result.reasoning[:100]}...")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Verify and correct dataset labels using GPT-OSS-120B"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="llm-semantic-router/feedback-detector-dataset",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to verify (train, validation, test)",
    )

    # Sampling arguments
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples to verify (default: 100)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Verify entire dataset (overrides --sample)"
    )

    # Processing arguments
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--correct", action="store_true", help="Apply corrections to the dataset"
    )
    parser.add_argument(
        "--confidence",
        type=str,
        choices=["high", "medium", "low"],
        default="medium",
        help="Minimum confidence for corrections",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/verified_dataset",
        help="Output directory for corrected dataset",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="verification_report.json",
        help="Output file for verification report",
    )

    # API arguments
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="vLLM API URL (default: http://localhost:8000/v1/chat/completions)",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name for vLLM API"
    )

    args = parser.parse_args()

    # Set API configuration
    global VLLM_API_URL, VLLM_MODEL
    if args.api_url:
        VLLM_API_URL = args.api_url
    if args.model:
        VLLM_MODEL = args.model

    # Determine number of samples
    num_samples = None if args.all else (args.sample or 100)

    logger.info("=" * 60)
    logger.info("DATASET LABEL VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Samples: {'all' if num_samples is None else num_samples}")
    logger.info(f"Model: {VLLM_MODEL}")
    logger.info(f"API URL: {VLLM_API_URL}")

    # Check if vLLM is running
    try:
        health_url = VLLM_API_URL.replace("/v1/chat/completions", "/health")
        response = requests.get(health_url, timeout=5)
        if response.status_code != 200:
            logger.warning("vLLM server may not be healthy")
    except Exception as e:
        logger.error(f"Cannot connect to vLLM server: {e}")
        logger.error("Make sure the vLLM server is running!")
        return

    # Load dataset
    logger.info(f"\nLoading dataset from HuggingFace...")
    try:
        dataset = load_dataset(args.dataset)
        if args.split not in dataset:
            logger.error(
                f"Split '{args.split}' not found. Available: {list(dataset.keys())}"
            )
            return
        dataset_split = dataset[args.split]
        logger.info(f"Loaded {len(dataset_split)} examples from '{args.split}' split")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Verify dataset
    results, stats = verify_dataset(
        dataset_split, num_samples=num_samples, num_workers=args.workers, progress=True
    )

    # Print report
    print_report(stats, results)

    # Save report
    report_path = Path(args.output_dir) / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = {
        "stats": stats.to_dict(),
        "results": [
            {
                "index": r.index,
                "text": r.text,
                "original_label": r.original_label,
                "predicted_label": r.predicted_label,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "is_correct": r.is_correct,
                "suggested_correction": r.suggested_correction,
            }
            for r in results
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"\n📄 Report saved to: {report_path}")

    # Apply corrections if requested
    if args.correct and stats.incorrect > 0:
        logger.info(
            f"\nApplying corrections with {args.confidence} confidence threshold..."
        )

        corrected_dataset = correct_dataset(
            dataset_split, results, confidence_threshold=args.confidence
        )

        # Save corrected dataset
        output_path = Path(args.output_dir) / f"corrected_{args.split}"
        corrected_dataset.save_to_disk(str(output_path))
        logger.info(f"✅ Corrected dataset saved to: {output_path}")

        # Also save as JSONL for easy viewing
        jsonl_path = Path(args.output_dir) / f"corrected_{args.split}.jsonl"
        with open(jsonl_path, "w") as f:
            for example in corrected_dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"✅ JSONL saved to: {jsonl_path}")

    logger.info("\n✅ Verification complete!")


if __name__ == "__main__":
    main()
