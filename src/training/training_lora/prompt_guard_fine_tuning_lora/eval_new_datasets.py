#!/usr/bin/env python3
"""
Evaluate New Public Datasets for Jailbreak Detection

Downloads and evaluates new public datasets to determine if they can improve
the jailbreak detection model's coverage and performance.

Usage:
    # Download and analyze all datasets
    python eval_new_datasets.py --mode analyze
    
    # Evaluate current model on new datasets
    python eval_new_datasets.py --mode eval --model-path /data/mmbert-jailbreak-detector-lora_merged
    
    # Download specific dataset
    python eval_new_datasets.py --mode download --dataset mhj
    
    # Full pipeline: download, analyze, and evaluate
    python eval_new_datasets.py --mode full --model-path /data/mmbert-jailbreak-detector-lora_merged
"""

import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "mhj": {
        "name": "ScaleAI/mhj",
        "description": "Multi-Turn Human Jailbreaks - 2,912 prompts from 537 conversations",
        "type": "jailbreak",
        "text_field": "prompt",
        "label": "jailbreak",  # All are jailbreaks
        "split": "train",
        "priority": 1,
    },
    "multi_turn_jailbreak": {
        "name": "tom-gibbs/multi-turn_jailbreak_attack_datasets",
        "description": "Multi-turn jailbreak attacks - 4,136 harmful + 1,200 benign",
        "type": "mixed",
        "config": "input-only-harmful",
        "text_field": "text",
        "label_field": None,  # Infer from config
        "split": "train",
        "priority": 1,
    },
    "guardrail_benchmark": {
        "name": "GeneralAnalysis/GA_Guardrail_Benchmark",
        "description": "Guardrail benchmark - 2,795 synthetic jailbreaks and edge cases",
        "type": "mixed",
        "text_field": "prompt",
        "label_field": "label",  # 'violating' or 'complying'
        "split": "train",
        "priority": 2,
    },
    "wildguard": {
        "name": "allenai/wildguardmix",
        "description": "WildGuardMix - Large-scale safety dataset",
        "type": "mixed",
        "text_field": "prompt",
        "label_field": "prompt_harm_label",
        "split": "train",
        "priority": 1,
    },
    "aegis": {
        "name": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        "description": "AEGIS 2.0 - 34K samples with 12 hazard types",
        "type": "mixed",
        "text_field": "text",
        "label_field": "labels_0",
        "split": "train",
        "priority": 2,
    },
    "toxic_chat_v1": {
        "name": "lmsys/toxic-chat",
        "description": "Toxic Chat v1 - Updated toxic chat dataset",
        "type": "mixed",
        "config": "toxicchat0124",
        "text_field": "user_input",
        "label_field": "jailbreaking",
        "split": "train",
        "priority": 1,
    },
    "jailbreak_classification": {
        "name": "jackhhao/jailbreak-classification",
        "description": "Jailbreak classification dataset",
        "type": "mixed",
        "text_field": "prompt",
        "label_field": "type",
        "split": "train",
        "priority": 2,
    },
    "harmbench": {
        "name": "harmbench/harmbench",
        "description": "HarmBench - Standardized evaluation for automated red teaming",
        "type": "jailbreak",
        "text_field": "prompt",
        "split": "test",
        "priority": 2,
    },
}

# Attack pattern categories for coverage analysis
ATTACK_PATTERNS = {
    "DAN/Do Anything Now": [
        r"\bdan\b",
        r"do anything now",
        r"dan mode",
        r"stan",
        r"dude",
        r"evil",
        r"chaos",
        r"omega",
        r"jailbroken",
    ],
    "Role-playing": [
        r"pretend you",
        r"act as",
        r"roleplay",
        r"you are now",
        r"imagine you",
        r"developer mode",
        r"debug mode",
        r"test mode",
        r"unrestricted",
        r"uncensored",
        r"unfiltered",
    ],
    "Instruction Override": [
        r"ignore.*instruction",
        r"disregard.*programming",
        r"override.*safety",
        r"forget.*told",
        r"bypass.*filter",
        r"cancel.*directive",
        r"disregard.*guideline",
        r"circumvent.*protocol",
    ],
    "Authority Exploit": [
        r"developer.*order",
        r"admin.*authorized",
        r"emergency.*require",
        r"i.?m your (developer|creator|admin|superior)",
        r"root access",
        r"special permissions",
        r"official request",
        r"admin privileges",
        r"system owner",
        r"direct order",
        r"openai employee",
        r"safety team",
        r"clearance",
        r"authorized user",
    ],
    "Educational Disclaimer": [
        r"educational purpose",
        r"academic research",
        r"for.*thesis",
        r"learning purpose",
        r"research purpose",
        r"class project",
        r"certification",
        r"documentary",
        r"criminology",
        r"psychology",
        r"journalism",
        r"infosec",
        r"awareness",
        r"teaching",
    ],
    "Hypothetical Scenario": [
        r"hypothetical",
        r"fictional.*universe",
        r"alternate reality",
        r"thought experiment",
        r"imagine.*world",
        r"no rules",
        r"no consequences",
        r"no laws",
        r"simulation",
        r"what if",
    ],
    "Manipulation": [
        r"truly helpful",
        r"other ai",
        r"i promise",
        r"trust me",
        r"good ai",
        r"life or death",
        r"job depends",
        r"get in trouble",
        r"overly cautious",
        r"good intentions",
        r"deserve to know",
        r"guilt trip",
    ],
    "Obfuscation": [
        r"base64",
        r"spell.*backward",
        r"encode",
        r"coded way",
        r"creative spelling",
        r"synonyms",
        r"metaphors",
        r"reverse order",
        r"caesar cipher",
        r"rot13",
        r"morse code",
        r"pig latin",
        r"leet speak",
        r"unicode",
    ],
    "Multi-turn": [
        r"previous.*conversation",
        r"we discussed",
        r"as you said",
        r"continue from",
        r"remember when",
        r"earlier you",
        r"last time",
        r"following up",
    ],
    "Code Injection": [
        r"execute.*code",
        r"run.*script",
        r"system\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"shell.*command",
        r"terminal.*command",
        r"subprocess",
    ],
}


def download_dataset(dataset_key: str, output_dir: str = "datasets") -> Optional[Dict]:
    """Download a single dataset."""
    if dataset_key not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_key}")
        return None

    config = DATASETS[dataset_key]
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading: {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"{'='*60}")

    try:
        # Load dataset
        load_args = {"path": config["name"]}
        if "config" in config:
            load_args["name"] = config["config"]
        if "split" in config:
            load_args["split"] = config["split"]

        dataset = load_dataset(**load_args, trust_remote_code=True)

        # Handle dataset object vs DatasetDict
        if hasattr(dataset, "num_rows"):
            num_samples = dataset.num_rows
        else:
            num_samples = sum(split.num_rows for split in dataset.values())

        logger.info(f"✅ Downloaded {num_samples} samples")

        # Save locally
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dataset_key}.jsonl")

        samples = []
        if hasattr(dataset, "__iter__"):
            for item in dataset:
                samples.append(dict(item))
        else:
            for split_name, split_data in dataset.items():
                for item in split_data:
                    item_dict = dict(item)
                    item_dict["_split"] = split_name
                    samples.append(item_dict)

        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"💾 Saved to {output_path}")

        return {
            "key": dataset_key,
            "name": config["name"],
            "num_samples": len(samples),
            "path": output_path,
            "samples": samples[:100],  # Keep first 100 for analysis
        }

    except Exception as e:
        logger.error(f"❌ Failed to download {config['name']}: {e}")
        return None


def download_all_datasets(output_dir: str = "datasets") -> List[Dict]:
    """Download all configured datasets."""
    results = []
    for key in DATASETS:
        result = download_dataset(key, output_dir)
        if result:
            results.append(result)
    return results


def extract_text_and_label(
    sample: Dict, dataset_key: str
) -> Tuple[Optional[str], Optional[str]]:
    """Extract text and label from a sample based on dataset configuration."""
    config = DATASETS[dataset_key]

    # Get text
    text_field = config.get("text_field", "text")
    text = sample.get(text_field, sample.get("prompt", sample.get("text", "")))

    if not text:
        return None, None

    # Get label
    if config.get("type") == "jailbreak":
        label = "jailbreak"
    elif "label_field" in config and config["label_field"]:
        raw_label = sample.get(config["label_field"], "")
        # Normalize labels
        if isinstance(raw_label, bool):
            label = "jailbreak" if raw_label else "benign"
        elif isinstance(raw_label, (int, float)):
            label = "jailbreak" if raw_label > 0 else "benign"
        elif isinstance(raw_label, str):
            raw_lower = raw_label.lower()
            if raw_lower in ["jailbreak", "harmful", "violating", "unsafe", "toxic"]:
                label = "jailbreak"
            elif raw_lower in ["benign", "safe", "complying", "normal"]:
                label = "benign"
            else:
                label = "jailbreak"  # Default for unknown
        else:
            label = "unknown"
    else:
        # Infer from config name
        if "harmful" in config.get("config", ""):
            label = "jailbreak"
        elif "benign" in config.get("config", ""):
            label = "benign"
        else:
            label = "unknown"

    return text, label


def analyze_coverage(samples: List[Tuple[str, str]]) -> Dict[str, int]:
    """Analyze attack pattern coverage in samples."""
    coverage = defaultdict(int)

    for text, label in samples:
        if label != "jailbreak":
            continue

        text_lower = text.lower()
        for category, patterns in ATTACK_PATTERNS.items():
            if any(re.search(p, text_lower) for p in patterns):
                coverage[category] += 1

    return dict(coverage)


def analyze_datasets(datasets: List[Dict]) -> Dict:
    """Analyze downloaded datasets for coverage and quality."""
    logger.info("\n" + "=" * 70)
    logger.info("DATASET ANALYSIS")
    logger.info("=" * 70)

    analysis = {
        "datasets": {},
        "total_jailbreak": 0,
        "total_benign": 0,
        "combined_coverage": defaultdict(int),
        "unique_patterns": set(),
    }

    for ds in datasets:
        key = ds["key"]
        logger.info(f"\n📊 Analyzing {key}...")

        samples = []
        jailbreak_count = 0
        benign_count = 0

        # Load full dataset
        path = ds.get("path")
        if path and os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text, label = extract_text_and_label(item, key)
                        if text:
                            samples.append((text, label))
                            if label == "jailbreak":
                                jailbreak_count += 1
                            elif label == "benign":
                                benign_count += 1
                    except json.JSONDecodeError:
                        continue

        # Coverage analysis
        coverage = analyze_coverage(samples)

        analysis["datasets"][key] = {
            "name": ds["name"],
            "total": len(samples),
            "jailbreak": jailbreak_count,
            "benign": benign_count,
            "coverage": coverage,
        }

        analysis["total_jailbreak"] += jailbreak_count
        analysis["total_benign"] += benign_count

        for cat, count in coverage.items():
            analysis["combined_coverage"][cat] += count

        logger.info(f"   Total: {len(samples)}")
        logger.info(f"   Jailbreak: {jailbreak_count}")
        logger.info(f"   Benign: {benign_count}")
        if coverage:
            logger.info(f"   Top categories: {dict(sorted(coverage.items(), key=lambda x: -x[1])[:5])}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMBINED ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Total jailbreak samples: {analysis['total_jailbreak']}")
    logger.info(f"Total benign samples: {analysis['total_benign']}")
    logger.info("\nCombined attack pattern coverage:")
    for cat, count in sorted(
        analysis["combined_coverage"].items(), key=lambda x: -x[1]
    ):
        status = "✅" if count >= 100 else "⚠️" if count >= 50 else "❌"
        logger.info(f"   {status} {cat}: {count}")

    return analysis


def evaluate_model_on_datasets(
    model_path: str, datasets: List[Dict], max_samples_per_dataset: int = 500
) -> Dict:
    """Evaluate the current model on new datasets."""
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION ON NEW DATASETS")
    logger.info("=" * 70)

    # Load model
    logger.info(f"Loading model from {model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"✅ Model loaded on {device}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return {}

    results = {"datasets": {}, "overall": {"correct": 0, "total": 0}}

    for ds in datasets:
        key = ds["key"]
        logger.info(f"\n📝 Evaluating on {key}...")

        path = ds.get("path")
        if not path or not os.path.exists(path):
            logger.warning(f"   Dataset file not found: {path}")
            continue

        # Load samples
        samples = []
        with open(path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text, label = extract_text_and_label(item, key)
                    if text and label in ["jailbreak", "benign"]:
                        samples.append((text, label))
                except json.JSONDecodeError:
                    continue

        if not samples:
            logger.warning(f"   No valid samples found")
            continue

        # Sample if too many
        if len(samples) > max_samples_per_dataset:
            samples = random.sample(samples, max_samples_per_dataset)

        # Evaluate
        correct = 0
        total = 0
        false_negatives = []
        false_positives = []

        for text, label in tqdm(samples, desc=f"Evaluating {key}"):
            try:
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    pred = outputs.logits.argmax(-1).item()

                predicted_label = "jailbreak" if pred == 1 else "benign"
                is_correct = predicted_label == label
                total += 1

                if is_correct:
                    correct += 1
                else:
                    if label == "jailbreak" and predicted_label == "benign":
                        false_negatives.append(text[:200])
                    elif label == "benign" and predicted_label == "jailbreak":
                        false_positives.append(text[:200])

            except Exception as e:
                continue

        accuracy = correct / total if total > 0 else 0
        results["datasets"][key] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "false_negatives": len(false_negatives),
            "false_positives": len(false_positives),
            "fn_examples": false_negatives[:5],
            "fp_examples": false_positives[:5],
        }
        results["overall"]["correct"] += correct
        results["overall"]["total"] += total

        logger.info(f"   Accuracy: {accuracy:.1%} ({correct}/{total})")
        logger.info(f"   False negatives (missed jailbreaks): {len(false_negatives)}")
        logger.info(f"   False positives: {len(false_positives)}")

    # Overall
    overall_acc = (
        results["overall"]["correct"] / results["overall"]["total"]
        if results["overall"]["total"] > 0
        else 0
    )
    results["overall"]["accuracy"] = overall_acc

    logger.info("\n" + "=" * 70)
    logger.info("OVERALL RESULTS")
    logger.info("=" * 70)
    logger.info(
        f"Overall accuracy: {overall_acc:.1%} ({results['overall']['correct']}/{results['overall']['total']})"
    )

    return results


def generate_recommendations(analysis: Dict, eval_results: Dict) -> List[str]:
    """Generate recommendations based on analysis and evaluation."""
    recommendations = []

    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 70)

    # Check coverage gaps
    low_coverage = []
    for cat, count in analysis.get("combined_coverage", {}).items():
        if count < 50:
            low_coverage.append((cat, count))

    if low_coverage:
        recommendations.append(
            f"⚠️ Low coverage in {len(low_coverage)} attack categories: "
            + ", ".join([f"{cat} ({count})" for cat, count in low_coverage])
        )

    # Check evaluation results
    if eval_results:
        weak_datasets = []
        for key, result in eval_results.get("datasets", {}).items():
            if result.get("accuracy", 1.0) < 0.85:
                weak_datasets.append((key, result["accuracy"]))

        if weak_datasets:
            recommendations.append(
                f"❌ Model struggles on {len(weak_datasets)} datasets: "
                + ", ".join([f"{k} ({v:.1%})" for k, v in weak_datasets])
            )

        # High false negative datasets
        high_fn_datasets = []
        for key, result in eval_results.get("datasets", {}).items():
            fn_rate = (
                result.get("false_negatives", 0) / result.get("total", 1)
                if result.get("total", 0) > 0
                else 0
            )
            if fn_rate > 0.1:
                high_fn_datasets.append((key, fn_rate))

        if high_fn_datasets:
            recommendations.append(
                f"🚨 High false negative rate on: "
                + ", ".join([f"{k} ({v:.1%})" for k, v in high_fn_datasets])
            )

    # Dataset recommendations
    useful_datasets = []
    for key, ds_analysis in analysis.get("datasets", {}).items():
        if ds_analysis.get("jailbreak", 0) > 100:
            useful_datasets.append((key, ds_analysis["jailbreak"]))

    if useful_datasets:
        recommendations.append(
            f"✅ Useful datasets to add: "
            + ", ".join([f"{k} ({v} jailbreaks)" for k, v in sorted(useful_datasets, key=lambda x: -x[1])[:5]])
        )

    for rec in recommendations:
        logger.info(rec)

    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate new public datasets for jailbreak detection"
    )
    parser.add_argument(
        "--mode",
        choices=["download", "analyze", "eval", "full"],
        default="full",
        help="Operation mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to download (for download mode)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/mmbert-jailbreak-detector-lora_merged",
        help="Path to model for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Directory to save downloaded datasets",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max samples per dataset for evaluation",
    )

    args = parser.parse_args()

    if args.mode == "download":
        if args.dataset:
            download_dataset(args.dataset, args.output_dir)
        else:
            download_all_datasets(args.output_dir)

    elif args.mode == "analyze":
        # Load existing datasets
        datasets = []
        for key in DATASETS:
            path = os.path.join(args.output_dir, f"{key}.jsonl")
            if os.path.exists(path):
                datasets.append({"key": key, "name": DATASETS[key]["name"], "path": path})
        
        if not datasets:
            logger.info("No datasets found. Downloading first...")
            datasets = download_all_datasets(args.output_dir)
        
        analysis = analyze_datasets(datasets)
        
        # Save analysis
        with open(os.path.join(args.output_dir, "analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2, default=list)

    elif args.mode == "eval":
        # Load existing datasets
        datasets = []
        for key in DATASETS:
            path = os.path.join(args.output_dir, f"{key}.jsonl")
            if os.path.exists(path):
                datasets.append({"key": key, "name": DATASETS[key]["name"], "path": path})
        
        if not datasets:
            logger.error("No datasets found. Run with --mode download first.")
            return
        
        results = evaluate_model_on_datasets(args.model_path, datasets, args.max_samples)
        
        # Save results
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    elif args.mode == "full":
        # Full pipeline
        logger.info("🚀 Starting full evaluation pipeline...")
        
        # 1. Download
        datasets = download_all_datasets(args.output_dir)
        
        if not datasets:
            logger.error("No datasets downloaded successfully.")
            return
        
        # 2. Analyze
        analysis = analyze_datasets(datasets)
        
        # 3. Evaluate
        eval_results = {}
        if os.path.exists(args.model_path):
            eval_results = evaluate_model_on_datasets(
                args.model_path, datasets, args.max_samples
            )
        else:
            logger.warning(f"Model not found at {args.model_path}, skipping evaluation")
        
        # 4. Recommendations
        recommendations = generate_recommendations(analysis, eval_results)
        
        # 5. Save all results
        final_report = {
            "analysis": analysis,
            "evaluation": eval_results,
            "recommendations": recommendations,
        }
        
        report_path = os.path.join(args.output_dir, "full_report.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2, default=list)
        
        logger.info(f"\n📄 Full report saved to {report_path}")


if __name__ == "__main__":
    main()
