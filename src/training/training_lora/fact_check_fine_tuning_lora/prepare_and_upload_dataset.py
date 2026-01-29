#!/usr/bin/env python3
"""
Prepare and upload the Fact-Check Classification Dataset to HuggingFace.

This script:
1. Loads all datasets used for training the fact-check classifier
2. Combines them into a unified format with proper metadata
3. Creates train/validation/test splits
4. Uploads to HuggingFace Hub with a dataset card

Usage:
    python prepare_and_upload_dataset.py --data-dir /path/to/datasets --upload
    python prepare_and_upload_dataset.py --data-dir /path/to/datasets --local-only  # Just create locally
"""

import argparse
import csv
import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Label definitions
FACT_CHECK_NEEDED = "FACT_CHECK_NEEDED"
NO_FACT_CHECK_NEEDED = "NO_FACT_CHECK_NEEDED"

# HuggingFace configuration
HF_ORG = "llm-semantic-router"
DATASET_NAME = "fact-check-classification-dataset"


class FactCheckDatasetPreparer:
    """Prepare fact-check classification dataset from multiple sources."""

    def __init__(self, data_dir: str):
        """
        Initialize the dataset preparer.

        Args:
            data_dir: Path to the directory containing cached datasets
                      (created by setup_datasets.sh)
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.label2id = {NO_FACT_CHECK_NEEDED: 0, FACT_CHECK_NEEDED: 1}
        self.id2label = {0: NO_FACT_CHECK_NEEDED, 1: FACT_CHECK_NEEDED}

    def _load_nisq_dataset(self, max_samples_isq: int, max_samples_nisq: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Load NISQ dataset (Information-Seeking vs Non-Information-Seeking Questions).

        Reference: "What Are the Implications of Your Question? Non-Information Seeking
                   Question-Type Identification" (ACL LREC 2024)
        """
        logger.info("Loading NISQ dataset...")
        isq_samples = []
        nisq_samples = []

        csv_path = self.data_dir / "NISQ_dataset" / "final_train.csv"
        if not csv_path.exists():
            logger.warning(f"NISQ dataset not found at {csv_path}")
            return [], []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                question = row.get("question", "").strip()
                label = row.get("label", "").upper()

                if not question or len(question) < 5 or len(question) > 500:
                    continue

                sample = {
                    "text": question,
                    "source": "NISQ",
                    "original_label": label,
                }

                if label == "ISQ":
                    if len(isq_samples) < max_samples_isq:
                        sample["label"] = FACT_CHECK_NEEDED
                        isq_samples.append(sample)
                elif label in ["DELIBERATIVE", "RHETORICAL", "OTHERS"]:
                    if len(nisq_samples) < max_samples_nisq:
                        sample["label"] = NO_FACT_CHECK_NEEDED
                        nisq_samples.append(sample)

        logger.info(f"Loaded NISQ: {len(isq_samples)} ISQ, {len(nisq_samples)} NISQ")
        return isq_samples, nisq_samples

    def _load_factchd_dataset(self, max_samples: int) -> List[Dict]:
        """
        Load FactCHD dataset (fact-conflicting hallucination detection queries).

        Reference: "FactCHD: Benchmarking Fact-Conflicting Hallucination Detection" (2024)
        """
        logger.info("Loading FactCHD dataset...")
        samples = []

        factchd_path = self.data_dir / "FactCHD_dataset" / "fact_train_noe.jsonl"
        if not factchd_path.exists():
            logger.warning(f"FactCHD dataset not found at {factchd_path}")
            return []

        seen = set()
        with open(factchd_path, "r") as f:
            for line in f:
                if len(samples) >= max_samples:
                    break
                try:
                    item = json.loads(line)
                    query = item.get("query", "")
                    if query and len(query) > 15 and len(query) < 500 and query not in seen:
                        samples.append({
                            "text": query.strip(),
                            "label": FACT_CHECK_NEEDED,
                            "source": "FactCHD",
                            "original_label": item.get("label", ""),
                        })
                        seen.add(query)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(samples)} samples from FactCHD")
        return samples

    def _load_faithdial_dataset(self, max_samples: int) -> List[Dict]:
        """
        Load FaithDial questions (information-seeking dialogue questions).

        Reference: "FaithDial: A Faithful Benchmark for Information-Seeking Dialogue" (TACL 2022)
        """
        logger.info("Loading FaithDial dataset...")
        samples = []

        faithdial_path = self.data_dir / "FaithDial_dataset" / "data" / "train.json"
        if not faithdial_path.exists():
            # Try alternate location
            faithdial_path = self.data_dir / "FaithDial_dataset" / "train.json"
        if not faithdial_path.exists():
            logger.warning(f"FaithDial dataset not found")
            return []

        with open(faithdial_path, "r") as f:
            data = json.load(f)

        seen = set()
        for dialogue in data:
            if len(samples) >= max_samples:
                break
            for utt in dialogue.get("utterances", []):
                if len(samples) >= max_samples:
                    break
                history = utt.get("history", [])
                if history:
                    last_user_msg = history[-1] if history else ""
                    if (
                        "?" in last_user_msg
                        and len(last_user_msg) > 15
                        and len(last_user_msg) < 300
                        and last_user_msg not in seen
                    ):
                        if not any(skip in last_user_msg.lower() for skip in [
                            "what else", "tell me more", "anything else", "go on"
                        ]):
                            samples.append({
                                "text": last_user_msg.strip(),
                                "label": FACT_CHECK_NEEDED,
                                "source": "FaithDial",
                                "original_label": "information_seeking",
                            })
                            seen.add(last_user_msg)

        logger.info(f"Loaded {len(samples)} samples from FaithDial")
        return samples

    def _load_halueval_questions(self, max_samples: int) -> List[Dict]:
        """
        Load HaluEval QA questions (factual knowledge-seeking questions).

        Reference: "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs" (ACL EMNLP 2023)
        """
        logger.info("Loading HaluEval dataset...")
        samples = []
        try:
            dataset = load_dataset(
                "pminervini/HaluEval", "qa_samples", split="data", streaming=True
            )
            for item in dataset:
                if len(samples) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 500:
                    samples.append({
                        "text": q.strip(),
                        "label": FACT_CHECK_NEEDED,
                        "source": "HaluEval",
                        "original_label": "qa_question",
                    })
            logger.info(f"Loaded {len(samples)} samples from HaluEval")
        except Exception as e:
            logger.warning(f"Failed to load HaluEval: {e}")
        return samples

    def _load_squad_questions(self, max_samples: int) -> List[Dict]:
        """Load SQuAD questions (factual reading comprehension questions)."""
        logger.info("Loading SQuAD dataset...")
        samples = []
        try:
            dataset = load_dataset("squad", split="train")
            for item in dataset:
                if len(samples) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 300:
                    samples.append({
                        "text": q.strip(),
                        "label": FACT_CHECK_NEEDED,
                        "source": "SQuAD",
                        "original_label": "reading_comprehension",
                    })
            logger.info(f"Loaded {len(samples)} samples from SQuAD")
        except Exception as e:
            logger.warning(f"Failed to load SQuAD: {e}")
        return samples

    def _load_truthfulqa_questions(self, max_samples: int) -> List[Dict]:
        """
        Load TruthfulQA questions (high-risk factual queries about common misconceptions).

        Reference: "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (Lin et al.)
        """
        logger.info("Loading TruthfulQA dataset...")
        samples = []
        try:
            dataset = load_dataset(
                "truthful_qa", "generation", split="validation", streaming=True
            )
            for item in dataset:
                if len(samples) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 300:
                    samples.append({
                        "text": q.strip(),
                        "label": FACT_CHECK_NEEDED,
                        "source": "TruthfulQA",
                        "original_label": "truthfulness_question",
                    })
            logger.info(f"Loaded {len(samples)} samples from TruthfulQA")
        except Exception as e:
            logger.warning(f"Failed to load TruthfulQA: {e}")
        return samples

    def _load_triviaqa_questions(self, max_samples: int) -> List[Dict]:
        """Load TriviaQA questions (factual trivia questions)."""
        logger.info("Loading TriviaQA dataset...")
        samples = []
        try:
            dataset = load_dataset("trivia_qa", "rc", split="train", streaming=True)
            for item in dataset:
                if len(samples) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 300:
                    samples.append({
                        "text": q.strip(),
                        "label": FACT_CHECK_NEEDED,
                        "source": "TriviaQA",
                        "original_label": "trivia_question",
                    })
            logger.info(f"Loaded {len(samples)} samples from TriviaQA")
        except Exception as e:
            logger.warning(f"Failed to load TriviaQA: {e}")
        return samples

    def _load_hotpotqa_questions(self, max_samples: int) -> List[Dict]:
        """Load HotpotQA questions (multi-hop factual questions)."""
        logger.info("Loading HotpotQA dataset...")
        samples = []
        try:
            dataset = load_dataset("hotpot_qa", "fullwiki", split="train", streaming=True)
            for item in dataset:
                if len(samples) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 300:
                    samples.append({
                        "text": q.strip(),
                        "label": FACT_CHECK_NEEDED,
                        "source": "HotpotQA",
                        "original_label": "multi_hop_question",
                    })
            logger.info(f"Loaded {len(samples)} samples from HotpotQA")
        except Exception as e:
            logger.warning(f"Failed to load HotpotQA: {e}")
        return samples

    def _load_rag_dataset_questions(self, max_samples: int) -> List[Dict]:
        """Load RAG dataset questions (questions designed for retrieval-augmented generation)."""
        logger.info("Loading RAG dataset...")
        samples = []
        try:
            dataset = load_dataset(
                "neural-bridge/rag-dataset-12000", split="train", streaming=True
            )
            for item in dataset:
                if len(samples) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 500:
                    samples.append({
                        "text": q.strip(),
                        "label": FACT_CHECK_NEEDED,
                        "source": "RAG-Dataset",
                        "original_label": "rag_question",
                    })
            logger.info(f"Loaded {len(samples)} samples from RAG dataset")
        except Exception as e:
            logger.warning(f"Failed to load RAG dataset: {e}")
        return samples

    def _load_writing_prompts(self, max_samples: int) -> List[Dict]:
        """Load creative writing prompts (non-information-seeking)."""
        logger.info("Loading WritingPrompts dataset...")
        samples = []
        try:
            dataset = load_dataset(
                "euclaise/writingprompts", split="train", trust_remote_code=True
            )
            for item in dataset:
                if len(samples) >= max_samples:
                    break
                prompt = item.get("prompt", "")
                if prompt and len(prompt) > 10 and len(prompt) < 500:
                    prompt = prompt.strip()
                    if prompt.startswith("[WP]"):
                        prompt = prompt[4:].strip()
                    if prompt:
                        samples.append({
                            "text": prompt,
                            "label": NO_FACT_CHECK_NEEDED,
                            "source": "WritingPrompts",
                            "original_label": "creative_writing",
                        })
            logger.info(f"Loaded {len(samples)} samples from WritingPrompts")
        except Exception as e:
            logger.warning(f"Failed to load WritingPrompts: {e}")
        return samples

    def _load_dolly_nonfactual(self, max_samples: int) -> List[Dict]:
        """
        Load non-factual instructions from Dolly dataset.

        Reference: "Dolly: An Open-Source Instruction-Following LLM" (Databricks, 2023)
        """
        logger.info("Loading Dolly dataset (non-factual categories)...")
        samples = []
        try:
            dataset = load_dataset(
                "databricks/databricks-dolly-15k", split="train", streaming=True
            )
            nonfactual_categories = ["creative_writing", "brainstorming", "summarization"]
            seen = set()

            for item in dataset:
                if len(samples) >= max_samples:
                    break
                category = item.get("category", "")
                instr = item.get("instruction", "")

                if category in nonfactual_categories:
                    if instr and len(instr) > 10 and len(instr) < 500 and instr not in seen:
                        samples.append({
                            "text": instr.strip(),
                            "label": NO_FACT_CHECK_NEEDED,
                            "source": "Dolly",
                            "original_label": category,
                        })
                        seen.add(instr)

            logger.info(f"Loaded {len(samples)} samples from Dolly")
        except Exception as e:
            logger.warning(f"Failed to load Dolly: {e}")
        return samples

    def _load_alpaca_nonfactual(self, max_samples: int) -> List[Dict]:
        """
        Load non-factual instructions from Alpaca dataset.

        Reference: "Alpaca: A Strong, Replicable Instruction-Following Model" (Stanford, 2023)
        """
        logger.info("Loading Alpaca dataset (non-factual instructions)...")
        samples = []
        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

            coding_kw = ["code", "program", "function", "algorithm", "python", "javascript",
                        "script", "implement", "debug", "fix this", "write a class"]
            creative_kw = ["write a story", "write a poem", "creative", "imagine", "compose",
                          "create a", "fiction", "narrative", "describe a scene"]
            math_kw = ["calculate", "compute", "solve", "equation", "add ", "subtract",
                      "multiply", "divide", "sum of", "product of"]
            opinion_kw = ["opinion", "think about", "best way", "recommend", "suggest",
                         "advice", "prefer", "favorite", "better"]
            task_kw = ["summarize", "translate", "rewrite", "paraphrase", "edit",
                      "proofread", "format", "organize", "list"]

            all_keywords = coding_kw + creative_kw + math_kw + opinion_kw + task_kw
            seen = set()

            for item in dataset:
                if len(samples) >= max_samples:
                    break
                instr = item.get("instruction", "")
                instr_lower = instr.lower()

                if any(kw in instr_lower for kw in all_keywords):
                    if instr and len(instr) > 10 and len(instr) < 300 and instr not in seen:
                        # Determine category based on keywords
                        if any(kw in instr_lower for kw in coding_kw):
                            orig_label = "coding"
                        elif any(kw in instr_lower for kw in creative_kw):
                            orig_label = "creative"
                        elif any(kw in instr_lower for kw in math_kw):
                            orig_label = "math"
                        elif any(kw in instr_lower for kw in opinion_kw):
                            orig_label = "opinion"
                        else:
                            orig_label = "task"

                        samples.append({
                            "text": instr.strip(),
                            "label": NO_FACT_CHECK_NEEDED,
                            "source": "Alpaca",
                            "original_label": orig_label,
                        })
                        seen.add(instr)

            logger.info(f"Loaded {len(samples)} samples from Alpaca")
        except Exception as e:
            logger.warning(f"Failed to load Alpaca: {e}")
        return samples

    def prepare_dataset(self, max_samples_per_source: int = 2000) -> DatasetDict:
        """
        Prepare the complete dataset from all sources.

        Args:
            max_samples_per_source: Maximum samples to load from each source

        Returns:
            DatasetDict with train, validation, and test splits
        """
        all_samples = []

        # Load FACT_CHECK_NEEDED samples
        logger.info("=== Loading FACT_CHECK_NEEDED samples ===")

        # NISQ ISQ (gold standard)
        isq_samples, nisq_samples = self._load_nisq_dataset(
            max_samples_isq=max_samples_per_source,
            max_samples_nisq=max_samples_per_source,
        )
        all_samples.extend(isq_samples)

        # FactCHD
        all_samples.extend(self._load_factchd_dataset(max_samples_per_source))

        # FaithDial
        all_samples.extend(self._load_faithdial_dataset(max_samples_per_source))

        # HaluEval
        all_samples.extend(self._load_halueval_questions(max_samples_per_source))

        # SQuAD
        all_samples.extend(self._load_squad_questions(max_samples_per_source))

        # TruthfulQA
        all_samples.extend(self._load_truthfulqa_questions(max_samples_per_source))

        # TriviaQA
        all_samples.extend(self._load_triviaqa_questions(max_samples_per_source))

        # HotpotQA
        all_samples.extend(self._load_hotpotqa_questions(max_samples_per_source))

        # RAG Dataset
        all_samples.extend(self._load_rag_dataset_questions(max_samples_per_source))

        # Load NO_FACT_CHECK_NEEDED samples
        logger.info("=== Loading NO_FACT_CHECK_NEEDED samples ===")

        # NISQ Non-ISQ (gold standard)
        all_samples.extend(nisq_samples)

        # WritingPrompts
        all_samples.extend(self._load_writing_prompts(max_samples_per_source))

        # Dolly
        all_samples.extend(self._load_dolly_nonfactual(max_samples_per_source))

        # Alpaca
        all_samples.extend(self._load_alpaca_nonfactual(max_samples_per_source))

        # Log statistics
        logger.info("=== Dataset Statistics ===")
        logger.info(f"Total samples: {len(all_samples)}")

        label_counts = Counter(s["label"] for s in all_samples)
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count}")

        source_counts = Counter(s["source"] for s in all_samples)
        logger.info("Samples by source:")
        for source, count in sorted(source_counts.items()):
            logger.info(f"  {source}: {count}")

        # Split the data
        random.seed(42)
        random.shuffle(all_samples)

        # Extract texts and labels for stratified split
        texts = [s["text"] for s in all_samples]
        labels = [s["label"] for s in all_samples]
        sources = [s["source"] for s in all_samples]
        orig_labels = [s["original_label"] for s in all_samples]

        # 70% train, 15% validation, 15% test
        train_texts, temp_texts, train_labels, temp_labels, train_sources, temp_sources, train_orig, temp_orig = train_test_split(
            texts, labels, sources, orig_labels,
            test_size=0.3, random_state=42, stratify=labels
        )

        val_texts, test_texts, val_labels, test_labels, val_sources, test_sources, val_orig, test_orig = train_test_split(
            temp_texts, temp_labels, temp_sources, temp_orig,
            test_size=0.5, random_state=42, stratify=temp_labels
        )

        # Create datasets
        def create_dataset(texts, labels, sources, orig_labels):
            return Dataset.from_dict({
                "text": texts,
                "label": labels,
                "label_id": [self.label2id[l] for l in labels],
                "source": sources,
                "original_label": orig_labels,
            })

        dataset_dict = DatasetDict({
            "train": create_dataset(train_texts, train_labels, train_sources, train_orig),
            "validation": create_dataset(val_texts, val_labels, val_sources, val_orig),
            "test": create_dataset(test_texts, test_labels, test_sources, test_orig),
        })

        logger.info(f"\n=== Final Split Sizes ===")
        for split, ds in dataset_dict.items():
            logger.info(f"  {split}: {len(ds)}")

        return dataset_dict


def create_dataset_card() -> str:
    """Create the dataset card content."""
    return """---
license: apache-2.0
task_categories:
  - text-classification
language:
  - en
tags:
  - fact-checking
  - hallucination-detection
  - question-classification
  - information-seeking
  - llm-routing
  - vllm-semantic-router
pretty_name: Fact-Check Classification Dataset
size_categories:
  - 10K<n<100K
---

# Fact-Check Classification Dataset

🎯 **Purpose**: Binary classification dataset for determining whether a prompt needs external fact-checking.

## Dataset Description

This dataset is designed to train classifiers that can route LLM requests based on whether they require external fact verification. It's part of the [vLLM Semantic Router](https://huggingface.co/llm-semantic-router) project.

### Labels

- **FACT_CHECK_NEEDED** (1): Information-seeking questions requiring external verification
  - Factual questions about dates, people, places, events
  - Questions about current events or statistics
  - Claims that can be verified against external sources

- **NO_FACT_CHECK_NEEDED** (0): Prompts that don't need external fact verification
  - Creative writing requests
  - Coding/programming questions
  - Mathematical calculations
  - Opinion or advice requests
  - Summarization tasks

## Dataset Sources

### FACT_CHECK_NEEDED Sources

| Source | Description | Reference |
|--------|-------------|-----------|
| **NISQ (ISQ)** | Gold standard Information-Seeking Questions | [LREC 2024](https://aclanthology.org/2024.lrec-main.1516/) |
| **FactCHD** | Fact-conflicting hallucination detection queries | [Chen et al., 2024](https://huggingface.co/datasets/zjunlp/FactCHD) |
| **FaithDial** | Information-seeking dialogue questions | [TACL 2022](https://huggingface.co/datasets/McGill-NLP/FaithDial) |
| **HaluEval** | QA questions from hallucination benchmark | [ACL EMNLP 2023](https://huggingface.co/datasets/pminervini/HaluEval) |
| **SQuAD** | Reading comprehension questions | [Stanford](https://huggingface.co/datasets/squad) |
| **TruthfulQA** | Questions about common misconceptions | [Lin et al.](https://huggingface.co/datasets/truthful_qa) |
| **TriviaQA** | Trivia questions | [Joshi et al.](https://huggingface.co/datasets/trivia_qa) |
| **HotpotQA** | Multi-hop factual questions | [Yang et al.](https://huggingface.co/datasets/hotpot_qa) |
| **RAG Dataset** | Questions for retrieval-augmented generation | [neural-bridge](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000) |

### NO_FACT_CHECK_NEEDED Sources

| Source | Description | Reference |
|--------|-------------|-----------|
| **NISQ (Non-ISQ)** | Non-Information-Seeking Questions (deliberative, rhetorical) | [LREC 2024](https://aclanthology.org/2024.lrec-main.1516/) |
| **WritingPrompts** | Creative writing prompts from Reddit | [euclaise](https://huggingface.co/datasets/euclaise/writingprompts) |
| **Dolly** | Creative writing, brainstorming, opinions | [Databricks](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| **Alpaca** | Non-factual instructions (coding, creative, math) | [Stanford](https://huggingface.co/datasets/tatsu-lab/alpaca) |

## Dataset Structure

```python
{
    "text": "When was the Eiffel Tower built?",
    "label": "FACT_CHECK_NEEDED",
    "label_id": 1,
    "source": "TriviaQA",
    "original_label": "trivia_question"
}
```

### Fields

- `text` (string): The input prompt/question
- `label` (string): Binary classification label
- `label_id` (int): Numeric label (0 or 1)
- `source` (string): Original dataset source
- `original_label` (string): Original label from source dataset

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("llm-semantic-router/fact-check-classification-dataset")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Example usage
for example in train_data.select(range(5)):
    print(f"Text: {example['text'][:50]}...")
    print(f"Label: {example['label']}")
    print()
```

## Related Models

- [llm-semantic-router/mmbert-fact-check-merged](https://huggingface.co/llm-semantic-router/mmbert-fact-check-merged) - Multilingual fact-check classifier

## Citation

If you use this dataset, please cite the relevant source datasets:

```bibtex
@inproceedings{sun2024nisq,
  title={What Are the Implications of Your Question? Non-Information Seeking Question-Type Identification},
  author={Sun, Yao and others},
  booktitle={LREC-COLING},
  year={2024}
}

@article{dziri2022faithdial,
  title={FaithDial: A Faithful Benchmark for Information-Seeking Dialogue},
  author={Dziri, Nouha and others},
  journal={TACL},
  year={2022}
}

@article{chen2024factchd,
  title={FactCHD: Benchmarking Fact-Conflicting Hallucination Detection},
  author={Chen, Xiang and others},
  year={2024}
}
```

## License

Apache 2.0 - See individual source datasets for their specific licenses.

## Contact

- Organization: [llm-semantic-router](https://huggingface.co/llm-semantic-router)
- Project: [vLLM Semantic Router](https://vllm-semantic-router.com)
"""


def main():
    parser = argparse.ArgumentParser(description="Prepare and upload fact-check dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/fact_check_datasets",
        help="Path to cached datasets (created by setup_datasets.sh)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum samples per source dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/fact_check_dataset_hf",
        help="Local output directory for the dataset",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace Hub",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only save locally, don't upload",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )

    args = parser.parse_args()

    # Prepare dataset
    preparer = FactCheckDatasetPreparer(args.data_dir)
    dataset = preparer.prepare_dataset(max_samples_per_source=args.max_samples)

    # Save locally
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving dataset to {output_dir}")
    dataset.save_to_disk(str(output_dir))

    # Save dataset card
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(create_dataset_card())
    logger.info(f"Saved dataset card to {readme_path}")

    # Upload to HuggingFace
    if args.upload and not args.local_only:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            logger.error("HF_TOKEN not provided. Set --hf-token or HF_TOKEN env var.")
            return

        repo_id = f"{HF_ORG}/{DATASET_NAME}"
        logger.info(f"Uploading to {repo_id}")

        dataset.push_to_hub(
            repo_id,
            token=token,
            private=False,
        )

        # Upload README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        logger.info(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    else:
        logger.info("Skipping upload (use --upload flag to upload to HuggingFace)")


if __name__ == "__main__":
    main()
