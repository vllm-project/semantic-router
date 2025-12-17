"""Dataset loaders for hallucination detection benchmarks.

Supports:
- HaluEval: General hallucination evaluation dataset
- Custom datasets in JSONL format
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterator

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class HallucinationSample:
    """A single sample for hallucination detection evaluation."""

    id: str
    context: str  # Ground truth context (tool results / RAG context)
    question: str  # User question
    gold_answer: str  # Correct answer from context
    llm_response: Optional[str] = None  # Generated LLM response
    hallucination_spans: Optional[List[dict]] = None  # Annotated hallucination spans
    is_faithful: Optional[bool] = None  # Whether response is faithful to context
    metadata: Optional[dict] = None  # Additional metadata


class DatasetInterface(ABC):
    """Abstract interface for hallucination datasets."""

    @abstractmethod
    def load(self, max_samples: Optional[int] = None) -> List[HallucinationSample]:
        """Load the dataset."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass


class HaluEvalDataset(DatasetInterface):
    """HaluEval dataset loader - general hallucination evaluation."""

    def name(self) -> str:
        return "halueval"

    def load(self, max_samples: Optional[int] = None) -> List[HallucinationSample]:
        """Load HaluEval dataset from HuggingFace."""
        if not HAS_DATASETS:
            raise ImportError(
                "datasets package not installed. Run: pip install datasets"
            )

        print(f"Loading HaluEval dataset...")

        try:
            dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        except Exception as e:
            print(f"Error loading HaluEval: {e}")
            raise

        samples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            samples.append(
                HallucinationSample(
                    id=f"halueval_{i}",
                    context=item.get("knowledge", ""),
                    question=item.get("question", ""),
                    gold_answer=item.get("right_answer", ""),
                    llm_response=item.get(
                        "hallucinated_answer", item.get("answer", "")
                    ),
                    is_faithful=item.get("hallucination", "no") == "no",
                    metadata={
                        "dataset": "halueval",
                    },
                )
            )

        print(f"Loaded {len(samples)} samples from HaluEval")
        return samples


class FinancialFactEvalDataset(DatasetInterface):
    """FinanceBench dataset loader.
    Loads `PatronusAI/financebench` from the Hugging Face datasets hub.
    The dataset's schema varies between versions and splits, so the loader
    maps common fields with sensible fallbacks.
    """

    def name(self) -> str:
        return "financebench"

    def load(self, max_samples: Optional[int] = None) -> List[HallucinationSample]:
        if not HAS_DATASETS:
            raise ImportError(
                "datasets package not installed. Run: pip install datasets"
            )

        print("Loading FinanceBench dataset...")
        try:
            ds = load_dataset("PatronusAI/financebench")
        except Exception as e:
            raise RuntimeError(f"Failed to load PatronusAI/financebench: {e}")

        samples: List[HallucinationSample] = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            # Map common fields with fallbacks for different dataset schemas.
            # The financebench dataset contains financial claims/questions and
            # references; exact field names may vary across releases.
            id_ = item.get("id") or item.get("financebench_id") or f"finance_{i}"
            evidence = item.get("evidence")
            if isinstance(evidence, dict) and "evidence_text" in evidence:
                context = evidence["evidence_text"]
            else:
                context = ""
                print(
                    f"Warning: 'evidence_text' not found in evidence for sample {id_}"
                )
            question = item.get("question")
            gold_answer = item.get("answer")
            llm_response = None
            hallucination_spans = None
            is_faithful = None
            # Some datasets include a binary correctness/faithful flag.
            if "is_faithful" in item:
                is_faithful = bool(item.get("is_faithful"))

            samples.append(
                HallucinationSample(
                    id=str(id_),
                    context=str(context or ""),
                    question=str(question or ""),
                    gold_answer=str(gold_answer or ""),
                    llm_response=(
                        str(llm_response) if llm_response is not None else None
                    ),
                    hallucination_spans=hallucination_spans,
                    is_faithful=is_faithful,
                    metadata={"dataset": "financebench", "raw_keys": list(item.keys())},
                )
            )

        print(f"Loaded {len(samples)} samples from FinanceBench")
        return samples


class CustomDataset(DatasetInterface):
    """Load custom dataset from JSONL file."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._name = self.file_path.stem

    def name(self) -> str:
        return self._name

    def load(self, max_samples: Optional[int] = None) -> List[HallucinationSample]:
        """Load dataset from JSONL file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        samples = []
        with open(self.file_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                item = json.loads(line.strip())
                samples.append(
                    HallucinationSample(
                        id=item.get("id", f"custom_{i}"),
                        context=item.get("context", item.get("knowledge", "")),
                        question=item.get("question", item.get("query", "")),
                        gold_answer=item.get("gold_answer", item.get("answer", "")),
                        llm_response=item.get("llm_response", item.get("response", "")),
                        hallucination_spans=item.get("hallucination_spans", []),
                        is_faithful=item.get("is_faithful"),
                        metadata=item.get("metadata", {}),
                    )
                )

        print(f"Loaded {len(samples)} samples from {self.file_path}")
        return samples


def get_dataset(name: str, **kwargs) -> DatasetInterface:
    """Factory function to get a dataset by name."""
    datasets = {
        "halueval": HaluEvalDataset,
        "financebench": FinancialFactEvalDataset,
    }

    if name in datasets:
        return datasets[name]()
    elif Path(name).exists():
        return CustomDataset(name)
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
