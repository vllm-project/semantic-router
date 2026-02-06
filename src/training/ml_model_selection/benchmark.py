#!/usr/bin/env python3
"""
Benchmark script for ML model selection training data generation.

This script:
1. Takes input file (JSONL format) - can be queries or existing training data
2. Automatically extracts unique queries with their metadata (category, ground_truth, etc.)
3. Runs each query against multiple LLM endpoints
4. Measures performance (accuracy) and response_time
5. Outputs training-ready JSONL with category preserved

Input formats supported:
- Simple queries: {"query": "...", "ground_truth": "...", "category": "..."}
- Existing training data: {"query": "...", "model_name": "...", "performance": ..., "category": "..."}
  (Will extract unique queries and re-benchmark against YOUR models)

Usage:
    # Use existing training data file - extracts queries and benchmarks your models
    python benchmark.py --queries training_data.jsonl --model-config models.yaml

    # Simple: All models on same endpoint (e.g., vLLM or Ollama)
    python benchmark.py --queries queries.jsonl --models llama3.2:1b,mistral:7b

    # Different endpoints/auth per model: Use config file
    python benchmark.py --queries queries.jsonl --model-config models.yaml

    # Output to specific file
    python benchmark.py --queries queries.jsonl --models llama3.2:1b --output my_benchmark.jsonl

Config file format (models.yaml):
    models:
      - name: llama3.2:1b
        endpoint: http://localhost:11434/v1  # Ollama

      - name: llama3.2:3b
        endpoint: http://localhost:11434/v1

      - name: gpt-4
        endpoint: https://api.openai.com/v1
        api_key: ${OPENAI_API_KEY}  # Environment variable

      - name: custom-model
        endpoint: https://custom.api.com/v1
        headers:
          Authorization: Bearer ${CUSTOM_TOKEN}
          X-Custom-Header: value

Output is training-ready (includes category if present in input):
    {"query": "...", "model_name": "...", "performance": 0.85, "response_time": 1.2, "category": "math"}

Then train with:
    python train.py --data-file benchmark_output.jsonl --output-dir models/
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai library required. Install with: pip install openai")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None  # Will error if config file is used

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    endpoint: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    max_tokens: int = 1024
    temperature: float = 0.0

    def get_client(self) -> OpenAI:
        """Create OpenAI client for this model."""
        # Resolve environment variables in api_key
        api_key = self.api_key or "dummy"
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "dummy")

        # Resolve environment variables in headers
        default_headers = None
        if self.headers:
            default_headers = {}
            for key, value in self.headers.items():
                if (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    env_var = value[2:-1]
                    value = os.environ.get(env_var, "")
                default_headers[key] = value

        return OpenAI(
            base_url=self.endpoint,
            api_key=api_key,
            default_headers=default_headers,
        )


def load_model_configs(config_path: Path) -> List[ModelConfig]:
    """Load model configurations from YAML file."""
    if yaml is None:
        print(
            "Error: PyYAML required for config files. Install with: pip install pyyaml"
        )
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    configs = []
    for model_data in data.get("models", []):
        config = ModelConfig(
            name=model_data["name"],
            endpoint=model_data.get("endpoint", "http://localhost:8000/v1"),
            api_key=model_data.get("api_key"),
            headers=model_data.get("headers"),
            max_tokens=model_data.get("max_tokens", 1024),
            temperature=model_data.get("temperature", 0.0),
        )
        configs.append(config)

    return configs


def create_model_configs_from_list(
    models: List[str],
    endpoint: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> List[ModelConfig]:
    """Create model configs from simple comma-separated list."""
    return [
        ModelConfig(
            name=model,
            endpoint=endpoint,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for model in models
    ]


@dataclass
class QueryRecord:
    """A single query record for benchmarking."""

    query: str
    ground_truth: Optional[str] = None
    task_name: Optional[str] = None
    metric: Optional[str] = None
    embedding_id: Optional[int] = None
    choices: Optional[str] = None
    category: Optional[str] = None  # Domain category (e.g., "math", "physics")
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single query against a model."""

    query: str
    model_name: str
    response: str
    performance: float  # 0.0 - 1.0
    response_time: float  # seconds
    ground_truth: Optional[str] = None
    task_name: Optional[str] = None
    metric: Optional[str] = None
    embedding_id: Optional[int] = None
    choices: Optional[str] = None
    category: Optional[str] = None  # Domain category (preserved from input)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """Convert to JSONL-compatible dict (same format as benchmark_training_data.jsonl)."""
        result = {
            "query": self.query,
            "model_name": self.model_name,
            "response": self.response,
            "performance": self.performance,
            "response_time": self.response_time,
        }

        # Add optional fields
        if self.ground_truth is not None:
            result["ground_truth"] = self.ground_truth
        if self.task_name is not None:
            result["task_name"] = self.task_name
        if self.metric is not None:
            result["metric"] = self.metric
        if self.embedding_id is not None:
            result["embedding_id"] = self.embedding_id
        if self.choices is not None:
            result["choices"] = self.choices
        if self.category is not None:
            result["category"] = self.category

        # Add any extra fields from input
        result.update(self.extra_fields)

        return result


def format_concise_query(
    query: str, metric: Optional[str] = None, choices: Optional[str] = None
) -> str:
    """
    Format query with concise prompts to get shorter responses.
    Similar to Go benchmark runner's formatQueryForTask.
    """
    # Multiple choice questions
    if choices or metric == "em_mc":
        return f"Answer with ONLY the letter of the correct choice (A, B, C, or D). Do not explain.\n\nQuestion: {query}"

    # Math problems
    if metric in ("MATH", "GSM8K"):
        return (
            f"{query}\n\nAnswer with ONLY the final number or expression. Be concise."
        )

    # Code generation
    if metric == "code_eval":
        return f"Write code to solve this problem. Output ONLY the code, no explanations:\n\n{query}"

    # QA and general questions
    return f"Answer the following question concisely in one sentence:\n\n{query}"


def load_queries(file_path: Path, deduplicate: bool = True) -> List[QueryRecord]:
    """
    Load queries from JSONL file.

    Supports both formats:
    - Simple queries: {"query": "...", "ground_truth": "...", "category": "..."}
    - Full training data: {"query": "...", "model_name": "...", "performance": ..., "category": "..."}

    If deduplicate=True (default), extracts unique queries and preserves their metadata.
    This allows using existing training data files as input for benchmarking new models.

    Args:
        file_path: Path to JSONL file
        deduplicate: If True, return only unique queries (first occurrence wins for metadata)

    Returns:
        List of QueryRecord objects
    """
    seen_queries: Dict[str, QueryRecord] = {}  # query text -> record
    total_records = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                total_records += 1

                query = data.get("query", "")
                if not query:
                    print(f"Warning: Skipping line {line_num} - no query field")
                    continue

                # Skip if we've seen this query and deduplication is enabled
                if deduplicate and query in seen_queries:
                    continue

                # Extract known fields
                record = QueryRecord(
                    query=query,
                    ground_truth=data.get("ground_truth"),
                    task_name=data.get("task_name"),
                    metric=data.get("metric"),
                    embedding_id=data.get("embedding_id"),
                    choices=data.get("choices"),
                    category=data.get("category"),  # Preserve category from input
                )

                # Store any extra fields (excluding benchmark-specific fields)
                known_fields = {
                    "query",
                    "ground_truth",
                    "task_name",
                    "metric",
                    "embedding_id",
                    "choices",
                    "model_name",
                    "response",
                    "performance",
                    "response_time",
                    "category",
                }
                for key, value in data.items():
                    if key not in known_fields:
                        record.extra_fields[key] = value

                seen_queries[query] = record

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")

    records = list(seen_queries.values())

    if deduplicate and total_records != len(records):
        print(f"Loaded {total_records} records from {file_path}")
        print(f"Extracted {len(records)} unique queries (deduplicated)")
    else:
        print(f"Loaded {len(records)} queries from {file_path}")

    # Print category distribution if categories exist
    categories = [r.category for r in records if r.category]
    if categories:
        from collections import Counter

        cat_counts = Counter(categories)
        print(f"Categories: {dict(cat_counts)}")

    return records


def evaluate_response(
    response: str,
    ground_truth: Optional[str],
    metric: Optional[str] = None,
    choices: Optional[str] = None,
) -> float:
    """
    Evaluate response against ground truth using metric-specific logic.

    Returns:
        Performance score between 0.0 and 1.0
    """
    if ground_truth is None:
        # No ground truth - return 0.5 as neutral score
        return 0.5

    response_lower = response.lower().strip()
    truth_lower = ground_truth.lower().strip()

    # Exact match (works for any metric)
    if response_lower == truth_lower:
        return 1.0

    # Metric-specific evaluation
    if metric == "em_mc" or choices:
        # Multiple choice - extract letter from response
        return _evaluate_multiple_choice(response, ground_truth, choices)

    elif metric == "GSM8K":
        # GSM8K - extract number after #### delimiter
        return _evaluate_gsm8k(response, ground_truth)

    elif metric == "MATH":
        # MATH - extract from \boxed{} with LaTeX normalization
        return _evaluate_math(response, ground_truth)

    elif metric == "f1_score":
        # F1 score based on word overlap
        return _evaluate_f1(response, ground_truth)

    elif metric == "code_eval":
        # Code evaluation - try to run assertions
        return _evaluate_code(response, ground_truth)

    elif metric == "commongen_coverage":
        # Check how many required words appear in response
        return _evaluate_commongen(response, ground_truth)

    else:
        # Default: CEM (Conditional Exact Match) - LLMRouter's default
        return _evaluate_cem(response, ground_truth)


def _evaluate_multiple_choice(
    response: str, ground_truth: str, choices: Optional[str]
) -> float:
    """
    Evaluate multiple choice questions by extracting answer letter.
    Aligned with LLMRouter's em_mc metric.
    """
    response_text = response.strip()
    truth_upper = ground_truth.upper().strip()

    # If ground truth is a single letter, look for it in response
    if len(truth_upper) == 1 and truth_upper in "ABCDEFGHIJ":
        # LLMRouter's approach: look for (A), (B), etc. pattern
        parenthesis_pattern = re.findall(r"\(\s*([a-zA-Z])\s*\)", response_text)
        if parenthesis_pattern:
            # Take the last match (usually the final answer)
            found_letter = parenthesis_pattern[-1].upper()
            return 1.0 if found_letter == truth_upper else 0.0

        # Additional patterns for various response styles
        patterns = [
            r"(?:answer(?:\s*is)?:?\s*)([A-J])\b",  # "answer is X"
            r"(?:it['\u2019]?s|is)\s+([A-J])\b",  # "it's X" or "is X"
            r"['\u2019]s\s+([A-J])\b",  # "'s X" pattern
            r"\b([A-J])\s+(?:because|since|as)",  # "X because..."
            r"(?:think|believe|choose)\s+([A-J])\b",  # "think X"
            r"\b([A-J])\s*[.)\]:]",  # Letter followed by punctuation
            r"^([A-J])[.)\]:\s]*$",  # Just the letter (with optional punctuation)
            r"\b([A-J])$",  # Ends with letter
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                found_letter = match.group(1).upper()
                # Skip "I" as a pronoun unless it's clearly an answer
                if found_letter == "I" and not re.match(
                    r"^I[.)\]:\s]*$", response_text.strip(), re.IGNORECASE
                ):
                    continue
                if found_letter == truth_upper:
                    return 1.0
                else:
                    return 0.0  # Wrong letter

        # Fallback: check if the letter appears standalone
        if truth_upper != "I" and re.search(
            r"\b" + truth_upper + r"\b", response_text.upper()
        ):
            return 0.8
        if truth_upper == "I" and re.search(
            r"(?:answer|choice|option)[:\s]+I\b", response_text, re.IGNORECASE
        ):
            return 0.8

    return 0.0


def _evaluate_gsm8k(response: str, ground_truth: str) -> float:
    """
    Evaluate GSM8K math problems.
    Aligned with LLMRouter's gsm8k metric - splits on #### delimiter.
    """
    # Extract answer from ground truth (format: "explanation #### answer")
    if "####" in ground_truth:
        ground_truth_processed = ground_truth.split("####")[-1]
    else:
        ground_truth_processed = ground_truth

    # Clean the ground truth answer
    ground_truth_processed = (
        ground_truth_processed.replace(",", "")
        .replace("$", "")
        .replace(".", "")
        .strip()
    )

    # Extract numbers from response
    numbers = re.findall(r"(\-?[0-9\.\,]+)", response)
    if not numbers:
        return 0.0

    # Find the last valid number (usually the final answer)
    invalid_str = ["", "."]
    final_answer = None
    for answer in reversed(numbers):
        if answer not in invalid_str:
            final_answer = answer
            break

    if final_answer is None:
        return 0.0

    # Clean the predicted answer
    final_answer = (
        final_answer.replace(",", "").replace("$", "").replace(".", "").strip()
    )

    return 1.0 if final_answer == ground_truth_processed else 0.0


def _strip_latex_string(string: str) -> str:
    """
    Normalize LaTeX string for comparison.
    Aligned with LLMRouter's strip_string function.
    """
    # Remove linebreaks and spaces
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")

    # Normalize fractions
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # Remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove degrees
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs and percentage
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # Handle decimal points
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if string and string[0] == ".":
        string = "0" + string

    # Remove "x = " or "k = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # Remove spaces
    string = string.replace(" ", "")

    return string.strip()


def _last_boxed_string(text: str) -> Optional[str]:
    """
    Extract the last \\boxed{} content from text.
    Aligned with LLMRouter's last_boxed_only_string function.
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
    if idx < 0:
        return None

    # Find matching braces
    i = idx
    num_left_braces = 0
    right_brace_idx = None

    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return text[idx : right_brace_idx + 1]


def _remove_boxed(text: str) -> str:
    """Remove \\boxed{} wrapper and return content."""
    if "\\boxed{" in text:
        # Find the content inside \boxed{}
        start = text.find("\\boxed{") + len("\\boxed{")
        depth = 1
        end = start
        while end < len(text) and depth > 0:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        return text[start : end - 1]
    elif "\\boxed " in text:
        return text.split("\\boxed ")[-1].split()[0]
    return text


def _evaluate_math(response: str, ground_truth: str) -> float:
    """
    Evaluate MATH problems by extracting \\boxed{} answers.
    Aligned with LLMRouter's math metric with LaTeX normalization.
    """
    # Extract ground truth from \boxed{} if present
    gt_boxed = _last_boxed_string(ground_truth)
    if gt_boxed:
        ground_truth_processed = _remove_boxed(gt_boxed)
    else:
        ground_truth_processed = ground_truth.strip()

    # Try to extract answer from response's \boxed{}
    try:
        response_boxed = _last_boxed_string(response)
        if response_boxed:
            response_answer = _remove_boxed(response_boxed)
            # Compare with LaTeX normalization
            if _strip_latex_string(response_answer) == _strip_latex_string(
                ground_truth_processed
            ):
                return 1.0
    except Exception:
        pass

    # Fallback: check if normalized ground truth appears in response
    gt_normalized = _strip_latex_string(ground_truth_processed)
    response_normalized = _strip_latex_string(response)

    if gt_normalized and gt_normalized in response_normalized:
        return 0.8

    # Try numeric comparison
    try:
        gt_nums = re.findall(r"-?\d+\.?\d*", ground_truth_processed)
        resp_nums = re.findall(r"-?\d+\.?\d*", response)
        if gt_nums and resp_nums:
            if gt_nums[-1] in resp_nums:
                return 0.7
    except Exception:
        pass

    return 0.0


def _evaluate_f1(response: str, ground_truth: str) -> float:
    """Calculate F1 score based on word overlap."""
    # Clean punctuation and normalize
    import string

    def clean_words(text: str) -> set:
        # Remove punctuation and split
        text = text.lower()
        for p in string.punctuation:
            text = text.replace(p, " ")
        return set(text.split())

    response_words = clean_words(response)
    truth_words = clean_words(ground_truth)

    if not truth_words:
        return 0.0

    # For short ground truths (1-2 words), check containment first
    if len(truth_words) <= 2:
        truth_text = ground_truth.lower()
        for p in string.punctuation:
            truth_text = truth_text.replace(p, "")
        if truth_text.strip() in response.lower():
            return 1.0

    # Remove common stopwords for better matching
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "in",
        "to",
        "and",
        "or",
    }
    response_content = response_words - stopwords
    truth_content = truth_words - stopwords

    # If truth was only stopwords, use original
    if not truth_content:
        truth_content = truth_words
    if not response_content:
        response_content = response_words

    overlap = response_content & truth_content

    if not overlap:
        return 0.0

    precision = len(overlap) / len(response_content) if response_content else 0
    recall = len(overlap) / len(truth_content)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _evaluate_code(response: str, ground_truth: str, timeout: int = 5) -> float:
    """
    Evaluate code by trying to run assertions.
    Aligned with LLMRouter's evaluate_code with timeout protection.
    """
    import signal
    import sys

    # Try to extract code from response
    code_patterns = [
        r"```python\n(.*?)```",
        r"```\n(.*?)```",
        r"def\s+\w+\s*\([^)]*\):.*?(?=\n\n|\Z)",
    ]

    code = response
    for pattern in code_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            code = match.group(1) if match.lastindex else match.group(0)
            break

    # Try to run assertions from ground truth
    try:
        # Ground truth is usually a list of assertions like:
        # "['assert func([1,2])==3', 'assert func([4])==4']"
        if ground_truth.startswith("[") and "assert" in ground_truth:
            assertions = eval(ground_truth)
            if isinstance(assertions, list):
                passed = 0
                total = len(assertions)

                # Timeout handler (Unix only)
                def timeout_handler(signum, frame):
                    raise TimeoutError("Code execution timed out")

                # Set timeout if signal.SIGALRM is available (Unix)
                alarm_supported = hasattr(signal, "SIGALRM")

                for assertion in assertions:
                    try:
                        if alarm_supported:
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(timeout)

                        # Execute the code first, then the assertion
                        local_vars = {}
                        exec(code, {}, local_vars)
                        exec(assertion, local_vars)
                        passed += 1

                    except (AssertionError, TimeoutError):
                        pass
                    except Exception:
                        pass
                    finally:
                        if alarm_supported:
                            signal.alarm(0)

                return passed / total if total > 0 else 0.0
    except Exception:
        pass

    # Fallback: check if function structure matches
    func_match = re.search(r"def\s+(\w+)", response)
    if func_match:
        func_name = func_match.group(1)
        if func_name in ground_truth.lower():
            return 0.5

    return 0.3  # Gave some code response


def _evaluate_commongen(response: str, ground_truth: str) -> float:
    """Evaluate commongen by checking word coverage."""
    # Ground truth is a comma-separated list of words
    required_words = set(w.strip().lower() for w in ground_truth.split(","))
    response_lower = response.lower()

    found = sum(1 for word in required_words if word in response_lower)

    return found / len(required_words) if required_words else 0.0


def _normalize_answer(text: str) -> str:
    """
    Normalize text for evaluation.
    Aligned with LLMRouter's normalize_answer function.
    """
    import string

    # Lowercase
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Fix whitespace
    text = " ".join(text.split())

    return text


def _evaluate_cem(response: str, ground_truth: str) -> float:
    """
    CEM (Conditional Exact Match) evaluation - LLMRouter's default.
    Returns 1.0 if exact match OR ground_truth contained in response, else 0.0
    """
    norm_response = _normalize_answer(response)
    norm_gt = _normalize_answer(ground_truth)

    # Exact match or containment
    if norm_response == norm_gt or norm_gt in norm_response:
        return 1.0

    return 0.0


def benchmark_query(
    model_config: ModelConfig,
    query: QueryRecord,
    concise: bool = False,
) -> BenchmarkResult:
    """Benchmark a single query against a model."""

    client = model_config.get_client()
    start_time = time.time()

    # Format query with concise prompts if enabled
    # But skip concise for code_eval - code needs full response
    query_text = query.query
    if concise and query.metric != "code_eval":
        query_text = format_concise_query(query.query, query.metric, query.choices)

    # Use higher max_tokens for code_eval (code needs more tokens)
    max_tokens = model_config.max_tokens
    if query.metric == "code_eval" and max_tokens < 256:
        max_tokens = 256

    try:
        response = client.chat.completions.create(
            model=model_config.name,
            messages=[{"role": "user", "content": query_text}],
            max_tokens=max_tokens,
            temperature=model_config.temperature,
        )

        response_text = response.choices[0].message.content or ""
        success = True

    except Exception as e:
        response_text = f"Error: {str(e)}"
        success = False

    end_time = time.time()
    response_time = end_time - start_time

    # Evaluate performance
    if success:
        performance = evaluate_response(
            response_text,
            query.ground_truth,
            query.metric,
            query.choices,
        )
    else:
        performance = 0.0

    return BenchmarkResult(
        query=query.query,
        model_name=model_config.name,
        response=response_text,
        performance=performance,
        response_time=response_time,
        ground_truth=query.ground_truth,
        task_name=query.task_name,
        metric=query.metric,
        embedding_id=query.embedding_id,
        choices=query.choices,
        category=query.category,  # Preserve category from input
        extra_fields=query.extra_fields,
    )


def run_benchmark(
    queries: List[QueryRecord],
    model_configs: List[ModelConfig],
    concurrency: int = 4,
    progress: bool = True,
    concise: bool = False,
) -> List[BenchmarkResult]:
    """Run benchmark for all queries against all models."""

    results = []

    # Create tasks: (query, model_config) pairs
    # Group by model to minimize model reloading (important for Ollama/local inference)
    tasks = [(q, m) for m in model_configs for q in queries]
    total_tasks = len(tasks)

    # Group models by endpoint for display
    endpoints = set(m.endpoint for m in model_configs)
    model_names = [m.name for m in model_configs]

    print(
        f"\nBenchmarking {len(queries)} queries × {len(model_configs)} models = {total_tasks} requests"
    )
    print(f"Models: {', '.join(model_names)}")
    if len(endpoints) == 1:
        print(f"Endpoint: {list(endpoints)[0]}")
    else:
        print(f"Endpoints: {len(endpoints)} different endpoints")
        for m in model_configs:
            auth_info = "with API key" if m.api_key else "no auth"
            print(f"  - {m.name}: {m.endpoint} ({auth_info})")
    print(f"Concurrency: {concurrency}")
    print()

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}

        for query, model_config in tasks:
            future = executor.submit(
                benchmark_query,
                model_config,
                query,
                concise,
            )
            futures[future] = (query, model_config)

        # Process results as they complete
        iterator = as_completed(futures)
        if progress:
            iterator = tqdm(iterator, total=total_tasks, desc="Benchmarking")

        for future in iterator:
            query, model_config = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if result.performance == 0.0 and "Error" in result.response:
                    failed += 1

            except Exception as e:
                print(f"\nError processing {model_config.name}: {e}")
                failed += 1

    print(f"\nCompleted: {completed}/{total_tasks} ({failed} errors)")

    return results


def save_results(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSONL file."""

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_jsonl_dict()) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print benchmark summary."""

    print("\n" + "=" * 60)
    print("  Benchmark Summary")
    print("=" * 60)

    # Group by model
    model_stats: Dict[str, Dict[str, Any]] = {}

    for result in results:
        if result.model_name not in model_stats:
            model_stats[result.model_name] = {
                "count": 0,
                "total_perf": 0.0,
                "total_time": 0.0,
                "successes": 0,
            }

        stats = model_stats[result.model_name]
        stats["count"] += 1
        stats["total_perf"] += result.performance
        stats["total_time"] += result.response_time

        if result.performance > 0:
            stats["successes"] += 1

    print(
        f"\n{'Model':<25} {'Queries':>8} {'Avg Perf':>10} {'Avg Time':>10} {'Success':>10}"
    )
    print("-" * 65)

    for model, stats in sorted(model_stats.items()):
        avg_perf = stats["total_perf"] / stats["count"] if stats["count"] > 0 else 0
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        success_rate = (
            stats["successes"] / stats["count"] * 100 if stats["count"] > 0 else 0
        )

        print(
            f"{model:<25} {stats['count']:>8} {avg_perf:>10.3f} {avg_time:>9.2f}s {success_rate:>9.1f}%"
        )

    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs for ML model selection training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing training data - extracts unique queries and benchmarks your models
  python benchmark.py --queries training_data_with_category.jsonl --model-config models.yaml

  # Ollama models (recommended config file approach)
  python benchmark.py --queries queries.jsonl --model-config ollama_models.yaml

  # Simple: All models on same endpoint (local vLLM)
  python benchmark.py --queries queries.jsonl --models llama-3.2-1b,mistral-7b

  # With custom endpoint and API key (OpenAI)
  python benchmark.py --queries queries.jsonl --models gpt-4 \\
      --endpoint https://api.openai.com/v1 --api-key $OPENAI_API_KEY

  # High concurrency for faster benchmarking
  python benchmark.py --queries queries.jsonl --models model1,model2 --concurrency 16

Config file format (models.yaml) - supports Ollama, vLLM, OpenAI, etc:
  models:
    - name: llama3.2:1b            # Ollama models (all same endpoint)
      endpoint: http://localhost:11434/v1
    - name: llama3.2:3b
      endpoint: http://localhost:11434/v1
    - name: mistral:7b
      endpoint: http://localhost:11434/v1
    - name: codellama:7b
      endpoint: http://localhost:11434/v1

    - name: gpt-4                   # OpenAI with API key
      endpoint: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}

    - name: custom-model            # Custom headers
      endpoint: https://custom.api.com/v1
      headers:
        Authorization: Bearer ${CUSTOM_TOKEN}

After benchmarking, train directly (category is preserved from input):
  python train.py --data-file benchmark_output.jsonl --output-dir models/ --device cuda
        """,
    )

    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to JSONL input file. Can be simple queries or existing training data. "
        "Unique queries are extracted automatically. Supports: query, ground_truth, category fields.",
    )

    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of model names (all use same endpoint)",
    )
    model_group.add_argument(
        "--model-config",
        type=str,
        help="Path to YAML config file with model definitions (supports different endpoints/auth per model)",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1"),
        help="API endpoint for --models (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get(
            "LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy")
        ),
        help="API key for --models (uses LLM_API_KEY or OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_output.jsonl",
        help="Output file path (default: benchmark_output.jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in response (default: 1024, used with --models)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0, used with --models)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent requests (default: 4)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to process (for testing). Default: no limit",
    )
    parser.add_argument(
        "--concise",
        action="store_true",
        help="Use concise prompts to get shorter responses (faster inference)",
    )

    args = parser.parse_args()

    # Load queries
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: Queries file not found: {queries_path}")
        sys.exit(1)

    queries = load_queries(queries_path)

    if not queries:
        print("Error: No queries loaded")
        sys.exit(1)

    # Apply limit if specified
    if args.limit and args.limit > 0:
        original_count = len(queries)
        queries = queries[: args.limit]
        print(f"Limited to {len(queries)} queries (from {original_count})")

    # Load model configs
    if args.model_config:
        # Load from YAML config file
        config_path = Path(args.model_config)
        if not config_path.exists():
            print(f"Error: Model config file not found: {config_path}")
            sys.exit(1)
        model_configs = load_model_configs(config_path)
        print(f"Loaded {len(model_configs)} model configurations from {config_path}")
    else:
        # Create from simple model list
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            print("Error: No models specified")
            sys.exit(1)
        model_configs = create_model_configs_from_list(
            models=models,
            endpoint=args.endpoint,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    if not model_configs:
        print("Error: No model configurations loaded")
        sys.exit(1)

    # Run benchmark
    if args.concise:
        print("Using concise prompts for faster inference")

    results = run_benchmark(
        queries=queries,
        model_configs=model_configs,
        concurrency=args.concurrency,
        progress=not args.no_progress,
        concise=args.concise,
    )

    # Save results
    output_path = Path(args.output)
    save_results(results, output_path)

    # Print summary
    print_summary(results)

    print("Next steps:")
    print("  1. Add categories using VSR classifier:")
    print(
        f"     python add_category_to_training_data.py --input {output_path} --output benchmark_with_category.jsonl"
    )
    print("  2. Train models:")
    print(
        "     python train.py --data-file benchmark_with_category.jsonl --output-dir models/"
    )
    print()


if __name__ == "__main__":
    main()
