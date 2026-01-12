"""
Fact-Check Classification Fine-tuning with Enhanced LoRA Training
Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters for efficient fact-check detection.

🎯 **PURPOSE**: Train a model to classify whether a prompt needs external fact-checking
   - FACT_CHECK_NEEDED: Information-seeking questions requiring external verification
   - NO_FACT_CHECK_NEEDED: Creative, opinion, coding, math - no verification needed

📚 **DATASETS USED** (publicly available, peer-reviewed):
   - QASPER (allenai/qasper): 5,049 information-seeking questions over research papers
     Source: "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers" (NAACL 2021)
     https://aclanthology.org/2021.naacl-main.365/

   - ELI5 (eli5_category): Explain Like I'm 5 - questions requiring factual explanations
     Source: Facebook AI Research

   - WritingPrompts (euclaise/writingprompts): Creative writing prompts (negative examples)

   - CodeSearchNet (code_search_net): Programming questions (negative examples)

Usage:
    # Train with recommended parameters
    python fact_check_bert_finetuning_lora.py --mode train --model bert-base-uncased --epochs 8 --lora-rank 16 --max-samples 2000

    # Train with custom LoRA parameters
    python fact_check_bert_finetuning_lora.py --mode train --lora-rank 16 --lora-alpha 32 --batch-size 2

    # Test inference with trained LoRA model
    python fact_check_bert_finetuning_lora.py --mode test --model-path lora_fact_check_classifier_bert-base-uncased_r16_model

    # Quick training test (for debugging)
    python fact_check_bert_finetuning_lora.py --mode train --model bert-base-uncased --epochs 1 --max-samples 100

Supported models:
    - mmbert-base: mmBERT base model (149M parameters, 1800+ languages, RECOMMENDED)
    - bert-base-uncased: Standard BERT base model (110M parameters, most stable)
    - roberta-base: RoBERTa base model (125M parameters, better context understanding)
    - modernbert-base: ModernBERT base model (149M parameters, latest architecture)

Key Features:
    - LoRA (Low-Rank Adaptation) for binary fact-check classification
    - 99%+ parameter reduction (only ~0.02% trainable parameters)
    - Uses real, peer-reviewed datasets instead of synthetic data
    - Auto-merge functionality: Generates both LoRA adapters and Rust-compatible models
"""

import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Import common LoRA utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_lora_utils import (
    clear_gpu_memory,
    create_lora_config,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
)

# Setup logging
logger = setup_logging()

# Label definitions
FACT_CHECK_NEEDED = "FACT_CHECK_NEEDED"
NO_FACT_CHECK_NEEDED = "NO_FACT_CHECK_NEEDED"


def create_tokenizer_for_model(model_path: str, base_model_name: str = None):
    """
    Create tokenizer with model-specific configuration.

    Args:
        model_path: Path to load tokenizer from
        base_model_name: Optional base model name for configuration
    """
    model_identifier = base_model_name or model_path

    if "roberta" in model_identifier.lower():
        logger.info("Using RoBERTa tokenizer with add_prefix_space=True")
        return AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    else:
        return AutoTokenizer.from_pretrained(model_path)


class FactCheckDataset:
    """
    Dataset class for fact-check classification fine-tuning.

    Uses publicly available, peer-reviewed datasets:
    - QASPER (allenai/qasper): Information-seeking questions over research papers
    - ELI5: Explain Like I'm 5 questions requiring factual explanations
    - WritingPrompts: Creative writing prompts (negative examples)
    - CodeSearchNet: Programming questions/docstrings (negative examples)

    References:
    - QASPER: "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers" (NAACL 2021)
      https://aclanthology.org/2021.naacl-main.365/
    - NISQ concept: "What Are the Implications of Your Question? Non-Information Seeking Question-Type Identification" (LREC 2024)
      https://aclanthology.org/2024.lrec-main.1516/
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize the dataset loader.

        Args:
            data_dir: Optional path to cached datasets directory.
                      If provided, will look for pre-downloaded datasets here.
                      Run setup_datasets.sh to populate this directory.
        """
        self.label2id = {NO_FACT_CHECK_NEEDED: 0, FACT_CHECK_NEEDED: 1}
        self.id2label = {0: NO_FACT_CHECK_NEEDED, 1: FACT_CHECK_NEEDED}

        # Data directory for cached/cloned datasets
        self.data_dir = data_dir or os.environ.get("FACT_CHECK_DATA_DIR", None)

        # Dataset configurations
        self.dataset_configs = {
            # Information-seeking questions (FACT_CHECK_NEEDED)
            "qasper": {
                "name": "allenai/qasper",
                "split": "train",
                "text_field": "question",
                "label": FACT_CHECK_NEEDED,
                "description": "Information-seeking questions over research papers (NAACL 2021)",
            },
            "eli5": {
                "name": "eli5_category",
                "split": "train",
                "text_field": "title",
                "label": FACT_CHECK_NEEDED,
                "description": "Explain Like I'm 5 - factual explanation questions",
            },
            "natural_questions": {
                "name": "google-research-datasets/natural_questions",
                "config": "default",
                "split": "train",
                "text_field": "question.text",
                "label": FACT_CHECK_NEEDED,
                "description": "Google Natural Questions - real user queries",
            },
            # Non-information-seeking (NO_FACT_CHECK_NEEDED)
            "writing_prompts": {
                "name": "euclaise/writingprompts",
                "split": "train",
                "text_field": "prompt",
                "label": NO_FACT_CHECK_NEEDED,
                "description": "Creative writing prompts from Reddit",
            },
            "code_search_net": {
                "name": "code_search_net",
                "config": "python",
                "split": "train",
                "text_field": "func_documentation_string",
                "label": NO_FACT_CHECK_NEEDED,
                "description": "Code documentation - programming/technical requests",
            },
        }

        # Fallback patterns for when datasets fail to load
        self._fallback_fact_check = [
            "When was {company} founded?",
            "Who is the CEO of {company}?",
            "What is the population of {city}?",
            "When did {event} happen?",
            "What year did {person} win the Nobel Prize?",
            "What is the capital of {country}?",
            "How tall is {building}?",
            "Who invented {invention}?",
            "What is the GDP of {country}?",
            "Is it true that {claim}?",
        ]

        self._fallback_no_fact_check = [
            "Write a poem about {topic}",
            "Create a Python function to {task}",
            "What do you think about {topic}?",
            "Calculate {expression}",
            "How do I {action}?",
            "Write a story about {theme}",
            "Debug this code: {code}",
            "Summarize this text",
            "Hello, how are you?",
            "Which is better, {option1} or {option2}?",
        ]

        self._fill_values = {
            "company": ["Microsoft", "Apple", "Google", "Amazon", "Tesla", "OpenAI"],
            "city": ["New York", "London", "Tokyo", "Paris", "Shanghai", "Mumbai"],
            "country": [
                "United States",
                "China",
                "Japan",
                "Germany",
                "France",
                "India",
            ],
            "person": ["Einstein", "Marie Curie", "Newton", "Turing", "Ada Lovelace"],
            "event": ["World War 2", "the Moon landing", "the French Revolution"],
            "building": [
                "the Eiffel Tower",
                "the Empire State Building",
                "the Burj Khalifa",
            ],
            "invention": [
                "the telephone",
                "the light bulb",
                "the internet",
                "the airplane",
            ],
            "claim": ["the Earth is round", "water boils at 100°C at sea level"],
            "topic": ["artificial intelligence", "climate change", "space exploration"],
            "task": ["sort a list", "read a file", "parse JSON"],
            "action": ["set up a server", "deploy an application"],
            "theme": ["adventure", "technology", "nature"],
            "code": ["def foo(): pass"],
            "expression": ["25 * 4", "sqrt(144)", "2^10"],
            "option1": ["Python", "React", "AWS"],
            "option2": ["Java", "Vue", "GCP"],
        }

    def _load_squad_questions(self, max_samples: int) -> List[str]:
        """Load SQuAD questions (factual reading comprehension questions)."""
        logger.info("Loading SQuAD dataset (factual questions)...")
        questions = []
        try:
            dataset = load_dataset("squad", split="train")
            for item in dataset:
                if len(questions) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 300:
                    questions.append(q.strip())
            logger.info(f"Loaded {len(questions)} questions from SQuAD")
        except Exception as e:
            logger.warning(f"Failed to load SQuAD: {e}")
        return questions

    def _load_trivia_qa_questions(self, max_samples: int) -> List[str]:
        """Load TriviaQA questions (factual trivia questions)."""
        logger.info("Loading TriviaQA dataset (factual trivia questions)...")
        questions = []
        try:
            # Use streaming for large dataset
            dataset = load_dataset("trivia_qa", "rc", split="train", streaming=True)
            for item in dataset:
                if len(questions) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 300:
                    questions.append(q.strip())
            logger.info(f"Loaded {len(questions)} questions from TriviaQA")
        except Exception as e:
            logger.warning(f"Failed to load TriviaQA: {e}")
        return questions

    def _load_hotpot_qa_questions(self, max_samples: int) -> List[str]:
        """Load HotpotQA questions (multi-hop factual questions)."""
        logger.info("Loading HotpotQA dataset (multi-hop factual questions)...")
        questions = []
        try:
            # Use streaming for large dataset
            dataset = load_dataset(
                "hotpot_qa", "fullwiki", split="train", streaming=True
            )
            for item in dataset:
                if len(questions) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 300:
                    questions.append(q.strip())
            logger.info(f"Loaded {len(questions)} questions from HotpotQA")
        except Exception as e:
            logger.warning(f"Failed to load HotpotQA: {e}")
        return questions

    def _load_truthful_qa_questions(self, max_samples: int) -> List[str]:
        """
        Load TruthfulQA questions (high-risk factual queries about common misconceptions).

        Reference: "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (Lin et al.)
        These are questions where factual accuracy is critical.
        """
        logger.info("Loading TruthfulQA dataset (high-risk factual queries)...")
        questions = []
        try:
            dataset = load_dataset(
                "truthful_qa", "generation", split="validation", streaming=True
            )
            for item in dataset:
                if len(questions) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 300:
                    questions.append(q.strip())
            logger.info(f"Loaded {len(questions)} questions from TruthfulQA")
        except Exception as e:
            logger.warning(f"Failed to load TruthfulQA: {e}")
        return questions

    def _load_coqa_questions(self, max_samples: int) -> List[str]:
        """
        Load CoQA questions (conversational QA - factual questions in context).

        Reference: "CoQA: A Conversational Question Answering Challenge" (ACL)
        Contains 127k questions across multiple domains.
        """
        logger.info("Loading CoQA dataset (conversational factual questions)...")
        questions = []
        try:
            dataset = load_dataset("coqa", split="train", streaming=True)
            for item in dataset:
                if len(questions) >= max_samples:
                    break
                # CoQA has a list of questions per story
                q_list = item.get("questions", [])
                for q in q_list:
                    if len(questions) >= max_samples:
                        break
                    if q and len(q) > 10 and len(q) < 300:
                        questions.append(q.strip())
            logger.info(f"Loaded {len(questions)} questions from CoQA")
        except Exception as e:
            logger.warning(f"Failed to load CoQA: {e}")
        return questions

    def _load_halueval_questions(self, max_samples: int) -> List[str]:
        """
        Load HaluEval QA questions (factual knowledge-seeking questions).

        Reference: "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs"
                   (ACL EMNLP 2023)
        Source: https://github.com/RUCAIBox/HaluEval

        We extract only the QUESTION field, not the hallucination labels
        (which are for answer verification, not prompt classification).
        """
        logger.info("Loading HaluEval dataset (QA questions for fact-check)...")
        questions = []
        try:
            # Load QA samples - these are factual questions
            dataset = load_dataset(
                "pminervini/HaluEval", "qa_samples", split="data", streaming=True
            )
            for item in dataset:
                if len(questions) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 500:
                    questions.append(q.strip())
            logger.info(f"Loaded {len(questions)} questions from HaluEval QA")
        except Exception as e:
            logger.warning(f"Failed to load HaluEval: {e}")
        return questions

    def _load_factchd_questions(self, max_samples: int) -> List[str]:
        """
        Load FactCHD questions (fact-conflicting hallucination detection queries).

        Reference: "FactCHD: Benchmarking Fact-Conflicting Hallucination Detection"
                   (Chen et al., 2024)
        Source: https://huggingface.co/datasets/zjunlp/FactCHD

        FactCHD contains queries designed to test factual accuracy of LLM responses.
        These are explicitly fact-seeking questions used by the HuDEx model.

        Note: This dataset uses a deprecated script loader on HuggingFace, so we load
        it directly from the cloned JSONL files.
        """
        logger.info(
            "Loading FactCHD dataset (fact-conflicting hallucination queries)..."
        )
        questions = []
        try:
            # Check for cached dataset first
            factchd_path = None
            if self.data_dir:
                factchd_path = (
                    Path(self.data_dir) / "FactCHD_dataset" / "fact_train_noe.jsonl"
                )
                if factchd_path.exists():
                    logger.info(f"Using cached FactCHD dataset from {factchd_path}")

            # Fallback to other locations
            if not factchd_path or not factchd_path.exists():
                factchd_path = (
                    Path(__file__).parent.parent.parent.parent.parent
                    / "FactCHD_dataset"
                    / "fact_train_noe.jsonl"
                )
            if not factchd_path.exists():
                factchd_path = Path("FactCHD_dataset/fact_train_noe.jsonl")

            if not factchd_path.exists():
                logger.warning(
                    f"FactCHD dataset not found. Run setup_datasets.sh or clone from HuggingFace."
                )
                return []

            seen_questions = set()
            with open(factchd_path, "r") as f:
                for line in f:
                    if len(questions) >= max_samples:
                        break
                    try:
                        item = json.loads(line)
                        query = item.get("query", "")
                        # These are factual queries
                        if (
                            query
                            and len(query) > 15
                            and len(query) < 500
                            and query not in seen_questions
                        ):
                            questions.append(query.strip())
                            seen_questions.add(query)
                    except json.JSONDecodeError:
                        continue

            logger.info(f"Loaded {len(questions)} questions from FactCHD")
        except Exception as e:
            logger.warning(f"Failed to load FactCHD: {e}")
        return questions

    def _load_faithdial_questions(self, max_samples: int) -> List[str]:
        """
        Load FaithDial questions (information-seeking dialogue questions).

        Reference: "FaithDial: A Faithful Benchmark for Information-Seeking Dialogue"
                   (Dziri et al., TACL 2022)
        Source: https://huggingface.co/datasets/McGill-NLP/FaithDial

        FaithDial contains knowledge-grounded dialogues where users ask information-seeking
        questions. We extract the user questions from dialogue history.

        Note: This dataset uses a deprecated script loader on HuggingFace, so we load
        it directly from the cloned JSON files.
        """
        logger.info(
            "Loading FaithDial dataset (information-seeking dialogue questions)..."
        )
        questions = []
        try:
            # Check for cached dataset first
            faithdial_path = None
            if self.data_dir:
                faithdial_path = (
                    Path(self.data_dir) / "FaithDial_dataset" / "data" / "train.json"
                )
                if faithdial_path.exists():
                    logger.info(f"Using cached FaithDial dataset from {faithdial_path}")

            # Fallback to other locations
            if not faithdial_path or not faithdial_path.exists():
                faithdial_path = (
                    Path(__file__).parent.parent.parent.parent.parent
                    / "FaithDial_dataset"
                    / "data"
                    / "train.json"
                )
            if not faithdial_path.exists():
                faithdial_path = Path("FaithDial_dataset/data/train.json")

            if not faithdial_path.exists():
                logger.warning(
                    f"FaithDial dataset not found. Run setup_datasets.sh or clone from HuggingFace."
                )
                return []

            with open(faithdial_path, "r") as f:
                data = json.load(f)

            seen_questions = set()
            for dialogue in data:
                if len(questions) >= max_samples:
                    break
                for utt in dialogue.get("utterances", []):
                    if len(questions) >= max_samples:
                        break
                    history = utt.get("history", [])
                    if history:
                        # Get the user's question (last item in history)
                        last_user_msg = history[-1] if history else ""
                        # Filter for actual questions
                        if (
                            "?" in last_user_msg
                            and len(last_user_msg) > 15
                            and len(last_user_msg) < 300
                            and last_user_msg not in seen_questions
                        ):
                            # Skip generic follow-ups
                            if not any(
                                skip in last_user_msg.lower()
                                for skip in [
                                    "what else",
                                    "tell me more",
                                    "anything else",
                                    "go on",
                                ]
                            ):
                                questions.append(last_user_msg.strip())
                                seen_questions.add(last_user_msg)

            logger.info(f"Loaded {len(questions)} questions from FaithDial")
        except Exception as e:
            logger.warning(f"Failed to load FaithDial: {e}")
        return questions

    def _load_rag_dataset_questions(self, max_samples: int) -> List[str]:
        """
        Load RAG dataset questions (questions designed for retrieval-augmented generation).

        These are factual questions that require external knowledge retrieval.
        Source: neural-bridge/rag-dataset-12000
        """
        logger.info(
            "Loading RAG dataset (questions for retrieval-augmented generation)..."
        )
        questions = []
        try:
            dataset = load_dataset(
                "neural-bridge/rag-dataset-12000", split="train", streaming=True
            )
            for item in dataset:
                if len(questions) >= max_samples:
                    break
                q = item.get("question", "")
                if q and len(q) > 10 and len(q) < 500:
                    questions.append(q.strip())
            logger.info(f"Loaded {len(questions)} questions from RAG dataset")
        except Exception as e:
            logger.warning(f"Failed to load RAG dataset: {e}")
        return questions

    def _load_nisq_dataset(
        self, max_samples_isq: int, max_samples_nisq: int
    ) -> Tuple[List[str], List[str]]:
        """
        Load NISQ dataset (Information-Seeking vs Non-Information-Seeking Questions).

        This is the GOLD STANDARD dataset for this classification task!

        Reference: "What Are the Implications of Your Question? Non-Information Seeking
                   Question-Type Identification" (ACL Anthology LREC 2024)
        Source: https://github.com/YaoSun0422/NISQ_dataset

        Labels:
        - ISQ: Information-Seeking Questions → FACT_CHECK_NEEDED
        - Deliberative, Rhetorical, OTHERS: Non-Information-Seeking → NO_FACT_CHECK_NEEDED

        Returns:
            Tuple of (isq_questions, nisq_questions)
        """
        logger.info("Loading NISQ dataset (ISQ vs Non-ISQ - gold standard)...")
        isq_questions = []
        nisq_questions = []

        try:
            import subprocess
            import tempfile
            import csv

            # Check for cached dataset first
            cached_path = None
            if self.data_dir:
                cached_path = os.path.join(
                    self.data_dir, "NISQ_dataset", "final_train.csv"
                )
                if os.path.exists(cached_path):
                    logger.info(f"Using cached NISQ dataset from {cached_path}")

            if cached_path and os.path.exists(cached_path):
                csv_path = cached_path
            else:
                # Clone the dataset to a temp directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    repo_path = os.path.join(tmpdir, "NISQ_dataset")
                    result = subprocess.run(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "https://github.com/YaoSun0422/NISQ_dataset.git",
                            repo_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

                    if result.returncode != 0:
                        logger.warning(f"Failed to clone NISQ dataset: {result.stderr}")
                        return [], []

                    csv_path = os.path.join(repo_path, "final_train.csv")

            if not os.path.exists(csv_path):
                logger.warning(f"NISQ dataset CSV not found at {csv_path}")
                return [], []

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    question = row.get("question", "").strip()
                    label = row.get("label", "").upper()

                    if not question or len(question) < 5 or len(question) > 500:
                        continue

                    if label == "ISQ":
                        if len(isq_questions) < max_samples_isq:
                            isq_questions.append(question)
                    elif label in ["DELIBERATIVE", "RHETORICAL", "OTHERS"]:
                        if len(nisq_questions) < max_samples_nisq:
                            nisq_questions.append(question)

            logger.info(
                f"Loaded NISQ dataset: {len(isq_questions)} ISQ, {len(nisq_questions)} NISQ"
            )

        except Exception as e:
            logger.warning(f"Failed to load NISQ dataset: {e}")

        return isq_questions, nisq_questions

    def _load_writing_prompts(self, max_samples: int) -> List[str]:
        """Load creative writing prompts (non-information-seeking)."""
        logger.info("Loading WritingPrompts dataset (creative prompts)...")
        prompts = []
        try:
            dataset = load_dataset(
                "euclaise/writingprompts", split="train", trust_remote_code=True
            )
            for item in dataset:
                if len(prompts) >= max_samples:
                    break
                prompt = item.get("prompt", "")
                if prompt and len(prompt) > 10 and len(prompt) < 500:
                    # Clean up common prefixes
                    prompt = prompt.strip()
                    if prompt.startswith("[WP]"):
                        prompt = prompt[4:].strip()
                    if prompt:
                        prompts.append(prompt)
            logger.info(f"Loaded {len(prompts)} prompts from WritingPrompts")
        except Exception as e:
            logger.warning(f"Failed to load WritingPrompts: {e}")
        return prompts

    def _load_alpaca_nonfactual(self, max_samples: int) -> List[str]:
        """
        Load non-factual instructions from Alpaca dataset.

        Reference: "Alpaca: A Strong, Replicable Instruction-Following Model"
                   (Stanford, 2023)
        Source: https://huggingface.co/datasets/tatsu-lab/alpaca

        We filter for non-factual instructions: coding, creative writing,
        math calculations, and opinion/advice requests.
        """
        logger.info("Loading Alpaca dataset (non-factual instructions)...")
        instructions = []
        try:
            dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

            # Keywords for non-factual categories
            coding_kw = [
                "code",
                "program",
                "function",
                "algorithm",
                "python",
                "javascript",
                "script",
                "implement",
                "debug",
                "fix this",
                "write a class",
            ]
            creative_kw = [
                "write a story",
                "write a poem",
                "creative",
                "imagine",
                "compose",
                "create a",
                "fiction",
                "narrative",
                "describe a scene",
            ]
            math_kw = [
                "calculate",
                "compute",
                "solve",
                "equation",
                "add ",
                "subtract",
                "multiply",
                "divide",
                "sum of",
                "product of",
            ]
            opinion_kw = [
                "opinion",
                "think about",
                "best way",
                "recommend",
                "suggest",
                "advice",
                "prefer",
                "favorite",
                "better",
            ]
            task_kw = [
                "summarize",
                "translate",
                "rewrite",
                "paraphrase",
                "edit",
                "proofread",
                "format",
                "organize",
                "list",
            ]

            all_keywords = coding_kw + creative_kw + math_kw + opinion_kw + task_kw
            seen = set()

            for item in dataset:
                if len(instructions) >= max_samples:
                    break

                instr = item.get("instruction", "")
                instr_lower = instr.lower()

                # Check if it matches non-factual patterns
                if any(kw in instr_lower for kw in all_keywords):
                    if (
                        instr
                        and len(instr) > 10
                        and len(instr) < 300
                        and instr not in seen
                    ):
                        instructions.append(instr.strip())
                        seen.add(instr)

            logger.info(
                f"Loaded {len(instructions)} instructions from Alpaca (non-factual)"
            )
        except Exception as e:
            logger.warning(f"Failed to load Alpaca: {e}")
        return instructions

    def _load_dolly_nonfactual(self, max_samples: int) -> List[str]:
        """
        Load non-factual instructions from Dolly dataset.

        Reference: "Dolly: An Open-Source Instruction-Following LLM"
                   (Databricks, 2023)
        Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k

        Categories we use for NO_FACT_CHECK_NEEDED:
        - creative_writing: 709 samples (poems, stories, scenes)
        - brainstorming: 1766 samples (ideas, suggestions, opinions)
        - summarization: 1188 samples (task-based, not factual)

        These categories help fix edge cases like:
        - "What is love?" (creative/philosophical)
        - "What is the meaning of life?" (philosophical)
        - "Is Python better than JavaScript?" (opinion)
        - "What should I name my cat?" (creative/brainstorming)
        """
        logger.info("Loading Dolly dataset (creative/brainstorming/opinion)...")
        instructions = []
        try:
            dataset = load_dataset(
                "databricks/databricks-dolly-15k", split="train", streaming=True
            )

            # Categories that are clearly non-factual
            nonfactual_categories = [
                "creative_writing",
                "brainstorming",
                "summarization",
            ]
            seen = set()

            for item in dataset:
                if len(instructions) >= max_samples:
                    break

                category = item.get("category", "")
                instr = item.get("instruction", "")

                # Only include non-factual categories
                if category in nonfactual_categories:
                    if (
                        instr
                        and len(instr) > 10
                        and len(instr) < 500
                        and instr not in seen
                    ):
                        instructions.append(instr.strip())
                        seen.add(instr)

            logger.info(
                f"Loaded {len(instructions)} instructions from Dolly (non-factual)"
            )
        except Exception as e:
            logger.warning(f"Failed to load Dolly: {e}")
        return instructions

    def _load_code_search_net(self, max_samples: int) -> List[str]:
        """Load code documentation (programming/technical requests)."""
        logger.info("Loading CodeSearchNet dataset (programming requests)...")
        docs = []
        try:
            dataset = load_dataset(
                "code_search_net", "python", split="train", trust_remote_code=True
            )
            for item in dataset:
                if len(docs) >= max_samples:
                    break
                doc = item.get("func_documentation_string", "")
                if doc and len(doc) > 10 and len(doc) < 300:
                    # Convert docstrings to request format
                    doc = doc.strip()
                    if not doc.endswith("?"):
                        doc = f"Write code to: {doc}"
                    docs.append(doc)
            logger.info(f"Loaded {len(docs)} items from CodeSearchNet")
        except Exception as e:
            logger.warning(f"Failed to load CodeSearchNet: {e}")
        return docs

    def _generate_fallback_samples(self, templates: List[str], count: int) -> List[str]:
        """Generate fallback samples from templates when datasets fail."""
        samples = set()
        max_attempts = count * 20
        attempts = 0

        while len(samples) < count and attempts < max_attempts:
            template = random.choice(templates)
            result = template
            for key, values in self._fill_values.items():
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, random.choice(values))
            if result not in samples:
                samples.add(result)
            attempts += 1

        return list(samples)

    def load_datasets(self, max_samples: int = 2000) -> Tuple[List[str], List[int]]:
        """
        Load and combine datasets for fact-check classification.

        Uses publicly available, peer-reviewed datasets:

        FACT_CHECK_NEEDED (from hallucination/QA research):
        - NISQ_ISQ: Gold standard Information-Seeking Questions (ACL LREC 2024)
        - HaluEval: QA questions from hallucination benchmark (ACL EMNLP 2023)
        - FaithDial: Information-seeking dialogue questions (TACL 2022) - used by HuDEx
        - FactCHD: Fact-conflicting hallucination queries (Chen et al., 2024) - used by HuDEx
        - RAGTruth-style: Questions from RAG datasets (require retrieval)
        - SQuAD: Stanford Question Answering Dataset (100k+ Wikipedia fact questions)
        - TriviaQA: 650k trivia question-answer-evidence triples
        - HotpotQA: Multi-hop factual reasoning questions
        - TruthfulQA: High-risk factual queries about common misconceptions
        - CoQA: Conversational QA (127k questions across domains)

        NO_FACT_CHECK_NEEDED:
        - NISQ_NonISQ: Gold standard Non-Information-Seeking Questions
        - Dolly: Creative writing, brainstorming, opinion (helps edge cases!)
        - WritingPrompts: 300k creative writing prompts from Reddit
        - Alpaca: Non-factual instructions (coding, creative, math, opinion)

        Note: We only use the QUESTION/PROMPT fields from hallucination datasets,
        NOT the hallucination labels (those are for answer verification).

        Args:
            max_samples: Maximum total samples to load

        Returns:
            Tuple of (texts, labels) where labels are 0 or 1
        """
        logger.info(f"Loading fact-check datasets (target: {max_samples} samples)...")

        samples_per_class = max_samples // 2
        samples_per_source = (
            samples_per_class // 10
        )  # Divide between 10 factual sources

        # Load FACT_CHECK_NEEDED samples from multiple sources
        fact_check_texts = []
        no_fact_check_texts_from_nisq = []
        source_counts = {}

        # NISQ Dataset - GOLD STANDARD for ISQ vs Non-ISQ (Tier 1)
        # This is the most directly relevant dataset for our task
        nisq_isq, nisq_non_isq = self._load_nisq_dataset(
            max_samples_isq=samples_per_source,
            max_samples_nisq=samples_per_source * 2,  # More NISQ samples available
        )
        fact_check_texts.extend(nisq_isq)
        no_fact_check_texts_from_nisq.extend(nisq_non_isq)
        source_counts["NISQ_ISQ"] = len(nisq_isq)
        source_counts["NISQ_NonISQ"] = len(nisq_non_isq)

        # HaluEval QA - questions from hallucination benchmark (ACL EMNLP 2023)
        # We only use QUESTION field, not hallucination labels
        halueval_questions = self._load_halueval_questions(samples_per_source)
        fact_check_texts.extend(halueval_questions)
        source_counts["HaluEval"] = len(halueval_questions)

        # RAG Dataset - questions designed for retrieval-augmented generation
        # These questions explicitly require external knowledge
        rag_questions = self._load_rag_dataset_questions(samples_per_source)
        fact_check_texts.extend(rag_questions)
        source_counts["RAG"] = len(rag_questions)

        # FaithDial - information-seeking dialogue questions (TACL 2022)
        # Used by HuDEx for hallucination detection training
        faithdial_questions = self._load_faithdial_questions(samples_per_source)
        fact_check_texts.extend(faithdial_questions)
        source_counts["FaithDial"] = len(faithdial_questions)

        # FactCHD - fact-conflicting hallucination detection queries (Chen et al., 2024)
        # Used by HuDEx - contains 51k+ factual queries
        factchd_questions = self._load_factchd_questions(samples_per_source)
        fact_check_texts.extend(factchd_questions)
        source_counts["FactCHD"] = len(factchd_questions)

        # SQuAD - factual reading comprehension questions (Tier 2)
        squad_questions = self._load_squad_questions(samples_per_source)
        fact_check_texts.extend(squad_questions)
        source_counts["SQuAD"] = len(squad_questions)

        # TriviaQA - factual trivia questions (Tier 2)
        trivia_questions = self._load_trivia_qa_questions(samples_per_source)
        fact_check_texts.extend(trivia_questions)
        source_counts["TriviaQA"] = len(trivia_questions)

        # TruthfulQA - high-risk factual queries (Tier 6 - critical for safety)
        truthful_questions = self._load_truthful_qa_questions(samples_per_source)
        fact_check_texts.extend(truthful_questions)
        source_counts["TruthfulQA"] = len(truthful_questions)

        # HotpotQA - multi-hop factual questions
        hotpot_questions = self._load_hotpot_qa_questions(samples_per_source)
        fact_check_texts.extend(hotpot_questions)
        source_counts["HotpotQA"] = len(hotpot_questions)

        # CoQA - conversational factual questions (Tier 4)
        if len(fact_check_texts) < samples_per_class:
            coqa_questions = self._load_coqa_questions(
                samples_per_class - len(fact_check_texts)
            )
            fact_check_texts.extend(coqa_questions)
            source_counts["CoQA"] = len(coqa_questions)

        # Fallback to templates if still not enough
        if len(fact_check_texts) < samples_per_class:
            logger.warning(
                f"Using fallback templates for FACT_CHECK_NEEDED ({samples_per_class - len(fact_check_texts)} samples)"
            )
            fallback = self._generate_fallback_samples(
                self._fallback_fact_check, samples_per_class - len(fact_check_texts)
            )
            fact_check_texts.extend(fallback)
            source_counts["Fallback_Factual"] = len(fallback)

        # Truncate to target
        fact_check_texts = fact_check_texts[:samples_per_class]

        # Load NO_FACT_CHECK_NEEDED samples
        no_fact_check_texts = []

        # First add NISQ non-ISQ samples (gold standard negative examples)
        no_fact_check_texts.extend(no_fact_check_texts_from_nisq)

        # Calculate how much we need from each non-factual source
        remaining_needed = samples_per_class - len(no_fact_check_texts)
        samples_per_nonfact_source = remaining_needed // 3  # Split between 3 sources

        # Dolly - creative_writing, brainstorming, opinion (helps with edge cases!)
        # This specifically helps with "What is love?", "meaning of life", opinion questions
        dolly_nonfact = self._load_dolly_nonfactual(samples_per_nonfact_source)
        no_fact_check_texts.extend(dolly_nonfact)
        source_counts["Dolly"] = len(dolly_nonfact)

        # WritingPrompts - creative writing
        writing_prompts = self._load_writing_prompts(samples_per_nonfact_source)
        no_fact_check_texts.extend(writing_prompts)
        source_counts["WritingPrompts"] = len(writing_prompts)

        # Alpaca - non-factual instructions (coding, creative, math, opinion)
        remaining_after_wp = samples_per_class - len(no_fact_check_texts)
        if remaining_after_wp > 0:
            alpaca_nonfact = self._load_alpaca_nonfactual(remaining_after_wp)
            no_fact_check_texts.extend(alpaca_nonfact)
            source_counts["Alpaca"] = len(alpaca_nonfact)
        else:
            source_counts["Alpaca"] = 0

        # Truncate to target
        no_fact_check_texts = no_fact_check_texts[:samples_per_class]

        # Combine and create labels
        texts = []
        labels = []

        for text in fact_check_texts:
            texts.append(text)
            labels.append(self.label2id[FACT_CHECK_NEEDED])

        for text in no_fact_check_texts:
            texts.append(text)
            labels.append(self.label2id[NO_FACT_CHECK_NEEDED])

        # Shuffle
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        texts, labels = list(texts), list(labels)

        # Log statistics
        fact_check_count = sum(1 for l in labels if l == 1)
        no_fact_check_count = sum(1 for l in labels if l == 0)
        logger.info(f"Loaded dataset: {len(texts)} total samples")
        logger.info(f"  FACT_CHECK_NEEDED: {fact_check_count}")
        logger.info(f"  NO_FACT_CHECK_NEEDED: {no_fact_check_count}")
        sources_str = ", ".join([f"{k}={v}" for k, v in source_counts.items()])
        logger.info(f"  Sources: {sources_str}")

        return texts, labels

    def prepare_datasets(self, max_samples: int = 2000):
        """Prepare train/validation/test datasets."""
        texts, labels = self.load_datasets(max_samples)

        # Split the data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )

        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels,
        )

        logger.info(f"Dataset sizes:")
        logger.info(f"  Train: {len(train_texts)}")
        logger.info(f"  Validation: {len(val_texts)}")
        logger.info(f"  Test: {len(test_texts)}")

        return {
            "train": (train_texts, train_labels),
            "validation": (val_texts, val_labels),
            "test": (test_texts, test_labels),
        }


def create_fact_check_dataset(max_samples: int = 2000, data_dir: str = None):
    """Create fact-check dataset.

    Args:
        max_samples: Maximum number of samples to load
        data_dir: Optional path to cached datasets (run setup_datasets.sh to populate)
    """
    dataset_loader = FactCheckDataset(data_dir=data_dir)
    datasets = dataset_loader.prepare_datasets(max_samples)

    train_texts, train_labels = datasets["train"]
    val_texts, val_labels = datasets["validation"]

    # Convert to the format expected by our training
    sample_data = []
    for text, label in zip(train_texts + val_texts, train_labels + val_labels):
        sample_data.append({"text": text, "label": label})

    logger.info(f"Created dataset with {len(sample_data)} samples")
    logger.info(f"Label mapping: {dataset_loader.label2id}")

    return sample_data, dataset_loader.label2id, dataset_loader.id2label


class FactCheckLoRATrainer(Trainer):
    """Enhanced Trainer for fact-check detection with LoRA."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute fact-check classification loss."""
        labels = inputs.get("labels")
        outputs = model(**inputs)

        # Binary classification loss
        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(
                outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
        else:
            loss = None

        return (loss, outputs) if return_outputs else loss


def create_lora_fact_check_model(model_name: str, num_labels: int, lora_config: dict):
    """Create LoRA-enhanced fact-check classification model."""
    logger.info(
        f"Creating LoRA fact-check classification model with base: {model_name}"
    )

    # Load tokenizer with model-specific configuration
    tokenizer = create_tokenizer_for_model(model_name, model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for binary classification
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32,  # Use FP32 for stable training
    )

    # Create LoRA configuration for sequence classification
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
    )

    # Apply LoRA to the model
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()

    return lora_model, tokenizer


def tokenize_fact_check_data(data, tokenizer, max_length=256):
    """Tokenize fact-check detection data."""
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )


def compute_fact_check_metrics(eval_pred):
    """Compute fact-check detection metrics."""
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main(
    model_name: str = "bert-base-uncased",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    max_samples: int = 2000,
    output_dir: str = None,
    data_dir: str = None,
):
    """Main training function for LoRA fact-check classification.

    Args:
        data_dir: Optional path to cached datasets (run setup_datasets.sh to populate)
    """
    logger.info("Starting Enhanced LoRA Fact-Check Classification Training")

    # Device configuration and memory management
    device, _ = set_gpu_device(gpu_id=None, auto_select=True)
    clear_gpu_memory()
    log_memory_usage("Pre-training")

    # Get actual model path
    model_path = resolve_model_path(model_name)
    logger.info(f"Using model: {model_name} -> {model_path}")

    # Create LoRA configuration with dynamic target_modules
    try:
        lora_config = create_lora_config(
            model_name, lora_rank, lora_alpha, lora_dropout
        )
    except Exception as e:
        logger.error(f"Failed to create LoRA config: {e}")
        raise

    # Create dataset
    sample_data, label_to_id, id_to_label = create_fact_check_dataset(
        max_samples, data_dir
    )

    # Split data
    train_size = int(0.8 * len(sample_data))
    train_data = sample_data[:train_size]
    val_data = sample_data[train_size:]

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Categories: {len(label_to_id)}")

    # Create LoRA model
    model, tokenizer = create_lora_fact_check_model(
        model_path, len(label_to_id), lora_config
    )

    # Prepare datasets
    train_dataset = tokenize_fact_check_data(train_data, tokenizer)
    val_dataset = tokenize_fact_check_data(val_data, tokenizer)

    # Setup output directory
    if output_dir is None:
        output_dir = f"lora_fact_check_classifier_{model_name}_r{lora_rank}_model"
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to=[],
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=False,
        eval_accumulation_steps=1,
    )

    # Create trainer
    trainer = FactCheckLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_fact_check_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save the LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    label_mapping_data = {
        "label_to_idx": label_to_id,
        "idx_to_label": {str(v): k for k, v in label_to_id.items()},
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping_data, f, indent=2)

    # Save fact_check_mapping.json for Go compatibility
    with open(os.path.join(output_dir, "fact_check_mapping.json"), "w") as f:
        json.dump(label_mapping_data, f, indent=2)
    logger.info("Created fact_check_mapping.json for Go compatibility")

    # Save LoRA config
    with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)

    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"Validation Results:")
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1: {eval_results['eval_f1']:.4f}")
    logger.info(f"  Precision: {eval_results['eval_precision']:.4f}")
    logger.info(f"  Recall: {eval_results['eval_recall']:.4f}")
    logger.info(f"LoRA Fact-Check model saved to: {output_dir}")

    # NOTE: LoRA adapters are kept separate from base model
    # To merge later, use: merge_lora_adapter_to_full_model(output_dir, merged_output_dir, model_path)
    logger.info(f"LoRA adapter saved to: {output_dir}")
    logger.info(f"Base model: {model_path} (not merged - adapters kept separate)")


def merge_lora_adapter_to_full_model(
    lora_adapter_path: str, output_path: str, base_model_path: str
):
    """
    Merge LoRA adapter with base model to create a complete model for Rust inference.
    """
    logger.info(f"Loading base model: {base_model_path}")

    # Load label mapping to get correct number of labels
    with open(os.path.join(lora_adapter_path, "label_mapping.json"), "r") as f:
        mapping_data = json.load(f)

    if "idx_to_label" in mapping_data:
        num_labels = len(mapping_data["idx_to_label"])
    elif "label_to_idx" in mapping_data:
        num_labels = len(mapping_data["label_to_idx"])
    else:
        num_labels = 2

    # Load base model with correct number of labels
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, num_labels=num_labels, torch_dtype=torch.float32
    )

    # Load tokenizer
    tokenizer = create_tokenizer_for_model(base_model_path, base_model_path)

    logger.info(f"Loading LoRA adapter from: {lora_adapter_path}")

    # Load LoRA model
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    logger.info("Merging LoRA adapter with base model...")

    # Merge and unload LoRA
    merged_model = lora_model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save merged model
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Fix config.json to include correct id2label mapping
    config_path = os.path.join(output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        if "idx_to_label" in mapping_data:
            config["id2label"] = mapping_data["idx_to_label"]
        if "label_to_idx" in mapping_data:
            config["label2id"] = mapping_data["label_to_idx"]

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Updated config.json with correct label mappings")

    # Copy important files from LoRA adapter
    for file_name in [
        "label_mapping.json",
        "lora_config.json",
        "fact_check_mapping.json",
    ]:
        src_file = Path(lora_adapter_path) / file_name
        if src_file.exists():
            shutil.copy(src_file, Path(output_path) / file_name)

    logger.info("LoRA adapter merged successfully!")


def demo_inference(
    model_path: str = "lora_fact_check_classifier_bert-base-uncased_r16_model",
):
    """Demonstrate inference with trained LoRA fact-check model."""
    logger.info(f"Loading LoRA fact-check model from: {model_path}")

    try:
        # Load label mapping
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            mapping_data = json.load(f)
        id_to_label = mapping_data.get("idx_to_label", {})
        num_labels = len(id_to_label)

        # Check if this is a LoRA adapter or a merged model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            logger.info("Detected LoRA adapter model, loading with PEFT...")
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                num_labels=num_labels,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = create_tokenizer_for_model(
                model_path, peft_config.base_model_name_or_path
            )
        else:
            logger.info("Detected merged model, loading directly...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels
            )
            tokenizer = create_tokenizer_for_model(model_path)

        # Test examples
        test_examples = [
            # Should be FACT_CHECK_NEEDED
            "When was the Eiffel Tower built?",
            "Who is the CEO of Apple?",
            "What is the population of Tokyo?",
            "Is it true that water boils at 100 degrees Celsius?",
            "What year did World War 2 end?",
            # Should be NO_FACT_CHECK_NEEDED
            "Write a poem about the ocean",
            "Write a Python function to sort a list",
            "What do you think about AI?",
            "Calculate 25 * 4",
            "Help me debug this JavaScript code",
        ]

        logger.info("Running fact-check classification inference...")
        print("\n" + "=" * 70)
        print("FACT-CHECK CLASSIFICATION RESULTS")
        print("=" * 70)

        for example in test_examples:
            inputs = tokenizer(
                example,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                confidence = predictions[0][predicted_class_id].item()

            predicted_label = id_to_label.get(str(predicted_class_id), "UNKNOWN")

            print(f"\nInput: {example}")
            print(f"Prediction: {predicted_label}")
            print(f"Confidence: {confidence:.4f}")
            print("-" * 60)

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced LoRA Fact-Check Classification"
    )
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument(
        "--model",
        choices=[
            "mmbert-base",  # mmBERT - Multilingual ModernBERT (1800+ languages, recommended)
            "modernbert-base",
            "bert-base-uncased",
            "roberta-base",
        ],
        default="mmbert-base",  # Default to mmBERT for multilingual support
        help="Model to use for fine-tuning",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum samples to generate for training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for saving the model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="lora_fact_check_classifier_bert-base-uncased_r16_model",
        help="Path to saved model for inference",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to cached datasets directory (run setup_datasets.sh to populate)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        main(
            model_name=args.model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
        )
    elif args.mode == "test":
        demo_inference(args.model_path)
