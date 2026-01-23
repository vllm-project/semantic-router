"""
Dataset download and labeling utilities for feedback detector training.

Downloads all required public datasets:
- MIMICS / MIMICS-Duo (Microsoft)
- INSCIT (Google Research)
- MultiWOZ
- SGD (Schema-Guided Dialogue)
- ReDial
- Hazumi (via HuggingFace)

Then labels them using GPT-OSS-120B via vLLM for the 4-class feedback schema:
- SAT: User is satisfied
- NEED_CLARIFICATION: User needs more information
- WRONG_ANSWER: System gave incorrect response
- WANT_DIFFERENT: User wants something different
"""

import os
import subprocess
import zipfile
import tarfile
import requests
import json
import glob
import re
import logging
import time
import random
import hashlib
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Generator, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DataConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Valid labels for feedback classification
VALID_LABELS = ["SAT", "NEED_CLARIFICATION", "WRONG_ANSWER", "WANT_DIFFERENT"]

# Default vLLM API settings
DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


@dataclass
class LabelingConfig:
    """Configuration for production labeling."""

    api_url: str = DEFAULT_API_URL
    model: str = DEFAULT_MODEL
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_rps: float = 5.0  # Requests per second
    timeout: int = 60
    max_tokens: int = 300
    checkpoint_interval: int = 500  # Save every N examples
    workers: int = 4

    # Garbage protection settings
    garbage_consecutive_threshold: int = (
        5  # Consecutive garbage to trigger circuit breaker
    )
    garbage_window_size: int = 20  # Sliding window for ratio calculation
    garbage_ratio_threshold: float = 0.5  # Max acceptable garbage ratio
    garbage_cooldown_base: float = 30.0  # Base cooldown in seconds
    garbage_max_cooldown: float = 300.0  # Maximum cooldown in seconds
    garbage_max_total: int = 100  # Max total garbage before abort (0 = no limit)
    skip_health_check: bool = False  # Skip initial health check (not recommended)


class RateLimiter:
    """Thread-safe rate limiter."""

    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = Lock()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()


class GarbageDetector:
    """
    Detects garbage output from LLM and implements circuit breaker pattern.

    Garbage patterns:
    - Repeated characters (e.g., "!!!!!!!!!!")
    - Very low character diversity
    - Empty or near-empty responses
    - Responses that are all punctuation/symbols
    """

    def __init__(
        self,
        consecutive_threshold: int = 5,
        window_size: int = 20,
        garbage_ratio_threshold: float = 0.5,
        cooldown_base: float = 30.0,
        max_cooldown: float = 300.0,
    ):
        """
        Args:
            consecutive_threshold: Consecutive garbage responses to trigger circuit breaker
            window_size: Size of sliding window for ratio calculation
            garbage_ratio_threshold: Ratio of garbage in window to trigger slowdown
            cooldown_base: Base cooldown time in seconds
            max_cooldown: Maximum cooldown time in seconds
        """
        self.consecutive_threshold = consecutive_threshold
        self.window_size = window_size
        self.garbage_ratio_threshold = garbage_ratio_threshold
        self.cooldown_base = cooldown_base
        self.max_cooldown = max_cooldown

        self.consecutive_garbage = 0
        self.recent_results: List[bool] = []  # True = garbage, False = valid
        self.circuit_open = False
        self.cooldown_multiplier = 1
        self.lock = Lock()

    def is_garbage(self, content: str) -> bool:
        """Check if content is garbage output."""
        if not content:
            return True

        # Too short
        if len(content) < 10:
            return True

        # Very low character diversity (e.g., "!!!!!!!!")
        unique_chars = len(set(content))
        if unique_chars < 5:
            return True

        # High ratio of repeated character
        char_counts = {}
        for c in content:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_char_ratio = max(char_counts.values()) / len(content)
        if max_char_ratio > 0.5:  # Single char is >50% of content
            return True

        # Check for known garbage patterns
        garbage_patterns = [
            r'^[!@#$%^&*()_+=\[\]{}|\\:";\'<>?,./\s]+$',  # Only punctuation
            r"analysis[!]+$",  # "analysis!!!!" pattern
            r"^[!]+\s*$",  # Just exclamation marks
            r"(.)\1{10,}",  # Any character repeated 10+ times
        ]

        for pattern in garbage_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def record_result(self, is_garbage: bool):
        """Record a result and update state."""
        with self.lock:
            # Update consecutive counter
            if is_garbage:
                self.consecutive_garbage += 1
            else:
                self.consecutive_garbage = 0
                self.cooldown_multiplier = max(1, self.cooldown_multiplier - 1)

            # Update sliding window
            self.recent_results.append(is_garbage)
            if len(self.recent_results) > self.window_size:
                self.recent_results.pop(0)

            # Check circuit breaker
            if self.consecutive_garbage >= self.consecutive_threshold:
                self.circuit_open = True
                self.cooldown_multiplier = min(self.cooldown_multiplier * 2, 10)
                logger.warning(
                    f"Circuit breaker OPEN: {self.consecutive_garbage} consecutive garbage responses. "
                    f"Cooldown multiplier: {self.cooldown_multiplier}x"
                )

    def get_garbage_ratio(self) -> float:
        """Get ratio of garbage in recent window."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    def should_pause(self) -> tuple:
        """
        Check if processing should pause.

        Returns:
            (should_pause: bool, pause_duration: float)
        """
        with self.lock:
            # Check sliding window ratio
            ratio = self.get_garbage_ratio()

            if self.circuit_open:
                pause_time = min(
                    self.cooldown_base * self.cooldown_multiplier, self.max_cooldown
                )
                self.circuit_open = False  # Will reopen if garbage continues
                return True, pause_time

            if ratio > self.garbage_ratio_threshold and len(self.recent_results) >= 5:
                # High garbage ratio - apply shorter pause
                pause_time = self.cooldown_base * (ratio / self.garbage_ratio_threshold)
                return True, min(pause_time, self.cooldown_base * 2)

            return False, 0.0

    def reset(self):
        """Reset all counters."""
        with self.lock:
            self.consecutive_garbage = 0
            self.recent_results = []
            self.circuit_open = False
            self.cooldown_multiplier = 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self.lock:
            return {
                "consecutive_garbage": self.consecutive_garbage,
                "recent_garbage_ratio": self.get_garbage_ratio(),
                "circuit_open": self.circuit_open,
                "cooldown_multiplier": self.cooldown_multiplier,
                "window_size": len(self.recent_results),
            }


class CheckpointManager:
    """Manages checkpointing for resumable labeling."""

    def __init__(self, checkpoint_dir: str, dataset_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.dataset_name = dataset_name
        self.checkpoint_file = os.path.join(
            checkpoint_dir, f"{dataset_name}_checkpoint.json"
        )
        self.labeled_file = os.path.join(
            checkpoint_dir, f"{dataset_name}_labeled.jsonl"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.processed_hashes: Set[str] = set()
        self.stats = {label: 0 for label in VALID_LABELS}
        self.stats["errors"] = 0
        self.stats["total"] = 0
        self._load_checkpoint()

    def _hash_text(self, text: str) -> str:
        """Create hash of text for deduplication."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _load_checkpoint(self):
        """Load existing checkpoint if available."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    self.processed_hashes = set(data.get("processed_hashes", []))
                    self.stats = data.get("stats", self.stats)
                logger.info(
                    f"Loaded checkpoint: {len(self.processed_hashes)} already processed"
                )
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")

    def save_checkpoint(self):
        """Save current progress."""
        with open(self.checkpoint_file, "w") as f:
            json.dump(
                {"processed_hashes": list(self.processed_hashes), "stats": self.stats},
                f,
            )

    def is_processed(self, text: str) -> bool:
        """Check if text has already been processed."""
        return self._hash_text(text) in self.processed_hashes

    def mark_processed(self, text: str, result: Optional[Dict] = None):
        """Mark text as processed and optionally save result."""
        text_hash = self._hash_text(text)
        self.processed_hashes.add(text_hash)
        self.stats["total"] += 1

        if result and "label_name" in result:
            self.stats[result["label_name"]] += 1
            # Append to labeled file
            with open(self.labeled_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            self.stats["errors"] += 1


def download_file(url: str, dest_path: str, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="iB", unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def clone_repo(repo_url: str, dest_dir: str, desc: str = "Repository") -> bool:
    """Clone a git repository."""
    if os.path.exists(dest_dir):
        print(f"{desc} already exists at {dest_dir}, skipping clone...")
        return True

    try:
        print(f"Cloning {desc} from {repo_url}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, dest_dir],
            check=True,
            capture_output=True,
        )
        print(f"Successfully cloned {desc}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error cloning {repo_url}: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Git not found. Please install git.")
        return False


def extract_archive(archive_path: str, dest_dir: str) -> bool:
    """Extract zip or tar archive."""
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        elif archive_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest_dir)
        elif archive_path.endswith(".tar"):
            with tarfile.open(archive_path, "r") as tf:
                tf.extractall(dest_dir)
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False


def download_mimics(raw_data_dir: str) -> bool:
    """
    Download MIMICS and MIMICS-Duo datasets.

    MIMICS: A Large-Scale Data Collection for Search Clarification
    https://github.com/microsoft/MIMICS
    """
    dest_dir = os.path.join(raw_data_dir, "mimics")
    return clone_repo("https://github.com/microsoft/MIMICS.git", dest_dir, "MIMICS")


def download_inscit(raw_data_dir: str) -> bool:
    """
    Download INSCIT dataset.

    Information-Seeking Conversations with Mixed-Initiative Interactions
    https://github.com/ellenmellon/INSCIT
    """
    dest_dir = os.path.join(raw_data_dir, "inscit")
    return clone_repo("https://github.com/ellenmellon/INSCIT.git", dest_dir, "INSCIT")


def download_multiwoz(raw_data_dir: str) -> bool:
    """
    Download MultiWOZ 2.2 dataset.

    Multi-Domain Wizard-of-Oz dataset
    """
    dest_dir = os.path.join(raw_data_dir, "multiwoz")
    os.makedirs(dest_dir, exist_ok=True)

    # Clone the repository which contains the data
    return clone_repo(
        "https://github.com/budzianowski/multiwoz.git", dest_dir, "MultiWOZ"
    )


def download_sgd(raw_data_dir: str) -> bool:
    """
    Download Schema-Guided Dialogue (SGD) dataset.

    DSTC8 Schema-Guided Dialogue
    https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
    """
    dest_dir = os.path.join(raw_data_dir, "sgd")
    return clone_repo(
        "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git",
        dest_dir,
        "SGD",
    )


def download_redial(raw_data_dir: str) -> bool:
    """
    Download ReDial dataset.

    Recommendation Dialogues dataset
    https://github.com/ReDialData/website
    """
    dest_dir = os.path.join(raw_data_dir, "redial")

    # Clone the website repo which contains download links
    success = clone_repo(
        "https://github.com/ReDialData/website.git", dest_dir, "ReDial"
    )

    if success:
        # The actual data files are hosted separately
        # Download from the Zenodo/Google Drive links in the repo
        data_dir = os.path.join(dest_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # ReDial data URL (from the official source)
        redial_data_url = "https://raw.githubusercontent.com/ReDialData/website/main/data/train_data.jsonl"
        train_path = os.path.join(data_dir, "train_data.jsonl")

        if not os.path.exists(train_path):
            print("Note: ReDial requires manual download from their website.")
            print("Please visit: https://redialdata.github.io/website/")
            print("Download train_data.jsonl and test_data.jsonl to:", data_dir)

    return success


def download_hazumi(raw_data_dir: str) -> bool:
    """
    Download Hazumi Affirmative/Negative dataset from HuggingFace.

    https://huggingface.co/datasets/ouktlab/Hazumi-AffNeg-Data
    """
    try:
        from datasets import load_dataset

        dest_dir = os.path.join(raw_data_dir, "hazumi")
        os.makedirs(dest_dir, exist_ok=True)

        print("Downloading Hazumi dataset from HuggingFace...")
        dataset = load_dataset("ouktlab/Hazumi-AffNeg-Data")

        # Save to disk
        dataset.save_to_disk(dest_dir)
        print(f"Hazumi dataset saved to {dest_dir}")
        return True

    except Exception as e:
        print(f"Error downloading Hazumi dataset: {e}")
        print("Try: pip install datasets")
        return False


def download_customer_complaints(raw_data_dir: str) -> bool:
    """
    Download customer complaints dataset from HuggingFace.

    Good source for WRONG_ANSWER examples - contains actual complaints
    about incorrect information, wrong billing, etc.

    https://huggingface.co/datasets/determined-ai/customers-complaints
    """
    try:
        from datasets import load_dataset

        dest_dir = os.path.join(raw_data_dir, "customer_complaints")
        os.makedirs(dest_dir, exist_ok=True)

        print("Downloading customer complaints dataset from HuggingFace...")
        dataset = load_dataset("determined-ai/customers-complaints")

        # Save to disk
        dataset.save_to_disk(dest_dir)
        print(f"Customer complaints dataset saved to {dest_dir}")
        return True

    except Exception as e:
        print(f"Error downloading customer complaints dataset: {e}")
        print("Try: pip install datasets")
        return False


def download_consumer_complaints_medium(raw_data_dir: str) -> bool:
    """
    Download medium-sized consumer complaints dataset from HuggingFace.

    ~85K examples with Issue categories - good for WRONG_ANSWER detection.
    Categories include billing disputes, incorrect information, etc.

    https://huggingface.co/datasets/determined-ai/consumer_complaints_medium
    """
    try:
        from datasets import load_dataset

        dest_dir = os.path.join(raw_data_dir, "consumer_complaints_medium")
        os.makedirs(dest_dir, exist_ok=True)

        print("Downloading consumer complaints medium dataset from HuggingFace...")
        dataset = load_dataset("determined-ai/consumer_complaints_medium")

        # Save to disk
        dataset.save_to_disk(dest_dir)
        print(f"Consumer complaints medium dataset saved to {dest_dir}")
        return True

    except Exception as e:
        print(f"Error downloading consumer complaints medium dataset: {e}")
        print("Try: pip install datasets")
        return False


def download_turkish_complaints(raw_data_dir: str) -> bool:
    """
    Download Turkish complaint classification dataset from HuggingFace.

    ~7K Turkish complaints - adds multilingual diversity.

    https://huggingface.co/datasets/nanelimon/complaint-classification-dataset
    """
    try:
        from datasets import load_dataset

        dest_dir = os.path.join(raw_data_dir, "turkish_complaints")
        os.makedirs(dest_dir, exist_ok=True)

        print("Downloading Turkish complaints dataset from HuggingFace...")
        dataset = load_dataset("nanelimon/complaint-classification-dataset")

        # Save to disk
        dataset.save_to_disk(dest_dir)
        print(f"Turkish complaints dataset saved to {dest_dir}")
        return True

    except Exception as e:
        print(f"Error downloading Turkish complaints dataset: {e}")
        print("Try: pip install datasets")
        return False


def download_all_datasets(config: Optional[DataConfig] = None) -> dict:
    """
    Download all required datasets.

    Returns:
        dict: Status of each download (True/False)
    """
    if config is None:
        config = DataConfig()

    raw_data_dir = config.raw_data_dir
    os.makedirs(raw_data_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading all datasets for feedback detector")
    print("=" * 60)

    results = {}

    # Download each dataset
    print("\n[1/9] Downloading MIMICS / MIMICS-Duo...")
    results["mimics"] = download_mimics(raw_data_dir)

    print("\n[2/9] Downloading INSCIT...")
    results["inscit"] = download_inscit(raw_data_dir)

    print("\n[3/9] Downloading MultiWOZ...")
    results["multiwoz"] = download_multiwoz(raw_data_dir)

    print("\n[4/9] Downloading SGD...")
    results["sgd"] = download_sgd(raw_data_dir)

    print("\n[5/9] Downloading ReDial...")
    results["redial"] = download_redial(raw_data_dir)

    print("\n[6/9] Downloading Hazumi...")
    results["hazumi"] = download_hazumi(raw_data_dir)

    # New complaint datasets for WRONG_ANSWER class
    print("\n[7/9] Downloading Customer Complaints (for WRONG_ANSWER)...")
    results["customer_complaints"] = download_customer_complaints(raw_data_dir)

    print("\n[8/9] Downloading Consumer Complaints Medium (for WRONG_ANSWER)...")
    results["consumer_complaints_medium"] = download_consumer_complaints_medium(
        raw_data_dir
    )

    print("\n[9/9] Downloading Turkish Complaints (multilingual)...")
    results["turkish_complaints"] = download_turkish_complaints(raw_data_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary:")
    print("=" * 60)
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name}: {status}")

    print("\n📊 Dataset categories:")
    print("  - Dialogue: mimics, inscit, multiwoz, sgd, redial, hazumi")
    print(
        "  - Complaints (WRONG_ANSWER): customer_complaints, consumer_complaints_medium"
    )
    print("  - Multilingual: hazumi (Japanese), turkish_complaints (Turkish)")

    return results


# =============================================================================
# LLM Labeling Functions
# =============================================================================


def check_model_health(
    api_url: str = DEFAULT_API_URL, model: str = DEFAULT_MODEL, timeout: int = 30
) -> tuple:
    """
    Check if the model is healthy and returning valid responses.

    Returns:
        (is_healthy: bool, message: str)
    """
    test_prompts = [
        ("Say 'OK'", lambda r: "ok" in r.lower()),
        ("What is 2+2? Answer with just the number.", lambda r: "4" in r),
        (
            'Classify "Thanks!" as POSITIVE or NEGATIVE. JSON: {"label": "..."}',
            lambda r: "positive" in r.lower() or '"label"' in r.lower(),
        ),
    ]

    for prompt, validator in test_prompts:
        try:
            response = requests.post(
                api_url,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 50,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            # Check for garbage
            if len(set(content)) < 5:
                return False, f"Garbage output detected: {repr(content[:50])}"

            # Check if response is sensible
            if not validator(content):
                logger.warning(f"Unexpected response to health check: {content[:100]}")
                continue

            return True, "Model is healthy"

        except requests.exceptions.Timeout:
            return False, "Model timeout - server may be overloaded"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to model server"
        except Exception as e:
            return False, f"Health check failed: {e}"

    return False, "Model returning unexpected responses"


def extract_label_from_response(content: str) -> Optional[str]:
    """
    Extract label from LLM response, handling various output formats.

    GPT-OSS outputs: analysis...assistantfinal{JSON}
    """
    if not content:
        return None

    # Method 1: Find assistantfinal{JSON} pattern (most reliable)
    final_match = re.search(r"assistantfinal\s*(\{[^{}]*\})", content, re.IGNORECASE)
    if final_match:
        try:
            data = json.loads(final_match.group(1))
            label = str(data.get("label", "")).upper().strip()
            if label in VALID_LABELS:
                return label
        except (json.JSONDecodeError, AttributeError):
            pass

    # Method 2: Find any JSON with label field
    json_matches = re.findall(r'\{\s*"label"\s*:\s*"([^"]+)"\s*\}', content)
    for match in json_matches:
        label = match.upper().strip()
        if label in VALID_LABELS:
            return label

    # Method 3: Find "label": "VALUE" pattern
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
    if label_match:
        label = label_match.group(1).upper().strip()
        if label in VALID_LABELS:
            return label

    # Method 4: Find label after "final" keyword (near end of response)
    final_idx = content.lower().rfind("final")
    if final_idx > 0:
        end_part = content[final_idx:].upper()
        for label in VALID_LABELS:
            if label in end_part:
                return label

    # Method 5: Last resort - look for any valid label in the content
    # But prioritize labels that appear in quotes or after colons
    content_upper = content.upper()
    for label in VALID_LABELS:
        patterns = [f'"{label}"', f"'{label}'", f": {label}", f":{label}"]
        for pattern in patterns:
            if pattern in content_upper:
                return label

    return None


def is_garbage_output(content: str) -> bool:
    """
    Quick check if content is garbage output.

    Garbage patterns:
    - Very low character diversity
    - Repeated characters
    - Only punctuation/symbols
    """
    if not content or len(content) < 10:
        return True

    # Very low character diversity
    unique_chars = len(set(content))
    if unique_chars < 5:
        return True

    # High ratio of repeated character
    char_counts = {}
    for c in content:
        char_counts[c] = char_counts.get(c, 0) + 1
    max_char_ratio = max(char_counts.values()) / len(content)
    if max_char_ratio > 0.5:
        return True

    # Check for repeated pattern like "analysis!!!!!"
    if re.search(r"(.)\1{10,}", content):
        return True

    return False


def call_llm_api(
    text: str,
    config: Optional[LabelingConfig] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> Optional[Dict[str, Any]]:
    """
    Call the LLM API to classify a text into one of the 4 feedback categories.
    Includes retry logic and rate limiting for production use.

    Returns:
        Dict with 'label' if successful, None otherwise.
    """
    if config is None:
        config = LabelingConfig()

    # Truncate very long text
    text_truncated = text[:500] if len(text) > 500 else text

    # Use a cleaner, more direct prompt
    prompt = f"""Classify this text into ONE category and output JSON only.

Categories:
- SAT: Satisfied, happy, thankful
- NEED_CLARIFICATION: Asking questions, needs more info
- WRONG_ANSWER: Complaint about incorrect info
- WANT_DIFFERENT: Wants alternative, different option

Text: "{text_truncated}"

JSON output:"""

    garbage_count = 0
    for attempt in range(config.max_retries):
        try:
            # Rate limiting
            if rate_limiter:
                rate_limiter.wait()

            response = requests.post(
                config.api_url,
                json={
                    "model": config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": 'You classify user feedback. Output only {"label": "CATEGORY"}.',
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": config.max_tokens,
                },
                timeout=config.timeout,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Check for garbage output
            if is_garbage_output(content):
                garbage_count += 1
                logger.debug(
                    f"Garbage output detected (attempt {attempt + 1}): {repr(content[:50])}"
                )

                # If we're getting consistent garbage, back off more aggressively
                if garbage_count >= 2:
                    backoff = config.retry_delay * (4**attempt) + random.uniform(1, 5)
                    logger.debug(
                        f"Multiple garbage responses, backing off {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                else:
                    time.sleep(config.retry_delay * (attempt + 1))
                continue

            # Extract label using robust extraction
            label = extract_label_from_response(content)
            if label:
                return {"label": label}

            logger.debug(f"Could not extract label from: {content[:100]}")

        except requests.exceptions.Timeout:
            logger.debug(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.debug(f"Error on attempt {attempt + 1}: {e}")

        # Exponential backoff with jitter
        if attempt < config.max_retries - 1:
            delay = config.retry_delay * (2**attempt) + random.uniform(0, 1)
            time.sleep(delay)

    return None


def call_llm_api_simple(
    text: str,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """Simple version of call_llm_api for backward compatibility."""
    config = LabelingConfig(
        api_url=api_url,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=1,
    )
    return call_llm_api(text, config)


def extract_mimics_examples(raw_data_dir: str) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from MIMICS dataset.

    MIMICS contains search clarification data - primarily NEED_CLARIFICATION examples.
    """
    mimics_dir = os.path.join(raw_data_dir, "mimics")

    # Find TSV/CSV files
    for pattern in ["*.tsv", "*.csv", "**/*.tsv", "**/*.csv"]:
        for filepath in glob.glob(os.path.join(mimics_dir, pattern), recursive=True):
            try:
                import csv

                delimiter = "\t" if filepath.endswith(".tsv") else ","

                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for row in reader:
                        # MIMICS has 'query' and 'clarification_need' columns
                        text = (
                            row.get("query", "")
                            or row.get("question", "")
                            or row.get("text", "")
                        )
                        if text and len(text) > 10:
                            yield {
                                "text": text,
                                "source": "mimics",
                                "file": os.path.basename(filepath),
                            }
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")


def extract_inscit_examples(raw_data_dir: str) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from INSCIT dataset.

    INSCIT contains information-seeking conversations with turns.
    Format: {dialogue_id: {seedArticle: {...}, turns: [{context: [...], response: ...}, ...]}}
    """
    inscit_dir = os.path.join(raw_data_dir, "inscit")

    # Look specifically in data/ folder
    for pattern in ["data/*.json", "**/*.json"]:
        for filepath in glob.glob(os.path.join(inscit_dir, pattern), recursive=True):
            if "eval" in filepath or "sample" in filepath:
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # INSCIT format: dict of dialogue_id -> dialogue
                    if isinstance(data, dict):
                        for dial_id, dialogue in data.items():
                            if isinstance(dialogue, dict) and "turns" in dialogue:
                                for turn in dialogue["turns"]:
                                    # Extract user questions from context
                                    context = turn.get("context", [])
                                    for ctx in context:
                                        if isinstance(ctx, str) and len(ctx) > 10:
                                            yield {
                                                "text": ctx,
                                                "source": "inscit",
                                                "file": os.path.basename(filepath),
                                                "dialogue_id": dial_id,
                                                "type": "question",
                                            }

                                    # Extract labels/responses
                                    labels = turn.get("labels", [])
                                    for label in labels:
                                        if isinstance(label, dict):
                                            response = label.get("response", "")
                                            if response and len(response) > 10:
                                                yield {
                                                    "text": response,
                                                    "source": "inscit",
                                                    "file": os.path.basename(filepath),
                                                    "dialogue_id": dial_id,
                                                    "type": "response",
                                                }
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")


def extract_multiwoz_examples(
    raw_data_dir: str,
) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from MultiWOZ dataset.

    MultiWOZ 2.2 format: list of dialogues with turns
    """
    multiwoz_dir = os.path.join(raw_data_dir, "multiwoz")

    # Look for dialogues files (MultiWOZ 2.2 format)
    for pattern in ["**/dialogues_*.json", "**/data.json"]:
        for filepath in glob.glob(os.path.join(multiwoz_dir, pattern), recursive=True):
            if "schema" in filepath.lower() or "db" in filepath:
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # MultiWOZ 2.2 format: list of dialogues
                    if isinstance(data, list):
                        for dialogue in data:
                            dial_id = dialogue.get("dialogue_id", "")
                            turns = dialogue.get("turns", [])
                            for turn in turns:
                                text = turn.get("utterance", "") or turn.get("text", "")
                                if text and len(text) > 10:
                                    yield {
                                        "text": text,
                                        "source": "multiwoz",
                                        "file": os.path.basename(filepath),
                                        "dialogue_id": dial_id,
                                        "speaker": turn.get("speaker", "unknown"),
                                    }
                    # MultiWOZ 2.1 format: dict of dialogue_id -> dialogue
                    elif isinstance(data, dict):
                        for dial_id, dialogue in data.items():
                            if isinstance(dialogue, dict) and "log" in dialogue:
                                for turn in dialogue["log"]:
                                    text = turn.get("text", "")
                                    if text and len(text) > 10:
                                        yield {
                                            "text": text,
                                            "source": "multiwoz",
                                            "file": os.path.basename(filepath),
                                            "dialogue_id": dial_id,
                                        }
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")


def extract_sgd_examples(raw_data_dir: str) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from SGD (Schema-Guided Dialogue) dataset.
    """
    sgd_dir = os.path.join(raw_data_dir, "sgd")

    for pattern in ["**/dialogues_*.json", "**/*.json"]:
        for filepath in glob.glob(os.path.join(sgd_dir, pattern), recursive=True):
            if "schema" in filepath.lower():
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        for dialogue in data:
                            if "turns" in dialogue:
                                for turn in dialogue["turns"]:
                                    text = turn.get("utterance", "")
                                    if text and len(text) > 10:
                                        yield {
                                            "text": text,
                                            "source": "sgd",
                                            "file": os.path.basename(filepath),
                                            "speaker": turn.get("speaker", "unknown"),
                                        }
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")


def extract_redial_examples(raw_data_dir: str) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from ReDial dataset.
    """
    redial_dir = os.path.join(raw_data_dir, "redial")

    for pattern in ["**/*.jsonl", "**/*.json"]:
        for filepath in glob.glob(os.path.join(redial_dir, pattern), recursive=True):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    if filepath.endswith(".jsonl"):
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                yield from _extract_dialogue_turns(
                                    data, "redial", filepath
                                )
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                yield from _extract_dialogue_turns(
                                    item, "redial", filepath
                                )
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")


def extract_hazumi_examples(raw_data_dir: str) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from Hazumi dataset (Japanese).

    Format: sentence1 (question), sentence2 (response), label (pos/neg)
    """
    hazumi_dir = os.path.join(raw_data_dir, "hazumi")

    try:
        from datasets import load_from_disk

        dataset = load_from_disk(hazumi_dir)

        for split_name in dataset.keys():
            for example in dataset[split_name]:
                # Hazumi has sentence pairs - sentence2 is the response
                sentence1 = example.get("sentence1", "")
                sentence2 = example.get("sentence2", "")
                original_label = example.get("label", "")  # pos/neg

                # Combine both sentences for context
                if sentence2 and len(sentence2) > 3:
                    combined = f"{sentence1} -> {sentence2}" if sentence1 else sentence2
                    yield {
                        "text": combined,
                        "source": "hazumi",
                        "split": split_name,
                        "language": "ja",
                        "original_label": original_label,
                        "sentence1": sentence1,
                        "sentence2": sentence2,
                    }

    except Exception as e:
        logger.warning(f"Error loading Hazumi dataset: {e}")


def extract_customer_complaints_examples(
    raw_data_dir: str,
) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from customer complaints dataset.

    These are actual consumer complaints - excellent source for WRONG_ANSWER examples.
    Complaints often contain phrases like "incorrect", "wrong", "not what I asked for", etc.
    """
    complaints_dir = os.path.join(raw_data_dir, "customer_complaints")

    try:
        from datasets import load_from_disk

        dataset = load_from_disk(complaints_dir)

        for split_name in dataset.keys():
            for example in dataset[split_name]:
                # Field name is 'Consumer_complaint_narrative' with underscores
                text = (
                    example.get("Consumer_complaint_narrative", "")
                    or example.get("Consumer complaint narrative", "")
                    or example.get("complaint", "")
                    or example.get("text", "")
                    or example.get("narrative", "")
                )

                if text and len(text) > 20:
                    # Extract product/sub-product for metadata
                    product = example.get("Product", "") or example.get("product", "")
                    sub_product = example.get("Sub_product", "") or example.get(
                        "sub_product", ""
                    )

                    yield {
                        "text": text,
                        "source": "customer_complaints",
                        "split": split_name,
                        "product": product,
                        "sub_product": sub_product,
                        "likely_label": "WRONG_ANSWER",  # Hint for labeling
                    }

    except Exception as e:
        logger.warning(f"Error loading customer complaints dataset: {e}")


def extract_consumer_complaints_medium_examples(
    raw_data_dir: str,
) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from consumer complaints medium dataset.

    ~85K examples with Issue categories like:
    - Incorrect information on your report
    - Billing disputes
    - Problem with a credit reporting company

    Excellent for WRONG_ANSWER class.
    """
    complaints_dir = os.path.join(raw_data_dir, "consumer_complaints_medium")

    try:
        from datasets import load_from_disk

        dataset = load_from_disk(complaints_dir)

        for split_name in dataset.keys():
            for example in dataset[split_name]:
                # The medium dataset has 'Consumer Complaint' and 'Issue' fields
                text = (
                    example.get("Consumer Complaint", "")
                    or example.get("consumer_complaint", "")
                    or example.get("text", "")
                )

                if text and len(text) > 20:
                    issue = example.get("Issue", "") or example.get("issue", "")

                    # Map some issues to likely labels
                    likely_label = "WRONG_ANSWER"
                    issue_lower = issue.lower() if issue else ""
                    if "incorrect" in issue_lower or "wrong" in issue_lower:
                        likely_label = "WRONG_ANSWER"
                    elif "information" in issue_lower and "need" in issue_lower:
                        likely_label = "NEED_CLARIFICATION"

                    yield {
                        "text": text,
                        "source": "consumer_complaints_medium",
                        "split": split_name,
                        "issue": issue,
                        "likely_label": likely_label,
                    }

    except Exception as e:
        logger.warning(f"Error loading consumer complaints medium dataset: {e}")


def extract_turkish_complaints_examples(
    raw_data_dir: str,
) -> Generator[Dict[str, str], None, None]:
    """
    Extract examples from Turkish complaints dataset.

    Adds multilingual diversity - Turkish language complaints.
    """
    complaints_dir = os.path.join(raw_data_dir, "turkish_complaints")

    try:
        from datasets import load_from_disk

        dataset = load_from_disk(complaints_dir)

        for split_name in dataset.keys():
            for example in dataset[split_name]:
                # Turkish dataset has 'Text' and 'Label' fields (capitalized)
                text = (
                    example.get("Text", "")
                    or example.get("text", "")
                    or example.get("complaint", "")
                )
                category = (
                    example.get("Label", "")
                    or example.get("label", "")
                    or example.get("category", "")
                )

                if text and len(text) > 10:
                    yield {
                        "text": text,
                        "source": "turkish_complaints",
                        "split": split_name,
                        "language": "tr",
                        "category": str(category),
                        "likely_label": "WRONG_ANSWER",  # Complaints are likely WRONG_ANSWER
                    }

    except Exception as e:
        logger.warning(f"Error loading Turkish complaints dataset: {e}")


def _extract_dialogue_turns(
    data: Dict, source: str, filepath: str
) -> Generator[Dict[str, str], None, None]:
    """
    Generic dialogue turn extractor.
    """
    # Common dialogue formats
    turns_keys = ["turns", "messages", "dialogue", "conversation", "utterances"]
    text_keys = ["text", "utterance", "content", "message", "turn"]

    for turns_key in turns_keys:
        if turns_key in data and isinstance(data[turns_key], list):
            for turn in data[turns_key]:
                if isinstance(turn, str):
                    if len(turn) > 10:
                        yield {
                            "text": turn,
                            "source": source,
                            "file": os.path.basename(filepath),
                        }
                elif isinstance(turn, dict):
                    for text_key in text_keys:
                        if text_key in turn and turn[text_key]:
                            text = turn[text_key]
                            if len(text) > 10:
                                yield {
                                    "text": text,
                                    "source": source,
                                    "file": os.path.basename(filepath),
                                }
                            break
            return

    # Single text field
    for text_key in text_keys:
        if text_key in data and data[text_key] and len(str(data[text_key])) > 10:
            yield {
                "text": str(data[text_key]),
                "source": source,
                "file": os.path.basename(filepath),
            }
            return


def label_dataset(
    examples: List[Dict[str, str]],
    output_path: str,
    config: Optional[LabelingConfig] = None,
    dataset_name: str = "dataset",
) -> Dict[str, int]:
    """
    Label a list of examples using the LLM API with production features.

    Features:
    - Retry logic for failed API calls
    - Rate limiting to avoid overload
    - Checkpointing for resumable processing
    - Progress tracking with ETA
    - Garbage detection with circuit breaker pattern
    - Model health check before starting

    Args:
        examples: List of dicts with 'text' and metadata
        output_path: Path to save labeled JSONL
        config: Labeling configuration
        dataset_name: Name for checkpoint files

    Returns:
        Dict with labeling statistics
    """
    if config is None:
        config = LabelingConfig()

    # Health check before starting
    if not config.skip_health_check:
        logger.info("Performing model health check...")
        is_healthy, health_msg = check_model_health(config.api_url, config.model)
        if not is_healthy:
            logger.error(f"Model health check failed: {health_msg}")
            logger.error(
                "Please ensure the vLLM server is running and healthy before labeling."
            )
            raise RuntimeError(f"Model health check failed: {health_msg}")
        logger.info(f"✓ {health_msg}")
    else:
        logger.warning("⚠️ Skipping health check (--skip-health-check)")

    # Setup checkpoint manager
    checkpoint_dir = os.path.dirname(output_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = CheckpointManager(checkpoint_dir, dataset_name)

    # Setup rate limiter
    rate_limiter = RateLimiter(config.rate_limit_rps)

    # Setup garbage detector
    garbage_detector = GarbageDetector(
        consecutive_threshold=config.garbage_consecutive_threshold,
        window_size=config.garbage_window_size,
        garbage_ratio_threshold=config.garbage_ratio_threshold,
        cooldown_base=config.garbage_cooldown_base,
        max_cooldown=config.garbage_max_cooldown,
    )

    # Filter out already processed examples
    examples_to_process = [
        ex for ex in examples if not checkpoint.is_processed(ex["text"])
    ]

    if len(examples_to_process) < len(examples):
        logger.info(
            f"Resuming: {len(examples) - len(examples_to_process)} already processed"
        )

    if not examples_to_process:
        logger.info("All examples already processed")
        return checkpoint.stats

    # Clear output file if starting fresh
    if checkpoint.stats["total"] == 0 and os.path.exists(output_path):
        os.remove(output_path)

    # Track garbage stats
    total_garbage = 0
    aborted = False
    abort_reason = ""

    def label_one(example: Dict) -> tuple:
        """Label a single example with retry logic."""
        text = example["text"]
        result = call_llm_api(text, config, rate_limiter)

        if result:
            labeled = {
                "text": text,
                "label_name": result["label"],
                "source": example.get("source", "unknown"),
                **{k: v for k, v in example.items() if k not in ["text", "source"]},
            }
            return (text, labeled, False)  # is_garbage = False
        return (text, None, True)  # is_garbage = True

    # Process sequentially with garbage protection (thread pool can still handle retries internally)
    processed_count = 0

    with tqdm(total=len(examples_to_process), desc=f"Labeling {dataset_name}") as pbar:
        # Process in batches - larger batches = better throughput
        # But smaller batches = better garbage detection response time
        # Default: 50 examples per batch (good balance)
        batch_size = max(50, config.workers * 10)

        for batch_start in range(0, len(examples_to_process), batch_size):
            if aborted:
                break

            batch = examples_to_process[batch_start : batch_start + batch_size]

            # Check for garbage-induced pause
            should_pause, pause_duration = garbage_detector.should_pause()
            if should_pause:
                logger.warning(
                    f"⚠️ High garbage rate detected. Pausing for {pause_duration:.0f}s..."
                )
                pbar.set_postfix({"status": f"PAUSED {pause_duration:.0f}s"})
                time.sleep(pause_duration)

                # Re-check health after pause
                is_healthy, health_msg = check_model_health(
                    config.api_url, config.model
                )
                if not is_healthy:
                    logger.error(f"Model still unhealthy after pause: {health_msg}")
                    logger.error("Consider stopping and restarting the model server.")
                    # Continue but with caution
                else:
                    logger.info(f"✓ Model recovered: {health_msg}")
                    garbage_detector.reset()  # Reset on recovery

            with ThreadPoolExecutor(max_workers=config.workers) as executor:
                futures = {executor.submit(label_one, ex): ex for ex in batch}

                for future in as_completed(futures):
                    text, result, is_garbage = future.result()

                    # Record garbage result
                    garbage_detector.record_result(is_garbage)
                    if is_garbage:
                        total_garbage += 1

                    checkpoint.mark_processed(text, result)
                    processed_count += 1

                    # Check for abort conditions
                    if (
                        config.garbage_max_total > 0
                        and total_garbage >= config.garbage_max_total
                    ):
                        aborted = True
                        abort_reason = f"Max garbage limit reached ({total_garbage})"
                        break

                    gc_stats = garbage_detector.get_stats()
                    if (
                        gc_stats["consecutive_garbage"]
                        >= config.garbage_consecutive_threshold * 2
                    ):
                        aborted = True
                        abort_reason = f"Too many consecutive garbage responses ({gc_stats['consecutive_garbage']})"
                        break

                    # Update progress bar with stats
                    success_rate = (
                        (checkpoint.stats["total"] - checkpoint.stats["errors"])
                        / max(1, checkpoint.stats["total"])
                        * 100
                    )
                    garbage_ratio = garbage_detector.get_garbage_ratio() * 100
                    pbar.set_postfix(
                        {
                            "success": f"{success_rate:.0f}%",
                            "garbage": f"{garbage_ratio:.0f}%",
                            "errors": checkpoint.stats["errors"],
                        }
                    )
                    pbar.update(1)

                    # Checkpoint periodically
                    if processed_count % config.checkpoint_interval == 0:
                        checkpoint.save_checkpoint()
                        logger.info(
                            f"Checkpoint saved: {processed_count} processed, {total_garbage} garbage"
                        )

    # Final checkpoint
    checkpoint.save_checkpoint()

    if aborted:
        logger.error(f"⛔ Labeling ABORTED: {abort_reason}")
        logger.error(
            f"   Progress saved. Run with --resume to continue when model is healthy."
        )

    # Add garbage stats to result
    checkpoint.stats["garbage_total"] = total_garbage
    checkpoint.stats["aborted"] = aborted
    if abort_reason:
        checkpoint.stats["abort_reason"] = abort_reason

    return checkpoint.stats


def label_dataset_simple(
    examples: List[Dict[str, str]],
    output_path: str,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    workers: int = 4,
    batch_size: int = 100,
) -> Dict[str, int]:
    """Simple version for backward compatibility."""
    config = LabelingConfig(
        api_url=api_url, model=model, workers=workers, checkpoint_interval=batch_size
    )
    return label_dataset(examples, output_path, config, "dataset")


def _append_jsonl(path: str, items: List[Dict]):
    """Append items to JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def label_all_datasets(
    data_config: Optional[DataConfig] = None,
    labeling_config: Optional[LabelingConfig] = None,
    max_per_dataset: int = 10000,
    resume: bool = True,
) -> Dict[str, Dict]:
    """
    Extract and label all downloaded datasets with production features.

    Features:
    - Resumable processing with checkpoints
    - Rate limiting to avoid API overload
    - Retry logic for failed requests
    - Progress tracking with ETA

    Args:
        data_config: Data configuration
        labeling_config: Labeling configuration (retries, rate limit, etc.)
        max_per_dataset: Maximum examples per dataset
        resume: Whether to resume from checkpoint

    Returns:
        Dict mapping dataset name to labeling stats
    """
    if data_config is None:
        data_config = DataConfig()
    if labeling_config is None:
        labeling_config = LabelingConfig()

    raw_data_dir = data_config.raw_data_dir
    labeled_dir = os.path.join(os.path.dirname(raw_data_dir), "labeled_data")
    os.makedirs(labeled_dir, exist_ok=True)

    # Dataset extractors with descriptions
    extractors = {
        # Original dialogue datasets (mostly SAT, NEED_CLARIFICATION, WANT_DIFFERENT)
        "mimics": (extract_mimics_examples, "Search Clarification"),
        "inscit": (extract_inscit_examples, "Information-Seeking"),
        "multiwoz": (extract_multiwoz_examples, "Task-Oriented Dialogue"),
        "sgd": (extract_sgd_examples, "Schema-Guided Dialogue"),
        "redial": (extract_redial_examples, "Recommendation"),
        "hazumi": (extract_hazumi_examples, "Japanese Dialogue"),
        # NEW: Complaint datasets (for WRONG_ANSWER class)
        "customer_complaints": (
            extract_customer_complaints_examples,
            "Customer Complaints [WRONG_ANSWER]",
        ),
        "consumer_complaints_medium": (
            extract_consumer_complaints_medium_examples,
            "Consumer Complaints [WRONG_ANSWER]",
        ),
        "turkish_complaints": (
            extract_turkish_complaints_examples,
            "Turkish Complaints [Multilingual]",
        ),
    }

    all_stats = {}

    print("=" * 70)
    print("PRODUCTION DATASET LABELING")
    print("=" * 70)
    print(f"API URL: {labeling_config.api_url}")
    print(f"Model: {labeling_config.model}")
    print(f"Max per dataset: {max_per_dataset:,}")
    print(f"Workers: {labeling_config.workers}")
    print(f"Rate limit: {labeling_config.rate_limit_rps} req/s")
    print(f"Retries: {labeling_config.max_retries}")
    print(f"Resume from checkpoint: {resume}")
    print("=" * 70)

    for name, (extractor, description) in extractors.items():
        print(f"\n{'='*50}")
        print(f"[{name.upper()}] {description}")
        print(f"{'='*50}")

        # Check if dataset exists
        dataset_dir = os.path.join(raw_data_dir, name)
        if not os.path.exists(dataset_dir):
            print(f"  ⚠️  Dataset not downloaded, skipping...")
            continue

        # Extract examples
        print(f"  📥 Extracting examples...")
        examples = []
        try:
            for example in extractor(raw_data_dir):
                examples.append(example)
                if len(examples) >= max_per_dataset:
                    break
        except Exception as e:
            logger.error(f"Error extracting {name}: {e}")
            continue

        if not examples:
            print(f"  ⚠️  No examples found, skipping...")
            continue

        print(f"  ✓ Found {len(examples):,} examples")

        # Label with production config
        output_path = os.path.join(labeled_dir, f"{name}_labeled.jsonl")

        # Clear checkpoint if not resuming
        if not resume:
            checkpoint_file = os.path.join(labeled_dir, f"{name}_checkpoint.json")
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            if os.path.exists(output_path):
                os.remove(output_path)

        print(f"  🏷️  Labeling with {labeling_config.model}...")
        stats = label_dataset(
            examples, output_path, config=labeling_config, dataset_name=name
        )

        all_stats[name] = stats

        # Print results
        success = stats["total"] - stats["errors"]
        success_rate = success / max(1, stats["total"]) * 100
        print(f"  ✓ Results: {success}/{stats['total']} ({success_rate:.1f}% success)")
        print(
            f"    Labels: SAT={stats['SAT']}, NEED_CLAR={stats['NEED_CLARIFICATION']}, "
            f"WRONG={stats['WRONG_ANSWER']}, WANT_DIFF={stats['WANT_DIFFERENT']}"
        )

    # Combine all labeled data
    combined_path = os.path.join(labeled_dir, "combined_labeled.jsonl")
    print(f"\n📦 Combining all labeled data...")

    total_lines = 0
    with open(combined_path, "w", encoding="utf-8") as out_f:
        for name in extractors.keys():
            src_path = os.path.join(labeled_dir, f"{name}_labeled.jsonl")
            if os.path.exists(src_path):
                with open(src_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1

    print(f"✓ Combined {total_lines:,} labeled examples to {combined_path}")

    # Summary
    print("\n" + "=" * 70)
    print("LABELING SUMMARY")
    print("=" * 70)

    total = {"total": 0, "errors": 0, "garbage_total": 0}
    for label in VALID_LABELS:
        total[label] = 0

    any_aborted = False
    for name, stats in all_stats.items():
        success = stats["total"] - stats["errors"]
        rate = success / max(1, stats["total"]) * 100
        garbage = stats.get("garbage_total", 0)
        aborted = stats.get("aborted", False)
        status = " ⛔ ABORTED" if aborted else ""
        print(
            f"  {name:12s}: {success:6,}/{stats['total']:6,} ({rate:5.1f}%) [garbage: {garbage}]{status}"
        )
        for key in total:
            if key in stats:
                total[key] += stats[key]
        if aborted:
            any_aborted = True

    print("-" * 40)
    success = total["total"] - total["errors"]
    rate = success / max(1, total["total"]) * 100
    print(f"  {'TOTAL':12s}: {success:6,}/{total['total']:6,} ({rate:5.1f}%)")
    print(f"  Total garbage responses: {total['garbage_total']:,}")
    print(f"\n  Label distribution:")
    print(f"    SAT:               {total['SAT']:,}")
    print(f"    NEED_CLARIFICATION:{total['NEED_CLARIFICATION']:,}")
    print(f"    WRONG_ANSWER:      {total['WRONG_ANSWER']:,}")
    print(f"    WANT_DIFFERENT:    {total['WANT_DIFFERENT']:,}")

    if any_aborted:
        print("\n" + "=" * 70)
        print("⚠️  WARNING: Some datasets were aborted due to model issues.")
        print("   Restart the vLLM server and run with --resume to continue.")
        print("=" * 70)

    return all_stats


# Backward compatibility alias
def label_all_datasets_legacy(
    config: Optional[DataConfig] = None,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    max_per_dataset: int = 10000,
    workers: int = 8,
) -> Dict[str, Dict]:
    """Legacy interface for backward compatibility."""
    labeling_config = LabelingConfig(api_url=api_url, model=model, workers=workers)
    return label_all_datasets(config, labeling_config, max_per_dataset)


def download_and_label(
    data_config: Optional[DataConfig] = None,
    labeling_config: Optional[LabelingConfig] = None,
    max_per_dataset: int = 10000,
    skip_download: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: download all datasets and label them.

    Args:
        data_config: Data configuration
        labeling_config: Labeling configuration
        max_per_dataset: Max examples per dataset
        skip_download: Skip download step if datasets exist
        resume: Resume from checkpoint

    Returns:
        Combined results
    """
    results = {"download": {}, "labeling": {}}

    # Download
    if not skip_download:
        results["download"] = download_all_datasets(data_config)
    else:
        print("Skipping download (--skip-download)")

    # Label
    results["labeling"] = label_all_datasets(
        data_config=data_config,
        labeling_config=labeling_config,
        max_per_dataset=max_per_dataset,
        resume=resume,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and label datasets for feedback detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and label with defaults
  python download_datasets.py

  # Only download datasets
  python download_datasets.py --download-only

  # Label with custom settings (production)
  python download_datasets.py --label-only --max-per-dataset 50000 \\
      --workers 8 --rate-limit 10 --retries 5

  # Resume interrupted labeling
  python download_datasets.py --label-only --resume

  # Start fresh (ignore checkpoints)
  python download_datasets.py --label-only --no-resume
""",
    )

    # Action selection
    parser.add_argument(
        "--download-only", action="store_true", help="Only download, don't label"
    )
    parser.add_argument(
        "--label-only", action="store_true", help="Only label (skip download)"
    )

    # API settings
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"vLLM API URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})"
    )

    # Labeling settings
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=10000,
        help="Max examples per dataset (default: 10000)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel workers (default: 4)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=5.0,
        help="Max requests per second (default: 5.0)",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Max retries per request (default: 3)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Save checkpoint every N examples (default: 500)",
    )

    # Checkpoint control
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Start fresh, ignore checkpoints"
    )

    # Garbage protection
    parser.add_argument(
        "--garbage-threshold",
        type=int,
        default=5,
        help="Consecutive garbage responses to trigger circuit breaker (default: 5)",
    )
    parser.add_argument(
        "--garbage-max",
        type=int,
        default=100,
        help="Max total garbage before abort, 0=no limit (default: 100)",
    )
    parser.add_argument(
        "--garbage-cooldown",
        type=float,
        default=30.0,
        help="Base cooldown in seconds when garbage detected (default: 30)",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip model health check (not recommended)",
    )

    args = parser.parse_args()

    # Build labeling config
    labeling_config = LabelingConfig(
        api_url=args.api_url,
        model=args.model,
        workers=args.workers,
        rate_limit_rps=args.rate_limit,
        max_retries=args.retries,
        checkpoint_interval=args.checkpoint_interval,
        garbage_consecutive_threshold=args.garbage_threshold,
        garbage_max_total=args.garbage_max,
        garbage_cooldown_base=args.garbage_cooldown,
        skip_health_check=args.skip_health_check,
    )

    resume = args.resume and not args.no_resume

    if args.download_only:
        download_all_datasets()
    elif args.label_only:
        label_all_datasets(
            labeling_config=labeling_config,
            max_per_dataset=args.max_per_dataset,
            resume=resume,
        )
    else:
        download_and_label(
            labeling_config=labeling_config,
            max_per_dataset=args.max_per_dataset,
            skip_download=False,
            resume=resume,
        )
