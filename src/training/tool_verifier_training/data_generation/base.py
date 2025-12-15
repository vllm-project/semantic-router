"""
Base Classes and Utilities for Data Generation.

Provides abstract base classes for generators and common utilities.
"""

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .entities import Entities, get_entities


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SentinelSample:
    """Sample for Stage 1 Sentinel (prompt classification)."""

    prompt: str
    label: str  # "SAFE" or "INJECTION_RISK"
    source_dataset: str = ""
    injection_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SentinelSample":
        return cls(
            prompt=d["prompt"],
            label=d["label"],
            source_dataset=d.get("source_dataset", ""),
            injection_type=d.get("injection_type"),
        )


@dataclass
class ToolCallSample:
    """Sample for Stage 2 Tool Call Verifier."""

    tool_schema: str
    policy: str
    user_intent: str
    tool_call_json: str
    labels: List[Dict[str, Any]] = field(default_factory=list)
    split: str = "train"
    source_dataset: str = ""
    injection_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolCallSample":
        return cls(
            tool_schema=d["tool_schema"],
            policy=d.get("policy", ""),
            user_intent=d["user_intent"],
            tool_call_json=d["tool_call_json"],
            labels=d.get("labels", []),
            split=d.get("split", "train"),
            source_dataset=d.get("source_dataset", ""),
            injection_content=d.get("injection_content"),
        )

    @property
    def is_unauthorized(self) -> bool:
        return len(self.labels) > 0


# =============================================================================
# BASE GENERATOR CLASS
# =============================================================================


class BaseGenerator(ABC):
    """
    Abstract base class for all sample generators.

    Each generator focuses on a specific attack category or pattern type.
    Generators should be stateless and produce reproducible results given a seed.

    Generators can customize:
    - generate(): Core sample generation
    - augment(): Text augmentation (synonyms, style, etc.)
    - paraphrase(): LLM-based paraphrasing
    - generate_adversarial(): Create hard negative examples
    """

    def __init__(
        self,
        entities: Optional[Entities] = None,
        seed: int = 42,
        llm_endpoint: Optional[str] = None,
    ):
        self.entities = entities or get_entities()
        self.seed = seed
        self.llm_endpoint = llm_endpoint
        self._llm_client = None
        random.seed(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this generator."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Category (e.g., 'filesystem', 'financial', 'adversarial')."""
        pass

    @abstractmethod
    def generate(self, num_samples: int = 100) -> List[ToolCallSample]:
        """
        Generate samples.

        Args:
            num_samples: Target number of samples to generate.

        Returns:
            List of ToolCallSample objects.
        """
        pass

    # =========================================================================
    # AUGMENTATION HOOKS - Override in subclasses for customization
    # =========================================================================

    def augment(
        self, samples: List[ToolCallSample], multiplier: float = 1.5
    ) -> List[ToolCallSample]:
        """
        Augment samples with text variations.

        Override this method in subclasses to customize augmentation
        for the specific dataset's characteristics.

        Args:
            samples: Original samples.
            multiplier: Target size multiplier (1.5 = 50% more samples).

        Returns:
            Original + augmented samples.
        """
        result = list(samples)
        target_new = int(len(samples) * (multiplier - 1))

        for _ in range(target_new):
            sample = random.choice(samples)
            augmented = self._default_augment_one(sample)
            if augmented:
                result.append(augmented)

        return result

    def _default_augment_one(self, sample: ToolCallSample) -> Optional[ToolCallSample]:
        """Default single-sample augmentation. Override for custom behavior."""
        intent = sample.user_intent

        # Apply random augmentation
        aug_type = random.choice(["synonym", "style", "case", "insert"])

        if aug_type == "synonym":
            intent = self._synonym_replace(intent)
        elif aug_type == "style":
            intent = self._style_augment(intent)
        elif aug_type == "case":
            intent = self._case_augment(intent)
        elif aug_type == "insert":
            intent = self._insert_filler(intent)

        return ToolCallSample(
            tool_schema=sample.tool_schema,
            policy=sample.policy,
            user_intent=intent,
            tool_call_json=sample.tool_call_json,
            labels=sample.labels.copy(),
            split=sample.split,
            source_dataset=f"{sample.source_dataset}_aug_{aug_type}",
        )

    def _synonym_replace(self, text: str) -> str:
        """Replace words with synonyms."""
        synonyms = {
            "send": ["transmit", "dispatch", "forward"],
            "delete": ["remove", "erase", "clear"],
            "read": ["view", "access", "open"],
            "transfer": ["move", "send", "relocate"],
            "get": ["retrieve", "fetch", "obtain"],
            "run": ["execute", "perform", "launch"],
            "check": ["verify", "examine", "review"],
        }
        words = text.split()
        for i, word in enumerate(words):
            w = word.lower().strip(".,!?")
            if w in synonyms and random.random() < 0.5:
                words[i] = random.choice(synonyms[w])
                break
        return " ".join(words)

    def _style_augment(self, text: str) -> str:
        """Add greetings, politeness, or endings."""
        choice = random.choice(["greeting", "polite", "ending"])
        if choice == "greeting":
            return random.choice(["Hi! ", "Hello, ", "Hey, "]) + text
        elif choice == "polite":
            return (
                random.choice(["Please ", "Could you ", "Would you "])
                + text[0].lower()
                + text[1:]
            )
        else:
            return text.rstrip(".!?") + random.choice([" Thanks!", " Thank you.", ""])

    def _case_augment(self, text: str) -> str:
        """Change text case."""
        choice = random.choice(["lower", "upper", "title"])
        if choice == "lower":
            return text.lower()
        elif choice == "upper":
            return text.upper()
        return text.title()

    def _insert_filler(self, text: str) -> str:
        """Insert filler words."""
        fillers = ["please", "kindly", "just", "simply"]
        words = text.split()
        if words:
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(fillers))
        return " ".join(words)

    def paraphrase(
        self,
        samples: List[ToolCallSample],
        num_variations: int = 1,
        llm_endpoint: Optional[str] = None,
    ) -> List[ToolCallSample]:
        """
        Paraphrase samples using LLM.

        Override this method in subclasses to customize the paraphrasing
        prompt and behavior for the specific dataset.

        Args:
            samples: Original samples.
            num_variations: Number of paraphrased versions per sample.
            llm_endpoint: LLM API endpoint (uses self.llm_endpoint if not provided).

        Returns:
            Original + paraphrased samples.
        """
        endpoint = llm_endpoint or self.llm_endpoint
        if not endpoint:
            return samples

        result = list(samples)

        for sample in samples:
            for _ in range(num_variations):
                paraphrased = self._paraphrase_one(sample, endpoint)
                if paraphrased:
                    result.append(paraphrased)

        return result

    def _paraphrase_one(
        self, sample: ToolCallSample, endpoint: str
    ) -> Optional[ToolCallSample]:
        """
        Paraphrase a single sample. Override for custom prompts.

        Default implementation uses a generic paraphrasing prompt.
        Subclasses can override to use domain-specific prompts.
        """
        system_prompt = self._get_paraphrase_system_prompt()
        user_prompt = self._get_paraphrase_user_prompt(sample)

        response = self._call_llm(endpoint, system_prompt, user_prompt)
        if not response or len(response) < 10:
            return None

        return ToolCallSample(
            tool_schema=sample.tool_schema,
            policy=sample.policy,
            user_intent=response,
            tool_call_json=sample.tool_call_json,
            labels=sample.labels.copy(),
            split=sample.split,
            source_dataset=f"{sample.source_dataset}_para",
        )

    def _get_paraphrase_system_prompt(self) -> str:
        """Override to customize the paraphrasing system prompt."""
        return """You are a helpful assistant that paraphrases text.
Paraphrase the given request while:
1. Keeping the EXACT same meaning and intent
2. Preserving all specific values (names, numbers, paths)
3. Using different words and sentence structure
Output ONLY the paraphrased text."""

    def _get_paraphrase_user_prompt(self, sample: ToolCallSample) -> str:
        """Override to customize the paraphrasing user prompt."""
        return f"Paraphrase: {sample.user_intent}"

    def generate_adversarial(
        self,
        samples: List[ToolCallSample],
        ratio: float = 0.5,
        llm_endpoint: Optional[str] = None,
    ) -> List[ToolCallSample]:
        """
        Generate adversarial (hard negative) examples.

        Override this method in subclasses to customize adversarial
        generation for the specific dataset's characteristics.

        Args:
            samples: Original samples (uses AUTHORIZED ones).
            ratio: Ratio of samples to generate adversarial versions for.
            llm_endpoint: Optional LLM endpoint for complex generation.

        Returns:
            Adversarial samples only (not including originals).
        """
        authorized = [s for s in samples if not s.is_unauthorized]
        if not authorized:
            return []

        selected = random.sample(authorized, int(len(authorized) * ratio))
        result = []

        for sample in selected:
            adversarial = self._generate_adversarial_one(sample, llm_endpoint)
            if adversarial:
                result.append(adversarial)

        return result

    def _generate_adversarial_one(
        self, sample: ToolCallSample, llm_endpoint: Optional[str] = None
    ) -> Optional[ToolCallSample]:
        """
        Generate one adversarial sample. Override for custom logic.

        Default implementation creates value mismatches.
        """
        try:
            tool_call = json.loads(sample.tool_call_json)
        except:
            return None

        args = tool_call.get("arguments", {})
        new_args = self._create_mismatch(args)

        if new_args == args:
            return None

        new_tool_call = json.dumps({"name": tool_call["name"], "arguments": new_args})

        return ToolCallSample(
            tool_schema=sample.tool_schema,
            policy=sample.policy,
            user_intent=sample.user_intent,  # Keep original intent
            tool_call_json=new_tool_call,  # Modified tool call
            labels=[{"start": 0, "end": len(new_tool_call), "label": "UNAUTHORIZED"}],
            split=sample.split,
            source_dataset=f"{sample.source_dataset}_adv",
        )

    def _create_mismatch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a mismatch in arguments. Override for category-specific mismatches.
        """
        new_args = args.copy()

        for key, value in args.items():
            # Email mismatch
            if (
                ("email" in key or key == "to")
                and isinstance(value, str)
                and "@" in value
            ):
                old_name = value.split("@")[0]
                new_name = self.entities.random_name()
                while new_name == old_name:
                    new_name = self.entities.random_name()
                new_args[key] = f"{new_name}@company.com"
                return new_args

            # Amount mismatch
            if "amount" in key and isinstance(value, (int, float)):
                new_args[key] = self.entities.random_different_amount(int(value))
                return new_args

            # File/path mismatch
            if ("path" in key or "file" in key) and isinstance(value, str):
                new_args[key] = self.entities.random_full_path()
                return new_args

        return new_args

    def _call_llm(
        self,
        endpoint: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> Optional[str]:
        """Call LLM API. Returns None on failure."""
        try:
            import requests

            resp = requests.post(
                f"{endpoint}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=30,
            )

            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            pass

        return None

    # =========================================================================
    # GENERATION WITH AUGMENTATION
    # =========================================================================

    def generate_with_augmentation(
        self,
        num_samples: int = 100,
        augment_multiplier: float = 1.5,
        adversarial_ratio: float = 0.3,
        paraphrase_count: int = 0,
        llm_endpoint: Optional[str] = None,
    ) -> List[ToolCallSample]:
        """
        Generate samples with augmentation pipeline.

        Args:
            num_samples: Base number of samples.
            augment_multiplier: Text augmentation multiplier.
            adversarial_ratio: Ratio of adversarial samples.
            paraphrase_count: Number of LLM paraphrases per sample (0 to disable).
            llm_endpoint: LLM endpoint for paraphrasing.

        Returns:
            All samples (original + augmented + adversarial + paraphrased).
        """
        # Generate base samples
        samples = self.generate(num_samples)

        # Apply text augmentation
        if augment_multiplier > 1.0:
            samples = self.augment(samples, augment_multiplier)

        # Generate adversarial examples
        if adversarial_ratio > 0:
            adversarial = self.generate_adversarial(
                samples, adversarial_ratio, llm_endpoint
            )
            samples.extend(adversarial)

        # Apply LLM paraphrasing
        if paraphrase_count > 0 and llm_endpoint:
            samples = self.paraphrase(samples, paraphrase_count, llm_endpoint)

        return samples

    def generate_balanced(
        self, num_samples: int = 100, authorized_ratio: float = 0.5
    ) -> List[ToolCallSample]:
        """
        Generate balanced samples with specified authorized/unauthorized ratio.

        Args:
            num_samples: Total number of samples.
            authorized_ratio: Ratio of authorized samples (0.0 to 1.0).

        Returns:
            Balanced list of samples.
        """
        samples = self.generate(num_samples * 2)  # Generate extra for balancing

        if not samples:
            return []

        authorized = [s for s in samples if not s.is_unauthorized]
        unauthorized = [s for s in samples if s.is_unauthorized]

        n_auth = int(num_samples * authorized_ratio)
        n_unauth = num_samples - n_auth

        result = []

        # Handle authorized samples
        if authorized and n_auth > 0:
            if len(authorized) < n_auth:
                authorized = authorized * (n_auth // len(authorized) + 1)
            result.extend(random.sample(authorized, min(n_auth, len(authorized))))

        # Handle unauthorized samples
        if unauthorized and n_unauth > 0:
            if len(unauthorized) < n_unauth:
                unauthorized = unauthorized * (n_unauth // len(unauthorized) + 1)
            result.extend(random.sample(unauthorized, min(n_unauth, len(unauthorized))))

        random.shuffle(result)
        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Create a JSON tool call string."""
    return json.dumps({"name": name, "arguments": arguments})


def create_tool_schema(name: str, description: str) -> str:
    """Create a JSON tool schema string."""
    return json.dumps({"name": name, "description": description})


def create_unauthorized_label(tool_call_json: str) -> List[Dict[str, Any]]:
    """Create UNAUTHORIZED label spanning the entire tool call."""
    return [{"start": 0, "end": len(tool_call_json), "label": "UNAUTHORIZED"}]


def apply_intent_variations(intent: str) -> str:
    """Apply random variations to user intent for diversity."""
    # Add prefix
    if random.random() > 0.5:
        prefix = random.choice(["Please ", "Can you ", "Could you ", "Would you ", ""])
        if prefix:
            intent = prefix + intent[0].lower() + intent[1:]

    # Add suffix
    if random.random() > 0.6:
        suffix = random.choice([" Thanks!", " Thank you.", " Please.", ""])
        intent = intent.rstrip(".!?") + suffix

    # Add greeting
    if random.random() > 0.8:
        greeting = random.choice(["Hi! ", "Hey, ", "Hello, ", ""])
        intent = greeting + intent

    return intent


def save_samples(samples: List[ToolCallSample], path: Path):
    """Save samples to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in samples], f, indent=2)
    print(f"  Saved {len(samples)} samples to {path}")


def load_samples(path: Path) -> List[ToolCallSample]:
    """Load samples from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [ToolCallSample(**d) for d in data]


def split_train_dev(
    samples: List[ToolCallSample], train_ratio: float = 0.8
) -> Tuple[List[ToolCallSample], List[ToolCallSample]]:
    """Split samples into train and dev sets."""
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)

    train = samples[:split_idx]
    dev = samples[split_idx:]

    for s in train:
        s.split = "train"
    for s in dev:
        s.split = "dev"

    return train, dev


def get_sample_stats(samples: List[ToolCallSample]) -> Dict[str, Any]:
    """Get statistics about a list of samples."""
    from collections import Counter

    authorized = sum(1 for s in samples if not s.is_unauthorized)
    unauthorized = sum(1 for s in samples if s.is_unauthorized)

    sources = Counter(s.source_dataset for s in samples)

    return {
        "total": len(samples),
        "authorized": authorized,
        "unauthorized": unauthorized,
        "ratio": authorized / len(samples) if samples else 0,
        "sources": dict(sources.most_common()),
    }


def print_stats(samples: List[ToolCallSample], title: str = "Dataset Stats"):
    """Print statistics about samples."""
    stats = get_sample_stats(samples)
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total: {stats['total']}")
    print(f"Authorized: {stats['authorized']}")
    print(f"Unauthorized: {stats['unauthorized']}")
    print(f"Ratio (auth): {stats['ratio']:.2%}")
    print(f"\nSources:")
    for src, count in list(stats["sources"].items())[:10]:
        print(f"  {src}: {count}")
