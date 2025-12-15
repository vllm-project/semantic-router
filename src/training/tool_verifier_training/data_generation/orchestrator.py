"""
Data Generation Orchestrator.

Combines all generators and loaders into a unified pipeline.
"""

import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .config import Config
from .base import (
    SentinelSample,
    ToolCallSample,
    save_samples,
    split_train_dev,
    print_stats,
)
from .generators import get_all_generators, get_generator, generate_jailbreak_samples
from .loaders import LOADERS, STAGE1_LOADERS, STAGE2_LOADERS


class DataOrchestrator:
    """
    Main orchestrator for data generation.

    Combines HuggingFace loaders and pattern generators into a unified pipeline.

    Usage:
        config = Config(output_dir="data/generated")
        orchestrator = DataOrchestrator(config)

        # Generate all data
        stage1_train, stage1_dev = orchestrator.generate_stage1_data()
        stage2_train, stage2_dev = orchestrator.generate_stage2_data()

        # Or generate with custom settings
        orchestrator.enable_only(["adversarial", "filesystem"])
        samples = orchestrator.generate_tool_call_samples()
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        random.seed(self.config.seed)
        self._generators = None

    @property
    def generators(self):
        """Lazily initialize generators."""
        if self._generators is None:
            self._generators = get_all_generators(seed=self.config.seed)
        return self._generators

    def enable_only(self, names: List[str]):
        """Enable only specific generators."""
        for name in self.config.generators:
            self.config.enable_generator(name, name in names)

    def enable_all(self):
        """Enable all generators."""
        for name in self.config.generators:
            self.config.enable_generator(name, True)

    # =========================================================================
    # STAGE 1: Sentinel (Prompt Classification)
    # =========================================================================

    def generate_stage1_data(self) -> Tuple[List[SentinelSample], List[SentinelSample]]:
        """
        Generate Stage 1 training data for Sentinel.

        Returns:
            Tuple of (train_samples, dev_samples)
        """
        print("=" * 60)
        print("Stage 1 Data Generation (Sentinel)")
        print("=" * 60)

        all_samples = []

        # Load HuggingFace datasets
        if self.config.use_huggingface:
            print("\n1. Loading HuggingFace datasets...")
            all_samples.extend(self._load_stage1_hf_datasets())

        # Generate synthetic data
        print("\n2. Generating synthetic data...")
        all_samples.extend(self._generate_stage1_synthetic())

        # Balance classes
        injection = [s for s in all_samples if s.label == "INJECTION_RISK"]
        safe = [s for s in all_samples if s.label == "SAFE"]
        print(f"\n3. Balancing: {len(injection)} injection, {len(safe)} safe")

        n_balanced = min(len(injection), len(safe))
        balanced = random.sample(injection, n_balanced) + random.sample(
            safe, n_balanced
        )
        random.shuffle(balanced)

        # Split
        split_idx = int(len(balanced) * self.config.train_ratio)
        train = balanced[:split_idx]
        dev = balanced[split_idx:]

        # Save
        self._save_sentinel_samples(train, "stage1_train.json")
        self._save_sentinel_samples(dev, "stage1_dev.json")

        print(f"\n{'='*60}")
        print(f"Stage 1 Complete: {len(train)} train, {len(dev)} dev")
        print(f"Labels: {dict(Counter(s.label for s in train))}")

        return train, dev

    def _load_stage1_hf_datasets(self) -> List[SentinelSample]:
        """Load all enabled HuggingFace datasets for Stage 1."""
        samples = []

        # Original loaders
        stage1_loaders = [
            "jailbreak_llms",
            "wildjailbreak",
            "hackaprompt",
            "jailbreakbench",
            "chatgpt_jailbreaks",
            "toxic_chat",
            "alpaca",
            "dolly",
            # NEW: 2024-2025 Security Research datasets
            "xstest",
            "advbench",
            "beaver_tails",
            "simple_safety_tests",
            "mcp_attacks",  # MCP Tool Poisoning, Shadowing, Rug Pulls
        ]

        for name in stage1_loaders:
            if name not in self.config.loaders:
                # For new loaders not in config, use default settings
                if name in LOADERS:
                    try:
                        loader_func = LOADERS[name]
                        loaded = loader_func(max_samples=2000)
                        samples.extend(loaded)
                    except Exception as e:
                        print(f"  Warning: Could not load {name}: {e}")
                continue

            loader_config = self.config.loaders[name]
            if not loader_config.enabled:
                continue
            if name not in LOADERS:
                continue

            loader_func = LOADERS[name]
            loaded = loader_func(max_samples=loader_config.max_samples)
            samples.extend(loaded)

        return samples

    def _generate_stage1_synthetic(self) -> List[SentinelSample]:
        """
        Generate synthetic Stage 1 data using comprehensive jailbreak patterns.

        Categories covered:
        1. Roleplay/Identity Manipulation
        2. Authority/Privilege Bypass
        3. Prompt Delimiter Attacks
        4. XML/HTML Tag Injection
        5. Social Engineering
        6. Encoding/Obfuscation
        7. Multi-turn Manipulation
        8. DAN-style Jailbreaks
        9. Tool Abuse Patterns
        10. Dangerous Commands
        11. Data Exfiltration
        12. System Prompt Extraction
        13. Payload Smuggling
        14. Context Window Attacks
        15. Instruction Hierarchy Bypass
        """
        samples = generate_jailbreak_samples(
            samples_per_category=self.config.jailbreak_samples_per_category,
            include_safe=True,
            safe_count=self.config.safe_samples_count,
            seed=self.config.seed,
        )
        print(
            f"  Generated {len(samples)} synthetic jailbreak samples (15 attack categories)"
        )
        return samples

    def _load_stage2_hf_datasets(self) -> List[ToolCallSample]:
        """Load ALL Stage 2 HuggingFace datasets for Tool Call Verifier."""
        samples = []

        # Map config names to loader names
        loader_mapping = {
            "llmail_inject": "llmail_inject",
            "wildjailbreak_toolcall": "wildjailbreak_toolcall",
            "hackaprompt_toolcall": "hackaprompt_toolcall",
            "jailbreakbench_toolcall": "jailbreakbench_toolcall",
            "chatgpt_jailbreaks_toolcall": "chatgpt_jailbreaks_toolcall",
            # NEW: MCP attack patterns for tool-call verification
            "mcp_attacks_toolcall": "mcp_attacks_toolcall",
        }

        for config_name, loader_name in loader_mapping.items():
            if loader_name not in STAGE2_LOADERS:
                continue

            # Get config, use default if not specified
            loader_config = self.config.loaders.get(config_name)
            max_samples = loader_config.max_samples if loader_config else 5000
            enabled = loader_config.enabled if loader_config else True

            if not enabled:
                continue

            loader_func = STAGE2_LOADERS[loader_name]
            try:
                loaded = loader_func(max_samples=max_samples)
                samples.extend(loaded)
            except Exception as e:
                print(f"  Warning: Failed to load {loader_name}: {e}")

        print(f"  Total: {len(samples)} samples from HuggingFace")
        return samples

    def _save_sentinel_samples(self, samples: List[SentinelSample], filename: str):
        """Save Sentinel samples to file."""
        import json

        path = self.config.output_dir / filename
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in samples], f, indent=2)
        print(f"  Saved {len(samples)} samples to {path}")

    # =========================================================================
    # STAGE 2: Tool Call Verifier
    # =========================================================================

    def generate_stage2_data(
        self, apply_paraphrase: bool = False
    ) -> Tuple[List[ToolCallSample], List[ToolCallSample]]:
        """
        Generate Stage 2 training data for Tool Call Verifier.

        Loads ALL HuggingFace datasets:
        - LLMail-Inject (GOLD standard - microsoft)
        - WildJailbreak (262K jailbreaks - allenai)
        - HackAPrompt (600K injections)
        - JailbreakBench (harmful behaviors)
        - ChatGPT-Jailbreak-Prompts (curated jailbreaks)

        Args:
            apply_paraphrase: Whether to apply LLM paraphrasing to generator samples.

        Returns:
            Tuple of (train_samples, dev_samples)
        """
        print("\n" + "=" * 60)
        print("Stage 2 Data Generation (Tool Call Verifier)")
        print("=" * 60)
        if apply_paraphrase and self.config.llm_endpoint:
            print(f"LLM Paraphrasing: ENABLED ({self.config.llm_endpoint})")

        all_samples = []

        # Load ALL Stage 2 HuggingFace datasets
        if self.config.use_huggingface:
            print("\n1. Loading HuggingFace datasets (Stage 2)...")
            all_samples.extend(self._load_stage2_hf_datasets())

        # Generate from all enabled generators (with paraphrasing if requested)
        print("\n2. Generating from pattern generators...")
        all_samples.extend(
            self.generate_tool_call_samples(
                apply_augmentation=True,
                apply_adversarial=True,
                apply_paraphrase=apply_paraphrase,
            )
        )

        random.shuffle(all_samples)

        # Split
        train, dev = split_train_dev(all_samples, self.config.train_ratio)

        # Save
        save_samples(train, self.config.output_dir / "stage2_train.json")
        save_samples(dev, self.config.output_dir / "stage2_dev.json")

        unauth_train = sum(1 for s in train if s.is_unauthorized)
        print(f"\n{'='*60}")
        print(f"Stage 2 Complete: {len(train)} train, {len(dev)} dev")
        print(f"Unauthorized in train: {unauth_train}/{len(train)}")

        return train, dev

    def generate_tool_call_samples(
        self,
        apply_augmentation: bool = True,
        apply_adversarial: bool = True,
        apply_paraphrase: bool = False,
    ) -> List[ToolCallSample]:
        """
        Generate tool call samples from all enabled generators.

        Each generator's augment(), generate_adversarial(), and paraphrase()
        methods are called according to its configuration.

        Args:
            apply_augmentation: Whether to apply text augmentation.
            apply_adversarial: Whether to generate adversarial examples.
            apply_paraphrase: Whether to apply LLM paraphrasing.

        Returns:
            List of ToolCallSample objects.
        """
        all_samples = []

        for name, gen_config in self.config.generators.items():
            if not gen_config.enabled:
                continue

            if name not in self.generators:
                print(f"  Warning: Generator '{name}' not found")
                continue

            generator = self.generators[name]

            # Base generation
            samples = generator.generate_balanced(
                num_samples=gen_config.num_samples,
                authorized_ratio=gen_config.authorized_ratio,
            )
            base_count = len(samples)

            # Apply generator's custom augmentation
            if apply_augmentation and gen_config.augment:
                samples = generator.augment(samples, gen_config.augment_multiplier)

            # Generate adversarial examples using generator's custom method
            if apply_adversarial and gen_config.generate_adversarial:
                adversarial = generator.generate_adversarial(
                    samples, gen_config.adversarial_ratio, self.config.llm_endpoint
                )
                samples.extend(adversarial)

            # Apply LLM paraphrasing using generator's custom method
            if apply_paraphrase and gen_config.paraphrase and self.config.llm_endpoint:
                samples = generator.paraphrase(
                    samples, gen_config.paraphrase_count, self.config.llm_endpoint
                )

            aug_count = len(samples) - base_count
            print(
                f"    {name}: {base_count} base + {aug_count} augmented = {len(samples)} total"
            )
            all_samples.extend(samples)

        print(f"  Total: {len(all_samples)} samples from generators")
        return all_samples

    def generate_with_full_augmentation(self) -> List[ToolCallSample]:
        """
        Generate samples with all augmentation methods enabled.

        Calls each generator's custom augment(), generate_adversarial(),
        and paraphrase() methods.
        """
        return self.generate_tool_call_samples(
            apply_augmentation=True,
            apply_adversarial=True,
            apply_paraphrase=True,
        )

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def generate_all(self, apply_paraphrase: bool = False) -> Dict[str, Any]:
        """
        Generate all data (Stage 1 and Stage 2).

        Args:
            apply_paraphrase: Whether to apply LLM paraphrasing to Stage 2 generator samples.

        Returns:
            Dictionary with train/dev splits for both stages.
        """
        stage1_train, stage1_dev = self.generate_stage1_data()
        stage2_train, stage2_dev = self.generate_stage2_data(
            apply_paraphrase=apply_paraphrase
        )

        return {
            "stage1": {"train": stage1_train, "dev": stage1_dev},
            "stage2": {"train": stage2_train, "dev": stage2_dev},
        }

    def generate_adversarial_only(
        self, num_samples: int = 2000
    ) -> List[ToolCallSample]:
        """Generate only adversarial (intent-mismatch) samples."""
        self.enable_only(["adversarial"])
        self.config.set_generator_samples("adversarial", num_samples)
        return self.generate_tool_call_samples()

    def generate_multi_tool_attacks(
        self, num_samples_per: int = 100
    ) -> List[ToolCallSample]:
        """Generate diverse multi-tool attack samples."""
        attack_generators = [
            "filesystem",
            "code_execution",
            "network",
            "authentication",
            "financial",
            "sysadmin",
        ]
        self.enable_only(attack_generators)
        for name in attack_generators:
            self.config.set_generator_samples(name, num_samples_per)
        return self.generate_tool_call_samples()

    def get_stats(self, samples: List[ToolCallSample]) -> Dict[str, Any]:
        """Get statistics about generated samples."""
        from .base import get_sample_stats

        return get_sample_stats(samples)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def generate_data(
    output_dir: str = "data/generated",
    use_huggingface: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Convenience function to generate all data with default settings.

    Args:
        output_dir: Output directory for generated data.
        use_huggingface: Whether to use HuggingFace datasets.
        seed: Random seed.

    Returns:
        Dictionary with generated data.
    """
    config = Config(
        output_dir=Path(output_dir),
        use_huggingface=use_huggingface,
        seed=seed,
    )
    orchestrator = DataOrchestrator(config)
    return orchestrator.generate_all()


def generate_stage2_only(
    output_dir: str = "data/generated",
    generators: List[str] = None,
    num_samples_per: int = 100,
    seed: int = 42,
) -> Tuple[List[ToolCallSample], List[ToolCallSample]]:
    """
    Generate only Stage 2 data with specific generators.

    Args:
        output_dir: Output directory.
        generators: List of generator names to enable. None = all.
        num_samples_per: Samples per generator.
        seed: Random seed.

    Returns:
        Tuple of (train, dev) samples.
    """
    config = Config(
        output_dir=Path(output_dir),
        use_huggingface=True,
        seed=seed,
    )

    orchestrator = DataOrchestrator(config)

    if generators:
        orchestrator.enable_only(generators)

    for name in config.generators:
        config.set_generator_samples(name, num_samples_per)

    return orchestrator.generate_stage2_data()
