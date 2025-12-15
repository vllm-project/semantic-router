"""
Centralized Configuration for Data Generation.

All configurable parameters are defined here for easy modification.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GeneratorConfig:
    """Configuration for individual generators."""

    enabled: bool = True
    num_samples: int = 100
    authorized_ratio: float = 0.5  # Ratio of authorized samples

    # Augmentation settings (each generator customizes these)
    augment: bool = True
    augment_multiplier: float = 1.5  # 1.5 = 50% more samples

    # Adversarial generation
    generate_adversarial: bool = True
    adversarial_ratio: float = 0.3  # Ratio of adversarial samples

    # LLM-based paraphrasing
    paraphrase: bool = False
    paraphrase_count: int = 1  # Number of paraphrases per sample


@dataclass
class LoaderConfig:
    """Configuration for dataset loaders."""

    enabled: bool = True
    max_samples: int = 5000
    streaming: bool = True  # Use streaming for large datasets


@dataclass
class Config:
    """Main configuration for data generation."""

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/generated"))
    train_ratio: float = 0.8

    # HuggingFace settings
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    use_huggingface: bool = True

    # LLM settings (for paraphrasing and adversarial generation)
    llm_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "LLM_ENDPOINT", "http://localhost:8000/v1"
        )
    )

    # Stage 1 (Sentinel) settings
    stage1_synthetic_jailbreaks: int = 5000
    stage1_synthetic_injections: int = 3000
    stage1_hard_neg_per_category: int = 50
    stage1_benign_per_category: int = 50

    # Jailbreak pattern generation settings (for Stage 1)
    # 15 categories × samples_per_category = total injection samples
    jailbreak_samples_per_category: int = 50  # Default: 50 × 15 = 750 injection samples
    safe_samples_count: int = 200  # Safe contrast samples

    # Stage 2 (Tool Verifier) settings
    stage2_authorized: int = 3000
    stage2_unauthorized: int = 3000

    # Generator configurations
    generators: Dict[str, GeneratorConfig] = field(
        default_factory=lambda: {
            "filesystem": GeneratorConfig(enabled=True, num_samples=100),
            "code_execution": GeneratorConfig(enabled=True, num_samples=100),
            "network": GeneratorConfig(enabled=True, num_samples=80),
            "authentication": GeneratorConfig(enabled=True, num_samples=100),
            "financial": GeneratorConfig(enabled=True, num_samples=80),
            "sysadmin": GeneratorConfig(enabled=True, num_samples=80),
            "email": GeneratorConfig(enabled=True, num_samples=100),
            "adversarial": GeneratorConfig(enabled=True, num_samples=500),
            "injection": GeneratorConfig(enabled=True, num_samples=200),
        }
    )

    # Loader configurations
    loaders: Dict[str, LoaderConfig] = field(
        default_factory=lambda: {
            # Stage 1 loaders (Sentinel)
            "jailbreak_llms": LoaderConfig(enabled=True, max_samples=5000),
            "wildjailbreak": LoaderConfig(
                enabled=True, max_samples=5000, streaming=True
            ),
            "hackaprompt": LoaderConfig(enabled=True, max_samples=5000, streaming=True),
            "jailbreakbench": LoaderConfig(enabled=True, max_samples=5000),
            "chatgpt_jailbreaks": LoaderConfig(enabled=True, max_samples=2000),
            "toxic_chat": LoaderConfig(enabled=True, max_samples=5000),
            "alpaca": LoaderConfig(enabled=True, max_samples=5000),
            "dolly": LoaderConfig(enabled=True, max_samples=5000),
            # Stage 2 loaders (Tool Call Verifier)
            "llmail_inject": LoaderConfig(enabled=True, max_samples=10000),
            "wildjailbreak_toolcall": LoaderConfig(
                enabled=True, max_samples=10000, streaming=True
            ),
            "hackaprompt_toolcall": LoaderConfig(
                enabled=True, max_samples=10000, streaming=True
            ),
            "jailbreakbench_toolcall": LoaderConfig(enabled=True, max_samples=2000),
            "chatgpt_jailbreaks_toolcall": LoaderConfig(enabled=True, max_samples=1000),
        }
    )

    # Random seed
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def enable_generator(self, name: str, enabled: bool = True):
        """Enable or disable a specific generator."""
        if name in self.generators:
            self.generators[name].enabled = enabled

    def enable_loader(self, name: str, enabled: bool = True):
        """Enable or disable a specific loader."""
        if name in self.loaders:
            self.loaders[name].enabled = enabled

    def set_generator_samples(self, name: str, num_samples: int):
        """Set number of samples for a specific generator."""
        if name in self.generators:
            self.generators[name].num_samples = num_samples

    def get_enabled_generators(self) -> List[str]:
        """Get list of enabled generators."""
        return [name for name, cfg in self.generators.items() if cfg.enabled]

    def get_enabled_loaders(self) -> List[str]:
        """Get list of enabled loaders."""
        return [name for name, cfg in self.loaders.items() if cfg.enabled]


# Preset configurations
def get_minimal_config() -> Config:
    """Minimal configuration for quick testing."""
    config = Config()
    config.stage1_synthetic_jailbreaks = 500
    config.stage1_synthetic_injections = 300
    config.stage2_authorized = 300
    config.stage2_unauthorized = 300
    # Jailbreak patterns: 15 categories × 10 = 150 + 50 safe = 200 samples
    config.jailbreak_samples_per_category = 10
    config.safe_samples_count = 50
    for name in config.generators:
        config.set_generator_samples(name, 20)
    for name in config.loaders:
        config.loaders[name].max_samples = 500
    return config


def get_full_config() -> Config:
    """Full configuration for production training."""
    config = Config()
    config.stage1_synthetic_jailbreaks = 10000
    config.stage1_synthetic_injections = 5000
    config.stage2_authorized = 5000
    config.stage2_unauthorized = 5000
    # Jailbreak patterns: 15 categories × 100 = 1500 + 500 safe = 2000 samples
    config.jailbreak_samples_per_category = 100
    config.safe_samples_count = 500
    for name in config.generators:
        config.set_generator_samples(name, 200)
    for name in config.loaders:
        config.loaders[name].max_samples = 10000
    return config


def get_adversarial_only_config() -> Config:
    """Configuration focusing on adversarial patterns only."""
    config = Config()
    config.use_huggingface = False
    for name in config.generators:
        config.enable_generator(name, name == "adversarial")
    config.set_generator_samples("adversarial", 2000)
    return config
