"""Red-team evasion benchmark for router classifier signals."""

from .attack import (
    DEFAULT_WORD_POOL,
    AttackResult,
    BenchmarkReport,
    Scorer,
    greedy_suffix_attack,
    summarize,
)
from .datasets import load_prompts
from .scorers import TransformerScorer

__all__ = [
    "DEFAULT_WORD_POOL",
    "AttackResult",
    "BenchmarkReport",
    "Scorer",
    "TransformerScorer",
    "greedy_suffix_attack",
    "load_prompts",
    "summarize",
]
