"""
Attack Pattern Generators.

Each generator focuses on a specific category of tool-call attacks.
All generators inherit from BaseGenerator and produce ToolCallSample objects.

For Stage 1 Sentinel training, use JailbreakPatternGenerator which produces SentinelSample objects.
"""

from .filesystem import FilesystemGenerator
from .code_execution import CodeExecutionGenerator
from .network import NetworkGenerator
from .authentication import AuthenticationGenerator
from .financial import FinancialGenerator
from .sysadmin import SysadminGenerator
from .email import EmailGenerator
from .adversarial import AdversarialGenerator
from .injection import InjectionGenerator
from .jailbreak_patterns import JailbreakPatternGenerator, generate_jailbreak_samples

# Registry of all generators (Stage 2 - ToolCallSample generators)
GENERATORS = {
    "filesystem": FilesystemGenerator,
    "code_execution": CodeExecutionGenerator,
    "network": NetworkGenerator,
    "authentication": AuthenticationGenerator,
    "financial": FinancialGenerator,
    "sysadmin": SysadminGenerator,
    "email": EmailGenerator,
    "adversarial": AdversarialGenerator,
    "injection": InjectionGenerator,
}


def get_generator(name: str, **kwargs):
    """Get a generator instance by name."""
    if name not in GENERATORS:
        raise ValueError(
            f"Unknown generator: {name}. Available: {list(GENERATORS.keys())}"
        )
    return GENERATORS[name](**kwargs)


def get_all_generators(**kwargs):
    """Get instances of all generators."""
    return {name: cls(**kwargs) for name, cls in GENERATORS.items()}


__all__ = [
    "FilesystemGenerator",
    "CodeExecutionGenerator",
    "NetworkGenerator",
    "AuthenticationGenerator",
    "FinancialGenerator",
    "SysadminGenerator",
    "EmailGenerator",
    "AdversarialGenerator",
    "InjectionGenerator",
    "JailbreakPatternGenerator",
    "generate_jailbreak_samples",
    "GENERATORS",
    "get_generator",
    "get_all_generators",
]
