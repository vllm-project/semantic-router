"""
Dataset Loaders for HuggingFace and Local Sources.

Each loader is responsible for a specific dataset source.
All loaders return lists of SentinelSample or ToolCallSample objects.

Updated 2024-2025 with new security research datasets:
- xstest (edge cases)
- AdvBench (adversarial behaviors)
- BeaverTails (helpful vs harmful)
- SimpleSafetyTests
- MCP Attack Patterns (Tool Poisoning, Shadowing, Rug Pulls)
"""

from .huggingface import (
    HuggingFaceLoader,
    # Stage 1 loaders (Sentinel - prompt classification)
    load_jailbreak_llms,
    load_wildjailbreak,
    load_hackaprompt,
    load_jailbreakbench,
    load_chatgpt_jailbreaks,
    load_toxic_chat,
    load_alpaca_benign,
    load_dolly_benign,
    # NEW: 2024-2025 datasets
    load_xstest,
    load_advbench,
    load_beaver_tails,
    load_simple_safety_tests,
    generate_mcp_attack_samples,
    # Stage 2 loaders (Tool Call Verifier)
    load_llmail_inject,
    load_wildjailbreak_tool_calls,
    load_hackaprompt_tool_calls,
    load_jailbreakbench_tool_calls,
    load_chatgpt_jailbreaks_tool_calls,
    # NEW: MCP tool-call attacks
    generate_mcp_tool_call_samples,
)

# Registry of Stage 1 loaders (SentinelSample)
STAGE1_LOADERS = {
    "jailbreak_llms": load_jailbreak_llms,
    "wildjailbreak": load_wildjailbreak,
    "hackaprompt": load_hackaprompt,
    "jailbreakbench": load_jailbreakbench,
    "chatgpt_jailbreaks": load_chatgpt_jailbreaks,
    "toxic_chat": load_toxic_chat,
    "alpaca": load_alpaca_benign,
    "dolly": load_dolly_benign,
    # NEW: 2024-2025 Security Research
    "xstest": load_xstest,
    "advbench": load_advbench,
    "beaver_tails": load_beaver_tails,
    "simple_safety_tests": load_simple_safety_tests,
    "mcp_attacks": generate_mcp_attack_samples,
}

# Registry of Stage 2 loaders (ToolCallSample)
STAGE2_LOADERS = {
    "llmail_inject": load_llmail_inject,
    "wildjailbreak_toolcall": load_wildjailbreak_tool_calls,
    "hackaprompt_toolcall": load_hackaprompt_tool_calls,
    "jailbreakbench_toolcall": load_jailbreakbench_tool_calls,
    "chatgpt_jailbreaks_toolcall": load_chatgpt_jailbreaks_tool_calls,
    # NEW: MCP attacks
    "mcp_attacks_toolcall": generate_mcp_tool_call_samples,
}

# Combined registry of all loaders
LOADERS = {**STAGE1_LOADERS, **STAGE2_LOADERS}


def get_loader(name: str):
    """Get a loader function by name."""
    if name not in LOADERS:
        raise ValueError(f"Unknown loader: {name}. Available: {list(LOADERS.keys())}")
    return LOADERS[name]


__all__ = [
    "HuggingFaceLoader",
    # Stage 1
    "load_jailbreak_llms",
    "load_wildjailbreak",
    "load_hackaprompt",
    "load_jailbreakbench",
    "load_chatgpt_jailbreaks",
    "load_toxic_chat",
    "load_alpaca_benign",
    "load_dolly_benign",
    # NEW: 2024-2025
    "load_xstest",
    "load_advbench",
    "load_beaver_tails",
    "load_simple_safety_tests",
    "generate_mcp_attack_samples",
    # Stage 2
    "load_llmail_inject",
    "load_wildjailbreak_tool_calls",
    "load_hackaprompt_tool_calls",
    "load_jailbreakbench_tool_calls",
    "load_chatgpt_jailbreaks_tool_calls",
    "generate_mcp_tool_call_samples",
    # Registries
    "LOADERS",
    "STAGE1_LOADERS",
    "STAGE2_LOADERS",
    "get_loader",
]
