"""Tokenizer-only cost probe for the prospective 0.6B option-isolation path.

Run in a private CPU container with an already cached, pinned official Qwen
source.  It does not load weights, infer, train, or read any benchmark labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .option_isolation import token_work

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd4"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
TOKENIZER_CONFIG_SHA256 = (
    "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"
)
CONFIG_SHA256 = "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59"
CONTEXT_CAP = 8192  # Existing Decision 2.0 0.6B native complete-input cap.


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(task_type: str, count: int, state_words: int) -> dict[str, Any]:
    clause = "Shipment evidence arrives on Tuesday under the revised policy."
    state = " ".join([clause] * ((state_words + 9) // 10))
    if task_type == "noul":
        options = [
            {"key": "false", "description": "Evidence does not establish the claim"},
            {"key": "true", "description": "Evidence establishes the claim"},
        ]
    else:
        options = [
            {
                "key": str(index),
                "description": f"Action {index}: route using the relevant evidence and exceptions.",
            }
            for index in range(count)
        ]
    return {
        "state": {"document": state, "case_id": "synthetic-cost-only"},
        "instructions": "Choose the best supported action under the current policy.",
        "options": options,
        "task_type": task_type,
    }


def measure(tokenizer: Any, config: Any) -> dict[str, Any]:
    layers = config.num_hidden_layers
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * 2  # K/V, BF16.
    scenarios = []
    for state_words in (64, 512, 4096):
        for task_type, counts in (
            ("choice", (2, 3, 10, 255)),
            ("noul", (2,)),
            ("score", (2, 3, 10)),
        ):
            for count in counts:
                work = token_work(fixture(task_type, count, state_words), tokenizer)
                reference = work["reference_tokens"]
                no_cache = work["independent_total_tokens"]
                shared = work["ideal_shared_prefix_tokens"]
                prefix = work["shared_prefix_tokens"]
                tails = work["independent_tail_tokens"]
                baseline_attention_pairs = reference * (reference + 1) // 2
                no_cache_attention_pairs = sum(
                    (prefix + tail) * (prefix + tail + 1) // 2 for tail in tails
                )
                shared_attention_pairs = prefix * (prefix + 1) // 2 + sum(
                    tail * prefix + tail * (tail + 1) // 2 for tail in tails
                )
                scenarios.append(
                    {
                        "task_type": task_type,
                        "candidate_count": count,
                        "synthetic_state_words": state_words,
                        "reference_tokens": reference,
                        "independent_total_tokens": no_cache,
                        "independent_shared_prefix_bound_tokens": shared,
                        "independent_max_branch_tokens": work[
                            "independent_longest_prompt_tokens"
                        ],
                        "shared_prefix_tokens": prefix,
                        "no_cache_token_multiplier": round(no_cache / reference, 3),
                        "shared_prefix_token_bound_multiplier": round(
                            shared / reference, 3
                        ),
                        "no_cache_attention_pair_multiplier": round(
                            no_cache_attention_pairs / baseline_attention_pairs, 3
                        ),
                        "shared_prefix_attention_pair_bound_multiplier": round(
                            shared_attention_pairs / baseline_attention_pairs, 3
                        ),
                        "reference_exceeds_current_cap": reference > CONTEXT_CAP,
                        "independent_branch_exceeds_current_cap": (
                            work["independent_longest_prompt_tokens"] > CONTEXT_CAP
                        ),
                        "bf16_reference_kv_mib": round(
                            reference * kv_bytes_per_token / 2**20, 1
                        ),
                        "bf16_independent_batched_no_cache_kv_mib": round(
                            no_cache * kv_bytes_per_token / 2**20, 1
                        ),
                        "bf16_independent_sequential_peak_kv_mib": round(
                            work["independent_longest_prompt_tokens"]
                            * kv_bytes_per_token
                            / 2**20,
                            1,
                        ),
                        "bf16_independent_shared_prefix_kv_bound_mib": round(
                            shared * kv_bytes_per_token / 2**20, 1
                        ),
                    }
                )
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "config_sha256": CONFIG_SHA256,
        "context_cap": CONTEXT_CAP,
        "kv_bytes_per_token_bf16": kv_bytes_per_token,
        "model_shape": {
            "layers": layers,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    for name, expected in (
        ("tokenizer.json", TOKENIZER_SHA256),
        ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256),
        ("config.json", CONFIG_SHA256),
    ):
        if sha256(args.source / name) != expected:
            raise ValueError(f"Pinned official source file changed: {name}")
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.source, local_files_only=True, trust_remote_code=False
    )
    config = AutoConfig.from_pretrained(
        args.source, local_files_only=True, trust_remote_code=False
    )
    if config.model_type != "qwen3":
        raise ValueError("Expected Qwen3 official source")
    print(json.dumps(measure(tokenizer, config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
