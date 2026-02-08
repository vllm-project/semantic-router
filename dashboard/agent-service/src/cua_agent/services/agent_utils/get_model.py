"""Model provider utilities for the Computer Use Agent."""

import os
from typing import List

from smolagents import InferenceClientModel, LiteLLMModel, Model

from cua_agent.services.agent_utils.envoy_model import EnvoyModel

# Available model IDs
AVAILABLE_MODELS: List[str] = [
    # Semantic Router (routes through Envoy to best model)
    "envoy/auto",
    # HuggingFace Vision Models
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    # Ollama Vision Models (local)
    "ollama/qwen3-vl:8b",  # Recommended - best quality for local
    "ollama/qwen2.5vl:7b",
    "ollama/llava:7b",
    "ollama/llava:13b",
    # OpenAI Vision Models
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]


def get_model(model_id: str) -> Model:
    """
    Get the appropriate model based on the model_id.

    Supports:
    - Envoy/Semantic Router (routes through Envoy to best model)
    - HuggingFace Inference API (default for Qwen models)
    - Ollama (local vision models)
    - OpenAI (GPT-4V)
    """
    # Check for Envoy/Semantic Router models
    # These route through Envoy proxy which handles model selection via decision engine
    model_lower = model_id.lower()
    if model_lower in ("envoy/auto", "envoy", "mom", "auto"):
        envoy_url = os.getenv("ENVOY_URL", "http://localhost:8801")
        return EnvoyModel(envoy_url=envoy_url, model_id="MoM")

    # Check for Ollama models
    if model_id.startswith("ollama/"):
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        actual_model = model_id.replace("ollama/", "")
        return LiteLLMModel(
            model_id=f"ollama/{actual_model}",
            api_base=ollama_host,
        )

    # Check for OpenAI models
    if model_id.startswith("openai/"):
        actual_model = model_id.replace("openai/", "")
        return LiteLLMModel(
            model_id=actual_model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    # Default to HuggingFace Inference API
    return InferenceClientModel(model_id=model_id)
