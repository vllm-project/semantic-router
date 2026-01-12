"""
Configuration management for LLM Katan

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServerConfig:
    """Configuration for the LLM Katan server"""

    model_name: str
    served_model_name: Optional[str] = None
    port: int = 8000
    host: str = "0.0.0.0"
    backend: str = "transformers"  # "transformers" or "vllm"
    max_tokens: int = 512
    temperature: float = 0.7
    device: str = "auto"  # "auto", "cpu", "cuda", "xpu"
    quantize: bool = True  # Enable int8 quantization for CPU (default: enabled)

    def __post_init__(self):
        """Post-initialization processing"""
        # If no served model name specified, use the actual model name
        if self.served_model_name is None:
            self.served_model_name = self.model_name

        # Apply environment variable overrides
        self.model_name = os.getenv("YLLM_MODEL", self.model_name)
        self.served_model_name = os.getenv(
            "YLLM_SERVED_MODEL_NAME", self.served_model_name
        )
        self.port = int(os.getenv("YLLM_PORT", str(self.port)))
        self.backend = os.getenv("YLLM_BACKEND", self.backend)
        self.host = os.getenv("YLLM_HOST", self.host)

        # Validate backend
        if self.backend not in ["transformers", "vllm"]:
            raise ValueError(
                f"Invalid backend: {self.backend}. Must be 'transformers' or 'vllm'"
            )

    @property
    def device_auto(self) -> str:
        """Auto-detect the best device"""
        if self.device == "auto":
            try:
                import torch

                if torch.xpu.is_available():
                    return "xpu"
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device
