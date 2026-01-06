"""
Model backend implementations for LLM Katan

Supports HuggingFace transformers and optionally vLLM for efficient inference.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional

from .config import ServerConfig

logger = logging.getLogger(__name__)


class ModelBackend(ABC):
    """Abstract base class for model backends"""

    def __init__(self, config: ServerConfig):
        self.config = config

    @abstractmethod
    async def load_model(self) -> None:
        """Load the model"""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> AsyncGenerator[Dict, None]:
        """Generate response from messages"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        pass


class TransformersBackend(ModelBackend):
    """HuggingFace Transformers backend"""

    def __init__(self, config: ServerConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    async def load_model(self) -> None:
        """Load model using HuggingFace transformers"""
        logger.info(f"Loading model {self.config.model_name} with transformers backend")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for TransformersBackend. "
                "Install with: pip install transformers torch"
            ) from e

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        device = self.config.device_auto
        is_gpu_device = device in ["xpu", "cuda"]
        torch_dtype = torch.float16 if is_gpu_device else torch.float32
        if device == "xpu":
            device_map = f"xpu:0"
        else:
            device_map = ("auto" if device == "cuda" else None,)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        if device == "cpu":
            self.model = self.model.to("cpu")

            # Apply quantization for faster CPU inference (2-4x speedup)
            if self.config.quantize:
                logger.info("Applying int8 quantization for CPU optimization...")
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info(
                        "✓ Quantization applied (2-4x faster inference, 4x less memory)"
                    )
                except RuntimeError as e:
                    if "NoQEngine" in str(e):
                        logger.warning(
                            "⚠️  Quantization not supported on this platform - "
                            "continuing with full precision"
                        )
                        logger.info(
                            "Note: PyTorch quantization requires specific CPU features. "
                            "Your model will run without quantization."
                        )
                    else:
                        raise
            else:
                logger.info("Quantization disabled - using full precision (slower)")

        logger.info(f"Model loaded successfully on {device}")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> AsyncGenerator[Dict, None]:
        """Generate response using transformers"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_tokens = max_tokens or self.config.max_tokens
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        if self.config.device_auto in ["cuda", "xpu"]:
            inputs = {k: v.to(self.config.device_auto) for k, v in inputs.items()}

        # Generate in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self._generate_sync, inputs, max_tokens, temperature
        )

        # Calculate token usage
        prompt_tokens = len(inputs["input_ids"][0])
        completion_tokens = len(response) - prompt_tokens
        total_tokens = prompt_tokens + completion_tokens

        # Decode response
        full_response = self.tokenizer.decode(response, skip_special_tokens=True)
        generated_text = full_response[len(prompt) :].strip()

        # Create response in OpenAI format
        response_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.served_model_name,
            "system_fingerprint": "llm-katan-transformers",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }

        # Add token_usage as alias for better SDK compatibility
        response_data["token_usage"] = response_data["usage"]

        if stream:
            # For streaming, yield chunks
            words = generated_text.split()
            for i, word in enumerate(words):
                chunk = {
                    "id": response_data["id"],
                    "object": "chat.completion.chunk",
                    "created": response_data["created"],
                    "model": self.config.served_model_name,
                    "system_fingerprint": "llm-katan-transformers",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": word + " " if i < len(words) - 1 else word
                            },
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield chunk
                await asyncio.sleep(0.05)  # Simulate streaming delay

            # Final chunk
            final_chunk = {
                "id": response_data["id"],
                "object": "chat.completion.chunk",
                "created": response_data["created"],
                "model": self.config.served_model_name,
                "system_fingerprint": "llm-katan-transformers",
                "choices": [
                    {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
            yield final_chunk
        else:
            yield response_data

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI messages format to prompt string"""
        # Simple prompt format - can be enhanced for specific models
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt

    def _generate_sync(self, inputs, max_tokens: int, temperature: float):
        """Synchronous generation for executor"""
        import torch

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return output[0]

    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            "id": self.config.served_model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llm-katan",
            "permission": [],
            "root": self.config.served_model_name,
            "parent": None,
        }


class VLLMBackend(ModelBackend):
    """vLLM backend for efficient inference"""

    def __init__(self, config: ServerConfig):
        super().__init__(config)
        self.engine = None

    async def load_model(self) -> None:
        """Load model using vLLM"""
        logger.info(f"Loading model {self.config.model_name} with vLLM backend")

        try:
            from vllm import LLM
            from vllm.sampling_params import SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is required for VLLMBackend. Install with: pip install vllm"
            ) from e

        # Load model with vLLM
        self.engine = LLM(
            model=self.config.model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
        )
        logger.info("vLLM model loaded successfully")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> AsyncGenerator[Dict, None]:
        """Generate response using vLLM"""
        if self.engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from vllm.sampling_params import SamplingParams

        max_tokens = max_tokens or self.config.max_tokens
        temperature = (
            temperature if temperature is not None else self.config.temperature
        )

        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, stop=["User:", "System:"]
        )

        # Generate
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None, self.engine.generate, [prompt], sampling_params
        )

        output = outputs[0]
        generated_text = output.outputs[0].text.strip()

        # Create response in OpenAI format
        response_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.served_model_name,
            "system_fingerprint": "llm-katan-vllm",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids)
                + len(output.outputs[0].token_ids),
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }

        # Add token_usage as alias for better SDK compatibility
        response_data["token_usage"] = response_data["usage"]

        if stream:
            # For streaming, yield chunks (simplified for now)
            words = generated_text.split()
            for i, word in enumerate(words):
                chunk = {
                    "id": response_data["id"],
                    "object": "chat.completion.chunk",
                    "created": response_data["created"],
                    "model": self.config.served_model_name,
                    "system_fingerprint": "llm-katan-vllm",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": word + " " if i < len(words) - 1 else word
                            },
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield chunk
                await asyncio.sleep(0.05)

            # Final chunk
            final_chunk = {
                "id": response_data["id"],
                "object": "chat.completion.chunk",
                "created": response_data["created"],
                "model": self.config.served_model_name,
                "system_fingerprint": "llm-katan-vllm",
                "choices": [
                    {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids)
                    + len(output.outputs[0].token_ids),
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
            yield final_chunk
        else:
            yield response_data

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI messages format to prompt string"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt

    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            "id": self.config.served_model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llm-katan",
            "permission": [],
            "root": self.config.served_model_name,
            "parent": None,
        }


def create_backend(config: ServerConfig) -> ModelBackend:
    """Factory function to create the appropriate backend"""
    if config.backend == "vllm":
        return VLLMBackend(config)
    elif config.backend == "transformers":
        return TransformersBackend(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
