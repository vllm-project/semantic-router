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
            device_map = "xpu:0"
        else:
            device_map = "auto" if device == "cuda" else None
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


class EchoBackend(ModelBackend):
    """
    Echo backend - returns the full prompt for testing.

    This backend is useful for integration tests where you need to verify
    that content (like memory context) was properly injected into the prompt.
    The response contains all messages (system, user, assistant) that were sent.

    Special handling for memory extraction:
    - Detects extraction prompts (containing "Extract important information")
    - Returns valid JSON facts based on keyword matching in the conversation
    - This allows testing the full memory extraction → storage → retrieval flow

    Usage:
        llm-katan --backend echo --served-model-name test-model
    """

    # Keywords to extract as facts (keyword -> fact template)
    # Format: "User's X is Y" to match query rewrites like "User's X"
    # Note: Include common nouns (dog, car) as well as specific values (max, tesla)
    EXTRACTION_KEYWORDS = {
        # Car-related facts
        "car": ("semantic", "User's car is a blue Tesla Model 3 from 2023"),
        "tesla": ("semantic", "User's car is a blue Tesla Model 3 from 2023"),
        "model 3": ("semantic", "User's car is a blue Tesla Model 3 from 2023"),
        "drive": ("semantic", "User's car is a blue Tesla Model 3 from 2023"),
        # Dog-related facts
        "dog": ("semantic", "User's dog's name is Max, a golden retriever"),
        "max": ("semantic", "User's dog's name is Max, a golden retriever"),
        "golden retriever": (
            "semantic",
            "User's dog's name is Max, a golden retriever",
        ),
        "pet": ("semantic", "User's dog's name is Max, a golden retriever"),
        # Project codename
        "phoenix": ("semantic", "User's secret project codename is Phoenix-2026"),
        "phoenix-2026": ("semantic", "User's secret project codename is Phoenix-2026"),
        "codename": ("semantic", "User's secret project codename is Phoenix-2026"),
        # Other personal facts
        "purple": ("semantic", "User's favorite color is purple"),
        "color": ("semantic", "User's favorite color is purple"),
        "google": ("semantic", "User works as a software engineer at Google"),
        "engineer": ("semantic", "User works as a software engineer at Google"),
        "japan": (
            "semantic",
            "User is planning a trip to Japan with a budget of $5000",
        ),
        "trip": ("semantic", "User is planning a trip to Japan with a budget of $5000"),
        "$5000": ("semantic", "User's budget is $5000"),
        "5000": ("semantic", "User's budget is $5000"),
        "budget": ("semantic", "User's budget is $5000"),
        "sarah": ("semantic", "User's friend Sarah is a doctor who lives in Boston"),
        "doctor": ("semantic", "User's friend Sarah is a doctor who lives in Boston"),
        "boston": ("semantic", "User's friend Sarah lives in Boston"),
        "friend": ("semantic", "User's friend Sarah is a doctor who lives in Boston"),
        "mit": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "college": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "university": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "graduated": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "computer science": (
            "semantic",
            "User graduated from MIT in 2020 with Computer Science degree",
        ),
        "italian place": (
            "semantic",
            "User's favorite restaurant is The Italian Place on 5th Avenue",
        ),
        "5th avenue": (
            "semantic",
            "User's favorite restaurant is The Italian Place on 5th Avenue",
        ),
        "tom": ("semantic", "User's brother is named Tom"),
        "anna": ("semantic", "Tom is getting married to Anna next spring"),
        "macbook": ("semantic", "User bought a MacBook Pro M3"),
        "m3": ("semantic", "User bought a MacBook Pro M3"),
        "sushi": ("episodic", "User had lunch at a sushi place downtown"),
        # User isolation tests - PIN and password
        "pin": ("semantic", "User's secret PIN is 9876"),
        "9876": ("semantic", "User's secret PIN is 9876"),
        "password": ("semantic", "User's password is hunter2"),
        "hunter2": ("semantic", "User's password is hunter2"),
        # Address tests
        "123 main": ("semantic", "User's home address is 123 Main Street, New York"),
        "main street": ("semantic", "User's home address is 123 Main Street, New York"),
        "home address": (
            "semantic",
            "User's home address is 123 Main Street, New York",
        ),
        "456 business": ("semantic", "User's work address is 456 Business Ave, Boston"),
        "business ave": ("semantic", "User's work address is 456 Business Ave, Boston"),
        "work address": ("semantic", "User's work address is 456 Business Ave, Boston"),
        # Deduplication test
        "phone": ("semantic", "User's phone number is 555-123-4567"),
        "555-123-4567": ("semantic", "User's phone number is 555-123-4567"),
        "phone number": ("semantic", "User's phone number is 555-123-4567"),
        # Wedding/multi-turn test
        "wedding": (
            "semantic",
            "User's wedding is on June 15th, 2026 with fiancée Emily",
        ),
        "june 15": ("semantic", "User's wedding is on June 15th, 2026"),
        "june": ("semantic", "User's wedding is on June 15th, 2026"),
        "2026": ("semantic", "User's wedding is on June 15th, 2026"),
        "emily": ("semantic", "User's fiancée is named Emily"),
        "fiancée": ("semantic", "User's fiancée is named Emily"),
        "beach": ("semantic", "User is having a beach wedding with 150 guests"),
        "venue": ("semantic", "User is having a beach wedding with 150 guests"),
        "beach venue": ("semantic", "User is having a beach wedding with 150 guests"),
        "150 people": ("semantic", "User is having 150 guests at the wedding"),
        "150 guests": ("semantic", "User is having 150 guests at the wedding"),
        "150": ("semantic", "User is having 150 guests at the wedding"),
        "guests": ("semantic", "User is having 150 guests at the wedding"),
        "$50,000": ("semantic", "User's wedding budget is $50,000"),
        "50000": ("semantic", "User's wedding budget is $50,000"),
    }

    def __init__(self, config: ServerConfig):
        super().__init__(config)

    async def load_model(self) -> None:
        """No model to load for echo backend"""
        logger.info("🔊 Echo backend ready (no model to load)")
        logger.info("   Smart extraction: enabled (detects extraction prompts)")

    def _is_extraction_prompt(self, messages: List[Dict[str, str]]) -> bool:
        """Check if this is a memory extraction prompt."""
        for msg in messages:
            content = msg.get("content", "").lower()
            if "extract important information" in content:
                return True
            if "memory extraction system" in content:
                return True
        return False

    def _is_query_rewrite_prompt(self, messages: List[Dict[str, str]]) -> bool:
        """Check if this is a query rewriting prompt."""
        for msg in messages:
            content = msg.get("content", "").lower()
            if "query rewriter" in content:
                return True
        return False

    def _extract_facts_from_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict]:
        """Extract facts from conversation using keyword matching.

        IMPORTANT: Only searches USER messages, not system prompts.
        The extraction system prompt contains example keywords (e.g., 'budget is $5000', 'MIT')
        that would cause false matches if we searched system messages too.
        """
        # Only look at user messages - NOT system messages which contain examples
        # that would trigger false keyword matches
        user_content = " ".join(
            msg.get("content", "") for msg in messages if msg.get("role") == "user"
        ).lower()

        facts = []
        seen_facts = set()  # Avoid duplicates

        for keyword, (fact_type, fact_content) in self.EXTRACTION_KEYWORDS.items():
            if keyword in user_content and fact_content not in seen_facts:
                facts.append({"type": fact_type, "content": fact_content})
                seen_facts.add(fact_content)

        logger.info(
            f"🧠 Smart extraction: found {len(facts)} facts from {len([m for m in messages if m.get('role') == 'user'])} user messages"
        )
        return facts

    def _rewrite_query(self, messages: List[Dict[str, str]]) -> str:
        """Query rewriting - enrich query with conversation context.

        Production behavior (from queryRewriteSystemPrompt):
        - PRESERVE query type: questions stay questions, statements stay statements
        - Add context from conversation history
        - Make query self-contained for semantic search

        The router sends query rewriting requests with format:
        [system]: You are a query rewriter...
        [user]: History:
               [user]: <previous messages>
               Query: <the actual query>
               Rewritten query:

        Example:
        - History: "My budget is $50K for a trip to Israel"
        - Query: "Which hotel should I choose?"
        - Rewritten: "Which hotel should I choose for my trip to Israel with $50K budget?"
        """
        query = ""
        history_context = []

        # Parse the user message to extract History and Query
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            lines = content.split("\n")

            in_history = False
            for line in lines:
                stripped = line.strip()

                # Start of history section
                if stripped.startswith("History:"):
                    in_history = True
                    continue

                # Query line - end of history
                if stripped.startswith("Query:") and "Rewritten" not in stripped:
                    query = stripped.replace("Query:", "").strip()
                    in_history = False
                    continue

                # Collect history context (look for user messages)
                if in_history and "[user]:" in stripped:
                    # Extract the user's message from history
                    user_msg = stripped.split("[user]:", 1)[-1].strip()
                    if user_msg:
                        history_context.append(user_msg)

            if query:
                break

        # Fallback: return last user message content
        if not query:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    query = msg.get("content", "").split("\n")[0].strip()
                    break

        if not query:
            return ""

        # If no history context, return query unchanged
        if not history_context:
            logger.info(f"🔄 Query unchanged (no history): '{query}'")
            return query

        # Simple context enrichment: append relevant context to the query
        # Keep the query as-is (preserve type) and add context
        context_summary = " ".join(history_context[-2:])  # Use last 2 history items

        # Only add context if query seems to need it (short or has pronouns)
        if len(query.split()) < 6 or any(
            word in query.lower() for word in ["it", "that", "this", "my"]
        ):
            # Append context in parentheses to keep query structure intact
            rewritten = f"{query} (context: {context_summary})"
            logger.info(f"🔄 Enriched query: '{query}' → '{rewritten}'")
            return rewritten

        # Query is already self-contained
        logger.info(f"🔄 Query unchanged (self-contained): '{query}'")
        return query

        # Pattern: "When is my X?" → "User's X is" (e.g., "When is my wedding?" → "User's wedding is")
        match = re.match(r"(?:When is my )(.+?)[\?\.]*$", query, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            rewritten = f"User's {topic} is"
            logger.info(f"🔄 Rewrote query: '{query}' → '{rewritten}'")
            return rewritten

        # Pattern: "Where did I go to X?" → "User graduated from" or "User's X"
        # e.g., "Where did I go to college?" → "User graduated from"
        match = re.match(
            r"(?:Where did I (?:go to|attend) )(.+?)[\?\.]*$", query, re.IGNORECASE
        )
        if match:
            topic = match.group(1).strip()
            if (
                "college" in topic.lower()
                or "university" in topic.lower()
                or "school" in topic.lower()
            ):
                rewritten = "User graduated from"
            else:
                rewritten = f"User's {topic}"
            logger.info(f"🔄 Rewrote query: '{query}' → '{rewritten}'")
            return rewritten

        # Pattern: "Where does X live?" → "X lives in" (e.g., "Where does she live?" → "lives in")
        # For pronouns, we return a generic location query
        match = re.match(
            r"(?:Where does (?:she|he|my friend|sarah) live)[\?\.]*$",
            query,
            re.IGNORECASE,
        )
        if match:
            rewritten = "lives in Boston"  # Return something that matches stored facts
            logger.info(f"🔄 Rewrote query: '{query}' → '{rewritten}'")
            return rewritten

        # Pattern: "How much can I spend on X?" → "User's budget is"
        match = re.match(
            r"(?:How much (?:can I|do I|should I) (?:spend|budget|afford)).*[\?\.]*$",
            query,
            re.IGNORECASE,
        )
        if match:
            rewritten = "User's budget is"
            logger.info(f"🔄 Rewrote query: '{query}' → '{rewritten}'")
            return rewritten

        # Pattern: "How many X?" → "User's X" or specific patterns
        # e.g., "How many guests are we expecting?" → "guests"
        match = re.match(
            r"(?:How many )(.+?)(?:are we|do we|will we|expecting).*[\?\.]*$",
            query,
            re.IGNORECASE,
        )
        if match:
            topic = match.group(1).strip()
            rewritten = f"{topic}"  # Just the topic for semantic matching
            logger.info(f"🔄 Rewrote query: '{query}' → '{rewritten}'")
            return rewritten

        # No rewrite needed
        return query

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> AsyncGenerator[Dict, None]:
        """Echo back all messages - with smart handling for extraction prompts."""
        import json

        # Check for special prompt types
        if self._is_extraction_prompt(messages):
            # Return valid JSON facts for extraction
            facts = self._extract_facts_from_messages(messages)
            echo_content = json.dumps(facts)
            logger.info(f"📝 Extraction response: {echo_content[:100]}...")
        elif self._is_query_rewrite_prompt(messages):
            # Return the query unchanged (simple passthrough)
            echo_content = self._rewrite_query(messages)
            logger.info(f"🔄 Query rewrite response: {echo_content[:50]}...")
        else:
            # Standard echo: format all messages
            echo_parts = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                echo_parts.append(f"[{role}]: {content}")
            echo_content = "\n".join(echo_parts)

        # Calculate rough token count
        token_count = len(echo_content.split())

        response_data = {
            "id": f"echo-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.served_model_name,
            "system_fingerprint": "llm-katan-echo",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": echo_content},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": token_count,
                "completion_tokens": token_count,
                "total_tokens": token_count * 2,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }

        # Add token_usage as alias for better SDK compatibility
        response_data["token_usage"] = response_data["usage"]

        if stream:
            # For streaming, yield the content in chunks
            words = echo_content.split()
            for i, word in enumerate(words):
                chunk = {
                    "id": response_data["id"],
                    "object": "chat.completion.chunk",
                    "created": response_data["created"],
                    "model": self.config.served_model_name,
                    "system_fingerprint": "llm-katan-echo",
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
                await asyncio.sleep(0.01)  # Fast streaming for tests

            # Final chunk
            final_chunk = {
                "id": response_data["id"],
                "object": "chat.completion.chunk",
                "created": response_data["created"],
                "model": self.config.served_model_name,
                "system_fingerprint": "llm-katan-echo",
                "choices": [
                    {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}
                ],
                "usage": response_data["usage"],
            }
            yield final_chunk
        else:
            yield response_data

    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            "id": self.config.served_model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llm-katan-echo",
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
    elif config.backend == "echo":
        return EchoBackend(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
