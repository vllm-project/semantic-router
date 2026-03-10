#!/usr/bin/env python3
"""
Router-R1 LLM-as-Router Server
Based on arXiv:2506.09033 - Router-R1: Teaching LLMs Multi-Round Routing and Aggregation

This server runs a small LLM to make routing decisions using the Router-R1 pattern:
1. <think>...</think> - Internal deliberation about which model to use
2. <search>LLM: query</search> - Route to a specific LLM

The server exposes a simple HTTP API for VSR to call.

Usage:
    python router_r1_server.py --port 8888 --model "microsoft/phi-2"
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.error("transformers not installed")


@dataclass
class LLMDescriptor:
    """Descriptor for an LLM in the routing pool."""

    name: str
    description: str
    cost_per_1m_tokens: float
    latency_ms: float
    capabilities: List[str]


class RouterR1Engine:
    """
    Router-R1 engine that uses an LLM to make routing decisions.

    Implements the think/route action pattern from the paper:
    - <think>reason about which model to use</think>
    - <search>model_name: query</search>
    """

    SYSTEM_PROMPT = """You are a routing agent that decides which LLM to use for a given query.
You have access to the following LLMs:

{llm_descriptions}

For each query, you must:
1. Think about which model would be best (wrap in <think>...</think>)
2. Route to a specific model (use <search>MODEL_NAME: query</search>)

Only route to ONE model. Choose based on:
- Query complexity (harder queries -> larger models)
- Cost efficiency (simpler queries -> cheaper models)
- Model capabilities (match query type to model strengths)

Example:
User: Write a haiku about nature
<think>This is a simple creative writing task. A smaller model like phi-2 can handle this well.</think>
<search>phi-2: Write a haiku about nature</search>

User: Explain the P vs NP problem and its implications for cryptography
<think>This requires deep technical knowledge. I should use a larger, more capable model.</think>
<search>gpt-4: Explain the P vs NP problem and its implications for cryptography</search>
"""

    def __init__(self, model_name: str = "microsoft/phi-2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.llm_pool: Dict[str, LLMDescriptor] = {}

        if HAS_TRANSFORMERS:
            logger.info(f"Loading router model: {model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Router model loaded successfully")
        else:
            self.tokenizer = None
            self.model = None

    def register_llm(self, descriptor: LLMDescriptor):
        """Register an LLM in the routing pool."""
        self.llm_pool[descriptor.name] = descriptor
        logger.info(f"Registered LLM: {descriptor.name}")

    def _build_llm_descriptions(self) -> str:
        """Build formatted descriptions of all available LLMs."""
        descriptions = []
        for name, desc in self.llm_pool.items():
            cap_str = ", ".join(desc.capabilities)
            descriptions.append(
                f"- {name}: {desc.description}\n"
                f"  Cost: ${desc.cost_per_1m_tokens}/1M tokens, Latency: {desc.latency_ms}ms\n"
                f"  Capabilities: {cap_str}"
            )
        return "\n".join(descriptions)

    def route(self, query: str) -> Tuple[str, str, str]:
        """
        Route a query to the best LLM using the think/route pattern.

        Returns:
            (selected_model, thinking, full_response)
        """
        if not self.model:
            # Fallback: simple heuristic routing
            return self._heuristic_route(query)

        # Build prompt
        system = self.SYSTEM_PROMPT.format(
            llm_descriptions=self._build_llm_descriptions()
        )
        prompt = f"{system}\n\nUser: {query}\n"

        # Generate routing decision
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt) :]  # Remove the prompt

        # Parse think/route pattern
        thinking, selected_model = self._parse_response(response)

        # Fallback if parsing failed
        if not selected_model and self.llm_pool:
            selected_model = list(self.llm_pool.keys())[0]
            thinking = f"Fallback to {selected_model} (parsing failed)"

        logger.info(f"Routed query to: {selected_model}")
        return selected_model, thinking, response

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the think/route pattern from the response."""
        thinking = ""
        selected_model = ""

        # Extract thinking
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()

        # Extract routing decision
        search_match = re.search(r"<search>([^:]+):", response)
        if search_match:
            selected_model = search_match.group(1).strip()
            # Normalize model name
            for pool_name in self.llm_pool:
                if pool_name.lower() in selected_model.lower():
                    selected_model = pool_name
                    break

        return thinking, selected_model

    def _heuristic_route(self, query: str) -> Tuple[str, str, str]:
        """Simple heuristic routing when LLM is not available."""
        query_lower = query.lower()

        # Simple heuristics
        if any(
            word in query_lower for word in ["code", "program", "function", "algorithm"]
        ):
            for model in ["gpt-4", "claude-3-opus", "phi-4"]:
                if model in self.llm_pool:
                    return (
                        model,
                        "Coding query -> capable model",
                        f"<think>Coding task</think><search>{model}: {query}</search>",
                    )

        if len(query.split()) < 10:
            for model in ["phi-2", "llama3.2:3b", "gpt-3.5-turbo"]:
                if model in self.llm_pool:
                    return (
                        model,
                        "Short query -> fast model",
                        f"<think>Simple query</think><search>{model}: {query}</search>",
                    )

        # Default to first model
        if self.llm_pool:
            model = list(self.llm_pool.keys())[0]
            return (
                model,
                "Default routing",
                f"<think>Default</think><search>{model}: {query}</search>",
            )

        return "", "No models registered", ""


class RouterR1Handler(BaseHTTPRequestHandler):
    """HTTP handler for Router-R1 requests."""

    engine: RouterR1Engine = None  # Set by main

    def do_POST(self):
        if self.path == "/route":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                query = data.get("query", "")

                if not query:
                    self._send_error(400, "Missing 'query' field")
                    return

                # Route the query
                selected_model, thinking, full_response = self.engine.route(query)

                response = {
                    "selected_model": selected_model,
                    "thinking": thinking,
                    "full_response": full_response,
                }

                self._send_json(200, response)

            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
            except Exception as e:
                logger.exception("Error processing request")
                self._send_error(500, str(e))
        else:
            self._send_error(404, "Not found")

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": self.engine.model_name})
        elif self.path == "/models":
            models = [
                {"name": k, "description": v.description}
                for k, v in self.engine.llm_pool.items()
            ]
            self._send_json(200, {"models": models})
        else:
            self._send_error(404, "Not found")

    def _send_json(self, status: int, data: dict):
        response = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response)

    def _send_error(self, status: int, message: str):
        self._send_json(status, {"error": message})

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="Router-R1 LLM-as-Router Server")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    parser.add_argument(
        "--model", type=str, default="microsoft/phi-2", help="Router model"
    )
    parser.add_argument("--test", action="store_true", help="Run test routing")
    args = parser.parse_args()

    # Initialize engine
    engine = RouterR1Engine(model_name=args.model)

    # Register default LLM pool
    engine.register_llm(
        LLMDescriptor(
            name="gpt-4",
            description="Most capable OpenAI model for complex reasoning",
            cost_per_1m_tokens=30.0,
            latency_ms=2000,
            capabilities=["reasoning", "coding", "math", "creative_writing"],
        )
    )
    engine.register_llm(
        LLMDescriptor(
            name="gpt-3.5-turbo",
            description="Fast and cheap OpenAI model for simple tasks",
            cost_per_1m_tokens=1.0,
            latency_ms=500,
            capabilities=["general", "simple_coding", "chat"],
        )
    )
    engine.register_llm(
        LLMDescriptor(
            name="phi-2",
            description="Compact 2.7B parameter model, efficient for simple tasks",
            cost_per_1m_tokens=0.1,
            latency_ms=100,
            capabilities=["simple_qa", "basic_coding"],
        )
    )
    engine.register_llm(
        LLMDescriptor(
            name="llama3.2:3b",
            description="Meta's efficient 3B model",
            cost_per_1m_tokens=0.2,
            latency_ms=150,
            capabilities=["general", "chat", "basic_reasoning"],
        )
    )

    if args.test:
        # Test routing
        queries = [
            "What is 2+2?",
            "Write a Python function to implement quicksort",
            "Explain the theory of relativity in detail",
        ]
        for query in queries:
            model, thinking, _ = engine.route(query)
            print(f"\nQuery: {query}")
            print(f"Thinking: {thinking}")
            print(f"Selected: {model}")
        return

    # Start server
    RouterR1Handler.engine = engine
    server = HTTPServer(("0.0.0.0", args.port), RouterR1Handler)
    logger.info(f"Router-R1 server listening on port {args.port}")
    logger.info(f"Using model: {args.model}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
