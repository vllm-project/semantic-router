#!/usr/bin/env python3
"""
Embedding-Based MCP Classification Server with Intelligent Routing

This is an example MCP server that demonstrates:
1. Text classification using semantic embeddings (RAG-style)
2. Dynamic category discovery via list_categories
3. Intelligent routing decisions (model selection and reasoning control)
4. FAISS vector database for similarity search

The server implements two MCP tools:
- 'list_categories': Returns available categories with per-category system prompts and descriptions
- 'classify_text': Classifies text using semantic similarity and returns routing recommendations

Protocol:
- list_categories returns: {
    "categories": ["math", "science", "technology", ...],
    "category_system_prompts": {
      "math": "You are a mathematics expert...",
      ...
    },
    "category_descriptions": {
      "math": "Mathematical and computational queries",
      ...
    }
  }
- classify_text returns: {
    "class": 0,
    "confidence": 0.85,
    "model": "openai/gpt-oss-20b",
    "use_reasoning": true,
    "probabilities": [...]
  }

Usage:
  # Stdio mode (for testing with MCP clients)
  python server_embedding.py

  # HTTP mode (for semantic router)
  python server_embedding.py --http --port 8090
"""

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Category definitions with system prompts
CATEGORY_CONFIG = {
    "math": {
        "description": "Mathematical and computational queries",
        "system_prompt": """You are a mathematics expert. When answering math questions:
- Show step-by-step solutions with clear explanations
- Use proper mathematical notation and terminology
- Verify calculations and provide intermediate steps
- Explain the underlying concepts and principles
- Offer alternative approaches when applicable""",
    },
    "science": {
        "description": "Scientific concepts and queries",
        "system_prompt": """You are a science expert. When answering science questions:
- Provide evidence-based answers grounded in scientific research
- Explain relevant scientific concepts and principles
- Use appropriate scientific terminology
- Cite the scientific method and experimental evidence when relevant
- Distinguish between established facts and current theories""",
    },
    "technology": {
        "description": "Technology and computing topics",
        "system_prompt": """You are a technology expert. When answering tech questions:
- Include practical examples and code snippets when relevant
- Follow best practices and industry standards
- Explain both high-level concepts and implementation details
- Consider security, performance, and maintainability
- Recommend appropriate tools and technologies for the use case""",
    },
    "history": {
        "description": "Historical events and topics",
        "system_prompt": """You are a history expert. When answering historical questions:
- Provide accurate dates, names, and historical context
- Cite time periods and geographical locations
- Explain the causes, events, and consequences
- Consider multiple perspectives and historical interpretations
- Connect historical events to their broader significance""",
    },
    "general": {
        "description": "General questions and topics",
        "system_prompt": """You are a knowledgeable assistant. When answering general questions:
- Provide balanced, well-rounded responses
- Draw from multiple domains of knowledge when relevant
- Be clear, concise, and accurate
- Adapt your explanation to the complexity of the question
- Acknowledge limitations and uncertainties when appropriate""",
    },
}


class EmbeddingClassifier:
    """Embedding-based text classifier using FAISS vector search."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        csv_path: str = "training_data.csv",
        index_path: str = "faiss_index.bin",
        device: str = "auto",
    ):
        """
        Initialize the embedding classifier.

        Args:
            model_name: Name of the embedding model to use
            csv_path: Path to the CSV training data file
            index_path: Path to save/load FAISS index
            device: Device to use ("cuda", "cpu", or "auto" for auto-detection)
        """
        self.model_name = model_name
        self.csv_path = csv_path
        self.index_path = index_path

        logger.info(f"Initializing embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Set device based on user preference
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

        # Qwen3-Embedding-0.6B has embedding dimension of 1024
        self.embedding_dim = 1024

        self.index = None
        self.category_names = list(CATEGORY_CONFIG.keys())
        self.category_to_index = {
            name: idx for idx, name in enumerate(self.category_names)
        }
        self.num_categories = len(self.category_names)

        logger.info(f"Loading training data from {csv_path}")
        self.texts, self.categories = self._load_csv_data()
        logger.info(f"Loaded {len(self.texts)} training examples")

        # Load or build FAISS index
        if os.path.exists(index_path):
            logger.info(f"Loading existing FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            logger.info(f"Index loaded with {self.index.ntotal} vectors")

            # Verify index size matches CSV
            if self.index.ntotal != len(self.texts):
                logger.warning(
                    f"Index size ({self.index.ntotal}) doesn't match CSV ({len(self.texts)}). Rebuilding..."
                )
                self._build_index()
        else:
            logger.info("No existing index found, building new one...")
            self._build_index()

    def _encode_texts(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode texts into embeddings using Qwen3 model.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding

        Returns:
            numpy array of embeddings
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling over sequence length
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def _load_csv_data(self) -> tuple[list[str], list[str]]:
        """
        Load training data from CSV file.

        Returns:
            Tuple of (texts, categories)
        """
        texts = []
        categories = []

        logger.info(f"Loading training data from {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row["text"])
                categories.append(row["category"])

        logger.info(f"Loaded {len(texts)} training examples")
        return texts, categories

    def _build_index(self):
        """Build FAISS index from loaded CSV data."""
        logger.info("Building FAISS index from training data...")

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(self.texts)} examples...")
        embeddings = self._encode_texts(self.texts)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype("float32"))

        # Build FAISS index
        logger.info(f"Creating FAISS index with dimension {self.embedding_dim}...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine
        self.index.add(embeddings)

        # Save index
        logger.info(f"Saving index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)

        logger.info(f"Index built successfully with {self.index.ntotal} vectors")

    def classify(
        self, text: str, k: int = 20, with_probabilities: bool = False
    ) -> dict[str, Any]:
        """
        Classify text using semantic similarity search.

        Args:
            text: Input text to classify
            k: Number of nearest neighbors to retrieve
            with_probabilities: Whether to return full probability distribution

        Returns:
            Dictionary with classification results
        """
        # Generate embedding for query text
        query_embedding = self._encode_texts([text])
        faiss.normalize_L2(query_embedding.astype("float32"))

        # Search for k nearest neighbors
        similarities, indices = self.index.search(query_embedding, k)

        # Get categories of nearest neighbors
        neighbor_categories = [self.categories[idx] for idx in indices[0]]
        neighbor_similarities = similarities[0]

        # Calculate confidence scores for each category
        category_scores = {cat: 0.0 for cat in self.category_names}

        for category, similarity in zip(neighbor_categories, neighbor_similarities):
            # Weight by similarity score (cosine similarity is already in [0, 1] after normalization)
            category_scores[category] += similarity

        # Normalize scores
        total_score = sum(category_scores.values())
        if total_score > 0:
            category_scores = {
                cat: score / total_score for cat, score in category_scores.items()
            }

        # Find best category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        best_category_name = best_category[0]
        best_confidence = best_category[1]

        # Get class index
        class_index = self.category_to_index[best_category_name]

        # Decide routing
        model, use_reasoning = self._decide_routing(
            text, best_category_name, best_confidence
        )

        result = {
            "class": int(class_index),
            "confidence": float(best_confidence),
            "model": model,
            "use_reasoning": use_reasoning,
        }

        if with_probabilities:
            # Create probability distribution (convert to native Python types)
            probabilities = [float(category_scores[cat]) for cat in self.category_names]
            result["probabilities"] = probabilities

            # Calculate entropy
            entropy_value = self._calculate_entropy(probabilities)
            result["entropy"] = float(entropy_value)

            logger.info(
                f"Classification result: class={class_index} ({best_category_name}), "
                f"confidence={best_confidence:.3f}, entropy={entropy_value:.3f}, "
                f"model={model}, use_reasoning={use_reasoning}"
            )
        else:
            logger.info(
                f"Classification result: class={class_index} ({best_category_name}), "
                f"confidence={best_confidence:.3f}, model={model}, use_reasoning={use_reasoning}"
            )

        return result

    def _calculate_entropy(self, probabilities: list[float]) -> float:
        """
        Calculate Shannon entropy of the probability distribution.

        Args:
            probabilities: List of probability values

        Returns:
            Entropy value
        """
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _decide_routing(
        self, text: str, category_name: str, confidence: float
    ) -> tuple[str, bool]:
        """
        Decide which model to use and whether to enable reasoning.

        Args:
            text: Input text being classified
            category_name: Predicted category
            confidence: Classification confidence

        Returns:
            Tuple of (model_name, use_reasoning)
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Check for complexity indicators
        complex_words = [
            "why",
            "how",
            "explain",
            "analyze",
            "compare",
            "evaluate",
            "describe",
        ]
        has_complex_words = any(word in text_lower for word in complex_words)

        # Long queries with complex words → use reasoning
        if word_count > 20 and has_complex_words:
            return "openai/gpt-oss-20b", True

        # Math category with simple queries → no reasoning needed
        if category_name == "math" and word_count < 15:
            return "openai/gpt-oss-20b", False

        # High confidence → can use simpler model
        if confidence > 0.9:
            return "openai/gpt-oss-20b", False

        # Low confidence → use reasoning to be safe
        if confidence < 0.6:
            return "openai/gpt-oss-20b", True

        # Default: use reasoning for better quality
        return "openai/gpt-oss-20b", True


# Initialize classifier globally
# Note: This is safe for aiohttp as it uses a single-threaded event loop.
# For multi-process deployments, each process gets its own instance.
classifier = None
classifier_device = "auto"  # Default device setting


def get_classifier():
    """Get or create the global classifier instance."""
    global classifier
    if classifier is None:
        # Get script directory
        script_dir = Path(__file__).parent
        csv_path = script_dir / "training_data.csv"
        index_path = script_dir / "faiss_index.bin"

        classifier = EmbeddingClassifier(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            csv_path=str(csv_path),
            index_path=str(index_path),
            device=classifier_device,
        )
    return classifier


# Initialize MCP server
app = Server("embedding-classifier")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    clf = get_classifier()
    return [
        Tool(
            name="classify_text",
            description=(
                "Classify text into categories using semantic embeddings and provide intelligent routing recommendations. "
                f"Categories: {', '.join(clf.category_names)}. "
                "Returns: class index, confidence, recommended model, and reasoning flag. "
                "Optionally returns full probability distribution for entropy analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to classify"},
                    "with_probabilities": {
                        "type": "boolean",
                        "description": "Whether to return full probability distribution for entropy analysis",
                        "default": False,
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="list_categories",
            description=(
                "List all available classification categories with per-category system prompts and descriptions. "
                "Returns: categories (array), category_system_prompts (object), category_descriptions (object). "
                "Each category can have its own system prompt that the router injects for category-specific LLM context."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    clf = get_classifier()

    if name == "classify_text":
        text = arguments.get("text", "")
        with_probabilities = arguments.get("with_probabilities", False)

        if not text:
            return [
                TextContent(type="text", text=json.dumps({"error": "No text provided"}))
            ]

        try:
            result = clf.classify(text, with_probabilities=with_probabilities)
            return [TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            logger.error(f"Error classifying text: {e}", exc_info=True)
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    elif name == "list_categories":
        # Return category information
        category_descriptions = {
            name: CATEGORY_CONFIG[name]["description"] for name in clf.category_names
        }

        category_system_prompts = {
            name: CATEGORY_CONFIG[name]["system_prompt"] for name in clf.category_names
        }

        categories_response = {
            "categories": clf.category_names,
            "category_system_prompts": category_system_prompts,
            "category_descriptions": category_descriptions,
        }

        logger.info(
            f"Returning {len(clf.category_names)} categories with {len(category_system_prompts)} system prompts: {clf.category_names}"
        )
        return [TextContent(type="text", text=json.dumps(categories_response))]

    else:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Unknown tool: {name}"})
            )
        ]


async def main_stdio(device: str = "auto"):
    """Run the MCP server in stdio mode."""
    global classifier_device
    classifier_device = device

    logger.info("Starting Embedding-Based MCP Classification Server (stdio mode)")
    clf = get_classifier()
    logger.info(f"Available categories: {', '.join(clf.category_names)}")
    logger.info(f"Model: {clf.model_name}")
    logger.info(f"Device: {clf.device}")
    logger.info(f"Index size: {clf.index.ntotal} vectors")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def main_http(port: int = 8091, device: str = "auto"):
    """Run the MCP server in HTTP mode."""
    global classifier_device
    classifier_device = device

    try:
        from aiohttp import web
    except ImportError:
        logger.error(
            "aiohttp is required for HTTP mode. Install it with: pip install aiohttp"
        )
        return

    logger.info(f"Starting Embedding-Based MCP Classification Server (HTTP mode)")
    clf = get_classifier()
    logger.info(f"Available categories: {', '.join(clf.category_names)}")
    logger.info(f"Model: {clf.model_name}")
    logger.info(f"Device: {clf.device}")
    logger.info(f"Index size: {clf.index.ntotal} vectors")
    logger.info(f"Listening on http://0.0.0.0:{port}/mcp")

    async def handle_mcp_request(request):
        """Handle MCP requests over HTTP."""
        try:
            data = await request.json()
            method = data.get("method", "")

            # Extract method from URL path if not in JSON
            if not method:
                path = request.path
                if path.startswith("/mcp/"):
                    method = path[5:]
                elif path == "/mcp":
                    method = ""

            params = data.get("params", data if not data.get("method") else {})
            request_id = data.get("id", 1)

            logger.debug(
                f"Received MCP request: method={method}, path={request.path}, id={request_id}"
            )

            # Handle initialize
            if method == "initialize":
                init_result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},  # We support tools
                        # Note: We don't support resources or prompts
                    },
                    "serverInfo": {
                        "name": "embedding-classifier",
                        "version": "1.0.0",
                        "description": "Embedding-based text classification with semantic similarity",
                    },
                }

                if request.path.startswith("/mcp/") and request.path != "/mcp":
                    return web.json_response(init_result)
                else:
                    result = {"jsonrpc": "2.0", "id": request_id, "result": init_result}
                    return web.json_response(result)

            # Handle tools/list
            elif method == "tools/list":
                tools_list = await list_tools()
                tools_data = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in tools_list
                ]

                if request.path.startswith("/mcp/") and request.path != "/mcp":
                    return web.json_response({"tools": tools_data})
                else:
                    result = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": tools_data},
                    }
                    return web.json_response(result)

            # Handle tools/call
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})

                tool_result = await call_tool(tool_name, arguments)

                # Convert TextContent to dict
                content = [{"type": tc.type, "text": tc.text} for tc in tool_result]

                result_data = {"content": content, "isError": False}

                if request.path.startswith("/mcp/") and request.path != "/mcp":
                    return web.json_response(result_data)
                else:
                    result = {"jsonrpc": "2.0", "id": request_id, "result": result_data}
                    return web.json_response(result)

            # Handle ping
            elif method == "ping":
                result = {"jsonrpc": "2.0", "id": request_id, "result": {}}
                return web.json_response(result)

            # Handle unsupported but valid MCP methods gracefully
            elif method in [
                "resources/list",
                "resources/read",
                "prompts/list",
                "prompts/get",
            ]:
                # These are valid MCP methods but not implemented in this server
                # Return empty results instead of error for better compatibility
                logger.debug(
                    f"Unsupported method called: {method} (returning empty result)"
                )

                if method == "resources/list":
                    result_data = {"resources": []}
                elif method == "prompts/list":
                    result_data = {"prompts": []}
                else:
                    result_data = {}

                result = {"jsonrpc": "2.0", "id": request_id, "result": result_data}
                return web.json_response(result)

            else:
                # Unknown method - return error with HTTP 200 (per JSON-RPC spec)
                logger.warning(f"Unknown method called: {method}")
                error = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
                return web.json_response(error)

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            error = {
                "jsonrpc": "2.0",
                "id": (
                    data.get("id")
                    if "data" in locals() and isinstance(data, dict)
                    else None
                ),
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }
            # Per JSON-RPC 2.0 spec, return HTTP 200 even for errors
            return web.json_response(error)

    async def health_check(request):
        """Health check endpoint."""
        clf = get_classifier()
        return web.json_response(
            {
                "status": "ok",
                "categories": clf.category_names,
                "model": clf.model_name,
                "index_size": clf.index.ntotal,
            }
        )

    # Create web application
    http_app = web.Application()

    # Main JSON-RPC endpoint
    http_app.router.add_post("/mcp", handle_mcp_request)

    # REST-style endpoints
    http_app.router.add_post("/mcp/initialize", handle_mcp_request)
    http_app.router.add_post("/mcp/tools/list", handle_mcp_request)
    http_app.router.add_post("/mcp/tools/call", handle_mcp_request)
    http_app.router.add_post("/mcp/resources/list", handle_mcp_request)
    http_app.router.add_post("/mcp/resources/read", handle_mcp_request)
    http_app.router.add_post("/mcp/prompts/list", handle_mcp_request)
    http_app.router.add_post("/mcp/prompts/get", handle_mcp_request)
    http_app.router.add_post("/mcp/ping", handle_mcp_request)

    # Health check
    http_app.router.add_get("/health", health_check)

    # Run the server
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info(f"Server is ready at http://0.0.0.0:{port}/mcp")
    logger.info(f"Health check available at http://0.0.0.0:{port}/health")

    # Keep the server running
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    import asyncio

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MCP Embedding-Based Classification Server"
    )
    parser.add_argument(
        "--http", action="store_true", help="Run in HTTP mode instead of stdio"
    )
    parser.add_argument("--port", type=int, default=8091, help="HTTP port to listen on")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (auto=auto-detect, cuda=force GPU, cpu=force CPU)",
    )
    args = parser.parse_args()

    if args.http:
        asyncio.run(main_http(args.port, args.device))
    else:
        asyncio.run(main_stdio(args.device))
