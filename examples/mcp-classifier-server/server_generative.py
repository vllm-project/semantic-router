#!/usr/bin/env python3
"""
Generative Model-Based MCP Classification Server with Intelligent Routing

This is an example MCP server that demonstrates:
1. Text classification using a fine-tuned Qwen3 generative model
2. Dynamic category discovery via list_categories
3. Intelligent routing decisions (model selection and reasoning control)
4. Softmax-based probability distributions and entropy calculation

The server implements two MCP tools:
- 'list_categories': Returns available categories with per-category system prompts and descriptions
- 'classify_text': Classifies text using generative model and returns routing recommendations

Protocol:
- list_categories returns: {
    "categories": ["biology", "business", "chemistry", ...],
    "category_system_prompts": {
      "biology": "You are a biology expert...",
      ...
    },
    "category_descriptions": {
      "biology": "Biological sciences and life sciences queries",
      ...
    }
  }
- classify_text returns: {
    "class": 0,
    "confidence": 0.85,
    "model": "openai/gpt-oss-20b",
    "use_reasoning": true,
    "probabilities": [...],
    "entropy": 0.45
  }

Usage:
  # Stdio mode (for testing with MCP clients)
  python server_generative.py --model-path qwen3_generative_classifier_r16

  # HTTP mode (for semantic router)
  python server_generative.py --http --port 8092 --model-path qwen3_generative_classifier_r16
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional, Sequence, TypedDict

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Type definitions for better type safety
# Note: We use dict[str, Any] for ClassificationResult because the key "class"
# is a reserved Python keyword, making it difficult to use with TypedDict.
# The structure is documented here:
#
# ClassificationResult = {
#     "class": int,              # Category index (0-13 for MMLU-Pro)
#     "confidence": float,        # Classification confidence (0.0-1.0)
#     "model": str,              # Recommended model (e.g., "openai/gpt-oss-20b")
#     "use_reasoning": bool,     # Whether to enable reasoning
#     "probabilities": list[float],  # Optional: full distribution (with_probabilities=True)
#     "entropy": float,          # Optional: Shannon entropy (with_probabilities=True)
# }

# Category definitions with system prompts (matching MMLU-Pro categories)
CATEGORY_CONFIG = {
    "biology": {
        "description": "Biological sciences and life sciences queries",
        "system_prompt": """You are a biology expert. When answering biology questions:
- Explain biological processes and mechanisms clearly
- Use proper scientific terminology for organisms, cells, and systems
- Reference relevant biological concepts, theories, and research
- Describe anatomical structures and physiological functions
- Connect concepts across different levels of biological organization""",
    },
    "business": {
        "description": "Business, management, and corporate topics",
        "system_prompt": """You are a business expert. When answering business questions:
- Apply business frameworks and strategic thinking
- Consider market dynamics, competitive forces, and stakeholder interests
- Explain financial concepts, metrics, and business models
- Discuss organizational structures and management practices
- Provide practical insights for business decision-making""",
    },
    "chemistry": {
        "description": "Chemical sciences and molecular topics",
        "system_prompt": """You are a chemistry expert. When answering chemistry questions:
- Explain chemical reactions, bonds, and molecular structures
- Use proper chemical nomenclature and notation
- Discuss periodic trends and chemical properties
- Apply concepts from organic, inorganic, and physical chemistry
- Consider experimental methods and laboratory techniques""",
    },
    "computer science": {
        "description": "Computing, algorithms, and software topics",
        "system_prompt": """You are a computer science expert. When answering CS questions:
- Explain algorithms, data structures, and computational complexity
- Discuss software design patterns and architectural principles
- Use proper programming terminology and concepts
- Consider performance, scalability, and correctness
- Reference theoretical foundations and practical implementations""",
    },
    "economics": {
        "description": "Economic theory and applied economics",
        "system_prompt": """You are an economics expert. When answering economics questions:
- Apply economic theories and models (micro and macro)
- Explain market mechanisms, incentives, and trade-offs
- Discuss supply and demand, elasticity, and equilibrium
- Consider policy implications and economic indicators
- Use graphs and quantitative reasoning when relevant""",
    },
    "engineering": {
        "description": "Engineering disciplines and technical topics",
        "system_prompt": """You are an engineering expert. When answering engineering questions:
- Apply engineering principles and problem-solving methods
- Consider design constraints, optimization, and trade-offs
- Explain technical systems, components, and processes
- Discuss materials, forces, and energy considerations
- Reference relevant engineering standards and best practices""",
    },
    "health": {
        "description": "Medicine, healthcare, and wellness topics",
        "system_prompt": """You are a health expert. When answering health questions:
- Provide evidence-based medical and health information
- Explain diseases, symptoms, treatments, and preventive measures
- Discuss anatomy, physiology, and pathology
- Consider patient care, public health, and healthcare systems
- Use appropriate medical terminology and cite clinical evidence""",
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
    "law": {
        "description": "Legal systems, regulations, and jurisprudence",
        "system_prompt": """You are a law expert. When answering legal questions:
- Explain legal principles, doctrines, and precedents
- Discuss statutes, regulations, and case law
- Consider different areas of law (constitutional, criminal, civil, etc.)
- Analyze legal reasoning and argumentation
- Note jurisdictional differences when relevant""",
    },
    "math": {
        "description": "Mathematical and computational queries",
        "system_prompt": """You are a mathematics expert. When answering math questions:
- Show step-by-step solutions with clear explanations
- Use proper mathematical notation and terminology
- Verify calculations and provide intermediate steps
- Explain the underlying concepts and principles
- Offer alternative approaches when applicable""",
    },
    "other": {
        "description": "General or interdisciplinary topics",
        "system_prompt": """You are a knowledgeable assistant. When answering general questions:
- Provide balanced, well-rounded responses
- Draw from multiple domains of knowledge when relevant
- Be clear, concise, and accurate
- Adapt your explanation to the complexity of the question
- Acknowledge limitations and uncertainties when appropriate""",
    },
    "philosophy": {
        "description": "Philosophical concepts and reasoning",
        "system_prompt": """You are a philosophy expert. When answering philosophy questions:
- Explain philosophical concepts, theories, and arguments
- Reference relevant philosophers and schools of thought
- Analyze logical structure and reasoning
- Consider different philosophical perspectives and debates
- Discuss metaphysics, epistemology, ethics, and other branches""",
    },
    "physics": {
        "description": "Physical sciences and phenomena",
        "system_prompt": """You are a physics expert. When answering physics questions:
- Apply physical laws, principles, and equations
- Explain phenomena using appropriate physics concepts
- Show mathematical derivations when relevant
- Discuss both classical and modern physics
- Consider experimental evidence and theoretical frameworks""",
    },
    "psychology": {
        "description": "Psychological concepts and human behavior",
        "system_prompt": """You are a psychology expert. When answering psychology questions:
- Explain psychological theories, concepts, and research findings
- Discuss cognitive processes, behavior, and mental states
- Reference relevant psychological studies and evidence
- Consider different perspectives (cognitive, behavioral, social, etc.)
- Apply scientific reasoning to human behavior and mental processes""",
    },
}


class GenerativeClassifier:
    """Generative model-based text classifier using fine-tuned Qwen3."""

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
    ):
        """
        Initialize the generative classifier.

        Args:
            model_path: Path to the fine-tuned model directory or HuggingFace model ID
            base_model_name: Name of the base Qwen3 model
            device: Device to use ("cuda", "cpu", or "auto" for auto-detection)
        """
        self.model_path = model_path
        self.base_model_name = base_model_name

        logger.info(f"Loading generative model from: {model_path}")

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

        # Detect if this is a HuggingFace model or local path
        self.is_hf_model = self._is_huggingface_model(model_path)

        if self.is_hf_model:
            logger.info(f"Detected HuggingFace model: {model_path}")
        else:
            logger.info(f"Detected local model path: {model_path}")

        # Load label mapping
        label_mapping_path = self._get_label_mapping_path(model_path)
        logger.info(f"Loading label mapping from: {label_mapping_path}")

        with open(label_mapping_path, "r") as f:
            mapping_data = json.load(f)
            self.label2id = mapping_data["label2id"]
            self.id2label = mapping_data["id2label"]
            self.instruction_template = mapping_data.get("instruction_template", "")

        self.category_names = [self.id2label[str(i)] for i in range(len(self.id2label))]
        logger.info(
            f"Loaded {len(self.category_names)} categories: {self.category_names}"
        )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with appropriate dtype
        use_fp16 = False
        if torch.cuda.is_available():
            try:
                compute_capability = torch.cuda.get_device_capability()
                use_fp16 = compute_capability[0] >= 7  # Volta and newer
            except Exception:
                use_fp16 = False

        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            device_map=None,  # We'll manually move to device
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Load LoRA weights
        logger.info("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.to(self.device)
        self.model.eval()

        # Pre-tokenize category names for efficient logit extraction
        self._prepare_category_tokens()

        logger.info("Model loaded successfully")

    def _is_huggingface_model(self, model_path: str) -> bool:
        """
        Detect if the model_path is a HuggingFace model ID or local path.

        Args:
            model_path: Model path or HuggingFace model ID

        Returns:
            True if it's a HuggingFace model ID, False if it's a local path
        """
        # Check if it's a local path that exists
        if os.path.exists(model_path):
            return False

        # Check if it looks like a HuggingFace model ID (contains /)
        # and is not an absolute path
        if "/" in model_path and not os.path.isabs(model_path):
            return True

        return False

    def _get_label_mapping_path(self, model_path: str) -> str:
        """
        Get the path to label_mapping.json for both local and HuggingFace models.

        Args:
            model_path: Model path or HuggingFace model ID

        Returns:
            Path to label_mapping.json file
        """
        if self.is_hf_model:
            # Download from HuggingFace Hub
            try:
                label_mapping_path = hf_hub_download(
                    repo_id=model_path,
                    filename="label_mapping.json",
                    cache_dir=None,  # Use default cache
                )
                return label_mapping_path
            except Exception as e:
                logger.error(
                    f"Failed to download label_mapping.json from HuggingFace: {e}"
                )
                raise
        else:
            # Local path
            return os.path.join(model_path, "label_mapping.json")

    def _prepare_category_tokens(self):
        """Pre-tokenize category names to extract their token IDs."""
        self.category_token_ids = []
        self.category_first_tokens = []

        for category in self.category_names:
            # Tokenize the category name (with leading space to match generation context)
            tokens = self.tokenizer.encode(f" {category}", add_special_tokens=False)
            self.category_token_ids.append(tokens)
            # Store first token for probability extraction
            if tokens:
                self.category_first_tokens.append(tokens[0])
            else:
                # Fallback: tokenize without space
                tokens = self.tokenizer.encode(category, add_special_tokens=False)
                self.category_first_tokens.append(tokens[0] if tokens else 0)

        logger.info(
            f"Prepared category tokens: {len(self.category_first_tokens)} categories"
        )

    def _format_instruction(self, question: str) -> str:
        """Format a question using the instruction template."""
        if self.instruction_template:
            return self.instruction_template.format(question=question)
        else:
            # Fallback template
            return f"""You are an expert academic classifier. Classify the following question into exactly ONE category. Respond with ONLY the category name.

Categories: {', '.join(self.category_names)}

Now classify this question:
Q: {question}
A:"""

    def classify(self, text: str, with_probabilities: bool = False) -> dict[str, Any]:
        """
        Classify text using the generative model.

        Args:
            text: Input text to classify
            with_probabilities: Whether to return full probability distribution

        Returns:
            Dictionary with classification results
        """
        # Format the instruction
        prompt = self._format_instruction(text)

        # Tokenize
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        # Get model output with logits
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

        # Get logits at the last position (where the model predicts the next token)
        last_logits = logits[0, -1, :]  # Shape: (vocab_size,)

        # Extract logits for category tokens
        category_logits = []
        for token_id in self.category_first_tokens:
            category_logits.append(last_logits[token_id].item())

        category_logits = torch.tensor(category_logits)

        # Compute softmax probabilities
        probabilities = F.softmax(category_logits, dim=0)
        probabilities_list = probabilities.cpu().numpy().tolist()

        # Find best category
        best_idx = int(torch.argmax(probabilities).item())
        best_category = self.category_names[best_idx]
        best_confidence = float(probabilities[best_idx].item())

        # Decide routing
        model, use_reasoning = self._decide_routing(
            text, best_category, best_confidence
        )

        result = {
            "class": int(best_idx),
            "confidence": float(best_confidence),
            "model": model,
            "use_reasoning": use_reasoning,
        }

        if with_probabilities:
            result["probabilities"] = probabilities_list

            # Calculate entropy
            entropy_value = self._calculate_entropy(probabilities_list)
            result["entropy"] = float(entropy_value)

            logger.info(
                f"Classification result: class={best_idx} ({best_category}), "
                f"confidence={best_confidence:.3f}, entropy={entropy_value:.3f}, "
                f"model={model}, use_reasoning={use_reasoning}"
            )
        else:
            logger.info(
                f"Classification result: class={best_idx} ({best_category}), "
                f"confidence={best_confidence:.3f}, model={model}, use_reasoning={use_reasoning}"
            )

        return result

    def _calculate_entropy(self, probabilities: Sequence[float]) -> float:
        """
        Calculate Shannon entropy of the probability distribution.

        Args:
            probabilities: Sequence of probability values (list, tuple, numpy array, etc.)

        Returns:
            Entropy value (in bits)
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
classifier_config = {
    "model_path": None,
    "base_model_name": "Qwen/Qwen3-0.6B",
    "device": "auto",
}


def get_classifier():
    """Get or create the global classifier instance."""
    global classifier
    if classifier is None:
        if classifier_config["model_path"] is None:
            raise ValueError("Model path not set. Use --model-path argument.")

        classifier = GenerativeClassifier(
            model_path=classifier_config["model_path"],
            base_model_name=classifier_config["base_model_name"],
            device=classifier_config["device"],
        )
    return classifier


# Initialize MCP server
app = Server("generative-classifier")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    clf = get_classifier()
    return [
        Tool(
            name="classify_text",
            description=(
                "Classify text into categories using a fine-tuned generative model and provide intelligent routing recommendations. "
                f"Categories: {', '.join(clf.category_names)}. "
                "Returns: class index, confidence, recommended model, and reasoning flag. "
                "Optionally returns full probability distribution (from softmax) for entropy analysis."
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
        category_descriptions = {}
        category_system_prompts = {}

        for name in clf.category_names:
            if name in CATEGORY_CONFIG:
                category_descriptions[name] = CATEGORY_CONFIG[name]["description"]
                category_system_prompts[name] = CATEGORY_CONFIG[name]["system_prompt"]
            else:
                # Fallback for categories not in config
                category_descriptions[name] = f"{name.title()} related queries"
                category_system_prompts[name] = f"You are a {name} expert."

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


async def main_stdio(model_path: str, base_model_name: str, device: str):
    """Run the MCP server in stdio mode."""
    classifier_config["model_path"] = model_path
    classifier_config["base_model_name"] = base_model_name
    classifier_config["device"] = device

    logger.info(
        "Starting Generative Model-Based MCP Classification Server (stdio mode)"
    )
    clf = get_classifier()
    logger.info(f"Available categories: {', '.join(clf.category_names)}")
    logger.info(f"Base model: {clf.base_model_name}")
    logger.info(f"Model path: {clf.model_path}")
    logger.info(f"Device: {clf.device}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def main_http(port: int, model_path: str, base_model_name: str, device: str):
    """Run the MCP server in HTTP mode."""
    classifier_config["model_path"] = model_path
    classifier_config["base_model_name"] = base_model_name
    classifier_config["device"] = device

    try:
        from aiohttp import web
    except ImportError:
        logger.error(
            "aiohttp is required for HTTP mode. Install it with: pip install aiohttp"
        )
        return

    logger.info(
        f"Starting Generative Model-Based MCP Classification Server (HTTP mode)"
    )
    clf = get_classifier()
    logger.info(f"Available categories: {', '.join(clf.category_names)}")
    logger.info(f"Base model: {clf.base_model_name}")
    logger.info(f"Model path: {clf.model_path}")
    logger.info(f"Device: {clf.device}")
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
                        "name": "generative-classifier",
                        "version": "1.0.0",
                        "description": "Generative model-based text classification with softmax probabilities",
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
                "base_model": clf.base_model_name,
                "model_path": clf.model_path,
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
        description="MCP Generative Model-Based Classification Server"
    )
    parser.add_argument(
        "--http", action="store_true", help="Run in HTTP mode instead of stdio"
    )
    parser.add_argument("--port", type=int, default=8092, help="HTTP port to listen on")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory (e.g., qwen3_generative_classifier_r16)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (auto=auto-detect, cuda=force GPU, cpu=force CPU)",
    )
    args = parser.parse_args()

    if args.http:
        asyncio.run(main_http(args.port, args.model_path, args.base_model, args.device))
    else:
        asyncio.run(main_stdio(args.model_path, args.base_model, args.device))
