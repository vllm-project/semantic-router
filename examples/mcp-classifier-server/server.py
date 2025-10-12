#!/usr/bin/env python3
"""
Regex-Based MCP Classification Server with Intelligent Routing

This is an example MCP server that demonstrates:
1. Text classification using regex patterns
2. Dynamic category discovery via list_categories
3. Intelligent routing decisions (model selection and reasoning control)

The server implements two MCP tools:
- 'list_categories': Returns available categories with per-category system prompts and descriptions
- 'classify_text': Classifies text and returns routing recommendations

Protocol:
- list_categories returns: {
    "categories": ["math", "science", "technology", ...],
    "category_system_prompts": {  # optional, per-category system prompts
      "math": "You are a mathematics expert. When answering math questions...",
      "science": "You are a science expert. When answering science questions...",
      "technology": "You are a technology expert. When answering tech questions..."
    },
    "category_descriptions": {  # optional
      "math": "Mathematical and computational queries",
      "science": "Scientific concepts and queries"
    }
  }
- classify_text returns: {
    "class": 0,
    "confidence": 0.85,
    "model": "openai/gpt-oss-20b",
    "use_reasoning": true,
    "probabilities": [...]  # optional
  }

Usage:
  # Stdio mode (for testing with MCP clients)
  python server.py

  # HTTP mode (for semantic router)
  python server.py --http --port 8080
"""

import argparse
import json
import logging
import math
import re
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define classification categories with their regex patterns, descriptions, and system prompts
# Each category has its own system prompt for specialized context
CATEGORIES = {
    "math": {
        "patterns": [
            r"\b(calculate|compute|solve|equation|formula|algebra|geometry|trigonometry)\b",
            r"\b(integral|derivative|differential|matrix|vector|probability)\b",
            r"\b(\d+\s*[+\-*/]\s*\d+|x\s*[=<>]\s*\d+)\b",
            r"\b(sin|cos|tan|log|sqrt|sum|average|mean)\b",
        ],
        "description": "Mathematical and computational queries",
        "system_prompt": """You are a mathematics expert. When answering math questions:
- Show step-by-step solutions with clear explanations
- Use proper mathematical notation and terminology
- Verify calculations and provide intermediate steps
- Explain the underlying concepts and principles
- Offer alternative approaches when applicable""",
    },
    "science": {
        "patterns": [
            r"\b(physics|chemistry|biology|astronomy|geology|ecology)\b",
            r"\b(atom|molecule|cell|DNA|RNA|evolution|gravity|energy)\b",
            r"\b(experiment|hypothesis|theory|scientific|research)\b",
            r"\b(planet|star|galaxy|universe|ecosystem|organism)\b",
        ],
        "description": "Scientific concepts and queries",
        "system_prompt": """You are a science expert. When answering science questions:
- Provide evidence-based answers grounded in scientific research
- Explain relevant scientific concepts and principles
- Use appropriate scientific terminology
- Cite the scientific method and experimental evidence when relevant
- Distinguish between established facts and current theories""",
    },
    "technology": {
        "patterns": [
            r"\b(computer|software|hardware|programming|code|algorithm)\b",
            r"\b(internet|network|database|server|cloud|API)\b",
            r"\b(machine learning|AI|artificial intelligence|neural network)\b",
            r"\b(python|java|javascript|C\+\+|golang|rust)\b",
        ],
        "description": "Technology and computing topics",
        "system_prompt": """You are a technology expert. When answering tech questions:
- Include practical examples and code snippets when relevant
- Follow best practices and industry standards
- Explain both high-level concepts and implementation details
- Consider security, performance, and maintainability
- Recommend appropriate tools and technologies for the use case""",
    },
    "history": {
        "patterns": [
            r"\b(ancient|medieval|renaissance|revolution|war|empire)\b",
            r"\b(century|era|period|historical|historian|archaeology)\b",
            r"\b(civilization|dynasty|monarchy|republic|democracy)\b",
            r"\b(BCE|CE|AD|BC|\d{4})\b.*\b(year|century|ago)\b",
        ],
        "description": "Historical events and topics",
        "system_prompt": """You are a history expert. When answering historical questions:
- Provide accurate dates, names, and historical context
- Cite time periods and geographical locations
- Explain the causes, events, and consequences
- Consider multiple perspectives and historical interpretations
- Connect historical events to their broader significance""",
    },
    "general": {
        "patterns": [r".*"],  # Catch-all pattern
        "description": "General questions and topics",
        "system_prompt": """You are a knowledgeable assistant. When answering general questions:
- Provide balanced, well-rounded responses
- Draw from multiple domains of knowledge when relevant
- Be clear, concise, and accurate
- Adapt your explanation to the complexity of the question
- Acknowledge limitations and uncertainties when appropriate""",
    },
}

# Build index from category names to indices
CATEGORY_NAMES = list(CATEGORIES.keys())
CATEGORY_TO_INDEX = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}
NUM_CATEGORIES = len(CATEGORY_NAMES)


def calculate_confidence(text: str, category_name: str) -> float:
    """
    Calculate confidence score for a category based on pattern matches.

    Args:
        text: Input text to classify
        category_name: Category name to check

    Returns:
        Confidence score between 0 and 1
    """
    patterns = CATEGORIES[category_name]["patterns"]
    matches = 0
    total_patterns = len(patterns)

    text_lower = text.lower()

    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matches += 1

    # Calculate base confidence from match ratio
    if total_patterns == 0:
        return 0.0

    match_ratio = matches / total_patterns

    # Boost confidence for multiple matches
    confidence = min(0.5 + (match_ratio * 0.5), 1.0)

    return confidence


def calculate_entropy(probabilities: list[float]) -> float:
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


def decide_routing(
    text: str, category_name: str, confidence: float
) -> tuple[str, bool]:
    """
    Decide which model to use and whether to enable reasoning based on query analysis.

    This is a simple example that demonstrates intelligent routing. In production,
    this could use ML models, query complexity analysis, etc.

    Args:
        text: Input text being classified
        category_name: Predicted category
        confidence: Classification confidence

    Returns:
        Tuple of (model_name, use_reasoning)
    """
    # Analyze query complexity
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

    # Simple routing logic (you can make this much more sophisticated)

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


def classify_text(text: str, with_probabilities: bool = False) -> dict[str, Any]:
    """
    Classify text using regex patterns and return routing recommendations.

    Args:
        text: Input text to classify
        with_probabilities: Whether to return full probability distribution

    Returns:
        Dictionary with classification results including model and use_reasoning
    """
    logger.info(
        f"Classifying text: '{text[:100]}...' (with_probabilities={with_probabilities})"
    )

    # Calculate confidence for each category
    confidences = {}
    for category_name in CATEGORY_NAMES:
        confidences[category_name] = calculate_confidence(text, category_name)

    # Find the category with highest confidence
    best_category = max(confidences.items(), key=lambda x: x[1])
    best_category_name = best_category[0]
    best_confidence = best_category[1]

    # Get the class index
    class_index = CATEGORY_TO_INDEX[best_category_name]

    # Decide routing (model and reasoning)
    model, use_reasoning = decide_routing(text, best_category_name, best_confidence)

    result = {
        "class": class_index,
        "confidence": best_confidence,
        "model": model,
        "use_reasoning": use_reasoning,
    }

    if with_probabilities:
        # Normalize confidences to probabilities
        total = sum(confidences.values())
        if total > 0:
            probabilities = [confidences[cat] / total for cat in CATEGORY_NAMES]
        else:
            # Uniform distribution if no matches
            probabilities = [1.0 / NUM_CATEGORIES] * NUM_CATEGORIES

        result["probabilities"] = probabilities

        # Calculate and add entropy
        entropy_value = calculate_entropy(probabilities)
        result["entropy"] = entropy_value

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


# Initialize MCP server
app = Server("regex-classifier")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="classify_text",
            description=(
                "Classify text into categories and provide intelligent routing recommendations. "
                f"Categories: {', '.join(CATEGORY_NAMES)}. "
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
    if name == "classify_text":
        text = arguments.get("text", "")
        with_probabilities = arguments.get("with_probabilities", False)

        if not text:
            return [
                TextContent(type="text", text=json.dumps({"error": "No text provided"}))
            ]

        try:
            result = classify_text(text, with_probabilities)
            return [TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            logger.error(f"Error classifying text: {e}", exc_info=True)
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    elif name == "list_categories":
        # Return category information including per-category system prompts and descriptions
        # This allows the router to get category-specific instructions from the MCP server
        category_descriptions = {
            name: CATEGORIES[name]["description"] for name in CATEGORY_NAMES
        }

        category_system_prompts = {
            name: CATEGORIES[name]["system_prompt"]
            for name in CATEGORY_NAMES
            if "system_prompt" in CATEGORIES[name]
        }

        categories_response = {
            "categories": CATEGORY_NAMES,
            "category_system_prompts": category_system_prompts,
            "category_descriptions": category_descriptions,
        }

        logger.info(
            f"Returning {len(CATEGORY_NAMES)} categories with {len(category_system_prompts)} system prompts: {CATEGORY_NAMES}"
        )
        return [TextContent(type="text", text=json.dumps(categories_response))]

    else:
        return [
            TextContent(
                type="text", text=json.dumps({"error": f"Unknown tool: {name}"})
            )
        ]


async def main_stdio():
    """Run the MCP server in stdio mode."""
    logger.info("Starting Regex-Based MCP Classification Server (stdio mode)")
    logger.info(f"Available categories: {', '.join(CATEGORY_NAMES)}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def main_http(port: int = 8080):
    """Run the MCP server in HTTP mode."""
    try:
        from aiohttp import web
    except ImportError:
        logger.error(
            "aiohttp is required for HTTP mode. Install it with: pip install aiohttp"
        )
        return

    logger.info(f"Starting Regex-Based MCP Classification Server (HTTP mode)")
    logger.info(f"Available categories: {', '.join(CATEGORY_NAMES)}")
    logger.info(f"Listening on http://0.0.0.0:{port}/mcp")

    async def handle_mcp_request(request):
        """Handle MCP requests over HTTP."""
        try:
            # Parse JSON-RPC request or extract method from URL path
            data = await request.json()

            # Support both styles:
            # 1. JSON-RPC style: {"method": "tools/list", "params": {...}}
            # 2. REST style: POST to /mcp/tools/list with direct params
            method = data.get("method", "")

            # If no method in JSON, extract from URL path
            if not method:
                path = request.path
                if path.startswith("/mcp/"):
                    method = path[5:]  # Remove '/mcp/' prefix
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
                        "tools": {},
                    },
                    "serverInfo": {"name": "regex-classifier", "version": "1.0.0"},
                }

                # For REST-style endpoints, return direct result
                # For JSON-RPC style, return full JSON-RPC response
                if request.path.startswith("/mcp/") and request.path != "/mcp":
                    # REST style - return direct result
                    return web.json_response(init_result)
                else:
                    # JSON-RPC style
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

                # For REST-style endpoints (path-based), return direct result
                # For JSON-RPC style (method in body), return full JSON-RPC response
                if request.path.startswith("/mcp/") and request.path != "/mcp":
                    # REST style - return direct result
                    return web.json_response({"tools": tools_data})
                else:
                    # JSON-RPC style
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

                # For REST-style endpoints, return direct result
                # For JSON-RPC style, return full JSON-RPC response
                if request.path.startswith("/mcp/") and request.path != "/mcp":
                    # REST style - return direct result
                    return web.json_response(result_data)
                else:
                    # JSON-RPC style
                    result = {"jsonrpc": "2.0", "id": request_id, "result": result_data}
                    return web.json_response(result)

            # Handle ping
            elif method == "ping":
                result = {"jsonrpc": "2.0", "id": request_id, "result": {}}
                return web.json_response(result)

            else:
                error = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
                return web.json_response(error, status=404)

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            error = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }
            return web.json_response(error, status=500)

    async def health_check(request):
        """Health check endpoint."""
        return web.json_response({"status": "ok", "categories": CATEGORY_NAMES})

    # Create web application
    http_app = web.Application()

    # Main JSON-RPC endpoint (single endpoint style)
    http_app.router.add_post("/mcp", handle_mcp_request)

    # REST-style endpoints (for Go client compatibility)
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
    parser = argparse.ArgumentParser(description="MCP Classification Server")
    parser.add_argument(
        "--http", action="store_true", help="Run in HTTP mode instead of stdio"
    )
    parser.add_argument("--port", type=int, default=8090, help="HTTP port to listen on")
    args = parser.parse_args()

    if args.http:
        asyncio.run(main_http(args.port))
    else:
        asyncio.run(main_stdio())
