#!/usr/bin/env python3
"""
Mock web search service for hallucination detection demo.

Returns context data that can be used to verify LLM responses.
The contexts are designed to conflict with mock vLLM's hallucinated responses.

Usage:
    python mock_web_search.py --port 8003
"""

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

# Search results - keyword to context mapping
# These contexts are the "ground truth" that the hallucination detector uses
SEARCH_CONTEXTS = {
    "eiffel": {
        "query": "Eiffel Tower facts",
        "context": "The Eiffel Tower is located in Paris, France. Construction began in 1887 and was completed in 1889 for the World's Fair. The tower stands 330 meters (1,083 ft) tall. It was originally painted brown, not red. The tower was designed by Gustave Eiffel's engineering company, but the architectural design was by Maurice Koechlin and Émile Nouguier.",
        "source": "Wikipedia - Eiffel Tower",
        "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
    },
    "tokyo": {
        "query": "Tokyo population",
        "context": "Tokyo is the capital city of Japan. The Tokyo Metropolis has a population of approximately 14 million people. The Greater Tokyo Area is the most populous metropolitan area in the world with about 37 million residents. Tokyo was originally named Edo and was renamed Tokyo in 1868.",
        "source": "Japan Statistics Bureau",
        "url": "https://www.stat.go.jp/",
    },
    "apple": {
        "query": "Apple Inc founding",
        "context": "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. The company was incorporated in 1977. The Apple I computer was sold for $666.66. Bill Gates was never a founder of Apple; he co-founded Microsoft.",
        "source": "Apple Inc. History",
        "url": "https://www.apple.com/about/",
    },
    "amazon": {
        "query": "Amazon founding history",
        "context": "Amazon was founded by Jeff Bezos on July 5, 1994, in Bellevue, Washington. The company started as an online bookstore. Amazon went public in 1997 at $18 per share. The company is headquartered in Seattle, Washington.",
        "source": "Amazon About Us",
        "url": "https://www.aboutamazon.com/",
    },
    "default": {
        "query": "General search",
        "context": "This is general information retrieved from web search. Please verify specific claims against authoritative sources.",
        "source": "Web Search",
        "url": "https://example.com",
    },
}


class MockWebSearchHandler(BaseHTTPRequestHandler):
    """Handler for mock web search requests."""

    def log_message(self, format, *args):
        print(f"[Mock Search] {args[0]}")

    def send_json_response(self, data: Dict[str, Any], status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_json_response({"status": "healthy"})
            return

        # Parse query from URL: /search?q=query
        parsed = urlparse(self.path)
        if parsed.path == "/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].lower()
            self._handle_search(query)
        else:
            self.send_json_response({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/search":
            self.send_json_response({"error": "Not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body)
            query = request.get("query", "").lower()
            self._handle_search(query)
        except json.JSONDecodeError as e:
            self.send_json_response({"error": f"Invalid JSON: {e}"}, 400)

    def _handle_search(self, query: str):
        # Find matching context
        result_key = "default"
        for key in SEARCH_CONTEXTS:
            if key != "default" and key in query:
                result_key = key
                break

        result = SEARCH_CONTEXTS[result_key]
        print(f"[Mock Search] Query: '{query}' -> Using '{result_key}' context")

        self.send_json_response(
            {
                "query": query,
                "results": [
                    {
                        "title": result["source"],
                        "url": result["url"],
                        "snippet": result["context"],
                    }
                ],
                "context": result["context"],  # Convenient field for direct use
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="Mock web search for hallucination demo"
    )
    parser.add_argument("--port", type=int, default=8003, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockWebSearchHandler)
    print(f"[Mock Search] Starting server on {args.host}:{args.port}")
    print(f"[Mock Search] Endpoints:")
    print(f"  - GET  /health")
    print(f"  - GET  /search?q=<query>")
    print(f'  - POST /search {{"query": "..."}}')
    print(f"[Mock Search] Available topics: {list(SEARCH_CONTEXTS.keys())}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Mock Search] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
