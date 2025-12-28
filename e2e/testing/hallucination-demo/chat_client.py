#!/usr/bin/env python3
"""
Interactive chat client with tool calling for hallucination detection demo.

This client:
1. Sends user questions to the semantic router (via Envoy)
2. Handles tool_calls by executing mock web search
3. Sends tool results back to complete the conversation
4. Displays hallucination detection headers from the response

Usage:
    python chat_client.py --router-url http://localhost:8801 --search-url http://localhost:8003
"""

import argparse
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple


# ANSI colors
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


def print_step(step: int, text: str):
    print(f"{Colors.YELLOW}[Step {step}]{Colors.END} {text}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.END} {text}")


class HallucinationDemoClient:
    """Chat client with tool calling support for hallucination demo."""

    def __init__(self, router_url: str, search_url: str, model: str = "qwen3"):
        self.router_url = router_url.rstrip("/")
        self.search_url = search_url.rstrip("/")
        self.model = model
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    def web_search(self, query: str) -> str:
        """Execute mock web search."""
        try:
            data = json.dumps({"query": query}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.search_url}/search",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("context", "No results found.")
        except urllib.error.HTTPError as e:
            return f"Search error: HTTP {e.code}"
        except Exception as e:
            return f"Search error: {e}"

    def chat(self, question: str) -> Tuple[str, Dict[str, str]]:
        """
        Send a chat message and handle tool calls.
        Returns: (response_content, hallucination_headers)
        """
        messages = [{"role": "user", "content": question}]

        print_step(1, f"Sending question to router: {question[:50]}...")

        # First request - may get tool_calls
        resp = self._send_request(messages, include_tools=True)
        if not resp:
            return "Error: Failed to get response", {}

        response_data, headers = resp
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Check if we got tool_calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            print_success(f"Got {len(tool_calls)} tool call(s)")

            # Add assistant message with tool_calls
            messages.append(message)

            # Execute each tool call
            for tc in tool_calls:
                func = tc.get("function", {})
                func_name = func.get("name", "")
                func_args = json.loads(func.get("arguments", "{}"))

                print_step(2, f"Executing tool: {func_name}({func_args})")

                if func_name == "web_search":
                    result = self.web_search(func_args.get("query", ""))
                    print_success(f"Got search context: {len(result)} chars")
                    print(
                        f"   {Colors.CYAN}Context preview:{Colors.END} {result[:100]}..."
                    )
                else:
                    result = f"Unknown tool: {func_name}"

                # Add tool result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result,
                    }
                )

            # Second request - with tool results
            print_step(3, "Sending tool results to get final response...")
            resp = self._send_request(messages, include_tools=False)
            if not resp:
                return "Error: Failed to get final response", {}

            response_data, headers = resp
            choice = response_data.get("choices", [{}])[0]
            message = choice.get("message", {})

        content = message.get("content", "")
        return content, headers

    def _send_request(
        self, messages: List[Dict], include_tools: bool
    ) -> Optional[Tuple[Dict, Dict]]:
        """Send request to router."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        if include_tools:
            payload["tools"] = self.tools

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.router_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                # Extract hallucination-related headers
                hal_headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if "hallucination" in k.lower()
                    or "fact-check" in k.lower()
                    or "unverified" in k.lower()
                }

                response_data = json.loads(resp.read().decode("utf-8"))
                return response_data, hal_headers
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")[:200] if e.fp else ""
            print_error(f"HTTP {e.code}: {body}")
            return None
        except Exception as e:
            print_error(f"Request failed: {e}")
            return None


def print_hallucination_results(headers: Dict[str, str], response: str):
    """Print hallucination detection results."""
    print(f"\n{Colors.BOLD}{'─'*60}{Colors.END}")
    print(f"{Colors.BOLD}LLM Response:{Colors.END}")
    print(f"  {response[:200]}{'...' if len(response) > 200 else ''}")

    print(f"\n{Colors.BOLD}Hallucination Detection Results:{Colors.END}")

    if not headers:
        print(f"  {Colors.YELLOW}No hallucination headers in response{Colors.END}")
        print(f"  (Headers may be returned as trailers in streaming mode)")
    else:
        for key, value in headers.items():
            if "detected" in key.lower() and value.lower() == "true":
                print(f"  {Colors.RED}🚨 {key}: {value}{Colors.END}")
            elif "detected" in key.lower():
                print(f"  {Colors.GREEN}✓ {key}: {value}{Colors.END}")
            else:
                print(f"  {Colors.CYAN}• {key}: {value}{Colors.END}")

    print(f"{'─'*60}\n")


def interactive_mode(client: HallucinationDemoClient):
    """Run interactive chat mode."""
    print_header("Hallucination Detection Demo")
    print("Type your questions to test hallucination detection.")
    print("The demo will:")
    print("  1. Send your question to the mock LLM")
    print("  2. Execute web_search tool calls")
    print("  3. Display the response with hallucination detection headers")
    print()
    print("Try questions like:")
    print(f"  {Colors.CYAN}• When was the Eiffel Tower built?{Colors.END}")
    print(f"  {Colors.CYAN}• What is the population of Tokyo?{Colors.END}")
    print(f"  {Colors.CYAN}• Who founded Apple?{Colors.END}")
    print()
    print(
        f"Type {Colors.YELLOW}'quit'{Colors.END} or {Colors.YELLOW}'q'{Colors.END} to exit.\n"
    )

    while True:
        try:
            question = input(f"{Colors.GREEN}You:{Colors.END} ").strip()
            if not question:
                continue
            if question.lower() in ["quit", "q", "exit"]:
                print(f"\n{Colors.CYAN}Goodbye!{Colors.END}\n")
                break

            print()
            response, headers = client.chat(question)
            print_hallucination_results(headers, response)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Goodbye!{Colors.END}\n")
            break
        except EOFError:
            break


def demo_mode(client: HallucinationDemoClient):
    """Run predefined demo questions."""
    print_header("Hallucination Detection Demo - Auto Mode")

    demo_questions = [
        "When was the Eiffel Tower built and how tall is it?",
        "What is the population of Tokyo?",
        "Who founded Apple Inc?",
    ]

    for i, question in enumerate(demo_questions, 1):
        print(f"\n{Colors.BOLD}Demo Question {i}/{len(demo_questions)}:{Colors.END}")
        print(f"{Colors.GREEN}You:{Colors.END} {question}\n")

        response, headers = client.chat(question)
        print_hallucination_results(headers, response)

        if i < len(demo_questions):
            input(f"Press Enter to continue to next question...")


def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Demo Client")
    parser.add_argument(
        "--router-url",
        type=str,
        default="http://localhost:8801",
        help="Semantic Router URL (default: http://localhost:8801)",
    )
    parser.add_argument(
        "--search-url",
        type=str,
        default="http://localhost:8003",
        help="Mock Web Search URL (default: http://localhost:8003)",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3", help="Model name to use (default: qwen3)"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run predefined demo questions"
    )
    args = parser.parse_args()

    client = HallucinationDemoClient(
        router_url=args.router_url, search_url=args.search_url, model=args.model
    )

    if args.demo:
        demo_mode(client)
    else:
        interactive_mode(client)


if __name__ == "__main__":
    main()
