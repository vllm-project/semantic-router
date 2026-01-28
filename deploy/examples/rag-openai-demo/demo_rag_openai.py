#!/usr/bin/env python3
"""
OpenAI RAG Demo Application

This demo showcases the OpenAI RAG functionality integrated with Semantic Router.
It demonstrates the complete workflow from file upload to RAG-enhanced responses.

Usage:
    python3 demo_rag_openai.py

Requirements:
    - OPENAI_API_KEY environment variable
    - SEMANTIC_ROUTER_URL environment variable (default: http://localhost:8080)
"""

import json
import os
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEMANTIC_ROUTER_URL = os.getenv("SEMANTIC_ROUTER_URL", "http://localhost:8080")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_step(step: int, text: str):
    """Print a formatted step"""
    print(f"{Colors.OKCYAN}[Step {step}]{Colors.ENDC} {text}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ{Colors.ENDC} {text}")


def create_sample_pdf_content() -> bytes:
    """Create a sample PDF content for demo purposes"""
    # In a real scenario, you would read from an actual PDF file
    # This is a simplified example
    sample_text = """
    Semantic Router Documentation

    Overview:
    The Semantic Router is an intelligent routing system that uses machine learning
    to route requests to the most appropriate LLM model based on the content and
    context of the request.

    Key Features:
    1. Signal-driven routing decisions
    2. Multi-model support
    3. RAG (Retrieval-Augmented Generation) integration
    4. Semantic caching
    5. Hallucination detection

    Configuration:
    The router uses YAML configuration files to define routing rules, signals,
    and plugin configurations.
    """
    return sample_text.encode("utf-8")


def upload_file_to_openai(
    filename: str, content: bytes, purpose: str = "assistants"
) -> Dict:
    """Upload a file to OpenAI File Store"""
    print_info(f"Uploading {filename} to OpenAI File Store...")

    try:
        # Create a temporary file for upload
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Upload file
        with open(tmp_file_path, "rb") as file:
            file_obj = openai_client.files.create(file=file, purpose=purpose)

        # Clean up temp file
        os.unlink(tmp_file_path)

        print_success(f"File uploaded: {file_obj.id} ({file_obj.bytes} bytes)")
        return {
            "id": file_obj.id,
            "filename": file_obj.filename,
            "bytes": file_obj.bytes,
            "purpose": file_obj.purpose,
        }
    except Exception as e:
        print_error(f"Failed to upload file: {e}")
        raise


def create_vector_store(name: str, file_ids: Optional[List[str]] = None) -> str:
    """Create a vector store and attach files"""
    print_info(f"Creating vector store: {name}")

    try:
        vector_store = openai_client.beta.vector_stores.create(
            name=name, file_ids=file_ids or []
        )

        print_success(f"Vector store created: {vector_store.id}")

        # Wait for files to be processed
        if file_ids:
            print_info("Waiting for files to be processed...")
            max_wait = 60  # seconds
            start_time = time.time()

            while time.time() - start_time < max_wait:
                vs = openai_client.beta.vector_stores.retrieve(vector_store.id)
                if vs.file_counts.completed == len(file_ids):
                    print_success("All files processed!")
                    break
                time.sleep(2)
            else:
                print_warning("Timeout waiting for file processing")

        return vector_store.id
    except Exception as e:
        print_error(f"Failed to create vector store: {e}")
        raise


def send_chat_request(
    query: str, vector_store_id: str, mode: str = "direct_search"
) -> Dict:
    """Send a chat request to Semantic Router with RAG"""
    print_info(f"Sending request with RAG ({mode} mode)...")

    headers = {
        "Content-Type": "application/json",
        "X-VSR-Selected-Decision": "rag-openai-decision",  # Decision name with RAG config
    }

    request_body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": query}],
    }

    # For tool_based mode, the router will add file_search tool
    # For direct_search mode, the router will retrieve context first

    try:
        response = requests.post(
            f"{SEMANTIC_ROUTER_URL}/v1/chat/completions",
            headers=headers,
            json=request_body,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print_error(f"Request failed: {e}")
        raise


def compare_responses(query: str, vector_store_id: str):
    """Compare responses with and without RAG"""
    print_header("Response Comparison")

    print_step(1, "Request WITHOUT RAG (baseline)")
    try:
        # Send request without RAG (use a different decision or no decision header)
        headers = {"Content-Type": "application/json"}
        request_body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": query}],
        }

        response = requests.post(
            f"{SEMANTIC_ROUTER_URL}/v1/chat/completions",
            headers=headers,
            json=request_body,
            timeout=60,
        )
        response.raise_for_status()
        baseline = response.json()

        print_success("Baseline response received")
        print(f"\n{Colors.WARNING}Baseline Response:{Colors.ENDC}")
        print(baseline["choices"][0]["message"]["content"][:200] + "...")
    except Exception as e:
        print_error(f"Baseline request failed: {e}")
        baseline = None

    print_step(2, "Request WITH RAG (direct_search mode)")
    try:
        rag_response = send_chat_request(query, vector_store_id, mode="direct_search")
        print_success("RAG response received")
        print(f"\n{Colors.OKGREEN}RAG Response:{Colors.ENDC}")
        print(rag_response["choices"][0]["message"]["content"][:200] + "...")
    except Exception as e:
        print_error(f"RAG request failed: {e}")
        rag_response = None

    if baseline and rag_response:
        print_info("Compare the responses above to see the difference RAG makes!")


def cleanup_resources(file_ids: List[str], vector_store_id: str):
    """Clean up uploaded files and vector store"""
    print_header("Cleanup")

    # Delete files
    for file_id in file_ids:
        try:
            openai_client.files.delete(file_id)
            print_success(f"Deleted file: {file_id}")
        except Exception as e:
            print_error(f"Failed to delete file {file_id}: {e}")

    # Delete vector store
    try:
        openai_client.beta.vector_stores.delete(vector_store_id)
        print_success(f"Deleted vector store: {vector_store_id}")
    except Exception as e:
        print_error(f"Failed to delete vector store: {e}")


def main():
    """Main demo function"""
    print_header("OpenAI RAG Demo Application")

    if not OPENAI_API_KEY:
        print_error("OPENAI_API_KEY environment variable not set!")
        return

    print_info(f"Semantic Router URL: {SEMANTIC_ROUTER_URL}")
    print_info(f"OpenAI Base URL: {OPENAI_BASE_URL}")

    file_ids = []
    vector_store_id = None

    try:
        # Step 1: Upload files
        print_header("Step 1: Upload Files to OpenAI File Store")
        print_step(1, "Creating sample PDF content...")
        sample_content = create_sample_pdf_content()

        print_step(2, "Uploading file to OpenAI...")
        file_info = upload_file_to_openai("semantic_router_docs.txt", sample_content)
        file_ids.append(file_info["id"])

        # Step 2: Create vector store
        print_header("Step 2: Create Vector Store")
        vector_store_id = create_vector_store("Demo Vector Store", file_ids)

        # Step 3: Test RAG queries
        print_header("Step 3: Test RAG Queries")

        test_queries = [
            "What is the Semantic Router?",
            "What are the key features?",
            "How do I configure the router?",
        ]

        for i, query in enumerate(test_queries, 1):
            print_step(i, f"Query: {query}")
            try:
                response = send_chat_request(
                    query, vector_store_id, mode="direct_search"
                )
                answer = response["choices"][0]["message"]["content"]
                print_success(f"Answer: {answer[:100]}...")
            except Exception as e:
                print_error(f"Query failed: {e}")

        # Step 4: Compare responses
        compare_responses("What is the Semantic Router?", vector_store_id)

        print_header("Demo Complete!")
        print_success("All steps completed successfully")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print_error(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        if file_ids and vector_store_id:
            response = input("\nClean up resources? (y/n): ")
            if response.lower() == "y":
                cleanup_resources(file_ids, vector_store_id)


if __name__ == "__main__":
    main()
