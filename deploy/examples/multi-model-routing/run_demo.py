#!/usr/bin/env python3

import requests
import time
from typing import Dict

ROUTER_API_URL = "http://localhost:8888/v1/chat/completions"

# Demo queries grouped by category
DEMO_QUERIES = {
    "Math": [
        "What is the derivative of x^3 + 2x?",
        "Solve the equation: 2x + 5 = 15",
        "Calculate the area of a circle with radius 7.5 cm",
    ],
    "Physics": [
        "Explain Newton's second law of motion",
        "What is the speed of light in vacuum?",
    ],
    "Chemistry": [
        "What is the molecular structure of water?",
        "Explain the difference between ionic and covalent bonds",
    ],
    "Computer Science": [
        "Write a Python function to implement binary search",
        "Explain the difference between stack and queue",
        "What is Big O notation?",
    ],
    "History": [
        "When did World War 2 end?",
        "Who was Julius Caesar and why was he important?",
    ],
    "Literature": [
        "Summarize Romeo and Juliet in 3 sentences",
        "Write a short poem about autumn",
    ],
    "Business": [
        "What is a SWOT analysis?",
        "Explain supply and demand",
    ],
    "General": [
        "Tell me a short story about a robot learning to paint",
        "What's the best way to learn a new language?",
    ],
}


def send_query(query: str) -> Dict:
    """Send query to router and return response"""
    payload = {
        "model": "MoM",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "max_tokens": 150,
    }

    start = time.time()
    resp = requests.post(
        ROUTER_API_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    elapsed = time.time() - start

    resp.raise_for_status()
    data = resp.json()

    return {
        "answer": data["choices"][0]["message"]["content"].strip(),
        "model": data.get("model", "unknown"),
        "latency": elapsed,
        "cached": resp.headers.get("x-vsr-cache-hit") == "true",
    }


def main():
    print("Semantic Router Demo - Multi-Model Routing")

    total = sum(len(queries) for queries in DEMO_QUERIES.values())
    count = 0

    # Test category-based routing
    for category, queries in DEMO_QUERIES.items():
        print(f"\n{category}:")
        for query in queries:
            count += 1
            try:
                info = send_query(query)
                cache = "[CACHED] " if info["cached"] else ""
                print(
                    f"\n[{count}/{total}] {cache}Model: {info['model']} ({info['latency']:.1f}s)"
                )
                print(f"Q: {query}")
                # Only showing the first 200 chars of answer
                answer = info["answer"][:200]
                print(f"A: {answer}\n")
            except Exception as e:
                print(f"\n[{count}/{total}] Error: {e}")

    # Semantic caching
    print("Testing Semantic Cache")
    test_query = "What is 5 + 10?"
    print(f"Q: {test_query}")
    print("\n  First request...")
    info1 = send_query(test_query)
    print(f"{info1['model']} - {info1['latency']:.2f}s")

    time.sleep(1)

    print("\n  Second request (same query)...")
    info2 = send_query(test_query)
    print(f"{info2['model']} - {info2['latency']:.2f}s")

    if info2["cached"]:
        speedup = info1["latency"] / info2["latency"]
        print(f"Cache hit! {speedup:.1f}x faster")
    else:
        print("  No cache hit detected")


if __name__ == "__main__":
    main()
