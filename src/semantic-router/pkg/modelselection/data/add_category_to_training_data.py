#!/usr/bin/env python3
"""
Add category field to training data using VSR's category classifier.

This script:
1. Reads training_data.jsonl (50,544 records)
2. Groups by embedding_id (5,616 unique queries)
3. Calls VSR's /api/v1/classify/intent for each unique query
4. Adds category field to all records
5. Outputs enriched training data

Usage:
    # Start VSR first, then run:
    python add_category_to_training_data.py --vsr-url http://localhost:8080
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


def classify_with_vsr(text: str, vsr_url: str, timeout: int = 30) -> Optional[dict]:
    """Call VSR's classification API."""
    try:
        import requests
    except ImportError:
        print("Error: requests library required. Install with: pip install requests")
        sys.exit(1)

    try:
        response = requests.post(
            f"{vsr_url}/api/v1/classify/intent",
            json={"text": text},
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            return {
                "category": data.get("classification", {}).get("category", "other"),
                "confidence": data.get("classification", {}).get("confidence", 0.0),
            }
        else:
            print(f"  Error: API returned {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  Error: API call failed: {e}")
        return None


def test_vsr_connection(vsr_url: str) -> bool:
    """Test connection to VSR API."""
    try:
        import requests
    except ImportError:
        print("Error: requests library required. Install with: pip install requests")
        sys.exit(1)

    try:
        response = requests.get(f"{vsr_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add category field to training data using VSR classifier"
    )
    parser.add_argument(
        "--vsr-url",
        default="http://localhost:8080",
        help="VSR API URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--input",
        default="training_data.jsonl",
        help="Input file (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="training_data_with_category.jsonl",
        help="Output file (default: training_data_with_category.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Print progress every N queries (default: 100)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per query on failure (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Delay between retries in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("Add Category to Training Data using VSR Classifier")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"VSR URL: {args.vsr_url}")
    print()

    # Test VSR connection
    print("Testing VSR connection...")
    if not test_vsr_connection(args.vsr_url):
        print(f"Error: Cannot connect to VSR at {args.vsr_url}")
        print("Please ensure VSR is running with the category classifier enabled.")
        print()
        print("To start VSR:")
        print("  make run-router")
        print()
        sys.exit(1)
    print("  VSR connection OK!")
    print()

    # Step 1: Load and group records by embedding_id
    print("Step 1: Loading training data...")
    records_by_embedding = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            embedding_id = record.get("embedding_id", 0)
            records_by_embedding[embedding_id].append(record)

    total_records = sum(len(recs) for recs in records_by_embedding.values())
    unique_queries = len(records_by_embedding)

    print(f"  Total records: {total_records}")
    print(f"  Unique queries (embedding_id): {unique_queries}")
    print()

    # Step 2: Classify each unique query using VSR
    print("Step 2: Classifying queries with VSR...")
    category_cache = {}  # embedding_id -> category
    category_counts = defaultdict(int)
    failed_queries = []

    start_time = time.time()

    for idx, (embedding_id, recs) in enumerate(records_by_embedding.items()):
        # Get query text from first record
        query = recs[0].get("query", "")

        # Classify with retries
        result = None
        for retry in range(args.max_retries):
            result = classify_with_vsr(query, args.vsr_url)
            if result is not None:
                break
            if retry < args.max_retries - 1:
                time.sleep(args.retry_delay)

        if result is None:
            # Failed after all retries
            failed_queries.append(
                {
                    "embedding_id": embedding_id,
                    "query": query[:100] + "..." if len(query) > 100 else query,
                }
            )
            category = "other"  # Mark as 'other' if classification fails
        else:
            category = result["category"]

        category_cache[embedding_id] = category
        category_counts[category] += len(recs)

        # Progress update
        if (idx + 1) % args.batch_size == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (unique_queries - idx - 1) / rate if rate > 0 else 0
            print(
                f"  Processed {idx + 1}/{unique_queries} queries "
                f"({(idx + 1) / unique_queries * 100:.1f}%) "
                f"ETA: {remaining:.0f}s"
            )

    elapsed_total = time.time() - start_time
    print(f"  Done! Classified {unique_queries} queries in {elapsed_total:.1f}s")

    if failed_queries:
        print(f"  Warning: {len(failed_queries)} queries failed classification")
    print()

    # Step 3: Write output with category field
    print("Step 3: Writing output file...")
    records_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for embedding_id, recs in records_by_embedding.items():
            category = category_cache.get(embedding_id, "other")

            for record in recs:
                # Add category field
                record["category"] = category
                f.write(json.dumps(record) + "\n")
                records_written += 1

    print(f"  Written {records_written} records to {output_path}")
    print()

    # Step 4: Show category distribution
    print("Step 4: Category Distribution (14 VSR Categories)")
    print("-" * 50)

    # All 14 VSR categories
    all_categories = [
        "biology",
        "business",
        "chemistry",
        "computer science",
        "economics",
        "engineering",
        "health",
        "history",
        "law",
        "math",
        "other",
        "philosophy",
        "physics",
        "psychology",
    ]

    for category in all_categories:
        count = category_counts.get(category, 0)
        pct = count / total_records * 100 if total_records > 0 else 0
        bar = "█" * int(pct / 2)
        status = "✓" if count > 0 else " "
        print(f"  {status} {category:20s} {count:6d} ({pct:5.1f}%) {bar}")

    # Show any unexpected categories
    unexpected = set(category_counts.keys()) - set(all_categories)
    if unexpected:
        print()
        print("  Unexpected categories:")
        for cat in unexpected:
            count = category_counts[cat]
            print(f"    {cat}: {count}")

    print()
    print(f"Done! Output file: {output_path}")
    print()

    # Show sample record
    print("Sample record (first line):")
    print("-" * 50)
    with open(output_path, "r") as f:
        sample = json.loads(f.readline())
        print(f"  query:       {sample['query'][:60]}...")
        print(f"  task_name:   {sample['task_name']}")
        print(f"  category:    {sample['category']}")
        print(f"  model_name:  {sample['model_name']}")
        print(f"  performance: {sample['performance']}")

    # Report failures if any
    if failed_queries:
        print()
        print(f"Failed queries ({len(failed_queries)}):")
        for fq in failed_queries[:5]:
            print(f"  - embedding_id={fq['embedding_id']}: {fq['query']}")
        if len(failed_queries) > 5:
            print(f"  ... and {len(failed_queries) - 5} more")


if __name__ == "__main__":
    main()
