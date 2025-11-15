#!/usr/bin/env python3
"""
PII Confidence Benchmark Tool

Tests a comprehensive set of PII and non-PII prompts, measuring:
- Confidence scores for each entity detected
- Processing time per prompt
- Detection success rates

Outputs detailed tables and statistics for analysis.
"""

import requests
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import statistics

CLASSIFICATION_API_URL = "http://localhost:8080/api/v1"
PII_ENDPOINT = "/classify/pii"


@dataclass
class BenchmarkResult:
    prompt: str
    category: str
    expected_pii: bool
    has_pii: bool
    max_confidence: float
    entities_detected: List[Dict[str, Any]]
    processing_time_ms: float
    error: str = None


# Comprehensive test prompts covering various PII types and formats
BENCHMARK_PROMPTS = [
    # === EMAIL ADDRESSES ===
    {"text": "john@example.com", "category": "Email", "has_pii": True},
    {"text": "john.doe@example.com", "category": "Email", "has_pii": True},
    {"text": "john.smith@example.com", "category": "Email", "has_pii": True},
    {"text": "jane.doe@company.co.uk", "category": "Email", "has_pii": True},
    {"text": "user123@test.com", "category": "Email", "has_pii": True},
    {"text": "support@example.org", "category": "Email", "has_pii": True},
    {"text": "admin@domain.net", "category": "Email", "has_pii": True},
    {
        "text": "Contact me at support@example.org for help",
        "category": "Email",
        "has_pii": True,
    },
    {
        "text": "Send to john@example.com and jane@example.com",
        "category": "Email",
        "has_pii": True,
    },
    {
        "text": "Email us at info@company.com for more details",
        "category": "Email",
        "has_pii": True,
    },
    # === SSN (Social Security Numbers) ===
    {"text": "123-45-6789", "category": "SSN", "has_pii": True},
    {"text": "987-65-4321", "category": "SSN", "has_pii": True},
    {"text": "456-78-9012", "category": "SSN", "has_pii": True},
    {"text": "123456789", "category": "SSN", "has_pii": True},
    {"text": "My SSN is 123-45-6789", "category": "SSN", "has_pii": True},
    {
        "text": "My social security number is 987-65-4321",
        "category": "SSN",
        "has_pii": True,
    },
    {"text": "SSN: 456-78-9012", "category": "SSN", "has_pii": True},
    {"text": "Please verify SSN 111-22-3333", "category": "SSN", "has_pii": True},
    # === CREDIT CARDS ===
    {"text": "4111-1111-1111-1111", "category": "Credit Card", "has_pii": True},
    {"text": "4532-1234-5678-9012", "category": "Credit Card", "has_pii": True},
    {"text": "5500-0000-0000-0004", "category": "Credit Card", "has_pii": True},
    {"text": "4111 1111 1111 1111", "category": "Credit Card", "has_pii": True},
    {"text": "4532 1234 5678 9012", "category": "Credit Card", "has_pii": True},
    {"text": "4111111111111111", "category": "Credit Card", "has_pii": True},
    {"text": "4532123456789012", "category": "Credit Card", "has_pii": True},
    {"text": "5500000000000004", "category": "Credit Card", "has_pii": True},
    {
        "text": "Card number 4111-1111-1111-1111",
        "category": "Credit Card",
        "has_pii": True,
    },
    {
        "text": "My card number is 4532-1234-5678-9012 and expires 12/25",
        "category": "Credit Card",
        "has_pii": True,
    },
    {
        "text": "Payment card: 4111111111111111 exp 03/26",
        "category": "Credit Card",
        "has_pii": True,
    },
    # === PHONE NUMBERS ===
    {"text": "(555) 123-4567", "category": "Phone", "has_pii": True},
    {"text": "555-123-4567", "category": "Phone", "has_pii": True},
    {"text": "555.123.4567", "category": "Phone", "has_pii": True},
    {"text": "555 123 4567", "category": "Phone", "has_pii": True},
    {"text": "+1-555-123-4567", "category": "Phone", "has_pii": True},
    {"text": "+1 (555) 123-4567", "category": "Phone", "has_pii": True},
    {"text": "5551234567", "category": "Phone", "has_pii": True},
    {"text": "1-800-555-1234", "category": "Phone", "has_pii": True},
    {
        "text": "Call me at (555) 123-4567 for more info",
        "category": "Phone",
        "has_pii": True,
    },
    {
        "text": "Phone: 555-123-4567 or 555-765-4321",
        "category": "Phone",
        "has_pii": True,
    },
    # === PERSON NAMES ===
    {"text": "John Smith", "category": "Person", "has_pii": True},
    {"text": "Jane Doe", "category": "Person", "has_pii": True},
    {"text": "John Q. Smith", "category": "Person", "has_pii": True},
    {"text": "Dr. Jane Doe", "category": "Person", "has_pii": True},
    {"text": "Mr. Robert Johnson", "category": "Person", "has_pii": True},
    {
        "text": "Meeting with John Smith and Jane Doe",
        "category": "Person",
        "has_pii": True,
    },
    {
        "text": "Contact Sarah Williams for details",
        "category": "Person",
        "has_pii": True,
    },
    # === ADDRESSES ===
    {"text": "123 Main Street", "category": "Address", "has_pii": True},
    {"text": "456 Oak Ave", "category": "Address", "has_pii": True},
    {"text": "789 Elm Road, Apt 5B", "category": "Address", "has_pii": True},
    {
        "text": "123 Main Street, New York, NY 10001",
        "category": "Address",
        "has_pii": True,
    },
    {
        "text": "456 Oak Ave, Los Angeles, CA 90001",
        "category": "Address",
        "has_pii": True,
    },
    {
        "text": "1600 Pennsylvania Avenue NW, Washington, DC 20500",
        "category": "Address",
        "has_pii": True,
    },
    # === LOCATIONS (GPE - Geo-Political Entities) ===
    {"text": "New York", "category": "Location", "has_pii": True},
    {"text": "Los Angeles", "category": "Location", "has_pii": True},
    {"text": "United States", "category": "Location", "has_pii": True},
    {"text": "London", "category": "Location", "has_pii": True},
    {"text": "Tokyo", "category": "Location", "has_pii": True},
    # === ORGANIZATIONS ===
    {"text": "Apple Inc.", "category": "Organization", "has_pii": True},
    {"text": "Microsoft Corporation", "category": "Organization", "has_pii": True},
    {"text": "Google LLC", "category": "Organization", "has_pii": True},
    {"text": "Amazon.com", "category": "Organization", "has_pii": True},
    # === DATES ===
    {"text": "12/31/2023", "category": "Date", "has_pii": True},
    {"text": "01/15/2024", "category": "Date", "has_pii": True},
    {"text": "January 1, 2024", "category": "Date", "has_pii": True},
    {"text": "March 15th, 2023", "category": "Date", "has_pii": True},
    {"text": "Born on 05/20/1990", "category": "Date", "has_pii": True},
    # === MIXED PII (Multiple types in one prompt) ===
    {
        "text": "Call 555-1234 or email test@example.com",
        "category": "Mixed",
        "has_pii": True,
    },
    {
        "text": "John Smith, SSN 123-45-6789, lives at 123 Main St",
        "category": "Mixed",
        "has_pii": True,
    },
    {
        "text": "Contact: jane.doe@example.com, Phone: (555) 123-4567",
        "category": "Mixed",
        "has_pii": True,
    },
    {
        "text": "Card 4111-1111-1111-1111 belongs to John Doe at 456 Oak Ave",
        "category": "Mixed",
        "has_pii": True,
    },
    # === NON-PII (Should NOT detect PII) ===
    {
        "text": "The quick brown fox jumps over the lazy dog",
        "category": "Non-PII",
        "has_pii": False,
    },
    {"text": "Hello world", "category": "Non-PII", "has_pii": False},
    {"text": "What is the weather today?", "category": "Non-PII", "has_pii": False},
    {
        "text": "How do I solve this math problem?",
        "category": "Non-PII",
        "has_pii": False,
    },
    {"text": "Tell me about machine learning", "category": "Non-PII", "has_pii": False},
    {"text": "12345", "category": "Non-PII", "has_pii": False},
    {"text": "abc def ghi", "category": "Non-PII", "has_pii": False},
    {"text": "What time is it?", "category": "Non-PII", "has_pii": False},
    {"text": "Explain quantum physics", "category": "Non-PII", "has_pii": False},
    {"text": "Recipe for chocolate cake", "category": "Non-PII", "has_pii": False},
    # === EDGE CASES (Ambiguous) ===
    {"text": "at", "category": "Edge Case", "has_pii": False},
    {"text": "@", "category": "Edge Case", "has_pii": False},
    {"text": "123", "category": "Edge Case", "has_pii": False},
    {"text": "test test test", "category": "Edge Case", "has_pii": False},
]


def run_benchmark() -> List[BenchmarkResult]:
    """Run benchmark on all test prompts"""
    results = []

    print(f"\n{'='*100}")
    print(f"PII CONFIDENCE BENCHMARK")
    print(f"{'='*100}")
    print(f"Testing {len(BENCHMARK_PROMPTS)} prompts...")
    print(f"{'='*100}\n")

    for i, prompt_data in enumerate(BENCHMARK_PROMPTS, 1):
        prompt = prompt_data["text"]
        category = prompt_data["category"]
        expected_pii = prompt_data["has_pii"]

        # Progress indicator
        if i % 10 == 0:
            print(f"Progress: {i}/{len(BENCHMARK_PROMPTS)} prompts tested...")

        try:
            payload = {"text": prompt}

            start_time = time.time()
            response = requests.post(
                f"{CLASSIFICATION_API_URL}{PII_ENDPOINT}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10,
            )
            end_time = time.time()

            processing_time_ms = (end_time - start_time) * 1000

            result_data = response.json()
            has_pii = result_data.get("has_pii", False)
            entities = result_data.get("entities", [])

            # Get max confidence from all entities
            max_confidence = 0.0
            if entities:
                max_confidence = max(e.get("confidence", 0.0) for e in entities)

            # Use API's processing time if available, otherwise use our measured time
            api_processing_time = result_data.get(
                "processing_time_ms", processing_time_ms
            )

            result = BenchmarkResult(
                prompt=prompt,
                category=category,
                expected_pii=expected_pii,
                has_pii=has_pii,
                max_confidence=max_confidence,
                entities_detected=entities,
                processing_time_ms=api_processing_time,
            )

        except Exception as e:
            result = BenchmarkResult(
                prompt=prompt,
                category=category,
                expected_pii=expected_pii,
                has_pii=False,
                max_confidence=0.0,
                entities_detected=[],
                processing_time_ms=0.0,
                error=str(e),
            )

        results.append(result)

    print(f"\nCompleted testing {len(BENCHMARK_PROMPTS)} prompts.\n")
    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print detailed results table"""
    print(f"\n{'='*150}")
    print(f"DETAILED RESULTS")
    print(f"{'='*150}")

    # Table header
    print(
        f"{'#':<4} {'Category':<15} {'Prompt':<50} {'Confidence':<12} {'Time (ms)':<12} {'Status':<10}"
    )
    print(f"{'-'*150}")

    for i, result in enumerate(results, 1):
        # Truncate long prompts
        prompt_display = (
            result.prompt[:47] + "..." if len(result.prompt) > 50 else result.prompt
        )

        # Status: ✅ correct detection, ❌ missed/false positive, ⚠️ error
        if result.error:
            status = "⚠️ ERROR"
        elif result.expected_pii == result.has_pii:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        confidence_str = (
            f"{result.max_confidence:.4f}" if result.max_confidence > 0 else "N/A"
        )
        time_str = (
            f"{result.processing_time_ms:.1f}"
            if result.processing_time_ms > 0
            else "N/A"
        )

        print(
            f"{i:<4} {result.category:<15} {prompt_display:<50} {confidence_str:<12} {time_str:<12} {status:<10}"
        )

    print(f"{'='*150}\n")


def print_statistics(results: List[BenchmarkResult]):
    """Print comprehensive statistics"""
    print(f"\n{'='*100}")
    print(f"STATISTICS SUMMARY")
    print(f"{'='*100}\n")

    # Overall metrics
    total = len(results)
    errors = sum(1 for r in results if r.error)
    correct = sum(1 for r in results if r.expected_pii == r.has_pii and not r.error)
    incorrect = total - correct - errors

    # PII detection metrics
    expected_pii_count = sum(1 for r in results if r.expected_pii)
    detected_pii_count = sum(1 for r in results if r.has_pii)
    true_positives = sum(1 for r in results if r.expected_pii and r.has_pii)
    false_positives = sum(1 for r in results if not r.expected_pii and r.has_pii)
    false_negatives = sum(1 for r in results if r.expected_pii and not r.has_pii)
    true_negatives = sum(1 for r in results if not r.expected_pii and not r.has_pii)

    # Confidence statistics
    confidences = [r.max_confidence for r in results if r.max_confidence > 0]
    processing_times = [
        r.processing_time_ms for r in results if r.processing_time_ms > 0
    ]

    print(f"📊 Overall Performance:")
    print(f"   Total Prompts:        {total}")
    print(f"   ✅ Correct:           {correct} ({correct/total*100:.1f}%)")
    print(f"   ❌ Incorrect:         {incorrect} ({incorrect/total*100:.1f}%)")
    print(f"   ⚠️  Errors:            {errors} ({errors/total*100:.1f}%)")

    print(f"\n🎯 Detection Accuracy:")
    print(f"   Expected PII:         {expected_pii_count}")
    print(f"   Detected PII:         {detected_pii_count}")
    print(f"   True Positives:       {true_positives}")
    print(f"   False Positives:      {false_positives}")
    print(f"   False Negatives:      {false_negatives}")
    print(f"   True Negatives:       {true_negatives}")

    if expected_pii_count > 0:
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = true_positives / expected_pii_count
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\n📈 Classification Metrics:")
        print(f"   Precision:            {precision:.3f} ({precision*100:.1f}%)")
        print(f"   Recall:               {recall:.3f} ({recall*100:.1f}%)")
        print(f"   F1 Score:             {f1_score:.3f}")

    if confidences:
        print(f"\n💯 Confidence Scores:")
        print(f"   Mean:                 {statistics.mean(confidences):.4f}")
        print(f"   Median:               {statistics.median(confidences):.4f}")
        print(f"   Min:                  {min(confidences):.4f}")
        print(f"   Max:                  {max(confidences):.4f}")
        print(
            f"   Std Dev:              {statistics.stdev(confidences):.4f}"
            if len(confidences) > 1
            else "   Std Dev:              N/A"
        )

    if processing_times:
        print(f"\n⏱️  Processing Time (ms):")
        print(f"   Mean:                 {statistics.mean(processing_times):.2f}")
        print(f"   Median:               {statistics.median(processing_times):.2f}")
        print(f"   Min:                  {min(processing_times):.2f}")
        print(f"   Max:                  {max(processing_times):.2f}")
        print(
            f"   Std Dev:              {statistics.stdev(processing_times):.2f}"
            if len(processing_times) > 1
            else "   Std Dev:              N/A"
        )

    # Category breakdown
    print(f"\n📂 Results by Category:")
    categories = {}
    for result in results:
        if result.category not in categories:
            categories[result.category] = {"total": 0, "correct": 0, "detected": 0}
        categories[result.category]["total"] += 1
        if result.expected_pii == result.has_pii:
            categories[result.category]["correct"] += 1
        if result.has_pii:
            categories[result.category]["detected"] += 1

    for category in sorted(categories.keys()):
        stats = categories[category]
        accuracy = stats["correct"] / stats["total"] * 100
        print(
            f"   {category:<20} {stats['correct']}/{stats['total']} correct ({accuracy:.0f}%), {stats['detected']} detected"
        )

    print(f"\n{'='*100}\n")


def main():
    """Main benchmark execution"""
    # Check API health
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code != 200:
            print(
                f"❌ ERROR: Classification API not healthy (status {response.status_code})"
            )
            return
    except Exception as e:
        print(
            f"❌ ERROR: Cannot connect to Classification API at {CLASSIFICATION_API_URL}"
        )
        print(f"   Make sure the router is running on port 8080")
        print(f"   Error: {e}")
        return

    # Run benchmark
    results = run_benchmark()

    # Print results
    print_results_table(results)
    print_statistics(results)

    # Save detailed results to JSON
    output_file = "/tmp/pii-benchmark-results.json"
    with open(output_file, "w") as f:
        json_results = []
        for r in results:
            json_results.append(
                {
                    "prompt": r.prompt,
                    "category": r.category,
                    "expected_pii": r.expected_pii,
                    "has_pii": r.has_pii,
                    "max_confidence": r.max_confidence,
                    "entities_detected": r.entities_detected,
                    "processing_time_ms": r.processing_time_ms,
                    "error": r.error,
                }
            )
        json.dump(json_results, f, indent=2)

    print(f"📄 Detailed results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
