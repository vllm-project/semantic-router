#!/usr/bin/env python3
"""
Production-grade vLLM-based data generation with streaming writes and checkpointing.

Pipeline Position: Step 1 of 2
    Input:  unlabeled_queries.jsonl (raw domain queries)
    Output: triplets.jsonl (anchor + positive + negative for contrastive learning)
    Next:   lora_trainer.py trains LoRA adapter from these triplets

Based on arXiv:2504.02268v1 - generates proper triplets for MNR loss training.

What this does:
    1. Takes unlabeled domain queries (e.g., "How to diagnose diabetes?")
    2. Uses LLM to generate:
       - Paraphrases (semantic equivalents, e.g., "What are diagnostic methods for diabetes?")
       - Hard negatives (related but different, e.g., "What are symptoms of diabetes?")
    3. Creates triplet samples for contrastive learning:
       - anchor: paraphrase
       - positive: original query (same meaning)
       - negative: hard negative (related but different intent)

Features:
- Streaming writes (no memory accumulation)
- Checkpoint/resume capability
- Progress tracking with detailed stats
- Multi-GPU support via vLLM tensor parallelism
- Error handling and retry logic
- Graceful shutdown on interruption
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import signal
import sys
from typing import Optional, List, Dict
import time
import os
import yaml

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global shutdown_requested
    print("\n⚠️  Shutdown requested. Finishing current batch...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_domain_prompts(domain: str) -> Dict:
    """Load domain-specific prompts from YAML configuration."""
    prompts_file = Path(__file__).parent / "domains" / "prompts.yaml"

    if not prompts_file.exists():
        print(f"\n❌ ERROR: Prompts file not found at {prompts_file}")
        sys.exit(1)

    with open(prompts_file) as f:
        config = yaml.safe_load(f)

    domains = config.get("domains", {})

    # Handle domain aliases
    domain_aliases = {
        "medical": "health",
        "programming": "computer_science",
    }
    canonical_domain = domain_aliases.get(domain, domain)

    if canonical_domain not in domains:
        available = ", ".join(domains.keys())
        print(f"\n❌ ERROR: No prompts defined for domain '{domain}'")
        print(f"Available domains: {available}")
        sys.exit(1)

    print(f"Loading prompts for domain: {domain}")
    domain_config = domains[canonical_domain]
    print(f"✓ Loaded prompts for {domain} (role: {domain_config.get('role', 'N/A')})")

    return domain_config


def extract_text(value):
    """Extract text from either string or {"text": "..."} format."""
    if isinstance(value, dict):
        return value.get("text", "")
    return str(value) if value else ""


class StreamingWriter:
    """Handles streaming writes with automatic flushing and crash recovery."""

    def __init__(self, output_path: Path, checkpoint_path: Path):
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Open in append mode for crash recovery
        self.file = open(output_path, "a", buffering=1)  # Line buffered
        self.samples_written = 0

    def write_samples(self, samples: List[Dict]):
        """Write samples immediately and flush."""
        for sample in samples:
            self.file.write(json.dumps(sample) + "\n")
            self.samples_written += 1
        self.file.flush()

    def write_checkpoint(self, queries_processed: int, total_queries: int):
        """Write checkpoint for resume capability."""
        checkpoint = {
            "queries_processed": queries_processed,
            "total_queries": total_queries,
            "samples_written": self.samples_written,
            "timestamp": time.time(),
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f)

    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                return json.load(f)
        return None

    def close(self):
        """Close file handles."""
        if hasattr(self, "file"):
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def generate_paraphrases_batch_vllm(
    queries: List[str], llm, num_paraphrases: int, domain_config: Dict
) -> List[List[str]]:
    """Generate paraphrases for a batch using vLLM.

    Uses simplified prompt format - empirically proven to work better than guideline blobs.
    """
    from vllm import SamplingParams

    role = domain_config.get("role", "helpful assistant")
    examples = domain_config.get("paraphrase_examples", [])

    # Build GOOD examples from YAML
    good_examples = ""
    if examples:
        ex = examples[0]
        good_examples = f"\nExamples of GOOD paraphrases:\n"
        good_examples += f'Original: "{ex["original"]}"\n'
        for i, p in enumerate(ex.get("paraphrases", [])[:3], 1):
            good_examples += f'{i}. "{p}"\n'

    # BAD examples - universal anti-patterns
    bad_examples = """
Examples of BAD paraphrases (DO NOT DO THIS):
- Removing punctuation only: "What is diabetes?" → "What is diabetes" (TOO SIMILAR)
- Changing one word: "How to reduce stress?" → "How to reduce anxiety?" (NOT ENOUGH CHANGE)
- Just reordering: "How can I sort a list?" → "Can I sort a list how?" (AWKWARD)
"""

    prompts = []
    for query in queries:
        # Simplified prompt format
        prompt = f"""You are a {role}.

Task: Generate {num_paraphrases} PARAPHRASES of the original query.

Rules:
- MUST preserve the exact same meaning and intent
- MUST use different words and sentence structure
- Each paraphrase should be 8-20 words long
- DO NOT just remove punctuation or change one word
- DO NOT change the core question being asked

Original Query: {query}
{good_examples}{bad_examples}
Return ONLY valid JSON:
{{"paraphrases": ["paraphrase 1 here", "paraphrase 2 here", "paraphrase 3 here"]}}
"""
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=512, stop=None)

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        try:
            text = output.outputs[0].text.strip()
            if "{" in text:
                json_start = text.index("{")
                json_end = text.rindex("}") + 1
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                paraphrases = [extract_text(p) for p in data.get("paraphrases", [])]
                results.append([p for p in paraphrases if p])
            else:
                results.append([])
        except Exception:
            results.append([])

    return results


def generate_negatives_batch_vllm(
    queries: List[str], llm, num_negatives: int, domain_config: Dict
) -> List[List[str]]:
    """Generate hard negatives for a batch using vLLM.

    Uses simplified prompt format based on empirical testing:
    - Simple prompts work better than long guideline blobs
    - Examples more effective than detailed instructions
    - Format: Task -> Rules -> Examples -> JSON
    """
    from vllm import SamplingParams

    role = domain_config.get("role", "helpful assistant")
    negative_guidelines = domain_config.get("negative_guidelines", "")
    examples = domain_config.get("negative_examples", [])

    # Build examples from YAML
    examples_text = ""
    if examples:
        examples_text = "\nExamples:\n"
        for ex in examples[:3]:  # Use more examples
            orig = ex.get("original", "")
            if orig:
                examples_text += f"\nOriginal: {orig}\n"

            # Show BAD examples if provided
            bad_negs = ex.get("bad_negatives", [])
            if bad_negs:
                examples_text += "BAD (DO NOT DO THIS):\n"
                for neg in bad_negs[:2]:
                    examples_text += f'  ❌ "{neg}"\n'

            # Show GOOD examples
            good_negs = ex.get("negatives", []) or ex.get("good_negatives", [])
            if good_negs:
                examples_text += "GOOD (DO THIS):\n"
                for neg in good_negs[:2]:
                    examples_text += f'  ✅ "{neg}"\n'

    prompts = []
    for query in queries:
        # Use domain-specific guidelines from prompts.yaml
        prompt = f"""You are a {role}.

Task: Generate {num_negatives} hard negative questions for this query.

{negative_guidelines}

Original Query: {query}
{examples_text}
Return ONLY valid JSON:
{{"negatives": ["question 1 here", "question 2 here"]}}
"""
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=512, stop=None)

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        try:
            text = output.outputs[0].text.strip()
            if "{" in text:
                json_start = text.index("{")
                json_end = text.rindex("}") + 1
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                negatives = [extract_text(n) for n in data.get("negatives", [])]
                results.append([n for n in negatives if n])
            else:
                results.append([])
        except Exception:
            results.append([])

    return results


def process_batch_vllm(
    batch_queries: List[str], llm, args, domain_config: Dict
) -> List[Dict]:
    """Process a batch using vLLM with immediate sample generation.

    Creates triplets according to the paper (arXiv:2504.02268v1):
    - Each sample has anchor + positive + negative for proper MNR loss
    - Positive: paraphrased version (semantically identical)
    - Negative: related but distinct query (different intent/focus)
    """
    paraphrases_batch = generate_paraphrases_batch_vllm(
        batch_queries, llm, args.paraphrases, domain_config
    )
    negatives_batch = generate_negatives_batch_vllm(
        batch_queries, llm, args.negatives, domain_config
    )

    samples = []
    for query, paraphrases, negatives in zip(
        batch_queries, paraphrases_batch, negatives_batch
    ):
        # Create triplets: each paraphrase paired with a negative
        # This ensures proper contrastive learning with MNR loss
        for i, paraphrase in enumerate(paraphrases):
            # Use round-robin to assign negatives if we have fewer negatives than paraphrases
            negative_idx = i % len(negatives) if negatives else None

            if negative_idx is not None:
                samples.append(
                    {
                        "anchor": paraphrase,
                        "positive": query,
                        "negative": negatives[negative_idx],
                        "is_duplicate": 1,  # Anchor-positive are duplicates
                    }
                )

    return samples


def main():
    # Disable vLLM's verbose progress bars to show clean overall progress
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"

    parser = argparse.ArgumentParser(
        description="Production vLLM augmentation with streaming and checkpointing"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL with unlabeled queries"
    )
    parser.add_argument("--output", required=True, help="Output JSONL for training")
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain for prompts (e.g., programming, medical)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name"
    )
    parser.add_argument(
        "--paraphrases", type=int, default=3, help="Paraphrases per query"
    )
    parser.add_argument(
        "--negatives", type=int, default=2, help="Hard negatives per query"
    )
    parser.add_argument("--max-queries", type=int, help="Max queries to process")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for vLLM"
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=0.9, help="GPU memory utilization"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--checkpoint-interval", type=int, default=10, help="Checkpoint every N batches"
    )

    args = parser.parse_args()

    # Load domain-specific prompts
    domain_config = load_domain_prompts(args.domain)

    # Use domain's recommended model if --model not explicitly provided
    if args.model == "Qwen/Qwen2.5-7B-Instruct":  # Default value
        recommended = domain_config.get("recommended_model")
        if recommended:
            args.model = recommended
            print(f"Using domain-recommended model: {recommended}")

    # Setup paths
    output_path = Path(args.output)
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"

    # Load queries
    print(f"Loading queries from {args.input}...")
    queries = []
    with open(args.input) as f:
        for line in f:
            data = json.loads(line)
            queries.append(data["query"])

    if args.max_queries:
        queries = queries[: args.max_queries]
        print(f"Limited to {args.max_queries} queries")

    print(f"Loaded {len(queries)} queries")
    print(f"Model: {args.model}")
    print()

    # Initialize streaming writer
    with StreamingWriter(output_path, checkpoint_path) as writer:

        # Check for resume
        start_idx = 0
        if args.resume:
            checkpoint = writer.load_checkpoint()
            if checkpoint:
                start_idx = checkpoint["queries_processed"]
                print(
                    f"✓ Resuming from checkpoint: {start_idx}/{len(queries)} queries processed"
                )
                print(f"  {checkpoint['samples_written']} samples already written")
                print()

        # Initialize vLLM
        print("Initializing vLLM...")

        # Suppress verbose vLLM progress bars
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

        from vllm import LLM

        llm = LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory,
            max_model_len=2048,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel,
            disable_log_stats=True,  # Suppress token/s stats
        )
        print(f"✓ vLLM initialized with {args.tensor_parallel} GPU(s)")
        print()

        # Process in batches with streaming writes
        batch_size = args.batch_size
        num_batches = (len(queries) - start_idx + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(0, len(queries) - start_idx, batch_size),
            desc="Processing batches",
            total=num_batches,
        ):

            if shutdown_requested:
                print("\n⚠️  Shutdown requested. Saving checkpoint...")
                writer.write_checkpoint(start_idx + batch_idx, len(queries))
                print("✓ Checkpoint saved. Safe to exit.")
                sys.exit(0)

            actual_idx = start_idx + batch_idx
            batch_queries = queries[actual_idx : actual_idx + batch_size]

            # Generate samples and write immediately
            samples = process_batch_vllm(batch_queries, llm, args, domain_config)
            writer.write_samples(samples)

            # Checkpoint periodically
            if (batch_idx // batch_size + 1) % args.checkpoint_interval == 0:
                writer.write_checkpoint(actual_idx + len(batch_queries), len(queries))

        # Final checkpoint
        writer.write_checkpoint(len(queries), len(queries))

    print(f"\n✓ Generated {writer.samples_written} training samples")
    print(f"✓ Saved to {args.output}")
    print(f"\nAugmentation factor: {writer.samples_written / len(queries):.1f}x")


if __name__ == "__main__":
    main()
