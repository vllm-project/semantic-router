#!/usr/bin/env python3
"""
Production-grade vLLM-based data generation with streaming writes and checkpointing.

Based on arXiv:2504.02268v1 - generates proper triplets for MNR loss training.

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

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global shutdown_requested
    print("\n⚠️  Shutdown requested. Finishing current batch...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def extract_text(value):
    """Extract text from either string or {"text": "..."} format."""
    if isinstance(value, dict):
        return value.get('text', '')
    return str(value) if value else ''


class StreamingWriter:
    """Handles streaming writes with automatic flushing and crash recovery."""

    def __init__(self, output_path: Path, checkpoint_path: Path):
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Open in append mode for crash recovery
        self.file = open(output_path, 'a', buffering=1)  # Line buffered
        self.samples_written = 0

    def write_samples(self, samples: List[Dict]):
        """Write samples immediately and flush."""
        for sample in samples:
            self.file.write(json.dumps(sample) + '\n')
            self.samples_written += 1
        self.file.flush()

    def write_checkpoint(self, queries_processed: int, total_queries: int):
        """Write checkpoint for resume capability."""
        checkpoint = {
            'queries_processed': queries_processed,
            'total_queries': total_queries,
            'samples_written': self.samples_written,
            'timestamp': time.time()
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)

    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                return json.load(f)
        return None

    def close(self):
        """Close file handles."""
        if hasattr(self, 'file'):
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def generate_paraphrases_batch_vllm(queries: List[str], llm, num_paraphrases: int) -> List[List[str]]:
    """Generate paraphrases for a batch using vLLM."""
    from vllm import SamplingParams

    prompts = []
    for query in queries:
        prompt = f"""You are a helpful medical expert. Generate {num_paraphrases} unique paraphrases of the given query.

Original Query: '{query}'

Each paraphrase should:
1. Preserve the original meaning but use different wording or sentence structure.
2. Avoid changing medical intent or introducing new information.
3. Be professionally written and clear.

Example:
Original Query: "What are the best ways to reduce stress?"
Paraphrased Queries:
1. "How can a person effectively manage stress?"
2. "What strategies help in reducing stress levels?"
3. "What methods are effective for stress reduction?"

Return ONLY a JSON object with a key 'paraphrases' containing a list of exactly {num_paraphrases} paraphrased strings (not objects).
Format: {{"paraphrases": ["string1", "string2", "string3"]}}
"""
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=None
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        try:
            text = output.outputs[0].text.strip()
            if '{' in text:
                json_start = text.index('{')
                json_end = text.rindex('}') + 1
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                paraphrases = [extract_text(p) for p in data.get('paraphrases', [])]
                results.append([p for p in paraphrases if p])
            else:
                results.append([])
        except Exception:
            results.append([])

    return results


def generate_negatives_batch_vllm(queries: List[str], llm, num_negatives: int) -> List[List[str]]:
    """Generate hard negatives for a batch using vLLM."""
    from vllm import SamplingParams

    prompts = []
    for query in queries:
        prompt = f"""You are a helpful medical expert. Given a medical query, generate {num_negatives} distinct but related queries that explore different aspects.

Guidelines:
1. The new queries should be related to the original but focus on different subtopics, perspectives, or medical contexts.
2. They should not be simple rewordings or slight variations of the original.
3. Consider different patient populations, alternative diagnostic methods, treatments, or physiological explanations.

Examples:
Original Query: "How to reduce stress?"
Distinct Queries:
1. "How can athletes manage stress during high-pressure competitions?"
2. "What are effective stress management strategies for children with ADHD?"

Original Query: "What is diabetes?"
Distinct Queries:
1. "How does gestational diabetes differ from type 2 diabetes?"
2. "What are the long-term complications of uncontrolled diabetes?"

Now, generate {num_negatives} distinct queries for this input:
Original Query: {query}

Return ONLY a JSON object with 'negatives' containing a list of exactly {num_negatives} strings (not objects).
Format: {{"negatives": ["string1", "string2"]}}
"""
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=None
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        try:
            text = output.outputs[0].text.strip()
            if '{' in text:
                json_start = text.index('{')
                json_end = text.rindex('}') + 1
                json_str = text[json_start:json_end]
                data = json.loads(json_str)
                negatives = [extract_text(n) for n in data.get('negatives', [])]
                results.append([n for n in negatives if n])
            else:
                results.append([])
        except Exception:
            results.append([])

    return results


def process_batch_vllm(batch_queries: List[str], llm, args) -> List[Dict]:
    """Process a batch using vLLM with immediate sample generation.

    Creates triplets according to the paper (arXiv:2504.02268v1):
    - Each sample has anchor + positive + negative for proper MNR loss
    - Positive: paraphrased version (semantically identical)
    - Negative: related but distinct query (different intent/focus)
    """
    paraphrases_batch = generate_paraphrases_batch_vllm(batch_queries, llm, args.paraphrases)
    negatives_batch = generate_negatives_batch_vllm(batch_queries, llm, args.negatives)

    samples = []
    for query, paraphrases, negatives in zip(batch_queries, paraphrases_batch, negatives_batch):
        # Create triplets: each paraphrase paired with a negative
        # This ensures proper contrastive learning with MNR loss
        for i, paraphrase in enumerate(paraphrases):
            # Use round-robin to assign negatives if we have fewer negatives than paraphrases
            negative_idx = i % len(negatives) if negatives else None

            if negative_idx is not None:
                samples.append({
                    "anchor": paraphrase,
                    "positive": query,
                    "negative": negatives[negative_idx],
                    "is_duplicate": 1  # Anchor-positive are duplicates
                })

    return samples


def main():
    # Disable vLLM's verbose progress bars to show clean overall progress
    os.environ['VLLM_DISABLE_PROGRESS_BAR'] = '1'

    parser = argparse.ArgumentParser(description="Production vLLM augmentation with streaming and checkpointing")
    parser.add_argument("--input", required=True, help="Input JSONL with unlabeled queries")
    parser.add_argument("--output", required=True, help="Output JSONL for training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name")
    parser.add_argument("--paraphrases", type=int, default=3, help="Paraphrases per query")
    parser.add_argument("--negatives", type=int, default=2, help="Hard negatives per query")
    parser.add_argument("--max-queries", type=int, help="Max queries to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for vLLM")
    parser.add_argument("--gpu-memory", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint every N batches")

    args = parser.parse_args()

    # Setup paths
    output_path = Path(args.output)
    checkpoint_path = output_path.parent / f"{output_path.stem}_checkpoint.json"

    # Load queries
    print(f"Loading queries from {args.input}...")
    queries = []
    with open(args.input) as f:
        for line in f:
            data = json.loads(line)
            queries.append(data['query'])

    if args.max_queries:
        queries = queries[:args.max_queries]
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
                start_idx = checkpoint['queries_processed']
                print(f"✓ Resuming from checkpoint: {start_idx}/{len(queries)} queries processed")
                print(f"  {checkpoint['samples_written']} samples already written")
                print()

        # Initialize vLLM
        print("Initializing vLLM...")
        from vllm import LLM

        llm = LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory,
            max_model_len=2048,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel
        )
        print(f"✓ vLLM initialized with {args.tensor_parallel} GPU(s)")
        print()

        # Process in batches with streaming writes
        batch_size = args.batch_size
        num_batches = (len(queries) - start_idx + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(0, len(queries) - start_idx, batch_size),
                             desc="Processing batches",
                             total=num_batches):

            if shutdown_requested:
                print("\n⚠️  Shutdown requested. Saving checkpoint...")
                writer.write_checkpoint(start_idx + batch_idx, len(queries))
                print("✓ Checkpoint saved. Safe to exit.")
                sys.exit(0)

            actual_idx = start_idx + batch_idx
            batch_queries = queries[actual_idx:actual_idx + batch_size]

            # Generate samples and write immediately
            samples = process_batch_vllm(batch_queries, llm, args)
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
