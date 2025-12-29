#!/usr/bin/env python3
"""
Command-Line Interface for Data Generation.

Usage:
    # Generate all data (Stage 1 + Stage 2)
    python -m data_generation.cli --output-dir data/generated

    # Generate only Stage 2 with specific generators
    python -m data_generation.cli --stage 2 --generators adversarial filesystem

    # Generate with minimal samples (for testing)
    python -m data_generation.cli --preset minimal

    # Skip HuggingFace datasets
    python -m data_generation.cli --no-huggingface
"""

import argparse
import sys
from pathlib import Path

from .config import Config, get_minimal_config, get_full_config
from .orchestrator import DataOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Unified Data Generation for Jailbreak Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all data with defaults
  python -m data_generation.cli

  # Generate only Stage 2 (Tool Verifier)
  python -m data_generation.cli --stage 2

  # Use only specific generators
  python -m data_generation.cli --generators adversarial filesystem network

  # Quick test with minimal data
  python -m data_generation.cli --preset minimal

  # Full production dataset
  python -m data_generation.cli --preset full
        """,
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/generated",
        help="Output directory for generated data",
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Generate only specific stage (1=Sentinel, 2=Verifier)",
    )

    parser.add_argument(
        "--generators",
        "-g",
        nargs="+",
        default=None,
        help="Enable only specific generators (e.g., adversarial filesystem)",
    )

    parser.add_argument(
        "--samples-per", type=int, default=None, help="Number of samples per generator"
    )

    parser.add_argument(
        "--no-huggingface",
        action="store_true",
        help="Skip HuggingFace datasets, use only synthetic",
    )

    parser.add_argument(
        "--preset",
        choices=["minimal", "full", "adversarial"],
        default=None,
        help="Use a preset configuration",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--list-generators",
        action="store_true",
        help="List available generators and exit",
    )

    args = parser.parse_args()

    # List generators
    if args.list_generators:
        from .generators import GENERATORS

        print("Available generators:")
        for name in sorted(GENERATORS.keys()):
            print(f"  - {name}")
        sys.exit(0)

    # Create config
    if args.preset == "minimal":
        config = get_minimal_config()
    elif args.preset == "full":
        config = get_full_config()
    else:
        config = Config()

    # Override with CLI args
    config.output_dir = Path(args.output_dir)
    config.seed = args.seed

    if args.no_huggingface:
        config.use_huggingface = False

    if args.samples_per:
        for name in config.generators:
            config.set_generator_samples(name, args.samples_per)

    # Create orchestrator
    orchestrator = DataOrchestrator(config)

    # Enable specific generators
    if args.generators:
        orchestrator.enable_only(args.generators)

    # Generate data
    if args.stage == 1:
        orchestrator.generate_stage1_data()
    elif args.stage == 2:
        orchestrator.generate_stage2_data()
    else:
        orchestrator.generate_all()

    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
