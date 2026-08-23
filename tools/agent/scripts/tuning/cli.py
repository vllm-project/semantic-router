#!/usr/bin/env python3
"""CLI entry point for the DSL tuning framework.

Usage:
    python -m tuning.cli <scenario> [options]

Examples:
    python -m tuning.cli privacy --config config.yaml --probes probes.yaml --candidate-config candidate.yaml
    python -m tuning.cli calibration --config config.yaml --probes probes.yaml --candidate-config candidate.yaml
    python -m tuning.cli list
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .client import RouterClient
from .probes import save_results
from .scenario import CandidateTuner

BUILTIN_SCENARIOS = {
    "privacy": "tuning.scenarios.privacy:PrivacyScenario",
    "calibration": "tuning.scenarios.calibration:CalibrationScenario",
}


def _load_scenario(name: str):
    """Load a scenario class by name or module:Class path."""
    if ":" in name:
        module_path, class_name = name.rsplit(":", 1)
    elif name in BUILTIN_SCENARIOS:
        module_path, class_name = BUILTIN_SCENARIOS[name].rsplit(":", 1)
    else:
        print(f"Unknown scenario: {name}")
        print(f"Available: {', '.join(BUILTIN_SCENARIOS)}")
        print("Or specify a module:Class path (e.g. my_scenarios.custom:MyScenario)")
        sys.exit(1)

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DSL Tuning Framework — analytical optimization for semantic router",
    )
    parser.add_argument(
        "scenario",
        help="Scenario name (privacy, calibration) or module:Class path",
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:8080", help="Router API endpoint"
    )
    parser.add_argument("--config", required=True, help="Path to router config YAML")
    parser.add_argument(
        "--probes", required=True, help="Path to probe definitions YAML"
    )
    parser.add_argument(
        "--candidate-config",
        required=True,
        help="Write the offline candidate here; must differ from --config",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON filename (default: <scenario>_eval.json)",
    )
    parser.add_argument(
        "--output-dir", default="", help="Output directory (default: ./results)"
    )
    return parser


def main() -> int:
    parser = build_parser()

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        print("Available scenarios:")
        for name, path in BUILTIN_SCENARIOS.items():
            print(f"  {name:20s} → {path}")
        return 0

    args = parser.parse_args()

    scenario = _load_scenario(args.scenario)
    router = RouterClient(args.endpoint)

    print("=" * 70)
    print("  DSL Tuning Framework")
    print(f"  Scenario:  {scenario.name}")
    print(f"  Endpoint:  {args.endpoint}")
    print(f"  Config:    {args.config}")
    print(f"  Probes:    {args.probes}")
    print(f"  Candidate: {args.candidate_config}")
    print("=" * 70)

    tuner = CandidateTuner(
        scenario=scenario,
        router=router,
        config_path=Path(args.config),
        probes_path=Path(args.probes),
        candidate_path=Path(args.candidate_config),
    )

    output = tuner.run()

    filename = args.output or f"{scenario.name}_eval.json"
    output_dir = Path(args.output_dir) if args.output_dir else None
    save_results(output, filename, output_dir)

    print(f"\n{'='*70}")
    print(f"  SUMMARY — {scenario.name}")
    print(f"{'='*70}")
    evaluation = output["evaluation"]
    print(
        f"  Evaluation: {evaluation['accuracy']}/{evaluation['total']} "
        f"({evaluation['pct']}%)"
    )
    if output.get("candidate_config"):
        print(f"  Candidate: {output['candidate_config']}")

    validation = output.get("validation")
    return 0 if validation is None or validation.get("valid", False) else 1


if __name__ == "__main__":
    sys.exit(main())
