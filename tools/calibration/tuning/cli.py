#!/usr/bin/env python3
"""CLI entry point for the DSL tuning framework.

Usage:
    python -m tuning.cli <scenario> [options]

Examples:
    python -m tuning.cli privacy --config config.yaml --probes probes.yaml
    python -m tuning.cli calibration --config config.yaml --probes probes.yaml --max-iter 15
    python -m tuning.cli list
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .client import RouterClient
from .probes import save_results
from .scenario import TuningLoop

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


def main() -> int:
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
        "--deploy-config",
        default="",
        help="Optional: write fixed config to this path instead",
    )
    parser.add_argument(
        "--router-pid", type=int, default=0, help="Router PID for SIGHUP hot-reload"
    )
    parser.add_argument(
        "--max-iter", type=int, default=10, help="Maximum tuning iterations"
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON filename (default: <scenario>_eval.json)",
    )
    parser.add_argument(
        "--output-dir", default="", help="Output directory (default: ./results)"
    )

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
    print(f"  Max iter:  {args.max_iter}")
    print("=" * 70)

    loop = TuningLoop(
        scenario=scenario,
        router=router,
        config_path=Path(args.config),
        probes_path=Path(args.probes),
        deploy_config=Path(args.deploy_config) if args.deploy_config else None,
        router_pid=args.router_pid,
        max_iterations=args.max_iter,
    )

    output = loop.run()

    filename = args.output or f"{scenario.name}_eval.json"
    output_dir = Path(args.output_dir) if args.output_dir else None
    save_results(output, filename, output_dir)

    print(f"\n{'='*70}")
    print(f"  SUMMARY — {scenario.name}")
    print(f"{'='*70}")
    for t in output.get("trajectory", []):
        print(
            f"  Iter {t['iteration']}: {t['accuracy']}/{t['total']} "
            f"({t['pct']}%)  sev_loss={t['severity_weighted_loss']}"
        )
    print(f"  Total fixes: {len(output.get('all_fixes_applied', []))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
