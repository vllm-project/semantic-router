"""Parser construction and entrypoint for the fleet-sim CLI."""

from __future__ import annotations

import argparse

from . import __version__ as FLEET_SIM_VERSION
from .cli_advanced import (
    cmd_disagg,
    cmd_grid_flex,
    cmd_simulate_fleet,
    cmd_tok_per_watt,
)
from .cli_common import GPU_REGISTRY
from .cli_core import (
    cmd_compare_routers,
    cmd_optimize,
    cmd_pareto,
    cmd_serve,
    cmd_simulate,
    cmd_whatif,
)


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-sr-sim",
        description="vllm-sr-sim: fleet-level LLM inference simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {FLEET_SIM_VERSION}"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    _add_optimize_parser(subparsers)
    _add_simulate_parser(subparsers)
    _add_whatif_parser(subparsers)
    _add_pareto_parser(subparsers)
    _add_compare_routers_parser(subparsers)
    _add_disagg_parser(subparsers)
    _add_grid_flex_parser(subparsers)
    _add_tok_per_watt_parser(subparsers)
    _add_simulate_fleet_parser(subparsers)
    _add_serve_parser(subparsers)
    return parser


def _add_common_args(parser) -> None:
    parser.add_argument("--cdf", required=True, help="Path to CDF JSON file")
    parser.add_argument("--lam", type=float, default=200, help="Arrival rate (req/s)")
    parser.add_argument("--slo", type=float, default=500, help="P99 TTFT SLO (ms)")
    parser.add_argument(
        "--b-short",
        type=int,
        default=4096,
        help="Short-pool context threshold (tokens)",
    )
    parser.add_argument(
        "--long-max-ctx", type=int, default=65536, help="Long-pool max context (tokens)"
    )
    parser.add_argument(
        "--gpu-short", default="a100", choices=list(GPU_REGISTRY.keys())
    )
    parser.add_argument("--gpu-long", default="a100", choices=list(GPU_REGISTRY.keys()))
    parser.add_argument("--out", default=None, help="Save results to JSON file")


def _add_optimize_parser(subparsers) -> None:
    parser = subparsers.add_parser("optimize", help="Find min-cost fleet meeting SLO")
    _add_common_args(parser)
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--n-sim-req", type=int, default=40000)
    parser.add_argument("--verify-top", type=int, default=3)
    parser.set_defaults(func=cmd_optimize)


def _add_simulate_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "simulate", help="Simulate a fixed fleet configuration"
    )
    _add_common_args(parser)
    parser.add_argument("--n-s", type=int, required=True, help="Short pool GPUs")
    parser.add_argument("--n-l", type=int, required=True, help="Long pool GPUs")
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="C&R gamma (1.0 = pool routing only)"
    )
    parser.add_argument("--n-req", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=cmd_simulate)


def _add_whatif_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "whatif", help="Sweep arrival rate and/or GPU types (what-if analysis)"
    )
    _add_common_args(parser)
    parser.add_argument(
        "--lam-range",
        type=float,
        nargs="+",
        default=[50, 100, 200, 500, 1000],
        help="List of arrival rates to sweep",
    )
    parser.add_argument(
        "--gpu-compare",
        type=str,
        nargs="+",
        metavar="GPU",
        help="Compare multiple GPU types side-by-side (e.g. --gpu-compare a100 h100 a10g). When set, --gpu-short/--gpu-long are ignored.",
    )
    parser.set_defaults(func=cmd_whatif)


def _add_pareto_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "pareto",
        help="Sweep all CDF breakpoints as B_short candidates and show Pareto frontier",
    )
    _add_common_args(parser)
    parser.set_defaults(func=cmd_pareto)


def _add_compare_routers_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "compare-routers", help="Compare routing algorithms on same fleet"
    )
    _add_common_args(parser)
    parser.add_argument("--n-s", type=int, required=True)
    parser.add_argument("--n-l", type=int, required=True)
    parser.add_argument("--n-req", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=cmd_compare_routers)


def _add_disagg_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "disagg",
        help="Disaggregated prefill/decode fleet optimizer (separate P and D pools)",
    )
    parser.add_argument("--cdf", required=True, help="Workload CDF JSON")
    parser.add_argument(
        "--lam", type=float, default=200, help="Target arrival rate (req/s)"
    )
    parser.add_argument(
        "--slo-ttft", type=float, default=500, help="TTFT SLO (ms, default 500)"
    )
    parser.add_argument(
        "--slo-tpot", type=float, default=100, help="TPOT SLO (ms, default 100)"
    )
    parser.add_argument(
        "--gpu-prefill", default="h100", choices=list(GPU_REGISTRY.keys())
    )
    parser.add_argument(
        "--gpu-decode", default="a100", choices=list(GPU_REGISTRY.keys())
    )
    parser.add_argument(
        "--max-ctx",
        type=int,
        default=8192,
        help="Max context length (tokens, default 8192)",
    )
    parser.add_argument(
        "--mean-isl",
        type=float,
        default=0,
        help="Mean input sequence length (0 = derive from CDF)",
    )
    parser.add_argument(
        "--mean-osl",
        type=float,
        default=0,
        help="Mean output sequence length (0 = derive from CDF)",
    )
    parser.add_argument(
        "--max-prefill",
        type=int,
        default=32,
        help="Max prefill workers to sweep (default 32)",
    )
    parser.add_argument(
        "--max-decode",
        type=int,
        default=64,
        help="Max decode workers to sweep (default 64)",
    )
    parser.add_argument("--out", default=None, help="Save results to JSON")
    parser.set_defaults(func=cmd_disagg)


def _add_grid_flex_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "grid-flex",
        help="Power-latency trade-off curve for demand-response (GPU-to-Grid)",
        description=(
            "Sweep batch-size curtailment levels and report P99 TTFT vs. power reduction. "
            "Models the GPU-to-Grid (G2G) demand-response mechanism."
        ),
    )
    parser.add_argument("--cdf", required=True, help="Workload CDF JSON")
    parser.add_argument("--lam", type=float, default=200, help="Arrival rate (req/s)")
    parser.add_argument(
        "--n-gpus", type=int, required=True, help="Fixed fleet size (GPUs) to evaluate"
    )
    parser.add_argument(
        "--gpu",
        default="h100",
        choices=list(GPU_REGISTRY.keys()),
        help="GPU type (default: h100)",
    )
    parser.add_argument("--slo", type=float, default=500, help="P99 TTFT SLO (ms)")
    parser.add_argument(
        "--max-ctx",
        type=int,
        default=8192,
        help="Max context window (tokens, default 8192)",
    )
    parser.add_argument(
        "--flex-pcts",
        type=float,
        nargs="+",
        default=None,
        metavar="PCT",
        help="Power-reduction percentages to sweep (default: 0 5 10 15 20 25 30 40 50)",
    )
    parser.add_argument(
        "--verify-des",
        type=int,
        default=0,
        metavar="N",
        help="DES-verify each flex level with N simulated requests (0 = analytical only)",
    )
    parser.add_argument("--out", default=None, help="Save results to JSON")
    parser.set_defaults(func=cmd_grid_flex)


def _add_tok_per_watt_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "tok-per-watt",
        help="Compare GPU energy efficiency (tokens/watt) at the SLO-optimal fleet",
    )
    parser.add_argument("--cdf", required=True, help="Workload CDF JSON")
    parser.add_argument(
        "--lam", type=float, default=100, help="Arrival rate (req/s, default: 100)"
    )
    parser.add_argument(
        "--slo", type=float, default=500, help="P99 TTFT SLO (ms, default: 500)"
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["h100", "a100", "a10g"],
        choices=list(GPU_REGISTRY.keys()),
        help="GPU types to compare in single-pool mode (default: h100 a100 a10g)",
    )
    parser.add_argument(
        "--max-ctx",
        type=int,
        default=8192,
        help="Max context window (tokens, default: 8192)",
    )
    parser.add_argument(
        "--rho-sweep", action="store_true", help="Also show tok/W at ρ=0.2,0.4,0.6,0.8"
    )
    parser.add_argument(
        "--b-short",
        type=int,
        default=None,
        metavar="TOKENS",
        help="[two-pool mode] Short-pool threshold",
    )
    parser.add_argument(
        "--gpu-short",
        default=None,
        choices=list(GPU_REGISTRY.keys()),
        help="[two-pool mode] Short-pool GPU",
    )
    parser.add_argument(
        "--gpu-long",
        default=None,
        choices=list(GPU_REGISTRY.keys()),
        help="[two-pool mode] Long-pool GPU",
    )
    parser.add_argument("--out", default=None, help="Save results to JSON")
    parser.set_defaults(func=cmd_tok_per_watt)


def _add_simulate_fleet_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "simulate-fleet",
        help="Simulate an arbitrary N-pool fleet from a JSON config",
    )
    parser.add_argument("fleet_config", help="Path to fleet JSON config file")
    parser.add_argument(
        "--cdf", default=None, help="Workload CDF JSON (overrides config workloads)"
    )
    parser.add_argument(
        "--lam", type=float, default=200, help="Total arrival rate req/s (default: 200)"
    )
    parser.add_argument(
        "--slo",
        type=float,
        default=500,
        help="P99 TTFT SLO ms for reporting (default: 500)",
    )
    parser.add_argument(
        "--n-req",
        type=int,
        default=30000,
        help="Total requests to simulate (default: 30000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Save results to JSON file")
    parser.set_defaults(func=cmd_simulate_fleet)


def _add_serve_parser(subparsers) -> None:
    parser = subparsers.add_parser("serve", help="Start the vllm-sr-sim HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default="info")
    parser.set_defaults(func=cmd_serve)
