"""Run with PYTHONPATH=tools/calibration python -m systemone_auto."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import canonical, file_digest, read_json, write_json
from .collection import collect
from .dataset import prepare
from .export import bind_policy
from .judge import collect_judge, score_judge
from .replay import replay
from .sources import download
from .timing import COST_METRICS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "download", help="download licensed public sources with content receipts"
    )
    fetch.add_argument("--directory", type=Path, required=True)
    fetch.add_argument("--lock", type=Path)
    prep = commands.add_parser(
        "prepare", help="make an independently labelled, grouped pilot"
    )
    prep.add_argument("--data-dir", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--per-source", type=int, default=128)
    prep.add_argument("--seed", type=int, default=42)
    gather = commands.add_parser(
        "collect", help="collect real native/Chat model observations"
    )
    gather.add_argument("--dataset", type=Path, required=True)
    gather.add_argument(
        "--targets",
        type=Path,
        required=True,
        help="private local target JSON; never copied to output",
    )
    gather.add_argument("--output-dir", type=Path, required=True)
    gather.add_argument("--timeout", type=float, default=120)
    gather.add_argument(
        "--target",
        action="append",
        help="collect only this named target; full manifest identity is retained",
    )
    gather.add_argument(
        "--limit",
        type=int,
        help="smoke subset; incomplete collections cannot be replayed",
    )
    compare = commands.add_parser(
        "replay", help="train and compare paired policies on held-out source groups"
    )
    compare.add_argument("--dataset", type=Path, required=True)
    compare.add_argument("--collection-dir", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument(
        "--protocol",
        type=Path,
        help="Frozen, separately declared restricted native-pool experiment",
    )
    compare.add_argument(
        "--cost-metric", choices=COST_METRICS, default="server_compute_ms"
    )
    compare.add_argument(
        "--base", required=True, help="target name for the first native stage"
    )
    export = commands.add_parser(
        "export-policy",
        help="bind every fitted action to deployment stage/provider aliases",
    )
    export.add_argument("--policy", type=Path, required=True)
    export.add_argument("--bindings", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    judge = commands.add_parser(
        "collect-judge",
        help="collect separate prior-answer-conditioned judge observations",
    )
    judge.add_argument("--requests", type=Path, required=True)
    judge.add_argument("--manifest", type=Path, required=True)
    judge.add_argument("--target", type=Path, required=True)
    judge.add_argument("--deployment-receipt", type=Path, required=True)
    judge.add_argument("--output-dir", type=Path, required=True)
    judge.add_argument("--limit", type=int)
    judge_score = commands.add_parser(
        "score-judge", help="score a complete selector component experiment"
    )
    judge_score.add_argument("--dataset", type=Path, required=True)
    judge_score.add_argument("--requests", type=Path, required=True)
    judge_score.add_argument("--collection-dir", type=Path, required=True)
    judge_score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        result = download(args.directory, args.lock)
        summary = {"sources": list(result["sources"])}
    elif args.command == "prepare":
        result = prepare(
            args.data_dir, args.output, per_source=args.per_source, seed=args.seed
        )
        summary = {"records_sha256": result["records_sha256"], **result["counts"]}
    elif args.command == "collect":
        result = collect(
            args.dataset,
            args.targets,
            args.output_dir,
            timeout=args.timeout,
            limit=args.limit,
            target_names=args.target,
        )
        summary = {
            key: result[key]
            for key in (
                "collection_identity",
                "observation_count",
                "expected_count",
                "complete",
            )
        }
    elif args.command == "collect-judge":
        result = collect_judge(
            args.requests,
            args.manifest,
            args.target,
            args.deployment_receipt,
            args.output_dir,
            limit=args.limit,
        )
        summary = {
            key: result[key]
            for key in (
                "collection_identity",
                "observation_count",
                "expected_count",
                "complete",
            )
        }
    elif args.command == "score-judge":
        result = score_judge(args.dataset, args.requests, args.collection_dir)
        write_json(args.output, result)
        summary = {
            "trace_sha256": result["trace_sha256"],
            "outcomes": result["outcomes"],
        }
    elif args.command == "export-policy":
        result = bind_policy(read_json(args.policy), read_json(args.bindings))
        write_json(args.output, result)
        summary = {
            "policy_sha256": file_digest(args.output),
            "actions": list(result["actions"]),
        }
    else:
        result = replay(
            args.dataset,
            args.collection_dir,
            args.output_dir,
            args.base,
            cost_metric=args.cost_metric,
            protocol_path=args.protocol,
        )
        summary = {
            "dataset_sha256": result["dataset_sha256"],
            "operating_points": len(result["curves"]),
        }
    print(canonical(summary))


if __name__ == "__main__":
    main()
