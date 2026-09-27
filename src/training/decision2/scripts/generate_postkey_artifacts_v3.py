"""Generate post-key v3 card artifacts against the original frozen scorer map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from publication import generate_arena_v3 as artifacts
from scripts.package_postkey_aggregate_v3 import _frozen_ranker


def generate(
    source_root: Path, config_path: Path, diagnostic: Path, output: Path
) -> dict:
    config = artifacts.load(config_path)
    arena_path = artifacts.relative(config_path, config["arena_rank"])
    sidecar = artifacts.load(diagnostic)
    if (
        sidecar.get("schema_version") != "jevarena-v3-postkey-answer-count-diagnostic/1"
        or sidecar.get("status") != "postkey_diagnostic_not_preregistered_release_gate"
        or sidecar.get("ranked_result") != artifacts.load(arena_path)
    ):
        raise ValueError("Card artifacts lack the exact post-key rank sidecar")
    frozen = _frozen_ranker(source_root)
    original = artifacts.SCORER_SOURCE_PATHS
    artifacts.SCORER_SOURCE_PATHS = frozen.SCORER_SOURCE_PATHS
    try:
        return artifacts.generate(config_path, output)
    finally:
        artifacts.SCORER_SOURCE_PATHS = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--postkey-rank-diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(
        args.frozen_source_root,
        args.config,
        args.postkey_rank_diagnostic,
        args.output,
    )
    print(json.dumps({"version": result["publication_version"]}, sort_keys=True))


if __name__ == "__main__":
    main()
