"""Seal official-source 4B formal predictions before any gold/scoring access."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from inference.run import file_digest

from .lock_postkey_qwen4b import ROSTER_SCHEMA, SCHEMA
from .seal_postkey_qwen06b import (
    PANELS,
    _candidate_rows,
    _exclusive_json,
    _object,
    _prompts,
    _source,
)

SEAL_SCHEMA = "decision2-official-qwen4b-v3-postkey-prediction-seal/1"


def seal(args: argparse.Namespace) -> dict:
    if file_digest(args.lock) != args.lock_sha256:
        raise ValueError("Prospective lock bytes changed")
    locked, roster = _object(args.lock), _object(args.roster)
    if (
        locked.get("schema") != SCHEMA
        or roster.get("schema") != ROSTER_SCHEMA
        or locked.get("roster_sha256") != file_digest(args.roster)
        or locked.get("post_key_same_panel") is not True
    ):
        raise ValueError("Wrong model roster, lock, or label chronology")
    _source(roster, args.source_root)
    paths = {panel: getattr(args, f"{panel}_prompts") for panel in PANELS}
    panels = _prompts(roster, paths)
    if locked["prompts_sha256"] != {
        panel: file_digest(path) for panel, path in paths.items()
    }:
        raise ValueError("Prompt bytes changed after lock")
    candidate = {
        panel: _candidate_rows(
            args.predictions / f"{panel}.predictions.jsonl",
            panels[panel],
            roster["candidate"],
            locked["prompts_sha256"][panel],
        )
        for panel in PANELS
    }
    result = {
        "schema": SEAL_SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_lock_sha256": file_digest(args.lock),
        "roster_sha256": file_digest(args.roster),
        "code_sha256": file_digest(Path(__file__)),
        "post_key_same_panel": True,
        "candidate": candidate,
        "controls": locked["controls"],
        "prompts_sha256": locked["prompts_sha256"],
        "scoring_sources_sha256": locked["scoring_sources_sha256"],
        "claim": "All three candidate predictions and six reused controls sealed before this run's first formal/public score; prior project label access is disclosed.",
    }
    _exclusive_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "lock",
        "roster",
        "source_root",
        "typed_prompts",
        "css_prompts",
        "public_prompts",
        "predictions",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--lock-sha256", required=True)
    args = parser.parse_args()
    result = seal(args)
    print({"schema": result["schema"], "sha256": file_digest(args.output)})


if __name__ == "__main__":
    main()
