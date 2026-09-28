"""Run the preregistered arm grid for one feature source, one fresh process per run.

Order: one-update preflight per arm (must pass), layer screen on the primary
seed, the frozen layer rule, then the remaining seeds at the chosen layer. A
failed preflight or run stops that arm only; nothing is retried.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import pins
from .arms import ARMS

PRIMARY_SEED = 20260928
EXTRA_SEEDS = (1, 2)


def run(command: list[str], log: Path) -> int:
    with log.open("x", encoding="utf-8") as stream:
        return subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT
        ).returncode


def choose_layer(results: dict[int, dict]) -> int:
    """SELECT family-macro accuracy desc, Brier asc; ties go to the later layer."""
    return min(
        results,
        key=lambda layer: (
            -results[layer]["select"]["family_macro_accuracy"],
            results[layer]["select"]["family_macro_brier"],
            -layer,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", required=True, help="comma-separated arm IDs")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    arms = args.arms.split(",")
    if any(arm not in ARMS for arm in arms):
        raise SystemExit(f"Unknown arm in {arms}")
    args.output.mkdir(parents=True, exist_ok=False)
    base = [
        sys.executable,
        "-m",
        "clm9b.train_heads",
        "--features",
        str(args.features),
        "--code-commit",
        args.code_commit,
    ]
    summary = {
        "features_manifest_sha256": pins.file_sha256(args.features / "manifest.json"),
        "arms": {},
    }

    def train(arm: str, layer: int, seed: int, preflight: bool = False) -> dict | None:
        name = f"{arm}-L{layer}-s{seed}" + ("-preflight" if preflight else "")
        out = args.output / name
        command = base + [
            "--arm",
            arm,
            "--layer",
            str(layer),
            "--seed",
            str(seed),
            "--output",
            str(out),
        ]
        if ARMS[arm]["objective"].endswith("replay"):
            command += ["--teacher", str(args.teacher)]
        if preflight:
            command.append("--preflight")
        started = time.time()
        code = run(command, args.output / f"{name}.log")
        print(
            json.dumps(
                {"run": name, "exit": code, "seconds": round(time.time() - started, 1)}
            ),
            flush=True,
        )
        if code != 0 or not (out / "BEST.json").exists():
            return None
        return json.loads((out / "BEST.json").read_text(encoding="utf-8"))

    for arm in arms:
        entry: dict = {
            "preflight": None,
            "layers": {},
            "chosen_layer": None,
            "seeds": {},
        }
        summary["arms"][arm] = entry
        if ARMS[arm]["objective"].endswith("replay") and args.teacher is None:
            entry["stopped"] = "no admitted teacher"
            continue
        check = train(arm, 32, PRIMARY_SEED, preflight=True)
        entry["preflight"] = check and {
            k: check[k] for k in ("reload_max_abs_drift", "select_zero_step")
        }
        if check is None:
            entry["stopped"] = "preflight failed"
            continue
        for layer in pins.LAYERS:
            best = train(arm, layer, PRIMARY_SEED)
            if best is None:
                entry["stopped"] = f"layer {layer} run failed"
                break
            entry["layers"][layer] = best
        if "stopped" in entry:
            continue
        layer = choose_layer(entry["layers"])
        entry["chosen_layer"] = layer
        entry["seeds"][PRIMARY_SEED] = entry["layers"][layer]
        for seed in EXTRA_SEEDS:
            best = train(arm, layer, seed)
            entry["seeds"][seed] = best
            if best is None:
                entry["stopped"] = f"seed {seed} run failed"
                break
        (args.output / "GRID.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
        )
    (args.output / "GRID.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
