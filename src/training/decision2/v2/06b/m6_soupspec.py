"""Write a uniform soup spec (`soup.py`) over finished same-init causal runs.

python3 m6_soupspec.py <output-spec.json> <soup-name> <run-or-group> [<run-or-group> ...] [--runs DIR]

A run is an arm directory name ending in -s<k> (e.g. m6-cx-s1); a group is an arm prefix
(e.g. m4-t-a7) and expands to its seeds s1-s3 that finished (COMPLETE); a group needs at
least two. Every seed gets the same weight. The recipe is the first ingredient's committed
arm spec; `allowed_differences` lists every field in which the ingredients' RUN.json specs
differ (top-level fields, and the fields of `start` and `data`). Ingredients that differ in
family, base model or head seed are refused.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

RUNS = "/data/dev2/runs/06b/m1/arms"
NESTED = ("start", "data")
REFUSED = ("family", "start.repo", "start.revision", "start.head_seed")
RECIPES = "/src/src/training/decision2/v2/06b/records/arms"


def status(run_dir: Path) -> str | None:
    path = run_dir / "COMPLETE.json"
    return json.loads(path.read_text()).get("status") if path.is_file() else None


def expand(items: list[str], runs: Path) -> tuple[list[str], dict[str, list[str]]]:
    arms, skipped = [], {}
    for item in items:
        if re.search(r"-s\d+$", item):
            if status(runs / item / "full") != "COMPLETE":
                raise ValueError(f"{item}: run is not COMPLETE")
            arms.append(item)
            continue
        seeds = [f"{item}-s{k}" for k in (1, 2, 3)]
        done = [s for s in seeds if status(runs / s / "full") == "COMPLETE"]
        if len(done) < 2:
            raise ValueError(f"{item}: fewer than two COMPLETE seeds")
        skipped[item] = [s for s in seeds if s not in done]
        arms.extend(done)
    if len(set(arms)) != len(arms) or len(arms) < 2:
        raise ValueError("A soup needs two or more distinct ingredients")
    return arms, skipped


def differences(specs: list[dict[str, Any]]) -> list[str]:
    """Dotted fields whose values differ across the specs."""
    out = []
    missing = object()
    for key in sorted({k for s in specs for k in s}):
        values = [s.get(key, missing) for s in specs]
        if all(v == values[0] for v in values):
            continue
        if key in NESTED and all(isinstance(v, dict) for v in values):
            for sub in sorted({k for v in values for k in v}):
                inner = [v.get(sub, missing) for v in values]
                if any(v is missing for v in inner):
                    raise ValueError(f"{key}.{sub}: present in only some ingredients")
                if any(v != inner[0] for v in inner):
                    out.append(f"{key}.{sub}")
            continue
        if any(v is missing for v in values):
            raise ValueError(f"{key}: present in only some ingredients")
        out.append(key)
    return out


def soup_spec(
    name: str, items: list[str], runs: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    arms, skipped = expand(items, runs)
    ingredients, specs = [], []
    for arm in arms:
        run_dir = runs / arm / "full"
        spec = json.loads((run_dir / "RUN.json").read_text())["spec"]
        if spec["arm"] != arm:
            raise ValueError(f"{arm}: run directory holds {spec['arm']}")
        best = json.loads((run_dir / "BEST.json").read_text())
        specs.append(spec)
        ingredients.append(
            {
                "arm": arm,
                "run": f"/runs/m1/arms/{arm}/full",
                "state_sha256": best["state_sha256"],
            }
        )
    if any(s["family"] != "qwen-causal" for s in specs):
        raise ValueError("Soups are implemented for the causal Qwen family")
    allowed = differences(specs)
    refused = [f for f in allowed if f in REFUSED]
    if refused:
        raise ValueError(f"Ingredients differ in {refused}: not the same init")
    spec = {
        "arm": name,
        "family": "qwen-causal",
        "recipe": f"{RECIPES}/{arms[0]}.json",
        "ingredients": ingredients,
        "allowed_differences": allowed,
        "export": True,
    }
    return spec, {"ingredients": arms, "skipped_seeds": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("name")
    parser.add_argument("items", nargs="+")
    parser.add_argument("--runs", type=Path, default=Path(RUNS))
    args = parser.parse_args()
    spec, info = soup_spec(args.name, args.items, args.runs)
    with args.output.open("x") as stream:
        json.dump(spec, stream, indent=2)
        stream.write("\n")
    print(json.dumps({**info, "allowed_differences": spec["allowed_differences"]}))


if __name__ == "__main__":
    main()
