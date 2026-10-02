"""Index-first release rule for the ~27B tier (M6 prereg amendment 7; COORDINATION 2026-10-02 09:55, a user
decision that supersedes the Index path and successor items 1-8 as release blockers).

    python3 -m v2.27b.m6.m6_index_first --boot NAME=paired-boot-vs-a20r.json ... --types NAME=types.json ... \
        [--family-delta NAME=family-delta-vs-a20r.json ...] --public OUT.json --private PRIVATE.json

Gate (the only quality blocker): the candidate's private Index delta vs the current release (A20r) is significantly
positive, i.e. the ``v2.eval.ix1.paired_boot`` headline 95% CI lower bound is > 0. Integrity: no decision type
collapsed on formal typed FINAL (``v2.eval.gates types``: every type OK). Choice: the largest lower bound among the
candidates that pass both; an exact tie goes to the candidate without in-distribution families, then by name. A
candidate without a bootstrap file has no Index run and cannot be chosen.

The public output and stdout hold pass / fail, the order and the choice only. Index values (deltas, intervals and the
transfer-only delta: the weighted per-benchmark contributions without the benchmarks whose train splits are in the
candidate's TRAIN and, for the IB1 arms, the format-matched BPoMP) go to the private output, which must lie under a
``private`` directory.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

index_path = importlib.import_module("v2.27b.m6.m6_index_path")

SCHEMA = "m6-index-first/1"
IN_DISTRIBUTION = {**index_path.IN_DISTRIBUTION, "M5-L128": ()}
FORMAT_MATCHED = {
    name: () if name == "M5-L128" else index_path.FORMAT_MATCHED
    for name in IN_DISTRIBUTION
}
RULE = (
    "prereg amendment 7 (Index-first): paired_boot headline 95% lower bound vs A20r > 0 and no type collapsed on "
    "formal typed FINAL; choice = the largest lower bound"
)


def pairs(specs: list[str]) -> dict[str, Path]:
    out = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if name not in IN_DISTRIBUTION or not path:
            raise SystemExit(f"expected a 27B candidate NAME=FILE, got {spec!r}")
        out[name] = Path(path)
    return out


def transfer_only(name: str, delta: dict[str, Any]) -> dict[str, Any]:
    rows = delta["benchmarks"]
    excluded = [n for n in IN_DISTRIBUTION[name] + FORMAT_MATCHED[name] if n in rows]
    shared = {
        n: rows[n]["weighted_delta"] for n in index_path.SHARED_WITH_A20R if n in rows
    }
    return {
        "excluded": excluded,
        "weighted_delta_sum": delta["weighted_delta_sum"],
        "transfer_only_weighted_delta": round(
            delta["weighted_delta_sum"]
            - sum(rows[n]["weighted_delta"] for n in excluded),
            3,
        ),
        "excluded_weighted_delta": {n: rows[n]["weighted_delta"] for n in excluded},
        "shared_with_A20r_weighted_delta": shared,
    }


def no_type_collapsed(types: dict[str, Any]) -> bool:
    verdicts = [entry["verdict"] for entry in types["types"].values()]
    return bool(verdicts) and all(v == "OK" for v in verdicts)


def decide(
    boots: dict[str, dict[str, Any]],
    types: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    public: dict[str, Any] = {"schema": SCHEMA, "rule": RULE, "candidates": {}}
    private: dict[str, Any] = {"schema": SCHEMA + "-private", "candidates": {}}
    eligible = []
    for name in sorted(set(boots) | set(types)):
        boot = boots.get(name)
        gate = None if boot is None else boot["headline"]["ci95"][0] > 0
        intact = None if name not in types else no_type_collapsed(types[name])
        public["candidates"][name] = {
            "index_run": boot is not None,
            "index_gate": gate,
            "no_type_collapsed": intact,
            "eligible": bool(gate) and bool(intact),
        }
        record: dict[str, Any] = {}
        if boot is not None:
            record["index_delta"] = {
                k: boot["headline"][k] for k in ("delta", "ci95", "se", "p_le_0")
            }
        if name in deltas:
            record["transfer_only"] = transfer_only(name, deltas[name])
        private["candidates"][name] = record
        if public["candidates"][name]["eligible"]:
            eligible.append(name)
    eligible.sort(
        key=lambda n: (
            -boots[n]["headline"]["ci95"][0],
            bool(IN_DISTRIBUTION[n]),
            n,
        )
    )
    public["order"] = eligible
    public["choice"] = eligible[0] if eligible else None
    private["order"] = eligible
    return public, private


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--boot", action="append", default=[])
    parser.add_argument("--types", action="append", default=[])
    parser.add_argument("--family-delta", action="append", default=[])
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args(argv)
    if "private" not in args.private.resolve().parts:
        raise SystemExit(f"{args.private} is not under a private directory")
    if "private" in args.public.resolve().parts:
        raise SystemExit(
            f"{args.public} is the public output; keep it out of private directories"
        )

    def load(specs: list[str]) -> dict[str, dict[str, Any]]:
        return {
            n: json.loads(p.read_text(encoding="utf-8"))
            for n, p in pairs(specs).items()
        }

    public, private = decide(load(args.boot), load(args.types), load(args.family_delta))
    with args.public.open("x", encoding="utf-8") as stream:
        json.dump(public, stream, indent=1, sort_keys=True)
        stream.write("\n")
    args.private.parent.mkdir(parents=True, exist_ok=True)
    with args.private.open("x", encoding="utf-8") as stream:
        json.dump(private, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: public[k] for k in ("candidates", "order", "choice")}))


if __name__ == "__main__":
    main()
