"""Milestone 6 Index path for the ~27B track (prereg amendment 4; COORDINATION 2026-10-02 02:05): item 1' and the
amended choice among passers, from the combined VERDICTS file and each finalist's private paired bootstrap.

    python3 -m v2.27b.m6.m6_index_path --verdicts VERDICTS.json [--boot ARM=paired-boot-vs-a20r.json ...] \
        [--family-delta ARM=family-delta-vs-a20r.json ...] --public OUT.json --private PRIVATE.json

Item 1' = (a) post-key v3 not significantly below A20r (the upper bound of the paired v3 CI vs A20r in the VERDICTS
file > 0) and (b) the private Index delta vs A20r significantly positive (``v2.eval.ix1.paired_boot`` headline 95% CI
lower bound > 0). A finalist without a bootstrap file cannot use the Index path. Index-path passers have item 1' and
items 2-7. Choice: (1) classic passers (items 1-7) that beat AutoJev-27B, by the lower bound vs AutoJev-27B; (2) other
classic passers, by the lower bound vs A20r; (3) Index-path passers, by the lower bound of the Index delta.

The public output holds pass / fail and the choice only. Index values (deltas, intervals and the transfer-only delta,
i.e. the weighted per-benchmark contributions without the benchmarks whose train splits are in the arm's TRAIN and
the format-matched BPoMP) go to the private output, which must lie under a ``private`` directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "m6-index-path/1"
ITEMS_2_7 = (
    "2_H_not_below_A20r",
    "3_no_type_collapsed",
    "4_mlx_card_eligible_not_below_A20r",
    "5_tier_gates",
    "6_no_overlap_exposure",
    "7_public231_not_regression",
)
IN_DISTRIBUTION = {
    "M6-IB": ("When2Call", "iSarcasmEval"),
    "M6-IBX": (),
    "M6-IB2": ("When2Call", "iSarcasmEval", "HoVer", "GSM8K"),
    "M6-IB2PN": ("When2Call", "iSarcasmEval", "HoVer", "GSM8K"),
}
FORMAT_MATCHED = ("BPoMP",)
SHARED_WITH_A20R = ("BANKING77", "CLINC150")


def pairs(specs: list[str]) -> dict[str, Path]:
    out = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if name not in IN_DISTRIBUTION or not path:
            raise SystemExit(f"expected M6 ARM=FILE, got {spec!r}")
        out[name] = Path(path)
    return out


def item_1_prime(
    finalist: dict[str, Any], boot: dict[str, Any] | None
) -> dict[str, Any]:
    v3_high = finalist["paired"]["A20r"]["ci95"][1]
    a = v3_high > 0
    b = boot is not None and boot["headline"]["ci95"][0] > 0
    return {
        "a_v3_not_significantly_below_A20r": a,
        "b_index_delta_significantly_positive": b if boot is not None else None,
        "pass": a and b,
    }


def transfer_only(arm: str, delta: dict[str, Any]) -> dict[str, Any]:
    rows = delta["benchmarks"]
    excluded = [n for n in IN_DISTRIBUTION[arm] + FORMAT_MATCHED if n in rows]
    shared = {n: rows[n]["weighted_delta"] for n in SHARED_WITH_A20R if n in rows}
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


def decide(
    verdicts: dict[str, Any],
    boots: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    public: dict[str, Any] = {"schema": SCHEMA, "finalists": {}}
    private: dict[str, Any] = {"schema": SCHEMA + "-private", "finalists": {}}
    index_passers = []
    for name, f in sorted(verdicts["finalists"].items()):
        items = f["successor_items"]
        rest = all(items[k]["pass"] for k in ITEMS_2_7)
        prime = item_1_prime(f, boots.get(name))
        index_path = prime["pass"] and rest
        public["finalists"][name] = {
            "classic_items_1_7": bool(f["successor_items_1_7"]),
            "beats_autojev27": bool(f["beats_autojev27"]["pass"]),
            "item_1_prime": prime,
            "items_2_7": rest,
            "index_path": index_path,
            "index_run": name in boots,
        }
        record: dict[str, Any] = {}
        if name in boots:
            head = boots[name]["headline"]
            record["index_delta"] = {
                k: head[k] for k in ("delta", "ci95", "se", "p_le_0")
            }
        if name in deltas:
            record["transfer_only"] = transfer_only(name, deltas[name])
        private["finalists"][name] = record
        if index_path:
            index_passers.append(name)
    index_passers.sort(key=lambda n: -boots[n]["headline"]["ci95"][0])
    beating = list(verdicts["beats_autojev_and_successor"])
    classic = list(verdicts["successor_items_1_7"])
    choice = (beating or classic or index_passers or [None])[0]
    public.update(
        {
            "rule": "prereg amendment 4: (1) items 1-7 and beats AutoJev-27B by the lower bound vs AutoJev-27B; "
            "(2) items 1-7 by the lower bound vs A20r; (3) items 1' and 2-7 by the Index delta's lower bound",
            "beats_autojev_and_successor": beating,
            "successor_items_1_7": classic,
            "index_path": index_passers,
            "choice": choice,
            "choice_path": (
                None
                if choice is None
                else ("classic" if choice in classic else "index")
            ),
        }
    )
    private["index_path_order"] = index_passers
    return public, private


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--boot", action="append", default=[])
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
    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8"))
    boots = {
        n: json.loads(p.read_text(encoding="utf-8"))
        for n, p in pairs(args.boot).items()
    }
    deltas = {
        n: json.loads(p.read_text(encoding="utf-8"))
        for n, p in pairs(args.family_delta).items()
    }
    for name in list(boots) + list(deltas):
        if name not in verdicts["finalists"]:
            raise SystemExit(f"{name} is not a finalist of {args.verdicts}")
    public, private = decide(verdicts, boots, deltas)
    with args.public.open("x", encoding="utf-8") as stream:
        json.dump(public, stream, indent=1, sort_keys=True)
        stream.write("\n")
    args.private.parent.mkdir(parents=True, exist_ok=True)
    with args.private.open("x", encoding="utf-8") as stream:
        json.dump(private, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: public[k] for k in ("index_path", "choice", "choice_path")}))


if __name__ == "__main__":
    main()
