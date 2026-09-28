"""M6 development selection: proxy Q, guards, family artifact and greedy step (stdlib only).

    python3 -m v2.06b.m6_select table NAME|READOUT_DIR ... [--json OUT] [--finalist-ref m5-z-soup]
    python3 -m v2.06b.m6_select seedmean --soup NAME SEED ... [--json OUT]
    python3 -m v2.06b.m6_select greedy-check S S_PLUS_G [--json OUT]

A NAME resolves to `<root>/<NAME>/readout` (default root /data/dev2/runs/06b/m1/arms); the
run's `full/` directory beside it holds SELECT/CAL metrics (`SOUP.json` for soups,
`BEST.json` for seeds; seeds have no CAL probabilities, so their CAL Noul is missing).

Development only (M6 prereg sections 3 and 4), never a release score:
- T_dev = typed-DEV macro family accuracy; H_pilot = CSS-pilot median task macro-F1;
  H3 = mean macro-F1 of the three pilot tasks; P = 100*sqrt(T_dev*H_pilot);
  Q = 100*sqrt(T_dev*H3).
- Guards: typed-DEV Choice >= 255 and Score >= 101, >= 3 distinct predicted Score levels,
  SELECT Noul and CAL Noul accuracy >= .85. A missing input fails its guard.
- seedmean: the family artifact is the soup if Q(soup) >= the seed-mean Q, else the
  median-Q seed (lower middle for an even count). Seeds with `full/STOPPED.json` (collapse
  stop) or without a readout are excluded.
- greedy-check S S+g: accept g iff Q(S+g) >= Q(S), H3(S+g) >= H3(S) - 0.010, typed-DEV
  Choice and Score of S+g each >= those of S minus 20, and the guards of S+g hold.
- table: finalist-eligible = contains an m6-cx / m6-mx ingredient, guards pass and
  Q >= Q(finalist-ref) - 4 (when the reference is in the table).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("/data/dev2/runs/06b/m1/arms")
PILOT_TASKS = ("discourse", "implicit_hate", "semeval_stance")
CHOICE_MIN, SCORE_MIN, LEVELS_MIN, NOUL_MIN = 255, 101, 3, 0.85
H3_SLACK, ITEM_SLACK, FINALIST_BAND = 0.010, 20, 4.0
NEW_FAMILIES = ("m6-cx", "m6-mx")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(ref: str, root: Path) -> tuple[str, Path]:
    if "/" in ref:
        path = Path(ref)
        return (path.parent.name if path.name == "readout" else path.name), path
    return ref, root / ref / "readout"


def score_levels(readout: dict[str, Any], directory: Path) -> dict[str, int]:
    levels = readout.get("score_levels_predicted")
    if levels is not None:
        return {k: int(v) for k, v in levels.items() if v}
    counts: Counter[str] = Counter()
    with (directory / "dev.predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            for answer in json.loads(line)["answers"].values():
                if answer.get("type") == "score" and "probabilities" in answer:
                    p = answer["probabilities"]
                    counts[max(p, key=p.get)] += 1
    return dict(counts)


def noul(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    cell = metrics["by_type"]["noul"]
    return {"accuracy": cell["accuracy"], "correct": cell["correct"], "n": cell["n"]}


def in_distribution(full: Path) -> tuple[dict | None, dict | None, list[str], str]:
    if (full / "SOUP.json").is_file():
        soup = load_json(full / "SOUP.json")
        arms = [i["arm"] for i in soup.get("ingredients", [])]
        return (
            noul((soup.get("select") or {}).get("metrics")),
            noul((soup.get("cal") or {}).get("metrics")),
            arms,
            "SOUP.json",
        )
    if (full / "BEST.json").is_file():
        return noul(load_json(full / "BEST.json").get("metrics")), None, [], "BEST.json"
    return None, None, [], "missing"


def guards(c: dict[str, Any]) -> dict[str, bool | None]:
    def at_least(value: Any, floor: float) -> bool | None:
        return None if value is None else value >= floor

    return {
        "choice_ge_255": at_least(c["typed"]["choice"], CHOICE_MIN),
        "score_ge_101": at_least(c["typed"]["score"], SCORE_MIN),
        "score_levels_ge_3": at_least(c["score_levels"], LEVELS_MIN),
        "select_noul_ge_085": at_least(
            (c["select_noul"] or {}).get("accuracy"), NOUL_MIN
        ),
        "cal_noul_ge_085": at_least((c["cal_noul"] or {}).get("accuracy"), NOUL_MIN),
    }


def candidate(ref: str, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    name, directory = resolve(ref, root)
    readout = load_json(directory / "READOUT.json")
    tasks = readout["css_pilot_tasks"]
    if set(tasks) != set(PILOT_TASKS):
        raise ValueError(
            f"{name}: CSS pilot tasks {sorted(tasks)} != {list(PILOT_TASKS)}"
        )
    t, h_pilot = readout["T_dev"], readout["H_pilot"]
    h3 = sum(tasks[k] for k in PILOT_TASKS) / len(PILOT_TASKS)
    levels = score_levels(readout, directory)
    full = directory.parent / "full"
    select_noul, cal_noul, ingredients, source = in_distribution(full)
    out = {
        "name": name,
        "readout_dir": str(directory),
        "T_dev": t,
        "H_pilot": h_pilot,
        "H3": h3,
        "pilot_tasks": {k: tasks[k] for k in PILOT_TASKS},
        "P": 100 * math.sqrt(t * h_pilot),
        "Q": 100 * math.sqrt(t * h3),
        "typed": {
            k: readout["typed_by_type"][k]["correct"]
            for k in ("choice", "noul", "score")
        },
        "typed_n": {
            k: readout["typed_by_type"][k]["n"] for k in ("choice", "noul", "score")
        },
        "score_levels": len(levels),
        "score_level_counts": dict(sorted(levels.items())),
        "select_noul": select_noul,
        "cal_noul": cal_noul,
        "in_distribution_source": source,
        "ingredients": ingredients,
        "collapse_stopped": (full / "STOPPED.json").is_file(),
    }
    out["guards"] = guards(out)
    out["guards_pass"] = all(v is True for v in out["guards"].values())
    out["guards_failed"] = [k for k, v in out["guards"].items() if v is not True]
    out["contains_new_family"] = any(
        arm.startswith(NEW_FAMILIES) for arm in [name, *ingredients]
    )
    return out


def fmt_noul(cell: dict[str, Any] | None) -> str:
    return "—" if cell is None else f"{cell['accuracy']:.3f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    head = (
        f"{'candidate':22s} {'Q':>6s} {'H3':>6s} {'T_dev':>6s} {'P':>6s} {'H_pil':>6s} "
        f"{'C/N/S':>13s} {'lvl':>3s} {'SELn':>5s} {'CALn':>5s}  guards"
    )
    print(head)
    for c in rows:
        typed = "/".join(str(c["typed"][k]) for k in ("choice", "noul", "score"))
        verdict = "pass" if c["guards_pass"] else "FAIL " + ",".join(c["guards_failed"])
        print(
            f"{c['name']:22s} {c['Q']:6.2f} {c['H3']:6.4f} {c['T_dev']:6.4f} {c['P']:6.2f} "
            f"{c['H_pilot']:6.4f} {typed:>13s} {c['score_levels']:3d} {fmt_noul(c['select_noul']):>5s} "
            f"{fmt_noul(c['cal_noul']):>5s}  {verdict}"
        )


def table(refs: list[str], root: Path, finalist_ref: str | None) -> dict[str, Any]:
    rows = [candidate(r, root) for r in refs]
    by_name = {c["name"]: c for c in rows}
    ref_q = by_name[finalist_ref]["Q"] if finalist_ref in by_name else None
    for c in rows:
        c["finalist_eligible"] = (
            None
            if ref_q is None
            else c["contains_new_family"]
            and c["guards_pass"]
            and c["Q"] >= ref_q - FINALIST_BAND
        )
    passing = sorted(
        (c for c in rows if c["guards_pass"]), key=lambda c: (-c["Q"], -c["H3"])
    )
    return {
        "schema": "dev2-06b-m6-select-table/1",
        "label": "development readout (never a release score)",
        "rules": {
            "Q": "100*sqrt(T_dev*H3)",
            "P": "100*sqrt(T_dev*H_pilot)",
            "guards": f"Choice >= {CHOICE_MIN}, Score >= {SCORE_MIN}, >= {LEVELS_MIN} Score levels, "
            f"SELECT and CAL Noul >= {NOUL_MIN}",
            "finalist": f"contains m6-cx/m6-mx, guards pass, Q >= Q({finalist_ref}) - {FINALIST_BAND}",
        },
        "finalist_ref_Q": ref_q,
        "guard_passing_by_Q": [c["name"] for c in passing],
        "candidates": rows,
    }


def seedmean(soup: str, seeds: list[str], root: Path) -> dict[str, Any]:
    s = candidate(soup, root)
    used, excluded = [], []
    for ref in seeds:
        name, directory = resolve(ref, root)
        if (directory.parent / "full" / "STOPPED.json").is_file():
            excluded.append({"name": name, "reason": "collapse stop"})
        elif not (directory / "READOUT.json").is_file():
            excluded.append({"name": name, "reason": "no readout"})
        else:
            used.append(candidate(ref, root))
    if len(used) < 2:
        raise ValueError("a family soup needs >= 2 non-collapsed seeds")
    qs = [c["Q"] for c in used]
    mean_q = statistics.fmean(qs)
    ordered = sorted(used, key=lambda c: (c["Q"], c["name"]))
    median_seed = ordered[(len(ordered) - 1) // 2]
    artifact = s if s["Q"] >= mean_q else median_seed
    return {
        "schema": "dev2-06b-m6-select-seedmean/1",
        "label": "development readout (never a release score)",
        "rule": "soup if Q(soup) >= seed-mean Q, else the median-Q seed (lower middle if even)",
        "soup": s,
        "seeds": used,
        "excluded_seeds": excluded,
        "seed_mean_Q": mean_q,
        "seed_sd_Q": statistics.stdev(qs) if len(qs) > 1 else None,
        "median_seed": median_seed["name"],
        "median_ambiguous": len(used) % 2 == 0,
        "artifact": artifact["name"],
        "artifact_is_soup": artifact is s,
        "artifact_guards_pass": artifact["guards_pass"],
        "artifact_guards_failed": artifact["guards_failed"],
    }


def greedy_check(base: str, extended: str, root: Path) -> dict[str, Any]:
    s, g = candidate(base, root), candidate(extended, root)
    checks = {
        "Q_not_lower": g["Q"] >= s["Q"],
        "H3_within_0.010": g["H3"] >= s["H3"] - H3_SLACK,
        "choice_within_20": g["typed"]["choice"] >= s["typed"]["choice"] - ITEM_SLACK,
        "score_within_20": g["typed"]["score"] >= s["typed"]["score"] - ITEM_SLACK,
        "guards_hold": g["guards_pass"],
    }
    return {
        "schema": "dev2-06b-m6-select-greedy/1",
        "label": "development readout (never a release score)",
        "S": s,
        "S_plus_g": g,
        "delta": {
            "Q": g["Q"] - s["Q"],
            "H3": g["H3"] - s["H3"],
            "choice": g["typed"]["choice"] - s["typed"]["choice"],
            "score": g["typed"]["score"] - s["typed"]["score"],
        },
        "checks": checks,
        "accept": all(checks.values()),
    }


def write(path: Path | None, value: Any) -> None:
    if path is None:
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("table")
    p.add_argument("refs", nargs="+")
    p.add_argument("--finalist-ref", default="m5-z-soup")
    p.add_argument("--json", type=Path)
    p = commands.add_parser("seedmean")
    p.add_argument("--soup", required=True)
    p.add_argument("seeds", nargs="+")
    p.add_argument("--json", type=Path)
    p = commands.add_parser("greedy-check")
    p.add_argument("base")
    p.add_argument("extended")
    p.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "table":
        result = table(args.refs, args.root, args.finalist_ref)
        print_table(result["candidates"])
        print(
            "guard-passing by Q: " + (", ".join(result["guard_passing_by_Q"]) or "none")
        )
        eligible = [c["name"] for c in result["candidates"] if c["finalist_eligible"]]
        if result["finalist_ref_Q"] is not None:
            print(
                f"finalist-eligible (new family, guards, Q >= {result['finalist_ref_Q'] - FINALIST_BAND:.2f}): "
                + (", ".join(eligible) or "none")
            )
    elif args.command == "seedmean":
        result = seedmean(args.soup, args.seeds, args.root)
        print_table(result["seeds"] + [result["soup"]])
        for e in result["excluded_seeds"]:
            print(f"excluded {e['name']}: {e['reason']}")
        print(
            f"seed-mean Q {result['seed_mean_Q']:.2f}; soup Q {result['soup']['Q']:.2f}; "
            f"artifact {result['artifact']} ({'soup' if result['artifact_is_soup'] else 'median-Q seed'}); "
            f"guards {'pass' if result['artifact_guards_pass'] else 'FAIL ' + ','.join(result['artifact_guards_failed'])}"
        )
    else:
        result = greedy_check(args.base, args.extended, args.root)
        print_table([result["S"], result["S_plus_g"]])
        d = result["delta"]
        print(
            f"dQ {d['Q']:+.2f}  dH3 {d['H3']:+.4f}  dChoice {d['choice']:+d}  dScore {d['score']:+d}; "
            + ", ".join(f"{k}={v}" for k, v in result["checks"].items())
        )
        print("ACCEPT" if result["accept"] else "REJECT")
    write(args.json, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
