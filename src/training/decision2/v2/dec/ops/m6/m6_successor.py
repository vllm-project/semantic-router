"""Decoder M6 successor rule per finalist and the per-tier choice (prereg dec-m6-prereg-2026-09-29.md, "Successor
rule", items 1-7, and "Choosing among passing finalists").

evaluate: one finalist's node-A run directory (REPORT.json, PAIRED-vs-<name>.json from m6-score.sh, M6-RECEIPT.json),
its types gate (TYPES.json), its mlx-diag paired file (MLX-PAIRED.json, v2.dec.mlx_paired), its overlap_effects run and
the exposure receipts of the candidate's new training files:

1. v3 paired 95% lower bound > 0 against the successor bar (PAIRED-vs-bar-t1: the current revision's T = 1 run).
2. `axis_ci95.H.delta.high` >= 0 against the bar.
3. No type COLLAPSED (`v2.eval.gates types`).
4. mlx-diag card-eligible type macro (Choice + Noul) paired 95% upper bound >= 0 (frozen panel, no problems).
5. Tier gates: lower bound > 0 against the tier's own 1.0 gate run (4B adopted Nox 1.0, 2B Sol 1.0 16K, 0.8B adopted
   Eos 1.0); v3 >= 55.7 (4B) / 44.5 (2B); H not significantly below (H high >= 0) Decider 4B and Jet v6.2 (4B) or
   Decider 2B (2B); no type collapsed.
6. (a) every new-training-file exposure receipt lists no group; (b) rules 1 and 5 hold on the reduced panels
   (the 84 flagged items removed, `overlap_effects run`), with no overlap problem and every stored file reproduced.
7. JevBench public 231: reported, never gating (unless an eval decision lands; see the note).

Reported: v3 / T / H, paired CIs against every comparator, the best same-size peer (highest v3 among the listed
peers) and public 231 by tier, the inherited exposure of a point that contains incumbent weights.

choose: among passing finalists of a tier, the highest paired lower bound against the best same-size peer, then the
higher lower bound against the bar (the incumbent's T = 1 run), then priority (the selection slot).

    python3 m6_successor.py evaluate --tier 4b --run RUN --types TYPES.json --mlx-paired MLX-PAIRED.json \
        --overlap overlap-effects.json [--exposure RECEIPT ...] --output PREFIX
    python3 m6_successor.py choose --tier 4b --result PREFIX.json [--result ...] --output PREFIX
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "dec-m6-successor/1"
CHOICE_SCHEMA = "dec-m6-successor-choice/1"
TIERS: dict[str, dict[str, Any]] = {
    "4b": {
        "label": "4B",
        "gate_own": "adopted-1.0",
        "v3_floor": 55.7,
        "h_peers": ["decider4b", "jet62"],
        "peers": ["decider4b", "jet62"],
        "inherited": "none (N4XF: no r2-excluded group)",
    },
    "2b": {
        "label": "2B",
        "gate_own": "same-limit-16k",
        "v3_floor": 44.5,
        "h_peers": ["decider2b"],
        "peers": ["decider2b", "thisthat12"],
        "inherited": "S2T: 24 groups / 57 rows of r2-excluded groups",
    },
    "08b": {
        "label": "0.8B",
        "gate_own": "adopted-1.0",
        "v3_floor": None,
        "h_peers": [],
        "peers": ["intern", "kev"],
        "inherited": "E8F: 33 r2-excluded groups",
    },
}
BAR = "bar-t1"
JEVBENCH_NOTE = (
    "public 231 reported without gating (prereg item 7); re-read the newest COORDINATION notes for an "
    "eval decision before selection"
)


def load(path: Path | None) -> dict[str, Any] | None:
    return (
        json.loads(Path(path).read_text())
        if path is not None and Path(path).is_file()
        else None
    )


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def item(passed: bool | None, **detail: Any) -> dict[str, Any]:
    return {"pass": passed, **detail}


def ci(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta": p["point"]["delta"]["score"],
        "ci95": p["ci95"],
        "H_delta": p["point"]["delta"]["H"],
        "H_ci95": p["axis_ci95"]["H"]["delta"],
        "right_v3": p["point"]["right"]["score"],
    }


def evaluate(
    tier: str,
    run: Path,
    types: dict[str, Any] | None,
    mlx: dict[str, Any] | None,
    overlap: dict[str, Any] | None,
    exposures: list[tuple[str, dict[str, Any] | None]],
) -> dict[str, Any]:
    cfg = TIERS[tier]
    report = load(run / "REPORT.json")
    receipt = load(run / "M6-RECEIPT.json") or {}
    paired = {
        p.name[len("PAIRED-vs-") : -len(".json")]: json.loads(p.read_text())
        for p in sorted(run.glob("PAIRED-vs-*.json"))
    }
    need = [BAR, cfg["gate_own"], *cfg["peers"]]
    missing = [n for n in need if n not in paired] + ([] if report else ["REPORT.json"])
    items: dict[str, Any] = {}
    bar = paired.get(BAR)
    items["1_v3_vs_bar"] = item(
        bar["ci95"]["low"] > 0 if bar else None, **(ci(bar) if bar else {})
    )
    items["2_H_vs_bar"] = item(
        bar["axis_ci95"]["H"]["delta"]["high"] >= 0 if bar else None,
        **({"H_ci95": bar["axis_ci95"]["H"]["delta"]} if bar else {}),
    )
    collapsed = None
    if types is not None:
        collapsed = sorted(k for k, v in types["types"].items() if v["verdict"] != "OK")
    items["3_types"] = item(
        None if types is None else not collapsed,
        collapsed=collapsed,
        verdicts=(
            None
            if types is None
            else {k: v["verdict"] for k, v in types["types"].items()}
        ),
    )
    if mlx is None:
        items["4_mlx_card_eligible"] = item(None, reason="no MLX-PAIRED file")
    else:
        ok = (
            mlx["card_eligible"]["ci95"]["high"] >= 0
            and mlx["frozen_panel"]
            and not mlx["problems"]
        )
        items["4_mlx_card_eligible"] = item(
            ok,
            reference=mlx["reference_name"],
            delta=mlx["card_eligible"]["delta"],
            ci95=mlx["card_eligible"]["ci95"],
            full_delta=mlx["full"]["delta"],
            full_ci95=mlx["full"]["ci95"],
            frozen_panel=mlx["frozen_panel"],
            problems=mlx["problems"],
        )
    own = paired.get(cfg["gate_own"])
    v3 = report["v3"]["score"] if report else None
    gate = {
        "own_1_0": item(
            own["ci95"]["low"] > 0 if own else None,
            name=cfg["gate_own"],
            **(ci(own) if own else {}),
        ),
        "v3_floor": item(
            None if v3 is None else (cfg["v3_floor"] is None or v3 >= cfg["v3_floor"]),
            v3=v3,
            floor=cfg["v3_floor"],
        ),
        "H_vs_peers": {
            n: item(
                (
                    paired[n]["axis_ci95"]["H"]["delta"]["high"] >= 0
                    if n in paired
                    else None
                ),
                **(
                    {"H_ci95": paired[n]["axis_ci95"]["H"]["delta"]}
                    if n in paired
                    else {}
                ),
            )
            for n in cfg["h_peers"]
        },
        "no_collapse": items["3_types"]["pass"],
    }
    parts = [
        gate["own_1_0"]["pass"],
        gate["v3_floor"]["pass"],
        gate["no_collapse"],
        *(v["pass"] for v in gate["H_vs_peers"].values()),
    ]
    items["5_tier_gates"] = item(all_true(parts), **gate)
    items["6a_exposure_new_files"] = item(
        (
            None
            if not exposures or any(d is None for _p, d in exposures)
            else all(not d["groups"] for _p, d in exposures)
        ),
        receipts=[
            {
                "file": p,
                "sha256": sha(Path(p)) if d is not None else None,
                "groups": None if d is None else len(d["groups"]),
            }
            for p, d in exposures
        ],
        reason=None if exposures else "no receipt for the new training files given",
    )
    items["6b_reduced_panels"] = reduced(cfg, overlap, run.name)
    public = report["panels"]["public231"] if report else None
    items["7_jevbench_public231"] = item(
        None,
        gating=False,
        note=JEVBENCH_NOTE,
        correct=public.get("correct") if public else None,
        tiers=public.get("tiers") if public else None,
    )
    peers = {n: ci(paired[n]) for n in cfg["peers"] if n in paired}
    best = max(peers, key=lambda n: peers[n]["right_v3"]) if peers else None
    weights = (receipt.get("selection") or {}).get("effective_weights") or {}
    rule = [
        items[k]["pass"]
        for k in (
            "1_v3_vs_bar",
            "2_H_vs_bar",
            "3_types",
            "4_mlx_card_eligible",
            "5_tier_gates",
            "6a_exposure_new_files",
            "6b_reduced_panels",
        )
    ]
    status = (
        "PASS"
        if all(r is True for r in rule)
        else ("FAIL" if any(r is False for r in rule) else "INCOMPLETE")
    )
    return {
        "schema": SCHEMA,
        "tier": tier,
        "run": str(run),
        "name": receipt.get("name") or run.name,
        "point": receipt.get("point"),
        "selection": receipt.get("selection"),
        "priority": (receipt.get("selection") or {}).get("slot"),
        "revision": receipt.get("revision"),
        "missing": missing,
        "v3": v3,
        "T": report["v3"]["T"] if report else None,
        "H": report["v3"]["H"] if report else None,
        "items": items,
        "status": status,
        "report_only": {
            "best_same_size_peer": best,
            "vs_best_peer": peers.get(best) if best else None,
            "vs_peers": peers,
            "vs_all": {n: ci(p) for n, p in paired.items()},
            "inherited_exposure": (
                cfg["inherited"]
                if "I" in weights
                else "none (no incumbent weight in the point)"
            ),
            "effective_weights": weights,
        },
        "inputs": {
            "report_sha256": sha(run / "REPORT.json") if report else None,
            "paired_sha256": {n: sha(run / f"PAIRED-vs-{n}.json") for n in paired},
        },
    }


def all_true(parts: list[bool | None]) -> bool | None:
    if any(p is False for p in parts):
        return False
    return True if all(p is True for p in parts) else None


def reduced(
    cfg: dict[str, Any], overlap: dict[str, Any] | None, cand: str
) -> dict[str, Any]:
    if overlap is None:
        return item(None, reason="no overlap_effects output")
    pairs = overlap["pairs"]

    def pair(right: str) -> dict[str, Any] | None:
        return pairs.get(f"{cand} - {right}")

    out: dict[str, Any] = {}
    bar = pair(BAR)
    own = pair(cfg["gate_own"])
    out["rule1_v3_vs_bar"] = item(
        bar["v3"]["reduced"]["ci95"]["low"] > 0 if bar else None,
        ci95=bar["v3"]["reduced"]["ci95"] if bar else None,
    )
    out["rule5_own_1_0"] = item(
        own["v3"]["reduced"]["ci95"]["low"] > 0 if own else None,
        ci95=own["v3"]["reduced"]["ci95"] if own else None,
    )
    red_v3 = overlap["models"].get(cand, {}).get("reduced", {}).get("v3")
    out["rule5_v3_floor"] = item(
        (
            None
            if red_v3 is None
            else (cfg["v3_floor"] is None or red_v3 >= cfg["v3_floor"])
        ),
        v3=red_v3,
        floor=cfg["v3_floor"],
    )
    out["rule5_H_vs_peers"] = {
        n: item(
            (
                pair(n)["v3"]["reduced"]["axis_ci95"]["H"]["delta"]["high"] >= 0
                if pair(n)
                else None
            ),
            H_ci95=(
                pair(n)["v3"]["reduced"]["axis_ci95"]["H"]["delta"] if pair(n) else None
            ),
        )
        for n in cfg["h_peers"]
    }
    repro = overlap.get("reproduction", [])
    out["overlap_problems"] = overlap.get("problems", [])
    out["reproduced"] = f"{sum(r['match'] for r in repro)}/{len(repro)}"
    parts = [
        out["rule1_v3_vs_bar"]["pass"],
        out["rule5_own_1_0"]["pass"],
        out["rule5_v3_floor"]["pass"],
        *(v["pass"] for v in out["rule5_H_vs_peers"].values()),
        not out["overlap_problems"] and all(r["match"] for r in repro),
    ]
    return item(all_true(parts), **out)


def fmt_ci(c: dict[str, float] | None, digits: int = 2) -> str:
    return "—" if not c else f"[{c['low']:+.{digits}f}, {c['high']:+.{digits}f}]"


def render(r: dict[str, Any]) -> str:
    it = r["items"]
    lines = [
        f"# M6 successor rule — {r['name']} ({TIERS[r['tier']]['label']})",
        "",
        f"Status: **{r['status']}**. Point `{r['point']}`, revision `{(r['revision'] or '')[:12]}`, "
        f"slot {r['priority']}. v3 {r['v3']}, T {r['T']}, H {r['H']}. Post-key same-panel evidence.",
        "",
        "| Item | Pass | Detail |",
        "| --- | --- | --- |",
    ]
    b = it["1_v3_vs_bar"]
    lines.append(
        f"| 1 v3 vs bar (T = 1) | {b['pass']} | {b.get('delta', 0):+.3f} {fmt_ci(b.get('ci95'))} |"
    )
    lines.append(
        f"| 2 H vs bar | {it['2_H_vs_bar']['pass']} | {fmt_ci(it['2_H_vs_bar'].get('H_ci95'), 4)} |"
    )
    lines.append(
        f"| 3 types | {it['3_types']['pass']} | collapsed: {it['3_types']['collapsed']} |"
    )
    m = it["4_mlx_card_eligible"]
    lines.append(
        f"| 4 mlx-diag card-eligible | {m['pass']} | {m.get('delta', 0):+.4f} {fmt_ci(m.get('ci95'), 4)} "
        f"(full {fmt_ci(m.get('full_ci95'), 4)}) |"
    )
    g = it["5_tier_gates"]
    lines.append(
        f"| 5 tier gates | {g['pass']} | own 1.0 {fmt_ci(g['own_1_0'].get('ci95'))}; v3 floor "
        f"{g['v3_floor']['floor']}; H vs peers {({k: v['pass'] for k, v in g['H_vs_peers'].items()})} |"
    )
    lines.append(
        f"| 6a exposure (new files) | {it['6a_exposure_new_files']['pass']} | "
        f"{[x['groups'] for x in it['6a_exposure_new_files']['receipts']]} groups |"
    )
    red = it["6b_reduced_panels"]
    lines.append(
        f"| 6b rules 1 and 5 without flagged items | {red['pass']} | "
        f"bar {fmt_ci((red.get('rule1_v3_vs_bar') or {}).get('ci95'))}; reproduced {red.get('reproduced')} |"
    )
    pub = it["7_jevbench_public231"]
    lines.append(
        f"| 7 public 231 (report) | — | {pub['correct']} correct; {pub['note']} |"
    )
    ro = r["report_only"]
    lines += [
        "",
        f"Best same-size peer: `{ro['best_same_size_peer']}` "
        f"{(ro['vs_best_peer'] or {}).get('delta', 0):+.3f} {fmt_ci((ro['vs_best_peer'] or {}).get('ci95'))}. "
        f"Inherited exposure: {ro['inherited_exposure']}.",
        "",
    ]
    if r["missing"]:
        lines += [f"Missing inputs: {r['missing']}", ""]
    return "\n".join(lines)


def choose(tier: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [r for r in results if r["status"] == "PASS"]

    def key(r: dict[str, Any]) -> tuple:
        peer = (
            (r["report_only"]["vs_best_peer"] or {})
            .get("ci95", {})
            .get("low", float("-inf"))
        )
        bar = r["items"]["1_v3_vs_bar"]["ci95"]["low"]
        return (-peer, -bar, r["priority"] if r["priority"] is not None else 99)

    ranked = sorted(passing, key=key)
    return {
        "schema": CHOICE_SCHEMA,
        "tier": tier,
        "rule": "highest paired lower bound vs the best same-size peer, then vs the bar (incumbent T = 1), then priority",
        "finalists": [
            {
                "name": r["name"],
                "point": r["point"],
                "status": r["status"],
                "priority": r["priority"],
                "lb_vs_best_peer": (r["report_only"]["vs_best_peer"] or {})
                .get("ci95", {})
                .get("low"),
                "lb_vs_bar": (r["items"]["1_v3_vs_bar"].get("ci95") or {}).get("low"),
                "run": r["run"],
                "revision": r["revision"],
            }
            for r in results
        ],
        "order": [r["name"] for r in ranked],
        "successor": ranked[0]["name"] if ranked else None,
        "successor_run": ranked[0]["run"] if ranked else None,
        "successor_revision": ranked[0]["revision"] if ranked else None,
        "pending": [r["name"] for r in results if r["status"] == "INCOMPLETE"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--tier", choices=sorted(TIERS), required=True)
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--types", type=Path)
    e.add_argument("--mlx-paired", type=Path)
    e.add_argument("--overlap", type=Path)
    e.add_argument("--exposure", action="append", default=[])
    e.add_argument(
        "--output",
        type=Path,
        required=True,
        help="prefix; writes PREFIX.json and PREFIX.md",
    )
    c = sub.add_parser("choose")
    c.add_argument("--tier", choices=sorted(TIERS), required=True)
    c.add_argument("--result", type=Path, action="append", required=True)
    c.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "evaluate":
        out = evaluate(
            args.tier,
            args.run,
            load(args.types),
            load(args.mlx_paired),
            load(args.overlap),
            [(x, load(Path(x))) for x in args.exposure],
        )
        Path(f"{args.output}.json").write_text(
            json.dumps(out, indent=1, sort_keys=True) + "\n"
        )
        Path(f"{args.output}.md").write_text(render(out) + "\n")
        print(
            json.dumps(
                {
                    "name": out["name"],
                    "status": out["status"],
                    "items": {k: v["pass"] for k, v in out["items"].items()},
                }
            )
        )
        return 0
    results = [json.loads(r.read_text()) for r in args.result]
    if any(r["tier"] != args.tier for r in results):
        p.error("results from another tier")
    out = choose(args.tier, results)
    Path(f"{args.output}.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n"
    )
    md = [
        f"# M6 successor choice ({TIERS[args.tier]['label']})",
        "",
        f"Rule: {out['rule']}.",
        "",
        "| Finalist | Status | Slot | LB vs best peer | LB vs bar |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for f in out["finalists"]:
        md.append(
            f"| {f['name']} | {f['status']} | {f['priority']} | {f['lb_vs_best_peer']} | {f['lb_vs_bar']} |"
        )
    md += [
        "",
        f"Successor: **{out['successor']}**"
        + (f"; pending: {out['pending']}" if out["pending"] else ""),
        "",
    ]
    Path(f"{args.output}.md").write_text("\n".join(md))
    print(
        json.dumps(
            {
                "successor": out["successor"],
                "order": out["order"],
                "pending": out["pending"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
