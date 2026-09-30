"""M4b deterministic rules over development readouts and formal gate outputs (CPU, stdlib).

``summary`` (host, after each readout): ``READOUT-M4B.json`` beside a readout's ``READOUT.json``:
P_dev, T_dev, H_pilot, H3 (the CSS-pilot three-task mean of macro-F1), per-type counts,
per-family accuracy, invalid answers, SELECT700 and CAL698 Brier. Gold-free.

``rules`` (host): per candidate (``NAME=READOUT_DIR``, the layout of ``run_readout.sh``)
the points and paired bootstrap of ``v2/27b/m3_contrast.py`` (typed-DEV item groups,
CSS-pilot items within task, AHO groups, CAL698 groups; 10,000 draws, seed 20260929),
extended by H3 on the same resamples, and the preregistered M4b rules:

- the 17:15 soup rule per arm with two seeds (``m3_contrast.soup_rule``: the soup if
  P_dev(soup) >= the seed mean, else the seed with the better SELECT700 — the trainer's
  when its run is given, else the readout's kernel-path SELECT700); a one-seed arm's
  artifact is that seed;
- proxy v2: a candidate is dropped only if its P_dev is >= 8 below the best M4b
  candidate's (every M4b readout except F1 and F1M);
- the interpolation line: F1M must agree with F1 on >= 99% of typed-DEV + CSS-pilot
  argmax answers, else the line stops. S is A2's artifact, else A1's. With F1's readout
  as the reference, alpha is eligible iff (i) every typed-DEV type's correct count is
  >= F1's - 0.03 n_t, (ii) H3 >= F1's H3, (iii) no typed-DEV family is more than 0.10
  below F1's. G(alpha) = T_dev(alpha) - T_dev(F1); G* = max G over eligible alpha in
  {1/3, 1/2, 2/3, 1} (alpha = 1 is S); alpha* = the smallest eligible alpha in
  {1/3, 1/2, 2/3} with G >= 0.75 G*; no pick if none is eligible, G* < 0.01 or only
  alpha = 1 qualifies. Counts and accuracies are compared as exact fractions;
- finalists (at most three) in fixed priority: A2's artifact, A1's artifact, theta(alpha*)
  when the line picks, A3-s1's BEST; a failed or proxy-dropped candidate passes its slot
  to the next in order (slots F-a, F-b, F-c are filled in that order);
- development contrasts A2 - A1 (per seed and pooled), A1 - F1 (per seed and pooled),
  A3-s1 - A2-s1 on T_dev, per type, H_pilot, H3, P_dev and CAL698 Brier, with the
  report-only retention floors of every M4b candidate vs F1.

``overlap-spec`` (host): the per-candidate ``v2.eval.overlap_effects run`` spec derived from
the committed template: the candidate named, sealed M4b finalists kept as internal peers,
the candidate's own exposure receipt and its stored paired files to reproduce.

``verdicts`` (host): the successor rule vs F1 and the "beats AutoJev" bar from the
``v2.eval.gates`` paired / types outputs, the overlap exposure receipt and, when it exists,
the mlx-diag paired interval of an ``overlap_effects run`` with the mlx panel.

Development readouts only for ``rules``: never release scores.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

m3 = importlib.import_module("v2.27b.m3_contrast")
contrast = importlib.import_module("v2.27b.contrast")

SCHEMA = "decision2-27b-m4b-rules/1"
PILOT_TASKS = ("discourse", "implicit_hate", "semeval_stance")
ALPHAS = (Fraction(1, 3), Fraction(1, 2), Fraction(2, 3))
TYPE_SLACK = Fraction(3, 100)
FAMILY_SLACK = Fraction(1, 10)
G_MIN = Fraction(1, 100)
G_SHARE = Fraction(3, 4)
AGREEMENT_MIN = Fraction(99, 100)
SLOTS = ("F-a", "F-b", "F-c")
V3_FLOOR = 64.92
AUTOJEV_V3 = 72.133
F1_NAME = "F1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------- summary


def summary(readout_dir: Path, checkpoint: str | None = None) -> dict[str, Any]:
    readout = read_json(readout_dir / "READOUT.json")
    typed, css = readout["typed_dev"], readout["css_pilot"]
    tasks = css["tasks"]
    if set(tasks) != set(PILOT_TASKS):
        raise ValueError(f"CSS pilot tasks {sorted(tasks)} != {list(PILOT_TASKS)}")
    cal_path = readout_dir / "cal698.summary.json"
    cal = read_json(cal_path) if cal_path.is_file() else None
    calibration = readout_dir / "cal" / "calibration.json"
    out = {
        "schema": "decision2-27b-m4b-readout-summary/1",
        "scope": "development readout; never a release or formal score",
        "label": readout.get("label"),
        "readout_dir": str(readout_dir),
        "checkpoint": checkpoint,
        "model_sha256": (
            read_json(calibration)["model_sha256"] if calibration.is_file() else None
        ),
        "P_dev": readout["development_proxy"],
        "T_dev": typed["T_dev"],
        "H_pilot": css["H_pilot"],
        "H3": sum(tasks[t] for t in PILOT_TASKS) / len(PILOT_TASKS),
        "css_pilot_tasks": {t: tasks[t] for t in PILOT_TASKS},
        "typed_by_type": typed["by_type"],
        "typed_by_family": typed["by_family"],
        "invalid_or_missing": {
            "typed_dev": typed["invalid_or_missing"],
            "css_pilot": css["invalid_or_missing"],
        },
        "select700": {
            k: (readout.get("select") or {}).get(k)
            for k in ("n", "correct", "family_macro_accuracy", "family_macro_brier")
        },
        "cal698": (
            None
            if cal is None
            else {
                "brier": sum(b["brier"] * b["n"] for b in cal["by_type"].values())
                / sum(b["n"] for b in cal["by_type"].values()),
                "family_macro_brier": cal["family_macro_brier"],
                "family_macro_accuracy": cal["family_macro_accuracy"],
            }
        ),
    }
    if not math.isclose(
        out["P_dev"], 100 * math.sqrt(out["T_dev"] * out["H_pilot"]), abs_tol=1e-9
    ):
        raise ValueError(
            "READOUT.json development_proxy is not 100*sqrt(T_dev*H_pilot)"
        )
    return out


# ---------------------------------------------------------------- rule metrics


def rule_metrics(panels: Any, candidate: Any) -> dict[str, Any]:
    """Exact per-type counts, per-family counts, T_dev (fraction) and H3 of one candidate."""
    types: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    families: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in candidate.dev:
        for family, kind, correct, _ in item:
            types[kind][0] += correct
            types[kind][1] += 1
            families[family][0] += correct
            families[family][1] += 1
    css = contrast.css_macro(panels, candidate.css, panels.task_items)
    t = sum(Fraction(c, n) for c, n in families.values()) / len(families)
    readout = candidate.readout.get("typed_dev", {})
    for kind, cell in (readout.get("by_type") or {}).items():
        if [cell["correct"], cell["n"]] != types[kind]:
            raise ValueError(f"{candidate.name}: READOUT.json {kind} counts differ")
    for family, acc in (readout.get("by_family") or {}).items():
        if not math.isclose(
            acc, families[family][0] / families[family][1], abs_tol=1e-12
        ):
            raise ValueError(f"{candidate.name}: READOUT.json family {family} differs")
    return {
        "types": {k: list(v) for k, v in sorted(types.items())},
        "families": {k: list(v) for k, v in sorted(families.items())},
        "T": t,
        "H3": statistics.fmean(css[f"css:{task}"] for task in panels.tasks),
    }


def eligibility(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    types = {
        k: Fraction(cand["types"].get(k, [0, n])[0]) >= c - TYPE_SLACK * n
        for k, (c, n) in ref["types"].items()
    }
    families = {
        f: Fraction(*cand["families"].get(f, [0, n])) >= Fraction(c, n) - FAMILY_SLACK
        for f, (c, n) in ref["families"].items()
    }
    checks = {
        "types_within_0.03n": all(types.values()),
        "h3_at_least_f1": cand["H3"] >= ref["H3"],
        "families_within_0.10": all(families.values()),
    }
    return {
        "checks": checks,
        "failed_types": sorted(k for k, ok in types.items() if not ok),
        "failed_families": sorted(f for f, ok in families.items() if not ok),
        "eligible": all(checks.values()),
    }


def alpha_rule(
    ref: dict[str, Any], line: dict[Fraction, dict[str, Any] | None]
) -> dict[str, Any]:
    """``line`` maps alpha in {1/3, 1/2, 2/3, 1} to rule metrics (None: no readout)."""
    rows: dict[str, Any] = {}
    eligible: dict[Fraction, Fraction] = {}
    for alpha in (*ALPHAS, Fraction(1)):
        metrics = line.get(alpha)
        if metrics is None:
            rows[str(alpha)] = {"readout": False, "eligible": False}
            continue
        check = eligibility(ref, metrics)
        g = metrics["T"] - ref["T"]
        rows[str(alpha)] = {"readout": True, **check, "G": float(g)}
        if check["eligible"]:
            eligible[alpha] = g
    out: dict[str, Any] = {"alphas": rows, "alpha_star": None}
    if not eligible:
        return {**out, "G_star": None, "pick": False, "reason": "no alpha is eligible"}
    g_star = max(eligible.values())
    out["G_star"] = float(g_star)
    if g_star < G_MIN:
        return {**out, "pick": False, "reason": "G* < 0.01"}
    qualified = [a for a in ALPHAS if a in eligible and eligible[a] >= G_SHARE * g_star]
    if not qualified:
        return {**out, "pick": False, "reason": "only alpha = 1 qualifies"}
    return {
        **out,
        "alpha_star": str(qualified[0]),
        "pick": True,
        "reason": "smallest eligible alpha in {1/3, 1/2, 2/3} with G >= 0.75 G*",
    }


def finalist_list(
    order: list[tuple[str, str | None]], usable: set[str], limit: int = len(SLOTS)
) -> list[dict[str, str]]:
    """``order``: (role, candidate or None) in priority; the first ``limit`` usable fill the slots."""
    picked = [
        (role, name) for role, name in order if name is not None and name in usable
    ]
    return [
        {"slot": slot, "role": role, "candidate": name}
        for slot, (role, name) in zip(SLOTS, picked[:limit])
    ]


# ---------------------------------------------------------------- answers agreement


def decision(answer: Any) -> Any:
    """The argmax decision of one answer (choice, noul >= 0.5, score); None if malformed."""
    if not isinstance(answer, dict):
        return None
    kind = answer.get("type")
    if kind == "noul":
        p = answer.get("noul")
        return (
            ("noul", p >= 0.5)
            if isinstance(p, (int, float)) and not isinstance(p, bool)
            else None
        )
    key = "score" if kind == "score" else "choice"
    return (kind or "choice", answer.get(key)) if answer.get(key) is not None else None


def decisions(path: Path) -> dict[tuple[str, str], Any]:
    return {
        (row["id"], key): decision(answer)
        for row in contrast.read_jsonl(path)
        for key, answer in (row.get("answers") or {}).items()
    }


def agreement(left: Path, right: Path, panels: Any) -> dict[str, Any]:
    """Share of typed-DEV questions and CSS-pilot items where both give the same valid argmax."""
    slots = {
        "typed-dev": [
            (item["id"], key) for item in panels.dev for key in item["questions"]
        ],
        "css-pilot": [(row["id"], "label") for row in panels.css],
    }
    out: dict[str, Any] = {}
    agree, total = 0, 0
    for panel, keys in slots.items():
        a = decisions(left / "output" / f"{panel}.predictions.jsonl")
        b = decisions(right / "output" / f"{panel}.predictions.jsonl")
        same = sum(a.get(k) is not None and a.get(k) == b.get(k) for k in keys)
        out[panel] = {"agree": same, "n": len(keys)}
        agree, total = agree + same, total + len(keys)
    return {
        **out,
        "agree": agree,
        "n": total,
        "rate": agree / total,
        "min_rate": float(AGREEMENT_MIN),
        "passes": Fraction(agree, total) >= AGREEMENT_MIN,
    }


# ---------------------------------------------------------------- paired bootstrap


def paired(
    panels: Any, left: list, right: list, draws: int, seed: int
) -> dict[str, Any]:
    """``contrast.paired`` on the same resamples, with H3 added."""
    rng = random.Random(seed)
    metrics: dict[str, list[float]] = defaultdict(list)
    slices = (
        sorted(set.intersection(*(set(a.aho) for a in left + right)))
        if left[0].aho
        else []
    )
    for _ in range(draws):
        groups = [
            panels.dev_groups[rng.randrange(len(panels.dev_groups))]
            for _ in panels.dev_groups
        ]
        dev_sample = [i for g in groups for i in g]
        css_sample = {
            t: [items[rng.randrange(len(items))] for _ in items]
            for t, items in panels.task_items.items()
        }
        aho_samples = {}
        for name in slices:
            keys = sorted(left[0].aho[name])
            aho_samples[name] = [keys[rng.randrange(len(keys))] for _ in keys]
        values = defaultdict(list)
        for sign, arms in ((1, left), (-1, right)):
            for arm in arms:
                dev = contrast.dev_metrics(panels, arm.dev, dev_sample)
                css = contrast.css_macro(panels, arm.css, css_sample)
                sample = {
                    **dev,
                    "H_pilot": css["H_pilot"],
                    "H3": statistics.fmean(css[f"css:{t}"] for t in panels.tasks),
                    "P_dev": 100 * math.sqrt(dev["T_dev"] * css["H_pilot"]),
                }
                for name, keys in aho_samples.items():
                    flat = [v for k in keys for v in arm.aho[name][k]]
                    sample[f"aho:{name}"] = sum(flat) / len(flat)
                for key, value in sample.items():
                    values[key].append(sign * value / len(arms))
        for key, parts in values.items():
            metrics[key].append(sum(parts))
    return {
        key: {
            "ci95": [contrast.percentile(v, 0.025), contrast.percentile(v, 0.975)],
            "draws": draws,
        }
        for key, v in sorted(metrics.items())
    }


def compare(
    points: dict[str, dict[str, float]], left: list[str], right: list[str], boot: dict
) -> dict:
    common = set.intersection(*(set(points[n]) for n in left + right))
    trim = lambda names: [
        {k: v for k, v in points[n].items() if k in common} for n in names
    ]  # noqa: E731
    decision = contrast.decide(trim(left), trim(right), boot, "T_dev", 0.0)
    floors = {k: decision["checks"][k] for k in m3.FLOORS}
    return {
        "delta": decision["delta"],
        "retention_floors": {
            **floors,
            "all_pass": all(floors.values()),
            "report_only": True,
        },
    }


def default_contrasts(arms: dict[str, list[str]], present: set[str]) -> list[str]:
    def seed(arm: str, k: int) -> str | None:
        name = f"{arm}-s{k}"
        return name if name in arms.get(arm, []) and name in present else None

    specs = []
    a1 = [seed("A1", k) for k in (1, 2)]
    a2 = [seed("A2", k) for k in (1, 2)]
    for x, y in zip(a2, a1):
        if x and y:
            specs.append(f"{x}:{y}")
    if all(a1) and all(a2):
        specs.append(f"{a2[0]}+{a2[1]}:{a1[0]}+{a1[1]}")
    for x in a1:
        if x:
            specs.append(f"{x}:{F1_NAME}")
    if all(a1):
        specs.append(f"{a1[0]}+{a1[1]}:{F1_NAME}")
    if seed("A3", 1) and a2[0]:
        specs.append(f"A3-s1:{a2[0]}")
    return specs


def parse_arm(spec: str) -> tuple[str, str | None, list[str]]:
    """ARM=SOUP:S1+S2 or ARM=SEED -> (arm, soup, seeds)."""
    arm, rest = spec.split("=", 1)
    if ":" in rest:
        soup, seeds = rest.split(":", 1)
        return arm, soup, seeds.split("+")
    return arm, None, [rest]


def rules(args: argparse.Namespace) -> dict[str, Any]:
    inputs = m3.Inputs()
    if args.dev_gold or args.css_gold:
        dev_gold, css_gold = args.dev_gold, args.css_gold
    else:
        m3.panel_registry.verify(args.panel_root, ["typed-dev", "css-pilot"])
        dev_gold = m3.panel_registry.path(args.panel_root, "typed-dev", "gold")
        css_gold = m3.panel_registry.path(args.panel_root, "css-pilot", "gold")
    panels = contrast.Panels(inputs(dev_gold), inputs(css_gold))
    if inputs.sha(args.cal_rows) != args.cal_sha256:
        raise ValueError("CAL rows differ from their pinned SHA-256")
    cal = m3.CalRows(args.cal_rows)
    aho = m3.pairs(args.aho)
    runs = m3.pairs(args.train_run)
    failed = set(args.failed)
    dirs = {F1_NAME: args.f1, **m3.pairs(args.candidate)}
    candidates = {
        name: m3.Candidate(name, directory, panels, cal, aho, inputs, runs.get(name))
        for name, directory in dirs.items()
        if name not in failed
    }
    points = {name: c.point(panels) for name, c in candidates.items()}
    metrics = {name: rule_metrics(panels, c) for name, c in candidates.items()}
    for name in points:
        points[name]["H3"] = metrics[name]["H3"]

    selects = {}
    for name, c in candidates.items():
        readout_select = c.readout.get("select") or {}
        selects[name] = c.select or (
            {**{k: readout_select.get(k) for k in ("n", "correct", "family_macro_accuracy", "family_macro_brier")},
             "source": "readout kernel-path SELECT700"}  # fmt: skip
            if readout_select.get("family_macro_accuracy") is not None
            else None
        )
    shim = {name: SimpleNamespace(select=s) for name, s in selects.items()}

    arms: dict[str, list[str]] = {}
    soup_rules, artifacts = {}, {}
    for spec in args.arm:
        arm, soup, seeds = parse_arm(spec)
        arms[arm] = seeds
        alive = [s for s in seeds if s in candidates]
        if not alive:
            soup_rules[arm] = {"artifact": None, "reason": "no seed has a readout"}
        elif soup is None or len(alive) == 1:
            soup_rules[arm] = {
                "artifact": alive[0],
                "seeds": alive,
                "reason": "one seed: its BEST",
            }
        elif soup not in candidates:
            ranked = sorted(
                alive,
                key=lambda s: (
                    -selects[s]["family_macro_accuracy"],
                    selects[s]["family_macro_brier"],
                ),
            )
            soup_rules[arm] = {"artifact": ranked[0], "soup": soup, "seeds": alive,
                               "reason": "soup failed or has no readout; seed with the higher SELECT700",
                               "select700": {s: selects[s] for s in alive}}  # fmt: skip
        else:
            soup_rules[arm] = m3.soup_rule(points, shim, soup, alive)
        artifacts[arm] = soup_rules[arm]["artifact"]

    pool = sorted(n for n in candidates if n not in (F1_NAME, args.f1m))
    screen = m3.proxy_screen(points, pool) if pool else {"candidates": {}}
    kept = {n for n, v in screen["candidates"].items() if v["kept"]}

    line: dict[str, Any] = {"S": artifacts.get("A2") or artifacts.get("A1")}
    line["S_arm"] = (
        "A2" if artifacts.get("A2") else ("A1" if artifacts.get("A1") else None)
    )
    if args.f1m and args.f1m in candidates:
        line["f1m_agreement"] = agreement(
            candidates[args.f1m].directory, candidates[F1_NAME].directory, panels
        )
    else:
        line["f1m_agreement"] = None
    alpha_names = {
        Fraction(s.split("=", 1)[0]): s.split("=", 1)[1] for s in args.line_alpha
    }
    line["alpha_candidates"] = {str(a): n for a, n in sorted(alpha_names.items())}
    if line["S"] is None:
        line["alpha_rule"] = {
            "pick": False,
            "reason": "no S artifact (A2 and A1 have none)",
        }
    elif line["f1m_agreement"] is None or not line["f1m_agreement"]["passes"]:
        line["alpha_rule"] = {
            "pick": False,
            "reason": "F1M readout missing or agrees with F1 on < 99%",
        }
    else:
        inputs_line = {a: metrics.get(n) for a, n in alpha_names.items()}
        inputs_line[Fraction(1)] = metrics[line["S"]]
        line["alpha_rule"] = alpha_rule(metrics[F1_NAME], inputs_line)
    pick = line["alpha_rule"].get("alpha_star")
    theta = alpha_names.get(Fraction(pick)) if pick else None

    a3 = next((s for s in arms.get("A3", []) if s.endswith("-s1")), None)
    order = [
        ("A2 artifact (both levers)", artifacts.get("A2")),
        ("A1 artifact (capacity control)", artifacts.get("A1")),
        (f"theta(alpha* = {pick})", theta),
        ("A3-s1 BEST", a3 if a3 in candidates else None),
    ]
    usable = {n for n in candidates if n in kept}
    finalists = finalist_list(order, usable)

    specs = args.contrast or default_contrasts(arms, set(candidates))
    contrasts = {}
    for spec in specs:
        treat, control = spec.split(":")
        left = [candidates[n] for n in treat.split("+")]
        right = [candidates[n] for n in control.split("+")]
        boot = paired(panels, left, right, args.draws, args.seed)
        boot["cal_brier"] = m3.paired_cal(
            cal, left, right, args.draws, f"{args.seed}/cal698"
        )
        contrasts[spec] = {
            "treatment": treat.split("+"),
            "control": control.split("+"),
            **compare(points, treat.split("+"), control.split("+"), boot),
            "bootstrap": boot,
        }
    no_ci = {k: {"ci95": [0.0, 0.0]} for k in ("T_dev",)}
    retention = {
        n: compare(points, [n], [F1_NAME], no_ci)["retention_floors"] for n in pool
    }

    inputs(Path(__file__))
    inputs(Path(m3.__file__))
    inputs(Path(contrast.__file__))
    return {
        "schema": SCHEMA,
        "label": "development readouts; never a release or formal score",
        "draws": args.draws,
        "seed": args.seed,
        "failed": sorted(failed),
        "points": points,
        "rule_metrics": {n: {**m, "T": float(m["T"])} for n, m in metrics.items()},
        "details": {name: c.details() for name, c in candidates.items()},
        "soup_rule": soup_rules,
        "proxy_screen": screen,
        "line": line,
        "finalists": {
            "priority": [{"role": r, "candidate": n} for r, n in order],
            "rule": "fixed priority; a failed or proxy-dropped candidate passes its slot to the next in order",
            "slots": finalists,
        },
        "contrasts": contrasts,
        "retention_floors_vs_F1": retention,
        "inputs_sha256": dict(sorted(inputs.files.items())),
    }


# ---------------------------------------------------------------- overlap spec


def overlap_spec(template: dict[str, Any], name: str, exposure: Path) -> dict[str, Any]:
    """Template keys: ``m4b`` (slot -> label/run/note), ``comparators`` (label -> spec model),
    ``tier`` (the 27B tier without ``candidate``), ``reproduce_names`` (right label -> stored
    file in the candidate's formal run) and the spec's own ``label``/``panel_root``/``focus_task``.
    """
    slots = template["m4b"]
    if name not in slots:
        raise ValueError(f"{name} is not an M4b finalist slot of the template")
    sealed = {
        slot: cfg
        for slot, cfg in slots.items()
        if (Path(cfg["run"]) / "SEAL.json").is_file()
    }
    if name not in sealed:
        raise ValueError(f"{name}: formal run {slots[name]['run']} is not sealed")
    label = slots[name]["label"]
    models = {}
    for slot, cfg in sealed.items():
        models[cfg["label"]] = {"run": cfg["run"], "note": cfg["note"]}
        if slot == name:
            models[cfg["label"]]["exposure"] = [str(exposure)]
    models.update(template["comparators"])
    others = [cfg["label"] for slot, cfg in sealed.items() if slot != name]
    tier = {
        **template["tier"],
        "candidate": label,
        "internal_peers": [*template["tier"]["internal_peers"], *others],
    }
    reproduce = [
        {
            "left": label,
            "right": right,
            "stored": str(Path(slots[name]["run"]) / stored),
        }
        for right, stored in template["reproduce_names"].items()
        if (Path(slots[name]["run"]) / stored).is_file()
    ]
    spec = {k: template[k] for k in ("label", "panel_root", "focus_task")}
    return {**spec, "models": models, "tiers": {"27B": tier}, "reproduce": reproduce}


# ---------------------------------------------------------------- verdicts


def low_high(ci: Any) -> tuple[float, float]:
    return (ci["low"], ci["high"]) if isinstance(ci, dict) else (ci[0], ci[1])


def item(value: Any, passed: bool | None, note: str = "") -> dict[str, Any]:
    status = "PENDING" if passed is None else ("PASS" if passed else "FAIL")
    return {"value": value, "status": status, **({"note": note} if note else {})}


def combined(items: list[dict[str, Any]]) -> str:
    states = {i["status"] for i in items}
    return (
        "FAIL" if "FAIL" in states else ("PENDING" if "PENDING" in states else "PASS")
    )


def verdicts(args: argparse.Namespace) -> dict[str, Any]:
    f1 = read_json(args.paired_f1)
    peers = {k: read_json(v) for k, v in m3.pairs(args.paired_peer).items()}
    types = read_json(args.types)["types"]
    exposure = read_json(args.exposure)
    report = read_json(args.report)
    v3 = f1["point"]["left"]["score"]
    ci = low_high(f1["ci95"])
    h_f1 = low_high(f1["axis_ci95"]["H"]["delta"])
    collapsed = sorted(k for k, v in types.items() if v["verdict"] != "OK")
    peer_h = {k: low_high(p["axis_ci95"]["H"]["delta"]) for k, p in peers.items()}
    exposed = bool(exposure["groups"]) or bool(exposure["matched_rows"])
    overlap = read_json(args.overlap) if args.overlap else None
    if overlap is not None:
        exposed = exposed or any(t.get("any") for t in overlap["exposure"].values())
    mlx = None
    if args.mlx_overlap:
        pairs = read_json(args.mlx_overlap)["pairs"]
        entry = pairs.get(f"{args.label} - {args.f1_label}") or {}
        if entry.get("mlx"):
            mlx = low_high(entry["mlx"]["full"]["ci95"])
    successor = {
        "1_v3_ci_lower_vs_F1_above_0": item(
            {"delta": f1["point"]["delta"]["score"], "ci95": ci}, ci[0] > 0
        ),
        "2_H_not_significantly_below_F1": item(
            {"delta": f1["point"]["delta"]["H"], "ci95": h_f1}, h_f1[1] >= 0
        ),
        "3_no_type_collapsed": item(
            {k: v["verdict"] for k, v in types.items()}, not collapsed
        ),
        "4_mlx_diag_not_significantly_below_F1": item(
            {"ci95": mlx},
            None if mlx is None else mlx[1] >= 0,
            (
                ""
                if mlx is not None
                else "needs the mlx-diag paired interval (overlap_effects run with the mlx panel)"
            ),
        ),
        "5_no_1_0_tier_gates": item(
            {
                "v3": v3,
                "v3_floor": V3_FLOOR,
                "H_vs_peers_ci95": peer_h,
                "collapsed": collapsed,
            },
            v3 >= V3_FLOOR
            and all(h[1] >= 0 for h in peer_h.values())
            and not collapsed
            and len(peer_h) == 3,
        ),
        "6_no_overlap_exposure": item(
            {
                "groups": len(exposure["groups"]),
                "overlap_run": str(args.overlap) if args.overlap else None,
            },
            not exposed,
        ),
    }
    public = report["panels"]["public231"]
    successor["7_public231_reported"] = {
        "value": {
            k: public.get(k)
            for k in ("correct", "items", "tier_macro_accuracy", "tiers")
        },
        "status": "REPORTED",
    }
    aj = peers.get(args.autojev)
    beats = None
    if aj is not None:
        aj_ci, aj_h = low_high(aj["ci95"]), peer_h[args.autojev]
        aj_v3 = aj["point"]["left"]["score"]
        beats = {
            "v3_above_72.133": item(aj_v3, aj_v3 > AUTOJEV_V3),
            "v3_ci_lower_vs_AutoJev_above_0": item(
                {"delta": aj["point"]["delta"]["score"], "ci95": aj_ci}, aj_ci[0] > 0
            ),
            "H_not_significantly_below_AutoJev": item(
                {"delta": aj["point"]["delta"]["H"], "ci95": aj_h}, aj_h[1] >= 0
            ),
        }
    gating = [v for k, v in successor.items() if not k.startswith("7_")]
    return {
        "schema": "decision2-27b-m4b-verdicts/1",
        "candidate": args.label,
        "inputs": {
            "paired_f1": str(args.paired_f1),
            "paired_peers": {k: str(v) for k, v in m3.pairs(args.paired_peer).items()},
            "types": str(args.types),
            "exposure": str(args.exposure),
            "overlap": str(args.overlap) if args.overlap else None,
            "mlx_overlap": str(args.mlx_overlap) if args.mlx_overlap else None,
            "report": str(args.report),
        },
        "successor_rule": {"items": successor, "verdict": combined(gating)},
        "beats_autojev": (
            None
            if beats is None
            else {"items": beats, "verdict": combined(list(beats.values()))}
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--readout-dir", type=Path, required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("rules")
    p.add_argument("--panel-root", type=Path, default=m3.panel_registry.DEFAULT_ROOT)
    p.add_argument("--dev-gold", type=Path, help="override (no panel check)")
    p.add_argument("--css-gold", type=Path, help="override (no panel check)")
    p.add_argument("--cal-rows", type=Path, required=True)
    p.add_argument("--cal-sha256", required=True)
    p.add_argument("--f1", type=Path, required=True, help="F1's readout directory")
    p.add_argument("--candidate", action="append", default=[], help="NAME=READOUT_DIR")
    p.add_argument(
        "--train-run", action="append", default=[], help="NAME=TRAINER_RUN_DIR"
    )
    p.add_argument(
        "--arm", action="append", default=[], help="ARM=SOUP:SEED1+SEED2 or ARM=SEED"
    )
    p.add_argument("--f1m", help="candidate name of F1M's readout")
    p.add_argument(
        "--line-alpha", action="append", default=[], help="ALPHA=NAME (1/3, 1/2, 2/3)"
    )
    p.add_argument(
        "--failed", action="append", default=[], help="failed candidate NAME"
    )
    p.add_argument("--aho", action="append", default=[], help="SLICE=ROWS")
    p.add_argument(
        "--contrast", action="append", default=[], help="TREAT[+T2]:CONTROL[+C2]"
    )
    p.add_argument("--draws", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("overlap-spec")
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--name", required=True, help="finalist slot (F-a, F-b, F-c)")
    p.add_argument("--exposure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("verdicts")
    p.add_argument("--label", required=True, help="the candidate's display name")
    p.add_argument("--f1-label", default="DEV2.0-27B (F1)")
    p.add_argument("--autojev", default="AutoJev-27B")
    p.add_argument(
        "--paired-f1", type=Path, required=True, help="gates paired, candidate - F1"
    )
    p.add_argument(
        "--paired-peer", action="append", default=[], help="NAME=gates paired JSON"
    )
    p.add_argument("--types", type=Path, required=True)
    p.add_argument("--exposure", type=Path, required=True)
    p.add_argument("--overlap", type=Path)
    p.add_argument(
        "--mlx-overlap", type=Path, help="overlap_effects run output with the mlx panel"
    )
    p.add_argument(
        "--report", type=Path, required=True, help="the candidate's formal REPORT.json"
    )
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "summary":
        result = summary(args.readout_dir, args.checkpoint)
        write_once(args.output, result)
        print(json.dumps({k: result[k] for k in ("P_dev", "T_dev", "H_pilot", "H3")}))
    elif args.mode == "rules":
        result = rules(args)
        write_once(args.output, result)
        print(json.dumps({"finalists": result["finalists"]["slots"], "alpha_rule": result["line"]["alpha_rule"].get("reason"),
                          "soup_rule": {a: r["artifact"] for a, r in result["soup_rule"].items()}}))  # fmt: skip
    elif args.mode == "overlap-spec":
        result = overlap_spec(read_json(args.template), args.name, args.exposure)
        write_once(args.output, result)
        print(
            json.dumps(
                {
                    "candidate": result["tiers"]["27B"]["candidate"],
                    "models": sorted(result["models"]),
                }
            )
        )
    else:
        result = verdicts(args)
        write_once(args.output, result)
        print(json.dumps({"successor_rule": result["successor_rule"]["verdict"],
                          "beats_autojev": (result["beats_autojev"] or {}).get("verdict")}))  # fmt: skip


if __name__ == "__main__":
    main()
