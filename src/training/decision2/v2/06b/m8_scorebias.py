"""M8 5-level Score offsets: fit, development checks D1-D3, selection, replay (stdlib only).

    python3 -m v2.06b.m8_scorebias fit --lam {1,0.5,0} --human-weight {0,1} --name NAME
        --output score_bias.json --report FIT.json
    python3 -m v2.06b.m8_scorebias check --score-bias score_bias.json --name NAME --output CHECK.json
    python3 -m v2.06b.m8_scorebias select --checks CHECK.json ... --output SELECT.json
    python3 -m v2.06b.m8_scorebias replay --online score5t-dev.predictions.jsonl
        --score-bias score_bias.json [--check CHECK.json] --output D4.json
    python3 -m v2.06b.m8_scorebias score-report --run RUN --gates RUN.gates --label NAME --output OUT
    python3 -m v2.06b.m8_scorebias mlx-paired --candidate-run RUN-mlx --released-run REL-mlx --output OUT
    python3 -m v2.06b.m8_scorebias xcheck --run RUN --mlx-run RUN-mlx --score-bias score_bias.json
        [--reference-run M6RUN --reference-mlx-run M6RUN-mlx] --output OUT
    python3 -m v2.06b.m8_scorebias successor --run RUN --gates RUN.gates --exposure EXPOSURE.json
        --d4 D4.json --label NAME --output SUCCESSOR.json
    python3 -m v2.06b.m8_scorebias choose --successor SUCCESSOR.json ... [--between PAIRED.json ...]
        --output RELEASE-CHOICE.json

M8 prereg sections 1-8. Every input defaults to its node path and is verified against the
preregistered SHA-256 (the combined panels through `v2.eval.panels.verify`).

Fit (section 3): offsets b_0..b_4 on 5-level Score logits z = log max(p, 1e-12). Ledger rows
are the 400 score5t-dev fit-half items (probabilities of the `m6-mxcx-soup` validation
collection); human rows (w_h > 0 only) are the M7(a) FIT_AHO and CAL698 rows with L = 5
minus every Score5-DEV panel row. Within source s, row i weighs
a_i = n_{s,y_i}^(-lam) / sum_j n_{s,y_j}^(-lam); J = sum_ledger a_i l_i + w_h sum_human a_i l_i,
minimized with b_0 = 0 by M7's Newton solver (float64, backtracking, gradient norm < 1e-9,
at most 200 steps), then shifted to mean zero and rounded to 6 decimals.

Checks (section 4) apply the rounded file values: D1 is `v2.eval.score5t.blocks` on the
combined gold in file order (check block: neither COLLAPSE nor WARN), D2 the typed-DEV Score
top share (<= .90), D3 on CHK_5 (paired delta upper bound >= 0, modal share <= .90,
accuracy - always-majority lower bound > 0; M7's statistics). Selection follows section 5,
replay is D4 (section 6).

After a formal run (sections 7-8): `score-report` applies `v2.eval.score5t.summary` (the D1
statistics: histogram, top share and accuracy with Wilson CIs, always-majority, accuracy -
always-majority by paired bootstrap, 5,000 draws, seed 20260929, COLLAPSE / WARN / NO-GAIN) to
the 400 typed FINAL Score slots, next to the R3 verdicts of `gates types`. `mlx-paired` (R4)
scores every mlx-diag item as `v2.eval.multilingual_panel.score` does, reproduces both runs'
stored per-type accuracies, and bootstraps candidate - released type-macro accuracy with items
resampled within type (the same draws for both runs; 5,000, seed 20260929): Choice + Noul
(gated, 95% upper bound >= 0) and all three types (reported). `xcheck` (report only) compares
the finalist's answers with the frozen soup's formal answers: Choice / Noul and non-5-level
Score identical, 5-level Score equal to the offline correction of the stored probabilities.
`successor` combines R1-R7 into SUCCESSOR.json; `choose` applies the release-choice rule to
finalists given in section 5 rank order.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.score import unique_argmax
from training.model.infer import normalized_answer
from training.model.score_bias import (
    SCORE_BIAS_FORMAT,
    load_score_bias,
    validate_score_bias,
)

from .common import file_sha256, read_jsonl

m7 = importlib.import_module("v2.06b.m7_scorebias")

PREREG = "v2/06b/records/m8-prereg-2026-09-29.md"
PREREG_COMMIT = "e6a5f4515b902f2760ff768d47bf50c9eafb3849"
LEVELS = 5
KEYS = [str(k) for k in range(LEVELS)]
MODEL_SHA256 = "01fae75098ce29314c5de2990321fcfd31457de7c02bb757bd8345662a823854"
EXPORT_MANIFEST_SHA256 = (
    "3ae009ad838487c0df839a55898a98f5665276136d2a8cab3c1260dd0b29dfa2"
)
TOP_SHARE = 0.9
REPLAY_TOLERANCE = 1e-4
FINALISTS = 3
CANDIDATES = {
    "s5-b1": (1.0, 0.0),
    "s5h-b1": (1.0, 1.0),
    "s5-b05": (0.5, 0.0),
    "s5h-b05": (0.5, 1.0),
    "s5-b0": (0.0, 0.0),
    "s5h-b0": (0.0, 1.0),
}

PANEL_ROOT = Path("/data/dev2/private/panels")
SCORE5T_RUN = Path("/data/dev2/runs/eval/score5t-dev/collect/m6-mxcx-soup")
SCORE5T_PREDICTIONS = SCORE5T_RUN / "output" / "score5t-dev.predictions.jsonl"
SCORE5T_READOUT = SCORE5T_RUN / "READOUT-score5t.json"
FIT_GOLD = PANEL_ROOT / "gold" / "score5t-dev.fit.gold.jsonl"
CHECK_GOLD = PANEL_ROOT / "gold" / "score5t-dev.check.gold.jsonl"
AHO_DIR = Path("/data/dev2/runs/06b/m7/a/aho")
PROBS = Path("/data/dev2/runs/06b/m7/a/probs/PROBS.json")
SOUP_DIR = Path("/data/dev2/runs/06b/m1/arms/m6-mxcx-soup/full")
SCORE5_PANEL_ROWS = Path("/data/dev2/runs/eval/score5-dev/build-v1/panel-rows.jsonl")
SCORE5_RUN = Path("/data/dev2/runs/eval/score5-dev/m6-mxcx-soup")
DEV_PREDICTIONS = Path(
    "/data/dev2/runs/06b/m1/arms/m6-mxcx-soup/readout/dev.predictions.jsonl"
)
EXPECTED = {
    "score5t_predictions": "033148fc38ab2aadbffd26d09c8516e4b16f39d286f425e7d8f2dd50731f9642",
    "score5t_fit_gold": "3d735da439ac7a459bffcc931b44202541e8a3d55b82960c70b0eedff9efb5f6",
    "score5t_check_gold": "2e4a9490642cb04da7c016c4ebd22a9366666f0e464cfbc418371a92266adb3d",
    "score5_panel_rows": "889ff2c7609f79d0af9eb9f470b0d76b45630a6956ab12c0abd34cb5e2599138",
    "fit_aho": "4bfe6731c5ef12782a626f4eacbd5844e0557deb468cd0e3662f08b754657a9e",
    "fit_aho_probs": "238ce555511fde5f18d0e4b1f65d374e2fb5a5a99dce365e7882948df7ec1292",
    "cal_probs": "df223972b1a642a500ab2f8f38af33108edcecc41ff6ee504e1320d6618fa127",
    "cal698_ids": "7e8c01c8552f56ba30f4dd1a5c0ca7c2e74baf4412d2533eecbd23b77300f947",
    "cal_gold": "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    "chk": "98425b276dd2540c2886b9a7ec904a85101f6f79d0fd642e5e9fafbf79282b16",
    "chk_probs": "ad845b8d3a935bd76a177fca3159f406bd126b3e1db52759c499519fb8abb934",
    "dev_predictions": "b008c258773699687827ba7ba1843475314d0b883a828647686620bd6770dadc",
}
OBJECTIVE = (
    "J(b) = sum_ledger a_i l_i(b) + w_h sum_human a_i l_i(b), "
    "l_i(b) = -log softmax(z_i + b)_{y_i}, z = log max(p, 1e-12), "
    "a_i = n_{s,y_i}^(-lam) / sum_{j in s} n_{s,y_j}^(-lam), b_0 = 0"
)
SOLVER = (
    f"Newton, float64, backtracking, gradient norm < {m7.GRAD_TOLERANCE}, "
    f"at most {m7.MAX_NEWTON_STEPS} steps (m7_scorebias)"
)
POST = f"shift to mean zero, round to {m7.DECIMALS} decimals"


def verified(path: Path, role: str) -> dict[str, str]:
    digest = file_sha256(path)
    if digest != EXPECTED[role]:
        raise ValueError(f"{path}: sha256 {digest} is not the preregistered {role}")
    return {"path": str(path), "sha256": digest}


def refuse_existing(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(path)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ fit


def source_weights(labels: list[int], lam: float) -> list[float]:
    """a_i = n_{y_i}^(-lam) / sum_j n_{y_j}^(-lam) within one source (sums to 1)."""
    counts = Counter(labels)
    raw = [counts[y] ** (-lam) for y in labels]
    total = sum(raw)
    return [value / total for value in raw]


def fit_weighted(
    rows: list[tuple[list[float], int]], weights: list[float], levels: int
) -> dict[str, Any]:
    """`m7_scorebias.fit_level` with given row weights (same loop and settings)."""
    present = Counter(y for (_, y), w in zip(rows, weights) if w > 0)
    missing = [k for k in range(levels) if present[k] == 0]
    if missing:
        raise m7.FitError(f"L={levels}: weighted rows miss level(s) {missing}")
    b = [0.0] * levels
    value = start = m7.balanced_nll(rows, weights, b)
    norm = math.inf
    for step in range(m7.MAX_NEWTON_STEPS + 1):
        g, h = m7.gradient_hessian(rows, weights, b)
        norm = math.sqrt(sum(x * x for x in g))
        if norm < m7.GRAD_TOLERANCE:
            break
        if step == m7.MAX_NEWTON_STEPS:
            raise m7.FitError(f"L={levels}: Newton did not converge (|g| = {norm:.3g})")
        d = m7.solve(h, [-x for x in g])
        t = 1.0
        while True:
            trial = [0.0] + [b[j + 1] + t * d[j] for j in range(levels - 1)]
            trial_value = m7.balanced_nll(rows, weights, trial)
            slope = sum(gj * dj for gj, dj in zip(g, d))
            if norm < 1e-3 or trial_value <= value + 1e-4 * t * slope or t < 1e-8:
                break
            t /= 2
        b, value = trial, trial_value
    mean = sum(b) / levels
    return {
        "offsets": [round(v - mean, m7.DECIMALS) + 0.0 for v in b],
        "raw_offsets": b,
        "newton_steps": step,
        "gradient_norm": norm,
        "objective_start": start,
        "objective_end": value,
    }


def objective_rows(
    sources: dict[str, list[dict[str, Any]]], lam: float, human_weight: float
) -> tuple[list[tuple[list[float], int]], list[float]]:
    """Rows and weights of J: each source's weights sum to 1, the human block's times w_h."""
    rows: list[tuple[list[float], int]] = []
    weights: list[float] = []
    for name, items in sources.items():
        scale = 1.0 if name == "ledger" else human_weight
        if scale == 0 or not items:
            continue
        labels = [i["y"] for i in items]
        rows += [(m7.logits_of(i["p"]), i["y"]) for i in items]
        weights += [scale * a for a in source_weights(labels, lam)]
    return rows, weights


def by_level(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(i["y"] for i in items)
    return {str(k): counts[k] for k in range(LEVELS)}


def score5t_gold(panel_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    from v2.eval import panels
    from v2.eval.score5t import PANEL

    observed = panels.verify(panel_root, [PANEL])
    return read_jsonl(panels.path(panel_root, PANEL, "gold")), observed


def score5t_predictions(path: Path) -> tuple[dict[str, dict[str, Any]], dict]:
    provenance = verified(path, "score5t_predictions")
    rows = {r["id"]: r for r in read_jsonl(path)}
    models = Counter(r.get("model_sha256") for r in rows.values())
    if set(models) != {MODEL_SHA256}:
        raise ValueError(
            f"{path}: model_sha256 {dict(models)} is not the frozen soup's"
        )
    return rows, provenance


def probabilities(answer: Any) -> list[float] | None:
    """The 5-level Score probabilities of an answer, or None when it has none."""
    if not isinstance(answer, dict) or answer.get("type") != "score":
        return None
    values = answer.get("probabilities")
    if not isinstance(values, dict) or set(values) != set(KEYS):
        return None
    out = [values[k] for k in KEYS]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in out):
        return None
    return [float(v) for v in out]


def ledger_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict]:
    fit_gold = read_jsonl(args.fit_gold)
    check_gold = read_jsonl(args.check_gold)
    inputs = {
        "score5t_fit_gold": verified(args.fit_gold, "score5t_fit_gold"),
        "score5t_check_gold": verified(args.check_gold, "score5t_check_gold"),
    }
    combined, panel = score5t_gold(args.panel_root)
    halves = {r["id"]: r["half"] for r in combined}
    fit_ids = [r["id"] for r in fit_gold]
    check_ids = {r["id"] for r in check_gold}
    if any(r["half"] != "fit" or halves.get(r["id"]) != "fit" for r in fit_gold):
        raise ValueError("a fit-gold row is not in the panel's fit half")
    if any(halves.get(k) != "check" for k in check_ids) or check_ids & set(fit_ids):
        raise ValueError("a check-half id is in the fit rows")
    if len(set(fit_ids)) != len(fit_ids):
        raise ValueError("fit ids repeat")
    predictions, inputs["score5t_predictions"] = score5t_predictions(args.predictions)
    if set(predictions) != set(halves):
        raise ValueError("predictions do not cover exactly the score5t-dev ids")
    items = []
    for row in fit_gold:
        p = probabilities(predictions[row["id"]]["answers"].get("decision"))
        y = row["gold"]["decision"]["value"]
        if p is None or y not in range(LEVELS):
            raise ValueError(f"{row['id']}: no 5-level probabilities or gold level")
        items.append({"id": row["id"], "source": "ledger", "p": p, "y": y})
    inputs["score5t_panel"] = panel
    return items, inputs


def human_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict]:
    for role, path in (
        ("fit_aho", args.aho_dir / "fit_aho.jsonl"),
        ("cal698_ids", args.aho_dir / "cal698.score.ids.json"),
        ("cal_probs", args.soup_dir / "cal.probs.jsonl"),
        ("cal_gold", args.cal_gold),
        ("score5_panel_rows", args.score5_panel_rows),
    ):
        verified(path, role)
    items, provenance = m7.fit_inputs(args)
    if provenance["fit_aho_probs_sha256"] != EXPECTED["fit_aho_probs"]:
        raise ValueError("FIT_AHO probabilities are not the preregistered file")
    if provenance["best_export_manifest_sha256"] != EXPORT_MANIFEST_SHA256:
        raise ValueError("CAL698 probabilities come from another export")
    five = [i for i in items if i["levels"] == LEVELS]
    panel = read_jsonl(args.score5_panel_rows)
    excluded = {r["source_row_id"] for r in panel} | {r["panel_id"] for r in panel}
    kept = [i for i in five if i["id"] not in excluded]
    removed = Counter(i["source"] for i in five if i["id"] in excluded)
    report = {
        "rows_l5_before_removal": len(five),
        "rows_l5_before_removal_by_source": dict(
            sorted(Counter(i["source"] for i in five).items())
        ),
        "score5_dev_panel_rows": len(panel),
        "score5_dev_rule": "drop a human row whose id is a panel row's source_row_id "
        "or panel_id",
        "removed_by_source": {
            s: removed[s] for s in sorted({i["source"] for i in five})
        },
        "removed": sum(removed.values()),
        "rows": len(kept),
        "rows_by_source": dict(sorted(Counter(i["source"] for i in kept).items())),
        "rows_by_level": by_level(kept),
        "inputs": {
            **provenance,
            "aho_dir": str(args.aho_dir),
            "probs_manifest": str(args.probs),
            "soup_dir": str(args.soup_dir),
            "cal_gold": str(args.cal_gold),
            "score5_panel_rows": {
                "path": str(args.score5_panel_rows),
                "sha256": EXPECTED["score5_panel_rows"],
            },
        },
    }
    return [{**i, "source_arm": i["source"], "source": "human"} for i in kept], report


def fit(args: argparse.Namespace) -> int:
    started = time.monotonic()
    refuse_existing(args.output, args.report)
    registered = CANDIDATES.get(args.name)
    if registered != (args.lam, args.human_weight):
        raise ValueError(f"{args.name}: (lam, w_h) differs from prereg section 3")
    model_sha = m7.soup_model_sha256(args.soup_dir)
    if model_sha != MODEL_SHA256:
        raise ValueError("the soup's best-export is not the frozen model")
    ledger, inputs = ledger_items(args)
    human, human_report = ([], None)
    if args.human_weight > 0:
        human, human_report = human_items(args)
    rows, weights = objective_rows(
        {"ledger": ledger, "human": human}, args.lam, args.human_weight
    )
    record: dict[str, Any] = {
        "prereg": f"{PREREG} section 3",
        "prereg_commit": PREREG_COMMIT,
        "candidate": args.name,
        "lam": args.lam,
        "human_weight": args.human_weight,
        "levels": LEVELS,
        "objective": OBJECTIVE,
        "solver": SOLVER,
        "post": POST,
        "model_sha256": model_sha,
        "export_manifest_sha256": EXPORT_MANIFEST_SHA256,
        "inputs": inputs,
        "rows": {
            "ledger": {"rows": len(ledger), "rows_by_level": by_level(ledger)},
            "human": human_report or {"rows": 0, "used": False},
            "objective_rows": len(rows),
            "block_weight": {"ledger": 1.0, "human": args.human_weight},
        },
    }
    report: dict[str, Any] = {"schema": "dev2-06b-m8-fit/1", "status": "FIT"}
    try:
        result = fit_weighted(rows, weights, LEVELS)
    except m7.FitError as exc:
        report.update(status="FIT_FAILED", error=str(exc), **record)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        m7.save(args.report, report)
        print(json.dumps({"status": "FIT_FAILED", "error": str(exc)}))
        return 1
    offsets = result.pop("offsets")
    record["fit"] = {**result, "offsets": offsets}
    if args.lam == 1 and args.human_weight == 0:
        reference = m7.fit_level([(m7.logits_of(i["p"]), i["y"]) for i in ledger], 5)
        record["fit"]["m7_fit_level_offsets"] = reference["offsets"]
        if reference["offsets"] != offsets:
            raise ValueError("lam = 1, w_h = 0 differs from m7_scorebias.fit_level")
    bias = {
        "format": SCORE_BIAS_FORMAT,
        "model_sha256": model_sha,
        "offsets": {str(LEVELS): offsets},
        "fit": record,
    }
    validate_score_bias(bias, MODEL_SHA256)
    sha = m7.save(args.output, bias)
    report.update(record)
    report.update(
        score_bias={"path": str(args.output), "sha256": sha},
        offsets={str(LEVELS): offsets},
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    m7.save(args.report, report)
    print(json.dumps({"candidate": args.name, "offsets": offsets, "sha256": sha}))
    return 0


# ---------------------------------------------------------------- checks


def corrected_answer(answer: Any, row: list[float]) -> Any:
    """The answer with offsets on log max(p, 1e-12); answers without 5-level
    probabilities are returned unchanged."""
    p = probabilities(answer)
    if p is None:
        return answer
    logits = [z + b for z, b in zip(m7.logits_of(p), row)]
    return normalized_answer("score", KEYS, logits, 1.0)


def corrected_predictions(
    rows: dict[str, dict[str, Any]], row: list[float]
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "answers": {"decision": corrected_answer(r["answers"].get("decision"), row)}
        }
        for key, r in rows.items()
    }


def d1_gate(check_block: dict[str, Any]) -> dict[str, Any]:
    flags = check_block["flags"]
    return {
        "block": "score5t-dev check half (v2.eval.score5t.blocks)",
        "flags": flags,
        "top_share": check_block["top_share"],
        "top_share_wilson95": check_block["top_share_wilson95"],
        "accuracy": check_block["accuracy"],
        "accuracy_wilson95": check_block["accuracy_wilson95"],
        "always_majority_accuracy": check_block["always_majority_accuracy"],
        "acc_minus_majority_boot95": check_block["acc_minus_majority_boot95"],
        "pass": "COLLAPSE" not in flags and "WARN" not in flags,
    }


def d3_gate(before: dict, after: dict, paired: dict) -> dict[str, Any]:
    checks = {
        "a_delta_upper_bound_ge_0": paired["ci95"][1] >= 0,
        "b_modal_share_le_0.90": after["top_share"] <= TOP_SHARE,
        "c_delta_vs_majority_lower_bound_gt_0": after["delta_vs_majority_ci95"][0] > 0,
    }
    return {
        "n": after["n"],
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "delta_accuracy": paired["delta_accuracy"],
        "delta_accuracy_ci95": paired["ci95"],
        "modal_share_after": after["top_share"],
        "modal_value_after": after["top_value"],
        "majority_accuracy": after["majority_accuracy"],
        "delta_vs_majority_after": after["delta_vs_majority"],
        "delta_vs_majority_ci95_after": after["delta_vs_majority_ci95"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def score5t_section(args, row: list[float]) -> tuple[dict[str, Any], dict]:
    from v2.eval import score5t

    gold, panel = score5t_gold(args.panel_root)
    predictions, provenance = score5t_predictions(args.predictions)
    fixed = corrected_predictions(predictions, row)
    before = score5t.blocks(gold, predictions)
    after = score5t.blocks(gold, fixed)
    changed = sum(
        m7.predicted(probabilities(predictions[k]["answers"].get("decision")) or [])
        != m7.predicted(probabilities(fixed[k]["answers"]["decision"]) or [])
        for k in predictions
        if probabilities(predictions[k]["answers"].get("decision"))
    )
    section: dict[str, Any] = {
        "inputs": {"panel": panel, "predictions": provenance},
        "uncorrectable_answers": sum(
            probabilities(r["answers"].get("decision")) is None
            for r in predictions.values()
        ),
        "changed_answers": changed,
        "before": before,
        "after": after,
    }
    if args.readout is not None and args.readout.is_file():
        stored = load_json(args.readout)["score5t"]
        section["uncorrected_matches_readout"] = {
            "path": str(args.readout),
            "sha256": file_sha256(args.readout),
            "match": all(stored[b] == before[b] for b in score5t.BLOCKS),
        }
    return section, d1_gate(after["check"])


def typed_dev_section(args, row: list[float]) -> tuple[dict[str, Any], dict]:
    inputs = {"dev_predictions": verified(args.dev_predictions, "dev_predictions")}
    dev = m7.dev_items(args.dev_predictions, args.dev_gold, {LEVELS: row})
    if set(dev["model_sha256"]) != {MODEL_SHA256}:
        raise ValueError("typed-DEV readout is not of the frozen model")
    before = m7.summary(dev["before"], dev["gold"], intervals=True)
    after = m7.summary(dev["after"], dev["gold"], intervals=True)
    inputs["dev_gold"] = {
        "path": str(args.dev_gold),
        "sha256": file_sha256(args.dev_gold),
    }
    gate = {
        "n": after["n"],
        "top_share": after["top_share"],
        "top_value": after["top_value"],
        "correct": after["correct"],
        "accuracy": after["accuracy"],
        "top_share_before": before["top_share"],
        "correct_before": before["correct"],
        "pass": after["top_share"] <= TOP_SHARE,
    }
    return {
        "inputs": inputs,
        "before": before,
        "after": after,
        "paired_vs_uncorrected": m7.paired_delta(
            dev["before"], dev["after"], dev["gold"]
        ),
    }, gate


def chk5_section(args, row: list[float]) -> tuple[dict[str, Any], dict]:
    m7.aho_manifest(args.aho_dir)
    inputs = {"chk": verified(args.aho_dir / "chk.jsonl", "chk")}
    path, probs = m7.probs_output(args.probs, args.aho_dir / "chk.jsonl")
    inputs["chk_probs"] = verified(path, "chk_probs")
    complete = load_json(args.soup_dir / "COMPLETE.json")
    if probs["state_sha256"] != complete["state_sha256"]:
        raise ValueError("CHK probabilities come from another state than the soup")
    if complete["best_export_manifest_sha256"] != EXPORT_MANIFEST_SHA256:
        raise ValueError("the soup's COMPLETE.json binds another export")
    items = [
        i
        for i in m7.aho_items(args.aho_dir, "chk.jsonl", args.probs)
        if i["levels"] == LEVELS
    ]
    gold = [i["y"] for i in items]
    before = [m7.predicted(i["p"]) for i in items]
    after = [m7.predicted(m7.corrected(i["p"], row)) for i in items]
    summaries = {
        "before": m7.summary(before, gold, intervals=True),
        "after": m7.summary(after, gold, intervals=True),
    }
    paired = m7.paired_delta(before, after, gold)
    section = {
        "inputs": {**inputs, "state_sha256": probs["state_sha256"]},
        **summaries,
        "paired_vs_uncorrected": paired,
        "by_arm": m7.by_source(items, {LEVELS: row}),
    }
    return section, d3_gate(summaries["before"], summaries["after"], paired)


def score5_dev_section(args, row: list[float]) -> dict[str, Any]:
    from v2.eval import panels, score5

    observed = panels.verify(args.panel_root, ["score5-dev"])
    gold = read_jsonl(panels.path(args.panel_root, "score5-dev", "gold"))
    path = args.score5_run / "output" / "score5-dev.predictions.jsonl"
    seal = load_json(args.score5_run / "SEAL-SCORE5.json")
    if file_sha256(path) != seal["predictions_sha256"]:
        raise ValueError(f"{path} changed after its seal")
    predictions = {r["id"]: r for r in read_jsonl(path)}
    if {r.get("model_sha256") for r in predictions.values()} != {MODEL_SHA256}:
        raise ValueError("Score5-DEV collection is not of the frozen model")
    return {
        "scope": "reported only (never gated or used for selection)",
        "inputs": {
            "panel": observed,
            "predictions": {"path": str(path), "sha256": seal["predictions_sha256"]},
        },
        "before": score5.summary(gold, predictions),
        "after": score5.summary(gold, corrected_predictions(predictions, row)),
    }


def check(args: argparse.Namespace) -> int:
    started = time.monotonic()
    refuse_existing(args.output)
    offsets, bias = load_score_bias(args.score_bias, MODEL_SHA256)
    if set(offsets) != {LEVELS}:
        raise ValueError("M8 offsets hold level count 5 only")
    if bias["fit"].get("candidate") != args.name:
        raise ValueError(f"{args.score_bias} was fitted for another candidate")
    row = offsets[LEVELS]
    score5t_block, d1 = score5t_section(args, row)
    dev_block, d2 = typed_dev_section(args, row)
    chk_block, d3 = chk5_section(args, row)
    gates = {"D1": d1, "D2": d2, "D3": d3}
    gates["eligible"] = all(g["pass"] for g in gates.values())
    result = {
        "schema": "dev2-06b-m8-check/1",
        "prereg": f"{PREREG} section 4",
        "prereg_commit": PREREG_COMMIT,
        "candidate": args.name,
        "lam": bias["fit"]["lam"],
        "human_weight": bias["fit"]["human_weight"],
        "model_sha256": MODEL_SHA256,
        "score_bias": {
            "path": str(args.score_bias),
            "sha256": file_sha256(args.score_bias),
            "offsets": bias["offsets"],
        },
        "gates": gates,
        "score5t": score5t_block,
        "typed_dev_score": dev_block,
        "chk5": chk_block,
        "score5_dev": score5_dev_section(args, row),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    m7.save(args.output, result)
    print(
        json.dumps(
            {
                "candidate": args.name,
                "D1": [d1["pass"], d1["flags"], d1["accuracy"], d1["top_share"]],
                "D2": [d2["pass"], d2["top_share"]],
                "D3": [d3["pass"], d3["checks"]],
                "eligible": gates["eligible"],
            }
        )
    )
    return 0


# --------------------------------------------------------------- select


def ranking(checks: list[dict[str, Any]]) -> dict[str, Any]:
    order = list(CANDIDATES)
    names = [c["candidate"] for c in checks]
    if len(set(names)) != len(names) or not set(names) <= set(order):
        raise ValueError(f"candidates {names} repeat or are not preregistered")
    table = []
    for c in checks:
        gates = c["gates"]
        table.append(
            {
                "candidate": c["candidate"],
                "order": order.index(c["candidate"]) + 1,
                "lam": c["lam"],
                "human_weight": c["human_weight"],
                "score_bias_sha256": c["score_bias"]["sha256"],
                "offsets": c["score_bias"]["offsets"][str(LEVELS)],
                "D1": gates["D1"]["pass"],
                "D2": gates["D2"]["pass"],
                "D3": gates["D3"]["pass"],
                "eligible": bool(
                    gates["D1"]["pass"] and gates["D2"]["pass"] and gates["D3"]["pass"]
                ),
                "check_flags": gates["D1"]["flags"],
                "check_accuracy": gates["D1"]["accuracy"],
                "check_top_share": gates["D1"]["top_share"],
                "chk5_accuracy": gates["D3"]["accuracy_after"],
            }
        )
    eligible = sorted(
        (r for r in table if r["eligible"]),
        key=lambda r: (
            -r["check_accuracy"],
            -r["chk5_accuracy"],
            r["check_top_share"],
            r["order"],
        ),
    )
    for rank, entry in enumerate(eligible, 1):
        entry["rank"] = rank
    for entry in table:
        entry.setdefault("rank", None)
    table.sort(key=lambda r: (r["rank"] is None, r["rank"] or 0, r["order"]))
    return {
        "rule": "eligible = D1 and D2 and D3; rank by check-half accuracy (desc), then "
        "CHK_5 corrected accuracy (desc), check-half top share (asc), section 3 order; "
        f"finalists = the first {FINALISTS} eligible",
        "candidates_checked": len(table),
        "candidates_missing": [n for n in order if n not in names],
        "table": table,
        "finalists": [r["candidate"] for r in eligible[:FINALISTS]],
        "successor_possible": bool(eligible),
    }


def select(args: argparse.Namespace) -> int:
    refuse_existing(args.output)
    checks = [load_json(path) for path in args.checks]
    for path, c in zip(args.checks, checks):
        if c.get("schema") != "dev2-06b-m8-check/1":
            raise ValueError(f"{path}: not an M8 CHECK.json")
    result = {
        "schema": "dev2-06b-m8-select/1",
        "prereg": f"{PREREG} section 5",
        "prereg_commit": PREREG_COMMIT,
        "checks": [{"path": str(p), "sha256": file_sha256(p)} for p in args.checks],
        **ranking(checks),
    }
    m7.save(args.output, result)
    print(
        json.dumps(
            {
                "finalists": result["finalists"],
                "table": [
                    [r["candidate"], r["rank"], r["eligible"], r["check_accuracy"]]
                    for r in result["table"]
                ],
            }
        )
    )
    return 0


# --------------------------------------------------------------- replay


def replay_result(
    gold: list[dict[str, Any]],
    logged: dict[str, dict[str, Any]],
    online: dict[str, dict[str, Any]],
    row: list[float],
    bias_sha: str,
    manifest: dict[str, Any] | None,
    replicates: int | None = None,
) -> dict[str, Any]:
    from v2.eval import score5t

    ids = {r["id"] for r in gold}
    offline = corrected_predictions(logged, row)
    worst, argmax_mismatch, missing = 0.0, 0, 0
    for key in sorted(ids & set(logged) & set(online)):
        expected = probabilities(offline[key]["answers"]["decision"])
        observed = probabilities((online[key].get("answers") or {}).get("decision"))
        if expected is None or observed is None:
            missing += 1
            continue
        worst = max(worst, max(abs(a - b) for a, b in zip(expected, observed)))
        argmax_mismatch += m7.predicted(expected) != m7.predicted(observed)
    extra = {} if replicates is None else {"replicates": replicates}
    online_flags = score5t.blocks(gold, online, **extra)["check"]["flags"]
    offline_flags = score5t.blocks(gold, offline, **extra)["check"]["flags"]
    checks = {
        "same_items": set(logged) == ids == set(online),
        "all_answers_have_probabilities": missing == 0,
        "same_argmax_all_items": argmax_mismatch == 0 and missing == 0,
        "max_abs_dp_le_1e-4": worst <= REPLAY_TOLERANCE,
        "check_flags_equal_offline": online_flags == offline_flags,
        "online_rows_score_bias_sha256": all(
            r.get("score_bias_sha256") == bias_sha for r in online.values()
        ),
        "online_rows_model_sha256": all(
            r.get("model_sha256") == MODEL_SHA256 for r in online.values()
        ),
        "logged_rows_model_sha256": all(
            r.get("model_sha256") == MODEL_SHA256 for r in logged.values()
        ),
    }
    if manifest is not None:
        checks["online_manifest_score_bias_sha256"] = (
            manifest.get("score_bias_sha256") == bias_sha
        )
        checks["online_manifest_model_sha256"] = (
            manifest.get("model_sha256") == MODEL_SHA256
        )
    return {
        "items": len(ids),
        "max_abs_dp": worst,
        "argmax_mismatches": argmax_mismatch,
        "answers_without_probabilities": missing,
        "check_flags_online": online_flags,
        "check_flags_offline": offline_flags,
        "checks": checks,
    }


def replay(args: argparse.Namespace) -> int:
    refuse_existing(args.output)
    offsets, _ = load_score_bias(args.score_bias, MODEL_SHA256)
    bias_sha = file_sha256(args.score_bias)
    gold, panel = score5t_gold(args.panel_root)
    logged, logged_provenance = score5t_predictions(args.logged)
    online = {r["id"]: r for r in read_jsonl(args.online)}
    manifest_path = args.online.with_name(args.online.name + ".manifest.json")
    manifest = load_json(manifest_path) if manifest_path.is_file() else None
    result = replay_result(gold, logged, online, offsets[LEVELS], bias_sha, manifest)
    if args.check is not None:
        stored = load_json(args.check)
        if stored["score_bias"]["sha256"] != bias_sha:
            raise ValueError(f"{args.check} checked another score_bias.json")
        result["checks"]["check_flags_equal_d1"] = (
            result["check_flags_online"] == stored["gates"]["D1"]["flags"]
        )
    result = {
        "schema": "dev2-06b-m8-replay/1",
        "prereg": f"{PREREG} section 6 (D4)",
        "prereg_commit": PREREG_COMMIT,
        "panel": panel,
        "logged": logged_provenance,
        "online": {"path": str(args.online), "sha256": file_sha256(args.online)},
        "online_manifest": (
            {"path": str(manifest_path), "sha256": file_sha256(manifest_path)}
            if manifest is not None
            else None
        ),
        "check": (
            {"path": str(args.check), "sha256": file_sha256(args.check)}
            if args.check is not None
            else None
        ),
        "score_bias_sha256": bias_sha,
        "model_sha256": MODEL_SHA256,
        **result,
    }
    result["D4"] = {"pass": all(result["checks"].values())}
    m7.save(args.output, result)
    print(
        json.dumps(
            {
                "D4": result["D4"],
                "checks": result["checks"],
                "max_abs_dp": result["max_abs_dp"],
            }
        )
    )
    return 0


# --------------------------------------------------------- score report


def typed_final_score_rows(
    gold: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Each typed FINAL Score slot as a one-question `score5t` row plus its answer."""
    rows: list[dict[str, Any]] = []
    answers_by_slot: dict[str, dict[str, Any]] = {}
    for item in gold:
        answers = (predictions.get(item["id"]) or {}).get("answers")
        for key, question in item["questions"].items():
            if question["type"] != "score":
                continue
            if len(question["criteria"]) != LEVELS:
                raise ValueError(
                    f"{item['id']}/{key}: typed FINAL Score is not 5-level"
                )
            slot = f"{item['id']}#{key}"
            rows.append(
                {
                    "id": slot,
                    "task": "typed-final/score",
                    "group_id": slot,
                    "source": "typed-final",
                    "language": item.get("language", "en"),
                    "long": False,
                    "questions": {"decision": question},
                    "gold": {"decision": item["gold"][key]},
                }
            )
            if isinstance(answers, dict) and key in answers:
                answers_by_slot[slot] = {"answers": {"decision": answers[key]}}
    return rows, answers_by_slot


def score_report_result(
    gold: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    replicates: int | None = None,
) -> dict[str, Any]:
    from v2.eval import score5t
    from v2.eval.gates import type_summary
    from v2.eval.sealed.score import outcomes

    rows, answers = typed_final_score_rows(gold, predictions)
    extra = {} if replicates is None else {"replicates": replicates}
    score = score5t.summary(rows, answers, **extra)
    pairs = [(o[5], o[6]) for o in outcomes(rows, answers)]
    m7_stats = m7.summary([p for _, p in pairs], [g for g, _ in pairs], intervals=True)
    types = type_summary(gold, predictions)
    verdicts = {k: v["verdict"] for k, v in types.items()}
    return {
        "score": score,
        "no_gain": "NO-GAIN" in score["flags"],
        "m7_gate_statistics": {
            k: m7_stats[k]
            for k in (
                "correct",
                "accuracy",
                "majority_level",
                "majority_accuracy",
                "delta_vs_majority",
                "delta_vs_majority_ci95",
                "top_value",
                "top_share",
                "kappa",
                "recall_by_level",
            )
        },
        "agreement": {
            "correct_equals_gates_types": score["correct"]
            == round(types["score"]["accuracy"] * types["score"]["slots"]),
            "top_share_equals_gates_types": abs(
                score["top_share"] - types["score"]["predicted_top_share"]
            )
            < 1e-12,
            "correct_equals_m7_gate": score["correct"] == m7_stats["correct"],
        },
        "type_verdicts": verdicts,
        "R3": {
            "rule": "v2.eval.gates types marks no type COLLAPSED (top share < .90 and "
            "accuracy Wilson lower bound above chance)",
            "pass": all(v == "OK" for v in verdicts.values()),
        },
    }


def score_report(args: argparse.Namespace) -> int:
    from benchmark.score import load_jsonl
    from v2.eval import panels
    from v2.eval.gates import verified as sealed

    refuse_existing(args.output)
    gold_path = panels.path(args.panel_root, "typed-final", "gold")
    if file_sha256(gold_path) != panels.ALL["typed-final"]["gold_sha256"]:
        raise ValueError("typed FINAL gold differs from the frozen panel")
    gold = list(load_jsonl(gold_path).values())
    predictions_path = sealed(args.run, "typed-final")
    predictions = {r["id"]: r for r in read_jsonl(predictions_path)}
    result = score_report_result(gold, predictions)
    stored = args.gates / "types.json"
    if stored.is_file():
        verdicts = {k: v["verdict"] for k, v in load_json(stored)["types"].items()}
        result["types_json"] = {
            "path": str(stored),
            "sha256": file_sha256(stored),
            "agrees": verdicts == result["type_verdicts"],
        }
    result = {
        "schema": "dev2-06b-m8-score-report/1",
        "prereg": f"{PREREG} section 8 (R3)",
        "prereg_commit": PREREG_COMMIT,
        "label": args.label,
        "run": str(args.run),
        "typed_final_predictions_sha256": file_sha256(predictions_path),
        "typed_final_gold_sha256": file_sha256(gold_path),
        **result,
    }
    m7.save(args.output, result)
    s = result["score"]
    print(
        json.dumps(
            {
                "histogram": s["histogram"],
                "top_share": [s["top_share"], s["top_share_wilson95"]],
                "accuracy": [s["accuracy"], s["accuracy_wilson95"]],
                "always_majority": s["always_majority_accuracy"],
                "gain": [s["acc_minus_majority"], s["acc_minus_majority_boot95"]],
                "flags": s["flags"],
                "R3": result["R3"]["pass"],
                "type_verdicts": result["type_verdicts"],
            },
            sort_keys=True,
        )
    )
    return 0


# ----------------------------------------------------------- mlx paired

MLX_PANEL = PANEL_ROOT / "mlx-diag-v1"
MLX_TYPES = ("choice", "noul", "score")
CARD_TYPES = ("choice", "noul")


def mlx_outcomes(
    panel: Path, predictions_path: Path
) -> dict[str, tuple[str, str, int]]:
    """(type, language, correct) per item, exactly as `multilingual_panel.score` counts it."""
    from benchmark.score import evaluate_answer

    gold = {g["id"]: g for g in read_jsonl(panel / "gold.jsonl")}
    prompts = {p["id"]: p for p in read_jsonl(panel / "prompts.jsonl")}
    preds = {r["id"]: r for r in read_jsonl(predictions_path)}
    if set(preds) - set(gold):
        raise ValueError("predictions contain unknown ids")
    out: dict[str, tuple[str, str, int]] = {}
    for item_id, g in gold.items():
        prompt, pred = prompts[item_id], preds.get(item_id)
        question = prompt["questions"]["q"]
        if pred is not None and pred.get("source_input_sha256") != g["input_sha256"]:
            raise ValueError(f"{item_id}: prediction made from different input")
        answer = ((pred or {}).get("answers") or {}).get("q")
        if g["type"] == "choice":
            target = {
                "type": "choice",
                "value": g["value"],
                "label_to_semantic": {k: k for k in question["criteria"]},
                "semantic_value": g["value"],
            }
        else:
            target = {"type": g["type"], "value": g["value"]}
        result = (
            evaluate_answer(question, target, answer)
            if answer is not None
            else {"status": "missing"}
        )
        out[item_id] = (g["type"], g["language"], int(bool(result.get("correct"))))
    return out


def mlx_accuracies(outcomes: dict[str, tuple[str, str, int]]) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[int]] = {}
    for kind, lang, correct in outcomes.values():
        cell = cells.setdefault((kind, lang), [0, 0])
        cell[0] += correct
        cell[1] += 1
    by_type = {}
    for kind in MLX_TYPES:
        langs = {
            lang: {"correct": c, "n": n, "accuracy": c / n}
            for (t, lang), (c, n) in sorted(cells.items())
            if t == kind
        }
        if len({v["n"] for v in langs.values()}) != 1:
            raise ValueError(f"mlx-diag {kind}: languages are not balanced")
        by_type[kind] = {
            "languages": langs,
            "accuracy": sum(v["accuracy"] for v in langs.values()) / len(langs),
        }
    return {
        "by_type": by_type,
        "card_macro": sum(by_type[t]["accuracy"] for t in CARD_TYPES) / len(CARD_TYPES),
        "type_macro": sum(by_type[t]["accuracy"] for t in MLX_TYPES) / len(MLX_TYPES),
    }


def mlx_matches_stored(accuracies: dict[str, Any], stored: dict[str, Any]) -> bool:
    for kind in MLX_TYPES:
        mine = accuracies["by_type"][kind]["languages"]
        theirs = stored["by_type"][kind]["languages"]
        if set(mine) != set(theirs):
            return False
        for lang, cell in mine.items():
            if (cell["correct"], cell["n"]) != (
                theirs[lang]["correct"],
                theirs[lang]["n"],
            ):
                return False
    return abs(accuracies["type_macro"] - stored["type_macro_accuracy"]) < 1e-12


def mlx_bootstrap(
    candidate: dict[str, tuple[str, str, int]],
    released: dict[str, tuple[str, str, int]],
    draws: int = m7.DRAWS,
    seed: int = m7.SEED,
) -> dict[str, Any]:
    """Paired item bootstrap of candidate - released, items resampled within type."""
    from .contrast import percentile

    if set(candidate) != set(released):
        raise ValueError("mlx-diag runs cover different items")
    ids = {t: [k for k, v in candidate.items() if v[0] == t] for t in MLX_TYPES}
    if any(released[k][0] != candidate[k][0] for k in candidate):
        raise ValueError("mlx-diag item types differ between runs")
    left = {t: [candidate[k][2] for k in ids[t]] for t in MLX_TYPES}
    right = {t: [released[k][2] for k in ids[t]] for t in MLX_TYPES}
    rng = random.Random(seed)
    per_type: dict[str, list[float]] = {t: [] for t in MLX_TYPES}
    card, overall = [], []
    for _ in range(draws):
        diff = {}
        for t in MLX_TYPES:
            a, b, n = left[t], right[t], len(left[t])
            total = 0
            for _ in range(n):
                i = rng.randrange(n)
                total += a[i] - b[i]
            diff[t] = total / n
            per_type[t].append(diff[t])
        card.append(sum(diff[t] for t in CARD_TYPES) / len(CARD_TYPES))
        overall.append(sum(diff.values()) / len(MLX_TYPES))

    def ci(values: list[float]) -> list[float]:
        return [percentile(values, 2.5), percentile(values, 97.5)]

    return {
        "draws": draws,
        "seed": seed,
        "items_by_type": {t: len(ids[t]) for t in MLX_TYPES},
        "card_macro_ci95": ci(card),
        "type_macro_ci95": ci(overall),
        "per_type_ci95": {t: ci(v) for t, v in per_type.items()},
    }


def mlx_run(run: Path, panel: Path) -> tuple[dict, dict, dict]:
    path = run / "output" / "mlx-diag.predictions.jsonl"
    stored_path = run / "mlx-diag.score.json"
    stored = load_json(stored_path)
    digest = file_sha256(path)
    if stored["predictions_sha256"] != digest:
        raise ValueError(f"{path} differs from its mlx-diag.score.json")
    if stored["gold_sha256"] != file_sha256(panel / "gold.jsonl"):
        raise ValueError(f"{stored_path} scored another gold file")
    outcomes = mlx_outcomes(panel, path)
    accuracies = mlx_accuracies(outcomes)
    provenance = {
        "run": str(run),
        "predictions_sha256": digest,
        "score_json_sha256": file_sha256(stored_path),
        "reproduces_score_json": mlx_matches_stored(accuracies, stored),
    }
    return outcomes, accuracies, provenance


def mlx_paired_result(
    candidate: dict, released: dict, draws: int = m7.DRAWS
) -> dict[str, Any]:
    acc_c, acc_r = mlx_accuracies(candidate), mlx_accuracies(released)
    boot = mlx_bootstrap(candidate, released, draws)
    return {
        "candidate": acc_c,
        "released": acc_r,
        "delta": {
            "card_macro": acc_c["card_macro"] - acc_r["card_macro"],
            "type_macro": acc_c["type_macro"] - acc_r["type_macro"],
            "per_type": {
                t: acc_c["by_type"][t]["accuracy"] - acc_r["by_type"][t]["accuracy"]
                for t in MLX_TYPES
            },
        },
        "bootstrap": boot,
        "R4": {
            "rule": "Choice + Noul type-macro accuracy, candidate - released: paired "
            "item-bootstrap 95% upper bound >= 0 (items resampled within type)",
            "pass": boot["card_macro_ci95"][1] >= 0,
        },
    }


def mlx_paired(args: argparse.Namespace) -> int:
    refuse_existing(args.output)
    cand, _, cand_prov = mlx_run(args.candidate_run, args.panel)
    rel, _, rel_prov = mlx_run(args.released_run, args.panel)
    if not (cand_prov["reproduces_score_json"] and rel_prov["reproduces_score_json"]):
        raise ValueError("per-item outcomes do not reproduce mlx-diag.score.json")
    result = {
        "schema": "dev2-06b-m8-mlx-paired/1",
        "prereg": f"{PREREG} section 8 (R4)",
        "prereg_commit": PREREG_COMMIT,
        "panel_gold_sha256": file_sha256(args.panel / "gold.jsonl"),
        "runs": {"candidate": cand_prov, "released": rel_prov},
        "scope": "Choice + Noul card-eligible (gated); XNLI-based Score is NC, reported",
        **mlx_paired_result(cand, rel),
    }
    m7.save(args.output, result)
    print(
        json.dumps(
            {
                "card_macro": [
                    result["candidate"]["card_macro"],
                    result["released"]["card_macro"],
                    result["delta"]["card_macro"],
                    result["bootstrap"]["card_macro_ci95"],
                ],
                "type_macro": [
                    result["candidate"]["type_macro"],
                    result["released"]["type_macro"],
                    result["delta"]["type_macro"],
                    result["bootstrap"]["type_macro_ci95"],
                ],
                "R4": result["R4"]["pass"],
            }
        )
    )
    return 0


# --------------------------------------------------------------- xcheck

M6_FORMAL = Path("/data/dev2/runs/06b/m6/formal/m6-mxcx-soup")
FORMAL_PANELS = ("typed-final", "css15", "public231")


def probability_map(answer: dict[str, Any]) -> dict[str, float]:
    if answer.get("type") == "noul":
        return {"true": float(answer["noul"])}
    return {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}


def max_abs_dp(a: dict[str, Any], b: dict[str, Any]) -> float:
    pa, pb = probability_map(a), probability_map(b)
    if set(pa) != set(pb):
        return math.inf
    return max((abs(pa[k] - pb[k]) for k in pa), default=0.0)


def answer_point(answer: dict[str, Any]) -> Any:
    kind = answer.get("type")
    if kind == "choice":
        return answer.get("choice")
    if kind == "noul":
        return m7.noul_point(answer["noul"])
    return m7.predicted([p for _, p in sorted(probability_map(answer).items())])


def xcheck_rows(
    reference: dict[str, dict[str, Any]],
    finalist: dict[str, dict[str, Any]],
    row: list[float],
) -> dict[str, Any]:
    kinds = ("choice", "noul", "score5", "score_other", "invalid")
    cells = {
        k: {"slots": 0, "identical": 0, "same_answer": 0, "max_abs_dp": 0.0}
        for k in kinds
    }
    cells["score5"]["equal_offline_argmax"] = 0
    cells["score5"]["max_abs_dp_vs_offline"] = 0.0
    cells["score5"]["max_abs_dscore_vs_offline"] = 0.0
    cells["score5"]["changed_vs_reference"] = 0
    missing_keys = 0
    for key in sorted(reference):
        ra = reference[key].get("answers") or {}
        fa = (finalist.get(key) or {}).get("answers") or {}
        missing_keys += set(ra) != set(fa)
        for slot, ref in ra.items():
            new = fa.get(slot)
            if not isinstance(ref, dict) or not isinstance(new, dict):
                cell = cells["invalid"]
                cell["slots"] += 1
                cell["identical"] += ref == new
                cell["same_answer"] += ref == new
                continue
            kind = ref.get("type")
            if kind == "score":
                kind = "score5" if probabilities(ref) is not None else "score_other"
            cell = cells[kind if kind in cells else "invalid"]
            cell["slots"] += 1
            cell["identical"] += ref == new
            same = new.get("type") == ref.get("type")
            dp = max_abs_dp(ref, new) if same else math.inf
            cell["max_abs_dp"] = max(cell["max_abs_dp"], dp)
            if kind == "score5":
                expected = corrected_answer(ref, row)
                observed = probabilities(new)
                point = m7.predicted(observed) if observed is not None else None
                cell["same_answer"] += point == m7.predicted(probabilities(ref))
                cell["changed_vs_reference"] += point != m7.predicted(
                    probabilities(ref)
                )
                if observed is None:
                    cell["max_abs_dp_vs_offline"] = math.inf
                    continue
                target = probabilities(expected)
                cell["equal_offline_argmax"] += m7.predicted(target) == point
                cell["max_abs_dp_vs_offline"] = max(
                    cell["max_abs_dp_vs_offline"],
                    max(abs(a - b) for a, b in zip(target, observed)),
                )
                cell["max_abs_dscore_vs_offline"] = max(
                    cell["max_abs_dscore_vs_offline"],
                    abs(float(expected["score"]) - float(new["score"])),
                )
            else:
                cell["same_answer"] += same and answer_point(ref) == answer_point(new)
    checks = {
        "same_ids": set(reference) == set(finalist),
        "same_question_keys": missing_keys == 0,
        "choice_noul_identical": all(
            cells[k]["identical"] == cells[k]["slots"] for k in ("choice", "noul")
        ),
        "score_other_identical": cells["score_other"]["identical"]
        == cells["score_other"]["slots"],
        "score5_offline_argmax_all": cells["score5"]["equal_offline_argmax"]
        == cells["score5"]["slots"],
        "score5_max_abs_dp_le_1e-4": cells["score5"]["max_abs_dp_vs_offline"]
        <= REPLAY_TOLERANCE,
    }
    return {"items": len(reference), "by_kind": cells, "checks": checks}


def xcheck_bindings(
    reference: dict[str, dict[str, Any]], finalist: dict[str, dict[str, Any]], bias_sha
) -> dict[str, bool]:
    return {
        "reference_model_sha256": all(
            r.get("model_sha256") == MODEL_SHA256 for r in reference.values()
        ),
        "finalist_model_sha256": all(
            r.get("model_sha256") == MODEL_SHA256 for r in finalist.values()
        ),
        "finalist_score_bias_sha256": all(
            r.get("score_bias_sha256") == bias_sha for r in finalist.values()
        ),
        "reference_without_score_bias": all(
            "score_bias_sha256" not in r for r in reference.values()
        ),
    }


def xcheck(args: argparse.Namespace) -> int:
    from v2.eval.gates import verified as sealed

    refuse_existing(args.output)
    offsets, _ = load_score_bias(args.score_bias, MODEL_SHA256)
    bias_sha = file_sha256(args.score_bias)
    row = offsets[LEVELS]
    sources = {
        panel: (sealed(args.reference_run, panel), sealed(args.run, panel))
        for panel in FORMAL_PANELS
    }
    sources["mlx-diag"] = tuple(
        run / "output" / "mlx-diag.predictions.jsonl"
        for run in (args.reference_mlx_run, args.mlx_run)
    )
    panels_out = {}
    for panel, (ref_path, new_path) in sources.items():
        reference = {r["id"]: r for r in read_jsonl(ref_path)}
        finalist = {r["id"]: r for r in read_jsonl(new_path)}
        result = xcheck_rows(reference, finalist, row)
        result["bindings"] = xcheck_bindings(reference, finalist, bias_sha)
        result["reference_sha256"] = file_sha256(ref_path)
        result["finalist_sha256"] = file_sha256(new_path)
        panels_out[panel] = result
    result = {
        "schema": "dev2-06b-m8-xcheck/1",
        "prereg": f"{PREREG} section 7 (report only)",
        "prereg_commit": PREREG_COMMIT,
        "runs": {
            "finalist": str(args.run),
            "finalist_mlx": str(args.mlx_run),
            "reference": str(args.reference_run),
            "reference_mlx": str(args.reference_mlx_run),
        },
        "score_bias_sha256": bias_sha,
        "panels": panels_out,
        "all_checks": all(
            all(p["checks"].values()) and all(p["bindings"].values())
            for p in panels_out.values()
        ),
    }
    m7.save(args.output, result)
    print(
        json.dumps(
            {
                panel: {
                    "checks": p["checks"],
                    "bindings": p["bindings"],
                    "slots": {k: v["slots"] for k, v in p["by_kind"].items()},
                    "score5_max_abs_dp_vs_offline": p["by_kind"]["score5"][
                        "max_abs_dp_vs_offline"
                    ],
                    "score5_changed_vs_reference": p["by_kind"]["score5"][
                        "changed_vs_reference"
                    ],
                }
                for panel, p in panels_out.items()
            },
            sort_keys=True,
        )
    )
    return 0


# ------------------------------------------------------------ successor

V3_TIER_FLOOR = 38.3
EXCLUDED_GROUPS = Path(
    "/data/dev2/runs/eval/m5/overlap-effects/final/excluded-groups.json"
)
EXCLUDED_GROUPS_SHA256 = (
    "2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914"
)
M6_MIXTURES = {
    "/data/dev2/runs/06b/m6/data/m6-mx-s1.train.jsonl": "a277bd45a61662b8edbdc3472d0f74f605bc2a4d6e187605afc2bb3d4c998eae",
    "/data/dev2/runs/06b/m6/data/m6-mx-s2.train.jsonl": "5fafa72d6199025e36cda7eb315a1faa341d1f5164ac1d06ccafd69f4e7ed8c0",
    "/data/dev2/runs/06b/m6/data/m6-mx-s3.train.jsonl": "e484c99b1fe4ca5051a2f33ca032edda71f07ff601f6ace859f82c687e31f653",
    "/data/dev2/runs/06b/m6/data/m6-cx-s1.train.jsonl": "ad925c379fe4f1c658087d99e88dcd4ac3cfdab372e999b592eb2f6c4e6186fc",
    "/data/dev2/runs/06b/m6/data/m6-cx-s2.train.jsonl": "420fd988f06c0fe5deb4aed72fcc71f3d2a2af414ccea2597d652679ad664489",
    "/data/dev2/runs/06b/m6/data/m6-cx-s3.train.jsonl": "9131be73edce1e3ba79a9b91adcff7807035e12352545fbbfd3b5e0e44178aa5",
}


def v3_interval(paired_doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta_v3": paired_doc["point"]["delta"]["score"],
        "ci95_v3": [paired_doc["ci95"]["low"], paired_doc["ci95"]["high"]],
        "delta_H": paired_doc["point"]["delta"]["H"],
        "ci95_H": [
            paired_doc["axis_ci95"]["H"]["delta"]["low"],
            paired_doc["axis_ci95"]["H"]["delta"]["high"],
        ],
        "models": paired_doc.get("models"),
    }


def exposure_check(doc: dict[str, Any]) -> dict[str, Any]:
    files = {f["path"]: f["sha256"] for f in doc["files"]}
    checks = {
        "payload_is_rescreen_excluded_groups": doc["payload_sha256"]
        == EXCLUDED_GROUPS_SHA256,
        "six_m6_mixtures": files == M6_MIXTURES,
        "methods_agree": bool(doc["methods_agree"]),
        "no_group_found": not doc["groups"] and not doc["matched_rows"],
    }
    return {
        "files": len(files),
        "rows": sum(f["rows"] for f in doc["files"]),
        "groups": len(doc["groups"]),
        "checks": checks,
        "pass": all(checks.values()),
    }


def successor_result(
    report: dict[str, Any],
    paired: dict[str, dict[str, Any]],
    types: dict[str, Any],
    mlx: dict[str, Any],
    exposure: dict[str, Any],
    public: dict[str, Any],
    d4: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    released = v3_interval(paired["released"])
    kai1 = v3_interval(paired["kai1"])
    gliner = v3_interval(paired["gliner25"])
    verdicts = {k: v["verdict"] for k, v in types["types"].items()}
    r3 = all(v == "OK" for v in verdicts.values())
    v3 = report["v3"]["score"]
    r5 = {
        "v3_vs_kai1_lower_bound_gt_0": kai1["ci95_v3"][0] > 0,
        f"v3_ge_{V3_TIER_FLOOR}": v3 >= V3_TIER_FLOOR,
        "H_vs_gliner25_upper_bound_ge_0": gliner["ci95_H"][1] >= 0,
        "no_type_collapsed": r3,
    }
    exposure_gate = exposure_check(exposure)
    rules = {
        "R1": {
            "rule": "post-key v3 paired delta vs released: 95% CI lower bound > 0",
            **released,
            "pass": released["ci95_v3"][0] > 0,
        },
        "R2": {
            "rule": "delta H vs released: 95% CI upper bound >= 0",
            "delta_H": released["delta_H"],
            "ci95_H": released["ci95_H"],
            "pass": released["ci95_H"][1] >= 0,
        },
        "R3": {
            "rule": "v2.eval.gates types: no type COLLAPSED",
            "verdicts": verdicts,
            "score_no_gain_disclosed": "NO-GAIN" in score["score"]["flags"],
            "score_flags": score["score"]["flags"],
            "pass": r3,
        },
        "R4": {
            "rule": mlx["R4"]["rule"],
            "card_macro_delta": mlx["delta"]["card_macro"],
            "card_macro_ci95": mlx["bootstrap"]["card_macro_ci95"],
            "type_macro_delta_reported": mlx["delta"]["type_macro"],
            "type_macro_ci95_reported": mlx["bootstrap"]["type_macro_ci95"],
            "pass": mlx["bootstrap"]["card_macro_ci95"][1] >= 0,
        },
        "R5": {
            "rule": "paired v3 vs Kai1 lower bound > 0; v3 >= 38.3; delta H vs "
            "GLiNER2.5-Decide upper bound >= 0; no type collapsed",
            "v3": v3,
            "vs_kai1": kai1,
            "vs_gliner25": gliner,
            "checks": r5,
            "pass": all(r5.values()),
        },
        "R6": {
            "rule": "overlap_effects exposure of the six M6 mixtures against the "
            "rescreen's excluded groups finds none",
            **exposure_gate,
        },
        "R7": {
            "rule": "v2.eval.gates public231 vs the current revision is not REGRESSION",
            "delta_items": public["delta"],
            "left_correct": public["left_correct"],
            "right_correct": public["right_correct"],
            "mcnemar_exact_p": public["mcnemar_exact_p"],
            "verdict": public["verdict"],
            "pass": public["verdict"] != "REGRESSION",
        },
    }
    d4_pass = bool(d4["D4"]["pass"])
    return {
        "D4_pass": d4_pass,
        "rules": rules,
        "verdict": d4_pass and all(r["pass"] for r in rules.values()),
    }


def successor(args: argparse.Namespace) -> int:
    refuse_existing(args.output)
    inputs = {
        "report": args.run / "REPORT.json",
        "released": args.gates / "paired-vs-released.json",
        "kai1": args.run / "PAIRED-vs-kai1.json",
        "gliner25": args.gates / "paired-vs-gliner25.json",
        "types": args.gates / "types.json",
        "mlx": args.gates / "mlx-paired.json",
        "public231": args.gates / "public231-vs-released.json",
        "score": args.gates / "score-report.json",
        "exposure": args.exposure,
        "d4": args.d4,
    }
    docs = {k: load_json(v) for k, v in inputs.items()}
    stored = load_json(args.run / "PAIRED-vs-released.json")
    result = successor_result(
        docs["report"],
        {k: docs[k] for k in ("released", "kai1", "gliner25")},
        docs["types"],
        docs["mlx"],
        docs["exposure"],
        docs["public231"],
        docs["d4"],
        docs["score"],
    )
    result = {
        "schema": "dev2-06b-m8-successor/1",
        "prereg": f"{PREREG} section 8",
        "prereg_commit": PREREG_COMMIT,
        "label": args.label,
        "run": str(args.run),
        "seal_sha256": docs["report"].get("seal_sha256"),
        "inputs": {
            k: {"path": str(v), "sha256": file_sha256(v)} for k, v in inputs.items()
        },
        "gates_paired_equals_same_panel_compare": v3_interval(stored)["ci95_v3"]
        == v3_interval(docs["released"])["ci95_v3"],
        **result,
    }
    m7.save(args.output, result)
    print(
        json.dumps(
            {
                "label": args.label,
                "D4": result["D4_pass"],
                **{k: r["pass"] for k, r in result["rules"].items()},
                "verdict": result["verdict"],
            }
        )
    )
    return 0


def choose_result(
    successors: list[dict[str, Any]], between: list[dict[str, Any]]
) -> dict[str, Any]:
    ranked = [s["label"] for s in successors]
    passers = [s["label"] for s in successors if s["verdict"]]
    pairs = {(d["models"]["left"], d["models"]["right"]): d for d in between}
    comparisons = []
    chosen = passers[0] if passers else None
    for later in passers[1:]:
        doc = pairs.get((later, passers[0]))
        if doc is None:
            raise ValueError(f"no paired file {later} vs {passers[0]}")
        interval = v3_interval(doc)
        better = interval["ci95_v3"][0] > 0
        comparisons.append(
            {
                "later": later,
                "first": passers[0],
                **interval,
                "significantly_better": better,
            }
        )
        if better and chosen == passers[0]:
            chosen = later
    return {
        "rule": "the first passer in the section 5 ranking, unless a later passer is "
        "significantly better on v3 (paired delta later - first, 95% lower bound > 0)",
        "ranking": ranked,
        "passers": passers,
        "comparisons": comparisons,
        "successor": chosen,
    }


def choose(args: argparse.Namespace) -> int:
    refuse_existing(args.output)
    successors = [load_json(p) for p in args.successor]
    between = [load_json(p) for p in args.between]
    result = {
        "schema": "dev2-06b-m8-release-choice/1",
        "prereg": f"{PREREG} section 8 (release choice)",
        "prereg_commit": PREREG_COMMIT,
        "inputs": [
            {"path": str(p), "sha256": file_sha256(p)}
            for p in [*args.successor, *args.between]
        ],
        **choose_result(successors, between),
    }
    m7.save(args.output, result)
    print(json.dumps({k: result[k] for k in ("passers", "successor", "comparisons")}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def inputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", required=True, choices=list(CANDIDATES))
        p.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
        p.add_argument("--predictions", type=Path, default=SCORE5T_PREDICTIONS)
        p.add_argument("--aho-dir", type=Path, default=AHO_DIR)
        p.add_argument("--probs", type=Path, default=PROBS)
        p.add_argument("--soup-dir", type=Path, default=SOUP_DIR)
        p.add_argument("--cal-gold", type=Path, default=m7.CAL_GOLD)
        p.add_argument("--output", type=Path, required=True)

    one = commands.add_parser("fit")
    inputs(one)
    one.add_argument("--lam", type=float, required=True, choices=[1.0, 0.5, 0.0])
    one.add_argument("--human-weight", type=float, required=True, choices=[0.0, 1.0])
    one.add_argument("--fit-gold", type=Path, default=FIT_GOLD)
    one.add_argument("--check-gold", type=Path, default=CHECK_GOLD)
    one.add_argument("--score5-panel-rows", type=Path, default=SCORE5_PANEL_ROWS)
    one.add_argument("--report", type=Path, required=True)
    two = commands.add_parser("check")
    inputs(two)
    two.add_argument("--score-bias", type=Path, required=True)
    two.add_argument("--readout", type=Path, default=SCORE5T_READOUT)
    two.add_argument("--dev-predictions", type=Path, default=DEV_PREDICTIONS)
    two.add_argument("--dev-gold", type=Path, default=m7.DEV_GOLD)
    two.add_argument("--score5-run", type=Path, default=SCORE5_RUN)
    three = commands.add_parser("select")
    three.add_argument("--checks", type=Path, nargs="+", required=True)
    three.add_argument("--output", type=Path, required=True)
    four = commands.add_parser("replay")
    four.add_argument("--online", type=Path, required=True)
    four.add_argument("--logged", type=Path, default=SCORE5T_PREDICTIONS)
    four.add_argument("--score-bias", type=Path, required=True)
    four.add_argument("--check", type=Path)
    four.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    four.add_argument("--output", type=Path, required=True)
    five = commands.add_parser("score-report")
    five.add_argument("--run", type=Path, required=True)
    five.add_argument("--gates", type=Path, required=True)
    five.add_argument("--label", required=True)
    five.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    five.add_argument("--output", type=Path, required=True)
    six = commands.add_parser("mlx-paired")
    six.add_argument("--candidate-run", type=Path, required=True)
    six.add_argument("--released-run", type=Path, required=True)
    six.add_argument("--panel", type=Path, default=MLX_PANEL)
    six.add_argument("--output", type=Path, required=True)
    seven = commands.add_parser("xcheck")
    seven.add_argument("--run", type=Path, required=True)
    seven.add_argument("--mlx-run", type=Path, required=True)
    seven.add_argument("--score-bias", type=Path, required=True)
    seven.add_argument("--reference-run", type=Path, default=M6_FORMAL)
    seven.add_argument(
        "--reference-mlx-run", type=Path, default=Path(f"{M6_FORMAL}-mlx")
    )
    seven.add_argument("--output", type=Path, required=True)
    eight = commands.add_parser("successor")
    eight.add_argument("--run", type=Path, required=True)
    eight.add_argument("--gates", type=Path, required=True)
    eight.add_argument("--exposure", type=Path, required=True)
    eight.add_argument("--d4", type=Path, required=True)
    eight.add_argument("--label", required=True)
    eight.add_argument("--output", type=Path, required=True)
    nine = commands.add_parser("choose")
    nine.add_argument("--successor", type=Path, action="append", required=True)
    nine.add_argument("--between", type=Path, action="append", default=[])
    nine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {
        "fit": fit,
        "check": check,
        "select": select,
        "replay": replay,
        "score-report": score_report,
        "mlx-paired": mlx_paired,
        "xcheck": xcheck,
        "successor": successor,
        "choose": choose,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
