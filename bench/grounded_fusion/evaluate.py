"""Grounded-fusion benchmark: run ONE arm against the router, grade with DRACO.

For each DRACO problem this:
  1. drives the router fusion looper (runner.run_fusion) -> final answer + trace,
  2. grades the final answer with the DRACO rubric (Level-2),
  3. (optional) grades each panel response and correlates grounding score with
     panel quality (Level-1 intrinsic scorer validity).

Run two arms (grounding on / off, via two router configs) then diff with
``compare.py``. Results are written per-sample so a killed run can be resumed.

Example:
    .venv-bench/bin/python -m bench.grounded_fusion.evaluate \
        --endpoint http://localhost:8801 \
        --draco-path ~/Downloads/draco.json \
        --arm on --max-samples 8 --domains Medicine,Law \
        --grader-model qwen3:14b --grade-panel
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .datasets import get_dataset
from .llm_client import ChatClient
from .metrics import discard_quality, spearman
from .rubric_judge import RubricJudge
from .runner import FusionResult, run_fusion


def grade_sample(
    judge: RubricJudge, sample, fusion: FusionResult, grade_panel: bool
) -> dict:
    """Grade one sample's final answer (+ optionally its panel) with the rubric."""
    final = judge.score(sample.problem, fusion.final_answer, sample.rubric)
    rec = {
        "id": sample.id,
        "domain": sample.domain,
        "error": fusion.error,
        "latency_ms": fusion.latency_ms,
        "usage": fusion.usage,
        "grounding_present": fusion.grounding_present,
        "reference_mode": fusion.reference_mode,
        "final": {
            "total": final.total,
            "normalized": final.normalized,
            "negative_penalty": final.negative_penalty,
            "n_negative_triggered": final.n_negative_triggered,
            "per_section": final.per_section,
        },
    }
    # Always record the panel's grounding metadata (score + drop decision) — it
    # comes from the fusion trace for free and is what compare.py uses to detect
    # the "contested" slice. Rubric grading of each panel response (Level-1) is
    # the expensive part and stays gated behind --grade-panel.
    if fusion.panel:
        if grade_panel:
            rec["panel"] = _grade_panel(judge, sample, fusion)
        else:
            rec["panel"] = [
                {
                    "model": p.model,
                    "grounding_score": p.grounding_score,
                    "dropped": p.dropped,
                    "flagged": p.flagged,
                }
                for p in fusion.panel
            ]
    return rec


def _grade_panel(judge: RubricJudge, sample, fusion: FusionResult) -> list[dict]:
    out = []
    for p in fusion.panel:
        ps = judge.score(sample.problem, p.content, sample.rubric)
        out.append(
            {
                "model": p.model,
                "grounding_score": p.grounding_score,
                "dropped": p.dropped,
                "rubric_total": ps.total,
                "rubric_normalized": ps.normalized,
                "negative_penalty": ps.negative_penalty,
            }
        )
    return out


def summarize(records: list[dict]) -> dict:
    """Aggregate Level-2 (answer quality) and Level-1 (scorer validity)."""
    ok = [r for r in records if not r.get("error")]
    finals = [r["final"] for r in ok]
    summary = {
        "n": len(records),
        "n_ok": len(ok),
        "n_grounding_present": sum(1 for r in ok if r.get("grounding_present")),
        "level2": _level2(finals, ok),
        "level1": _level1(ok),
    }
    return summary


def _level2(finals, ok) -> dict:
    if not finals:
        return {}

    def mean(key):
        return sum(f[key] for f in finals) / len(finals)

    by_dom: dict = {}
    for r in ok:
        by_dom.setdefault(r["domain"], []).append(r["final"]["normalized"])
    return {
        "mean_normalized": mean("normalized"),
        "mean_total": mean("total"),
        "mean_negative_penalty": mean("negative_penalty"),
        "frac_with_any_penalty": sum(1 for f in finals if f["n_negative_triggered"] > 0)
        / len(finals),
        "per_domain_mean_normalized": {d: sum(v) / len(v) for d, v in by_dom.items()},
    }


def _level1(ok) -> dict:
    """Pool all graded panel responses: correlate grounding score with quality."""
    gscores, qscores, dropped = [], [], []
    for r in ok:
        for p in r.get("panel", []):
            # Only panels that were rubric-graded (--grade-panel) contribute to
            # Level-1; ungraded panels carry drop flags only.
            if p.get("grounding_score") is None or "rubric_normalized" not in p:
                continue
            gscores.append(float(p["grounding_score"]))
            qscores.append(float(p["rubric_normalized"]))
            dropped.append(bool(p["dropped"]))
    if not gscores:
        return {"n_panel_graded": 0}
    res = {
        "n_panel_graded": len(gscores),
        "spearman_score_vs_quality": spearman(gscores, qscores),
    }
    res.update(discard_quality(gscores, qscores, dropped))
    return res


def run_arm(args) -> dict:
    samples = get_dataset(
        "draco",
        path=args.draco_path,
        max_samples=args.max_samples,
        domains=args.domains.split(",") if args.domains else None,
        seed=args.seed,
    )
    judge = RubricJudge(
        ChatClient(args.grader_base_url, model=args.grader_model, timeout=args.timeout),
        batch_size=args.grader_batch_size,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / f"samples_{args.arm}.jsonl"

    records: list[dict] = []
    done_ids = _resume_ids(per_sample_path) if args.resume else set()
    fh = per_sample_path.open("a" if args.resume else "w")
    print(
        f"[{args.arm}] {len(samples)} samples | grader={args.grader_model} | resume_skip={len(done_ids)}"
    )
    for i, s in enumerate(samples, 1):
        if s.id in done_ids:
            continue
        fusion = run_fusion(
            args.endpoint,
            s.id,
            s.problem,
            model=args.fusion_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        if (
            args.assert_grounding
            and args.arm == "on"
            and not fusion.grounding_present
            and not fusion.error
        ):
            raise SystemExit(
                f"[FATAL] sample {s.id}: grounding NOT present in trace -- the NLI backend "
                "is not wired (enable hallucination_mitigation.nli_model). Aborting so you "
                "don't measure plain fusion twice."
            )
        rec = grade_sample(judge, s, fusion, args.grade_panel)
        records.append(rec)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        flag = "" if fusion.grounding_present else " (no-grounding)"
        err = f" ERROR={fusion.error}" if fusion.error else ""
        print(
            f"  [{i}/{len(samples)}] {s.domain[:14]:14} norm={rec['final']['normalized']:.3f} "
            f"pen={rec['final']['negative_penalty']:.0f}{flag}{err}"
        )
    fh.close()

    if args.resume:
        records = [
            json.loads(line)
            for line in per_sample_path.read_text().splitlines()
            if line.strip()
        ]
    summary = summarize(records)
    summary_path = out_dir / f"summary_{args.arm}.json"
    summary_path.write_text(
        json.dumps({"args": _sanitized_args(args), "summary": summary}, indent=2)
    )
    print(f"\n[{args.arm}] summary -> {summary_path}")
    print(json.dumps(summary, indent=2))
    return summary


def _sanitized_args(args) -> dict:
    """Strip personal filesystem paths from recorded args so no absolute home
    paths or local identifiers land in shared/committed result JSON. The dataset
    path is reduced to its basename; any other home-rooted path is rewritten to
    a ``~``-relative form."""
    home = str(Path.home())
    out = {}
    for k, v in vars(args).items():
        masked = v
        if isinstance(v, str):
            if k == "draco_path":
                masked = Path(v).name
            elif v.startswith(home):
                masked = "~" + v[len(home) :]
        out[k] = masked
    return out


def _resume_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {
        json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grounded-fusion DRACO benchmark (one arm)")
    p.add_argument(
        "--endpoint", default="http://localhost:8801", help="router OpenAI endpoint"
    )
    p.add_argument("--draco-path", required=True)
    p.add_argument(
        "--arm",
        choices=["on", "off"],
        required=True,
        help="label (match the router config grounding state)",
    )
    p.add_argument(
        "--fusion-model",
        default="vllm-sr/fusion",
        help="model name that triggers the fusion looper",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--domains",
        default=None,
        help="comma-separated domain filter (e.g. Medicine,Law)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument(
        "--grade-panel", action="store_true", help="grade each panel response (Level-1)"
    )
    p.add_argument("--grader-base-url", default="http://localhost:11435/v1")
    p.add_argument("--grader-model", default="qwen3:14b")
    p.add_argument("--grader-batch-size", type=int, default=8)
    p.add_argument(
        "--assert-grounding",
        action="store_true",
        help="abort if arm=on lacks a grounding trace",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default="bench/grounded_fusion/results")
    return p


def main():
    args = build_parser().parse_args()
    t0 = time.time()
    run_arm(args)
    print(f"\nwall-clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
