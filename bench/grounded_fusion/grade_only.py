"""Grade pre-generated arm answers (from the fusioneval driver) with the DRACO rubric.

The cached-panel paired evaluator (``cmd/fusioneval``) emits ``answers_{arm}.jsonl``:
one final answer per item, already synthesized from the byte-identical cached panel.
This grades those answers with the SAME ``RubricJudge`` + ``summarize`` the live
harness uses (``evaluate.grade_sample`` / ``evaluate.summarize``) — only the answer
SOURCE changes (disk, not a live router call). Output matches ``evaluate.py``:
``samples_{arm}.jsonl`` + ``summary_{arm}.json``, so ``compare_multiarm.py`` consumes
it unchanged.

    python -m bench.grounded_fusion.grade_only --answers results/answers_C.jsonl \
        --arm C --draco-path draco.json --grader-model qwen3:14b
    # add --grade-panel --panel-cache results/panel_cache.jsonl for Level-1 scoring
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .datasets import get_dataset
from .evaluate import _resume_ids, grade_sample, summarize
from .llm_client import ChatClient
from .rubric_judge import RubricJudge
from .runner import FusionResult, PanelEntry


def _fusion_from_record(rec: dict, contents: dict[str, str]) -> FusionResult:
    """Rebuild the FusionResult grade_sample expects from a fusioneval answer row.

    Panel content is only needed for Level-1 (--grade-panel); it is sourced from the
    panel cache (``contents``: model -> content). Without it, panel entries carry the
    grounding metadata only (enough for Level-2 + the contested slice)."""
    panel = []
    for p in rec.get("panel", []):
        model = p.get("model", "")
        panel.append(
            PanelEntry(
                model=model,
                content=contents.get(model, ""),
                grounding_score=p.get("grounding_score"),
                dropped=bool(p.get("dropped", False)),
                flagged=p.get("flagged") or [],
            )
        )
    return FusionResult(
        sample_id=rec["id"],
        final_answer=rec.get("final_answer", "") or "",
        panel=panel,
        grounding_present=bool(rec.get("grounding_present", False)),
        reference_mode=rec.get("reference_mode", ""),
        usage=rec.get("usage", {}) or {},
        error=rec.get("error") or None,
    )


def _load_panel_contents(panel_cache: str | None) -> dict[str, dict[str, str]]:
    """item_id -> {model -> content} from the fusioneval panel cache (for Level-1)."""
    out: dict[str, dict[str, str]] = {}
    if not panel_cache:
        return out
    for line in Path(panel_cache).read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        out[e["item_id"]] = {
            p["model"]: p.get("content", "") for p in e.get("panel", [])
        }
    return out


def grade_arm(args) -> dict:
    if args.grade_panel and not args.panel_cache:
        raise SystemExit(
            "--grade-panel requires --panel-cache (panel content for Level-1)"
        )

    samples = {
        s.id: s
        for s in get_dataset(
            "draco",
            path=args.draco_path,
            domains=args.domains.split(",") if args.domains else None,
            seed=args.seed,
        )
    }
    judge = RubricJudge(
        ChatClient(args.grader_base_url, model=args.grader_model, timeout=args.timeout),
        batch_size=args.grader_batch_size,
    )
    content_by_item = _load_panel_contents(args.panel_cache) if args.grade_panel else {}

    answers = [
        json.loads(line)
        for line in Path(args.answers).read_text().splitlines()
        if line.strip()
    ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / f"samples_{args.arm}.jsonl"
    done = _resume_ids(per_sample_path) if args.resume else set()
    fh = per_sample_path.open("a" if args.resume else "w")
    print(f"[{args.arm}] grading {len(answers)} answers | resume_skip={len(done)}")

    records: list[dict] = []
    for i, rec in enumerate(answers, 1):
        if rec["id"] in done:
            continue
        s = samples.get(rec["id"])
        if s is None:
            print(f"  skip {rec['id']}: no DRACO rubric for this id")
            continue
        fusion = _fusion_from_record(rec, content_by_item.get(rec["id"], {}))
        graded = grade_sample(judge, s, fusion, args.grade_panel)
        records.append(graded)
        fh.write(json.dumps(graded) + "\n")
        fh.flush()
        err = f" ERROR={graded['error']}" if graded.get("error") else ""
        print(
            f"  [{i}/{len(answers)}] {s.domain[:14]:14} norm={graded['final']['normalized']:.3f} "
            f"pen={graded['final']['negative_penalty']:.0f}{err}"
        )
    fh.close()

    if args.resume:
        records = [
            json.loads(line)
            for line in per_sample_path.read_text().splitlines()
            if line.strip()
        ]
    summary = summarize(records)
    (out_dir / f"summary_{args.arm}.json").write_text(
        json.dumps({"arm": args.arm, "summary": summary}, indent=2)
    )
    print(f"\n[{args.arm}] summary -> {out_dir / f'summary_{args.arm}.json'}")
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grade fusioneval arm answers with the DRACO rubric"
    )
    p.add_argument(
        "--answers", required=True, help="answers_{arm}.jsonl from fusioneval"
    )
    p.add_argument("--arm", required=True, help="arm label (A/B/C/D/annotate/filter)")
    p.add_argument("--draco-path", required=True)
    p.add_argument("--domains", default=None, help="comma-separated domain filter")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--grade-panel", action="store_true", help="grade each panel response (Level-1)"
    )
    p.add_argument(
        "--panel-cache",
        default=None,
        help="panel_cache.jsonl (panel content for --grade-panel)",
    )
    p.add_argument("--grader-base-url", default="http://localhost:11435/v1")
    p.add_argument("--grader-model", default="qwen3:14b")
    p.add_argument("--grader-batch-size", type=int, default=8)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", default="bench/grounded_fusion/results")
    return p


def main():
    args = build_parser().parse_args()
    t0 = time.time()
    grade_arm(args)
    print(f"\nwall-clock: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
