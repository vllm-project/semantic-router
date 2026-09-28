import argparse
import json
import time
from pathlib import Path
from typing import Any, Protocol

from .evaluate_detectors import RESULTS_DIR, build_detector, prf, score_gold_spans
from .trajectories import DEFAULT_FIXTURE, Trajectory, load_trajectories, step_checks


class SpanDetector(Protocol):
    def predict(
        self, *, context: list[str], question: str, answer: str, output_format: str
    ) -> list[dict[str, Any]]: ...


def evaluate_steps(
    detector: SpanDetector, trajectories: list[Trajectory]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the detector on every assistant step and score it against gold spans."""
    steps = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    char_tp = char_fp = char_fn = 0
    latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for check in step_checks(trajectory):
            started = time.perf_counter()
            spans = detector.predict(
                context=[check.context],
                question=check.question,
                answer=check.answer,
                output_format="spans",
            )
            latencies.append((time.perf_counter() - started) * 1000)

            flagged = bool(spans)
            if check.gold_spans:
                outcome = "tp" if flagged else "fn"
            else:
                outcome = "fp" if flagged else "tn"
            steps[outcome] += 1
            # Every step is labeled, so detections on a supported step count
            # against character precision instead of being skipped.
            ctp, cfp, cfn, *_ = score_gold_spans(
                spans, list(check.gold_spans), check.answer
            )
            char_tp, char_fp, char_fn = char_tp + ctp, char_fp + cfp, char_fn + cfn
            rows.append(
                {
                    "trajectory": check.trajectory_id,
                    "step": check.step,
                    "outcome": outcome,
                    "gold_spans": list(check.gold_spans),
                    "spans": spans,
                }
            )

    step_p, step_r, step_f1 = prf(steps["tp"], steps["fp"], steps["fn"])
    char_p, char_r, char_f1 = prf(char_tp, char_fp, char_fn)
    metrics = {
        "trajectories": len(trajectories),
        "steps": len(rows),
        "step_level": {**steps, "precision": step_p, "recall": step_r, "f1": step_f1},
        "char_level": {"precision": char_p, "recall": char_r, "f1": char_f1},
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else None,
    }
    return metrics, rows


def format_row(row: dict[str, Any]) -> str:
    gold = ", ".join(repr(span["text"]) for span in row["gold_spans"]) or "-"
    detected = ", ".join(repr(span["text"]) for span in row["spans"]) or "-"
    return (
        f"{row['trajectory']} step {row['step']}: {row['outcome']}"
        f"  gold={gold}  detected={detected}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a hallucination detector on every assistant step of "
        "agent trajectories."
    )
    parser.add_argument(
        "--detector",
        required=True,
        help="llm:<model> or transformer:<path>[:<taxonomy_head>]",
    )
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint for llm detectors (e.g. vLLM)",
    )
    args = parser.parse_args(argv)

    trajectories = load_trajectories(args.trajectories)
    metrics, rows = evaluate_steps(
        detector=build_detector(args.detector, args.base_url),
        trajectories=trajectories,
    )
    for row in rows:
        print(format_row(row))
    print(json.dumps(metrics, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    safe = args.detector.replace("/", "_").replace(":", "-")
    out = RESULTS_DIR / f"trajectories_{args.trajectories.stem}_{safe}.json"
    out.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
