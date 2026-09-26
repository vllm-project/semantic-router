import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .evaluate_detectors import RESULTS_DIR, build_detector, prf, score_gold_spans
from .trajectories import DEFAULT_FIXTURE, Trajectory, load_trajectories, step_checks

# Optional like lettucedetect: only --context-window needs a tokenizer.
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class SpanDetector(Protocol):
    def predict(
        self, *, context: list[str], question: str, answer: str, output_format: str
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ContextWindow:
    tokens: int
    count_tokens: Callable[[str], int]

    def fit(self, *, context: str, question: str, answer: str) -> str:
        """The longest start of the context that fits with the question and answer."""
        # Mirrors the router's ModernBERT detector, which appends this tail to
        # the context and cuts the context from the end to fit, in
        # candle-binding/src/ffi/instances/tasks.rs.
        tail = f" Question: {question} [SEP] {answer}"
        low, high = 0, len(context)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count_tokens(context[:middle] + tail) <= self.tokens:
                low = middle
            else:
                high = middle - 1
        return context[:low]


def evaluate_steps(
    detector: SpanDetector,
    trajectories: list[Trajectory],
    context_window: ContextWindow | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the detector on every assistant step and score it against gold spans."""
    steps = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    truncated = dict.fromkeys(steps, 0)
    char_tp = char_fp = char_fn = 0
    latencies: list[float] = []
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for check in step_checks(trajectory):
            context = check.context
            if context_window is not None:
                context = context_window.fit(
                    context=context, question=check.question, answer=check.answer
                )
            started = time.perf_counter()
            spans = detector.predict(
                context=[context],
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
            if len(context) < len(check.context):
                truncated[outcome] += 1
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
                    "context_chars": len(check.context),
                    "context_chars_seen": len(context),
                    "truncated": len(context) < len(check.context),
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
        "first_false_step": summarize_first_false_steps(first_false_steps(rows)),
        "context_window_tokens": context_window.tokens if context_window else None,
        "truncated_steps": truncated,
        "latency_ms_avg": sum(latencies) / len(latencies) if latencies else None,
    }
    return metrics, rows


def first_false_steps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Where each trajectory first goes wrong and when the detector catches it."""
    by_trajectory: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trajectory.setdefault(row["trajectory"], []).append(row)
    results: list[dict[str, Any]] = []
    for trajectory, checked in by_trajectory.items():
        unsupported = [index for index, row in enumerate(checked) if row["gold_spans"]]
        if not unsupported:
            continue
        first = unsupported[0]
        caught = next((index for index in unsupported if checked[index]["spans"]), None)
        results.append(
            {
                "trajectory": trajectory,
                "first_false_step": checked[first]["step"],
                "first_caught_step": (
                    None if caught is None else checked[caught]["step"]
                ),
                # Counted in assistant steps, the only steps a detector checks.
                "steps_late": None if caught is None else caught - first,
                "false_alarm_before": any(row["spans"] for row in checked[:first]),
            }
        )
    return results


def summarize_first_false_steps(results: list[dict[str, Any]]) -> dict[str, Any]:
    delays: list[int] = [
        result["steps_late"] for result in results if result["steps_late"] is not None
    ]
    return {
        "trajectories": len(results),
        "caught_at_first": delays.count(0),
        "caught_late": len(delays) - delays.count(0),
        "missed": len(results) - len(delays),
        "mean_steps_late": sum(delays) / len(delays) if delays else None,
        "false_alarm_before": sum(result["false_alarm_before"] for result in results),
    }


def format_row(row: dict[str, Any]) -> str:
    gold = ", ".join(repr(span["text"]) for span in row["gold_spans"]) or "-"
    detected = ", ".join(repr(span["text"]) for span in row["spans"]) or "-"
    seen = f"{row['context_chars_seen']}/{row['context_chars']} chars"
    return (
        f"{row['trajectory']} step {row['step']}: {row['outcome']}"
        f"  gold={gold}  detected={detected}  context={seen}"
        + ("  truncated" if row["truncated"] else "")
    )


def format_first_false_step(result: dict[str, Any]) -> str:
    where = f"{result['trajectory']}: first false step {result['first_false_step']}"
    if result["steps_late"] is None:
        verdict = "missed"
    else:
        verdict = (
            f"caught at step {result['first_caught_step']}"
            f" (steps late: {result['steps_late']})"
        )
    alarm = ", after a false alarm" if result["false_alarm_before"] else ""
    return f"{where}, {verdict}{alarm}"


def token_counter(name: str) -> Callable[[str], int]:
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers not installed. Run: pip install lettucedetect")
    tokenizer = AutoTokenizer.from_pretrained(name)
    return lambda text: len(tokenizer(text)["input_ids"])


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
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="keep only the start of each step's context that fits this many "
        "tokens with the question and answer, as the router's ModernBERT "
        "detector does at 512",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer that counts --context-window tokens "
        "(default: the transformer detector's model)",
    )
    args = parser.parse_args(argv)

    context_window: ContextWindow | None = None
    if args.context_window is not None:
        method, _, model = args.detector.partition(":")
        tokenizer_name = args.tokenizer or (
            model.partition(":")[0] if method == "transformer" else None
        )
        if tokenizer_name is None:
            parser.error("--context-window needs --tokenizer for an llm detector")
        context_window = ContextWindow(
            tokens=args.context_window, count_tokens=token_counter(tokenizer_name)
        )

    trajectories = load_trajectories(args.trajectories)
    metrics, rows = evaluate_steps(
        detector=build_detector(args.detector, args.base_url),
        trajectories=trajectories,
        context_window=context_window,
    )
    for row in rows:
        print(format_row(row))
    first_false = first_false_steps(rows)
    for result in first_false:
        print(format_first_false_step(result))
    print(json.dumps(metrics, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    safe = args.detector.replace("/", "_").replace(":", "-")
    out = RESULTS_DIR / f"trajectories_{args.trajectories.stem}_{safe}.json"
    out.write_text(
        json.dumps(
            {"metrics": metrics, "first_false_steps": first_false, "rows": rows},
            indent=2,
        )
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
