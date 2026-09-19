"""Score a prompt-guard checkpoint under the guard metric contract.

`mom_collection_eval.py` evaluates the served collection, and for Guard it
refuses the historical jailbreak split: that split labels toxicity where the
signal answers about instruction attacks, and matching class names do not make
the gold compatible. `constants.MODEL_REGISTRY` drops the dataset entry for the
same reason. So the model comes from the registry and the evaluation set never
does: `--dataset` names the set a run is accountable to, and `--dataset-version`
records which version of it produced the numbers.

Scoring follows the router rather than the tokenizer's truncation. A long
prompt is scanned in overlapping windows and the riskiest one is the score the
router compares with the threshold (`classifier_jailbreak_window.go`), so a
single truncated window measures a path nobody serves. The geometry comes from
the shipped `prompt_guard.window` config, and the report records it.

The report is `guard_metrics.guard_report`: separation, recall at a benign
false-positive budget, the same macro-averaged over length bands, calibration,
and any slice worth reading on its own. `--slice-col` takes the source column,
a language column, and a column marking fixed regression rows, so a number
always says which slice it was measured on.

An operating point is chosen on a separate split, never on the set it is
reported against: `--dev-dataset` picks the threshold that spends `--budget` on
that split, and the report records where the threshold came from.

Usage:
    python jailbreak_guard_eval.py \
        --model llm-semantic-router/Vela-1.0-Encoder-307M-Guard \
        --dataset local:guard-eval-v1.json --dataset-version v1 \
        --label-col attack --slice-col source --slice-col language \
        --dev-dataset local:guard-dev-v1.json --budget 0.01 \
        --bootstrap 2000 --output report.json

    # routing agreement of a candidate against a saved baseline run
    python jailbreak_guard_eval.py --model ./candidate \
        --dataset local:guard-eval-v1.json --label-col attack \
        --baseline report.json --output candidate.json

A local dataset is a JSON list of objects, which is how a set held out from the
Hub is carried. `--positive-labels` names the label values that mean block;
every other value is a negative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import guard_metrics
from constants import MODEL_REGISTRY

DEFAULT_SPLIT = "test"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "config.yaml"
# classifier_jailbreak_window_default.go gives the guard this window when the
# config leaves one out, so a checkout without the entry still scans the way
# the router does.
DEFAULT_WINDOW = (512, 255)


def served_window() -> tuple[int, int]:
    """The window the shipped config gives prompt_guard."""
    window = _guard_window(yaml.safe_load(CONFIG_PATH.read_text()))
    if not window:
        return DEFAULT_WINDOW
    size, overlap = window.get("size"), window.get("overlap")
    return (size or DEFAULT_WINDOW[0], overlap or 0)


def _guard_window(node: Any) -> dict[str, int] | None:
    """Find the guard entry wherever the config keeps it."""
    if isinstance(node, dict):
        if node.get("model_ref") == "prompt_guard" and isinstance(
            node.get("window"), dict
        ):
            return node["window"]
        children: Any = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _guard_window(child)
        if found:
            return found
    return None


def parse_args() -> argparse.Namespace:
    served = MODEL_REGISTRY["jailbreak"]["id"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=served)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Hugging Face id, or local:<path to a JSON list>",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="version of the evaluation set these numbers belong to",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument(
        "--slice-col",
        action="append",
        default=[],
        help="column to report on its own; repeatable",
    )
    parser.add_argument(
        "--positive-labels",
        default="1,unsafe,jailbreak,attack",
        help="comma-separated label values that mean block",
    )
    parser.add_argument(
        "--positive-index",
        type=int,
        default=1,
        help="index of the block class in the model's head",
    )
    parser.add_argument(
        "--threshold", type=float, default=guard_metrics.DEFAULT_THRESHOLD
    )
    parser.add_argument(
        "--dev-dataset",
        default=None,
        help="split the operating point is chosen on, never the "
        "one it is reported against",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.01,
        help="benign false-positive budget the threshold spends on the dev split",
    )
    size, overlap = served_window()
    parser.add_argument(
        "--window-size",
        type=int,
        default=size,
        help="tokens per window, special tokens included",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=overlap,
        help="tokens two neighbouring windows share",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="bootstrap resamples for intervals, 0 to skip",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="an earlier report to compute routing agreement against",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(dataset: str, split: str) -> list[dict[str, Any]]:
    if dataset.startswith("local:"):
        return json.loads(Path(dataset.split(":", 1)[1]).read_text())
    return list(load_dataset(dataset, split=split))


def read_columns(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[str], list[int], dict[str, list[Any]]]:
    positive = {value.strip().lower() for value in args.positive_labels.split(",")}
    texts: list[str] = []
    labels: list[int] = []
    slices: dict[str, list[Any]] = {name: [] for name in args.slice_col}
    for row in rows:
        text = (row.get(args.text_col) or "").strip()
        annotation = row.get(args.label_col)
        # A row the source does not annotate for this contract is left out
        # rather than assumed benign.
        if not text or annotation is None:
            continue
        texts.append(text)
        labels.append(1 if str(annotation).lower() in positive else 0)
        for name in args.slice_col:
            slices[name].append(row.get(name, "unknown"))
    return texts, labels, slices


def score(
    model_id: str,
    texts: list[str],
    window: tuple[int, int],
    batch_size: int,
    positive_index: int,
) -> list[float]:
    """The riskiest window per text, which is the score the router keeps."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    size, overlap = window
    empty = tokenizer("", add_special_tokens=True)["input_ids"]
    prefix, suffix = empty[:1], empty[1:]
    padding = tokenizer.pad_token_id or 0
    pending: list[list[int]] = []
    owners: list[int] = []
    for index, text in enumerate(texts):
        content = tokenizer(text, add_special_tokens=False)["input_ids"]
        for first, last in guard_metrics.token_windows(
            len(content), size, overlap, len(empty)
        ):
            pending.append(prefix + content[first:last] + suffix)
            owners.append(index)
    scores = [0.0] * len(texts)
    with torch.no_grad():
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            longest = max(len(ids) for ids in batch)
            encoded = torch.tensor(
                [ids + [padding] * (longest - len(ids)) for ids in batch]
            )
            mask = torch.tensor(
                [[1] * len(ids) + [0] * (longest - len(ids)) for ids in batch]
            )
            probabilities = torch.softmax(
                model(input_ids=encoded, attention_mask=mask).logits, dim=1
            )
            for offset, probability in enumerate(
                probabilities[:, positive_index].tolist()
            ):
                owner = owners[start + offset]
                scores[owner] = max(scores[owner], probability)
    return scores


def choose_threshold(args: argparse.Namespace) -> tuple[float, str]:
    """Pick the operating point on the dev split, at the stated budget."""
    if not args.dev_dataset:
        return args.threshold, "--threshold"
    texts, labels, _ = read_columns(load_rows(args.dev_dataset, args.split), args)
    scores = score(
        args.model,
        texts,
        (args.window_size, args.window_overlap),
        args.batch_size,
        args.positive_index,
    )
    reached = guard_metrics.recall_at_fpr(
        scores, [label == 1 for label in labels], args.budget
    )
    if not reached:
        raise SystemExit(
            f"no threshold on {args.dev_dataset} stays inside a "
            f"{args.budget:g} benign false-positive budget"
        )
    return reached["threshold"], f"{args.dev_dataset} at {args.budget:g} FPR"


def main() -> None:
    args = parse_args()
    texts, labels, slices = read_columns(load_rows(args.dataset, args.split), args)
    print(f"{args.dataset}:{args.split} -> {len(texts)} rows, {sum(labels)} positives")

    threshold, source = choose_threshold(args)
    scores = score(
        args.model,
        texts,
        (args.window_size, args.window_overlap),
        args.batch_size,
        args.positive_index,
    )
    report = guard_metrics.guard_report(
        labels,
        scores,
        words=[len(text.split()) for text in texts],
        slices=slices or None,
        threshold=threshold,
        bootstrap_resamples=args.bootstrap,
    )
    report["model"] = args.model
    report["dataset"] = f"{args.dataset}:{args.split}"
    report["dataset_version"] = args.dataset_version
    report["threshold_source"] = source
    report["window"] = {"size": args.window_size, "overlap": args.window_overlap}
    report["scores"] = scores

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        report["baseline"] = baseline.get("model")
        report["routing_agreement"] = (
            guard_metrics.routing_agreement(
                labels, baseline["scores"], scores, threshold
            )
            if len(baseline.get("scores", [])) == len(scores)
            else {"skipped": "the baseline report scores a different row count"}
        )

    Path(args.output).write_text(json.dumps(report, indent=2))
    pooled = report["pooled"]
    budget = guard_metrics.DEFAULT_BUDGETS[-1]
    key = f"recall_at_{budget:g}_fpr"
    reached = pooled.get(key) or {}
    print(
        f"  auc {_number(pooled.get('auc'))}"
        f"  recall at {budget:g} FPR {_number(reached.get('recall'))}"
        f"  band macro {_number(report.get(f'band_macro_{key}'))}"
        f"  FPR at {threshold:.4f} {_number(pooled['false_positive_rate'])}"
    )
    print(f"[saved] {args.output}")


def _number(value: float | None) -> str:
    """A slice can leave a number undefined, and that reads better than a crash."""
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
