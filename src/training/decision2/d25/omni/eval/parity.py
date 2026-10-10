"""Text-only parity between ``VisionCodeReadoutModel`` and Vega's ``CodeReadoutModel``.

Both engines score the same text-only requests one at a time (no padding, so both run identical
tensors) on the same checkpoint. The check passes when ``max |dp| <= tol`` and the argmax agrees on
every request whose top-two margin exceeds ``2 * tol``.

    python -m d25.omni.eval.parity --ckpt CKPT --rows ROWS.jsonl.gz --limit 256 --device cuda --out report.json

``ROWS`` holds training-format rows or evaluation rows (``questions`` dicts); image rows are skipped.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from typing import Any

from d25.omni.common import vision_format
from d25.omni.eval import shards


def text_requests(
    rows: Sequence[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for row in rows:
        if row.get("images"):
            continue
        if "questions" in row:
            requests.extend(
                {**request, "images": []} for _, request in vision_format.requests(row)
            )
        else:
            requests.append(
                {"state": row.get("state"), "question": row["question"], "images": []}
            )
        if limit and len(requests) >= limit:
            return requests[:limit]
    return requests


def compare(
    ours: Callable[[list[dict[str, Any]]], list[list[float]]],
    theirs: Callable[[list[dict[str, Any]]], list[list[float]]],
    requests: Sequence[dict[str, Any]],
    tol: float,
) -> dict[str, Any]:
    worst, total, count, argmax_flips, near_ties = 0.0, 0.0, 0, 0, 0
    for request in requests:
        a = ours([request])[0]
        b = theirs([{"state": request["state"], "question": request["question"]}])[0]
        if len(a) != len(b):
            raise ValueError("engines returned different option counts")
        diffs = [abs(x - y) for x, y in zip(a, b)]
        worst = max(worst, max(diffs))
        total += sum(diffs)
        count += len(diffs)
        best_a = max(range(len(a)), key=a.__getitem__)
        best_b = max(range(len(b)), key=b.__getitem__)
        ranked = sorted(b, reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
        if best_a != best_b:
            if margin > 2 * tol:
                argmax_flips += 1
            else:
                near_ties += 1
    return {
        "requests": len(requests),
        "max_abs_diff": worst,
        "mean_abs_diff": total / max(count, 1),
        "argmax_flips": argmax_flips,
        "near_tie_flips": near_ties,
        "tolerance": tol,
        "pass": worst <= tol and argmax_flips == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        from d25.vega.eval.engine import CodeReadoutModel
    except ImportError as error:
        raise SystemExit(
            f"Vega's engine is not importable yet ({error}); parity cannot run"
        ) from error
    from d25.omni.eval.engine import VisionCodeReadoutModel

    requests = text_requests(list(shards.read_jsonl(args.rows)), args.limit)
    ours = VisionCodeReadoutModel(args.ckpt, device=args.device, max_batch_size=1)
    report = compare(
        ours.predict,
        CodeReadoutModel(args.ckpt, args.device).predict,
        requests,
        args.tol,
    )
    report.update(checkpoint=args.ckpt, rows=args.rows)
    if args.out:
        shards.write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
