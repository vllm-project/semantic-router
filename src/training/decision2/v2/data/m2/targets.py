"""Convert native teacher output on RP-v2 prompts into compact target files.

    python3 -m v2.data.m2.targets --rows rp-v2.rows.jsonl --teacher-output lux.wave1.jsonl \\
        --teacher lux --model-id llm-semantic-router/Decision-1.0-Lux-9B --revision bd45a30a... \\
        --out lux.wave1.targets.jsonl --report lux.wave1.report.json

Every receipt is checked by ``v2.data.replay_targets.convert`` (model identity,
attested revision, validated runtime, prompt digest bound to the training row).
Rows the teacher could not answer natively get no target and are counted. The
output keeps ``{id, input_sha256, teacher_probs}`` sorted by id; join by id to
``m2/arms/<ARM>/train.jsonl`` (or v1 arm files) to build replay rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from training.model.data import canonical
from v2.data.build_a0_variants import native_prompt
from v2.data.m2.common import read_jsonl
from v2.data.replay_targets import convert


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    receipts = list(read_jsonl(args.teacher_output))
    wanted = {r["id"] for r in receipts}
    rows = [row for row in read_jsonl(args.rows) if row["id"] in wanted]
    if len(rows) != len(wanted):
        raise ValueError("teacher output names ids outside the RP-v2 rows")
    replay, report = convert(
        rows,
        [native_prompt(row) for row in rows],
        receipts,
        teacher=args.teacher,
        model_id=args.model_id,
        revision=args.revision,
    )
    data = "".join(
        canonical(
            {
                "id": r["id"],
                "input_sha256": r["input_sha256"],
                "teacher_probs": r["teacher_probs"],
            }
        )
        + "\n"
        for r in sorted(replay, key=lambda r: r["id"])
    ).encode("utf-8")
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    report.update(
        rows=len(replay),
        prompts=len(rows),
        content_sha256=hashlib.sha256(data).hexdigest(),
        teacher_output_sha256=hashlib.sha256(
            args.teacher_output.read_bytes()
        ).hexdigest(),
    )
    fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
    print(json.dumps({k: report[k] for k in ("rows", "prompts", "content_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
