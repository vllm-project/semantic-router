"""Apply or undo per-type temperatures on a sealed prediction file (stdlib; answers unchanged).

For a calibration-only package change: predictions scored with CAL temperatures
become the T = 1 predictions of the same logits (``--undo``: softmax(log q * T)),
or raw predictions become calibrated ones. Every row must carry the expected
``calibration_sha256`` when undoing; the output drops it. The receipt records
input and output digests and confirms that no answer changed. Adopt the output
with ``v2.eval.same_panel adopt`` (stating this derivation as the reason).

    python3 -m v2.release.retemper_predictions --predictions IN.jsonl --prompts PROMPTS.jsonl \
        --calibration calibration.json [--undo] --output OUT.jsonl --receipt receipt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from v2.release.dev_calibration import _jsonl, answer_changes, rescale
from v2.release.layout import canonical, sha_file, write_json
from v2.release.temperature_parity import question_kinds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--undo", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    temperatures = json.loads(args.calibration.read_text(encoding="utf-8"))[
        "temperature_by_type"
    ]
    rows = _jsonl(args.predictions)
    calibration = sha_file(args.calibration)
    if args.undo and any(row.get("calibration_sha256") != calibration for row in rows):
        raise ValueError("rows were not scored with this calibration file")
    out = rescale(rows, question_kinds(_jsonl(args.prompts)), temperatures, args.undo)
    changes = answer_changes(rows, out)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in out:
            stream.write(canonical(row) + "\n")
    write_json(
        args.receipt,
        {
            "schema": "dev2-release-retemper/1",
            "mode": "undo" if args.undo else "apply",
            "input_sha256": sha_file(args.predictions),
            "output_sha256": sha_file(args.output),
            "prompts_sha256": sha_file(args.prompts),
            "calibration_sha256": calibration,
            "temperature_by_type": temperatures,
            "rows": len(rows),
            "slots": sum(len(r["answers"]) for r in rows),
            "answer_changes": changes,
            "module_sha256": sha_file(Path(__file__)),
            "passed": changes == 0,
        },
    )
    print(json.dumps({"rows": len(rows), "answer_changes": changes}))
    sys.exit(0 if changes == 0 else 1)


if __name__ == "__main__":
    main()
