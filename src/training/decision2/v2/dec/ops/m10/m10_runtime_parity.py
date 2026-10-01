"""Decoder M10: release-runtime parity of a label-token checkpoint (native `label_token` path vs `infer_dec`).

The runtime is staged exactly as the package builder stages it (``v2.release.build.vendor_runtime`` with the
checkpoint's identity, so ``label_token.py`` is vendored with package-relative imports), imported as the package
``decision2`` from the stage, and loaded through ``QwenDecision.load`` with a minimal manifest (full profile, the
checkpoint's identity, no calibration). Every prompt item is answered with ``system_one`` and compared with the stored
``infer_dec`` predictions of the same checkpoint: decisions (Choice key, Noul side of 0.5, Score argmax level) and the
largest probability difference. PASS = 0 differing decisions.

usage: m10_runtime_parity.py --checkpoint CK --prompts P --predictions PRED --output OUT [--max-items N]
       [--max-length 16384]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

from v2.dec.label_token import label_fingerprint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m10_compare import decision, probabilities  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()
    from v2.release import build

    identity = label_fingerprint(args.checkpoint, None)
    with tempfile.TemporaryDirectory() as scratch:
        stage = Path(scratch)
        records = build.vendor_runtime(
            {"profile": "qwen-full"},
            stage,
            {"head_variant": "shared", "dec_residual": False, "readout": "label_token"},
        )
        sys.path.insert(0, str(stage))
        qwen = importlib.import_module("decision2.qwen")
        manifest = {
            "profile": "qwen-full",
            "identity": {"model_sha256": identity["model_sha256"]},
            "max_input_tokens": args.max_length,
            "calibration": None,
        }
        backend = qwen.QwenDecision.load(
            args.checkpoint, manifest, device="cuda:0", base_path=None, threads=None
        )
        stored = {
            json.loads(line)["id"]: json.loads(line)["answers"]
            for line in args.predictions.open(encoding="utf-8")
        }
        items = [json.loads(line) for line in args.prompts.open(encoding="utf-8")]
        if args.max_items:
            items = items[: args.max_items]
        questions = differ = 0
        drift = 0.0
        for item in items:
            answers, _ = backend.system_one(item["state"], item["questions"])
            for qid, answer in answers.items():
                other = stored[item["id"]][qid]
                questions += 1
                differ += decision(answer) != decision(other)
                p, q = probabilities(answer), probabilities(other)
                if p is not None and q is not None:
                    drift = max(drift, max(abs(p[k] - q[k]) for k in p))
        vendored = sorted(k for k in records if "label_token" in k)
    report = {
        "checkpoint": str(args.checkpoint),
        "model_sha256": identity["model_sha256"],
        "prompts": str(args.prompts),
        "items": len(items),
        "questions": questions,
        "decisions_differ": differ,
        "max_probability_drift": drift,
        "vendored": vendored,
        "status": "PASS" if differ == 0 else "FAIL",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
