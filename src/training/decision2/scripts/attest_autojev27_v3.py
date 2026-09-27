"""Build a private native-package attestation for the fixed AutoJev 27B peer.

The independently recorded loaded count comes from a successful native GPU
smoke, which itself checks its loaded tensors against the full released weight
inventory. This command only audits those inputs and writes private receipts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.autojev27 import (
    BACKEND,
    MODEL_ID,
    MODEL_REVISION,
    verify_release,
)
from scripts.baseline_attestation_v3 import build_attestation
from scripts.plan_final_eval import NativeModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source_root",
        "model_root",
        "external_root",
        "native_count_receipt",
        "receipt_output",
        "attestation_output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    model = NativeModel(
        "autojev27b",
        "AutoJev 27B",
        "open",
        "27B",
        MODEL_ID,
        MODEL_REVISION,
        BACKEND,
        "inference.autojev27",
        "autojev-27b",
        "autojev-source",
    )
    release = verify_release(
        args.model_root / model.model_dir,
        args.external_root / model.source_dir,
        model.revision,
    )
    count_receipt = json.loads(args.native_count_receipt.read_text(encoding="utf-8"))
    if (
        count_receipt.get("model_id") != model.model_id
        or count_receipt.get("model_revision") != model.revision
        or count_receipt.get("native_model_sha256") != release["native_model_sha256"]
        or count_receipt.get("runtime_source_sha256")
        != release["runtime_source_sha256"]
        or count_receipt.get("loaded_parameters") != release["loaded_parameters"]
        or count_receipt.get("input_items") != 32
    ):
        raise ValueError("AutoJev native GPU count receipt differs from pinned release")
    build_attestation(
        model,
        source_root=args.source_root,
        model_root=args.model_root,
        external_root=args.external_root,
        receipt_output=args.receipt_output,
        attestation_output=args.attestation_output,
        loaded_parameter_count=release["loaded_parameters"],
        calibration_path=args.model_root / model.model_dir / "decision_config.json",
    )
    print(json.dumps({"status": "attested", "model_id": model.model_id}))


if __name__ == "__main__":
    main()
