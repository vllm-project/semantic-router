"""Re-export an existing Kai checkpoint for a no-training selection ablation.

This verifies the completed trainer, exact parent native, checkpoint roster and
frozen parameters, then uses the published native exporter. The output is a
research artifact and never represents a newly selected training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from inference.kai_lex import MODELS, verify_native_bundle
from inference.kai_continuation import PARENT_MANIFEST_SHA, verify_export
from inference.run import file_digest


def export(
    *, base: Path, run: Path, step: int, checkpoint_sha: str, output: Path
) -> dict:
    if output.exists() or output.is_symlink() or step < 0:
        raise ValueError("New output and nonnegative existing step required")
    base, run = base.resolve(strict=True), run.resolve(strict=True)
    verify_native_bundle(base, "kai", MODELS["kai"]["revision"])
    receipt_path, run_path = run / "COMPLETE.json", run / "RUN.json"
    complete, run_receipt = json.loads(receipt_path.read_text()), json.loads(
        run_path.read_text()
    )
    identity = complete.get("identity")
    if (
        complete.get("status") != "COMPLETE_DECISION_FINETUNE"
        or identity != run_receipt.get("identity")
        or step not in {point["step"] for point in complete.get("curve", [])}
        or file_digest(Path(run_receipt["train_path"])) != identity["train_sha256"]
        or file_digest(Path(run_receipt["dev_path"])) != identity["dev_sha256"]
    ):
        raise ValueError("Completed run, data, or evaluated checkpoint differs")
    parent_native = Path(run_receipt["native_path"]).resolve(strict=True)
    parent_sha = identity["native_manifest_sha256"]
    if parent_sha == PARENT_MANIFEST_SHA:
        if parent_native != base / "native":
            raise ValueError("Kai source native path differs")
    else:
        parent_run = parent_native.parents[2]
        parent = verify_export(parent_run, parent_native, parent_sha)
        if parent["base_manifest_sha256"] != PARENT_MANIFEST_SHA:
            raise ValueError("Unsupported nested Kai source ancestry")
    sys.path.insert(0, str(base))
    import torch
    import decision_runtime as api
    from decision_finetune import run as trainer, state
    from safetensors.torch import load_file

    if trainer.source_identity() != identity["sources"]:
        raise ValueError("Pinned trainer/runtime source code differs")
    checkpoint = run / "attempts" / "0001" / f"checkpoint-{step:06d}"
    checkpoint, metadata = state.verify_checkpoint(checkpoint, checkpoint_sha)
    if metadata.get("identity") != identity or metadata.get("step") != step:
        raise ValueError("Checkpoint training identity or step differs")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    native = api.load_native(
        parent_native, expected_manifest_sha256=parent_sha, device="cuda:0"
    )
    native.collator = type(native.collator)(
        native.collator.tokenizer, max_length=1024, state_truncation="error"
    )
    api.configure_training(native, max_input_tokens=1024)
    frozen = api.frozen_snapshot(native)
    if metadata["frozen_hashes"] != {
        name: val["sha256"] for name, val in frozen.items()
    }:
        raise ValueError("Frozen checkpoint shared parameters differ from source")
    native.model.load_state_dict(
        load_file(str(checkpoint / "model.safetensors"), device="cpu"), strict=True
    )
    api.assert_frozen(native, frozen, content=True, versions=False)
    api.restore_native_policy_for_export(native)
    api.export_native(
        native,
        output,
        provenance={
            "cli": "research.kai06b_checkpoint_export",
            "research_only": True,
            "source_run_complete_sha256": file_digest(receipt_path),
            "source_run_json_sha256": file_digest(run_path),
            "source_native_manifest_sha256": parent_sha,
            "checkpoint_step": step,
            "checkpoint_manifest_sha256": checkpoint_sha,
            "selection_ablation": "retrospective macro-F1 SELECT; not the original trainer-selected export",
            "release_qualified": False,
        },
    )
    report = {
        "schema_version": "kai06b-existing-checkpoint-export/1",
        "research_only": True,
        "release_qualified": False,
        "source_run_complete_sha256": file_digest(receipt_path),
        "source_run_json_sha256": file_digest(run_path),
        "source_native_manifest_sha256": parent_sha,
        "checkpoint_step": step,
        "checkpoint_manifest_sha256": checkpoint_sha,
        "export_manifest_sha256": file_digest(output / "MANIFEST.json"),
        "export_path": str(output),
    }
    report_path = output.with_suffix(".receipt.json")
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                base=args.base_model,
                run=args.run,
                step=args.step,
                checkpoint_sha=args.checkpoint_sha256,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
