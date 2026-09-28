"""Gold-free multi-panel collection for one sealed PEFT package (one model load).

Reuses ``publication.package_native_arena`` unchanged: the package's own loader,
manifest and base checks, per-item collection, prediction manifests and final
bundle re-verification. Only the loop over several prompt files is added, so a
~27B model is loaded once for DEV, CSS pilot or the formal panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from publication import package_native_arena as native


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument(
        "--panel", action="append", required=True, help="NAME=GOLD_FREE_PROMPTS"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    source = args.source.resolve(strict=True)
    revision = f"package-sha256:{args.expected_package_sha256}"
    contract, package_sha256 = native._package_manifest(
        package, args.expected_package_sha256, args.model_id, revision
    )
    panels = []
    for spec in args.panel:
        name, path = spec.split("=", 1)
        prompts = Path(path)
        output = args.output_dir / f"{name}.predictions.jsonl"
        if output.exists() or output.with_name(output.name + ".manifest.json").exists():
            raise FileExistsError(f"{output} already exists")
        panels.append((name, prompts, native._sha_file(prompts), output))
    api, sealed_infer = native._import_sealed_package(package)
    started = time.perf_counter()
    model = api.Decision2.from_pretrained(
        package, source_path=source, device=args.device
    )
    load_seconds = time.perf_counter() - started
    adapter_sha256 = hashlib.sha256(
        native._canonical(contract["loader_files_sha256"]).encode("utf-8")
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "load_seconds": load_seconds,
        "package_manifest_sha256": package_sha256,
        "panels": {},
    }
    for name, prompts, prompt_sha256, output in panels:
        rows = native.load_gold_free(prompts)
        if any(
            native.input_digest(row) != sealed_infer.prompt_input_sha256(row)
            for row in rows
        ):
            raise ValueError(
                f"{name}: package prompt digest differs from the panel contract"
            )
        panel_started = time.perf_counter()
        predictions, counts = native.collect(
            rows,
            model=model,
            model_id=args.model_id,
            model_revision=revision,
            manifest=contract,
            package_sha256=package_sha256,
            adapter_sha256=adapter_sha256,
            question_to_row=sealed_infer.question_to_row,
        )
        if native._sha_file(prompts) != prompt_sha256:
            raise ValueError(f"{name}: prompt bytes changed during inference")
        receipt = native.prediction_manifest(
            package=contract,
            package_sha256=package_sha256,
            model_id=args.model_id,
            model_revision=revision,
            input_sha256=prompt_sha256,
            input_items=len(rows),
            adapter_sha256=adapter_sha256,
            counts=counts,
        )
        receipt = native.write_predictions(output, predictions, receipt)
        summary["panels"][name] = {
            "input_sha256": prompt_sha256,
            "predictions_sha256": receipt["predictions_sha256"],
            "counts": counts,
            "seconds": time.perf_counter() - panel_started,
        }
        print(json.dumps({"panel": name, **summary["panels"][name]}), flush=True)
    if (
        native._sha_file(package / "MODEL_MANIFEST.json") != package_sha256
        or api.verify_bundle(package, source) != contract
    ):
        raise ValueError("Package or external base changed during inference")
    (args.output_dir / "COLLECT-SUMMARY.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
