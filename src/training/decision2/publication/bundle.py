"""Assemble a portable Hugging Face Decision 2.0 bundle without running a GPU.

Only a completed, full materialization is accepted. The selected LoRA CAL
report is copied byte-for-byte and bound to the merged model by its receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from training.model.calibration import load_calibration, verified_materialization_origin
from training.model.infer import checkpoint_fingerprint

from .training_record import bind_training_record

BUNDLE_VERSION = "decision2-self-contained-bundle/1"
MODEL_SUFFIXES = {".json", ".safetensors", ".model", ".txt", ".jinja"}
RUNTIME_SOURCES = (
    "calibration.py",
    "data.py",
    "decision_model.py",
    "infer.py",
    "lora.py",
    "source.py",
)
CARD_FILES = ("score-table.md", "ranking.svg", "matrix.svg", "manifest.json")
SECRET = re.compile(
    r"(?i)(?:\bhf_[A-Za-z0-9]{20,}|\bjv_live_[A-Za-z0-9_-]{16,}"
    r"|\bapikey_[A-Za-z0-9_-]{16,}|\bsk-[A-Za-z0-9_-]{16,}"
    r"|Authorization\s*:\s*Bearer\s+\S+)"
)
HOST_PATH = re.compile(
    r"(?<![A-Za-z0-9:/])/(?:home|root|data|work|mnt|tmp|private|Users|var|opt)/[^\s\"'<>]+"
    r"|\b[A-Za-z]:[\\/](?:Users|Documents|ProgramData|Windows)[\\/][^\s\"'<>]+"
)
PUBLIC_MODEL_ID = re.compile(r"llm-semantic-router/dev-2\.0-(?:27b|9b|4b|2b|0\.8b)\Z")
HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
LICENSE = re.compile(r"[a-z0-9][a-z0-9.-]*\Z")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _reject_absolute_values(value: Any, filename: str) -> None:
    if isinstance(value, str) and value.startswith("/"):
        raise ValueError(f"Absolute path value found in {filename}")
    if isinstance(value, dict):
        for item in value.values():
            _reject_absolute_values(item, filename)
    elif isinstance(value, list):
        for item in value:
            _reject_absolute_values(item, filename)


def _screen_text(path: Path) -> None:
    """Reject common credential formats and workstation paths before copying."""
    content = path.read_text(encoding="utf-8")
    if SECRET.search(content):
        raise ValueError(f"Credential-like text found in {path.name}")
    if HOST_PATH.search(content):
        raise ValueError(f"Absolute host path found in {path.name}")
    if path.suffix == ".json":
        _reject_absolute_values(json.loads(content), path.name)


def _screen_safetensors(path: Path) -> None:
    with path.open("rb") as source:
        header_length = int.from_bytes(source.read(8), "little")
        if not 2 <= header_length <= 128 << 20:
            raise ValueError(f"Invalid safetensors header in {path.name}")
        header = source.read(header_length)
        if len(header) != header_length:
            raise ValueError(f"Truncated safetensors header in {path.name}")
    # Tensor bytes are opaque; metadata is the only human-readable portion.
    metadata = json.loads(header)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid safetensors metadata in {path.name}")
    encoded = json.dumps(metadata, ensure_ascii=False)
    if SECRET.search(encoded) or HOST_PATH.search(encoded):
        raise ValueError(f"Private text found in safetensors metadata: {path.name}")
    _reject_absolute_values(metadata, path.name)


def _screen_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Bundle input must be a regular file: {path.name}")
    if path.suffix == ".safetensors":
        _screen_safetensors(path)
    else:
        _screen_text(path)


def _model_files(checkpoint: Path) -> list[Path]:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("Checkpoint must be a regular directory")
    files = []
    for file in sorted(checkpoint.rglob("*")):
        if file.is_symlink():
            raise ValueError("Materialized checkpoint contains a symlink")
        if file.is_dir():
            if file != checkpoint / "backbone":
                raise ValueError("Materialized checkpoint has an unexpected directory")
            continue
        if file.suffix not in MODEL_SUFFIXES or file.relative_to(checkpoint).parts[
            0
        ] not in {file.name, "backbone"}:
            raise ValueError(
                f"Materialized checkpoint has a non-model file: {file.name}"
            )
        _screen_file(file)
        files.append(file)
    required = {
        "decision_config.json",
        "decision_head.safetensors",
        "materialization_receipt.json",
    }
    if not required.issubset(
        {file.name for file in files if file.parent == checkpoint}
    ):
        raise ValueError(
            "Materialized checkpoint lacks config, head, or portable receipt"
        )
    if not (checkpoint / "backbone").is_dir():
        raise ValueError("Materialized checkpoint lacks a backbone directory")
    if not any(
        file.suffix == ".safetensors" and file.parent == checkpoint / "backbone"
        for file in files
    ):
        raise ValueError("Materialized checkpoint has no safe backbone weights")
    if not any(
        file.name.startswith("tokenizer") for file in files if file.parent == checkpoint
    ):
        raise ValueError("Materialized checkpoint lacks tokenizer files")
    return files


def _card_artifacts(
    path: Path, score_key: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Card artifacts must be a regular directory")
    for name in CARD_FILES:
        file = path / name
        _screen_file(file)
    manifest = _json(path / "manifest.json")
    if manifest.get("publication_version") != "decision-model-card-artifacts/2":
        raise ValueError("Card artifacts have an unknown generator version")
    digests = manifest.get("artifacts_sha256")
    if not isinstance(digests, dict) or any(
        digests.get(name) != sha_file(path / name) for name in CARD_FILES[:3]
    ):
        raise ValueError("Card artifact hashes differ from their generator manifest")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ValueError("Card manifest lacks model identities")
    matches = [
        entry
        for entry in models
        if isinstance(entry, dict) and entry.get("key") == score_key
    ]
    if len(matches) != 1 or matches[0].get("group") != "decision2":
        raise ValueError("The selected score key must identify one Decision 2.0 model")
    return manifest, matches[0]


def _md(value: Any) -> str:
    """Render one untrusted metadata value as inert Markdown text."""
    escaped = str(value).replace("\\", "\\\\").replace("<", "&lt;").replace(">", "&gt;")
    for mark in "|[]`*_~#!(){}":
        escaped = escaped.replace(mark, "\\" + mark)
    return escaped.replace("\n", " ").replace("\r", " ")


def _training_section(training: dict[str, Any]) -> str:
    declaration = training["declaration"]
    verified = training["verification"]
    optimization = training["optimization"]
    calibration = training["calibration"]
    init = declaration["initialization"]
    source_rows = [
        f"| {_md(source['source'])} | {source['rows']} | "
        f"[{_md(source['attribution'])}]({source['url']}) | {_md(source['license_status'])} |"
        for source in declaration["sources"]
    ]
    lora = optimization["lora"]
    rights = training["rights"]
    rights_lines = [
        "## Source rights and release scope",
        "",
        f"Declared scope: {_md(rights['publication_scope'])}. "
        "The dataset manifest and any research-use attestation "
        "were bound to the exact TRAIN/SELECT/CAL and run hashes.",
        "",
    ]
    if rights["mode"] == "noncommercial_research":
        rights_lines.extend(
            [
                "This model package is for noncommercial research only. It does not "
                "redistribute the upstream TRAIN, SELECT or CAL text. The mixed "
                "source conditions below are disclosures, not a new license grant.",
                "",
                "| Source condition | Terms and evidence |",
                "| --- | --- |",
                *(
                    f"| {_md(name)} | {_md(value['terms'])} {_md(value['evidence'])} |"
                    for name, value in sorted(rights["conditions"].items())
                ),
                "",
            ]
        )
        rights_lines.extend(f"- {_md(note)}" for note in rights.get("limitations", []))
    else:
        rights_lines.extend(
            [
                "Additional dataset conditions:",
                "",
                *(f"- {_md(note)}" for note in rights["conditions"]),
                "",
            ]
        )
    rights_lines.extend(
        [
            "SELECT/CAL source counts:",
            "",
            "| Split | Source | Rows |",
            "| --- | --- | ---: |",
        ]
    )
    for role, counts in sorted(rights["holdout_source_counts"].items()):
        rights_lines.extend(
            f"| {_md(role.upper())} | {_md(name)} | {count} |"
            for name, count in sorted(counts.items())
        )
    rights_lines.append("")
    lines = [
        "## Training and data",
        "",
        f"Initialization: [{_md(init['model_id'])}](https://huggingface.co/{init['model_id']}) "
        f"at immutable revision `{init['revision']}`; declared source license status: "
        f"{_md(init['license_status'])}. The initialization source-file fingerprint was "
        "checked against the completed training run and merged-model receipt.",
        "",
        f"TRAIN {verified['partition_rows']['train']:,} rows (SHA-256 "
        f"`{verified['partition_sha256']['train']}`), SELECT "
        f"{verified['partition_rows']['select']:,} rows (SHA-256 "
        f"`{verified['partition_sha256']['select']}`), and CAL "
        f"{verified['partition_rows']['cal']:,} rows (SHA-256 "
        f"`{verified['partition_sha256']['cal']}`). Native task counts in TRAIN: "
        + ", ".join(
            f"{_md(kind)} {count:,}"
            for kind, count in sorted(verified["task_type_counts"].items())
        )
        + ".",
        "",
        "| Training source | Rows | Attribution | License and rights status |",
        "| --- | ---: | --- | --- |",
        *source_rows,
        "",
        "The source counts and partition hashes were checked against the private "
        "training-data builder manifest and the run provenance. Rights statements "
        "are reviewed declarations, not independently established by this packager.",
        "",
        "## Optimization, selection, and calibration",
        "",
        f"Training used `{_md(optimization['train_mode'])}` with "
        f"`{_md(optimization['objective'])}` objective and Brier weight "
        f"`{optimization['brier_weight']}`, {optimization['epochs']} epoch(s), "
        f"{optimization['planned_updates']} optimizer updates, seed "
        f"`{optimization['seed']}`, microbatch {optimization['microbatch']} × "
        f"accumulation {optimization['accumulation']}, LoRA rank {lora['rank']} / "
        f"alpha {lora['alpha']} / dropout {lora['dropout']}, LoRA LR "
        f"`{lora['lr']}`, head LR `{optimization['head_lr']}`, and maximum "
        f"context {optimization['max_length']} tokens. Precision: "
        f"{_md(optimization['precision'])}.",
        "",
        f"Selected checkpoint: `{verified['selected_checkpoint']}` at optimizer "
        f"step {verified['selected_step']}. Selection rule: "
        f"{_md(optimization['selection'])}. The independent SELECT partition "
        "chose this checkpoint; CAL was not used for checkpoint selection.",
        "",
        f"CAL fit {calibration['cal_rows']:,} audited examples after training "
        f"completion using `{_md(calibration['selection_policy'])}`. "
        "Per-type temperatures: "
        + ", ".join(
            f"{_md(kind)} {value:.4f}"
            for kind, value in sorted(calibration["temperature_by_type"].items())
        )
        + ".",
        "",
        "Exact run receipt and builder-manifest SHA-256 values, source/model "
        "fingerprints and optimization settings are in `training-provenance.json`.",
        "",
        "## Evaluation exposure and limitations",
        "",
        "Known overlap and interpretation notes declared for this release:",
        "",
        *(f"- {_md(note)}" for note in training["known_overlap"]),
        *(f"- {_md(note)}" for note in training["evaluation_interpretation"]),
        "",
        "Training-data builder limitations:",
        "",
        *(f"- {_md(note)}" for note in training["builder_limitations"]),
        "",
        "Additional model limitations:",
        "",
        *(f"- {_md(note)}" for note in training["limitations"]),
        "",
    ]
    return "\n".join([*lines, *rights_lines])


def _card(
    *,
    model_id: str,
    base_model_id: str,
    license_id: str,
    score_table: str,
    score_identity: dict[str, Any],
    evaluation_binding: str,
    max_length: int,
    training: dict[str, Any],
) -> str:
    note = (
        "The supplied scored prediction manifest identifies these exact published merged weights."
        if evaluation_binding == "merged"
        else "The supplied scored prediction manifest identifies the selected pre-merge LoRA checkpoint; "
        "numerical parity with the published merged weights still needs a separate check."
    )
    mixed_license = (
        "license_name: mixed-source-noncommercial-research-terms\n"
        if license_id == "other"
        else ""
    )
    return f"""---
license: {license_id}
{mixed_license}base_model: {base_model_id}
tags:
- decision-model
- typed-decision
- custom-code
- qwen3_5
---

# {model_id}

Give the model a state, questions and candidate answers. It returns native
Choice, Noul and Score decisions with probabilities. Candidate labels are
supplied at runtime; each question is evaluated independently.

| Type | Output | Candidate range |
| --- | --- | --- |
| Choice | Selected label and full distribution | 2–255 |
| Noul | Probability of true | false/true |
| Score | Expected ordinal index and full distribution | 2–10 levels |

## Measured capability

The table and figures are generated from frozen scorer reports. Scored model
identity: `{score_identity['id']}` at `{score_identity['revision']}`.
{note} Missing and invalid answers remain in the score denominator.

![Final benchmark ranking](ranking.svg)

![Family and task-type accuracy matrix](matrix.svg)

{score_table.rstrip()}

{_training_section(training)}

## Use

Install a PyTorch build for your device, a Transformers release with
`Qwen3_5TextModel`, `safetensors`, and `huggingface_hub`. Inspect the bundled
`decision2/` source before loading custom model code.

```python
import sys
from huggingface_hub import snapshot_download

snapshot = snapshot_download("{model_id}")
sys.path.insert(0, snapshot)
from decision2 import Decision2

model = Decision2.from_pretrained(snapshot, device="cuda:0")
result = model.system_one(
    state="The customer reports a duplicate charge and requests a refund.",
    questions={{
        "route": {{"type": "choice", "instructions": "Which team should respond?",
                  "criteria": {{"billing": "Payments and refunds",
                               "support": "Product troubleshooting"}}}},
        "refund_requested": {{"type": "noul",
                             "instructions": "Was a refund requested?"}},
        "urgency": {{"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Routine", "Soon", "Immediate"]}},
    }},
)
print(result["answers"])
```

Noul uses `false: No` and `true: Yes` descriptions when criteria are omitted.
The complete state, question and candidates must fit {max_length} tokens;
over-budget questions receive an explicit error without truncation. GPU
inference uses BF16 backbone compute and an FP32 decision head. CPU inference
uses FP32 and requires enough RAM. The checkpoint's CAL-fitted temperatures
are loaded automatically after verification of model and calibration hashes.

## Architecture and provenance

A Qwen3.5 text backbone feeds a shared candidate-endpoint and global-query
head. The bundle contains a full merged backbone, decision head, tokenizer,
calibration report, immutable materialization receipt, source inference code,
verified public training record, and `MODEL_MANIFEST.json` with SHA-256 hashes
for every packaged file.
`Decision2.from_pretrained` verifies the manifest and CAL lineage before
loading weights. The model scores supplied evidence; it does not retrieve
facts or guarantee calibrated probabilities outside the CAL distribution.
The benchmark's exact inputs, scoring policy and model comparisons are in
`score-table.md` and `card-artifacts/manifest.json`.
"""


def bundle(
    *,
    checkpoint: Path,
    calibration: Path,
    card_artifacts: Path,
    scored_manifest: Path,
    score_key: str,
    model_id: str,
    base_model_id: str,
    license_id: str,
    output: Path,
    training_record: Path,
    run_dir: Path,
    training_data_manifest: Path,
    rights_attestation: Path | None = None,
) -> dict[str, Any]:
    """Validate all inputs, then atomically create a new publishable directory."""
    if not PUBLIC_MODEL_ID.fullmatch(model_id) or not HF_ID.fullmatch(base_model_id):
        raise ValueError(
            "Model must be llm-semantic-router/dev-2.0-xxb and base must be a Hugging Face ID"
        )
    if not REVISION.fullmatch(score_key) or not LICENSE.fullmatch(license_id):
        raise ValueError("Score key or SPDX license ID is invalid")
    if any(
        path.is_symlink()
        for path in (
            checkpoint,
            calibration,
            card_artifacts,
            scored_manifest,
            training_record,
            run_dir,
            training_data_manifest,
            output,
        )
    ):
        raise ValueError("Bundle inputs and output must not be symlinks")
    if rights_attestation is not None and rights_attestation.is_symlink():
        raise ValueError("Rights attestation must not be a symlink")
    checkpoint = checkpoint.resolve(strict=True)
    calibration = calibration.resolve(strict=True)
    card_artifacts = card_artifacts.resolve(strict=True)
    scored_manifest = scored_manifest.resolve(strict=True)
    training_record = training_record.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    training_data_manifest = training_data_manifest.resolve(strict=True)
    if rights_attestation is not None:
        rights_attestation = rights_attestation.resolve(strict=True)
    output = output.resolve()
    if output.exists() or any(
        output.is_relative_to(source)
        for source in (checkpoint, card_artifacts, run_dir)
    ):
        raise FileExistsError("Output exists or is inside a bundle input")
    model_files = _model_files(checkpoint)
    model_identity = checkpoint_fingerprint(checkpoint)
    metadata = _json(checkpoint / "decision_config.json")
    if (
        metadata.get("checkpoint_format") != "full"
        or metadata.get("architecture")
        != "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp"
        or metadata.get("prompt_version")
        != "decision2-segmented-options-global-query-v1"
    ):
        raise ValueError("Input is not a materialized Decision 2.0 full checkpoint")
    origin = verified_materialization_origin(checkpoint, model_identity["model_sha256"])
    if origin is None:
        raise ValueError(
            "Input checkpoint has no verified LoRA materialization lineage"
        )
    receipt_path = checkpoint / "materialization_receipt.json"
    receipt = _json(receipt_path)
    if receipt.get("merged_model_files_sha256") != model_identity["files_sha256"]:
        raise ValueError("Materialization receipt differs from actual model files")
    _screen_file(calibration)
    temperatures, cal_report = load_calibration(
        calibration,
        model_identity["model_sha256"],
        materialized_source_sha256=origin["source_model_sha256"],
    )
    max_length = cal_report.get("inference", {}).get("max_length")
    if type(max_length) is not int or max_length < 1:
        raise ValueError("CAL report lacks a positive audited inference max_length")
    _screen_file(training_record)
    if rights_attestation is not None:
        _screen_file(rights_attestation)
    training = bind_training_record(
        record_path=training_record,
        run_dir=run_dir,
        data_manifest_path=training_data_manifest,
        calibration=cal_report,
        merged_receipt=receipt,
        source_model_sha256=origin["source_model_sha256"],
        rights_attestation_path=rights_attestation,
        license_id=license_id,
    )
    card_manifest, selected = _card_artifacts(card_artifacts, score_key)
    published_size = model_id.removeprefix("llm-semantic-router/dev-2.0-")
    if (
        selected.get("size") is not None
        and str(selected["size"]).lower() != published_size
    ):
        raise ValueError("Score artifact size differs from the published model ID")
    _screen_file(scored_manifest)
    scored = _json(scored_manifest)
    identity = selected.get("identity")
    if not isinstance(identity, dict) or any(
        scored.get(field) != identity.get(key)
        for field, key in (("model_id", "id"), ("model_revision", "revision"))
    ):
        raise ValueError(
            "Score artifact identity differs from scored prediction manifest"
        )
    benchmark = selected.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get(
        "predictions_sha256"
    ) != scored.get("predictions_sha256"):
        raise ValueError("Score table is not bound to these scored predictions")
    scored_model_sha = scored.get("model_sha256")
    if scored_model_sha not in {
        model_identity["model_sha256"],
        origin["source_model_sha256"],
    }:
        raise ValueError("Scored predictions came from unrelated model weights")
    if (
        scored.get("calibration", {}).get("file_sha256") != sha_file(calibration)
        or scored.get("max_length") != max_length
    ):
        raise ValueError("Scored predictions used another calibration or context limit")
    evaluation_binding = (
        "merged"
        if scored_model_sha == model_identity["model_sha256"]
        else "premerge_source"
    )

    runtime_root = Path(__file__).resolve().parents[1] / "training" / "model"
    source_files = [runtime_root / name for name in RUNTIME_SOURCES]
    source_files.append(Path(__file__).with_name("runtime_api.py"))
    for file in source_files:
        _screen_file(file)
    score_table = (card_artifacts / "score-table.md").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "model" / "backbone").mkdir(parents=True)
        for source in model_files:
            relative = source.relative_to(checkpoint)
            destination = temporary / "model" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        shutil.copyfile(calibration, temporary / "calibration.json")
        (temporary / "decision2").mkdir()
        for source in source_files:
            name = "api.py" if source.name == "runtime_api.py" else source.name
            shutil.copyfile(source, temporary / "decision2" / name)
        (temporary / "decision2" / "__init__.py").write_text(
            "from .api import Decision2, verify_bundle\n",
            encoding="utf-8",
        )
        (temporary / "card-artifacts").mkdir()
        for name in CARD_FILES:
            shutil.copyfile(card_artifacts / name, temporary / "card-artifacts" / name)
        for name in ("score-table.md", "ranking.svg", "matrix.svg"):
            shutil.copyfile(card_artifacts / name, temporary / name)
        shutil.copyfile(scored_manifest, temporary / "scored-predictions.manifest.json")
        shutil.copyfile(training_record, temporary / "training-record.json")
        if rights_attestation is not None:
            shutil.copyfile(rights_attestation, temporary / "rights-attestation.json")
        (temporary / "training-provenance.json").write_text(
            json.dumps(
                training, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            _card(
                model_id=model_id,
                base_model_id=base_model_id,
                license_id=license_id,
                score_table=score_table,
                score_identity=identity,
                evaluation_binding=evaluation_binding,
                max_length=max_length,
                training=training,
            ),
            encoding="utf-8",
        )
        (temporary / "requirements.txt").write_text(
            "torch\ntransformers\nsafetensors\nhuggingface_hub\n",
            encoding="utf-8",
        )
        files_sha256 = {}
        for file in sorted(temporary.rglob("*")):
            if file.is_file():
                _screen_file(file)
                files_sha256[str(file.relative_to(temporary))] = sha_file(file)
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "model_id": model_id,
            "base_model_id": base_model_id,
            "license": license_id,
            "model_sha256": model_identity["model_sha256"],
            "model_files_sha256": model_identity["files_sha256"],
            "source_model_sha256": origin["source_model_sha256"],
            "materialization_sha256": origin["receipt_sha256"],
            "calibration_sha256": sha_file(calibration),
            "temperature_by_type": temperatures,
            "max_length": max_length,
            "score_key": score_key,
            "scored_identity": identity,
            "scored_predictions_sha256": scored["predictions_sha256"],
            "evaluation_weight_binding": evaluation_binding,
            "card_manifest_sha256": sha_file(card_artifacts / "manifest.json"),
            "frozen_gold_sha256": card_manifest.get("frozen_benchmark", {}).get(
                "gold_sha256"
            ),
            "training_record_sha256": sha_file(training_record),
            "training_provenance_sha256": sha_file(
                temporary / "training-provenance.json"
            ),
            "training_data_manifest_sha256": sha_file(training_data_manifest),
            "rights_mode": training["rights"]["mode"],
            "rights_attestation_sha256": training["rights"]["attestation_sha256"],
            "training_run_receipts_sha256": training["verification"][
                "run_receipts_sha256"
            ],
            "files_sha256": files_sha256,
        }
        (temporary / "MODEL_MANIFEST.json").write_text(
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        # Import from the staged bundle: its verifier must work without torch.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "decision2",
            temporary / "decision2" / "__init__.py",
            submodule_search_locations=[str(temporary / "decision2")],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import staged Decision 2.0 runtime")
        import sys

        previous = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name == "decision2" or name.startswith("decision2.")
        }
        try:
            module = importlib.util.module_from_spec(spec)
            sys.modules["decision2"] = module
            spec.loader.exec_module(module)
            module.verify_bundle(temporary)
        finally:
            for name in list(sys.modules):
                if name == "decision2" or name.startswith("decision2."):
                    del sys.modules[name]
            sys.modules.update(previous)
        if output.exists():
            raise FileExistsError(output)
        os.replace(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--card-artifacts", type=Path, required=True)
    parser.add_argument(
        "--scored-manifest",
        type=Path,
        required=True,
        help="training.model.infer prediction manifest for selected Decision 2.0 score",
    )
    parser.add_argument(
        "--training-record",
        type=Path,
        required=True,
        help="Reviewed public source/rights/limitations JSON disclosure",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed training run with BEST, COMPLETE and provenance receipts",
    )
    parser.add_argument(
        "--training-data-manifest",
        type=Path,
        required=True,
        help="Frozen source-data builder manifest with TRAIN/SELECT/CAL hashes and counts",
    )
    parser.add_argument(
        "--rights-attestation",
        type=Path,
        help="Exact noncommercial research source/run rights statement when using restricted pilot data",
    )
    parser.add_argument(
        "--score-key",
        required=True,
        help="Decision 2.0 key in card-artifacts/manifest.json",
    )
    parser.add_argument("--model-id", required=True, help="Publishable namespace/name")
    parser.add_argument(
        "--base-model-id", required=True, help="Attributed Hugging Face namespace/name"
    )
    parser.add_argument("--license", required=True, help="Reviewed SPDX license ID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = bundle(
        checkpoint=args.checkpoint,
        calibration=args.calibration,
        card_artifacts=args.card_artifacts,
        scored_manifest=args.scored_manifest,
        training_record=args.training_record,
        run_dir=args.run_dir,
        training_data_manifest=args.training_data_manifest,
        rights_attestation=args.rights_attestation,
        score_key=args.score_key,
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        license_id=args.license,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_sha256": manifest["model_sha256"],
                "evaluation_weight_binding": manifest["evaluation_weight_binding"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
