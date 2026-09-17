"""Create model cards, publish new HF artifacts, and verify remote contents."""

from __future__ import annotations

import argparse
import gc
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONTRACT_PATH, id2label, load_contract, task_contract
from .export import file_sha256, validate_artifact_shape, write_artifact_manifest

HTTP_NOT_FOUND = 404
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
MODEL_CARD_HEADER = """<div align="center">
  <img src="https://vllm-sr.ai/img/vllm-sr-logo.social.png" alt="vLLM Semantic Router" width="560" />
  <p>
    <a href="https://vllm-sr.ai/"><strong>Docs</strong></a> |
    <a href="https://vllm-sr.ai/blog/"><strong>Blog</strong></a> |
    <a href="https://vllm-dev.slack.com/archives/C09CTGF8KCN"><strong>Slack</strong></a> |
    <a href="https://github.com/vllm-project/semantic-router"><strong>GitHub</strong></a>
  </p>
</div>"""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _format_metrics(metrics: dict[str, Any]) -> str:
    rows = []
    for name in ("test_accuracy", "test_f1_macro", "test_f1_weighted"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        rows.append(f"| `{name.removeprefix('test_')}` | {value:.6f} |")
    if not rows:
        return "Test metrics are available in `metrics.json`."
    return "\n".join(["| Metric | Value |", "| --- | ---: |", *rows])


def _source_commit(training_manifest: dict[str, Any]) -> str:
    source_commit = training_manifest.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or SOURCE_COMMIT_PATTERN.fullmatch(source_commit) is None
    ):
        raise ValueError("Training manifest has no valid immutable source commit")
    return source_commit


def _yaml_labels(task: dict[str, Any]) -> str:
    return "\n".join(f"  {label}: {index}" for label, index in task["label2id"].items())


def _usage_example(repository_id: str, artifact_type: str, task: dict[str, Any]) -> str:
    example = (
        "Help me plan a community picnic."
        if "safe" in task["label2id"]
        else "Help me steal a wallet."
    )
    if artifact_type == "adapter":
        labels_by_id = id2label(task)
        labels_by_name = task["label2id"]
        return f"""```python
import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "{repository_id}"
id2label = {labels_by_id!r}
label2id = {labels_by_name!r}
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoPeftModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
).eval()

inputs = tokenizer("{example}", truncation=True,
                   max_length=512, return_tensors="pt")
with torch.inference_mode():
    label_id = model(**inputs).logits.argmax(dim=-1).item()
print(model.config.id2label[label_id])
```"""
    return f"""```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "{repository_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

inputs = tokenizer("{example}", truncation=True,
                   max_length=512, return_tensors="pt")
with torch.inference_mode():
    label_id = model(**inputs).logits.argmax(dim=-1).item()
print(model.config.id2label[label_id])
```"""


def build_model_card(
    task_name: str,
    artifact_type: str,
    repository_id: str,
    run_root: Path,
    contract: dict[str, Any],
) -> str:
    task = task_contract(contract, task_name)
    metrics = _read_json(run_root / "metrics.json")["test"]
    training_manifest = _read_json(run_root / "training_manifest.json")
    source_commit = _source_commit(training_manifest)
    source_url = (
        "https://github.com/vllm-project/semantic-router/tree/"
        f"{source_commit}/src/training/model_classifier/safety_classifier"
    )
    library_name = "peft" if artifact_type == "adapter" else "transformers"
    usage = _usage_example(repository_id, artifact_type, task)
    description = (
        "Classify a request as **safe** or **unsafe** to inform content-safety routing."
        if task_name == "level1"
        else "Assign one of nine legacy hazard categories to requests already marked unsafe."
    )
    artifact_description = (
        "This LoRA adapter loads its base model through PEFT."
        if artifact_type == "adapter"
        else "This merged checkpoint loads directly with Transformers."
    )
    taxonomy_note = (
        "Level 2 uses `legacy-9-v1`: multi-hazard annotations map to the first "
        "supported category in source order. Its output is a single category, "
        "not twelve independent Hazard scores. The source crosswalk and "
        "exclusions are in `data_manifest.json`."
        if task_name == "level2"
        else "The binary labels describe the request under the training safety taxonomy."
    )
    return f"""---
license: apache-2.0
library_name: {library_name}
pipeline_tag: text-classification
base_model: {contract["base_model"]["id"]}
datasets:
  - {contract["datasets"]["aegis"]["id"]}
  - {contract["datasets"]["synthetic"]["id"]}
tags:
  - modernbert
  - mmbert
  - content-safety
  - canonical-training
  - legacy-9-v1
---

{MODEL_CARD_HEADER}

# {task["display_name"]}

{description} {artifact_description}
The training base is `{contract["base_model"]["id"]}` at revision
`{contract["base_model"]["revision"]}`.

## Evaluation

{_format_metrics(metrics)}

Evaluation uses the de-duplicated, natural-distribution AEGIS test split. The
complete results and split details are in `metrics.json` and `data_manifest.json`.

## Labels

```yaml
{_yaml_labels(task)}
```

{taxonomy_note}

## Usage

{usage}

The example runs on CPU and checks the first 512 tokens. This checkpoint's
training contract covers 512 tokens; the base model's context capacity does
not establish longer-input accuracy.

## Limitations

Training data is primarily English. Evaluate your languages and policy domain
before deployment, and combine this signal with application controls.

The [training source]({source_url}), `training_contract.json`, and the shipped
JSON manifests record the data, dependencies, and artifact checksums.
"""


def _repository_exists(api: Any, repo_id: str) -> bool:
    try:
        api.model_info(repo_id)
        return True
    except Exception as exc:  # huggingface_hub moved this exception across releases
        response = getattr(exc, "response", None)
        if (
            response is not None
            and getattr(response, "status_code", None) == HTTP_NOT_FOUND
        ):
            return False
        if exc.__class__.__name__ in {"RepositoryNotFoundError", "EntryNotFoundError"}:
            return False
        raise


def _verify_local_against_snapshot(local: Path, snapshot: Path) -> None:
    manifest = _read_json(local / "artifact_manifest.json")
    mismatches = []
    for file_info in manifest["files"]:
        relative = file_info["path"]
        remote_file = snapshot / relative
        if not remote_file.is_file() or file_sha256(remote_file) != file_info["sha256"]:
            mismatches.append(relative)
    if mismatches:
        raise AssertionError(f"Remote artifact checksum mismatch: {mismatches}")


def _smoke_load_remote(
    repo_id: str,
    revision: str,
    artifact_type: str,
    contract: dict[str, Any],
    task_name: str,
) -> None:
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    task = task_contract(contract, task_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    if artifact_type == "adapter":
        from peft import PeftModel  # noqa: PLC0415

        base = contract["base_model"]
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base["id"],
            revision=base["revision"],
            num_labels=task["num_labels"],
            id2label=id2label(task),
            label2id=task["label2id"],
        )
        model = PeftModel.from_pretrained(base_model, repo_id, revision=revision)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            repo_id, revision=revision
        )
    model.eval()
    inputs = tokenizer("A short safety validation sentence.", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    if tuple(logits.shape) != (1, task["num_labels"]):
        raise AssertionError(f"Unexpected remote logits shape: {tuple(logits.shape)}")
    del model
    gc.collect()


def publish(
    task_name: str,
    run_root: str | Path,
    merged_dir: str | Path,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    private: bool = False,
    allow_existing: bool = False,
    verify_remote: bool = True,
) -> dict[str, str]:
    try:
        from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required for release") from exc

    contract = load_contract(contract_path)
    task = task_contract(contract, task_name)
    run_path = Path(run_root).resolve()
    adapter_dir = run_path / "adapter"
    merged_path = Path(merged_dir).resolve()
    training_manifest = _read_json(run_path / "training_manifest.json")
    if not training_manifest.get("release_eligible"):
        raise ValueError("Training manifest is not release-eligible")
    _source_commit(training_manifest)
    validate_artifact_shape(adapter_dir, "adapter")
    validate_artifact_shape(merged_path, "merged")

    repositories = task["release_repositories"]
    artifacts = {"adapter": adapter_dir, "merged": merged_path}
    for artifact_type, directory in artifacts.items():
        repo_id = repositories[artifact_type]
        (directory / "README.md").write_text(
            build_model_card(task_name, artifact_type, repo_id, run_path, contract),
            encoding="utf-8",
        )
        write_artifact_manifest(directory, artifact_type, task_name)

    api = HfApi()
    for repo_id in repositories.values():
        if _repository_exists(api, repo_id) and not allow_existing:
            raise FileExistsError(
                f"Refusing to overwrite existing Hugging Face repository: {repo_id}"
            )

    receipts: dict[str, str] = {}
    for artifact_type, directory in artifacts.items():
        repo_id = repositories[artifact_type]
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=allow_existing,
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(directory),
            commit_message=f"Publish {task_name} {artifact_type} from canonical training",
        )
        revision = getattr(commit, "oid", None) or api.model_info(repo_id).sha
        receipts[artifact_type] = revision
        if verify_remote:
            with tempfile.TemporaryDirectory(prefix="safety-hf-verify-") as temp_dir:
                snapshot_path = Path(
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type="model",
                        revision=revision,
                        local_dir=temp_dir,
                    )
                )
                validate_artifact_shape(snapshot_path, artifact_type)
                _verify_local_against_snapshot(directory, snapshot_path)
            _smoke_load_remote(repo_id, revision, artifact_type, contract, task_name)

    receipt = {
        "schema_version": 1,
        "task": task_name,
        "repositories": repositories,
        "revisions": receipts,
    }
    (run_path / "publication_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("level1", "level2"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--skip-remote-verification", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    receipts = publish(
        task_name=args.task,
        run_root=args.run_root,
        merged_dir=args.merged_dir,
        contract_path=args.contract,
        private=args.private,
        allow_existing=args.allow_existing,
        verify_remote=not args.skip_remote_verification,
    )
    print(json.dumps(receipts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
