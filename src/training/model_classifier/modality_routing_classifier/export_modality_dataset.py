#!/usr/bin/env python3
"""
Export a fixed modality routing dataset for local review or Hugging Face upload.

The modality routing classifier currently builds its training data dynamically from
multiple public sources plus optional vLLM-generated BOTH examples. This script
materializes that pipeline into deterministic train/validation/test splits and
writes upload-friendly artifacts:

- train.jsonl / validation.jsonl / test.jsonl
- README.md dataset card
- label_mapping.json
- dataset_stats.json
- export_config.json
- hf_dataset/ (DatasetDict saved with save_to_disk)

Optionally, the exporter can also create/update a Hugging Face dataset repo and
upload the generated artifacts in one step.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, DatasetDict

try:
    from huggingface_hub import HfApi, upload_folder
except ImportError:  # pragma: no cover - optional dependency
    HfApi = None
    upload_folder = None

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "exported_modality_routing_dataset"
)
LABEL_AR = "AR"
LABEL_DIFFUSION = "DIFFUSION"
LABEL_BOTH = "BOTH"
DATASET_SCHEMA = {
    "text": "string",
    "label": "int64",
    "label_name": "string",
}
LABEL_DESCRIPTIONS = {
    LABEL_AR: "Text-only requests that should route to an autoregressive LLM.",
    LABEL_DIFFUSION: "Image-generation requests that should route to a diffusion model.",
    LABEL_BOTH: "Requests that benefit from both text and image responses.",
}
SOURCE_DATASETS = [
    "FredZhang7/stable-diffusion-prompts-2.47M",
    "succinctly/midjourney-prompts",
    "Falah/image_generation_prompts_SDXL",
    "nateraw/parti-prompts",
    "fal/image-generation-prompts",
    "OpenAssistant/oasst2",
    "tatsu-lab/alpaca",
    "databricks/databricks-dolly-15k",
    "stingning/ultrachat",
    "lmsys/lmsys-chat-1m",
    "allenai/WildChat",
    "mqliu/InterleavedBench",
]


def load_modality_dataset_builder():
    """
    Import the shared modality dataset builder lazily.

    The main training module pulls in the full training stack, so the exporter keeps
    that import out of module scope. This allows `--help` and local helper tests to
    work even when the training dependencies are not installed or are temporarily
    inconsistent.
    """

    module_names = [
        "src.training.model_classifier.modality_routing_classifier.modality_routing_bert_finetuning_lora",
        "modality_routing_bert_finetuning_lora",
    ]
    last_error = None

    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return module.ModalityRoutingDataset
        except Exception as exc:  # pragma: no cover - error path
            last_error = exc

    raise RuntimeError(
        "Failed to import ModalityRoutingDataset from "
        "`modality_routing_bert_finetuning_lora.py`. "
        "The exporter reuses the training data builder, so this environment needs a "
        "compatible training stack (notably `transformers`, `peft`, and "
        "`huggingface_hub`). Original error: "
        f"{last_error}"
    ) from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the modality routing dataset into fixed train/validation/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the exported dataset artifacts will be written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=6000,
        help="Maximum total samples to materialize across all labels.",
    )
    parser.add_argument(
        "--synthesize-both",
        type=int,
        default=0,
        help="Number of BOTH-class samples to synthesize via the configured vLLM endpoint.",
    )
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default=None,
        help="Optional vLLM-compatible endpoint used for BOTH-class synthesis.",
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default=None,
        help="Optional model name for the vLLM synthesis endpoint.",
    )
    parser.add_argument(
        "--vllm-api-key",
        type=str,
        default=os.environ.get("VLLM_API_KEY"),
        help="Optional API key for the vLLM endpoint. Defaults to $VLLM_API_KEY.",
    )
    parser.add_argument(
        "--dataset-title",
        type=str,
        default="Modality Routing Dataset",
        help="Dataset title used in the generated dataset card.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the exported dataset directory to a Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face dataset repo id, for example your-org/modality-routing-dataset.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face dataset repo as private when used with --push-to-hub.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face token for upload. Defaults to $HF_TOKEN or $HUGGINGFACE_TOKEN.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload modality routing dataset export",
        help="Commit message used when uploading to Hugging Face.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory if it already exists.",
    )
    return parser.parse_args()


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Pass --overwrite to replace it."
        )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_split_dataset(
    texts: Sequence[str],
    labels: Sequence[int],
    id2label: Mapping[int, str],
) -> Dataset:
    if len(texts) != len(labels):
        raise ValueError(
            f"Split has mismatched lengths: {len(texts)} texts vs {len(labels)} labels"
        )

    rows = []
    for text, label in zip(texts, labels, strict=True):
        rows.append(
            {
                "text": text,
                "label": label,
                "label_name": id2label[label],
            }
        )
    return Dataset.from_list(rows)


def build_dataset_dict(
    split_payloads: Mapping[str, tuple[Sequence[str], Sequence[int]]],
    id2label: Mapping[int, str],
) -> DatasetDict:
    dataset_dict = DatasetDict()
    for split_name in SPLIT_NAMES:
        texts, labels = split_payloads[split_name]
        dataset_dict[split_name] = build_split_dataset(texts, labels, id2label)
    return dataset_dict


def compute_label_counts(
    labels: Iterable[int], id2label: Mapping[int, str]
) -> dict[str, int]:
    counter = Counter(labels)
    return {id2label[idx]: counter.get(idx, 0) for idx in sorted(id2label)}


def compute_dataset_stats(
    dataset_dict: DatasetDict,
    id2label: Mapping[int, str],
) -> dict[str, object]:
    split_stats = {}
    total_counter = Counter()

    for split_name in SPLIT_NAMES:
        dataset = dataset_dict[split_name]
        label_counts = compute_label_counts(dataset["label"], id2label)
        split_stats[split_name] = {
            "num_rows": len(dataset),
            "label_counts": label_counts,
        }
        total_counter.update(dataset["label"])

    total_rows = sum(split_stats[split]["num_rows"] for split in SPLIT_NAMES)
    return {
        "total_rows": total_rows,
        "label_counts": {
            id2label[idx]: total_counter.get(idx, 0) for idx in sorted(id2label)
        },
        "splits": split_stats,
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl_exports(output_dir: Path, dataset_dict: DatasetDict) -> None:
    for split_name in SPLIT_NAMES:
        rows = (
            {
                "text": row["text"],
                "label": row["label"],
                "label_name": row["label_name"],
            }
            for row in dataset_dict[split_name]
        )
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)


def build_export_config(
    args: argparse.Namespace, label2id: Mapping[str, int]
) -> dict[str, object]:
    return {
        "dataset_title": args.dataset_title,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_samples": args.max_samples,
        "synthesize_both": args.synthesize_both,
        "vllm_endpoint": args.vllm_endpoint,
        "vllm_model": args.vllm_model,
        "vllm_synthesis_enabled": bool(args.vllm_endpoint and args.synthesize_both > 0),
        "label2id": dict(label2id),
        "schema": DATASET_SCHEMA,
        "source_datasets": SOURCE_DATASETS,
        "split_strategy": "70% train / 15% validation / 15% test with random_state=42",
        "shuffle_seed": 42,
    }


def build_dataset_card(
    dataset_title: str,
    stats: Mapping[str, object],
    export_config: Mapping[str, object],
) -> str:
    split_rows = []
    for split_name in SPLIT_NAMES:
        split_stats = stats["splits"][split_name]
        split_rows.append(
            f"| {split_name} | {split_stats['num_rows']} | "
            f"{split_stats['label_counts'][LABEL_AR]} | "
            f"{split_stats['label_counts'][LABEL_DIFFUSION]} | "
            f"{split_stats['label_counts'][LABEL_BOTH]} |"
        )

    synthesis_note = (
        "enabled" if export_config["vllm_synthesis_enabled"] else "disabled"
    )

    return f"""---
pretty_name: {dataset_title}
task_categories:
- text-classification
language:
- en
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# {dataset_title}

This dataset materializes the dynamic modality routing data builder used by the local
mmBERT-32K modality router training pipeline. The export is intended for review,
versioning, and uploading to a Hugging Face dataset repository.

## Labels

| Label | ID | Description |
|-------|----|-------------|
| {LABEL_AR} | {export_config['label2id'][LABEL_AR]} | {LABEL_DESCRIPTIONS[LABEL_AR]} |
| {LABEL_DIFFUSION} | {export_config['label2id'][LABEL_DIFFUSION]} | {LABEL_DESCRIPTIONS[LABEL_DIFFUSION]} |
| {LABEL_BOTH} | {export_config['label2id'][LABEL_BOTH]} | {LABEL_DESCRIPTIONS[LABEL_BOTH]} |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| text | string | Input user prompt |
| label | int64 | Integer class id |
| label_name | string | Human-readable class label |

## Splits

| Split | Rows | {LABEL_AR} | {LABEL_DIFFUSION} | {LABEL_BOTH} |
|-------|------|------------|--------------------|--------------|
{chr(10).join(split_rows)}

## Export Configuration

- `max_samples`: {export_config['max_samples']}
- `synthesize_both`: {export_config['synthesize_both']}
- `vllm_synthesis_enabled`: {synthesis_note}
- `vllm_endpoint`: {export_config['vllm_endpoint'] or "None"}
- `vllm_model`: {export_config['vllm_model'] or "None"}
- `split_strategy`: {export_config['split_strategy']}

## Sources

- `FredZhang7/stable-diffusion-prompts-2.47M`
- `succinctly/midjourney-prompts`
- `Falah/image_generation_prompts_SDXL`
- `nateraw/parti-prompts`
- `fal/image-generation-prompts`
- `OpenAssistant/oasst2`
- `tatsu-lab/alpaca`
- `databricks/databricks-dolly-15k`
- `stingning/ultrachat`
- `lmsys/lmsys-chat-1m`
- `allenai/WildChat`
- `mqliu/InterleavedBench`
- Optional vLLM-generated BOTH prompts when enabled

## Files

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: upload-friendly JSONL splits
- `label_mapping.json`: label to integer mapping
- `dataset_stats.json`: row counts per split and label
- `export_config.json`: reproducibility metadata for this export
- `hf_dataset/`: local `DatasetDict.save_to_disk()` artifact
"""


def write_metadata_files(
    output_dir: Path,
    label2id: Mapping[str, int],
    stats: Mapping[str, object],
    export_config: Mapping[str, object],
) -> None:
    metadata_files = {
        "label_mapping.json": label2id,
        "dataset_stats.json": stats,
        "export_config.json": export_config,
    }
    for filename, payload in metadata_files.items():
        path = output_dir / filename
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def print_summary(output_dir: Path, stats: Mapping[str, object]) -> None:
    print(f"Exported modality dataset to: {output_dir}")
    print(f"Total rows: {stats['total_rows']}")
    for split_name in SPLIT_NAMES:
        split_stats = stats["splits"][split_name]
        counts = split_stats["label_counts"]
        print(
            f"  {split_name}: {split_stats['num_rows']} "
            f"(AR={counts[LABEL_AR]}, DIFFUSION={counts[LABEL_DIFFUSION]}, BOTH={counts[LABEL_BOTH]})"
        )
    print("Files:")
    print("  - train.jsonl / validation.jsonl / test.jsonl")
    print("  - README.md")
    print("  - label_mapping.json")
    print("  - dataset_stats.json")
    print("  - export_config.json")
    print("  - hf_dataset/")


def push_export_to_hub(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.repo_id:
        raise ValueError("--repo-id is required when using --push-to-hub")

    if HfApi is None or upload_folder is None:
        raise RuntimeError(
            "huggingface_hub is required for --push-to-hub. "
            "Install it in the export environment before uploading."
        )

    api = HfApi(token=args.hf_token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=args.hf_token,
    )
    upload_folder(
        folder_path=str(output_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.hf_token,
        commit_message=args.commit_message,
    )
    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


def export_dataset(args: argparse.Namespace) -> Path:
    validate_output_dir(args.output_dir, args.overwrite)
    modality_dataset_cls = load_modality_dataset_builder()

    dataset_loader = modality_dataset_cls(
        vllm_endpoint=args.vllm_endpoint,
        vllm_model=args.vllm_model,
        vllm_api_key=args.vllm_api_key,
    )
    split_payloads = dataset_loader.prepare_datasets(
        max_samples=args.max_samples,
        synthesize_both=args.synthesize_both,
    )

    dataset_dict = build_dataset_dict(split_payloads, dataset_loader.id2label)
    stats = compute_dataset_stats(dataset_dict, dataset_loader.id2label)
    export_config = build_export_config(args, dataset_loader.label2id)

    prepare_output_dir(args.output_dir, args.overwrite)
    dataset_dict.save_to_disk(str(args.output_dir / "hf_dataset"))
    write_jsonl_exports(args.output_dir, dataset_dict)
    write_metadata_files(
        args.output_dir,
        dataset_loader.label2id,
        stats,
        export_config,
    )

    dataset_card = build_dataset_card(args.dataset_title, stats, export_config)
    (args.output_dir / "README.md").write_text(dataset_card, encoding="utf-8")

    print_summary(args.output_dir, stats)
    if args.push_to_hub:
        push_export_to_hub(args, args.output_dir)
    return args.output_dir


def main() -> None:
    args = parse_args()

    if args.vllm_endpoint and args.synthesize_both == 0:
        print(
            "Warning: --vllm-endpoint is set but --synthesize-both is 0. "
            "The export will not add synthetic BOTH examples."
        )

    export_dataset(args)


if __name__ == "__main__":
    main()
