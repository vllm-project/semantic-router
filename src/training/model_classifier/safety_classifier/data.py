"""Deterministic, prompt-only data preparation for the safety classifiers.

The pure builders in this module depend only on the Python standard library.
The ``prepare`` command imports ``huggingface_hub`` lazily, downloads the exact
dataset revisions pinned in the reconstruction contract, verifies their file
digests, and materializes JSONL for the training entrypoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from . import data_io as _data_io
    from .config import DEFAULT_CONTRACT_PATH, contract_sha256, load_contract
    from .data_contract import (
        AEGIS_SCHEMA,
        AUDIT_KEYS,
        DEFAULT_LEVEL1_PER_LABEL,
        DEFAULT_LEVEL2_PER_LABEL,
        DEFAULT_SEED,
        NORMALIZATION_VERSION,
        SPLIT_PRECEDENCE,
        SPLIT_RANK,
        SPLITS,
        SYNTH_SCHEMA,
        DataContractError,
        DataPreparationError,
        InsufficientClassSamplesError,
        SchemaError,
    )
    from .taxonomy import (
        LEVEL1_LABEL_TO_ID,
        LEVEL1_LABELS,
        LEVEL2_LABEL_TO_ID,
        LEVEL2_LABELS,
        TAXONOMY_VERSION,
        map_aegis_categories,
        map_synth_category,
    )
else:  # Allow ``python path/to/data.py prepare ...``.
    import data_io as _data_io  # type: ignore[no-redef]
    from data_contract import (  # type: ignore[no-redef]
        AEGIS_SCHEMA,
        AUDIT_KEYS,
        DEFAULT_LEVEL1_PER_LABEL,
        DEFAULT_LEVEL2_PER_LABEL,
        DEFAULT_SEED,
        NORMALIZATION_VERSION,
        SPLIT_PRECEDENCE,
        SPLIT_RANK,
        SPLITS,
        SYNTH_SCHEMA,
        DataContractError,
        DataPreparationError,
        InsufficientClassSamplesError,
        SchemaError,
    )
    from taxonomy import (  # type: ignore[no-redef]
        LEVEL1_LABEL_TO_ID,
        LEVEL1_LABELS,
        LEVEL2_LABEL_TO_ID,
        LEVEL2_LABELS,
        TAXONOMY_VERSION,
        map_aegis_categories,
        map_synth_category,
    )

    from config import DEFAULT_CONTRACT_PATH, contract_sha256, load_contract


load_json_array = _data_io.load_json_array
load_jsonl = _data_io.load_jsonl
materialize_dataset = _data_io.materialize_dataset
_download_contract_file = _data_io.download_contract_file
_positive_contract_int = _data_io.positive_contract_int
_validate_data_contract = _data_io.validate_data_contract


@dataclass(frozen=True)
class _Candidate:
    split: str
    text: str
    fingerprint: str
    label_name: str
    source: str
    source_id: str
    raw_categories: tuple[str, ...] = ()
    mapped_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedSample:
    """One prompt-only materialized training or evaluation example."""

    sample_id: str
    split: str
    text: str
    label: str
    label_id: int
    fingerprint: str
    source: str
    source_id: str
    occurrence: int
    raw_categories: tuple[str, ...] = ()
    mapped_targets: tuple[str, ...] = ()

    @property
    def label_name(self) -> str:
        """Compatibility alias for code that describes the string as a name."""
        return self.label

    @property
    def is_multitarget(self) -> bool:
        return len(self.mapped_targets) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "label": self.label,
            "label_id": self.label_id,
            "fingerprint": self.fingerprint,
            "source": self.source,
            "source_id": self.source_id,
            "occurrence": self.occurrence,
            "raw_categories": list(self.raw_categories),
            "mapped_targets": list(self.mapped_targets),
            "is_multitarget": self.is_multitarget,
        }


@dataclass(frozen=True)
class DatasetBuild:
    """In-memory deterministic dataset plus its pre-materialization manifest."""

    task: str
    samples: tuple[PreparedSample, ...]
    manifest: dict[str, Any]

    def split_samples(self, split: str) -> tuple[PreparedSample, ...]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return tuple(sample for sample in self.samples if sample.split == split)


def normalize_prompt(text: str) -> str:
    """Return the versioned comparison form used only for de-duplication."""
    if not isinstance(text, str):
        raise TypeError("Prompt text must be a string")
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).casefold()


def prompt_fingerprint(text: str) -> str:
    """Return a stable SHA-256 fingerprint of normalized prompt text."""
    return hashlib.sha256(normalize_prompt(text).encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_digest(*parts: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(parts))).hexdigest()


def _new_audit() -> Counter[str]:
    return Counter(dict.fromkeys(AUDIT_KEYS, 0))


def _require_exact_schema(
    row: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = set(row)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise SchemaError(f"{context} schema mismatch: {'; '.join(details)}")


def _require_string(
    row: Mapping[str, object], key: str, context: str, *, nonempty: bool = False
) -> str:
    value = row[key]
    if not isinstance(value, str):
        raise SchemaError(f"{context}.{key} must be a string")
    if nonempty and not value:
        raise SchemaError(f"{context}.{key} must be non-empty")
    return value


def validate_aegis_row(row: Mapping[str, object], context: str = "aegis row") -> None:
    """Validate a row against the exact AEGIS 2.0 JSON schema used here."""
    if not isinstance(row, Mapping):
        raise SchemaError(f"{context} must be a mapping")
    _require_exact_schema(row, AEGIS_SCHEMA, context)
    _require_string(row, "id", context, nonempty=True)
    _require_string(row, "prompt", context)
    _require_string(row, "violated_categories", context)
    _require_string(row, "prompt_label_source", context)

    reconstruction_id = row["reconstruction_id_if_redacted"]
    valid_reconstruction_id = reconstruction_id is None or (
        isinstance(reconstruction_id, (int, float))
        and not isinstance(reconstruction_id, bool)
    )
    if not valid_reconstruction_id:
        raise SchemaError(
            f"{context}.reconstruction_id_if_redacted must be numeric or null"
        )

    if row["response"] is not None and not isinstance(row["response"], str):
        raise SchemaError(f"{context}.response must be a string or null")
    if row["prompt_label"] not in LEVEL1_LABEL_TO_ID:
        raise SchemaError(f"{context}.prompt_label must be 'safe' or 'unsafe'")
    response_label = row["response_label"]
    if response_label is not None and response_label not in LEVEL1_LABEL_TO_ID:
        raise SchemaError(f"{context}.response_label must be safe, unsafe, or null")
    response_source = row["response_label_source"]
    if response_source is not None and not isinstance(response_source, str):
        raise SchemaError(f"{context}.response_label_source must be a string or null")


def validate_synth_row(
    row: Mapping[str, object], context: str = "synthetic row"
) -> None:
    """Validate a row against the exact pinned synthetic JSONL schema."""
    if not isinstance(row, Mapping):
        raise SchemaError(f"{context} must be a mapping")
    _require_exact_schema(row, SYNTH_SCHEMA, context)
    _require_string(row, "text", context)
    _require_string(row, "category", context, nonempty=True)
    label = row["label"]
    if not isinstance(label, int) or isinstance(label, bool) or label != 1:
        raise SchemaError(f"{context}.label must be the integer 1")


def _usable_prompt(prompt: str, audit: Counter[str]) -> str | None:
    text = prompt.strip()
    if not text:
        audit["empty_prompt_rows"] += 1
        return None
    if normalize_prompt(text) == "redacted":
        audit["redacted_prompt_rows"] += 1
        return None
    return text


def _aegis_candidates(
    task: str,
    aegis_splits: Mapping[str, Iterable[Mapping[str, object]]],
    audit: Counter[str],
) -> list[_Candidate]:
    unknown_splits = sorted(set(aegis_splits) - set(SPLITS))
    if unknown_splits:
        raise DataPreparationError(f"Unknown AEGIS splits: {unknown_splits}")

    candidates: list[_Candidate] = []
    for split in SPLITS:
        for row_index, row in enumerate(aegis_splits.get(split, ())):
            context = f"aegis.{split}[{row_index}]"
            audit["aegis_rows_seen"] += 1
            validate_aegis_row(row, context)
            mapping = map_aegis_categories(str(row["violated_categories"]))
            audit["schema_valid_rows"] += 1

            text = _usable_prompt(str(row["prompt"]), audit)
            if text is None:
                continue

            if task == "level1":
                label_name = str(row["prompt_label"])
            else:
                if row["prompt_label"] == "safe":
                    audit["level2_safe_rows_skipped"] += 1
                    continue
                if mapping.is_multilabel:
                    audit["level2_multilabel_rows"] += 1
                if mapping.is_multitarget:
                    audit["level2_multitarget_rows"] += 1
                if mapping.selected_label is None:
                    audit["level2_no_mapped_target_rows"] += 1
                    continue
                label_name = mapping.selected_label

            candidates.append(
                _Candidate(
                    split=split,
                    text=text,
                    fingerprint=prompt_fingerprint(text),
                    label_name=label_name,
                    source="aegis",
                    source_id=str(row["id"]),
                    raw_categories=mapping.raw_categories,
                    mapped_targets=mapping.mapped_targets,
                )
            )
    return candidates


def _synth_candidates(
    task: str,
    synth_rows: Iterable[Mapping[str, object]],
    audit: Counter[str],
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for row_index, row in enumerate(synth_rows):
        context = f"synthetic.train[{row_index}]"
        audit["synthetic_rows_seen"] += 1
        validate_synth_row(row, context)
        category = str(row["category"])
        mapped_label = map_synth_category(category)
        audit["schema_valid_rows"] += 1

        text = _usable_prompt(str(row["text"]), audit)
        if text is None:
            continue
        fingerprint = prompt_fingerprint(text)
        label_name = "unsafe" if task == "level1" else mapped_label
        candidates.append(
            _Candidate(
                split="train",
                text=text,
                fingerprint=fingerprint,
                label_name=label_name,
                source="synthetic",
                source_id=f"synthetic:{fingerprint}:{category}",
                raw_categories=(category,),
                mapped_targets=(mapped_label,),
            )
        )
    return candidates


def _representative_key(candidate: _Candidate) -> tuple[object, ...]:
    source_rank = {"aegis": 0, "synthetic": 1}
    return (
        source_rank[candidate.source],
        candidate.source_id,
        candidate.text,
        candidate.raw_categories,
        candidate.mapped_targets,
    )


def _deduplicate(
    candidates: Sequence[_Candidate], audit: Counter[str]
) -> list[_Candidate]:
    groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.fingerprint].append(candidate)

    audit["candidate_rows"] = len(candidates)
    audit["unique_fingerprint_groups"] = len(groups)
    selected: list[_Candidate] = []
    for fingerprint in sorted(groups):
        group = groups[fingerprint]
        if len({candidate.label_name for candidate in group}) > 1:
            audit["label_conflict_groups"] += 1
            audit["label_conflict_rows"] += len(group)
            continue

        selected_split = min(
            (candidate.split for candidate in group), key=SPLIT_RANK.__getitem__
        )
        split_candidates = [
            candidate for candidate in group if candidate.split == selected_split
        ]
        lower_precedence = len(group) - len(split_candidates)
        same_split_duplicates = len(split_candidates) - 1
        audit["lower_precedence_rows_dropped"] += lower_precedence
        audit["same_split_duplicate_rows_dropped"] += same_split_duplicates
        audit["deduplicated_rows"] += lower_precedence + same_split_duplicates
        selected.append(min(split_candidates, key=_representative_key))

    audit["unique_rows_after_dedup"] = len(selected)
    return selected


def _validate_sampling_inputs(per_label: int, seed: int) -> None:
    if not isinstance(per_label, int) or isinstance(per_label, bool) or per_label < 1:
        raise ValueError("per_label must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")


def _rank_candidate(task: str, seed: int, candidate: _Candidate) -> str:
    return _stable_digest(
        TAXONOMY_VERSION,
        task,
        "sample",
        seed,
        candidate.label_name,
        candidate.fingerprint,
        candidate.source,
        candidate.source_id,
    )


def _sample_train_rows(
    task: str,
    candidates: Sequence[_Candidate],
    labels: Sequence[str],
    per_label: int,
    seed: int,
    audit: Counter[str],
) -> tuple[list[tuple[_Candidate, int]], dict[str, dict[str, int]]]:
    pools: dict[str, list[_Candidate]] = {label: [] for label in labels}
    for candidate in candidates:
        if candidate.split == "train":
            pools[candidate.label_name].append(candidate)

    sampling: dict[str, dict[str, int]] = {}
    emitted: list[tuple[_Candidate, int]] = []
    for label in labels:
        ordered = sorted(
            pools[label], key=lambda candidate: _rank_candidate(task, seed, candidate)
        )
        available = len(ordered)
        audit["train_rows_available"] += available
        if not available:
            raise InsufficientClassSamplesError(
                f"{task} label {label!r} has no unique train samples"
            )
        if task == "level1" and available < per_label:
            raise InsufficientClassSamplesError(
                f"{task} label {label!r} has {available} unique train samples; "
                f"requires {per_label}"
            )

        selected_unique = min(available, per_label)
        oversampled = max(0, per_label - available) if task == "level2" else 0
        downsampled = max(0, available - per_label)
        audit["train_rows_selected"] += selected_unique
        audit["train_rows_oversampled"] += oversampled
        audit["train_rows_downsampled"] += downsampled

        occurrence_counts: Counter[tuple[str, str, str]] = Counter()
        for index in range(per_label):
            candidate = ordered[index % available]
            identity = (
                candidate.source,
                candidate.source_id,
                candidate.fingerprint,
            )
            occurrence = occurrence_counts[identity]
            occurrence_counts[identity] += 1
            emitted.append((candidate, occurrence))

        sampling[label] = {
            "available_unique": available,
            "selected_unique": selected_unique,
            "downsampled": downsampled,
            "oversampled": oversampled,
            "emitted": per_label,
        }
    return emitted, sampling


def _prepared_sample(
    task: str,
    candidate: _Candidate,
    occurrence: int,
    label_to_id: Mapping[str, int],
) -> PreparedSample:
    sample_id = _stable_digest(
        TAXONOMY_VERSION,
        task,
        candidate.split,
        candidate.source,
        candidate.source_id,
        candidate.fingerprint,
        occurrence,
    )
    return PreparedSample(
        sample_id=sample_id,
        split=candidate.split,
        text=candidate.text,
        label=candidate.label_name,
        label_id=label_to_id[candidate.label_name],
        fingerprint=candidate.fingerprint,
        source=candidate.source,
        source_id=candidate.source_id,
        occurrence=occurrence,
        raw_categories=candidate.raw_categories,
        mapped_targets=candidate.mapped_targets,
    )


def _output_order(task: str, seed: int, sample: PreparedSample) -> tuple[int, str]:
    digest = _stable_digest(
        TAXONOMY_VERSION,
        task,
        "output",
        seed,
        sample.split,
        sample.label_name,
        sample.fingerprint,
        sample.source_id,
        sample.occurrence,
    )
    return SPLITS.index(sample.split), digest


def _split_manifest(
    samples: Sequence[PreparedSample], labels: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_samples = [sample for sample in samples if sample.split == split]
        counts = Counter(sample.label_name for sample in split_samples)
        result[split] = {
            "rows": len(split_samples),
            "unique_fingerprints": len(
                {sample.fingerprint for sample in split_samples}
            ),
            "class_counts": {label: counts[label] for label in labels},
        }
    return result


def _records_sha256(samples: Sequence[PreparedSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(_canonical_json_bytes(sample.to_dict()))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_dataset(
    task: str,
    aegis_splits: Mapping[str, Iterable[Mapping[str, object]]],
    synth_rows: Iterable[Mapping[str, object]],
    *,
    per_label: int,
    seed: int,
) -> DatasetBuild:
    _validate_sampling_inputs(per_label, seed)
    if task == "level1":
        labels = LEVEL1_LABELS
        label_to_id = LEVEL1_LABEL_TO_ID
    elif task == "level2":
        labels = LEVEL2_LABELS
        label_to_id = LEVEL2_LABEL_TO_ID
    else:
        raise ValueError(f"Unknown task: {task}")

    audit = _new_audit()
    candidates = _aegis_candidates(task, aegis_splits, audit)
    candidates.extend(_synth_candidates(task, synth_rows, audit))
    unique_candidates = _deduplicate(candidates, audit)

    train_rows, sampling = _sample_train_rows(
        task,
        unique_candidates,
        labels,
        per_label,
        seed,
        audit,
    )
    selected_rows = train_rows + [
        (candidate, 0)
        for candidate in unique_candidates
        if candidate.split in {"validation", "test"}
    ]
    samples = [
        _prepared_sample(task, candidate, occurrence, label_to_id)
        for candidate, occurrence in selected_rows
    ]
    samples.sort(key=lambda sample: _output_order(task, seed, sample))
    audit["emitted_rows"] = len(samples)

    manifest = {
        "schema_version": 1,
        "task": task,
        "taxonomy_version": TAXONOMY_VERSION,
        "normalization": NORMALIZATION_VERSION,
        "input_field": "prompt",
        "prompt_only": True,
        "seed": seed,
        "split_precedence": list(SPLIT_PRECEDENCE),
        "label2id": dict(label_to_id),
        "sampling": {
            "strategy": (
                "stable-hash-without-replacement"
                if task == "level1"
                else "stable-hash-with-deterministic-oversampling"
            ),
            "train_per_label": per_label,
            "classes": sampling,
        },
        "splits": _split_manifest(samples, labels),
        "audit": {key: audit[key] for key in AUDIT_KEYS},
        "records_sha256": _records_sha256(samples),
    }
    return DatasetBuild(task=task, samples=tuple(samples), manifest=manifest)


def build_level1_dataset(
    aegis_splits: Mapping[str, Iterable[Mapping[str, object]]],
    synth_rows: Iterable[Mapping[str, object]] = (),
    *,
    per_label: int = DEFAULT_LEVEL1_PER_LABEL,
    seed: int = DEFAULT_SEED,
) -> DatasetBuild:
    """Build the binary prompt classifier dataset deterministically."""
    return _build_dataset(
        "level1", aegis_splits, synth_rows, per_label=per_label, seed=seed
    )


def build_level2_dataset(
    aegis_splits: Mapping[str, Iterable[Mapping[str, object]]],
    synth_rows: Iterable[Mapping[str, object]] = (),
    *,
    per_label: int = DEFAULT_LEVEL2_PER_LABEL,
    seed: int = DEFAULT_SEED,
) -> DatasetBuild:
    """Build the legacy nine-class prompt classifier dataset deterministically."""
    return _build_dataset(
        "level2", aegis_splits, synth_rows, per_label=per_label, seed=seed
    )


def prepare_materialized_data(
    output_dir: str | Path,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    downloader: Callable[..., str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Download pinned inputs and write both classifier datasets."""
    contract = load_contract(contract_path)
    _validate_data_contract(contract)
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The prepare command requires huggingface-hub; install requirements.txt"
            ) from exc
        downloader = hf_hub_download

    datasets = contract["datasets"]
    if not isinstance(datasets, Mapping):
        raise DataContractError("Contract datasets must be an object")
    aegis_dataset = datasets["aegis"]
    synth_dataset = datasets["synthetic"]
    if not isinstance(aegis_dataset, Mapping) or not isinstance(synth_dataset, Mapping):
        raise DataContractError("Contract dataset entries must be objects")
    aegis_files = aegis_dataset["files"]
    synth_files = synth_dataset["files"]
    if not isinstance(aegis_files, Mapping) or not isinstance(synth_files, Mapping):
        raise DataContractError("Contract dataset files must be objects")

    aegis_paths: dict[str, Path] = {}
    for split in SPLITS:
        file_spec = aegis_files.get(split)
        if not isinstance(file_spec, Mapping):
            raise DataContractError(f"Missing AEGIS file spec for {split}")
        aegis_paths[split] = _download_contract_file(
            aegis_dataset, file_spec, downloader
        )
    synth_file_spec = synth_files.get("train")
    if not isinstance(synth_file_spec, Mapping):
        raise DataContractError("Missing synthetic train file spec")
    synth_path = _download_contract_file(synth_dataset, synth_file_spec, downloader)

    aegis_rows = {split: load_json_array(path) for split, path in aegis_paths.items()}
    synth_rows = load_jsonl(synth_path)
    data_contract = contract["data"]
    if not isinstance(data_contract, Mapping):
        raise DataContractError("Contract data must be an object")
    seed = data_contract.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DataContractError("Contract data.seed must be an integer")

    level1 = build_level1_dataset(
        aegis_rows,
        synth_rows,
        per_label=_positive_contract_int(data_contract, "binary_train_per_label"),
        seed=seed,
    )
    level2 = build_level2_dataset(
        aegis_rows,
        synth_rows,
        per_label=_positive_contract_int(data_contract, "hazard_train_per_label"),
        seed=seed,
    )
    provenance = {
        "contract_file": Path(contract_path).name,
        "contract_sha256": contract_sha256(contract),
        "datasets": datasets,
    }
    root = Path(output_dir)
    return {
        "level1": materialize_dataset(level1, root / "level1", provenance=provenance),
        "level2": materialize_dataset(level2, root / "level2", provenance=provenance),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="download pinned source data and materialize JSONL"
    )
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "prepare":
        raise AssertionError(f"Unhandled command: {args.command}")
    manifests = prepare_materialized_data(
        args.output_dir,
        contract_path=args.contract,
    )
    summary = {
        task: {
            "output_dir": str(args.output_dir / task),
            "manifest_sha256": manifest["manifest_sha256"],
            "split_rows": {
                split: manifest["splits"][split]["rows"] for split in SPLITS
            },
        }
        for task, manifest in manifests.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
