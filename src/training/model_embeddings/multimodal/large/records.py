"""Manifest records and Sentence Transformers dataset conversion."""

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, Features, IterableDataset, Value

SUPPORTED_MODALITIES = {"text", "image", "audio"}


@dataclass
class PairItem:
    modality: str
    value: Any


@dataclass
class TrainRecord:
    query: PairItem
    positive: PairItem
    negative: PairItem | None = None


def _parse_item(obj: Any, prefix: str) -> PairItem:
    if isinstance(obj, dict):
        modality = obj.get("type")
        value = obj.get("value")
    else:
        modality = None
        value = None

    if not modality or not value:
        raise ValueError(f"{prefix} must include type/value")
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported modality '{modality}' in {prefix}")
    return PairItem(modality=modality, value=value)


def parse_record(raw: dict[str, Any]) -> TrainRecord:
    if "query" in raw and "positive" in raw:
        query = _parse_item(raw["query"], "query")
        positive = _parse_item(raw["positive"], "positive")
        negative = (
            _parse_item(raw["negative"], "negative") if raw.get("negative") else None
        )
        return TrainRecord(query=query, positive=positive, negative=negative)

    # Compatibility with common pair formats in existing repos
    if "texts_a" in raw and "texts_b" in raw:
        query = PairItem("text", raw["texts_a"])
        positive = PairItem("text", raw["texts_b"])
        return TrainRecord(query=query, positive=positive)

    if "image_path" in raw and "caption" in raw:
        query = PairItem("image", raw["image_path"])
        positive = PairItem("text", raw["caption"])
        return TrainRecord(query=query, positive=positive)

    if "audio_path" in raw and "caption" in raw:
        query = PairItem("audio", raw["audio_path"])
        positive = PairItem("text", raw["caption"])
        return TrainRecord(query=query, positive=positive)

    raise ValueError("Record does not match supported schemas")


class JsonlManifestDataset:
    def __init__(
        self,
        manifest_path: str,
        image_root: str | None = None,
        audio_root: str | None = None,
        allow_missing_negative: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.image_root = image_root
        self.audio_root = audio_root
        self.allow_missing_negative = allow_missing_negative
        self.records = list(
            iter_manifest_records(
                manifest_path=self.manifest_path,
                image_root=self.image_root,
                audio_root=self.audio_root,
                allow_missing_negative=self.allow_missing_negative,
            )
        )
        if not self.records:
            raise ValueError(f"No records loaded from {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TrainRecord:
        return self.records[idx]


def _process_shard() -> tuple[int, int]:
    rank = int(
        os.environ.get("ACCELERATE_PROCESS_INDEX") or os.environ.get("RANK") or 0
    )
    world_size = int(
        os.environ.get("WORLD_SIZE") or os.environ.get("ACCELERATE_NUM_PROCESSES") or 1
    )
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return rank, max(world_size, 1)

    total_shards = max(world_size, 1) * worker_info.num_workers
    shard_id = rank * worker_info.num_workers + worker_info.id
    return shard_id, max(total_shards, 1)


def iter_sentence_transformers_rows(
    manifest_path: str,
    image_root: str | None,
    audio_root: str | None,
    allow_missing_negative: bool,
    allowed_modalities: list[str] | None,
    query_modalities: list[str] | None,
    positive_modalities: list[str] | None,
    negative_modalities: list[str] | None,
    use_negative_column: bool,
):
    allowed = set(allowed_modalities or [])
    allowed_query = set(query_modalities or [])
    allowed_positive = set(positive_modalities or [])
    allowed_negative = set(negative_modalities or [])
    shard_id, total_shards = _process_shard()
    matched_index = 0

    for record in iter_manifest_records(
        manifest_path=manifest_path,
        image_root=image_root,
        audio_root=audio_root,
        allow_missing_negative=allow_missing_negative,
    ):
        if not record_matches_filters(
            record,
            allowed=allowed,
            allowed_query=allowed_query,
            allowed_positive=allowed_positive,
            allowed_negative=allowed_negative,
        ):
            continue

        if matched_index % total_shards == shard_id:
            yield record_to_sentence_transformers_row(
                record, include_negative=use_negative_column
            )
        matched_index += 1


def collate_records(batch: list[TrainRecord]) -> dict[str, list[PairItem]]:
    return {
        "query": [r.query for r in batch],
        "positive": [r.positive for r in batch],
        "negative": [r.negative for r in batch],
    }


def sentence_transformers_input(item: PairItem) -> Any:
    payload: dict[str, Any] = {}
    if item.modality == "text":
        payload["text"] = item.value
        return payload
    if item.modality == "image":
        payload["image"] = item.value
        return payload
    if item.modality == "audio":
        payload["audio"] = item.value
        return payload
    return item.value


def resolve_media(
    item: PairItem, image_root: str | None, audio_root: str | None
) -> PairItem:
    if item.modality == "image" and image_root and not os.path.isabs(item.value):
        return PairItem(item.modality, os.path.join(image_root, item.value))
    if item.modality == "audio" and audio_root and not os.path.isabs(item.value):
        return PairItem(item.modality, os.path.join(audio_root, item.value))
    return item


def iter_manifest_records(
    manifest_path: str,
    image_root: str | None = None,
    audio_root: str | None = None,
    allow_missing_negative: bool = True,
) -> Iterable[TrainRecord]:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            raw = json.loads(line)
            record = parse_record(raw)
            record = TrainRecord(
                query=resolve_media(record.query, image_root, audio_root),
                positive=resolve_media(record.positive, image_root, audio_root),
                negative=(
                    resolve_media(record.negative, image_root, audio_root)
                    if record.negative
                    else None
                ),
            )
            if record.negative is None and not allow_missing_negative:
                raise ValueError(f"Missing negative at line {line_no}")
            yield record


def record_matches_filters(
    record: TrainRecord,
    allowed: set[str],
    allowed_query: set[str],
    allowed_positive: set[str],
    allowed_negative: set[str],
) -> bool:
    record_modalities = {record.query.modality, record.positive.modality}
    if record.negative is not None:
        record_modalities.add(record.negative.modality)
    if allowed and not record_modalities.issubset(allowed):
        return False
    if allowed_query and record.query.modality not in allowed_query:
        return False
    if allowed_positive and record.positive.modality not in allowed_positive:
        return False
    return not (
        record.negative is not None
        and allowed_negative
        and record.negative.modality not in allowed_negative
    )


def record_to_sentence_transformers_row(
    record: TrainRecord, include_negative: bool
) -> dict[str, Any]:
    row = {
        "query": sentence_transformers_input(record.query),
        "positive": sentence_transformers_input(record.positive),
    }
    if include_negative and record.negative is not None:
        row["negative_0"] = sentence_transformers_input(record.negative)
    return row


def summarize_manifest_records(
    manifest_path: str,
    image_root: str | None = None,
    audio_root: str | None = None,
    allow_missing_negative: bool = True,
    allowed_modalities: list[str] | None = None,
    query_modalities: list[str] | None = None,
    positive_modalities: list[str] | None = None,
    negative_modalities: list[str] | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    modalities = set()
    negatives_present = 0
    negatives_missing = 0
    skipped_rows = 0
    num_rows = 0
    allowed = set(allowed_modalities or [])
    allowed_query = set(query_modalities or [])
    allowed_positive = set(positive_modalities or [])
    allowed_negative = set(negative_modalities or [])

    for record in iter_manifest_records(
        manifest_path=manifest_path,
        image_root=image_root,
        audio_root=audio_root,
        allow_missing_negative=allow_missing_negative,
    ):
        if not record_matches_filters(
            record,
            allowed=allowed,
            allowed_query=allowed_query,
            allowed_positive=allowed_positive,
            allowed_negative=allowed_negative,
        ):
            skipped_rows += 1
            continue

        modalities.add(record.query.modality)
        modalities.add(record.positive.modality)
        if record.negative is not None:
            modalities.add(record.negative.modality)
            negatives_present += 1
        else:
            negatives_missing += 1
        num_rows += 1
        if max_records is not None and num_rows >= max_records:
            break

    if num_rows == 0:
        raise ValueError(f"No records loaded from {manifest_path}")

    return {
        "modalities": sorted(modalities),
        "num_rows": num_rows,
        "has_uniform_negatives": negatives_present > 0 and negatives_missing == 0,
        "num_negatives_present": negatives_present,
        "num_negatives_missing": negatives_missing,
        "skipped_rows": skipped_rows,
    }


def manifest_to_sentence_transformers_dataset(
    manifest_path: str,
    image_root: str | None = None,
    audio_root: str | None = None,
    allow_missing_negative: bool = True,
    allowed_modalities: list[str] | None = None,
    query_modalities: list[str] | None = None,
    positive_modalities: list[str] | None = None,
    negative_modalities: list[str] | None = None,
    as_iterable: bool = False,
    max_records: int | None = None,
) -> tuple[Dataset | IterableDataset, dict[str, Any]]:
    info = summarize_manifest_records(
        manifest_path=manifest_path,
        image_root=image_root,
        audio_root=audio_root,
        allow_missing_negative=allow_missing_negative,
        allowed_modalities=allowed_modalities,
        query_modalities=query_modalities,
        positive_modalities=positive_modalities,
        negative_modalities=negative_modalities,
        max_records=max_records,
    )

    dataset_out: Dataset | IterableDataset
    if as_iterable:
        column_names = ["query", "positive"]
        if info["has_uniform_negatives"]:
            column_names.append("negative_0")
        dataset_out = IterableDataset.from_generator(
            iter_sentence_transformers_rows,
            features=Features({key: Value("null") for key in column_names}),
            gen_kwargs={
                "manifest_path": manifest_path,
                "image_root": image_root,
                "audio_root": audio_root,
                "allow_missing_negative": allow_missing_negative,
                "allowed_modalities": allowed_modalities,
                "query_modalities": query_modalities,
                "positive_modalities": positive_modalities,
                "negative_modalities": negative_modalities,
                "use_negative_column": info["has_uniform_negatives"],
            },
        )
    else:
        dataset = JsonlManifestDataset(
            manifest_path=manifest_path,
            image_root=image_root,
            audio_root=audio_root,
            allow_missing_negative=allow_missing_negative,
        )
        allowed = set(allowed_modalities or [])
        allowed_query = set(query_modalities or [])
        allowed_positive = set(positive_modalities or [])
        allowed_negative = set(negative_modalities or [])
        rows: list[dict[str, Any]] = []
        for record in dataset.records:
            if not record_matches_filters(
                record,
                allowed=allowed,
                allowed_query=allowed_query,
                allowed_positive=allowed_positive,
                allowed_negative=allowed_negative,
            ):
                continue
            rows.append(
                record_to_sentence_transformers_row(
                    record, include_negative=info["has_uniform_negatives"]
                )
            )
            if max_records is not None and len(rows) >= max_records:
                break
        dataset_out = Dataset.from_list(rows)

    return dataset_out, info
