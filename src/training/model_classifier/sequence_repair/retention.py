"""Optional, identity-bound TRAIN replay targets and categorical retention loss."""

# Torch stays optional for cache identity and data checks.
# ruff: noqa: PLC0415

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from pathlib import Path

_LOGIT_RANK = 2
_MINIMUM_LABELS = 2


def masked_teacher_kl(
    logits, teacher_logits, eligible, *, examples_per_step, temperature=2.0
):
    """Return detached T² KL(teacher || student), summed over a logical batch."""
    import torch
    from torch.nn import functional

    if type(examples_per_step) is not int or examples_per_step <= 0:
        raise ValueError("Logical example count must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Retention temperature must be finite and positive")
    if (
        logits.ndim != _LOGIT_RANK
        or logits.shape[1] < _MINIMUM_LABELS
        or logits.shape[0] > examples_per_step
        or teacher_logits.shape != logits.shape
        or eligible.shape != logits.shape[:1]
        or eligible.dtype != torch.bool
        or not logits.is_floating_point()
        or not teacher_logits.is_floating_point()
        or teacher_logits.device != logits.device
        or eligible.device != logits.device
        or not torch.isfinite(logits).all()
    ):
        raise ValueError(
            "Retention logits and mask must have compatible finite geometry"
        )
    if not eligible.any():
        # Connected zero without summing possibly huge finite masked logits.
        return logits.float()[:0].sum()
    student = logits[eligible].float() / temperature
    teacher = teacher_logits.detach()[eligible].float() / temperature
    if not torch.isfinite(teacher).all():
        raise ValueError("Eligible teacher logits must be finite")
    student_logp = functional.log_softmax(student, dim=-1)
    teacher_logp = functional.log_softmax(teacher, dim=-1)
    loss = functional.kl_div(
        student_logp, teacher_logp, log_target=True, reduction="sum"
    ) * (temperature**2 / examples_per_step)
    if not torch.isfinite(loss):
        raise ValueError("Retention loss must be finite")
    return loss


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest_map(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("Artifact receipts must contain file digests")
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(
                "Artifact receipt contains an invalid relative name or SHA"
            )


def tokenizer_file_receipt(base):
    """Bind the complete local fast-tokenizer files used by ModernBERT."""
    base = Path(base)
    names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "chat_template.jinja",
    }
    names.update(
        str(path.relative_to(base))
        for path in (base / "chat_templates").glob("*.jinja")
    )
    result = {
        name: _sha(base / name) for name in sorted(names) if (base / name).is_file()
    }
    if not {"tokenizer.json", "tokenizer_config.json"} <= result.keys():
        raise ValueError("Retention requires the complete local fast tokenizer")
    return result


def row_identity(row, encoded):
    """Bind raw text and every unpadded model input, without storing text in targets."""
    if any(
        not isinstance(row.get(key), str) or not row[key]
        for key in ("id", "group_id", "text", "label")
    ):
        raise ValueError("Retention rows require string ID, group, text and label")
    if (
        not isinstance(encoded, Mapping)
        or not {"input_ids", "attention_mask"} <= encoded.keys()
    ):
        raise ValueError("Retention requires complete unpadded tokenizer inputs")
    encoded = dict(encoded)
    if not encoded.keys() <= {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    }:
        raise ValueError("Unexpected tokenizer input in retention identity")
    if not isinstance(encoded["input_ids"], list):
        raise ValueError("Retention input IDs must be an unpadded integer list")
    size = len(encoded["input_ids"])
    if (
        not size
        or any(
            not isinstance(values, list)
            or len(values) != size
            or any(type(value) is not int or value < 0 for value in values)
            for values in encoded.values()
        )
        or any(value != 1 for value in encoded["attention_mask"])
    ):
        raise ValueError("Retention inputs must be complete unpadded integer sequences")
    token_bytes = json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()
    return {
        "id": row["id"],
        "group_id": row["group_id"],
        "label": row["label"],
        "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
        "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
    }


def _labels(value):
    if (
        not isinstance(value, dict)
        or len(value) < _MINIMUM_LABELS
        or any(
            not isinstance(name, str) or not name or type(index) is not int
            for name, index in value.items()
        )
        or sorted(value.values()) != list(range(len(value)))
    ):
        raise ValueError("Retention labels must be a contiguous bijection")


def _record(record, labels):
    expected = {"id", "group_id", "label", "text_sha256", "tokens_sha256", "logits"}
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError("Retention target fields differ from the schema")
    if any(
        not isinstance(record[key], str) or not record[key]
        for key in ("id", "group_id", "label")
    ):
        raise ValueError("Invalid retention target identity")
    _digest_map({key: record[key] for key in ("text_sha256", "tokens_sha256")})
    values = record["logits"]
    if (
        record["label"] not in labels
        or not isinstance(values, list)
        or len(values) != len(labels)
    ):
        raise ValueError("Retention logits differ from the label geometry")
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Cached teacher logits must be finite float32 values")
        try:
            exact = struct.unpack("f", struct.pack("f", value))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError("Cached teacher logit exceeds float32") from error
        if exact != value:
            raise ValueError("Cached teacher logit was not serialized from float32")


def _metadata(initialization, contract_sha256, tokenizer_files, label_to_id):
    if not isinstance(initialization, dict) or set(initialization) != {"base_files"}:
        raise ValueError("Retention requires an exact complete-checkpoint receipt")
    _digest_map(initialization["base_files"])
    _digest_map({"contract": contract_sha256})
    _digest_map(tokenizer_files)
    _labels(label_to_id)
    for name, digest in tokenizer_files.items():
        if (
            name in initialization["base_files"]
            and initialization["base_files"][name] != digest
        ):
            raise ValueError("Tokenizer and initialization receipts disagree")


def write_retention_cache(
    destination,
    *,
    initialization,
    contract_sha256,
    tokenizer_files,
    label_to_id,
    records,
):
    """Write a new portable cache; the loader separately verifies admitted TRAIN scope."""
    _metadata(initialization, contract_sha256, tokenizer_files, label_to_id)
    records = list(records)
    for record in records:
        _record(record, label_to_id)
    records.sort(key=lambda record: record["id"])
    ids = [record["id"] for record in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Retention targets must be nonempty with unique IDs")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    targets = destination / "targets.jsonl"
    with targets.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    manifest = {
        "version": 1,
        "split": "train",
        "dtype": "float32",
        "initialization": initialization,
        "contract_sha256": contract_sha256,
        "tokenizer_files": tokenizer_files,
        "label_to_id": label_to_id,
        "replay_ids": ids,
        "rows": len(records),
        "records_file": targets.name,
        "records_sha256": _sha(targets),
    }
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


class RetentionCache:
    """Targets for explicitly designated old TRAIN rows; unknown rows are rejected."""

    @classmethod
    def load(
        cls,
        path,
        *,
        initialization,
        contract_sha256,
        tokenizer_files,
        train_rows,
        tokenized_rows,
        label_to_id,
        replay_ids,
    ):
        _metadata(initialization, contract_sha256, tokenizer_files, label_to_id)
        path = Path(path)
        raw = path.read_bytes()
        manifest = json.loads(raw)
        expected = {
            "version",
            "split",
            "dtype",
            "initialization",
            "contract_sha256",
            "tokenizer_files",
            "label_to_id",
            "replay_ids",
            "rows",
            "records_file",
            "records_sha256",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected:
            raise ValueError("Retention manifest fields differ from the schema")
        if (
            type(manifest["version"]) is not int
            or manifest["version"] != 1
            or manifest["split"] != "train"
            or manifest["dtype"] != "float32"
        ):
            raise ValueError("Retention requires version1 float32 TRAIN targets")
        for key, value in (
            ("initialization", initialization),
            ("contract_sha256", contract_sha256),
            ("tokenizer_files", tokenizer_files),
            ("label_to_id", label_to_id),
        ):
            if manifest[key] != value:
                raise ValueError(f"Retention cache differs from the actual {key}")
        identities, marked = {}, set()
        for row in train_rows:
            if row.get("split", "train") != "train":
                raise ValueError(
                    "Retention targets cannot include development or test rows"
                )
            if type(row.get("retention_replay", False)) is not bool:
                raise ValueError("TRAIN retention_replay designation must be boolean")
            if row["id"] in identities or row["id"] not in tokenized_rows:
                raise ValueError("Missing tokens or duplicate TRAIN retention ID")
            identities[row["id"]] = row_identity(row, tokenized_rows[row["id"]])
            if row["label"] not in label_to_id:
                raise ValueError("Unknown TRAIN retention label")
            if row.get("retention_replay", False):
                marked.add(row["id"])
        replay_ids = list(replay_ids)
        if (
            not marked
            or len(replay_ids) != len(set(replay_ids))
            or set(replay_ids) != marked
            or manifest["replay_ids"] != sorted(marked)
        ):
            raise ValueError("Cache must cover exactly the designated old TRAIN subset")
        filename = manifest["records_file"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in {"", ".", ".."}
        ):
            raise ValueError("Retention records must use a relative filename")
        targets = path.parent / filename
        data = targets.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["records_sha256"]:
            raise ValueError("Retention target bytes changed")
        records = [json.loads(line) for line in data.splitlines() if line.strip()]
        if (
            type(manifest["rows"]) is not int
            or len(records) != manifest["rows"]
            or len(records) != len(marked)
        ):
            raise ValueError("Retention target coverage is incomplete")
        cached = {}
        for record in records:
            _record(record, label_to_id)
            identity = {key: value for key, value in record.items() if key != "logits"}
            if (
                record["id"] not in marked
                or record["id"] in cached
                or identity != identities[record["id"]]
            ):
                raise ValueError(
                    "Retention target does not match its old TRAIN identity"
                )
            values = record["logits"]
            maximum = max(values)
            eligible = (
                values.count(maximum) == 1
                and values.index(maximum) == label_to_id[record["label"]]
            )
            cached[record["id"]] = (values, eligible)
        result = cls()
        result._identities, result._cached, result._width = (
            identities,
            cached,
            len(label_to_id),
        )
        result.receipt = {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "records_sha256": manifest["records_sha256"],
            "replay_rows": len(cached),
            "eligible_rows": sum(eligible for _, eligible in cached.values()),
        }
        return result

    def targets(self, rows, device):
        import torch

        values, masks = [], []
        for row in rows:
            identity = self._identities.get(row["id"])
            if (
                identity is None
                or any(row[key] != identity[key] for key in ("id", "group_id", "label"))
                or hashlib.sha256(row["text"].encode()).hexdigest()
                != identity["text_sha256"]
            ):
                raise ValueError(
                    "Retention batch is not part of the validated TRAIN corpus"
                )
            logits, eligible = self._cached.get(row["id"], ([0.0] * self._width, False))
            values.append(logits)
            masks.append(eligible)
        return (
            torch.tensor(values, dtype=torch.float32, device=device).reshape(
                len(rows), self._width
            ),
            torch.tensor(masks, dtype=torch.bool, device=device),
        )
