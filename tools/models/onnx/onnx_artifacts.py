"""Public ONNX artifact sanitation and reproducible, confined file digests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot_sha256(directory: Path) -> dict[str, str]:
    """Hash native weights, tokenizer and reader configuration, not reports."""
    patterns = (
        "config.json",
        "matryoshka_config.json",
        "modules.json",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
        "*.safetensors",
        "*.safetensors.index.json",
        "pytorch_model*.bin",
        "*.bin.index.json",
        "classification_heads.pt",
        "tokenizer*",
        "vocab*",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "1_Pooling/config.json",
    )
    files = {
        path
        for pattern in patterns
        for path in directory.glob(pattern)
        if path.is_file()
    }
    return {str(path.relative_to(directory)): sha256(path) for path in sorted(files)}


def prepare_export_directory(directory: Path, identity: dict) -> None:
    """Bind all precisions/exits to one source before changing output files."""
    manifest = directory / "source_identity.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(
                "Export source or exit inventory changed; use a separate directory"
            )
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(
            "Existing export has no source identity; use a separate directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(identity, indent=2) + "\n")


def strip_debug_annotations(message) -> None:
    """Remove recursive exporter metadata, preserving external tensor locations."""
    if hasattr(message, "doc_string"):
        message.doc_string = ""
    if hasattr(message, "metadata_props"):
        del message.metadata_props[:]
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        for child in value if field.is_repeated else (value,):
            strip_debug_annotations(child)


def external_data_sha256(message, graph_path: Path) -> dict[str, str]:
    """Hash every referenced tensor file, including subgraphs and attributes.

    Absolute paths, traversal, Windows drive paths and symlink escapes are
    rejected before any external file is read. Load the model with
    ``load_external_data=False`` to retain the locations for this check.
    """
    locations = set()

    def visit(item):
        if hasattr(item, "external_data"):
            entries = {entry.key: entry.value for entry in item.external_data}
            if "location" in entries:
                locations.add(entries["location"])
        for field, value in item.ListFields():
            if field.message_type is not None:
                for child in value if field.is_repeated else (value,):
                    visit(child)

    visit(message)
    root = Path(graph_path).parent.resolve()
    result = {}
    for name in sorted(locations):
        location = Path(name)
        windows = PureWindowsPath(name)
        resolved = (root / location).resolve()
        if (
            not name
            or location.is_absolute()
            or windows.drive
            or ".." in location.parts
            or ".." in windows.parts
            or not resolved.is_relative_to(root)
        ):
            raise ValueError("External data must stay inside its artifact directory")
        result[name] = sha256(resolved)
    return result
