"""Turn fetched HT-DEV raw artefacts into row files that `independence rows` can read.

    python3 -m v2.eval.htdev_iso.prepare --spec sources.json --iso <iso-dir> [--source KEY ...]

<iso>/raw/<key>/ (from fetch.sh) becomes <iso>/sources/<key>/ (mode 700):

- HF snapshots: every .jsonl/.json/.csv/.tsv/.parquet data file is copied as is
  (dataset_info(s).json skipped).
- Archives: members under the spec's `paths` are taken from the tarball (top directory
  stripped) or zip and converted per `convert`; converted files keep their relative path
  with the suffix `.jsonl`, one output row per input record (see the converters).

Writes <iso>/SOURCES-MANIFEST.json (mode 600): per key the origin, full revision, raw
files and source files with sha256, bytes, rows and rows with a >= 20-character string
leaf (the rows `independence rows` yields). Counts only; no text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator
from xml.etree import ElementTree

from v2.eval.sealed.independence import DATA_SUFFIXES, MIN_CHARS, data_rows, strings

SKIP_NAMES = ("dataset_info.json", "dataset_infos.json")
Rows = Iterator[dict[str, Any]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def tsv_header(data: bytes) -> Rows:
    csv.field_size_limit(1 << 30)
    reader = csv.reader(
        io.StringIO(_text(data)), delimiter="\t", quoting=csv.QUOTE_NONE
    )
    header = next(reader, None) or []
    for record in reader:
        if record:
            yield {
                (header[i] if i < len(header) else f"col{i}"): value
                for i, value in enumerate(record)
            }


def tsv_noheader(data: bytes) -> Rows:
    for line in _text(data).splitlines():
        if line.strip():
            text, _, labels = line.partition("\t")
            yield {"text": text, "labels": labels}


def diplomacy_messages(data: bytes) -> Rows:
    for number, line in enumerate(_text(data).splitlines()):
        if not line.strip():
            continue
        dialogue = json.loads(line)
        for index, message in enumerate(dialogue.get("messages", [])):
            yield {
                "dialogue_row": number,
                "message_index": index,
                "game_id": dialogue.get("game_id"),
                "text": message,
            }


def hatexplain(data: bytes) -> Rows:
    for post_id, post in json.loads(_text(data)).items():
        yield {"post_id": post_id, "text": " ".join(post.get("post_tokens", []))}


def swords(data: bytes) -> Rows:
    parsed = json.loads(gzip.decompress(data).decode("utf-8"))
    for context_id, context in sorted(parsed.get("contexts", {}).items()):
        yield {"context_id": context_id, "context": context.get("context", "")}


def _xml_articles(data: bytes) -> Rows:
    for _, element in ElementTree.iterparse(io.BytesIO(data), events=("end",)):
        if element.tag != "article":
            continue
        attributes = dict(element.attrib)
        paragraphs = [
            "".join(p.itertext()).strip() for p in element.iter() if p.tag == "p"
        ]
        if paragraphs or len(element):
            attributes["text"] = "\n".join(p for p in paragraphs if p)
        yield attributes
        element.clear()


CONVERTERS: dict[str, Callable[[bytes], Rows]] = {
    "tsv_header": tsv_header,
    "tsv_noheader": tsv_noheader,
    "diplomacy_messages": diplomacy_messages,
    "hatexplain": hatexplain,
    "swords": swords,
}


def _write_rows(target: Path, rows: Rows) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _selected(name: str, paths: list[str]) -> bool:
    return any(name == p or (p.endswith("/") and name.startswith(p)) for p in paths)


def _wictsv(members: dict[str, bytes], out: Path) -> None:
    splits: dict[str, dict[str, list[str]]] = {}
    for name, data in members.items():
        directory, _, file_name = name.rpartition("/")
        if not file_name.endswith(".txt") or "_" not in file_name:
            continue
        kind = file_name.rsplit("_", 1)[1][: -len(".txt")]
        splits.setdefault(directory, {})[kind] = _text(data).splitlines()
    for directory, parts in sorted(splits.items()):
        examples = parts.get("examples", [])
        rows = []
        for index, line in enumerate(examples):
            word, position, sentence = (line.split("\t") + ["", "", ""])[:3]
            row = {"target": word, "position": position, "context": sentence}
            for kind in ("definitions", "hypernyms", "labels", "domains"):
                if kind in parts and index < len(parts[kind]):
                    row[kind] = parts[kind][index]
            rows.append(row)
        _write_rows(out / directory / "examples.jsonl", iter(rows))


def _archive_members(raw: Path, paths: list[str]) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for archive in sorted(raw.iterdir()):
        if archive.name.endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name.split("/", 1)[1] if "/" in member.name else ""
                    if paths and not _selected(name, paths):
                        continue
                    stream = tar.extractfile(member)
                    if stream is not None:
                        members[name] = stream.read()
        elif archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zipped:
                for name in zipped.namelist():
                    if not name.endswith("/"):
                        members[f"{archive.stem}/{name}"] = zipped.read(name)
    return members


def prepare_key(key: str, spec: dict[str, Any], raw: Path, out: Path) -> None:
    out.mkdir(parents=True, mode=0o700)
    origin, convert = spec["origin"], spec.get("convert", "copy")
    if origin == "hf":
        for path in sorted(raw.rglob("*")):
            if (
                path.is_file()
                and path.name.endswith(DATA_SUFFIXES)
                and path.name not in SKIP_NAMES
                and ".cache" not in path.parts
            ):
                target = out / path.relative_to(raw)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        return
    members = _archive_members(raw, spec.get("paths", []))
    if convert == "wictsv":
        _wictsv(members, out)
        return
    for name, data in sorted(members.items()):
        base = name[: -len(".gz")] if name.endswith(".json.gz") else name
        if convert == "hyperpartisan_xml":
            if name.endswith(".xml"):
                _write_rows(
                    out / (name[: -len(".xml")] + ".jsonl"), _xml_articles(data)
                )
        elif convert in CONVERTERS:
            if base.endswith((".csv", ".tsv", ".jsonl", ".json")):
                stem = base.rsplit(".", 1)[0]
                _write_rows(out / (stem + ".jsonl"), CONVERTERS[convert](data))
        elif name.endswith(DATA_SUFFIXES) and name.rsplit("/", 1)[-1] not in SKIP_NAMES:
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def count_rows(path: Path) -> tuple[int, int]:
    total = with_text = 0
    for row in data_rows(path):
        total += 1
        with_text += any(len(t.strip()) >= MIN_CHARS for t in strings(row))
    return total, with_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--iso", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args(argv)
    os.umask(0o077)
    sources = json.loads(args.spec.read_text(encoding="utf-8"))["sources"]
    manifest: dict[str, Any] = {"schema": "htdev-iso-sources-manifest/1", "sources": {}}
    for key in args.source or list(sources):
        spec = sources[key]
        raw, out = args.iso / "raw" / key, args.iso / "sources" / key
        entry: dict[str, Any] = {
            k: spec.get(k) for k in ("task", "role", "origin", "repo", "revision")
        }
        if not raw.is_dir():
            entry["status"] = "not fetched"
            manifest["sources"][key] = entry
            continue
        if not out.exists():
            prepare_key(key, spec, raw, out)
        entry["raw"] = [
            {
                "path": str(p.relative_to(raw)),
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in sorted(raw.rglob("*"))
            if p.is_file()
        ]
        files = []
        for path in sorted(out.rglob("*")):
            if path.is_file():
                total, with_text = count_rows(path)
                files.append(
                    {
                        "path": str(path.relative_to(out)),
                        "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                        "rows": total,
                        "rows_with_text": with_text,
                    }
                )
        entry["files"] = files
        entry["rows_with_text"] = sum(f["rows_with_text"] for f in files)
        entry["status"] = "ok" if entry["rows_with_text"] > 0 else "EMPTY"
        manifest["sources"][key] = entry
        print(key, entry["status"], len(files), entry["rows_with_text"], flush=True)
    target = args.iso / "SOURCES-MANIFEST.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
