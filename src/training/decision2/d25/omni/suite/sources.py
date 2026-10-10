"""Pinned sources of the public Vision-board suite (0.3.1) and of the eval image bank.

Every Hub source is fixed to a commit; ``fetch`` downloads it into ``<root>/<key>/`` and writes
``FETCH.json`` with the sha256 and size of every file, checked against the Hub's LFS digests.
Plain-URL pools (COCO, ADE20K) record the digests observed at download time. A source whose
``FETCH.json`` verifies is skipped, so an interrupted fetch resumes where it stopped.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Source:
    key: str
    repo: str = ""
    revision: str = ""
    patterns: tuple[str, ...] = ()
    url: str = ""
    licence: str = ""
    gated: bool = False
    purpose: tuple[str, ...] = ("suite", "bank")
    notes: str = ""
    extra: dict = field(default_factory=dict)


SOURCES: dict[str, Source] = {
    s.key: s
    for s in (
        Source(
            "cvbench",
            "nyu-visionx/CV-Bench",
            "bc284db50d036958861cb60cdd7b77612052ce0d",
            (
                "test_2d.parquet",
                "test_3d.parquet",
                "test_2d.jsonl",
                "test_3d.jsonl",
                "README.md",
            ),
            licence="apache-2.0",
        ),
        Source(
            "blink",
            "BLINK-Benchmark/BLINK",
            "a3666eb249237ba3d5eca8db21176cc47967e040",
            ("*/val-*.parquet", "*/test-*.parquet", "README.md"),
            licence="apache-2.0",
        ),
        Source(
            "realworldqa",
            "xai-org/RealworldQA",
            "17e7f75e092e47169732462ea3cdfebe911105dd",
            ("data/test-*.parquet", "README.md"),
            licence="cc-by-nd-4.0",
        ),
        Source(
            "charxiv",
            "princeton-nlp/CharXiv",
            "f441eb632fc62f6f777830a0f47619e6e86459b0",
            ("val.parquet", "test.parquet", "README.md"),
            licence="cc-by-sa-4.0 (questions); charts belong to their authors; evaluation only",
        ),
        Source(
            "infovqa",
            "lmms-lab/DocVQA",
            "539088ef8a8ada01ac8e2e6d4e372586748a265e",
            (
                "InfographicVQA/validation-*.parquet",
                "InfographicVQA/test-*.parquet",
                "README.md",
            ),
            licence="apache-2.0 (card); InfographicVQA terms of the RRC",
        ),
        Source(
            "mind2web",
            "osunlp/Multimodal-Mind2Web",
            "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889",
            (
                "data/test_task-*.parquet",
                "data/test_website-*.parquet",
                "data/test_domain-*.parquet",
                "README.md",
            ),
            licence="openrail",
        ),
        Source(
            "mind2web_train",
            "osunlp/Multimodal-Mind2Web",
            "1b4c6a8cf9f77b7a5e0d641959935c80c4a05889",
            ("data/train-*.parquet",),
            licence="openrail",
            purpose=("bank",),
        ),
        Source(
            "winoground",
            "facebook/winoground",
            "b400e173549071916ad1b3d449293bc8d8b4b763",
            (
                "data/images.zip",
                "data/examples.jsonl",
                "README.md",
                "license_agreement.txt",
            ),
            licence="Meta Images Research License (evaluation only, never redistribute)",
            gated=True,
        ),
        Source(
            "cord",
            "naver-clova-ix/cord-v2",
            "7f0115a4b758a71d6473b8d085751692da2fef98",
            (
                "data/test-*.parquet",
                "data/validation-*.parquet",
                "data/train-*.parquet",
                "README.md",
            ),
            licence="cc-by-4.0",
        ),
        Source(
            "funsd",
            "nielsr/funsd",
            "7e7eeeedd84ce86540eb83cbbf7c75a3fcc7c7a5",
            ("data/test-*.parquet", "data/train-*.parquet", "README.md"),
            licence="FUNSD research terms (RVL-CDIP subset)",
        ),
        Source(
            "hateful_memes",
            "neuralcatcher/hateful_memes",
            "d201c488dc7024623d1ecbcc987b3f132c4c2e12",
            ("*.jsonl", "img/*", "LICENSE.txt", "README.md"),
            licence="Hateful Memes dataset licence (research, non-transferable)",
            notes="The repo holds 9,664 of the dataset's images; rows without an image are not buildable.",
        ),
        Source(
            "rbench",
            "R-Bench/R-Bench",
            "c0e92a6d90dad980ea51a47af64c12ec67b670ce",
            ("rbench-m_en/*.parquet", "rbench-m_zh/*.parquet", "README.md"),
            licence="apache-2.0",
        ),
        Source(
            "mmmu_pro",
            "MMMU/MMMU_Pro",
            "0d7426df4ccb3d8704a992fe6850abee7c262127",
            (
                "vision/*.parquet",
                "standard (10 options)/*.parquet",
                "standard (4 options)/*.parquet",
                "README.md",
            ),
            licence="apache-2.0",
            notes="2026-10-04 upstream fix touches test_Chemistry_240 and validation_Finance_5 against 563f3e84.",
        ),
        Source(
            "mmmu",
            "MMMU/MMMU",
            "876ce5cb130f7f7e290ce4d9984357737d4db5cf",
            (
                "*/dev-*.parquet",
                "*/validation-*.parquet",
                "*/test-*.parquet",
                "README.md",
            ),
            licence="apache-2.0",
            purpose=("bank",),
        ),
        Source(
            "coco_val2017",
            url="http://images.cocodataset.org/zips/val2017.zip",
            licence="COCO terms (Flickr images, CC licences per image)",
            purpose=("bank",),
        ),
        Source(
            "ade20k",
            url="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            licence="ADE20K terms (research)",
            purpose=("bank",),
            notes="Only images/validation is hashed into the bank.",
        ),
    )
}

SUITE_KEYS = tuple(k for k, s in SOURCES.items() if "suite" in s.purpose)
BANK_KEYS = tuple(SOURCES)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _token() -> str | bool:
    return os.environ.get("HF_TOKEN") or False


def _hub_files(source: Source) -> dict[str, dict]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(
        source.repo, revision=source.revision, files_metadata=True, token=_token()
    )
    files = {}
    for sibling in info.siblings or []:
        if any(fnmatch.fnmatch(sibling.rfilename, p) for p in source.patterns):
            lfs = sibling.lfs
            files[sibling.rfilename] = {
                "size": sibling.size,
                "lfs_sha256": (
                    lfs.get("sha256")
                    if isinstance(lfs, dict)
                    else getattr(lfs, "sha256", None)
                ),
            }
    return files


def _verified(record_path: Path, base: Path) -> bool:
    if not record_path.exists():
        return False
    record = json.loads(record_path.read_text())
    return all(
        (base / f["path"]).exists() and (base / f["path"]).stat().st_size == f["size"]
        for f in record["files"]
    )


def fetch(key: str, root: str | Path, workers: int = 8) -> dict:
    """Download one pinned source into ``root/key`` and return its FETCH record."""
    source = SOURCES[key]
    base = Path(root) / key
    record_path = base / "FETCH.json"
    if _verified(record_path, base):
        return json.loads(record_path.read_text())
    base.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if source.url:
        name = source.url.rsplit("/", 1)[1]
        target = base / name
        if not target.exists():
            part = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(source.url, timeout=120) as response, open(
                part, "wb"
            ) as out:
                shutil.copyfileobj(response, out, 1 << 22)
            part.replace(target)
        files = [
            {"path": name, "size": target.stat().st_size, "sha256": sha256_file(target)}
        ]
        record = {"key": key, "url": source.url, "files": files}
    else:
        from huggingface_hub import snapshot_download

        expected = _hub_files(source)
        if not expected:
            raise RuntimeError(
                f"{key}: no files match {source.patterns} at {source.revision}"
            )
        snapshot_download(
            source.repo,
            repo_type="dataset",
            revision=source.revision,
            allow_patterns=list(source.patterns),
            local_dir=base,
            token=_token(),
            max_workers=workers,
        )
        files = []
        for name, meta in sorted(expected.items()):
            path = base / name
            digest = sha256_file(path)
            if meta["lfs_sha256"] and digest != meta["lfs_sha256"]:
                raise RuntimeError(f"{key}: sha256 mismatch for {name}")
            if meta["size"] is not None and path.stat().st_size != meta["size"]:
                raise RuntimeError(f"{key}: size mismatch for {name}")
            files.append({"path": name, "size": path.stat().st_size, "sha256": digest})
        record = {
            "key": key,
            "repo": source.repo,
            "revision": source.revision,
            "files": files,
        }
    record.update(
        licence=source.licence,
        gated=source.gated,
        fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seconds=round(time.time() - started, 1),
    )
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    return record


def record(key: str, root: str | Path) -> dict:
    """The FETCH record of a fetched source (raises if the source was not fetched)."""
    path = Path(root) / key / "FETCH.json"
    if not path.exists():
        raise FileNotFoundError(f"source {key} not fetched under {root}")
    return json.loads(path.read_text())


def file_sha256(key: str, root: str | Path, name: str) -> str:
    for f in record(key, root)["files"]:
        if f["path"] == name:
            return f["sha256"]
    raise KeyError(f"{key}: {name} not in FETCH.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--only", default="", help="comma-separated source keys (default: all)"
    )
    parser.add_argument("--purpose", choices=("suite", "bank", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    keys = [k for k in args.only.split(",") if k] or [
        k
        for k, s in SOURCES.items()
        if args.purpose == "all" or args.purpose in s.purpose
    ]
    failures = {}
    for key in keys:
        try:
            rec = fetch(key, args.root, workers=args.workers)
            total = sum(f["size"] for f in rec["files"])
            print(
                json.dumps(
                    {
                        "key": key,
                        "files": len(rec["files"]),
                        "bytes": total,
                        "seconds": rec.get("seconds"),
                    }
                ),
                flush=True,
            )
        except (
            Exception
        ) as exc:  # keep going: one unreachable pool must not block the rest
            failures[key] = f"{type(exc).__name__}: {exc}"
            print(json.dumps({"key": key, "error": failures[key]}), flush=True)
    if failures:
        raise SystemExit(f"fetch failed for {sorted(failures)}")


if __name__ == "__main__":
    main()
