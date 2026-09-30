"""EXPERIMENTAL sr-bench-nano freezing and materialization from pinned sources."""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import nano
from .canonical import digest
from .source_records import normalize_records
from .sources import HF_SOURCES, _stratified_order, _write_dataset, read_records

STRATA = {
    "mmlu-pro": "category",
    "simpleqa-verified": "topic",
    "gpqa-diamond": "Subdomain",
    "livecodebench": "difficulty",
    "hle": "category",
}
POPULATIONS = {
    "mmlu-pro": "all test tasks",
    "simpleqa-verified": "all tasks",
    "gpqa-diamond": "all diamond tasks",
    "livecodebench": "release v5+v6 additions (test5.jsonl, test6.jsonl)",
    "hle": (
        "text-only (no image), answer_type=exactMatch, reference numeric "
        "(integer, decimal or a/b) or a plain string of <=4 words and <=40 characters"
    ),
}


def _files(benchmark):
    repo, revision, files = HF_SOURCES[benchmark]
    return repo, revision, nano.LCB_FILES if benchmark == "livecodebench" else files


def _fetch(benchmark, store, source_dir):
    """Return verified local paths for the pinned files of one benchmark."""
    repo, revision, files = _files(benchmark)
    paths = []
    for filename in files:
        if source_dir is not None:
            path = Path(source_dir).expanduser() / benchmark / Path(filename).name
            if not path.is_file():
                raise ValueError(f"Missing pinned source file {path}")
        else:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            path = Path(
                hf_hub_download(
                    repo,
                    filename,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=str(Path(store).expanduser().resolve() / "cache"),
                )
            )
        paths.append((filename, path))
    return repo, revision, paths


def _population(benchmark, store, source_dir):
    repo, revision, paths = _fetch(benchmark, store, source_dir)
    rows, files = [], []
    for filename, path in paths:
        rows.extend(read_records(path))
        files.append(
            {"name": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    cases = normalize_records(benchmark, rows, nano.SEED)
    if benchmark == "hle":
        cases = nano.render_hle(cases)
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate source task IDs")
    source = {
        "repository": "https://huggingface.co/datasets/" + repo,
        "revision": revision,
        "files": files,
    }
    return cases, source


def freeze(store, source_dir=None):
    """Build the frozen id-list document; stores ids and hashes, never text."""
    benchmarks = {}
    for benchmark in nano.BENCHMARKS:
        cases, source = _population(benchmark, store, source_dir)
        ordered = _stratified_order(cases, nano.SEED)
        splits = {}
        for profile, (offset, count) in nano.WINDOWS[benchmark].items():
            chosen = ordered[offset : offset + count]
            if len(chosen) != count:
                raise ValueError(f"{benchmark} has too few tasks for {profile}")
            tasks = [nano.task_record(case) for case in chosen]
            splits[profile] = {
                "split": nano.PROFILES[profile],
                "offset": offset,
                "count": count,
                "ids_sha256": digest([task["id"] for task in tasks]),
                "tasks": tasks,
            }
        strata = {}
        for case in cases:
            name = str(case["metadata"].get("stratum", "all"))
            strata[name] = strata.get(name, 0) + 1
        benchmarks[benchmark] = {
            "source": source,
            "population": {
                "filter": POPULATIONS[benchmark],
                "count": len(cases),
                "strata": dict(sorted(strata.items())),
            },
            "stratify_by": STRATA[benchmark],
            "weight": nano.WEIGHTS[benchmark],
            "splits": splits,
        }
    document = {
        "version": nano.IDS_VERSION,
        "seed": nano.SEED,
        "selection": (
            "sr-bench stratified-hash-v1 order of the normalized population; "
            "each split is the recorded contiguous [offset, offset+count) window"
        ),
        "hashing": (
            "sha256 over canonical JSON (sorted keys, compact separators, "
            "ensure_ascii=False); prompt_sha256=digest(messages); "
            "content_sha256=digest({messages, answer}) or, for LiveCodeBench, "
            "digest({messages, source: digest(source_record)})"
        ),
        "benchmarks": benchmarks,
    }
    document["sha256"] = nano.body_sha256(document)
    return document


def prepare(profile, store, source_dir=None, progress=None):
    """Materialize one frozen split into the store after verifying every hash."""
    if profile not in nano.PROFILES:
        raise ValueError("Select nano or nano-holdout")
    frozen = nano.frozen_ids()
    selected, sources = [], {}
    for benchmark in nano.BENCHMARKS:
        if progress:
            progress(benchmark)
        entry = frozen["benchmarks"][benchmark]
        cases, source = _population(benchmark, store, source_dir)
        if source != entry["source"]:
            raise ValueError(f"{benchmark} source files differ from the frozen pin")
        by_id = {case["id"]: case for case in cases}
        for task in entry["splits"][profile]["tasks"]:
            case = by_id.get(task["id"])
            if case is None or nano.task_record(case) != task:
                raise ValueError(f"{task['id']} differs from its frozen hashes")
            case["metadata"].update(
                {"split": nano.PROFILES[profile], "source_revision": source["revision"]}
            )
            selected.append(case)
        sources[benchmark] = {
            "url": source["repository"],
            "revision": source["revision"],
            "files": source["files"],
            "revision_verification": "pinned-huggingface-download",
            "license_url": source["repository"]
            + "/blob/"
            + source["revision"]
            + "/README.md",
            "access_note": "Upstream access and license terms apply; sr-bench does not redistribute the source dataset.",
            "normalizer": "sr-bench-normalizer-v1",
            "nano_ids": {"version": nano.IDS_VERSION, "sha256": frozen["sha256"]},
        }
    return _write_dataset(store, selected, profile, nano.SEED, sources)
