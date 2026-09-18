"""Pinned dataset acquisition and stable, disjoint sr-bench selections."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

from .contracts import VERSION, canonical, digest
from .source_records import normalize_records

# (repository, immutable revision, source files). No moving branch is used.
HF_SOURCES = {
    "mmlu-pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", ("data/test-00000-of-00001.parquet",)),
    "gpqa-diamond": ("Idavidrein/gpqa", "633f5ee89ab8ad4522a9f850766b73f62147ffdd", ("gpqa_diamond.csv",)),
    "hle": ("cais/hle", "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29", ("data/test-00000-of-00001.parquet",)),
    "livecodebench": ("livecodebench/code_generation_lite", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", tuple("test" + (str(i) if i > 1 else "") + ".jsonl" for i in range(1, 7))),
    "scicode": ("SciCode1/SciCode", "4510f6a6aa27c43fad7b43da2c59602a86e88480", ("problems_test.jsonl",)),
    "simpleqa-verified": ("google/simpleqa-verified", "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc", ("simpleqa_verified.csv",)),
}
COUNTS = {
    "mmlu-pro": (14, 500, 2000), "gpqa-diamond": (4, 40, 158),
    "hle": (4, 40, 200), "livecodebench": (2, 30, 150),
    "scicode": (1, 3, 20), "terminal-bench-2.1": (1, 3, 15),
    "simpleqa-verified": (5, 100, 500), "arc-agi-2": (2, 12, 80),
    "tau3": (3, 12, 60),
}
ARC_REVISION = "f3283f727488ad98fe575ea6a5ac981e4a188e49"


def read_records(path):
    path = Path(path)
    if path.suffix == ".csv":
        return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("Parquet preparation requires pip install 'vllm-sr[bench]'") from exc
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    raise ValueError("Source must contain an array of task records")


def _stratified_order(cases, seed):
    groups = defaultdict(list)
    for case in cases:
        groups[str(case.get("metadata", {}).get("stratum", "all"))].append(case)
    for rows in groups.values():
        rows.sort(key=lambda c: digest([seed, c["id"]]))
    # Proportional interleaving preserves subject representation at every prefix.
    ordered = []
    for name, rows in sorted(groups.items()):
        ordered.extend(((i + .5) / len(rows), name, row) for i, row in enumerate(rows))
    return [row for _, _, row in sorted(ordered, key=lambda x: (x[0], x[1]))]


def _acquire(benchmark, root):
    if benchmark not in HF_SOURCES:
        raise ValueError(f"{benchmark} requires --source-path to a pinned upstream task checkout; see benchmark setup")
    from huggingface_hub import hf_hub_download
    repo, revision, filenames = HF_SOURCES[benchmark]
    records, files = [], []
    for filename in filenames:
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(root / "cache")))
        records.extend(read_records(path))
        files.append({"name": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return records, {"url": "https://huggingface.co/datasets/" + repo, "revision": revision, "files": files}


def _local_source(benchmark, path, revision):
    path = Path(path).expanduser().resolve()
    files, records = [], []
    if path.is_file():
        records = read_records(path)
        files.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    elif benchmark == "arc-agi-2":
        task_dir = path / "data" / "evaluation" if (path / "data" / "evaluation").is_dir() else path
        for p in sorted(task_dir.glob("*.json")):
            records.append({"id": p.stem, **json.loads(p.read_text())})
            files.append({"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    elif benchmark == "tau3":
        for domain in ("airline", "retail", "telecom"):
            p = path / "data" / "tau2" / "domains" / domain / "tasks.json"
            rows = json.loads(p.read_text())
            records.extend({**row, "domain": domain} for row in rows)
            files.append({"name": str(p.relative_to(path)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    elif benchmark == "terminal-bench-2.1":
        for p in sorted(path.glob("*/task.toml")):
            # Whole sandbox task trees, including verifier, are source inputs.
            tree = {str(f.relative_to(p.parent)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.parent.rglob("*")) if f.is_file() and not f.is_symlink()}
            records.append({"id": p.parent.name, "task_path": str(p.parent), "tree_sha256": digest(tree)})
            files.append({"name": p.parent.name, "sha256": digest(tree)})
    else:
        raise ValueError("source-path must be a record file or a supported task checkout")
    if not records:
        raise ValueError("Source contains no supported tasks")
    if not revision:
        raise ValueError("Local source imports require --revision and record their actual content digest")
    return records, {"url": "local-import", "revision": revision, "files": files}


def prepare_dataset(*, benchmark, profile, store, source_path=None, revision=None, seed=20260918, limit=None):
    if benchmark not in COUNTS or profile not in {"smoke", "quick", "standard"}:
        raise ValueError("Unknown benchmark or profile")
    root = Path(store).expanduser().resolve()
    rows, source = _local_source(benchmark, source_path, revision) if source_path else _acquire(benchmark, root)
    cases = normalize_records(benchmark, rows, seed)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source task IDs")
    ordered = _stratified_order(cases, seed)
    smoke, quick, standard = COUNTS[benchmark]
    desired = {"smoke": smoke, "quick": quick, "standard": standard}[profile]
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= desired:
            raise ValueError("limit must be a positive integer within the profile budget")
        desired = limit
    offset = quick if profile == "standard" else 0
    selected = ordered[offset:offset + desired]
    if len(selected) != desired:
        raise ValueError(f"Source has insufficient tasks for disjoint {profile}: need {offset + desired}, found {len(ordered)}")
    for c in selected:
        c.setdefault("metadata", {}).update({"split": "holdout" if profile == "standard" else "dev", "source_revision": source["revision"]})
    return _write_dataset(root, selected, profile, seed, {benchmark: source}, limit is not None)


def _write_dataset(root, cases, profile, seed, sources, custom=False):
    content = "".join(canonical(c) + "\n" for c in cases).encode()
    sha = hashlib.sha256(content).hexdigest()
    identity = digest({"cases_sha256": sha, "sources": sources, "profile": profile, "seed": seed, "selection": "stratified-hash-v1"})
    destination = root / "datasets" / identity
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    data_path = destination / "cases.jsonl"
    data_path.write_bytes(content)
    manifest = {"version": VERSION, "id": identity, "name": "+".join(sorted(sources)) + "/" + profile, "path": str(data_path), "sha256": sha, "case_count": len(cases), "profile": profile, "custom_subset": custom, "benchmarks": sorted(sources), "sources": sources, "seed": seed, "selection": "stratified-hash-v1", "split": "holdout" if profile == "standard" else "dev"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    data_path.chmod(0o600)
    (destination / "manifest.json").chmod(0o600)
    return manifest


def combine_datasets(manifests, store):
    if not manifests or len({(m["profile"], m["seed"]) for m in manifests}) != 1:
        raise ValueError("Combining datasets requires one profile and seed")
    cases, sources = [], {}
    for m in manifests:
        p = Path(m["path"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != m["sha256"]:
            raise ValueError("Prepared dataset digest changed")
        if set(sources) & set(m["sources"]):
            raise ValueError("Duplicate benchmark in dataset bundle")
        cases.extend(read_records(p))
        sources.update(m["sources"])
    return _write_dataset(Path(store).expanduser().resolve(), cases, manifests[0]["profile"], manifests[0]["seed"], sources, any(m.get("custom_subset") for m in manifests))
