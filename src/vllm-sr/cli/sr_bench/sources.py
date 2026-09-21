"""Pinned dataset acquisition and stable, disjoint sr-bench selections."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from .contracts import VERSION, canonical, digest
from .dataset_io import (
    MAX_ROW_BYTES,
    MAX_ROWS,
    MAX_SCAN_BYTES,
    DatasetSizeLimitError,
    publish_dataset,
    verified_lines,
)
from .dataset_io import (
    identity as file_identity,
)
from .preparation import SELECTION, select_cases, validate_cases
from .source_records import normalize_records
from .task_identity import source_identity

# (repository, immutable revision, source files). No moving branch is used.
HF_SOURCES = {
    "mmlu-pro": (
        "TIGER-Lab/MMLU-Pro",
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        ("data/test-00000-of-00001.parquet",),
    ),
    "gpqa-diamond": (
        "Idavidrein/gpqa",
        "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        ("gpqa_diamond.csv",),
    ),
    "hle": (
        "cais/hle",
        "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
        ("data/test-00000-of-00001.parquet",),
    ),
    "livecodebench": (
        "livecodebench/code_generation_lite",
        "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        tuple("test" + (str(i) if i > 1 else "") + ".jsonl" for i in range(1, 7)),
    ),
    "scicode": (
        "SciCode1/SciCode",
        "4510f6a6aa27c43fad7b43da2c59602a86e88480",
        ("problems_test.jsonl",),
    ),
    "simpleqa-verified": (
        "google/simpleqa-verified",
        "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc",
        ("simpleqa_verified.csv",),
    ),
}
COUNTS = {
    "mmlu-pro": (14, 500, 2000),
    "gpqa-diamond": (4, 40, 158),
    "hle": (4, 40, 200),
    "livecodebench": (2, 30, 150),
    "scicode": (1, 3, 20),
    "terminal-bench-2.1": (1, 3, 15),
    "simpleqa-verified": (5, 100, 500),
    "arc-agi-2": (2, 12, 80),
    "tau3": (3, 12, 60),
}
ARC_REVISION = "f3283f727488ad98fe575ea6a5ac981e4a188e49"


def read_records(path):
    path = Path(path)
    if path.suffix == ".csv":
        return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415 - optional data preparation extra
        except ImportError as exc:
            raise ValueError(
                "Parquet preparation requires pip install 'vllm-sr[bench]'"
            ) from exc
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().split("\n") if line.strip()
        ]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    raise ValueError("Source must contain an array of task records")


def _stratified_order(cases, seed, balanced=False):
    groups = defaultdict(list)
    for case in cases:
        groups[str(case.get("metadata", {}).get("stratum", "all"))].append(case)
    for rows in groups.values():
        rows.sort(key=lambda c: digest([seed, c["id"]]))
    if balanced:
        return [
            rows[i]
            for i in range(max(map(len, groups.values())))
            for _, rows in sorted(groups.items())
            if i < len(rows)
        ]
    # Begin with one case from every stratum, then interleave proportionally.
    # Thus the 14-case MMLU smoke profile covers all 14 subjects.
    first = [rows.pop(0) for _, rows in sorted(groups.items())]
    ordered = []
    for name, rows in sorted(groups.items()):
        ordered.extend(((i + 0.5) / len(rows), name, row) for i, row in enumerate(rows))
    return first + [row for _, _, row in sorted(ordered, key=lambda x: (x[0], x[1]))]


def _acquire(benchmark, root):
    if benchmark not in HF_SOURCES:
        from .setup import (  # noqa: PLC0415 - setup reads dataset catalog lazily
            PACKAGES,
            TASK_SOURCES,
            _checkout,
            harness_paths,
            home,
        )

        if benchmark == "tau3":
            spec = PACKAGES[benchmark]
            path, _ = harness_paths(benchmark)
        else:
            spec = TASK_SOURCES[benchmark]
            path = home() / "sources" / benchmark
        _checkout(spec, path)
        return _local_source(benchmark, path, spec["revision"])
    # Only remote acquisition initializes Hub support.
    from huggingface_hub import (  # noqa: PLC0415
        hf_hub_download,
    )

    repo, revision, filenames = HF_SOURCES[benchmark]
    records, files = [], []
    for filename in filenames:
        path = Path(
            hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(root / "cache"),
            )
        )
        records.extend(read_records(path))
        files.append(
            {"name": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    source_url = "https://huggingface.co/datasets/" + repo
    return records, {
        "url": source_url,
        "revision": revision,
        "files": files,
        "revision_verification": "pinned-huggingface-download",
        "license_url": source_url + "/blob/" + revision + "/README.md",
        "access_note": "Upstream access and license terms apply; sr-bench does not redistribute the source dataset.",
        "normalizer": "sr-bench-normalizer-v1",
    }


def _local_source(benchmark, path, revision):
    source_path = Path(path).expanduser()
    path = source_path.resolve()
    files, records = [], []
    if path.is_file():
        # Hub cache filenames point to extensionless content-addressed blobs.
        # Preserve the supplied format while hashing the actual source bytes.
        records = read_records(source_path)
        files.append(
            {
                "name": source_path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    elif benchmark == "arc-agi-2":
        task_dir = (
            path / "data" / "evaluation"
            if (path / "data" / "evaluation").is_dir()
            else path
        )
        for p in sorted(task_dir.glob("*.json")):
            records.append({"id": p.stem, **json.loads(p.read_text())})
            files.append(
                {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            )
    elif benchmark == "tau3":
        for domain in ("airline", "retail", "telecom"):
            p = path / "data" / "tau2" / "domains" / domain / "tasks.json"
            rows = json.loads(p.read_text())
            if domain == "telecom":
                split_file = p.parent / "split_tasks.json"
                base_ids = set(json.loads(split_file.read_text())["base"])
                rows = [row for row in rows if row["id"] in base_ids]
                files.append(
                    {
                        "name": str(split_file.relative_to(path)),
                        "sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
                    }
                )
            records.extend({**row, "domain": domain} for row in rows)
            files.append(
                {
                    "name": str(p.relative_to(path)),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
            )
    elif benchmark == "terminal-bench-2.1":
        task_dir = path / "tasks" if (path / "tasks").is_dir() else path
        for p in sorted(task_dir.glob("*/task.toml")):
            # Whole sandbox task trees, including verifier, are source inputs.
            tree = {
                str(f.relative_to(p.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in sorted(p.parent.rglob("*"))
                if f.is_file() and not f.is_symlink()
            }
            records.append(
                {
                    "id": p.parent.name,
                    "task_path": str(p.parent),
                    "tree_sha256": digest(tree),
                }
            )
            files.append({"name": p.parent.name, "sha256": digest(tree)})
    else:
        raise ValueError(
            "source-path must be a record file or a supported task checkout"
        )
    if not records:
        raise ValueError("Source contains no supported tasks")
    if not revision:
        raise ValueError(
            "Local source imports require --revision and record their actual content digest"
        )
    if path.is_dir():
        actual = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if actual.returncode == 0 and actual.stdout.strip() != revision:
            raise ValueError(
                "Local task checkout differs from the declared source revision"
            )
    return records, {
        "url": "local-import",
        "revision": revision,
        "files": files,
        "revision_verification": (
            "git-head"
            if path.is_dir() and actual.returncode == 0
            else "declared-revision-with-content-digest"
        ),
        "access_note": "Operator-supplied source; retain upstream access and license records alongside this manifest.",
        "normalizer": "sr-bench-normalizer-v1",
    }


def prepare_dataset(
    *,
    benchmark,
    profile,
    store,
    source_path=None,
    revision=None,
    seed=20260918,
    limit=None,
    exclusion_snapshot=None,
    source_partition=None,
    evaluation_role=None,
):
    if benchmark not in COUNTS:
        if (
            exclusion_snapshot is not None
            or source_partition is not None
            or evaluation_role is not None
        ):
            raise ValueError(
                "This adapter does not support source-bound history preparation"
            )
        # Adapter registration imports source preparation.
        from .adapters import (  # noqa: PLC0415
            get_adapter,
        )

        adapter = get_adapter(benchmark)
        if not adapter.prepare:
            raise ValueError("This adapter requires operator-prepared task records")
        return adapter.prepare(
            benchmark=benchmark,
            profile=profile,
            store=store,
            source_path=source_path,
            revision=revision,
            seed=seed,
            limit=limit,
        )
    if profile not in {"smoke", "quick", "standard"}:
        raise ValueError("Unknown benchmark or profile")
    root = Path(store).expanduser().resolve()
    rows, source = (
        _local_source(benchmark, source_path, revision)
        if source_path
        else _acquire(benchmark, root)
    )
    task_source = (
        source_identity(source, source_partition)
        if source_partition is not None
        else None
    )
    cases = normalize_records(benchmark, rows, seed, task_source=task_source)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source task IDs")
    ordered = _stratified_order(cases, seed, balanced=benchmark == "tau3")
    smoke, quick, standard = COUNTS[benchmark]
    desired = {"smoke": smoke, "quick": quick, "standard": standard}[profile]
    if limit is not None:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 0 < limit <= desired
        ):
            raise ValueError(
                "limit must be a positive integer within the profile budget"
            )
        desired = limit
    offset = quick if profile == "standard" else 0
    preparation = None
    if (
        exclusion_snapshot is not None
        or source_partition is not None
        or evaluation_role is not None
    ):
        from .history_snapshot import load_snapshot  # noqa: PLC0415

        snapshot = (
            load_snapshot(exclusion_snapshot)
            if exclusion_snapshot is not None
            else None
        )
        selected, preparation = select_cases(
            ordered,
            benchmark,
            offset,
            desired,
            snapshot=snapshot,
            role=evaluation_role,
            task_source=task_source,
        )
    else:
        selected = ordered[offset : offset + desired]
    if len(selected) != desired:
        raise ValueError(
            f"Source has insufficient tasks for disjoint {profile}: need {offset + desired}, found {len(ordered)}"
        )
    for c in selected:
        c.setdefault("metadata", {}).update(
            {
                "split": "holdout" if profile == "standard" else "dev",
                "source_revision": source["revision"],
            }
        )
    return _write_dataset(
        root,
        selected,
        profile,
        seed,
        {benchmark: source},
        limit is not None,
        preparation=preparation,
    )


def _write_dataset(
    root, cases, profile, seed, sources, custom=False, *, preparation=None
):
    """Stream canonical rows to a private stage, then publish immutable artifacts."""
    root = Path(root).expanduser().resolve()
    datasets = root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True, mode=0o700)
    if datasets.is_symlink() or datasets.resolve() != datasets:
        raise ValueError("Prepared datasets must remain inside their store")
    with tempfile.TemporaryDirectory(prefix=".dataset-", dir=root) as temporary:
        staged = Path(temporary) / "cases.jsonl"
        checksum, count, total = hashlib.sha256(), 0, 0
        identities = []
        with staged.open("xb") as output:
            for case in cases:
                content = (canonical(case) + "\n").encode()
                if len(content) > MAX_ROW_BYTES:
                    raise DatasetSizeLimitError(staged, len(content), MAX_ROW_BYTES)
                count += 1
                total += len(content)
                if count > MAX_ROWS or total > MAX_SCAN_BYTES:
                    raise ValueError(
                        "Prepared dataset exceeds the supported size limit"
                    )
                output.write(content)
                checksum.update(content)
                if preparation:
                    identities.append(
                        {
                            "id": case["id"],
                            "benchmark": case["benchmark"],
                            "metadata": {
                                "task_identity": case.get("metadata", {}).get(
                                    "task_identity"
                                )
                            },
                        }
                    )
        sha = checksum.hexdigest()
        identity = digest(
            {
                "cases_sha256": sha,
                "sources": sources,
                "profile": profile,
                "seed": seed,
                "custom_subset": custom,
                "selection": SELECTION if preparation else "stratified-hash-v1",
                **({"preparation": preparation} if preparation else {}),
            }
        )
        destination = datasets / identity
        data_path = destination / "cases.jsonl"
        manifest = {
            "version": VERSION,
            "id": identity,
            "name": "+".join(sorted(sources)) + "/" + profile,
            "path": str(data_path),
            "sha256": sha,
            "case_count": count,
            "profile": profile,
            "custom_subset": custom,
            "benchmarks": sorted(sources),
            "sources": sources,
            "seed": seed,
            "selection": SELECTION if preparation else "stratified-hash-v1",
            **({"preparation": preparation} if preparation else {}),
            "split": "holdout" if profile == "standard" else "dev",
        }
        if preparation:
            validate_cases(manifest, identities)
        manifest_stage = Path(temporary) / "manifest.json"
        rendered = (json.dumps(manifest, indent=2) + "\n").encode()
        if len(rendered) > 2 * 1024 * 1024:
            raise ValueError("Prepared manifest exceeds its supported byte limit")
        manifest_stage.write_bytes(rendered)
        destination.mkdir(exist_ok=True, mode=0o700)
        if destination.is_symlink() or destination.resolve() != destination:
            raise ValueError("Prepared dataset must not be a symlink")
        publish_dataset(destination, staged, sha, manifest_stage, rendered)
        return manifest


def combine_datasets(manifests, store):
    if not manifests or len({(m["profile"], m["seed"]) for m in manifests}) != 1:
        raise ValueError("Combining datasets requires one profile and seed")
    sources, total, preparation = {}, 0, {}
    for manifest in manifests:
        if set(sources) & set(manifest["sources"]):
            raise ValueError("Duplicate benchmark in dataset bundle")
        sources.update(manifest["sources"])
        from .preparation import validate_manifest  # noqa: PLC0415

        validate_manifest(manifest)
        preparation.update(manifest.get("preparation", {}))

    def cases():
        nonlocal total
        ids = set()
        for manifest in manifests:
            path, count = Path(manifest["path"]), 0
            source_size = file_identity(path)[2]
            for line in verified_lines(
                path, manifest["sha256"], maximum=MAX_SCAN_BYTES - total
            ):
                case = json.loads(line)
                if (
                    not isinstance(case, dict)
                    or not isinstance(case.get("id"), str)
                    or case["id"] in ids
                ):
                    raise ValueError(
                        "Combined dataset contains invalid or duplicate case IDs"
                    )
                ids.add(case["id"])
                count += 1
                yield case
            total += source_size
            if count != manifest["case_count"]:
                raise ValueError("Prepared dataset case count does not match")

    return _write_dataset(
        Path(store).expanduser().resolve(),
        cases(),
        manifests[0]["profile"],
        manifests[0]["seed"],
        sources,
        any(m.get("custom_subset") for m in manifests),
        preparation=preparation or None,
    )
