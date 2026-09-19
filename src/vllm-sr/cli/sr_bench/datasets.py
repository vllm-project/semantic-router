"""Bounded, read-only views of prepared datasets, without grading material."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from urllib.parse import urlparse

from .contracts import canonical, catalog, digest
from .setup import harness_paths
from .sources import _write_dataset

MAX_DATA_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_CASE_BYTES = 2 * 1024 * 1024
MAX_PAGE_BYTES = 8 * 1024 * 1024
MAX_CASES = 50000
MAX_PAGE_SIZE = 100
MAX_QUERY_CHARS = 200
MAX_CASE_ID_CHARS = 512
MAX_BENCHMARK_ID_CHARS = 64
MAX_CACHE_BYTES = 32 * 1024 * 1024
MAX_CACHE_ENTRIES = 16
MAX_TASK_FILES = 4096
MAX_COMPOSE_DATASETS = 32
DATASET_ID = re.compile(r"[0-9a-f]{64}\Z")


def _identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Dataset input must be a regular file")
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _read(path, maximum):
    before = _identity(path)
    if before[2] > maximum:
        raise ValueError("Dataset input exceeds the supported size limit")
    with path.open("rb") as handle:
        content = handle.read(maximum + 1)
    if len(content) > maximum or _identity(path) != before:
        raise ValueError("Dataset input changed while being read")
    return content


def _text(value):
    return value if isinstance(value, str) else ""


def _messages(case):
    # Tool responses, assistant examples, and arbitrary message fields may
    # contain reference trajectories or credentials. They are never projected.
    return [
        {"role": message["role"], "content": message["content"]}
        for message in case.get("messages", [])
        if isinstance(message, dict)
        and message.get("role") in {"system", "user", "developer"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]


def _scicode_input(metadata):
    record = metadata.get("source_record", {})
    parts = []
    dependencies = _text(record.get("required_dependencies"))
    if dependencies:
        parts.append("Required dependencies:\n" + dependencies)
    for index, step in enumerate(record.get("sub_steps", []), 1):
        fields = [
            f"Subproblem {index}",
            _text(step.get("step_description_prompt")),
            "Function header:\n" + _text(step.get("function_header")),
            "Return line:\n" + _text(step.get("return_line")),
        ]
        parts.append("\n\n".join(fields))
    return "\n\n".join(parts)


def _terminal_input(metadata, inputs):
    task = Path(metadata["task_path"])
    if task.is_symlink() or not task.is_dir():
        raise ValueError("Task source is unavailable")
    tree, instruction = {}, None
    for index, file in enumerate(task.rglob("*")):
        if index >= MAX_TASK_FILES:
            raise ValueError("Task source exceeds the supported size limit")
        if file.is_symlink():
            raise ValueError("Task source contains a symbolic link")
        if file.is_file():
            content = inputs.read(file)
            tree[str(file.relative_to(task))] = hashlib.sha256(content).hexdigest()
            if file.name == "instruction.md" and file.parent == task:
                instruction = content
    if digest(tree) != metadata.get("tree_sha256"):
        raise ValueError("Task source differs from the frozen dataset")
    if instruction is None or len(instruction) > MAX_CASE_BYTES:
        raise ValueError("Task instructions are unavailable")
    return instruction.decode("utf-8")


def _tau_input(metadata, inputs):
    domain = metadata.get("domain")
    if domain not in {"airline", "retail", "telecom"}:
        raise ValueError("Task domain is unavailable")
    source, _ = harness_paths("tau3")
    path = source / "data" / "tau2" / "domains" / domain / "tasks.json"
    if path not in inputs.tau_tasks:
        inputs.tau_tasks[path] = json.loads(inputs.read(path))
    rows = inputs.tau_tasks[path]
    matches = [r for r in rows if str(r.get("id")) == str(metadata.get("task_id"))]
    if len(matches) != 1 or digest(matches[0]) != metadata.get("source_task_sha256"):
        raise ValueError("Task source differs from the frozen dataset")
    scenario = matches[0].get("user_scenario", {})
    parts = ["Simulated user scenario", _text(scenario.get("persona"))]
    instructions = scenario.get("instructions", {})
    if isinstance(instructions, str):
        parts.append(instructions)
    elif isinstance(instructions, dict):
        for key in (
            "domain",
            "reason_for_call",
            "known_info",
            "unknown_info",
            "task_instructions",
        ):
            value = _text(instructions.get(key))
            if value:
                parts.append(key.replace("_", " ").capitalize() + ":\n" + value)
    return "\n\n".join(p for p in parts if p)


class _TaskInputs:
    def __init__(self):
        self.remaining = MAX_DATA_BYTES
        self.tau_tasks = {}

    def read(self, path):
        content = _read(path, self.remaining)
        self.remaining -= len(content)
        return content


def _public_case(case, inputs):
    metadata = case.get("metadata", {})
    messages = _messages(case)
    notice = None
    if not messages:
        try:
            if case["benchmark"] == "scicode":
                question = _scicode_input(metadata)
            elif case["benchmark"] == "terminal-bench-2.1":
                question = _terminal_input(metadata, inputs)
            elif case["benchmark"] == "tau3":
                question = _tau_input(metadata, inputs)
            else:
                question = ""
        except (OSError, ValueError, KeyError, TypeError):
            question = ""
            notice = "Pinned task inputs are unavailable or no longer match the prepared dataset."
        if question:
            messages = [{"role": "user", "content": question}]
    question = "\n\n".join(m["content"] for m in messages)
    result = {
        "id": case["id"],
        "benchmark": case["benchmark"],
        "category": (
            metadata.get("stratum")
            if isinstance(metadata.get("stratum"), str)
            else "all"
        ),
        "question": question,
        "messages": messages,
        "input_status": "available" if question else "unavailable",
    }
    choices = case.get("choices")
    if isinstance(choices, list) and all(isinstance(c, str) for c in choices):
        result["choices"] = choices
    if not question:
        result["input_notice"] = (
            notice or "This prepared task has no displayable input."
        )
    if len(json.dumps(result).encode()) > MAX_CASE_BYTES:
        result.update(
            question="",
            messages=[],
            input_status="unavailable",
            input_notice="Task input exceeds the supported display size.",
        )
        result.pop("choices", None)
    return result


def _source_summary(benchmark, source):
    result = {"benchmark": benchmark, "file_count": len(source.get("files", []))}
    for field in ("revision", "revision_verification", "normalizer"):
        value = source.get(field)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
            result[field] = value
    for field in ("url", "license_url"):
        value = source.get(field, "")
        parsed = urlparse(value)
        if (
            parsed.scheme == "https"
            and parsed.hostname in {"github.com", "huggingface.co"}
            and not parsed.username
            and not parsed.password
        ):
            result[field] = parsed._replace(query="", fragment="").geturl()
        elif field == "url":
            result[field] = "local-import"
    result["access_note"] = (
        "Upstream access and license terms apply; task answers and grading materials are excluded from this view."
    )
    return result


class DatasetReader:
    """A stat-validated, size-bounded cache; no network, model, or task execution."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.lock = threading.Lock()

    def _paths(self, identity):
        if not isinstance(identity, str) or not DATASET_ID.fullmatch(identity):
            raise ValueError("Invalid dataset identifier")
        folder = self.root / "datasets" / identity
        if not folder.exists():
            raise KeyError(identity)
        if folder.resolve() != folder or folder.is_symlink():
            raise ValueError("Dataset must remain inside its registered store")
        return folder / "manifest.json", folder / "cases.jsonl"

    def _load(self, identity, task_inputs=None):
        manifest_path, data_path = self._paths(identity)
        key = (_identity(manifest_path), _identity(data_path))
        with self.lock:
            cached = self.cache.get(identity)
            if cached and cached[0] == key:
                self.cache.move_to_end(identity)
                return cached[1], cached[2]
            manifest = json.loads(_read(manifest_path, MAX_MANIFEST_BYTES))
            if not isinstance(manifest, dict) or any(
                key in manifest and not isinstance(manifest[key], str)
                for key in ("id", "name", "profile", "split")
            ):
                raise ValueError("Prepared dataset header has invalid field types")
            if (
                not isinstance(manifest.get("case_count"), int)
                or isinstance(manifest["case_count"], bool)
                or not 0 <= manifest["case_count"] <= MAX_CASES
            ):
                raise ValueError("Prepared dataset case count is invalid")
            if "custom_subset" in manifest and not isinstance(
                manifest["custom_subset"], bool
            ):
                raise ValueError(
                    "Prepared dataset custom subset marker must be boolean"
                )
            if not isinstance(manifest.get("sources", {}), dict) or any(
                not isinstance(k, str) or not isinstance(v, dict)
                for k, v in manifest.get("sources", {}).items()
            ):
                raise ValueError("Prepared dataset source provenance has invalid types")
            data = _read(data_path, MAX_DATA_BYTES)
            if manifest.get("id") != identity or hashlib.sha256(
                data
            ).hexdigest() != manifest.get("sha256"):
                raise ValueError(
                    "Prepared dataset identity or content digest does not match"
                )
            cases, ids, inputs, view_bytes = [], set(), task_inputs or _TaskInputs(), 0
            for line in data.split(b"\n"):
                if not line.strip():
                    continue
                if len(cases) >= MAX_CASES or len(line) > MAX_DATA_BYTES:
                    raise ValueError("Dataset exceeds the supported case limit")
                case = json.loads(line)
                if (
                    not isinstance(case, dict)
                    or not isinstance(case.get("id"), str)
                    or not case["id"]
                    or len(case["id"]) > MAX_CASE_ID_CHARS
                    or not isinstance(case.get("benchmark"), str)
                    or not case["benchmark"]
                    or len(case["benchmark"]) > MAX_BENCHMARK_ID_CHARS
                    or case["id"] in ids
                ):
                    raise ValueError("Dataset contains an invalid or duplicate case")
                if not isinstance(case.get("metadata", {}), dict) or not isinstance(
                    case.get("messages", []), list
                ):
                    raise ValueError(
                        "Dataset case metadata and messages have invalid types"
                    )
                category = case.get("metadata", {}).get("stratum")
                if isinstance(category, str) and len(category) > MAX_QUERY_CHARS:
                    raise ValueError(
                        "Dataset category exceeds the supported size limit"
                    )
                ids.add(case["id"])
                projected = _public_case(case, inputs)
                view_bytes += len(json.dumps(projected).encode())
                if view_bytes > MAX_DATA_BYTES:
                    raise ValueError("Dataset display exceeds the supported size limit")
                cases.append(projected)
            if len(cases) != manifest.get("case_count"):
                raise ValueError("Prepared dataset case count does not match")
            # Source-backed empty tasks are reverified for every request; a
            # changed checkout must not be hidden behind a dataset-file cache.
            cacheable = not any(
                c["benchmark"] in {"tau3", "terminal-bench-2.1"} for c in cases
            )
            size = len(json.dumps(cases).encode()) + len(json.dumps(manifest).encode())
            if identity in self.cache:
                self.cache_bytes -= self.cache.pop(identity)[3]
            if cacheable and size <= MAX_CACHE_BYTES:
                while self.cache and (
                    self.cache_bytes + size > MAX_CACHE_BYTES
                    or len(self.cache) >= MAX_CACHE_ENTRIES
                ):
                    self.cache_bytes -= self.cache.popitem(last=False)[1][3]
                self.cache[identity] = (key, manifest, cases, size)
                self.cache_bytes += size
            return manifest, cases

    def detail(self, identity):
        manifest, cases = self._load(identity)
        counts = Counter((c["benchmark"], c["category"]) for c in cases)
        titles = {b["id"]: b["title"] for b in catalog()["benchmarks"]}
        benchmarks = []
        for benchmark in sorted({c["benchmark"] for c in cases}):
            categories = [
                {"name": category, "count": n}
                for (b, category), n in sorted(counts.items())
                if b == benchmark
            ]
            benchmarks.append(
                {
                    "id": benchmark,
                    "title": titles.get(benchmark, benchmark),
                    "count": sum(c["count"] for c in categories),
                    "categories": categories,
                }
            )
        return {
            **{
                k: manifest[k]
                for k in (
                    "id",
                    "name",
                    "profile",
                    "split",
                    "custom_subset",
                    "case_count",
                )
                if k in manifest
            },
            "benchmarks": benchmarks,
            "categories": [
                {"benchmark": b, "name": c, "count": n}
                for (b, c), n in sorted(counts.items())
            ],
            "provenance": {
                "sha256": manifest["sha256"],
                **(
                    {"seed": manifest["seed"]}
                    if isinstance(manifest.get("seed"), int)
                    and not isinstance(manifest["seed"], bool)
                    else {}
                ),
                **(
                    {"selection": manifest["selection"]}
                    if isinstance(manifest.get("selection"), str)
                    and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", manifest["selection"])
                    else {}
                ),
                "sources": [
                    _source_summary(b, s)
                    for b, s in sorted(manifest.get("sources", {}).items())
                ],
            },
        }

    def page(
        self, identity, *, cursor="0", limit="25", benchmark="", category="", q=""
    ):
        if (
            len(str(cursor)) > len(str(MAX_CASES))
            or len(str(limit)) > len(str(MAX_PAGE_SIZE))
            or not str(cursor).isdigit()
            or not str(limit).isdigit()
        ):
            raise ValueError("Dataset cursor and limit must be non-negative integers")
        offset, count = int(cursor), int(limit)
        if offset > MAX_CASES or not 1 <= count <= MAX_PAGE_SIZE:
            raise ValueError("Dataset cursor or page size is out of range")
        if any(
            not isinstance(v, str) or len(v) > MAX_QUERY_CHARS
            for v in (benchmark, category, q)
        ):
            raise ValueError("Dataset filters must contain at most 200 characters")
        _, rows = self._load(identity)
        needle = q.casefold()
        matches = [
            c
            for c in rows
            if (not benchmark or c["benchmark"] == benchmark)
            and (not category or c["category"] == category)
            and (
                not needle
                or needle
                in (
                    c["id"]
                    + "\n"
                    + c["question"]
                    + "\n"
                    + "\n".join(c.get("choices", []))
                ).casefold()
            )
        ]
        page, size = [], 0
        for row in matches[offset : offset + count]:
            size += len(json.dumps(row).encode())
            if size > MAX_PAGE_BYTES:
                break
            page.append(row)
        following = offset + len(page)
        return {
            "dataset_id": identity,
            "cases": page,
            "total": len(matches),
            "dataset_total": len(rows),
            "next_cursor": str(following) if following < len(matches) else None,
            "limit": count,
        }

    def compose(self, dataset_ids, benchmarks):
        if (
            not isinstance(dataset_ids, list)
            or not 1 <= len(dataset_ids) <= MAX_COMPOSE_DATASETS
            or not all(isinstance(i, str) for i in dataset_ids)
        ):
            raise ValueError("Compose requires 1 to 32 registered dataset IDs")
        if (
            not isinstance(benchmarks, list)
            or not benchmarks
            or len(benchmarks) > MAX_COMPOSE_DATASETS
            or not all(isinstance(b, str) for b in benchmarks)
            or len(set(benchmarks)) != len(benchmarks)
        ):
            raise ValueError("Compose requires distinct benchmark IDs")
        groups, sources, identity, custom = {}, {}, None, False
        single_source = None
        input_bytes = 0
        task_inputs = _TaskInputs()
        for dataset_id in dict.fromkeys(dataset_ids):
            manifest, _ = self._load(dataset_id, task_inputs)
            current = (
                manifest.get("profile"),
                manifest.get("seed"),
                manifest.get("split"),
            )
            if current[0] not in {"smoke", "quick", "standard"} or current[2] != (
                "holdout" if current[0] == "standard" else "dev"
            ):
                raise ValueError("Dataset has an invalid profile or split")
            if not isinstance(current[1], int) or isinstance(current[1], bool):
                raise ValueError("Dataset seed must be a frozen integer")
            if identity is not None and identity != current:
                raise ValueError(
                    "Composing datasets requires the same profile, seed, and split"
                )
            identity = current
            custom = custom or bool(manifest.get("custom_subset"))
            _, path = self._paths(dataset_id)
            data = _read(path, MAX_DATA_BYTES - input_bytes)
            input_bytes += len(data)
            if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
                raise ValueError("Prepared dataset changed while composing")
            rows = [json.loads(line) for line in data.split(b"\n") if line.strip()]
            if any(
                row.get("metadata", {}).get(
                    "split", None if current[2] == "holdout" else current[2]
                )
                != current[2]
                for row in rows
            ):
                raise ValueError("Dataset case split differs from its frozen manifest")
            if len(set(dataset_ids)) == 1 and {row["benchmark"] for row in rows} == set(
                benchmarks
            ):
                single_source = {**manifest, "path": str(path)}
            for benchmark in benchmarks:
                selected = sorted(
                    (row for row in rows if row["benchmark"] == benchmark),
                    key=lambda row: row["id"],
                )
                if not selected:
                    continue
                source = manifest.get("sources", {}).get(benchmark)
                if not source:
                    raise ValueError(
                        "Selected benchmark has no frozen source provenance"
                    )
                if benchmark in groups and (
                    groups[benchmark] != selected or sources[benchmark] != source
                ):
                    raise ValueError(
                        "Selected datasets contain conflicting benchmark selections"
                    )
                groups[benchmark], sources[benchmark] = selected, source
        if set(groups) != set(benchmarks):
            raise ValueError(
                "Prepared datasets do not contain every selected benchmark"
            )
        if single_source is not None:
            return single_source
        cases = [case for benchmark in sorted(groups) for case in groups[benchmark]]
        if len({case["id"] for case in cases}) != len(cases):
            raise ValueError("Composed dataset contains duplicate case IDs")
        sources = {benchmark: sources[benchmark] for benchmark in sorted(sources)}
        if len(cases) > MAX_CASES or len(json.dumps(cases).encode()) > MAX_DATA_BYTES:
            raise ValueError("Composed dataset exceeds the supported size limit")
        content_sha = hashlib.sha256(
            "".join(canonical(case) + "\n" for case in cases).encode()
        ).hexdigest()
        output_id = digest(
            {
                "cases_sha256": content_sha,
                "sources": sources,
                "profile": identity[0],
                "seed": identity[1],
                "custom_subset": custom,
                "selection": "stratified-hash-v1",
            }
        )
        if (self.root / "datasets" / output_id).exists():
            existing, _ = self._load(output_id)
            if any(
                existing.get(key) != value
                for key, value in {
                    "sha256": content_sha,
                    "sources": sources,
                    "profile": identity[0],
                    "seed": identity[1],
                    "custom_subset": custom,
                }.items()
            ):
                raise ValueError("Existing composed dataset provenance changed")
            return {**existing, "path": str(self._paths(output_id)[1])}
        with self.lock:
            return _write_dataset(
                self.root, cases, identity[0], identity[1], sources, custom
            )
