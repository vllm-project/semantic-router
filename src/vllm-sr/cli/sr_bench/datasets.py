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
MAX_DATASETS = 4096
MAX_COMPOSE_DATASETS = 32
MAX_FINGERPRINT_SCAN_BYTES = 2 * 1024 * 1024 * 1024
MAX_FINGERPRINT_INDEX_BYTES = 32 * 1024 * 1024
MAX_FINGERPRINT_CACHE_BYTES = 4 * 1024 * 1024
MAX_FINGERPRINT_CACHE_ENTRIES = 64
DATASET_ID = re.compile(r"[0-9a-f]{64}\Z")


class DatasetSizeLimitError(ValueError):
    """A known input size cannot fit within this bounded dataset read."""

    def __init__(self, path, size, maximum, reason_code="source_size_limit"):
        super().__init__("Dataset input exceeds the supported size limit")
        self.path = path
        self.size = size
        self.maximum = maximum
        self.reason_code = reason_code


def _identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Dataset input must be a regular file")
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _read(path, maximum):
    before = _identity(path)
    if before[2] > maximum:
        raise DatasetSizeLimitError(path, before[2], maximum)
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


def _validate_frozen_case(case, manifest, ids):
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
        raise ValueError("Dataset case metadata and messages have invalid types")
    metadata = case.get("metadata", {})
    category = metadata.get("stratum")
    if isinstance(category, str) and len(category) > MAX_QUERY_CHARS:
        raise ValueError("Dataset category exceeds the supported size limit")
    expected_split = manifest["split"]
    if (
        metadata.get("split", None if expected_split == "holdout" else "dev")
        != expected_split
    ):
        raise ValueError("Dataset case split differs from its frozen manifest")
    ids.add(case["id"])


class DatasetReader:
    """A content-validated, size-bounded cache; no network or task execution."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.fingerprint_cache = OrderedDict()
        self.fingerprint_cache_bytes = 0
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

    def _read_manifest(self, identity):
        manifest_path, _ = self._paths(identity)
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
            raise ValueError("Prepared dataset custom subset marker must be boolean")
        if not isinstance(manifest.get("sources", {}), dict) or any(
            not isinstance(k, str) or not isinstance(v, dict)
            for k, v in manifest.get("sources", {}).items()
        ):
            raise ValueError("Prepared dataset source provenance has invalid types")
        if manifest.get("profile") not in {
            "smoke",
            "quick",
            "standard",
        } or manifest.get("split") != (
            "holdout" if manifest["profile"] == "standard" else "dev"
        ):
            raise ValueError("Dataset has an invalid profile or split")
        if manifest.get("id") != identity:
            raise ValueError(
                "Prepared dataset identity or content digest does not match"
            )
        return manifest

    def _read_frozen(self, identity, remaining=MAX_DATA_BYTES):
        """Validate stored rows before display, selection or composition uses them."""
        manifest = self._read_manifest(identity)
        _, data_path = self._paths(identity)
        data = _read(data_path, remaining)
        if hashlib.sha256(data).hexdigest() != manifest.get("sha256"):
            raise ValueError(
                "Prepared dataset identity or content digest does not match"
            )
        rows, ids = [], set()
        for line in data.split(b"\n"):
            if not line.strip():
                continue
            if len(rows) >= MAX_CASES or len(line) > MAX_DATA_BYTES:
                raise ValueError("Dataset exceeds the supported case limit")
            case = json.loads(line)
            _validate_frozen_case(case, manifest, ids)
            rows.append(case)
        if len(rows) != manifest["case_count"]:
            raise ValueError("Prepared dataset case count does not match")
        return manifest, rows, len(data)

    def _cached_content_matches(self, identity, manifest, maximum):
        # File timestamps can have coarser precision than consecutive writes.
        # A matching stat tuple alone is not proof of unchanged frozen inputs.
        if self._read_manifest(identity) != manifest:
            return False
        _, path = self._paths(identity)
        before = _identity(path)
        if before[2] > maximum:
            raise DatasetSizeLimitError(path, before[2], maximum)
        checksum, size = hashlib.sha256(), 0
        with path.open("rb") as source:
            while chunk := source.read(min(1024 * 1024, maximum - size + 1)):
                size += len(chunk)
                if size > maximum:
                    raise DatasetSizeLimitError(path, size, maximum)
                checksum.update(chunk)
        if _identity(path) != before:
            raise ValueError("Dataset input changed while being read")
        if checksum.hexdigest() != manifest.get("sha256"):
            raise ValueError(
                "Prepared dataset identity or content digest does not match"
            )
        return True

    def _fingerprint(self, identity, remaining):
        """Verify full content with bounded line/index memory, caching only proofs."""
        manifest_path, data_path = self._paths(identity)
        key = (_identity(manifest_path), _identity(data_path))
        size = key[1][2]
        if size > remaining:
            raise DatasetSizeLimitError(
                data_path, size, remaining, "scan_budget_exhausted"
            )
        with self.lock:
            cached = self.fingerprint_cache.get(identity)
            if (
                cached
                and cached[0] == key
                and self._cached_content_matches(identity, cached[1], remaining)
            ):
                self.fingerprint_cache.move_to_end(identity)
                return cached[1], cached[2], size
            manifest = self._read_manifest(identity)
            groups, ids, index_bytes, content = {}, set(), 0, hashlib.sha256()
            with data_path.open("rb") as source:
                while line := source.readline(MAX_DATA_BYTES + 1):
                    if len(line) > MAX_DATA_BYTES:
                        raise DatasetSizeLimitError(
                            data_path, len(line), MAX_DATA_BYTES
                        )
                    content.update(line)
                    if not line.strip():
                        continue
                    if len(ids) >= MAX_CASES:
                        raise ValueError("Dataset exceeds the supported case limit")
                    case = json.loads(line)
                    _validate_frozen_case(case, manifest, ids)
                    index_bytes += len(case["id"].encode()) + 256
                    if index_bytes > MAX_FINGERPRINT_INDEX_BYTES:
                        raise DatasetSizeLimitError(
                            data_path, index_bytes, MAX_FINGERPRINT_INDEX_BYTES
                        )
                    groups.setdefault(case["benchmark"], []).append(
                        (case["id"], digest(case))
                    )
            if (_identity(manifest_path), _identity(data_path)) != key:
                raise ValueError("Prepared dataset changed while fingerprinting")
            if content.hexdigest() != manifest.get("sha256"):
                raise ValueError(
                    "Prepared dataset identity or content digest does not match"
                )
            if len(ids) != manifest["case_count"]:
                raise ValueError("Prepared dataset case count does not match")
            proofs = {}
            for benchmark, rows in groups.items():
                provenance = manifest.get("sources", {}).get(benchmark)
                if not provenance:
                    raise ValueError("Prepared benchmark has no frozen provenance")
                proofs[benchmark] = {
                    "count": len(rows),
                    "proof": digest(
                        {
                            "cases": sorted(rows),
                            "source": provenance,
                            "seed": manifest.get("seed"),
                            "split": manifest["split"],
                        }
                    ),
                }
            entry_bytes = len(json.dumps([manifest, proofs]).encode())
            if identity in self.fingerprint_cache:
                self.fingerprint_cache_bytes -= self.fingerprint_cache.pop(identity)[3]
            if entry_bytes <= MAX_FINGERPRINT_CACHE_BYTES:
                while self.fingerprint_cache and (
                    self.fingerprint_cache_bytes + entry_bytes
                    > MAX_FINGERPRINT_CACHE_BYTES
                    or len(self.fingerprint_cache) >= MAX_FINGERPRINT_CACHE_ENTRIES
                ):
                    self.fingerprint_cache_bytes -= self.fingerprint_cache.popitem(
                        last=False
                    )[1][3]
                self.fingerprint_cache[identity] = (key, manifest, proofs, entry_bytes)
                self.fingerprint_cache_bytes += entry_bytes
            return manifest, proofs, size

    def _load(self, identity, task_inputs=None):
        manifest_path, data_path = self._paths(identity)
        key = (_identity(manifest_path), _identity(data_path))
        with self.lock:
            cached = self.cache.get(identity)
            if (
                cached
                and cached[0] == key
                and self._cached_content_matches(identity, cached[1], MAX_DATA_BYTES)
            ):
                self.cache.move_to_end(identity)
                return cached[1], cached[2]
            manifest, rows, _ = self._read_frozen(identity)
            cases, inputs, view_bytes = [], task_inputs or _TaskInputs(), 0
            for case in rows:
                projected = _public_case(case, inputs)
                view_bytes += len(json.dumps(projected).encode())
                if view_bytes > MAX_DATA_BYTES:
                    raise ValueError("Dataset display exceeds the supported size limit")
                cases.append(projected)
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

    def selection(self, profile):
        """Resolve default sources from frozen content, without arbitrary revision picks."""
        if profile not in {"smoke", "quick", "standard"}:
            raise ValueError("Unknown evaluation profile")
        split = "holdout" if profile == "standard" else "dev"
        groups, blocked, remaining = {}, {}, MAX_FINGERPRINT_SCAN_BYTES
        paths = sorted((self.root / "datasets").glob("*/manifest.json"))
        if len(paths) > MAX_DATASETS:
            raise ValueError("Dataset inventory exceeds the selection limit")
        for path in paths:
            identity = path.parent.name
            # Skip unrelated profiles before reading their potentially large case files.
            header = self._read_manifest(identity)
            if header["profile"] != profile or header.get("custom_subset"):
                continue
            try:
                manifest, proofs, size = self._fingerprint(identity, remaining)
            except DatasetSizeLimitError as error:
                if error.path != self._paths(identity)[1]:
                    raise
                declared = header.get("benchmarks")
                known = {item["id"] for item in catalog()["benchmarks"]}
                if (
                    not isinstance(declared, list)
                    or not declared
                    or not all(
                        isinstance(item, str) and item in known for item in declared
                    )
                    or len(set(declared)) != len(declared)
                    or set(declared) != set(header["sources"])
                ):
                    raise ValueError(
                        "Oversized dataset has no valid benchmark scope"
                    ) from error
                source_limit = error.reason_code != "scan_budget_exhausted"
                reason = (
                    "A prepared source exceeds the per-case or fingerprint index limit. All versions of this benchmark are unavailable until its source can be verified."
                    if source_limit
                    else "The dataset selection scan reached its byte limit. This benchmark could not be fully verified; narrow the prepared collection or select a custom dataset."
                )
                for benchmark in declared:
                    blocked[benchmark] = (
                        (
                            "source_size_limit"
                            if source_limit
                            else "scan_budget_exhausted"
                        ),
                        reason,
                    )
                if not source_limit:
                    remaining = 0
                else:
                    # Reserve the entire source even if validation stopped early.
                    # Repeated oversized rows must not bypass the scan budget.
                    remaining = max(
                        0, remaining - _identity(self._paths(identity)[1])[2]
                    )
                continue
            remaining -= size
            if manifest["profile"] != profile or manifest.get("custom_subset"):
                raise ValueError("Prepared dataset changed while selecting")
            seed = manifest.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("Prepared dataset has an invalid seed")
            for benchmark, proof in proofs.items():
                groups.setdefault(benchmark, []).append(
                    (proof["proof"], len(proofs), identity, proof["count"], seed, size)
                )
        entries = []
        for benchmark in catalog()["benchmarks"]:
            candidates = groups.get(benchmark["id"], [])
            conflict = len({item[0] for item in candidates}) > 1
            usable = [item for item in candidates if item[5] <= MAX_DATA_BYTES]
            chosen = (
                min(usable, key=lambda item: (item[1], item[5], item[2]))
                if usable
                else None
            )
            limitation = blocked.get(benchmark["id"])
            if not limitation and candidates and not usable:
                limitation = (
                    "source_size_limit",
                    "Full content was verified, but every prepared source exceeds the composition size limit. Prepare a smaller dedicated source for this benchmark.",
                )
            eligible = bool(chosen) and not conflict and limitation is None
            entries.append(
                {
                    "id": benchmark["id"],
                    "title": benchmark["title"],
                    "eligible": eligible,
                    "case_count": chosen[3] if eligible else 0,
                    "source_ids": [chosen[2]] if eligible else [],
                    "reason_code": (
                        limitation[0]
                        if limitation
                        else (
                            "source_conflict"
                            if conflict
                            else None if eligible else "not_prepared"
                        )
                    ),
                    "reason": (
                        limitation[1]
                        if limitation
                        else (
                            "Multiple frozen versions are available. Prepare one canonical collection or explicitly select a custom dataset."
                            if conflict
                            else (
                                None
                                if eligible
                                else "Prepare this benchmark for the selected profile."
                            )
                        )
                    ),
                }
            )
        seeds = {
            items[0][4]
            for benchmark, items in groups.items()
            if any(entry["id"] == benchmark and entry["eligible"] for entry in entries)
        }
        if len(seeds) > 1:
            for entry in entries:
                if entry["eligible"]:
                    entry.update(
                        eligible=False,
                        source_ids=[],
                        case_count=0,
                        reason="Prepared benchmarks use different seeds. Prepare a collection with one common seed or explicitly select a custom dataset.",
                        reason_code="seed_conflict",
                    )
        return {
            "profile": profile,
            "split": split,
            "seed": next(iter(seeds)) if len(seeds) == 1 else None,
            "benchmarks": entries,
            "model_requests": 0,
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
        for dataset_id in dict.fromkeys(dataset_ids):
            manifest, rows, size = self._read_frozen(
                dataset_id, MAX_DATA_BYTES - input_bytes
            )
            input_bytes += size
            current = (
                manifest.get("profile"),
                manifest.get("seed"),
                manifest.get("split"),
            )
            if not isinstance(current[1], int) or isinstance(current[1], bool):
                raise ValueError("Dataset seed must be a frozen integer")
            if identity is not None and identity != current:
                raise ValueError(
                    "Composing datasets requires the same profile, seed, and split"
                )
            identity = current
            custom = custom or bool(manifest.get("custom_subset"))
            _, path = self._paths(dataset_id)
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
            existing, _, _ = self._read_frozen(output_id)
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
