#!/usr/bin/env python3
"""Reject changed-source directives that can hide Go complexity findings."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import chain, combinations, product
from pathlib import Path

from go_complexity_baseline import resolve_target_tip
from go_complexity_identity import COMPLEXITY_LINTERS


_NOLINT_PATTERN = re.compile(r"//\s*nolint(?::\s*([A-Za-z0-9_,\-]+))?")
_GENERATED_PATTERN = re.compile(r"^// Code generated .* DO NOT EDIT\.$")


def _git_source(repo_root: Path, commit: str, path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _normalized_line(line: str) -> str:
    return re.sub(r"\s+", "", line)


def _next_code_line(lines: list[str], start: int) -> str:
    for candidate in lines[start:]:
        stripped = candidate.strip()
        if stripped and not stripped.startswith("//"):
            return _normalized_line(stripped)
    return "<eof>"


def _complexity_nolint_sites(source: bytes) -> Counter[str]:
    lines = source.decode("utf-8", errors="strict").splitlines()
    sites: Counter[str] = Counter()
    for index, line in enumerate(lines):
        for match in _NOLINT_PATTERN.finditer(line):
            raw_linters = match.group(1)
            requested = (
                frozenset(item for item in raw_linters.split(",") if item)
                if raw_linters
                else frozenset({"all"})
            )
            linters = COMPLEXITY_LINTERS if "all" in requested else requested
            affected = sorted(linters & COMPLEXITY_LINTERS)
            if not affected:
                continue
            prefix = line[: match.start()].strip()
            context = (
                _normalized_line(prefix)
                if prefix
                else _next_code_line(lines, index + 1)
            )
            sites[f"{','.join(affected)}:{context}"] += 1
    return sites


def _build_constraints(source: bytes) -> frozenset[str]:
    constraints = set()
    for line in source.decode("utf-8", errors="strict").splitlines():
        stripped = line.strip()
        if stripped.startswith("package "):
            break
        if stripped.startswith("//go:build") or stripped.startswith("// +build"):
            constraints.add(_normalized_line(stripped))
    return frozenset(constraints)


def _generated_markers(source: bytes) -> Counter[str]:
    return Counter(
        line.strip()
        for line in source.decode("utf-8", errors="strict").splitlines()
        if _GENERATED_PATTERN.fullmatch(line.strip())
    )


def _go_comments(source: bytes) -> list[tuple[int, bytes]]:
    comments: list[tuple[int, bytes]] = []
    index = 0
    while index < len(source):
        current = source[index : index + 1]
        if current in {b'"', b"'"}:
            quote = current
            index += 1
            while index < len(source):
                if source[index : index + 1] == b"\\":
                    index += 2
                elif source[index : index + 1] == quote:
                    index += 1
                    break
                else:
                    index += 1
            continue
        if current == b"`":
            end = source.find(b"`", index + 1)
            index = len(source) if end < 0 else end + 1
            continue
        if source[index : index + 2] == b"//":
            end = source.find(b"\n", index + 2)
            end = len(source) if end < 0 else end
            comments.append((index, source[index:end]))
            index = end
            continue
        if source[index : index + 2] == b"/*":
            end = source.find(b"*/", index + 2)
            end = len(source) if end < 0 else end + 2
            comments.append((index, source[index:end]))
            index = end
            continue
        index += 1
    return comments


def _line_directives(source: bytes) -> Counter[bytes]:
    directives: Counter[bytes] = Counter()
    for offset, comment in _go_comments(source):
        is_block = re.match(rb"/\*line(?:[ \t]+|:)", comment) is not None
        line_start = source.rfind(b"\n", 0, offset) + 1
        is_line = (
            offset == line_start
            and re.match(rb"//line(?:[ \t]+|:)", comment) is not None
        )
        if is_block or is_line:
            directives[re.sub(rb"\s+", b"", comment)] += 1
    return directives


_CONSTRAINT_TOKEN = re.compile(r"&&|\|\||!|\(|\)|[A-Za-z0-9_.]+")
_GOOS = frozenset(
    {
        "aix",
        "android",
        "darwin",
        "dragonfly",
        "freebsd",
        "illumos",
        "ios",
        "js",
        "linux",
        "netbsd",
        "openbsd",
        "plan9",
        "solaris",
        "wasip1",
        "windows",
    }
)
_GOARCH = frozenset(
    {
        "386",
        "amd64",
        "arm",
        "arm64",
        "loong64",
        "mips",
        "mips64",
        "mips64le",
        "mipsle",
        "ppc64",
        "ppc64le",
        "riscv64",
        "s390x",
        "wasm",
    }
)
_UNIX_GOOS = _GOOS - {"android", "ios", "js", "plan9", "wasip1", "windows"}
_MAX_CUSTOM_BUILD_TAGS = 5
_MAX_BUILD_CONTEXT_CANDIDATES = 4096


@dataclass(frozen=True)
class BuildContext:
    goos: str
    goarch: str
    cgo_enabled: bool
    build_tags: tuple[str, ...]


def _platform_identity() -> tuple[str, str, bool]:
    goos = os.getenv("GOOS") or {
        "darwin": "darwin",
        "linux": "linux",
        "win32": "windows",
    }.get(sys.platform, sys.platform)
    machine = platform.machine().lower()
    goarch = os.getenv("GOARCH") or {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine, machine)
    raw_cgo = os.getenv("CGO_ENABLED")
    cgo = raw_cgo == "1" or (
        raw_cgo is None
        and goos not in {"js", "plan9", "wasip1", "windows"}
        and bool(shutil.which("gcc") or shutil.which("clang"))
    )
    return goos, goarch, cgo


def _context_tags(
    goos: str,
    goarch: str,
    cgo: bool,
    custom: frozenset[str] = frozenset(),
) -> frozenset[str]:
    tags = {goos, goarch, "gc", *custom}
    if goos in _UNIX_GOOS:
        tags.add("unix")
    if cgo:
        tags.add("cgo")
    return frozenset(tags)


def _active_tags() -> frozenset[str]:
    return _context_tags(*_platform_identity())


class _ConstraintParser:
    def __init__(self, expression: str, active_tags: frozenset[str]):
        self.tokens = _CONSTRAINT_TOKEN.findall(expression)
        compact = re.sub(r"\s+", "", expression)
        if "".join(self.tokens) != compact:
            raise ValueError("unsupported Go build constraint syntax")
        self.active_tags = active_tags
        self.index = 0

    def parse(self) -> bool:
        value = self._or_expression()
        if self.index != len(self.tokens):
            raise ValueError("trailing Go build constraint tokens")
        return value

    def _or_expression(self) -> bool:
        value = self._and_expression()
        while self._take("||"):
            value = self._and_expression() or value
        return value

    def _and_expression(self) -> bool:
        value = self._unary_expression()
        while self._take("&&"):
            value = self._unary_expression() and value
        return value

    def _unary_expression(self) -> bool:
        if self._take("!"):
            return not self._unary_expression()
        if self._take("("):
            value = self._or_expression()
            if not self._take(")"):
                raise ValueError("unclosed Go build constraint")
            return value
        if self.index >= len(self.tokens):
            raise ValueError("incomplete Go build constraint")
        token = self.tokens[self.index]
        self.index += 1
        return token in self.active_tags

    def _take(self, token: str) -> bool:
        if self.index < len(self.tokens) and self.tokens[self.index] == token:
            self.index += 1
            return True
        return False


def _build_expression(source: bytes) -> str | None:
    for line in source.decode("utf-8", errors="strict").splitlines():
        stripped = line.strip()
        if stripped.startswith("//go:build"):
            return stripped.removeprefix("//go:build").strip()
        if stripped.startswith("// +build"):
            raise ValueError("legacy Go build constraints are not supported")
        if stripped.startswith("package "):
            break
    return None


def _build_constraint_active(
    source: bytes, active_tags: frozenset[str] | None = None
) -> bool:
    expression = _build_expression(source)
    return (
        expression is None
        or _ConstraintParser(expression, active_tags or _active_tags()).parse()
    )


def _filename_active(path: Path, active_tags: frozenset[str] | None = None) -> bool:
    basename = path.name
    if basename.startswith((".", "_")):
        return False
    stem = path.stem
    if stem.endswith("_test"):
        stem = stem.removesuffix("_test")
    parts = stem.split("_")
    tags = active_tags or _active_tags()
    suffixes: list[str] = []
    if parts and parts[-1] in _GOARCH:
        suffixes.append(parts.pop())
        if parts and parts[-1] in _GOOS:
            suffixes.append(parts.pop())
    elif parts and parts[-1] in _GOOS:
        suffixes.append(parts.pop())
    if any(suffix not in tags for suffix in suffixes):
        return False
    return True


def _custom_tag_subsets(tags: set[str]):
    ordered = sorted(tags)
    return chain.from_iterable(
        combinations(ordered, size) for size in range(len(ordered) + 1)
    )


def _candidate_build_contexts(
    custom: set[str],
    supported_platforms: set[tuple[str, str]] | None = None,
):
    current_goos, current_goarch, current_cgo = _platform_identity()
    platforms = supported_platforms or set(product(_GOOS, _GOARCH))
    ordered_platforms = sorted(
        platforms,
        key=lambda item: (item != (current_goos, current_goarch), item),
    )
    candidate_count = (2 ** len(custom)) * len(ordered_platforms) * 2
    if candidate_count > _MAX_BUILD_CONTEXT_CANDIDATES:
        raise ValueError("Go build constraint requires too many candidate contexts")
    for custom_tags, (goos, goarch) in product(
        _custom_tag_subsets(custom), ordered_platforms
    ):
        cgo_values = (
            (current_cgo, not current_cgo)
            if goos == current_goos and goarch == current_goarch
            else (False, True)
        )
        for cgo in cgo_values:
            yield BuildContext(goos, goarch, cgo, tuple(sorted(custom_tags)))


def candidate_build_contexts_for_source(
    source: bytes,
    path: Path,
    supported_platforms: set[tuple[str, str]] | None = None,
):
    if path.name.startswith((".", "_")):
        raise ValueError(f"Go permanently ignores source filename {path.name}")
    expression = _build_expression(source)
    tokens = {
        token
        for token in _CONSTRAINT_TOKEN.findall(expression or "")
        if re.fullmatch(r"[A-Za-z0-9_.]+", token)
    }
    custom = tokens - _GOOS - _GOARCH - {"cgo", "gc", "unix"}
    if len(custom) > _MAX_CUSTOM_BUILD_TAGS:
        raise ValueError("Go build constraint has too many custom tags")
    for context in _candidate_build_contexts(custom, supported_platforms):
        tags = _context_tags(
            context.goos,
            context.goarch,
            context.cgo_enabled,
            frozenset(context.build_tags),
        )
        if _build_constraint_active(source, tags) and _filename_active(path, tags):
            yield context


def build_context_for_source(source: bytes, path: Path) -> BuildContext | None:
    if _build_constraint_active(source) and _filename_active(path):
        return None
    for context in candidate_build_contexts_for_source(source, path):
        return context
    raise ValueError(f"no satisfiable Go build context for {path}")


def validate_changed_source_policy(
    repo_root: Path,
    base_ref: str,
    changed_paths: set[str],
    covered_paths: set[str] | None = None,
) -> list[str]:
    target_tip = resolve_target_tip(repo_root, base_ref)
    covered = covered_paths or set()
    errors = []
    for path in sorted(changed_paths):
        source_path = repo_root / path
        if source_path.suffix != ".go" or not source_path.exists():
            continue
        try:
            current = source_path.read_bytes()
            baseline_source = _git_source(repo_root, target_tip, path)
            baseline = baseline_source or b""
            current_nolint = _complexity_nolint_sites(current)
            added_constraints = _build_constraints(current) - _build_constraints(
                baseline
            )
            added_generated = _generated_markers(current) - _generated_markers(baseline)
            added_line_directives = _line_directives(current) - _line_directives(
                baseline
            )
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"cannot inspect complexity source policy for {path}: {exc}")
            continue
        if current_nolint:
            errors.append(
                f"changed Go file contains a complexity nolint directive: {path}"
            )
        if added_generated:
            errors.append(f"new generated-code marker is not allowed: {path}")
        if added_line_directives:
            errors.append(f"new Go line directive is not allowed: {path}")
        if added_constraints and baseline_source is not None:
            errors.append(f"new or changed Go build constraint is not allowed: {path}")
        if path not in covered:
            errors.append(f"changed Go file was not loaded by lint: {path}")
    return errors
