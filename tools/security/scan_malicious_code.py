#!/usr/bin/env python3
"""
Supply Chain Security Scanner — Codebase-wide malicious code detection.

Scans source trees for patterns seen in real supply chain attacks:
  - Base64/hex/marshal obfuscated execution
  - Credential harvesting and exfiltration
  - Binary backdoors without source
  - Suspicious subprocess/network calls

All regex signatures are in scan_sensitive_paths.toml (base64-encoded).
The entire tools/security/ directory is excluded from scans (SELF_DIR_PARTS),
matched by trailing path components so this still applies when the scanner
and the scanned tree are different checkouts (see _is_self_path).

Exit codes: 0 — clean, 1 — findings, 2 — scanner error
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import re
import stat
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import tomllib

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

SECURITY_DIR = Path(__file__).resolve().parent
SIGNATURES_CONFIG = SECURITY_DIR / "scan_sensitive_paths.toml"

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
    "egg-info",
    ".eggs",
    ".venv-onnx",
    "supply-chain-security-scan",
}

SELF_DIR_PARTS = SECURITY_DIR.parts[-2:]


def _is_self_path(fpath: Path) -> bool:
    """True if fpath lives in the scanner's own directory (tools/security).

    Matches by trailing directory components, not an absolute path, so this
    still excludes the scanner's own files when it runs from one checkout
    (e.g. a trusted base/ tree in CI) against a different checkout (e.g. an
    untrusted pr-code/ tree) that contains its own copy of these files.
    """
    try:
        return fpath.resolve().parent.parts[-len(SELF_DIR_PARTS) :] == SELF_DIR_PARTS
    except (OSError, ValueError):
        return False


SKIP_DIR_PATTERNS = {
    "model",
    "models",
    "weights",
    "checkpoint",
    "onnx",
    "safetensors",
    "target",
    "site-packages",
    "lora_",
    "mmbert",
    "pii_",
}

MAX_FILE_SIZE = 2 * 1024 * 1024

BINARY_EXTENSIONS = {
    ".so",
    ".dll",
    ".dylib",
    ".pyd",
    ".exe",
    ".bin",
    ".elf",
    ".o",
    ".a",
    ".lib",
    ".wasm",
}

KNOWN_BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".svg",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mkv",
    ".webm",
    ".ogg",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".safetensors",
    ".gguf",
    ".pt",
    ".pth",
    ".onnx",
    ".pb",
    ".parquet",
    ".arrow",
    ".feather",
    ".h5",
    ".hdf5",
    ".db",
    ".sqlite",
    ".sqlite3",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".pyw",
    ".pyi",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".rb",
    ".pl",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".json",
}

KNOWN_TEST_MARKERS = ("test", "spec", "mock", "fixture", "example")


@dataclass
class Finding:
    severity: str
    category: str
    file: str
    line: int
    message: str
    snippet: str = ""

    def __str__(self):
        loc = f"{self.file}:{self.line}" if self.line else self.file
        s = f"[{self.severity}] {self.category} — {loc}\n  {self.message}"
        if self.snippet:
            s += f"\n  >>> {self.snippet[:200]}"
        return s


@dataclass
class ScanResult:
    findings: list = field(default_factory=list)

    def add(self, severity, category, file, line, message, snippet=""):
        self.findings.append(Finding(severity, category, file, line, message, snippet))

    @property
    def worst_severity(self):
        if not self.findings:
            return None
        return min(
            self.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 99)
        ).severity

    def summary(self):
        counts = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts


def _compile_entry(entry: dict) -> tuple[re.Pattern[str], str, str]:
    raw = base64.b64decode(entry["pattern_b64"]).decode("utf-8")
    return (
        re.compile(raw),
        entry.get("severity", "MEDIUM"),
        entry.get("message", "Suspicious pattern"),
    )


def _compile_section(data: dict, name: str) -> list[tuple[re.Pattern[str], str, str]]:
    return [_compile_entry(e) for e in data.get(name, []) if e.get("pattern_b64")]


@lru_cache(maxsize=1)
def load_signatures() -> tuple[
    list[tuple[re.Pattern[str], str, str]],
    list[tuple[re.Pattern[str], str, str]],
    list[tuple[re.Pattern[str], str, str]],
    list[tuple[re.Pattern[str], str, str]],
    re.Pattern[str],
]:
    try:
        data = tomllib.loads(SIGNATURES_CONFIG.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"Error: cannot load {SIGNATURES_CONFIG}: {exc}", file=sys.stderr)
        sys.exit(2)

    obfuscation = _compile_section(data, "obfuscation")
    exfiltration = _compile_section(data, "exfiltration")
    credential = _compile_section(data, "credential_paths") + _compile_section(
        data, "cloud_metadata"
    )
    setup_hooks = _compile_section(data, "setup_hooks")
    long_b64_raw = base64.b64decode(data["long_base64"]["pattern_b64"]).decode("utf-8")
    long_b64 = re.compile(long_b64_raw)
    return obfuscation, exfiltration, credential, setup_hooks, long_b64


def should_skip_dir(dirname: str) -> bool:
    if dirname in SKIP_DIRS:
        return True
    dl = dirname.lower()
    return any(p in dl for p in SKIP_DIR_PATTERNS)


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    length = len(data)
    return -sum((c / length) * math.log2(c / length) for c in freq if c > 0)


def is_binary_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(8192)
    except (OSError, PermissionError):
        return False


def has_elf_pe_macho_header(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        if header[:4] == b"\x7fELF":
            return "ELF"
        if header[:2] == b"MZ":
            return "PE/MZ"
        if header[:4] in (
            b"\xfe\xed\xfa\xce",
            b"\xfe\xed\xfa\xcf",
            b"\xce\xfa\xed\xfe",
            b"\xcf\xfa\xed\xfe",
        ):
            return "Mach-O"
        if header[:4] == b"\xca\xfe\xba\xbe":
            return "Mach-O Universal"
    except (OSError, PermissionError):
        pass
    return None


def scan_source_patterns(
    root: Path,
    result: ScanResult,
    obfuscation: list,
    exfiltration: list,
    credential: list,
    setup_hooks: list,
    long_b64_re: re.Pattern[str],
):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if _is_self_path(fpath):
                continue
            suffix = fpath.suffix.lower()
            if suffix not in SOURCE_EXTENSIONS:
                continue
            try:
                if fpath.stat().st_size > MAX_FILE_SIZE:
                    continue
                content = fpath.read_text(errors="replace")
            except (OSError, PermissionError):
                continue

            lines = content.splitlines()
            for lineno, line in enumerate(lines, 1):
                for pattern, severity, desc in obfuscation:
                    if pattern.search(line):
                        result.add(
                            severity,
                            "OBFUSCATION",
                            str(fpath),
                            lineno,
                            desc,
                            line.strip(),
                        )
                for pattern, severity, desc in exfiltration:
                    if pattern.search(line):
                        result.add(
                            severity,
                            "EXFILTRATION",
                            str(fpath),
                            lineno,
                            desc,
                            line.strip(),
                        )
                for pattern, severity, desc in credential:
                    if pattern.search(line):
                        is_test = any(
                            t in str(fpath).lower() for t in KNOWN_TEST_MARKERS
                        )
                        result.add(
                            "LOW" if is_test else severity,
                            "CREDENTIAL_ACCESS",
                            str(fpath),
                            lineno,
                            desc,
                            line.strip(),
                        )

            if suffix in (".py", ".pyw") and "setup" in fname.lower():
                for pattern, severity, desc in setup_hooks:
                    for lineno, line in enumerate(lines, 1):
                        if pattern.search(line):
                            result.add(
                                severity,
                                "INSTALL_HOOK",
                                str(fpath),
                                lineno,
                                desc,
                                line.strip(),
                            )

            for match in long_b64_re.findall(content):
                if len(match) > 500:
                    for lineno, line in enumerate(lines, 1):
                        if match[:80] in line:
                            result.add(
                                "HIGH",
                                "LONG_BASE64",
                                str(fpath),
                                lineno,
                                f"Very long base64 string ({len(match)} chars) — "
                                "possible encoded payload",
                                match[:120] + "...",
                            )
                            break


def scan_binary_anomalies(root: Path, result: ScanResult):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        rel_dir = Path(dirpath).relative_to(root)
        in_build = any(
            p in str(rel_dir).lower()
            for p in (
                "build",
                "dist",
                "target",
                "release",
                "debug",
                "vendor",
                "third_party",
                "3rdparty",
            )
        )
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if _is_self_path(fpath):
                continue
            suffix = fpath.suffix.lower()

            if suffix in BINARY_EXTENSIONS and not in_build:
                exe_type = has_elf_pe_macho_header(fpath)
                if exe_type:
                    result.add(
                        "HIGH",
                        "BINARY_IN_SOURCE",
                        str(fpath),
                        0,
                        f"Compiled {exe_type} binary in source tree — verify intentional",
                    )
                else:
                    result.add(
                        "MEDIUM",
                        "BINARY_IN_SOURCE",
                        str(fpath),
                        0,
                        f"Binary file ({suffix}) in source tree",
                    )

            if suffix == ".pyc":
                py_source = fpath.with_suffix(".py")
                parent_py = fpath.parent.parent / fpath.stem
                if not py_source.exists() and not parent_py.with_suffix(".py").exists():
                    result.add(
                        "HIGH",
                        "ORPHAN_PYC",
                        str(fpath),
                        0,
                        "Compiled .pyc without .py source — possible bytecode injection",
                    )

            try:
                mode = fpath.stat().st_mode
            except (OSError, PermissionError):
                continue

            if not suffix and not in_build and mode & stat.S_IXUSR:
                exe_type = has_elf_pe_macho_header(fpath)
                if exe_type:
                    result.add(
                        "MEDIUM",
                        "BINARY_EXECUTABLE",
                        str(fpath),
                        0,
                        f"Executable {exe_type} binary without extension",
                    )

            if suffix in (".py", ".js", ".sh", ".ts") and is_binary_file(fpath):
                result.add(
                    "HIGH",
                    "BINARY_MASQUERADE",
                    str(fpath),
                    0,
                    f"Source extension ({suffix}) with binary data — possible masquerade",
                )

            if (
                not in_build
                and suffix not in BINARY_EXTENSIONS
                and suffix not in KNOWN_BINARY_EXTENSIONS
                and is_binary_file(fpath)
            ):
                try:
                    size = fpath.stat().st_size
                    if size > 1024:
                        with open(fpath, "rb") as f:
                            ent = entropy(f.read(min(size, 65536)))
                        if ent > 7.5:
                            result.add(
                                "MEDIUM",
                                "HIGH_ENTROPY",
                                str(fpath),
                                0,
                                f"High entropy binary ({ent:.2f} bits/byte) — "
                                "possible encrypted/compressed payload",
                            )
                except (OSError, PermissionError):
                    pass


def print_report(result: ScanResult, verbose: bool = False):
    if not result.findings:
        print("\n  [CLEAN] No supply chain security issues detected.\n")
        return

    result.findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 99))
    counts = result.summary()
    print("\n" + "=" * 72)
    print("  SUPPLY CHAIN SECURITY SCAN RESULTS")
    print("=" * 72)
    print(f"\n  Total findings: {len(result.findings)}")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if sev in counts:
            print(f"    {sev}: {counts[sev]}")
    print()
    prev_sev = None
    for f in result.findings:
        if not verbose and f.severity == "LOW":
            continue
        if f.severity != prev_sev:
            print(f"--- {f.severity} ---")
            prev_sev = f.severity
        print(f)
        print()
    if "CRITICAL" in counts:
        print("  *** CRITICAL findings require IMMEDIATE investigation ***")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scan codebase for supply chain attacks and malicious code"
    )
    parser.add_argument("path", nargs="?", default=".", help="Root directory to scan")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--fail-on",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default="HIGH",
    )
    args = parser.parse_args()
    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    obfuscation, exfiltration, credential, setup_hooks, long_b64_re = load_signatures()
    result = ScanResult()
    print(f"Scanning {root} ...")
    scan_source_patterns(
        root, result, obfuscation, exfiltration, credential, setup_hooks, long_b64_re
    )
    scan_binary_anomalies(root, result)
    print_report(result, verbose=args.verbose)

    fail_threshold = SEVERITY_ORDER.get(args.fail_on, 1)
    worst = result.worst_severity
    if worst and SEVERITY_ORDER.get(worst, 99) <= fail_threshold:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
