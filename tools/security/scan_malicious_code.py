#!/usr/bin/env python3
"""
Supply Chain Security Scanner — regex fallback for malicious code detection.

Regex signatures are base64-encoded below so this module does not embed
attack-shaped literals that trip static security tools.

Scans source trees for:
  - Obfuscated execution (base64/zlib/marshal/hex)
  - Credential harvesting and exfiltration
  - Binary backdoors without source
  - Suspicious subprocess/network calls

The entire tools/security/ directory is excluded from scans (SELF_DIR).

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
from typing import NamedTuple

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

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

SELF_DIR = str(Path(__file__).resolve().parent) + os.sep

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

# (base64-encoded regex, severity, message)
_OBFUSCATION_SIGS: tuple[tuple[str, str, str], ...] = (
    (
        "ZXhlY1xzKlwoXHMqYmFzZTY0XC5iNjRkZWNvZGVccypcKA==",
        "CRITICAL",
        "Dynamic exec after base64 decode (classic obfuscation)",
    ),
    (
        "ZXZhbFxzKlwoXHMqYmFzZTY0XC5iNjRkZWNvZGVccypcKA==",
        "CRITICAL",
        "Dynamic eval after base64 decode",
    ),
    (
        "ZXhlY1xzKlwoXHMqY29tcGlsZVxzKlwoXHMqYmFzZTY0",
        "CRITICAL",
        "Dynamic exec with compile and base64 input",
    ),
    (
        "YmFzZTY0XC5iNjRkZWNvZGVccypcKFxzKmJhc2U2NFwuYjY0ZGVjb2Rl",
        "CRITICAL",
        "Nested base64 decode (multi-layer obfuscation)",
    ),
    (
        "ZXhlY1xzKlwoXHMqemxpYlwuZGVjb21wcmVzc1xzKlwo",
        "HIGH",
        "Dynamic exec after zlib decompress",
    ),
    (
        "ZXhlY1xzKlwoXHMqZ3ppcFwuZGVjb21wcmVzc1xzKlwo",
        "HIGH",
        "Dynamic exec after gzip decompress",
    ),
    (
        "bWFyc2hhbFwubG9hZHNccypcKA==",
        "HIGH",
        "marshal.loads() bytecode deserialization",
    ),
    ("dHlwZXNcLkNvZGVUeXBlXHMqXCg=", "HIGH", "Dynamic CodeType construction"),
    (
        "ZXhlY1xzKlwoXHMqWydcIl1cXHhbMC05YS1mQS1GXQ==",
        "HIGH",
        "exec() with hex-escape string literal",
    ),
    (
        "ZXhlY1xzKlwoXHMqYnl0ZXNcLmZyb21oZXhccypcKA==",
        "HIGH",
        "Dynamic exec after bytes.fromhex",
    ),
    (
        "ZXhlY1xzKlwoXHMqY29kZWNzXC5kZWNvZGVccypcKA==",
        "HIGH",
        "Dynamic exec after codecs.decode",
    ),
    (
        "X19pbXBvcnRfX1xzKlwoXHMqWydcIl1vc1snXCJdXHMqXClccypcLlxzKnN5c3RlbQ==",
        "HIGH",
        "Dynamic import of os followed by system()",
    ),
    (
        "Z2V0YXR0clxzKlwoXHMqX19pbXBvcnRfXw==",
        "MEDIUM",
        "getattr on dynamic __import__ (evasion)",
    ),
)
_EXFILTRATION_SIGS: tuple[tuple[str, str, str], ...] = (
    (
        "cmVxdWVzdHNcLihwb3N0fHB1dClccypcKFxzKlsnXCJdaHR0cHM/Oi8vKD8hbG9jYWxob3N0fDEyN1wuMFwuMFwuMSk=",
        "HIGH",
        "HTTP POST/PUT to external URL",
    ),
    (
        "dXJsbGliXC5yZXF1ZXN0XC51cmxvcGVuXHMqXChccyp1cmxsaWJcLnJlcXVlc3RcLlJlcXVlc3RccypcKFteKV0qbWV0aG9kXHMqPVxzKlsnXCJdUE9TVA==",
        "HIGH",
        "urllib POST request",
    ),
    (
        "c3VicHJvY2Vzc1wuXHcrXHMqXChccypcWz9ccypbJ1wiXWN1cmxbJ1wiXS4qLVhccypQT1NU",
        "HIGH",
        "subprocess invoking curl POST",
    ),
    (
        "c3VicHJvY2Vzc1wuXHcrXHMqXChccypcWz9ccypbJ1wiXXdnZXRbJ1wiXS4qLS1wb3N0",
        "HIGH",
        "subprocess invoking wget POST",
    ),
    (
        "c29ja2V0XC5zb2NrZXRccypcKC4qXClccyouKlwuY29ubmVjdFxzKlwoXHMqXChccypbJ1wiXSg/ITEyN1wufGxvY2FsaG9zdCk=",
        "MEDIUM",
        "Raw socket connect to non-loopback host",
    ),
)
_CREDENTIAL_PATH_SIGS: tuple[tuple[str, str, str], ...] = (
    ("fi9cLnNzaC8=", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLmF3cy8=", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLmt1YmUv", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLmRvY2tlci8=", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLmdudXBnLw==", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLmNvbmZpZy9nY2xvdWQv", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLmF6dXJlLw==", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLm5wbXJj", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLnZhdWx0LXRva2Vu", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLm5ldHJj", "MEDIUM", "Reference to sensitive filesystem path"),
    ("fi9cLmdpdGNvbmZpZw==", "MEDIUM", "Reference to sensitive filesystem path"),
    (
        "fi9cLmdpdC1jcmVkZW50aWFscw==",
        "MEDIUM",
        "Reference to sensitive filesystem path",
    ),
    ("L2V0Yy9zc2wvcHJpdmF0ZQ==", "MEDIUM", "Reference to sensitive filesystem path"),
    ("L2V0Yy9rdWJlcm5ldGVzLw==", "MEDIUM", "Reference to sensitive filesystem path"),
    ("XC5iaXRjb2luLw==", "MEDIUM", "Reference to sensitive filesystem path"),
    ("XC5ldGhlcmV1bS8=", "MEDIUM", "Reference to sensitive filesystem path"),
    ("XC5zb2xhbmEv", "MEDIUM", "Reference to sensitive filesystem path"),
    ("XC5iYXNoX2hpc3Rvcnk=", "MEDIUM", "Reference to sensitive filesystem path"),
    ("XC56c2hfaGlzdG9yeQ==", "MEDIUM", "Reference to sensitive filesystem path"),
    ("XC5wZ3Bhc3M=", "MEDIUM", "Reference to sensitive filesystem path"),
    ("XC5teVwuY25m", "MEDIUM", "Reference to sensitive filesystem path"),
    ("XC5tb25nb3Jj", "MEDIUM", "Reference to sensitive filesystem path"),
)
_CLOUD_METADATA_SIGS: tuple[tuple[str, str, str], ...] = (
    ("MTY5XC4yNTRcLjE2OVwuMjU0", "HIGH", "Cloud instance metadata endpoint reference"),
    (
        "bWV0YWRhdGFcLmdvb2dsZVwuaW50ZXJuYWw=",
        "HIGH",
        "Cloud instance metadata endpoint reference",
    ),
    (
        "bWV0YWRhdGFcLmF6dXJlXC5jb20=",
        "HIGH",
        "Cloud instance metadata endpoint reference",
    ),
)
_SETUP_HOOK_SIGS: tuple[tuple[str, str, str], ...] = (
    (
        "Y2xhc3NccytcdyooSW5zdGFsbHxQb3N0SW5zdGFsbHxEZXZlbG9wKVx3KlxzKlwo",
        "MEDIUM",
        "Custom setuptools install command class",
    ),
    ("Y21kY2xhc3Nccyo9XHMqXHs=", "LOW", "Custom cmdclass in packaging setup"),
)
_LONG_B64_SIG = "W0EtWmEtejAtOSsvPV17MjAwLH0="


class CompiledRule(NamedTuple):
    pattern: re.Pattern[str]
    severity: str
    message: str


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
    findings: list[Finding] = field(default_factory=list)

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
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts


def _compile_sig(encoded: str, severity: str, message: str) -> CompiledRule:
    raw = base64.b64decode(encoded).decode("utf-8")
    return CompiledRule(re.compile(raw), severity, message)


def _compile_sigs(sigs: tuple[tuple[str, str, str], ...]) -> list[CompiledRule]:
    return [_compile_sig(e, s, m) for e, s, m in sigs]


@lru_cache(maxsize=1)
def load_signatures() -> tuple[
    list[CompiledRule],
    list[CompiledRule],
    list[CompiledRule],
    list[CompiledRule],
    re.Pattern[str],
]:
    """Compile base64-encoded signature tuples (cached)."""
    obfuscation = _compile_sigs(_OBFUSCATION_SIGS)
    exfiltration = _compile_sigs(_EXFILTRATION_SIGS)
    credential = _compile_sigs(_CREDENTIAL_PATH_SIGS + _CLOUD_METADATA_SIGS)
    setup_hooks = _compile_sigs(_SETUP_HOOK_SIGS)
    long_b64 = re.compile(base64.b64decode(_LONG_B64_SIG).decode("utf-8"))
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
            chunk = f.read(8192)
        return b"\x00" in chunk
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
    obfuscation: list[CompiledRule],
    exfiltration: list[CompiledRule],
    credential: list[CompiledRule],
    setup_hooks: list[CompiledRule],
    long_b64_re: re.Pattern[str],
):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if str(fpath.resolve()).startswith(SELF_DIR):
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
                for rule in obfuscation:
                    if rule.pattern.search(line):
                        result.add(
                            rule.severity,
                            "OBFUSCATION",
                            str(fpath),
                            lineno,
                            rule.message,
                            line.strip(),
                        )

                for rule in exfiltration:
                    if rule.pattern.search(line):
                        result.add(
                            rule.severity,
                            "EXFILTRATION",
                            str(fpath),
                            lineno,
                            rule.message,
                            line.strip(),
                        )

                for rule in credential:
                    if rule.pattern.search(line):
                        is_test = any(
                            t in str(fpath).lower() for t in KNOWN_TEST_MARKERS
                        )
                        actual_sev = "LOW" if is_test else rule.severity
                        result.add(
                            actual_sev,
                            "CREDENTIAL_ACCESS",
                            str(fpath),
                            lineno,
                            rule.message,
                            line.strip(),
                        )

            if suffix in (".py", ".pyw") and "setup" in fname.lower():
                for rule in setup_hooks:
                    for lineno, line in enumerate(lines, 1):
                        if rule.pattern.search(line):
                            result.add(
                                rule.severity,
                                "INSTALL_HOOK",
                                str(fpath),
                                lineno,
                                rule.message,
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
            if str(fpath.resolve()).startswith(SELF_DIR):
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
                        f"Compiled {exe_type} binary in source tree — "
                        "verify this is intentional",
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
                        "Compiled .pyc without corresponding .py source — "
                        "possible bytecode injection",
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
                    f"File has source extension ({suffix}) but contains "
                    "binary data — possible masquerade",
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
                            data = f.read(min(size, 65536))
                        ent = entropy(data)
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
    print("  SUPPLY CHAIN SECURITY SCAN RESULTS (regex)")
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
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current dir)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show LOW severity findings too"
    )
    parser.add_argument(
        "--fail-on",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default="HIGH",
        help="Exit non-zero if findings at this severity or above (default: HIGH)",
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
        root,
        result,
        obfuscation,
        exfiltration,
        credential,
        setup_hooks,
        long_b64_re,
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
