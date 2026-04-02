#!/usr/bin/env python3
"""
Supply Chain Security Scanner — Codebase-wide malicious code detection.

Scans source trees for patterns seen in real supply chain attacks:
  - Base64/hex/marshal obfuscated execution
  - Credential harvesting and exfiltration
  - Binary backdoors without source
  - Suspicious subprocess/network calls

Exit codes:
  0  — clean
  1  — findings detected (see output)
  2  — scanner error
"""

import argparse
import math
import os
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

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

MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB — skip scanning content of huge files

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

CREDENTIAL_PATHS = [
    r"~/\.ssh/",
    r"~/\.aws/",
    r"~/\.kube/",
    r"~/\.docker/",
    r"~/\.gnupg/",
    r"~/\.config/gcloud/",
    r"~/\.azure/",
    r"~/\.npmrc",
    r"~/\.vault-token",
    r"~/\.netrc",
    r"~/\.gitconfig",
    r"~/\.git-credentials",
    r"/etc/ssl/private",
    r"/etc/kubernetes/",
    r"\.bitcoin/",
    r"\.ethereum/",
    r"\.solana/",
    r"\.bash_history",
    r"\.zsh_history",
    r"\.pgpass",
    r"\.my\.cnf",
    r"\.mongorc",
]

CLOUD_METADATA_URLS = [
    r"169\.254\.169\.254",
    r"metadata\.google\.internal",
    r"metadata\.azure\.com",
]


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


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

OBFUSCATION_PATTERNS = [
    (
        re.compile(r"exec\s*\(\s*base64\.b64decode\s*\("),
        "CRITICAL",
        "exec(base64.b64decode(...)) — classic obfuscated payload execution",
    ),
    (
        re.compile(r"eval\s*\(\s*base64\.b64decode\s*\("),
        "CRITICAL",
        "eval(base64.b64decode(...)) — obfuscated payload execution",
    ),
    (
        re.compile(r"exec\s*\(\s*compile\s*\(\s*base64"),
        "CRITICAL",
        "exec(compile(base64...)) — compiled obfuscated payload",
    ),
    (
        re.compile(r"base64\.b64decode\s*\(\s*base64\.b64decode"),
        "CRITICAL",
        "Double base64 decoding — multi-layer obfuscation",
    ),
    (
        re.compile(r"exec\s*\(\s*zlib\.decompress\s*\("),
        "HIGH",
        "exec(zlib.decompress(...)) — compressed payload execution",
    ),
    (
        re.compile(r"exec\s*\(\s*gzip\.decompress\s*\("),
        "HIGH",
        "exec(gzip.decompress(...)) — compressed payload execution",
    ),
    (
        re.compile(r"marshal\.loads\s*\("),
        "HIGH",
        "marshal.loads() — bytecode deserialization, potential code injection",
    ),
    (
        re.compile(r"types\.CodeType\s*\("),
        "HIGH",
        "types.CodeType() construction — dynamic bytecode creation",
    ),
    (
        re.compile(r"exec\s*\(\s*['\"]\\x[0-9a-fA-F]"),
        "HIGH",
        "exec() with hex-encoded string",
    ),
    (
        re.compile(r"exec\s*\(\s*bytes\.fromhex\s*\("),
        "HIGH",
        "exec(bytes.fromhex(...)) — hex-encoded payload",
    ),
    (
        re.compile(r"exec\s*\(\s*codecs\.decode\s*\("),
        "HIGH",
        "exec(codecs.decode(...)) — encoded payload execution",
    ),
    (
        re.compile(r"__import__\s*\(\s*['\"]os['\"]\s*\)\s*\.\s*system"),
        "HIGH",
        "Dynamic import + os.system — evasion pattern",
    ),
    (
        re.compile(r"getattr\s*\(\s*__import__"),
        "MEDIUM",
        "getattr(__import__(...)) — dynamic import evasion",
    ),
]

EXFILTRATION_PATTERNS = [
    (
        re.compile(
            r"requests\.(post|put)\s*\(\s*['\"]https?://(?!localhost|127\.0\.0\.1)"
        ),
        "HIGH",
        "HTTP POST/PUT to external URL — potential exfiltration",
    ),
    (
        re.compile(
            r"urllib\.request\.urlopen\s*\(\s*urllib\.request\.Request\s*\([^)]*method\s*=\s*['\"]POST"
        ),
        "HIGH",
        "urllib POST to external URL",
    ),
    (
        re.compile(r"subprocess\.\w+\s*\(\s*\[?\s*['\"]curl['\"].*-X\s*POST"),
        "HIGH",
        "subprocess curl POST — potential exfiltration",
    ),
    (
        re.compile(r"subprocess\.\w+\s*\(\s*\[?\s*['\"]wget['\"].*--post"),
        "HIGH",
        "subprocess wget POST",
    ),
    (
        re.compile(
            r"socket\.socket\s*\(.*\)\s*.*\.connect\s*\(\s*\(\s*['\"](?!127\.|localhost)"
        ),
        "MEDIUM",
        "Raw socket connection to external host",
    ),
]

CREDENTIAL_PATTERNS = []
for cp in CREDENTIAL_PATHS:
    CREDENTIAL_PATTERNS.append(
        (re.compile(cp), "MEDIUM", f"Reference to sensitive path: {cp}")
    )
for url in CLOUD_METADATA_URLS:
    CREDENTIAL_PATTERNS.append(
        (re.compile(url), "HIGH", f"Cloud metadata endpoint access: {url}")
    )

SUSPECT_SETUP_PATTERNS = [
    (
        re.compile(r"class\s+\w*(Install|PostInstall|Develop)\w*\s*\("),
        "MEDIUM",
        "Custom install command class — may run arbitrary code on pip install",
    ),
    (
        re.compile(r"cmdclass\s*=\s*\{"),
        "LOW",
        "Custom cmdclass in setup.py — verify no malicious install hooks",
    ),
]

LONG_BASE64_RE = re.compile(r"[A-Za-z0-9+/=]{200,}")


def should_skip_dir(dirname: str) -> bool:
    """Check if directory should be skipped (exact match or pattern)."""
    if dirname in SKIP_DIRS:
        return True
    dl = dirname.lower()
    return any(p in dl for p in SKIP_DIR_PATTERNS)


def entropy(data: bytes) -> float:
    """Shannon entropy of byte sequence."""
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    length = len(data)
    return -sum((c / length) * math.log2(c / length) for c in freq if c > 0)


def is_binary_file(path: Path) -> bool:
    """Heuristic: file is binary if first 8KB contain null bytes."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except (OSError, PermissionError):
        return False


def has_elf_pe_macho_header(path: Path) -> str | None:
    """Check for known executable file headers."""
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


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------


def scan_source_patterns(root: Path, result: ScanResult):
    """Scan source files for obfuscation, exfiltration, credential access."""
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
                for pattern, severity, desc in OBFUSCATION_PATTERNS:
                    if pattern.search(line):
                        result.add(
                            severity,
                            "OBFUSCATION",
                            str(fpath),
                            lineno,
                            desc,
                            line.strip(),
                        )

                for pattern, severity, desc in EXFILTRATION_PATTERNS:
                    if pattern.search(line):
                        result.add(
                            severity,
                            "EXFILTRATION",
                            str(fpath),
                            lineno,
                            desc,
                            line.strip(),
                        )

                for pattern, severity, desc in CREDENTIAL_PATTERNS:
                    if pattern.search(line):
                        is_test = any(
                            t in str(fpath).lower()
                            for t in ("test", "spec", "mock", "fixture", "example")
                        )
                        actual_sev = "LOW" if is_test else severity
                        result.add(
                            actual_sev,
                            "CREDENTIAL_ACCESS",
                            str(fpath),
                            lineno,
                            desc,
                            line.strip(),
                        )

            if suffix in (".py", ".pyw"):
                for pattern, severity, desc in SUSPECT_SETUP_PATTERNS:
                    if "setup" in fname.lower():
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

            b64_matches = LONG_BASE64_RE.findall(content)
            for match in b64_matches:
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
    """Detect unexpected binary/compiled files in source tree."""
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

            if not suffix and not in_build and fpath.stat().st_mode & stat.S_IXUSR:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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

    result = ScanResult()

    print(f"Scanning {root} ...")
    scan_source_patterns(root, result)
    scan_binary_anomalies(root, result)

    print_report(result, verbose=args.verbose)

    fail_threshold = SEVERITY_ORDER.get(args.fail_on, 1)
    worst = result.worst_severity
    if worst and SEVERITY_ORDER.get(worst, 99) <= fail_threshold:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
