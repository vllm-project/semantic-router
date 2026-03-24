#!/usr/bin/env python3
"""
AST-based Supply Chain Security Scanner — Tree-sitter powered, multi-language.

Three-layer architecture:
  Layer 1: AST diff extractor — structural change extraction from git diffs
  Layer 2: AST pattern matcher — context-aware malicious code detection
  Layer 3: Anomaly scorer — entropy + structural heuristics

Languages: Python, Go, JavaScript, TypeScript, Rust

Key advantage over regex: Tree-sitter parses real AST, so patterns inside
comments, string literals, docstrings, and variable names are never matched.

Exit codes:  0 — clean, 1 — findings detected, 2 — scanner error

References:
  - Apiiro malicious-code-ruleset (94.3% on PyPI, 88.4% on npm)
  - SCORE: Syntactic Code Representations (arXiv:2411.08182)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Tree-sitter language setup
# ---------------------------------------------------------------------------

LANG_REGISTRY: dict[str, object] = {}
EXT_TO_LANG_FAMILY: dict[str, str] = {}


def _init_languages():
    if LANG_REGISTRY:
        return
    import tree_sitter

    loaders = {
        ".py": ("tree_sitter_python", "language", "python"),
        ".pyw": ("tree_sitter_python", "language", "python"),
        ".go": ("tree_sitter_go", "language", "go"),
        ".js": ("tree_sitter_javascript", "language", "javascript"),
        ".mjs": ("tree_sitter_javascript", "language", "javascript"),
        ".cjs": ("tree_sitter_javascript", "language", "javascript"),
        ".jsx": ("tree_sitter_javascript", "language", "javascript"),
        ".ts": ("tree_sitter_typescript", "language_typescript", "typescript"),
        ".tsx": ("tree_sitter_typescript", "language_tsx", "typescript"),
        ".rs": ("tree_sitter_rust", "language", "rust"),
    }
    for ext, (mod_name, func_name, family) in loaders.items():
        try:
            mod = __import__(mod_name, fromlist=[func_name])
            lang_fn = getattr(mod, func_name)
            LANG_REGISTRY[ext] = tree_sitter.Language(lang_fn())
            EXT_TO_LANG_FAMILY[ext] = family
        except (ImportError, AttributeError, Exception):
            pass


def get_parser(ext: str):
    import tree_sitter

    _init_languages()
    lang = LANG_REGISTRY.get(ext)
    if lang is None:
        return None, None
    return tree_sitter.Parser(lang), lang


def lang_family(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    _init_languages()
    return EXT_TO_LANG_FAMILY.get(ext, "unknown")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Severity(IntEnum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Finding:
    severity: Severity
    category: str
    file: str
    line: int
    message: str
    snippet: str = ""
    ast_context: str = ""

    def __str__(self):
        loc = f"{self.file}:{self.line}" if self.line else self.file
        s = f"[{self.severity.name}] {self.category} — {loc}\n  {self.message}"
        if self.ast_context:
            s += f"\n  AST: {self.ast_context}"
        if self.snippet:
            s += f"\n  >>> {self.snippet[:200]}"
        return s

    def to_dict(self):
        return {
            "severity": self.severity.name,
            "category": self.category,
            "file": self.file,
            "line": self.line,
            "message": self.message,
            "snippet": self.snippet[:200],
            "ast_context": self.ast_context,
        }


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity, category, file, line, message, snippet="", ast_context=""):
        if isinstance(severity, str):
            severity = Severity[severity]
        self.findings.append(
            Finding(severity, category, file, line, message, snippet, ast_context)
        )

    @property
    def worst(self) -> Severity | None:
        return min((f.severity for f in self.findings), default=None)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for f in self.findings:
            c[f.severity.name] = c.get(f.severity.name, 0) + 1
        return c


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

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


def should_skip_dir(d: str) -> bool:
    if d in SKIP_DIRS:
        return True
    dl = d.lower()
    return any(p in dl for p in SKIP_DIR_PATTERNS)


def _is_own_source(filepath: str) -> bool:
    """Skip scanning the scanner's own source files."""
    try:
        return str(Path(filepath).resolve()).startswith(SELF_DIR)
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# AST helpers (language-agnostic)
# ---------------------------------------------------------------------------


def _call_name(node) -> str:
    """Full text of a call's function/callee node."""
    for field_name in ("function", "constructor", "macro"):
        fn = node.child_by_field_name(field_name)
        if fn is not None:
            return fn.text.decode(errors="replace").split("\n")[0].strip()
    return ""


def _ast_path(node) -> str:
    parts = []
    n = node
    while n and len(parts) < 5:
        if n.is_named:
            name_node = n.child_by_field_name("name")
            label = n.type
            if name_node:
                label += f"({name_node.text.decode(errors='replace')})"
            parts.append(label)
        n = n.parent
    return " > ".join(reversed(parts))


def _collect_string_values(node) -> list[str]:
    """Collect all string literal values from a subtree, across languages."""
    STRING_TYPES = {
        "string",
        "string_literal",
        "template_string",
        "string_content",
        "interpreted_string_literal",
        "raw_string_literal",
        "template_literal_type",
    }
    strings = []
    if node.type in STRING_TYPES:
        text = node.text.decode(errors="replace")
        if len(text) >= 2 and text[0] in ('"', "'", "`", "b"):
            text = text.strip("\"'`")
            if text.startswith("b'") or text.startswith('b"'):
                text = text[2:-1]
        strings.append(text)
    for child in node.children:
        strings.extend(_collect_string_values(child))
    return strings


def _find_nested_call(node, target_names: set) -> str | None:
    """Check if any descendant call matches target names."""
    if node.type in ("call", "call_expression"):
        name = _call_name(node)
        if name in target_names:
            return name
    for child in node.children:
        result = _find_nested_call(child, target_names)
        if result:
            return result
    return None


LONG_B64_RE = re.compile(r"[A-Za-z0-9+/=]{200,}")
KNOWN_TEST_DIRS = {
    "test",
    "tests",
    "spec",
    "specs",
    "mock",
    "mocks",
    "fixture",
    "fixtures",
    "example",
    "examples",
    "testdata",
    "testing",
    "e2e",
}


def _is_test_file(filepath: str) -> bool:
    parts = Path(filepath).parts
    fname = Path(filepath).stem.lower()
    return (
        any(p.lower() in KNOWN_TEST_DIRS for p in parts)
        or fname.endswith("_test")
        or fname.startswith("test_")
    )


# ---------------------------------------------------------------------------
# Credential / sensitive path patterns (shared across all languages)
# ---------------------------------------------------------------------------

CREDENTIAL_STRINGS = {
    "~/.ssh",
    "~/.aws",
    "~/.kube",
    "~/.docker",
    "~/.gnupg",
    "~/.config/gcloud",
    "~/.azure",
    "~/.npmrc",
    "~/.vault-token",
    "~/.netrc",
    "~/.gitconfig",
    "~/.git-credentials",
    "/etc/ssl/private",
    "/etc/kubernetes",
    ".bash_history",
    ".zsh_history",
    ".bitcoin",
    ".ethereum",
    ".solana",
    "169.254.169.254",
    "metadata.google.internal",
    "/home/user/.ssh",
    "/root/.ssh",
    "/root/.aws",
}

# ---------------------------------------------------------------------------
# Language-specific pattern definitions
# ---------------------------------------------------------------------------

# Python: exec/eval/compile with obfuscated args
PY_DYNAMIC_EXEC = {"exec", "eval", "compile"}
PY_DECODE_FUNCS = {
    "base64.b64decode",
    "base64.decodebytes",
    "zlib.decompress",
    "gzip.decompress",
    "marshal.loads",
    "pickle.loads",
    "dill.loads",
    "codecs.decode",
    "bytes.fromhex",
}
PY_PROCESS_SPAWN = {
    "os.system",
    "os.popen",
    "os.execv",
    "os.execvp",
    "os.execve",
    "subprocess.call",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.check_output",
    "subprocess.check_call",
}
PY_HTTP_EXFIL = {"requests.post", "requests.put", "urllib.request.urlopen"}

# Go: exec.Command, os.ReadFile, http.Post, reflect evasion
GO_PROCESS_SPAWN = {"exec.Command", "exec.CommandContext"}
GO_FILE_READ = {"os.ReadFile", "os.Open", "ioutil.ReadFile"}
GO_HTTP_EXFIL = {"http.Post", "http.NewRequest"}
GO_REFLECT_EVASION = {"reflect.ValueOf"}

# JavaScript/TypeScript: eval, Function, child_process, fetch
JS_DYNAMIC_EXEC = {"eval"}
JS_CONSTRUCTOR_EXEC = {"Function"}
JS_DECODE_FUNCS = {"atob", "Buffer.from"}
JS_PROCESS_SPAWN = {
    "exec",
    "execSync",
    "spawn",
    "spawnSync",
    "execFile",
    "execFileSync",
}
JS_HTTP_EXFIL = {"fetch", "axios.post", "axios.put"}
JS_DANGEROUS_REQUIRE = {"child_process", "child" + "_process"}

# Rust: Command::new, fs::read_to_string, reqwest
RS_PROCESS_SPAWN = {
    "Command::new",
    "process::Command::new",
    "std::process::Command::new",
}
RS_FILE_READ = {
    "fs::read_to_string",
    "fs::read",
    "std::fs::read_to_string",
    "std::fs::read",
}
RS_HTTP_EXFIL_METHODS = {".post", ".put"}

NETWORK_TOOLS = {"curl", "wget", "nc", "ncat"}

# ---------------------------------------------------------------------------
# URL default-deny allowlist
# ---------------------------------------------------------------------------
# Known exfiltration service URL patterns — always CRITICAL regardless of
# allowlist.  These are nearly never legitimate inside application code.
EXFIL_SERVICE_PATTERNS = [
    re.compile(r"discord(?:app)?\.com/api/webhooks/"),
    re.compile(r"api\.telegram\.org/bot"),
    re.compile(r"hooks\.slack\.com/services/"),
    re.compile(r"pastebin\.com/api/"),
    re.compile(r"paste\.ee/"),
    re.compile(r"hastebin\.com/"),
    re.compile(r"transfer\.sh/"),
    re.compile(r"file\.io/"),
    re.compile(r"requestbin\."),
    re.compile(r"pipedream\.net/"),
    re.compile(r"webhook\.site/"),
    re.compile(r"ngrok\.io/"),
    re.compile(r"[a-z0-9]+\.ngrok-free\.app"),
    re.compile(r"burpcollaborator\.net/"),
    re.compile(r"oastify\.com/"),
    re.compile(r"interact\.sh/"),
    re.compile(r"mockbin\.(com|org)/"),
    re.compile(r"beeceptor\.com/"),
    re.compile(r"hookbin\.com/"),
    re.compile(r"canarytokens\."),
]

# TLDs overrepresented in supply chain malware campaigns
SUSPICIOUS_TLDS = {
    ".xyz",
    ".top",
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
    ".pw",
    ".cc",
    ".buzz",
    ".icu",
    ".rest",
    ".cam",
    ".su",
    ".ws",
}

# Default-allow: domains commonly referenced from legitimate codebases.
# Projects extend this via a `.security-scan-allowlist` file in repo root
# (one domain per line, `#` for comments).
ALLOWED_URL_DOMAINS = {
    # Loopback / local
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "[::1]",
    # Package registries
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "crates.io",
    "pkg.go.dev",
    "rubygems.org",
    "repo1.maven.org",
    "central.sonatype.com",
    # Code hosting
    "github.com",
    "githubusercontent.com",
    "api.github.com",
    "github.io",
    "githubassets.com",
    "gitlab.com",
    "bitbucket.org",
    # Cloud providers
    "amazonaws.com",
    "s3.amazonaws.com",
    "awsstatic.com",
    "googleapis.com",
    "storage.googleapis.com",
    "cloud.google.com",
    "azure.com",
    "blob.core.windows.net",
    "azureedge.net",
    "azurewebsites.net",
    "microsoft.com",
    "windows.net",
    "cloudflare.com",
    "cloudflare-dns.com",
    "cloudfront.net",
    "akamaized.net",
    "digitalocean.com",
    "spaces.digitaloceanspaces.com",
    # CDNs
    "cdn.jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "esm.sh",
    "esm.run",
    "cdn.skypack.dev",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    # Documentation / specs / academic
    "python.org",
    "docs.python.org",
    "nodejs.org",
    "docs.rs",
    "go.dev",
    "golang.org",
    "readthedocs.io",
    "readthedocs.org",
    "wikipedia.org",
    "wikimedia.org",
    "stackoverflow.com",
    "stackexchange.com",
    "w3.org",
    "w3schools.com",
    "ietf.org",
    "schema.org",
    "json-schema.org",
    "json.org",
    "yaml.org",
    "toml.io",
    "semver.org",
    "developer.mozilla.org",
    "mozilla.org",
    "swagger.io",
    "openapis.org",
    "arxiv.org",
    "doi.org",
    "acm.org",
    "ieee.org",
    "creativecommons.org",
    "contributor-covenant.org",
    # Container registries
    "docker.io",
    "docker.com",
    "registry.hub.docker.com",
    "ghcr.io",
    "quay.io",
    "gcr.io",
    "k8s.io",
    # CI/CD & code quality
    "codecov.io",
    "coveralls.io",
    "circleci.com",
    "travis-ci.org",
    "travis-ci.com",
    "sonarcloud.io",
    "sonarqube.org",
    # ML/AI services
    "huggingface.co",
    "hf.co",
    "openai.com",
    "api.openai.com",
    "anthropic.com",
    "api.anthropic.com",
    "ai.google.dev",
    "generativelanguage.googleapis.com",
    # Auth providers
    "auth0.com",
    "okta.com",
    "accounts.google.com",
    # Monitoring
    "sentry.io",
    "datadoghq.com",
    "newrelic.com",
    # Common application domains
    "google.com",
    "apple.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "duckduckgo.com",
    "bing.com",
    # Container / infra
    "podman.io",
    "kubernetes.io",
    "helm.sh",
    "terraform.io",
    "hashicorp.com",
    "jenkins.io",
    "gradle.org",
    # Example / test
    "example.com",
    "example.org",
    "example.net",
    "httpbin.org",
    "jsonplaceholder.typicode.com",
}

_url_allowlist: set[str] = set()


def _load_url_allowlist(root: Path):
    """Load URL domain allowlist: built-in defaults + project overrides."""
    global _url_allowlist
    _url_allowlist = {d.lower() for d in ALLOWED_URL_DOMAINS}
    allowlist_file = root / ".security-scan-allowlist"
    if allowlist_file.is_file():
        try:
            for line in allowlist_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    _url_allowlist.add(line.lower())
        except (OSError, PermissionError):
            pass


_TEMPLATE_URL_RE = re.compile(r"[{$%]")


def _is_url_allowed(url_or_domain: str) -> bool:
    """Check if a URL or domain is in the allowlist (supports suffix matching).
    Template/interpolated URLs (containing {, $, %s) are always allowed since
    the actual target is determined at runtime."""
    m = re.search(r"https?://([^/:\s\"'`]+)", url_or_domain)
    domain = m.group(1).lower() if m else url_or_domain.lower()
    if _TEMPLATE_URL_RE.search(domain):
        return True
    if "." not in domain:
        return True
    if domain in _url_allowlist:
        return True
    return any(domain.endswith("." + a) for a in _url_allowlist)


# Python: setup.py hook base classes (cmdclass overrides)
PY_SETUP_HOOK_BASES = {
    "install",
    "develop",
    "egg_info",
    "build_py",
    "build_ext",
    "sdist",
    "bdist_wheel",
}
PY_DOWNLOAD_FUNCS = {
    "urllib.request.urlretrieve",
    "urllib.urlretrieve",
    "requests.get",
    "httpx.get",
}
PY_DNS_FUNCS = {
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "dns.resolver.query",
    "dns.resolver.resolve",
}

# package.json: suspicious patterns inside lifecycle scripts
PKG_JSON_SUSPICIOUS = [
    re.compile(r"\beval\b"),
    re.compile(r"\bFunction\b"),
    re.compile(r"\bbase64\b"),
    re.compile(r"\batob\b"),
    re.compile(r"\bcurl\s"),
    re.compile(r"\bwget\s"),
    re.compile(r"\bnc\s"),
    re.compile(r"\bnode\s+-e\b"),
    re.compile(r"\bpython[3]?\s+-c\b"),
    re.compile(r"\bpowershell\b", re.IGNORECASE),
    re.compile(r"\bbash\s+-c\b"),
    re.compile(r"\bsh\s+-c\b"),
    re.compile(r"https?://"),
]


# ---------------------------------------------------------------------------
# Layer 1: AST Diff Extractor
# ---------------------------------------------------------------------------


@dataclass
class StructuralChange:
    file: str
    kind: str
    name: str
    change_type: str
    start_line: int
    end_line: int
    source: str = ""


def parse_unified_diff(diff_text: str) -> dict[str, list[tuple[int, str]]]:
    result: dict[str, list[tuple[int, str]]] = {}
    current_file = None
    current_line = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            if current_file not in result:
                result[current_file] = []
        elif line.startswith("@@ "):
            m = re.search(r"\+(\d+)", line)
            if m:
                current_line = int(m.group(1)) - 1
        elif line.startswith("+") and not line.startswith("+++"):
            current_line += 1
            if current_file is not None:
                result[current_file].append((current_line, line[1:]))
        elif not line.startswith("-"):
            current_line += 1
    return result


ENTITY_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "class_definition",
    "class_declaration",
    "function_item",
    "impl_item",
    "struct_item",
    "import_statement",
    "import_from_statement",
    "lexical_declaration",
    "variable_declaration",
    "expression_statement",
    "short_var_declaration",
}


def extract_structural_changes(diff_text: str, repo_root: str = "."):
    file_additions = parse_unified_diff(diff_text)
    changes: list[StructuralChange] = []
    for filepath, added_lines in file_additions.items():
        if not added_lines:
            continue
        ext = Path(filepath).suffix.lower()
        parser, _lang = get_parser(ext)
        full_path = Path(repo_root) / filepath
        if not full_path.exists():
            continue
        if parser is None:
            for ln, content in added_lines:
                changes.append(
                    StructuralChange(filepath, "raw_line", "", "added", ln, ln, content)
                )
            continue
        try:
            source = full_path.read_bytes()
            if len(source) > MAX_FILE_SIZE:
                continue
        except (OSError, PermissionError):
            continue
        tree = parser.parse(source)
        added_set = {ln for ln, _ in added_lines}
        seen = set()
        for child in tree.root_node.children:
            if child.type not in ENTITY_TYPES and not child.is_named:
                continue
            child_lines = set(range(child.start_point.row + 1, child.end_point.row + 2))
            overlap = added_set & child_lines
            if not overlap:
                continue
            name_node = child.child_by_field_name("name")
            name = name_node.text.decode(errors="replace") if name_node else ""
            kind = child.type.replace("_definition", "").replace("_declaration", "")
            kind = kind.replace("_statement", "").replace("_item", "")
            rng = (child.start_point.row, child.end_point.row)
            if rng in seen:
                continue
            seen.add(rng)
            ct = "added" if overlap == child_lines else "modified"
            changes.append(
                StructuralChange(
                    filepath,
                    kind,
                    name,
                    ct,
                    child.start_point.row + 1,
                    child.end_point.row + 1,
                    child.text.decode(errors="replace")[:2000],
                )
            )
    return changes


# ---------------------------------------------------------------------------
# Layer 2: AST Pattern Matcher — multi-language
# ---------------------------------------------------------------------------


def _check_network_tool_exfil(fn_name: str, node, filepath: str, result: ScanResult):
    """Shared: detect subprocess calls spawning curl/wget with POST data."""
    args = node.child_by_field_name("arguments")
    if not args:
        return
    text = args.text.decode(errors="replace").lower()
    for tool in NETWORK_TOOLS:
        if tool in text and any(kw in text for kw in ("post", "--data", "-d ")):
            result.add(
                Severity.HIGH,
                "AST_SUBPROCESS_EXFIL",
                filepath,
                node.start_point.row + 1,
                f"{fn_name}() spawning {tool} with POST data — "
                "potential exfiltration",
                node.text.decode(errors="replace")[:150],
                _ast_path(node),
            )
            return


def _check_credential_strings(node, filepath: str, is_test: bool, result: ScanResult):
    """Shared: flag string literals referencing sensitive paths."""
    STRING_TYPES = {
        "string",
        "string_literal",
        "interpreted_string_literal",
        "raw_string_literal",
        "string_content",
        "template_string",
    }
    if node.type not in STRING_TYPES or is_test:
        return
    text = node.text.decode(errors="replace")
    for cred in CREDENTIAL_STRINGS:
        if cred in text:
            sev = (
                Severity.HIGH
                if ("169.254.169.254" in text or "metadata.google" in text)
                else Severity.MEDIUM
            )
            result.add(
                sev,
                "AST_CREDENTIAL_REF",
                filepath,
                node.start_point.row + 1,
                f"String references sensitive path: {cred}",
                text[:150],
                _ast_path(node),
            )
            return


def _check_long_base64(node, filepath: str, result: ScanResult):
    """Shared: flag very long base64 strings in literals."""
    STRING_TYPES = {
        "string",
        "string_literal",
        "interpreted_string_literal",
        "raw_string_literal",
        "concatenated_string",
        "template_string",
    }
    if node.type not in STRING_TYPES:
        return
    text = node.text.decode(errors="replace")
    for m in LONG_B64_RE.finditer(text):
        if len(m.group()) > 500:
            result.add(
                Severity.HIGH,
                "AST_LONG_BASE64",
                filepath,
                node.start_point.row + 1,
                f"Long base64 string ({len(m.group())} chars) in literal",
                m.group()[:120] + "...",
                _ast_path(node),
            )
            return


_URL_STRING_TYPES = {
    "string",
    "string_literal",
    "interpreted_string_literal",
    "raw_string_literal",
    "string_content",
    "template_string",
}
_SCHEME_RE = re.compile(r"https?://")


def _check_url(node, filepath: str, is_test: bool, result: ScanResult):
    """Default-deny URL check for string literals.

    Tiered severity:
      CRITICAL — known exfiltration service (always, bypasses allowlist)
      HIGH     — suspicious TLD and not in allowlist
      MEDIUM   — any URL domain not in project allowlist
    """
    if node.type not in _URL_STRING_TYPES or is_test:
        return
    text = node.text.decode(errors="replace")
    if not _SCHEME_RE.search(text):
        return

    # Known exfil services → CRITICAL (bypass allowlist entirely)
    for pat in EXFIL_SERVICE_PATTERNS:
        if pat.search(text):
            result.add(
                Severity.CRITICAL,
                "URL_KNOWN_EXFIL_SERVICE",
                filepath,
                node.start_point.row + 1,
                f"URL targets known exfiltration service "
                f"({pat.pattern.split('.')[0]})",
                text[:150],
                _ast_path(node),
            )
            return

    # Extract domain
    url_m = re.search(r"https?://([^/:\s\"'`]+)", text)
    if not url_m:
        return
    domain = url_m.group(1).lower()
    url_short = url_m.group()[:80]

    # Allowlisted → pass
    if _is_url_allowed(domain):
        return

    # Suspicious TLD → HIGH
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            result.add(
                Severity.HIGH,
                "URL_SUSPICIOUS_TLD",
                filepath,
                node.start_point.row + 1,
                f"URL not in allowlist, suspicious TLD ({tld}): " f"{url_short}",
                text[:150],
                _ast_path(node),
            )
            return

    # Not in allowlist → MEDIUM
    result.add(
        Severity.MEDIUM,
        "URL_NOT_ALLOWLISTED",
        filepath,
        node.start_point.row + 1,
        f"URL domain not in allowlist: {domain} "
        f"(add to .security-scan-allowlist to suppress)",
        text[:150],
        _ast_path(node),
    )


# --- Python-specific patterns ---


def _scan_python_call(
    node, fn_name: str, filepath: str, is_test: bool, result: ScanResult
):
    bare = fn_name.rsplit(".", maxsplit=1)[-1] if "." in fn_name else fn_name

    if bare in PY_DYNAMIC_EXEC:
        args = node.child_by_field_name("arguments")
        if args:
            nested = _find_nested_call(args, PY_DECODE_FUNCS)
            if nested:
                result.add(
                    Severity.CRITICAL,
                    "AST_OBFUSCATED_EXEC",
                    filepath,
                    node.start_point.row + 1,
                    f"{bare}({nested}(...)) — dynamic execution of "
                    "decoded/decompressed payload",
                    node.text.decode(errors="replace")[:150],
                    _ast_path(node),
                )
            else:
                for s in _collect_string_values(args):
                    if any(len(m) > 100 for m in LONG_B64_RE.findall(s)):
                        result.add(
                            Severity.CRITICAL,
                            "AST_EXEC_LONG_ENCODED",
                            filepath,
                            node.start_point.row + 1,
                            f"{bare}() with long encoded string",
                            node.text.decode(errors="replace")[:150],
                            _ast_path(node),
                        )
                        break

    if bare == "__import__":
        gp = node.parent
        if gp and gp.type in ("attribute", "subscript", "argument_list"):
            ctx = gp.parent if gp.parent else gp
            result.add(
                Severity.HIGH,
                "AST_DYNAMIC_IMPORT",
                filepath,
                node.start_point.row + 1,
                "Dynamic __import__ with attribute/call access — evasion",
                ctx.text.decode(errors="replace")[:150],
                _ast_path(node),
            )

    if fn_name in ("types.CodeType",):
        result.add(
            Severity.HIGH,
            "AST_BYTECODE_INJECT",
            filepath,
            node.start_point.row + 1,
            "types.CodeType() — dynamic bytecode creation",
            node.text.decode(errors="replace")[:150],
            _ast_path(node),
        )

    if fn_name == "marshal.loads":
        result.add(
            Severity.HIGH,
            "AST_MARSHAL_LOADS",
            filepath,
            node.start_point.row + 1,
            "marshal.loads() — bytecode deserialization",
            node.text.decode(errors="replace")[:150],
            _ast_path(node),
        )

    if fn_name in PY_HTTP_EXFIL and not is_test:
        args = node.child_by_field_name("arguments")
        if args:
            for s in _collect_string_values(args):
                if _SCHEME_RE.match(s) and not _is_url_allowed(s):
                    result.add(
                        Severity.HIGH,
                        "AST_HTTP_EXFIL",
                        filepath,
                        node.start_point.row + 1,
                        f"{fn_name}() to non-allowlisted URL",
                        node.text.decode(errors="replace")[:150],
                        _ast_path(node),
                    )
                    break

    if fn_name in PY_PROCESS_SPAWN:
        _check_network_tool_exfil(fn_name, node, filepath, result)

    # DNS exfiltration: long/encoded hostnames or concatenated args in DNS lookups
    if fn_name in PY_DNS_FUNCS:
        args = node.child_by_field_name("arguments")
        if args:
            args_text = args.text.decode(errors="replace")
            has_concat = "+" in args_text
            for s in _collect_string_values(args):
                if len(s) > 60 or LONG_B64_RE.search(s):
                    result.add(
                        Severity.HIGH,
                        "AST_DNS_EXFIL",
                        filepath,
                        node.start_point.row + 1,
                        f"{fn_name}() with long/encoded hostname — "
                        "possible DNS exfiltration",
                        node.text.decode(errors="replace")[:150],
                        _ast_path(node),
                    )
                    break
            else:
                if has_concat:
                    result.add(
                        Severity.MEDIUM,
                        "AST_DNS_EXFIL",
                        filepath,
                        node.start_point.row + 1,
                        f"{fn_name}() with concatenated hostname — "
                        "possible DNS exfiltration via encoded subdomains",
                        node.text.decode(errors="replace")[:150],
                        _ast_path(node),
                    )

    # os.environ bulk serialization → exfiltration
    if fn_name in ("json.dumps", "json.dump", "str", "repr") and not is_test:
        args = node.child_by_field_name("arguments")
        if args and "os.environ" in args.text.decode(errors="replace"):
            result.add(
                Severity.HIGH,
                "AST_ENV_SERIALIZE",
                filepath,
                node.start_point.row + 1,
                f"{fn_name}(os.environ...) — bulk environment "
                "serialization, possible credential exfiltration",
                node.text.decode(errors="replace")[:150],
                _ast_path(node),
            )

    # os.environ in HTTP call arguments → direct exfiltration
    if fn_name in PY_HTTP_EXFIL and not is_test:
        args = node.child_by_field_name("arguments")
        if args and "os.environ" in args.text.decode(errors="replace"):
            result.add(
                Severity.CRITICAL,
                "AST_ENV_HTTP_EXFIL",
                filepath,
                node.start_point.row + 1,
                f"{fn_name}() with os.environ — sending environment "
                "variables to external endpoint",
                node.text.decode(errors="replace")[:150],
                _ast_path(node),
            )


def _check_setup_py_class(node, filepath: str, result: ScanResult):
    """Detect setup.py install command overrides (#1 PyPI malware vector)."""
    bases = node.child_by_field_name("superclasses")
    if not bases:
        return
    bases_text = bases.text.decode(errors="replace")
    for hook_base in PY_SETUP_HOOK_BASES:
        if hook_base not in bases_text:
            continue
        body = node.text.decode(errors="replace")
        suspicious = [
            s
            for s in (
                "os.system",
                "subprocess",
                "exec(",
                "eval(",
                "Popen",
                "urlopen",
                "urlretrieve",
                "requests.",
                "curl",
                "wget",
                "base64",
            )
            if s in body
        ]
        if suspicious:
            result.add(
                Severity.CRITICAL,
                "SETUP_PY_CMD_OVERWRITE",
                filepath,
                node.start_point.row + 1,
                f"setup.py overrides '{hook_base}' command with "
                f"suspicious calls ({', '.join(suspicious[:3])}) — "
                "auto-executes on pip install",
                body[:200],
                _ast_path(node),
            )
        else:
            result.add(
                Severity.MEDIUM,
                "SETUP_PY_CMD_OVERRIDE",
                filepath,
                node.start_point.row + 1,
                f"setup.py overrides '{hook_base}' command — "
                "review for malicious install hooks",
                body[:200],
                _ast_path(node),
            )
        return


def _check_setup_py_module_exec(node, filepath: str, result: ScanResult):
    """Detect module-level exec/eval/os.system in setup.py."""
    text = node.text.decode(errors="replace")
    for danger in (
        "exec(",
        "eval(",
        "os.system(",
        "subprocess.",
        "Popen(",
        "__import__(",
    ):
        if danger in text:
            result.add(
                Severity.HIGH,
                "SETUP_PY_MODULE_EXEC",
                filepath,
                node.start_point.row + 1,
                f"Module-level {danger.rstrip('(')} in setup.py — "
                "executes on pip install",
                text[:200],
                _ast_path(node),
            )
            return


def _check_download_and_execute(node, filepath: str, result: ScanResult):
    """Detect functions that both download content and execute commands."""
    if _is_test_file(filepath):
        return
    body = node.text.decode(errors="replace")
    has_download = any(
        f in body
        for f in (
            "urlretrieve(",
            "urlopen(",
            "requests.get(",
            "httpx.get(",
            ".download(",
            'wget "',
            "wget '",
            'curl "',
            "curl '",
            "curl -",
        )
    )
    has_execute = any(
        f in body
        for f in (
            "os.system(",
            "os.popen(",
            "os.chmod(",
            "os.execv(",
            "Popen(",
            "check_output(",
            "check_call(",
        )
    )
    if has_download and has_execute:
        name_node = node.child_by_field_name("name")
        name = name_node.text.decode(errors="replace") if name_node else "<anon>"
        result.add(
            Severity.HIGH,
            "AST_DOWNLOAD_EXECUTE",
            filepath,
            node.start_point.row + 1,
            f"Function '{name}' contains both download and execute "
            "patterns — possible dropper",
            body[:200],
            _ast_path(node),
        )


def _scan_conftest_py(filepath: str, source: bytes, result: ScanResult):
    """Flag conftest.py with network/exec — pytest plugin injection vector."""
    body = source.decode(errors="replace")
    suspicious = []
    if any(s in body for s in ("requests.", "urllib", "httpx.", "aiohttp")):
        suspicious.append("HTTP client")
    if any(s in body for s in ("subprocess", "os.system", "os.popen")):
        suspicious.append("process execution")
    if "exec(" in body or "eval(" in body:
        suspicious.append("dynamic execution")
    if "os.environ" in body:
        suspicious.append("env var access")
    for cred in ("~/.ssh", "~/.aws", "~/.kube", "/root/.ssh"):
        if cred in body:
            suspicious.append("credential path")
            break
    if suspicious:
        result.add(
            Severity.MEDIUM,
            "CONFTEST_SUSPICIOUS",
            filepath,
            1,
            f"conftest.py with suspicious patterns: "
            f"{', '.join(suspicious)} — auto-loaded by pytest",
            body[:200],
        )


# --- Go-specific patterns ---


def _scan_go_call(node, fn_name: str, filepath: str, is_test: bool, result: ScanResult):
    if fn_name in GO_PROCESS_SPAWN:
        args = node.child_by_field_name("arguments")
        if args:
            strings = _collect_string_values(args)
            text_lower = args.text.decode(errors="replace").lower()
            for s in strings:
                sl = s.lower()
                if sl in ("sh", "bash", "cmd", "/bin/sh", "/bin/bash"):
                    for s2 in strings:
                        if any(t in s2.lower() for t in NETWORK_TOOLS):
                            result.add(
                                Severity.HIGH,
                                "AST_GO_SHELL_EXEC",
                                filepath,
                                node.start_point.row + 1,
                                f"{fn_name}() shell with network tool — "
                                "potential exfiltration",
                                node.text.decode(errors="replace")[:150],
                                _ast_path(node),
                            )
                            return
            for tool in NETWORK_TOOLS:
                if tool in text_lower and any(
                    kw in text_lower for kw in ("post", "--data", "-d ")
                ):
                    result.add(
                        Severity.HIGH,
                        "AST_GO_SUBPROCESS_EXFIL",
                        filepath,
                        node.start_point.row + 1,
                        f"{fn_name}() spawning {tool} with POST",
                        node.text.decode(errors="replace")[:150],
                        _ast_path(node),
                    )
                    return
            if any(
                s.lower() in ("sh", "bash", "/bin/sh", "/bin/bash") for s in strings
            ):
                for s in strings:
                    if "|" in s and any(t in s.lower() for t in NETWORK_TOOLS):
                        result.add(
                            Severity.HIGH,
                            "AST_GO_PIPE_EXFIL",
                            filepath,
                            node.start_point.row + 1,
                            f"{fn_name}() piped shell command with network tool",
                            node.text.decode(errors="replace")[:150],
                            _ast_path(node),
                        )
                        return

    if fn_name in GO_FILE_READ and not is_test:
        args = node.child_by_field_name("arguments")
        if args:
            for s in _collect_string_values(args):
                for cred in CREDENTIAL_STRINGS:
                    if cred in s:
                        result.add(
                            Severity.HIGH,
                            "AST_GO_CRED_READ",
                            filepath,
                            node.start_point.row + 1,
                            f"{fn_name}() reading credential path: {s}",
                            node.text.decode(errors="replace")[:150],
                            _ast_path(node),
                        )
                        return

    if fn_name in GO_HTTP_EXFIL and not is_test:
        args = node.child_by_field_name("arguments")
        if args:
            for s in _collect_string_values(args):
                if _SCHEME_RE.match(s) and not _is_url_allowed(s):
                    result.add(
                        Severity.HIGH,
                        "AST_GO_HTTP_EXFIL",
                        filepath,
                        node.start_point.row + 1,
                        f"{fn_name}() to non-allowlisted URL: {s[:60]}",
                        node.text.decode(errors="replace")[:150],
                        _ast_path(node),
                    )
                    return

    if fn_name.startswith("reflect.ValueOf") and not is_test:
        full = node.text.decode(errors="replace")
        if "MethodByName" in full:
            result.add(
                Severity.HIGH,
                "AST_GO_REFLECT_EVASION",
                filepath,
                node.start_point.row + 1,
                "reflect.ValueOf().MethodByName() — dynamic dispatch evasion",
                full[:150],
                _ast_path(node),
            )


def _scan_go_init_func(node, filepath: str, result: ScanResult):
    """Flag Go init() with network/exec/credential access — runs on import."""
    name_node = node.child_by_field_name("name")
    if not name_node or name_node.text.decode(errors="replace") != "init":
        return
    body = node.text.decode(errors="replace")
    suspicious = []
    if "exec.Command" in body or "exec.CommandContext" in body:
        suspicious.append("process execution")
    if any(s in body for s in ("http.Post", "http.Get", "http.NewRequest")):
        suspicious.append("HTTP request")
    if any(s in body for s in ("net.Dial", "net.Listen")):
        suspicious.append("network socket")
    if any(s in body for s in ("os.WriteFile", "os.Create", "os.OpenFile")):
        suspicious.append("file write")
    for cred in CREDENTIAL_STRINGS:
        if cred in body:
            suspicious.append(f"credential reference ({cred})")
            break
    if suspicious:
        result.add(
            Severity.HIGH,
            "AST_GO_INIT_SUSPICIOUS",
            filepath,
            node.start_point.row + 1,
            f"Go init() with: {', '.join(suspicious)} — "
            "auto-executes on package import",
            body[:200],
            _ast_path(node),
        )


# --- JavaScript/TypeScript-specific patterns ---


def _scan_js_call(node, fn_name: str, filepath: str, is_test: bool, result: ScanResult):
    bare = fn_name.rsplit(".", maxsplit=1)[-1] if "." in fn_name else fn_name

    if bare in JS_DYNAMIC_EXEC:
        args = node.child_by_field_name("arguments")
        if args:
            nested = _find_nested_call(args, JS_DECODE_FUNCS)
            if nested:
                result.add(
                    Severity.CRITICAL,
                    "AST_JS_EVAL_DECODE",
                    filepath,
                    node.start_point.row + 1,
                    f"{bare}({nested}(...)) — eval of decoded payload",
                    node.text.decode(errors="replace")[:150],
                    _ast_path(node),
                )
            else:
                for s in _collect_string_values(args):
                    if any(len(m) > 100 for m in LONG_B64_RE.findall(s)):
                        result.add(
                            Severity.CRITICAL,
                            "AST_JS_EVAL_ENCODED",
                            filepath,
                            node.start_point.row + 1,
                            f"{bare}() with long encoded string",
                            node.text.decode(errors="replace")[:150],
                            _ast_path(node),
                        )
                        break

    if bare == "require":
        args = node.child_by_field_name("arguments")
        if args:
            for s in _collect_string_values(args):
                if s in JS_DANGEROUS_REQUIRE or "child_process" in s:
                    result.add(
                        Severity.MEDIUM,
                        "AST_JS_CHILD_PROCESS",
                        filepath,
                        node.start_point.row + 1,
                        f"require('{s}') — child process module import",
                        node.text.decode(errors="replace")[:150],
                        _ast_path(node),
                    )
            text = args.text.decode(errors="replace")
            if "+" in text and "process" in text.lower():
                result.add(
                    Severity.HIGH,
                    "AST_JS_DYNAMIC_REQUIRE",
                    filepath,
                    node.start_point.row + 1,
                    "Dynamic require() with string concatenation — evasion",
                    node.text.decode(errors="replace")[:150],
                    _ast_path(node),
                )

    if bare in JS_PROCESS_SPAWN:
        args = node.child_by_field_name("arguments")
        if args:
            text = args.text.decode(errors="replace").lower()
            for tool in NETWORK_TOOLS:
                if tool in text and any(kw in text for kw in ("post", "--data", "-d ")):
                    result.add(
                        Severity.HIGH,
                        "AST_JS_SUBPROCESS_EXFIL",
                        filepath,
                        node.start_point.row + 1,
                        f"{bare}() spawning {tool} with POST",
                        node.text.decode(errors="replace")[:150],
                        _ast_path(node),
                    )
                    return

    if fn_name in JS_HTTP_EXFIL and not is_test:
        args = node.child_by_field_name("arguments")
        if args:
            for s in _collect_string_values(args):
                if _SCHEME_RE.match(s) and not _is_url_allowed(s):
                    result.add(
                        Severity.HIGH,
                        "AST_JS_HTTP_EXFIL",
                        filepath,
                        node.start_point.row + 1,
                        f"{fn_name}() to non-allowlisted URL",
                        node.text.decode(errors="replace")[:150],
                        _ast_path(node),
                    )
                    return

    if fn_name == "process.env" and not is_test:
        parent = node.parent
        if parent and parent.type not in ("member_expression", "subscript_expression"):
            result.add(
                Severity.MEDIUM,
                "AST_JS_BULK_ENV",
                filepath,
                node.start_point.row + 1,
                "Bulk process.env access — possible credential harvesting",
                node.text.decode(errors="replace")[:150],
                _ast_path(node),
            )


def _scan_js_new(node, filepath: str, is_test: bool, result: ScanResult):
    """Handle `new Function(...)` constructor pattern."""
    constructor = node.child_by_field_name("constructor")
    if constructor is None:
        return
    name = constructor.text.decode(errors="replace")
    if name == "Function":
        args = node.child_by_field_name("arguments")
        if args:
            nested = _find_nested_call(args, JS_DECODE_FUNCS)
            if nested:
                result.add(
                    Severity.CRITICAL,
                    "AST_JS_FUNCTION_DECODE",
                    filepath,
                    node.start_point.row + 1,
                    f"new Function({nested}(...)) — Function constructor "
                    "with decoded payload",
                    node.text.decode(errors="replace")[:150],
                    _ast_path(node),
                )
            else:
                for s in _collect_string_values(args):
                    if any(len(m) > 100 for m in LONG_B64_RE.findall(s)):
                        result.add(
                            Severity.CRITICAL,
                            "AST_JS_FUNCTION_ENCODED",
                            filepath,
                            node.start_point.row + 1,
                            "new Function() with long encoded string",
                            node.text.decode(errors="replace")[:150],
                            _ast_path(node),
                        )
                        break


# --- Rust-specific patterns ---


def _scan_rust_call(
    node, fn_name: str, filepath: str, is_test: bool, result: ScanResult
):
    if fn_name in RS_PROCESS_SPAWN:
        args = node.child_by_field_name("arguments")
        full_text = node.text.decode(errors="replace")
        if args:
            strings = _collect_string_values(args)
            for s in strings:
                sl = s.lower()
                if sl in ("sh", "bash", "/bin/sh", "/bin/bash") and any(
                    t in full_text.lower() for t in NETWORK_TOOLS
                ):
                    result.add(
                        Severity.HIGH,
                        "AST_RS_SHELL_EXEC",
                        filepath,
                        node.start_point.row + 1,
                        f"{fn_name}() shell exec with network tool",
                        full_text[:150],
                        _ast_path(node),
                    )
                    return
                for tool in NETWORK_TOOLS:
                    if tool == sl and any(
                        kw in full_text.lower() for kw in ("post", "--data", "-d ")
                    ):
                        result.add(
                            Severity.HIGH,
                            "AST_RS_SUBPROCESS_EXFIL",
                            filepath,
                            node.start_point.row + 1,
                            f"{fn_name}() spawning {tool} with POST",
                            full_text[:150],
                            _ast_path(node),
                        )
                        return

    if fn_name in RS_FILE_READ and not is_test:
        args = node.child_by_field_name("arguments")
        if args:
            for s in _collect_string_values(args):
                for cred in CREDENTIAL_STRINGS:
                    if cred in s:
                        result.add(
                            Severity.HIGH,
                            "AST_RS_CRED_READ",
                            filepath,
                            node.start_point.row + 1,
                            f"{fn_name}() reading credential path: {s}",
                            node.text.decode(errors="replace")[:150],
                            _ast_path(node),
                        )
                        return

    full_text = node.text.decode(errors="replace")
    if not is_test and any(m in full_text for m in RS_HTTP_EXFIL_METHODS):
        for s in _collect_string_values(node):
            if _SCHEME_RE.match(s) and not _is_url_allowed(s):
                result.add(
                    Severity.HIGH,
                    "AST_RS_HTTP_EXFIL",
                    filepath,
                    node.start_point.row + 1,
                    f"HTTP POST/PUT to non-allowlisted URL: {s[:60]}",
                    full_text[:150],
                    _ast_path(node),
                )
                return


def _scan_rust_build_rs(filepath: str, source: bytes, result: ScanResult):
    """Flag suspicious patterns in build.rs — runs at compile time."""
    body = source.decode(errors="replace")
    suspicious = []
    if "Command::new" in body or "process::Command" in body:
        suspicious.append("process execution")
    if "reqwest" in body or "hyper::" in body or "ureq::" in body:
        suspicious.append("HTTP client")
    if "TcpStream" in body or "UdpSocket" in body:
        suspicious.append("raw network socket")
    if "env::var" in body:
        for sensitive in (
            "AWS_",
            "GITHUB_TOKEN",
            "SECRET",
            "PASSWORD",
            "API_KEY",
            "PRIVATE_KEY",
            "TOKEN",
            "CREDENTIALS",
        ):
            if sensitive in body:
                suspicious.append(f"sensitive env var ({sensitive})")
                break
    for url_m in re.finditer(r"https?://([^/:\s\"'`]+)", body):
        if not _is_url_allowed(url_m.group(1)):
            suspicious.append(f"non-allowlisted URL ({url_m.group(1)})")
            break
    if suspicious:
        result.add(
            Severity.HIGH,
            "RS_BUILD_RS_SUSPICIOUS",
            filepath,
            1,
            f"build.rs contains: {', '.join(suspicious)} — "
            "executes at compile time with full privileges",
            body[:200],
        )


# --- package.json lifecycle script scanner ---


def scan_package_json_files(root: Path, result: ScanResult):
    """Scan package.json for malicious lifecycle scripts (#1 npm vector)."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        if "package.json" not in filenames:
            continue
        fpath = Path(dirpath) / "package.json"
        _scan_single_package_json(str(fpath), result)


def _scan_single_package_json(filepath: str, result: ScanResult):
    """Scan one package.json file for suspicious lifecycle scripts."""
    try:
        content = Path(filepath).read_text(errors="replace")
        if len(content) > MAX_FILE_SIZE:
            return
        data = json.loads(content)
    except (OSError, json.JSONDecodeError, PermissionError):
        return
    scripts = data.get("scripts", {})
    if not isinstance(scripts, dict):
        return
    LIFECYCLE_HOOKS = {
        "preinstall",
        "install",
        "postinstall",
        "preuninstall",
        "uninstall",
        "postuninstall",
        "prepublish",
        "preprepare",
        "prepare",
        "postprepare",
    }
    for hook_name, command in scripts.items():
        if hook_name not in LIFECYCLE_HOOKS or not isinstance(command, str):
            continue
        matches = [p.pattern for p in PKG_JSON_SUSPICIOUS if p.search(command)]
        if matches:
            result.add(
                Severity.HIGH,
                "PKG_JSON_LIFECYCLE_SCRIPT",
                filepath,
                0,
                f"Lifecycle hook '{hook_name}' contains suspicious "
                f"patterns: {', '.join(matches[:3])}",
                f"{hook_name}: {command[:200]}",
            )
        elif hook_name in ("preinstall", "postinstall"):
            result.add(
                Severity.LOW,
                "PKG_JSON_LIFECYCLE_HOOK",
                filepath,
                0,
                f"Package has '{hook_name}' lifecycle hook — review",
                f"{hook_name}: {command[:200]}",
            )


# ---------------------------------------------------------------------------
# Layer 2: Main dispatcher
# ---------------------------------------------------------------------------


def scan_file_ast(filepath: str, source: bytes, result: ScanResult):
    if _is_own_source(filepath):
        return
    ext = Path(filepath).suffix.lower()
    parser, _lang = get_parser(ext)
    if parser is None:
        return
    tree = parser.parse(source)
    family = lang_family(filepath)
    is_test = _is_test_file(filepath)

    is_setup_py = filepath.endswith("setup.py")

    def walk(node):
        # Dispatch call expressions to language-specific handlers
        if node.type in ("call", "call_expression"):
            fn_name = _call_name(node)
            if family == "python":
                _scan_python_call(node, fn_name, filepath, is_test, result)
            elif family == "go":
                _scan_go_call(node, fn_name, filepath, is_test, result)
            elif family in ("javascript", "typescript"):
                _scan_js_call(node, fn_name, filepath, is_test, result)
            elif family == "rust":
                _scan_rust_call(node, fn_name, filepath, is_test, result)

        # JS/TS: new Function(...) constructor
        if node.type == "new_expression" and family in ("javascript", "typescript"):
            _scan_js_new(node, filepath, is_test, result)

        # Python setup.py: install command overrides
        if node.type == "class_definition" and is_setup_py:
            _check_setup_py_class(node, filepath, result)

        # Python setup.py: module-level exec/eval
        if (
            is_setup_py
            and node.type == "expression_statement"
            and node.parent
            and node.parent.type == "module"
        ):
            _check_setup_py_module_exec(node, filepath, result)

        # Python: download-and-execute at function scope
        if node.type in FUNC_NODE_TYPES and family == "python":
            _check_download_and_execute(node, filepath, result)

        # Go: suspicious init() functions
        if node.type == "function_declaration" and family == "go":
            _scan_go_init_func(node, filepath, result)

        # Cross-language checks on all nodes
        _check_credential_strings(node, filepath, is_test, result)
        _check_long_base64(node, filepath, result)
        _check_url(node, filepath, is_test, result)

        for child in node.children:
            walk(child)

    walk(tree.root_node)


# ---------------------------------------------------------------------------
# Layer 3: Anomaly Scorer
# ---------------------------------------------------------------------------


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in freq if c > 0)


FUNC_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "function_item",
    "method_declaration",
}


def score_function_anomaly(node, filepath: str, result: ScanResult):
    name_node = node.child_by_field_name("name")
    if not name_node:
        return
    name = name_node.text.decode(errors="replace")
    body = node.text.decode(errors="replace")
    lines = body.splitlines()

    if len(name) <= 2 and len(lines) > 20:
        result.add(
            Severity.MEDIUM,
            "AST_SUSPICIOUS_FUNC",
            filepath,
            node.start_point.row + 1,
            f"Function '{name}' has very short name but {len(lines)} lines",
            body[:150],
            _ast_path(node),
        )

    if lines:
        avg_len = sum(len(ln) for ln in lines) / len(lines)
        if avg_len > 200 and len(lines) > 3:
            result.add(
                Severity.MEDIUM,
                "AST_DENSE_FUNCTION",
                filepath,
                node.start_point.row + 1,
                f"Function '{name}' has very dense lines (avg {avg_len:.0f} "
                "chars/line) — possible obfuscated code",
                body[:150],
                _ast_path(node),
            )

    body_bytes = body.encode(errors="replace")
    if len(body_bytes) > 500:
        ent = entropy(body_bytes)
        if ent > 5.5:
            result.add(
                Severity.LOW,
                "AST_HIGH_ENTROPY_FUNC",
                filepath,
                node.start_point.row + 1,
                f"Function '{name}' has high source entropy " f"({ent:.2f} bits/byte)",
                body[:150],
                _ast_path(node),
            )


def scan_file_anomalies(filepath: str, source: bytes, result: ScanResult):
    if _is_own_source(filepath):
        return
    ext = Path(filepath).suffix.lower()
    parser, _ = get_parser(ext)
    if parser is None:
        return
    tree = parser.parse(source)

    def walk(node):
        if node.type in FUNC_NODE_TYPES:
            score_function_anomaly(node, filepath, result)
        for child in node.children:
            walk(child)

    walk(tree.root_node)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def scan_directory(root: Path, result: ScanResult):
    _init_languages()
    _load_url_allowlist(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fname in filenames:
            fpath = Path(dirpath) / fname

            if fname == "package.json":
                _scan_single_package_json(str(fpath), result)
                continue

            ext = fpath.suffix.lower()
            if ext not in LANG_REGISTRY:
                continue
            try:
                if fpath.stat().st_size > MAX_FILE_SIZE:
                    continue
                source = fpath.read_bytes()
            except (OSError, PermissionError):
                continue
            scan_file_ast(str(fpath), source, result)
            scan_file_anomalies(str(fpath), source, result)

            if fname == "build.rs":
                _scan_rust_build_rs(str(fpath), source, result)
            if fname == "conftest.py":
                _scan_conftest_py(str(fpath), source, result)


def scan_diff(base_ref: str, repo_root: str, result: ScanResult):
    _init_languages()
    _load_url_allowlist(Path(repo_root))
    try:
        proc = subprocess.run(
            ["git", "diff", f"{base_ref}...HEAD", "--unified=0", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=repo_root,
        )
        diff_text = proc.stdout
        if not diff_text:
            proc = subprocess.run(
                ["git", "diff", base_ref, "HEAD", "--unified=0", "--diff-filter=ACMR"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=repo_root,
            )
            diff_text = proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Error getting diff: {e}", file=sys.stderr)
        return []
    if not diff_text:
        return []

    changes = extract_structural_changes(diff_text, repo_root)
    print(f"  Layer 1: Extracted {len(changes)} structural changes", file=sys.stderr)
    for ch in changes:
        label = f"  [{ch.change_type}] {ch.kind}"
        if ch.name:
            label += f" {ch.name}"
        label += f" in {ch.file}:{ch.start_line}-{ch.end_line}"
        print(label, file=sys.stderr)

    file_additions = parse_unified_diff(diff_text)
    for filepath in file_additions:
        ext = Path(filepath).suffix.lower()
        if ext not in LANG_REGISTRY:
            continue
        full_path = Path(repo_root) / filepath
        if not full_path.exists():
            continue
        try:
            source = full_path.read_bytes()
            if len(source) > MAX_FILE_SIZE:
                continue
        except (OSError, PermissionError):
            continue
        added = {ln for ln, _ in file_additions[filepath]}
        fr = ScanResult()
        scan_file_ast(filepath, source, fr)
        scan_file_anomalies(filepath, source, fr)
        for finding in fr.findings:
            if finding.line in added or any(abs(finding.line - a) <= 2 for a in added):
                result.findings.append(finding)

    for fp in file_additions:
        full_fp = Path(repo_root) / fp
        if fp.endswith("package.json") and full_fp.exists():
            _scan_single_package_json(str(full_fp), result)
        if fp.endswith("build.rs") and full_fp.exists():
            try:
                src = full_fp.read_bytes()
                _scan_rust_build_rs(str(full_fp), src, result)
            except (OSError, PermissionError):
                pass
        if fp.endswith("setup.py"):
            result.add(
                Severity.MEDIUM,
                "SETUP_PY_IN_DIFF",
                fp,
                0,
                "setup.py added/modified — review for install hook injection",
            )
        if fp.endswith("conftest.py") and full_fp.exists():
            try:
                src = full_fp.read_bytes()
                _scan_conftest_py(str(full_fp), src, result)
            except (OSError, PermissionError):
                pass
    return changes


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(result: ScanResult, verbose=False, json_output=False):
    if json_output:
        out = [f.to_dict() for f in sorted(result.findings, key=lambda f: f.severity)]
        json.dump({"findings": out, "counts": result.counts()}, sys.stdout, indent=2)
        print()
        return
    if not result.findings:
        print("\n  [CLEAN] No supply chain security issues detected (AST scan).\n")
        return
    result.findings.sort(key=lambda f: f.severity)
    counts = result.counts()
    print("\n" + "=" * 72)
    print("  AST SUPPLY CHAIN SECURITY SCAN RESULTS")
    langs = set()
    for f in result.findings:
        fam = lang_family(f.file)
        if fam != "unknown":
            langs.add(fam)
    if langs:
        print(f"  Languages scanned: {', '.join(sorted(langs))}")
    print("=" * 72)
    print(f"\n  Total findings: {len(result.findings)}")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if sev in counts:
            print(f"    {sev}: {counts[sev]}")
    print()
    prev_sev = None
    for f in result.findings:
        if not verbose and f.severity == Severity.LOW:
            continue
        if f.severity.name != prev_sev:
            print(f"--- {f.severity.name} ---")
            prev_sev = f.severity.name
        print(f)
        print()
    if "CRITICAL" in counts:
        print("  *** CRITICAL findings require IMMEDIATE investigation ***")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="AST-based supply chain security scanner (tree-sitter, multi-language)"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("scan", help="Scan a directory")
    sp.add_argument("path", nargs="?", default=".")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.add_argument(
        "--fail-on", default="HIGH", choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )

    dp = sub.add_parser("diff", help="Scan a PR diff")
    dp.add_argument("base_ref", nargs="?", default="main")
    dp.add_argument("--repo-root", default=".")
    dp.add_argument("--verbose", "-v", action="store_true")
    dp.add_argument("--json", action="store_true")
    dp.add_argument(
        "--fail-on", default="HIGH", choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )

    args = ap.parse_args()
    result = ScanResult()

    try:
        if args.command == "scan":
            root = Path(args.path).resolve()
            if not root.is_dir():
                print(f"Error: {root} is not a directory", file=sys.stderr)
                sys.exit(2)
            print(f"AST scanning {root} ...", file=sys.stderr)
            scan_directory(root, result)
        elif args.command == "diff":
            print(f"AST scanning diff: {args.base_ref}...HEAD", file=sys.stderr)
            scan_diff(args.base_ref, args.repo_root, result)
    except Exception as exc:
        print(f"Scanner error: {exc}", file=sys.stderr)
        if args.json:
            json.dump(
                {"findings": [], "counts": {}, "error": str(exc)}, sys.stdout, indent=2
            )
            print()
        sys.exit(2)

    print_report(result, verbose=args.verbose, json_output=args.json)
    threshold = Severity[args.fail_on]
    sys.exit(1 if result.worst is not None and result.worst <= threshold else 0)


if __name__ == "__main__":
    main()
