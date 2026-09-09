#!/usr/bin/env python3
"""Check real imports, module cycles, and repository root ownership."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import os
import posixpath
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import tree_sitter_go
import tree_sitter_rust
import tree_sitter_typescript
import yaml
from tree_sitter import Language, Parser

REPO_ROOT = Path(__file__).resolve().parents[3]
RULES_PATH = REPO_ROOT / "tools" / "agent" / "structure-rules.yaml"


@dataclass(frozen=True)
class Finding:
    level: str
    scope: str
    message: str


def load_rules() -> dict:
    with RULES_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def focus_patterns(scope: dict) -> list[str]:
    return scope.get("focus", scope["include"])


def scope_is_touched(scope: dict, changed_files: set[str]) -> bool:
    return any(matches_any(path, focus_patterns(scope)) for path in changed_files)


def load_current_sources(scope: dict) -> dict[str, str]:
    sources: dict[str, str] = {}
    root = REPO_ROOT / scope["root"]
    if not root.exists():
        return sources
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if matches_any(relative, scope["include"]):
            sources[relative] = path.read_text(encoding="utf-8", errors="ignore")
    return sources


def load_revision_sources(scope: dict, base_ref: str | None) -> dict[str, str]:
    ref = base_ref or "HEAD"
    result = subprocess.run(
        ["git", "archive", "--format=tar", ref, "--", scope["root"]],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return {}
    sources: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not matches_any(member.name, scope["include"]):
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                sources[member.name] = extracted.read().decode("utf-8", errors="ignore")
    return sources


def python_module_name(path: str, module_root: str) -> str:
    relative = PurePosixPath(path).relative_to(PurePosixPath(module_root))
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_python_module(module_name: str, module_index: dict[str, str]) -> str | None:
    candidate = module_name
    while candidate:
        if candidate in module_index:
            return module_index[candidate]
        candidate = candidate.rpartition(".")[0]
    return None


def python_import_base(node: ast.ImportFrom, package_parts: list[str]) -> str:
    if not node.level:
        return node.module or ""
    retained = max(0, len(package_parts) - node.level + 1)
    base_parts = package_parts[:retained]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def python_dependencies(
    path: str, source: str, sources: dict[str, str], module_root: str
) -> set[str]:
    module_index = {
        python_module_name(candidate, module_root): candidate for candidate in sources
    }
    current_module = python_module_name(path, module_root)
    package_parts = current_module.split(".")
    if not path.endswith("/__init__.py"):
        package_parts.pop()
    dependencies: set[str] = set()

    def add_module(module_name: str) -> None:
        target = resolve_python_module(module_name, module_index)
        if target is not None and target != path:
            dependencies.add(target)

    for node in ast.walk(ast.parse(source, filename=path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base_module = python_import_base(node, package_parts)
            for alias in node.names:
                if alias.name != "*":
                    add_module(
                        ".".join(part for part in (base_module, alias.name) if part)
                    )
            if base_module:
                add_module(base_module)
    return dependencies


def walk_tree(node):
    yield node
    for child in node.named_children:
        yield from walk_tree(child)


def typescript_parser() -> Parser:
    parser = Parser()
    language = Language(tree_sitter_typescript.language_tsx())
    try:
        parser.language = language
    except AttributeError:
        parser.set_language(language)
    return parser


def decode_string_node(node, source_bytes: bytes) -> str:
    value = source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="ignore"
    )
    return value[1:-1] if value else ""


def typescript_specifiers(source: str, parser: Parser) -> set[str]:
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    specifiers: set[str] = set()
    for node in walk_tree(tree.root_node):
        if node.type in {"import_statement", "export_statement"}:
            source_node = node.child_by_field_name("source")
            if source_node is not None and source_node.type == "string":
                specifiers.add(decode_string_node(source_node, source_bytes))
        elif node.type == "call_expression":
            function_node = node.child_by_field_name("function")
            if function_node is None or function_node.type != "import":
                continue
            arguments = node.child_by_field_name("arguments")
            if arguments is None:
                continue
            string_node = next(
                (child for child in walk_tree(arguments) if child.type == "string"),
                None,
            )
            if string_node is not None:
                specifiers.add(decode_string_node(string_node, source_bytes))
    return specifiers


def resolve_typescript_specifier(
    path: str, specifier: str, sources: dict[str, str], aliases: dict[str, str]
) -> str | None:
    base: str | None = None
    if specifier.startswith("."):
        base = posixpath.normpath(posixpath.join(posixpath.dirname(path), specifier))
    else:
        for prefix, replacement in aliases.items():
            if specifier.startswith(prefix):
                base = posixpath.normpath(replacement + specifier[len(prefix) :])
                break
    if base is None:
        return None
    suffix = PurePosixPath(base).suffix
    candidates = [base]
    if suffix in {".js", ".jsx"}:
        stem = base[: -len(suffix)]
        candidates.extend((f"{stem}.ts", f"{stem}.tsx"))
    elif not suffix:
        candidates.extend(
            (f"{base}.ts", f"{base}.tsx", f"{base}/index.ts", f"{base}/index.tsx")
        )
    return next((candidate for candidate in candidates if candidate in sources), None)


def build_graph(scope: dict, sources: dict[str, str]) -> dict[str, set[str]]:
    language = scope["language"]
    graph = {path: set() for path in sources}
    parser = typescript_parser() if language == "typescript" else None
    for path, source in sources.items():
        if language == "python":
            graph[path] = python_dependencies(
                path, source, sources, scope["module_root"]
            )
        elif language == "typescript" and parser is not None:
            for specifier in typescript_specifiers(source, parser):
                target = resolve_typescript_specifier(
                    path, specifier, sources, scope.get("aliases", {})
                )
                if target is not None and target != path:
                    graph[path].add(target)
        else:
            raise ValueError(f"unsupported architecture graph language: {language}")
    return graph


def cyclic_components(graph: dict[str, set[str]]) -> tuple[frozenset[str], ...]:
    next_index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[frozenset[str]] = []

    def visit(node: str) -> None:
        nonlocal next_index
        indices[node] = next_index
        lowlinks[node] = next_index
        next_index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph[node]:
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] != indices[node]:
            return
        component: set[str] = set()
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.add(member)
            if member == node:
                break
        if len(component) > 1:
            components.append(frozenset(component))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return tuple(sorted(components, key=sorted))


def cycle_edges(graph: dict[str, set[str]]) -> set[tuple[str, str]]:
    """Keep existing cycle edges ratcheted even when a cycle splits or shrinks."""
    return {
        (source, target)
        for component in cyclic_components(graph)
        for source in component
        for target in graph[source].intersection(component)
    }


def evaluate_root_placement(path: str, rules: dict) -> list[Finding]:
    if "/" in path or not (REPO_ROOT / path).is_file():
        return []
    if path in rules["root_files"]["allowed"]:
        return []
    return [
        Finding(
            "ERROR",
            path,
            "root file is not allowlisted; place it under its owning subtree",
        )
    ]


def load_baseline_source(path: str, base_ref: str | None) -> str:
    result = subprocess.run(
        ["git", "show", f"{base_ref or 'HEAD'}:{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def rust_use_paths(node, source: bytes, prefix: str = "") -> set[str]:
    """Expand grouped imports without interpreting comments or arbitrary strings."""
    if node.type == "use_as_clause":
        return rust_use_paths(node.child_by_field_name("path"), source, prefix)
    if node.type == "scoped_use_list":
        parent = node.child_by_field_name("path")
        stem = source[parent.start_byte : parent.end_byte].decode() if parent else ""
        return rust_use_paths(
            node.child_by_field_name("list"), source, prefix + stem + "::"
        )
    if node.type == "use_list":
        return set().union(
            *(rust_use_paths(child, source, prefix) for child in node.named_children)
        )
    value = source[node.start_byte : node.end_byte].decode()
    return {prefix + value}


def import_specifiers(path: str, source: str) -> set[str]:
    suffix = PurePosixPath(path).suffix
    imports: set[str] = set()
    if suffix == ".py":
        for node in ast.walk(ast.parse(source, filename=path)):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                # Relative imports stay inside the package; the module graph
                # checks their internal boundaries using resolved file paths.
                if node.level:
                    continue
                base = node.module or ""
                imports.add(base)
                imports.update(
                    ".".join(filter(None, (base, alias.name))) for alias in node.names
                )
        return imports
    modules = {".go": tree_sitter_go, ".rs": tree_sitter_rust}
    if suffix not in modules:
        return imports
    parser = Parser(Language(modules[suffix].language()))
    source_bytes = source.encode()
    tree = parser.parse(source_bytes)
    if tree.root_node.has_error:
        raise ValueError(f"cannot parse imports in {path}")
    for node in walk_tree(tree.root_node):
        if node.type == "import_spec":
            value = node.child_by_field_name("path")
            literal = source_bytes[value.start_byte : value.end_byte].decode()
            imports.add(
                literal[1:-1] if literal.startswith("`") else ast.literal_eval(literal)
            )
        elif node.type == "use_declaration":
            imports.update(
                rust_use_paths(node.child_by_field_name("argument"), source_bytes)
            )
        elif node.type == "extern_crate_declaration":
            value = node.child_by_field_name("name")
            imports.add(source_bytes[value.start_byte : value.end_byte].decode())
    return {value.removeprefix("::").removeprefix("crate::") for value in imports}


def evaluate_dependency_rules(
    path: str, source: str, rules: dict, base_ref: str | None = None
) -> list[Finding]:
    applicable = [
        rule
        for rule in rules.get("dependency_rules", [])
        if matches_any(path, rule["applies_to"])
    ]
    if not applicable:
        return []
    current = import_specifiers(path, source)
    baseline = None
    findings: list[Finding] = []
    for rule in applicable:
        for imported in sorted(current):
            if not any(
                imported == forbidden
                or any(
                    imported.startswith(forbidden + separator)
                    for separator in ("/", ".", "::")
                )
                for forbidden in rule["forbidden_imports"]
            ):
                continue
            if baseline is None and rule.get("policy") == "no-new":
                baseline = import_specifiers(path, load_baseline_source(path, base_ref))
            pre_existing = baseline is not None and imported in baseline
            level = (
                "WARN" if pre_existing and rule.get("policy") == "no-new" else "ERROR"
            )
            disposition = "pre-existing" if pre_existing else "new"
            findings.append(
                Finding(
                    level,
                    path,
                    f"{rule['name']}: {disposition} forbidden import '{imported}'",
                )
            )
    return findings


def evaluate_forbidden_edges(
    scope: dict,
    current_graph: dict[str, set[str]],
    baseline_graph: dict[str, set[str]],
    changed_files: set[str],
) -> list[Finding]:
    findings: list[Finding] = []
    for boundary in scope.get("forbidden_edges", []):
        for source, targets in current_graph.items():
            if not matches_any(source, boundary["from"]):
                continue
            for target in targets:
                if not matches_any(target, boundary["to"]):
                    continue
                if source not in changed_files and target not in changed_files:
                    continue
                pre_existing = target in baseline_graph.get(source, set())
                level = (
                    "WARN"
                    if pre_existing and boundary.get("policy") == "no-new"
                    else "ERROR"
                )
                disposition = "pre-existing" if pre_existing else "new"
                findings.append(
                    Finding(
                        level,
                        scope["name"],
                        f"{boundary['name']}: {disposition} edge {source} -> {target}",
                    )
                )
    return findings


def evaluate_dependency_graph(
    scope: dict,
    current_sources: dict[str, str],
    baseline_sources: dict[str, str],
    changed_files: set[str],
) -> list[Finding]:
    if not scope_is_touched(scope, changed_files):
        return []
    current_graph = build_graph(scope, current_sources)
    baseline_graph = build_graph(scope, baseline_sources)
    baseline_cycle_edges = cycle_edges(baseline_graph)
    findings: list[Finding] = []
    findings.extend(
        evaluate_forbidden_edges(scope, current_graph, baseline_graph, changed_files)
    )
    for component in cyclic_components(current_graph):
        if component.isdisjoint(changed_files):
            continue
        current_edges = {
            (source, target)
            for source in component
            for target in current_graph[source].intersection(component)
        }
        pre_existing = current_edges.issubset(baseline_cycle_edges)
        level = (
            "WARN"
            if pre_existing and scope.get("cycle_policy") == "no-new"
            else "ERROR"
        )
        disposition = "pre-existing" if pre_existing else "new"
        members = " -> ".join(sorted(component))
        findings.append(
            Finding(level, scope["name"], f"{disposition} dependency cycle: {members}")
        )
    return findings


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check import boundaries, dependency cycles, and root ownership"
    )
    parser.add_argument("files", nargs="*")
    parser.add_argument("--base-ref", default=os.getenv("BASE_REF"))
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    changed_files = {
        normalize_path(path) for path in args.files if normalize_path(path)
    }
    rules = load_rules()
    architecture = rules.get("architecture", {})
    findings: list[Finding] = []
    for path in sorted(changed_files):
        findings.extend(evaluate_root_placement(path, rules))
        absolute = REPO_ROOT / path
        if (
            absolute.is_file()
            and not matches_any(path, rules.get("ignore_globs", []))
            and any(
                matches_any(path, rule["applies_to"])
                for rule in rules.get("dependency_rules", [])
            )
        ):
            findings.extend(
                evaluate_dependency_rules(
                    path, absolute.read_text(encoding="utf-8"), rules, args.base_ref
                )
            )
    for scope in architecture.get("dependency_graphs", []):
        if not scope_is_touched(scope, changed_files):
            continue
        findings.extend(
            evaluate_dependency_graph(
                scope,
                load_current_sources(scope),
                load_revision_sources(scope, args.base_ref),
                changed_files,
            )
        )
    if not findings:
        print("Architecture check passed (no configured scope changed).")
        return 0
    exit_code = 0
    for finding in findings:
        if finding.level == "ERROR":
            exit_code = 1
        print(f"[{finding.level}] {finding.scope} :: {finding.message}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
