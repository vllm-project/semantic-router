#!/usr/bin/env python3
"""Structural checks for changed files with optional AST-backed checks."""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_go
import tree_sitter_python
import tree_sitter_rust
import tree_sitter_typescript
import yaml
from tree_sitter import Language, Parser

REPO_ROOT = Path(__file__).resolve().parents[3]
RULES_PATH = REPO_ROOT / "tools" / "agent" / "structure-rules.yaml"


FUNCTION_NODE_TYPES = {
    "go": {"function_declaration", "method_declaration", "func_literal"},
    "python": {"function_definition", "async_function_definition"},
    "rust": {"function_item"},
    "typescript": {
        "arrow_function",
        "function_declaration",
        "function_expression",
        "generator_function",
        "generator_function_declaration",
        "method_definition",
    },
}

INTERFACE_NODE_TYPES = {
    "go": {"interface_type"},
    "rust": {"trait_item"},
    "typescript": {"interface_declaration"},
}

INTERFACE_METHOD_NODE_TYPES = {
    "go": {"method_elem", "method_spec"},
    "rust": {"function_item", "function_signature_item"},
    "typescript": {"method_signature"},
}

CONTROL_NODE_TYPES = {
    "go": {
        "if_statement",
        "for_statement",
        "expression_switch_statement",
        "type_switch_statement",
        "select_statement",
    },
    "python": {
        "if_statement",
        "for_statement",
        "while_statement",
        "with_statement",
        "try_statement",
        "match_statement",
    },
    "rust": {
        "if_expression",
        "for_expression",
        "while_expression",
        "loop_expression",
        "match_expression",
    },
    "typescript": {
        "do_statement",
        "for_in_statement",
        "for_statement",
        "if_statement",
        "switch_statement",
        "try_statement",
        "while_statement",
    },
}


@dataclass
class Finding:
    level: str
    file: str
    message: str


def load_rules() -> dict:
    with RULES_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_language(module) -> Language:
    return Language(module.language())


def build_parser(parser_name: str) -> Parser:
    language_map = {
        "go": build_language(tree_sitter_go),
        "python": build_language(tree_sitter_python),
        "rust": build_language(tree_sitter_rust),
        "typescript": Language(tree_sitter_typescript.language_tsx()),
    }
    parser = Parser()
    language = language_map[parser_name]
    try:
        parser.language = language
    except AttributeError:
        parser.set_language(language)
    return parser


def parser_name_for_language(language_name: str, rules: dict) -> str | None:
    language_config = rules["languages"][language_name]
    parser_name = language_config.get("parser", language_name)
    if parser_name == "none":
        return None
    return parser_name


def walk(node):
    yield node
    for child in node.named_children:
        yield from walk(child)


def max_nesting_depth(node, language_name: str, current: int = 0) -> int:
    if node.type in CONTROL_NODE_TYPES[language_name]:
        current += 1
    depth = current
    for child in node.named_children:
        depth = max(depth, max_nesting_depth(child, language_name, current))
    return depth


def count_interface_methods(node, language_name: str) -> int:
    return sum(
        1
        for child in walk(node)
        if child.type in INTERFACE_METHOD_NODE_TYPES.get(language_name, set())
    )


def detect_language(path: str, rules: dict) -> str | None:
    for language_name, config in rules["languages"].items():
        if any(fnmatch.fnmatch(path, pattern) for pattern in config["globs"]):
            return language_name
    return None


def should_ignore(path: str, rules: dict) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in rules["ignore_globs"])


def load_baseline_source(path: str, base_ref: str | None) -> str | None:
    ref = base_ref or "HEAD"
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def evaluate_dependency_rules(
    path: str, text: str, rules: dict, base_ref: str | None = None
) -> list[Finding]:
    findings: list[Finding] = []
    baseline_source: str | None = None
    for rule in rules["dependency_rules"]:
        if not any(fnmatch.fnmatch(path, pattern) for pattern in rule["applies_to"]):
            continue
        for literal in rule["forbidden_literals"]:
            current_count = text.count(literal)
            if current_count == 0:
                continue
            if rule.get("policy", "error") == "no-new":
                if baseline_source is None:
                    baseline_source = load_baseline_source(path, base_ref) or ""
                baseline_count = baseline_source.count(literal)
                if current_count <= baseline_count:
                    findings.append(
                        Finding(
                            level="WARN",
                            file=path,
                            message=(
                                f"{rule['name']}: pre-existing forbidden dependency "
                                f"'{literal}' did not grow from baseline {baseline_count}"
                            ),
                        )
                    )
                    continue
            findings.append(
                Finding(
                    level="ERROR",
                    file=path,
                    message=f"{rule['name']}: forbidden dependency '{literal}'",
                )
            )
    return findings


def evaluate_root_placement(path: str, rules: dict) -> list[Finding]:
    """Reject new root files that do not have a repository-wide contract."""
    absolute_path = REPO_ROOT / path
    if "/" in path or not absolute_path.is_file():
        return []
    if path in rules["root_files"]["allowed"]:
        return []
    return [
        Finding(
            level="ERROR",
            file=path,
            message="root file is not allowlisted; place it under its owning subtree",
        )
    ]


def evaluate_file_line_count(path: str, line_count: int, rules: dict) -> list[Finding]:
    warn_limit = rules["limits"]["file_lines"]["warn"]
    if line_count <= warn_limit:
        return []
    return [
        Finding(
            "WARN",
            path,
            f"file has {line_count} lines (advisory {warn_limit}); "
            "review cohesion before splitting",
        )
    ]


def evaluate_ast_rules(
    path: str,
    language_name: str,
    source_bytes: bytes,
    rules: dict,
    parser: Parser,
) -> list[Finding]:
    findings: list[Finding] = []
    tree = parser.parse(source_bytes)

    for node in walk(tree.root_node):
        if node.type in FUNCTION_NODE_TYPES[language_name]:
            function_lines = node.end_point.row - node.start_point.row + 1
            function_limit = rules["limits"]["function_lines"]["warn"]
            if function_lines > function_limit:
                findings.append(
                    Finding(
                        "WARN",
                        path,
                        f"function starting at line {node.start_point.row + 1} has "
                        f"{function_lines} lines (advisory {function_limit})",
                    )
                )
            nesting = max_nesting_depth(node, language_name)
            nesting_limit = rules["limits"]["nesting"]["warn"]
            if nesting > nesting_limit:
                findings.append(
                    Finding(
                        "WARN",
                        path,
                        f"function starting at line {node.start_point.row + 1} nests "
                        f"{nesting} levels (advisory {nesting_limit})",
                    )
                )

        if node.type in INTERFACE_NODE_TYPES.get(language_name, set()):
            method_count = count_interface_methods(node, language_name)
            interface_limit = rules["limits"]["interface_methods"]["warn"]
            if method_count > interface_limit:
                findings.append(
                    Finding(
                        "WARN",
                        path,
                        f"interface/trait starting at line {node.start_point.row + 1} "
                        f"has {method_count} methods (advisory {interface_limit})",
                    )
                )

    return findings


def evaluate_file(
    path: str, rules: dict, parsers: dict[str, Parser], base_ref: str | None
) -> list[Finding]:
    language_name = detect_language(path, rules)
    if language_name is None or should_ignore(path, rules):
        return []

    absolute_path = REPO_ROOT / path
    if not absolute_path.exists():
        return []

    source_bytes = absolute_path.read_bytes()
    source_text = source_bytes.decode("utf-8", errors="ignore")
    findings = evaluate_dependency_rules(path, source_text, rules, base_ref)
    findings.extend(evaluate_file_line_count(path, source_text.count("\n") + 1, rules))

    parser_name = parser_name_for_language(language_name, rules)
    if parser_name is None:
        return findings

    findings.extend(
        evaluate_ast_rules(
            path,
            language_name,
            source_bytes,
            rules,
            parsers[language_name],
        )
    )
    return findings


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run structure checks on changed files"
    )
    parser.add_argument("files", nargs="*")
    parser.add_argument("--base-ref", default=os.getenv("BASE_REF"))
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    rules = load_rules()
    parsers = {
        name: build_parser(parser_name)
        for name in rules["languages"]
        if (parser_name := parser_name_for_language(name, rules)) is not None
    }
    findings_by_file: dict[str, list[Finding]] = defaultdict(list)

    for raw_path in args.files:
        path = raw_path.strip()
        while path.startswith("./"):
            path = path[2:]
        if not path:
            continue
        for finding in evaluate_root_placement(path, rules):
            findings_by_file[finding.file].append(finding)
        for finding in evaluate_file(path, rules, parsers, args.base_ref):
            findings_by_file[finding.file].append(finding)

    exit_code = 0
    for file_path in sorted(findings_by_file):
        for finding in findings_by_file[file_path]:
            if finding.level == "ERROR":
                exit_code = 1
            print(f"[{finding.level}] {finding.file} :: {finding.message}")

    if not findings_by_file:
        print("Structure check passed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
