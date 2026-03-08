#!/usr/bin/env python3
"""Structural checks for changed files with optional AST-backed checks."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_go
import tree_sitter_python
import tree_sitter_rust
import yaml
from tree_sitter import Language, Parser

REPO_ROOT = Path(__file__).resolve().parents[3]
RULES_PATH = REPO_ROOT / "tools" / "agent" / "structure-rules.yaml"


FUNCTION_NODE_TYPES = {
    "go": {"function_declaration", "method_declaration", "func_literal"},
    "python": {"function_definition", "async_function_definition"},
    "rust": {"function_item"},
}

INTERFACE_NODE_TYPES = {
    "go": {"interface_type"},
    "rust": {"trait_item"},
}

INTERFACE_METHOD_NODE_TYPES = {
    "go": {"method_elem", "method_spec"},
    "rust": {"function_item", "function_signature_item"},
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
}


@dataclass
class Finding:
    level: str
    file: str
    message: str


@dataclass(frozen=True)
class FunctionMetrics:
    lines: int
    nesting: int


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


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def get_legacy_hotspot_rule(path: str, rules: dict) -> dict | None:
    for rule in rules.get("legacy_hotspots", []):
        patterns = rule.get("paths", [])
        if matches_any(path, patterns):
            return rule
    return None


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


def load_baseline_line_count(path: str, base_ref: str | None) -> int | None:
    baseline_source = load_baseline_source(path, base_ref)
    if baseline_source is None:
        return None
    return baseline_source.count("\n") + 1


def evaluate_dependency_rules(path: str, text: str, rules: dict) -> list[Finding]:
    findings: list[Finding] = []
    for rule in rules["dependency_rules"]:
        if not any(fnmatch.fnmatch(path, pattern) for pattern in rule["applies_to"]):
            continue
        for literal in rule["forbidden_literals"]:
            if literal in text:
                findings.append(
                    Finding(
                        level="ERROR",
                        file=path,
                        message=f"{rule['name']}: forbidden literal '{literal}'",
                    )
                )
    return findings


def evaluate_file_line_count(
    path: str, line_count: int, rules: dict, base_ref: str | None
) -> list[Finding]:
    findings: list[Finding] = []
    legacy_hotspot = get_legacy_hotspot_rule(path, rules)
    baseline_line_count = (
        load_baseline_line_count(path, base_ref) if legacy_hotspot else None
    )
    warn_limit = rules["limits"]["file_lines"]["warn"]
    error_limit = rules["limits"]["file_lines"]["error"]

    if line_count > error_limit:
        if (
            legacy_hotspot
            and baseline_line_count is not None
            and line_count <= baseline_line_count
        ):
            findings.append(
                Finding(
                    "WARN",
                    path,
                    f"legacy hotspot still has {line_count} lines (limit {error_limit}) "
                    f"but did not grow from baseline {baseline_line_count}",
                )
            )
        else:
            findings.append(
                Finding(
                    "ERROR", path, f"file has {line_count} lines (limit {error_limit})"
                )
            )
    elif line_count > warn_limit:
        findings.append(
            Finding("WARN", path, f"file has {line_count} lines (warn {warn_limit})")
        )

    return findings


def function_identifier(node, source_bytes: bytes) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
    contextual_identifier = anonymous_function_identifier(node, source_bytes)
    if contextual_identifier is not None:
        return contextual_identifier
    line_start = source_bytes.rfind(b"\n", 0, node.start_byte)
    prefix = source_bytes[line_start + 1 : node.start_byte].decode(
        "utf-8", errors="ignore"
    )
    normalized_prefix = " ".join(prefix.split())
    return f"{node.type}:{normalized_prefix[-80:]}"


def anonymous_function_identifier(node, source_bytes: bytes) -> str | None:
    contexts: list[str] = []
    current = node.parent
    while current is not None:
        if current.type == "call_expression" and (
            context := call_expression_identifier(current, node, source_bytes)
        ):
            contexts.append(context)
        current = current.parent
    if not contexts:
        return None
    return f"{node.type}:{' > '.join(reversed(contexts))}"


def call_expression_identifier(call_node, func_node, source_bytes: bytes) -> str | None:
    function_node = call_node.child_by_field_name("function")
    if function_node is None and call_node.named_children:
        function_node = call_node.named_children[0]
    if function_node is None:
        return None
    function_name = " ".join(
        source_bytes[function_node.start_byte : function_node.end_byte]
        .decode("utf-8", errors="ignore")
        .split()
    )
    if not function_name:
        return None

    label = ""
    for child in call_node.named_children:
        if child.type != "argument_list":
            continue
        for argument in child.named_children:
            if argument.start_byte >= func_node.start_byte:
                break
            label = format_argument_label(argument, source_bytes)
            if label:
                break
        break

    if label:
        return f"{function_name}({label})"
    return function_name


def format_argument_label(argument_node, source_bytes: bytes) -> str:
    label = " ".join(
        source_bytes[argument_node.start_byte : argument_node.end_byte]
        .decode("utf-8", errors="ignore")
        .split()
    )
    label = label.strip('"`')
    return label[:120]


def collect_function_metrics(
    tree, language_name: str, source_bytes: bytes
) -> dict[str, FunctionMetrics]:
    metrics: dict[str, FunctionMetrics] = {}
    occurrence_counts: dict[str, int] = defaultdict(int)
    for node in walk(tree.root_node):
        if node.type not in FUNCTION_NODE_TYPES[language_name]:
            continue
        base_identifier = function_identifier(node, source_bytes)
        occurrence_counts[base_identifier] += 1
        identifier = f"{base_identifier}#{occurrence_counts[base_identifier]}"
        metrics[identifier] = FunctionMetrics(
            lines=node.end_point.row - node.start_point.row + 1,
            nesting=max_nesting_depth(node, language_name),
        )
    return metrics


def load_baseline_function_metrics(
    path: str, language_name: str, parser: Parser, base_ref: str | None
) -> dict[str, FunctionMetrics]:
    baseline_source = load_baseline_source(path, base_ref)
    if baseline_source is None:
        return {}
    baseline_bytes = baseline_source.encode("utf-8")
    baseline_tree = parser.parse(baseline_bytes)
    return collect_function_metrics(baseline_tree, language_name, baseline_bytes)


def ratchet_or_error(
    path: str,
    message: str,
    identifier: str,
    current_value: int,
    baseline_metrics: dict[str, FunctionMetrics],
    baseline_value_getter,
) -> Finding:
    baseline = baseline_metrics.get(identifier)
    if baseline is not None and current_value <= baseline_value_getter(baseline):
        return Finding(
            "WARN",
            path,
            f"legacy hotspot {message} but did not exceed baseline {baseline_value_getter(baseline)}",
        )
    return Finding("ERROR", path, message)


def relax_function_checks(legacy_hotspot: dict | None) -> bool:
    return bool(legacy_hotspot and legacy_hotspot.get("function_checks") == "relaxed")


def function_metric_finding(
    path: str,
    message: str,
    identifier: str,
    current_value: int,
    baseline_metrics: dict[str, FunctionMetrics],
    baseline_value_getter,
    legacy_hotspot: dict | None,
) -> Finding:
    if relax_function_checks(legacy_hotspot):
        return Finding(
            "WARN",
            path,
            f"legacy hotspot {message}; per-function ratchet is relaxed for this file",
        )
    return ratchet_or_error(
        path,
        message,
        identifier,
        current_value,
        baseline_metrics,
        baseline_value_getter,
    )


def evaluate_ast_rules(
    path: str,
    language_name: str,
    source_bytes: bytes,
    rules: dict,
    parser: Parser,
    base_ref: str | None,
) -> list[Finding]:
    findings: list[Finding] = []
    tree = parser.parse(source_bytes)
    legacy_hotspot = get_legacy_hotspot_rule(path, rules)
    baseline_metrics = (
        load_baseline_function_metrics(path, language_name, parser, base_ref)
        if legacy_hotspot
        else {}
    )
    occurrence_counts: dict[str, int] = defaultdict(int)

    for node in walk(tree.root_node):
        if node.type in FUNCTION_NODE_TYPES[language_name]:
            base_identifier = function_identifier(node, source_bytes)
            occurrence_counts[base_identifier] += 1
            identifier = f"{base_identifier}#{occurrence_counts[base_identifier]}"
            function_lines = node.end_point.row - node.start_point.row + 1
            if function_lines > rules["limits"]["function_lines"]["error"]:
                message = (
                    f"function starting at line {node.start_point.row + 1} has {function_lines} lines "
                    f"(limit {rules['limits']['function_lines']['error']})"
                )
                findings.append(
                    function_metric_finding(
                        path,
                        message,
                        identifier,
                        function_lines,
                        baseline_metrics,
                        lambda metrics: metrics.lines,
                        legacy_hotspot,
                    )
                )
            nesting = max_nesting_depth(node, language_name)
            if nesting > rules["limits"]["nesting"]["error"]:
                message = (
                    f"function starting at line {node.start_point.row + 1} nests {nesting} levels "
                    f"(limit {rules['limits']['nesting']['error']})"
                )
                findings.append(
                    function_metric_finding(
                        path,
                        message,
                        identifier,
                        nesting,
                        baseline_metrics,
                        lambda metrics: metrics.nesting,
                        legacy_hotspot,
                    )
                )

        if node.type in INTERFACE_NODE_TYPES.get(language_name, set()):
            method_count = count_interface_methods(node, language_name)
            if method_count > rules["limits"]["interface_methods"]["error"]:
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        f"interface/trait starting at line {node.start_point.row + 1} has {method_count} methods "
                        f"(limit {rules['limits']['interface_methods']['error']})",
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
    findings = evaluate_dependency_rules(path, source_text, rules)
    findings.extend(
        evaluate_file_line_count(path, source_text.count("\n") + 1, rules, base_ref)
    )

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
            base_ref,
        )
    )
    return findings


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run structure checks on changed files"
    )
    parser.add_argument("files", nargs="*")
    parser.add_argument("--base-ref", default=None)
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
