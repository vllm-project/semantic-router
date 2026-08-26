#!/usr/bin/env python3
"""Normalize GolangCI complexity diagnostics to stable Go declarations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_go
import yaml
from tree_sitter import Language, Node, Parser

COMPLEXITY_LINTERS = frozenset(
    {"cyclop", "funlen", "gocognit", "interfacebloat", "nestif"}
)

_CYCLOP_PATTERN = re.compile(
    r"calculated cyclomatic complexity for function .+ is (?P<observed>\d+), "
    r"max is (?P<limit>\d+)$"
)
_FUNLEN_PATTERN = re.compile(
    r"Function '.+' (?:is too long|has too many statements) "
    r"\((?P<observed>\d+) > (?P<limit>\d+)\)$"
)
_GOCOGNIT_PATTERN = re.compile(
    r"cognitive complexity (?P<observed>\d+) of func `.+` is high "
    r"\(> (?P<limit>\d+)\)$"
)
_INTERFACEBLOAT_PATTERN = re.compile(
    r"the interface has more than (?P<limit>\d+) methods: "
    r"(?P<observed>\d+)$"
)
_NESTIF_PATTERN = re.compile(
    r"`.+` has complex nested blocks \(complexity: (?P<observed>\d+)\)$"
)


class ComplexityIdentityError(ValueError):
    """Raised when a diagnostic cannot be normalized without guessing."""


@dataclass(frozen=True, order=True)
class ComplexityIdentity:
    path: str
    declaration: str
    linter: str
    site: str = ""


@dataclass(frozen=True)
class ComplexityFinding:
    identity: ComplexityIdentity
    observed: int
    limit: int
    line: int
    column: int
    message: str


@dataclass(frozen=True)
class ComplexityLimits:
    cyclop: int
    funlen_lines: int
    funlen_statements: int
    gocognit: int
    interfacebloat: int
    nestif: int


def complexity_limits_from_config(config: object, source: str) -> ComplexityLimits:
    try:
        settings = config["linters"]["settings"]
        nestif_trigger = int(settings["nestif"]["min-complexity"])
        return ComplexityLimits(
            cyclop=int(settings["cyclop"]["max-complexity"]),
            funlen_lines=int(settings["funlen"]["lines"]),
            funlen_statements=int(settings["funlen"]["statements"]),
            gocognit=int(settings["gocognit"]["min-complexity"]),
            interfacebloat=int(settings["interfacebloat"]["max"]),
            nestif=nestif_trigger - 1,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ComplexityIdentityError(
            f"invalid Go complexity linter settings in {source}"
        ) from exc


def load_complexity_limits(config_path: Path) -> ComplexityLimits:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return complexity_limits_from_config(config, str(config_path))


def _build_go_parser() -> Parser:
    parser = Parser()
    language = Language(tree_sitter_go.language())
    try:
        parser.language = language
    except AttributeError:
        parser.set_language(language)
    return parser


def _node_text(source: bytes, node: Node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="strict")


def _walk(node: Node):
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _normalize_type(raw: str) -> str:
    return re.sub(r"\s+", "", raw)


def _receiver_type(source: bytes, receiver: Node | None) -> str:
    if receiver is None:
        raise ComplexityIdentityError("method declaration has no receiver")
    for node in _walk(receiver):
        if node.type != "parameter_declaration":
            continue
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            return _normalize_type(_node_text(source, type_node))
    raise ComplexityIdentityError("method receiver type cannot be resolved")


def _declaration_name(source: bytes, node: Node) -> str:
    if node.type in {"function_declaration", "method_declaration"}:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            raise ComplexityIdentityError(f"{node.type} has no name")
        name = _node_text(source, name_node)
        if node.type == "method_declaration":
            receiver = _receiver_type(source, node.child_by_field_name("receiver"))
            return f"({receiver}).{name}"
        return name

    if node.type == "type_spec":
        name_node = node.child_by_field_name("name")
        type_node = node.child_by_field_name("type")
        if name_node is None or type_node is None or type_node.type != "interface_type":
            raise ComplexityIdentityError("interface declaration cannot be resolved")
        return f"interface {_node_text(source, name_node)}"

    raise ComplexityIdentityError(f"unsupported declaration node {node.type}")


def _contains_row(node: Node, row: int) -> bool:
    return node.start_point.row <= row <= node.end_point.row


def _smallest_node(nodes: list[Node]) -> Node | None:
    if not nodes:
        return None
    return min(
        nodes,
        key=lambda node: (
            node.end_byte - node.start_byte,
            -node.start_byte,
        ),
    )


def _normalized_ast(node: Node, source: bytes) -> str:
    pieces: list[str] = []

    def visit(current: Node) -> None:
        if current.type == "comment":
            return
        pieces.append(current.type)
        children = [child for child in current.children if child.type != "comment"]
        if not children:
            text = re.sub(r"\s+", "", _node_text(source, current))
            if text:
                pieces.append(f"={text}")
            return
        pieces.append("(")
        for child in children:
            visit(child)
            pieces.append(",")
        pieces.append(")")

    visit(node)
    return "".join(pieces)


class GoSourceIndex:
    """Resolve line-oriented lint positions to stable Go AST identities."""

    def __init__(self, path: str, source: bytes) -> None:
        self.path = path
        self.source = source
        tree = _build_go_parser().parse(source)
        if tree.root_node.has_error:
            raise ComplexityIdentityError(f"cannot parse Go source {path}")
        nodes = list(_walk(tree.root_node))
        self.declarations = [
            node
            for node in nodes
            if node.type in {"function_declaration", "method_declaration", "type_spec"}
        ]
        self.if_statements = [node for node in nodes if node.type == "if_statement"]

    @classmethod
    def from_path(cls, repo_root: Path, path: str) -> GoSourceIndex:
        source_path = repo_root / path
        try:
            source = source_path.read_bytes()
        except OSError as exc:
            raise ComplexityIdentityError(
                f"cannot read Go source {path}: {exc}"
            ) from exc
        return cls(path, source)

    def declaration_at(self, line: int) -> tuple[str, Node]:
        row = line - 1
        candidates = [node for node in self.declarations if _contains_row(node, row)]
        declaration = _smallest_node(candidates)
        if declaration is None:
            raise ComplexityIdentityError(
                f"no Go declaration contains {self.path}:{line}"
            )
        return _declaration_name(self.source, declaration), declaration

    def nestif_site(self, line: int, declaration: Node) -> str:
        row = line - 1
        candidates = [
            node
            for node in self.if_statements
            if _contains_row(node, row)
            and declaration.start_byte <= node.start_byte
            and node.end_byte <= declaration.end_byte
        ]
        statement = _smallest_node(candidates)
        if statement is None:
            raise ComplexityIdentityError(
                f"no if statement contains nestif diagnostic {self.path}:{line}"
            )

        condition = statement.child_by_field_name("condition") or statement
        normalized = _normalized_ast(condition, self.source)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        siblings = [
            node
            for node in self.if_statements
            if declaration.start_byte <= node.start_byte
            and node.end_byte <= declaration.end_byte
            and hashlib.sha256(
                _normalized_ast(
                    node.child_by_field_name("condition") or node,
                    self.source,
                ).encode("utf-8")
            ).hexdigest()
            == digest
        ]
        ordinal = (
            sorted(siblings, key=lambda node: node.start_byte).index(statement) + 1
        )
        return f"sha256:{digest}:{ordinal}"


class ComplexityFindingNormalizer:
    def __init__(
        self,
        repo_root: Path,
        config_path: Path,
        source_loader: Callable[[str], bytes] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.limits = load_complexity_limits(config_path)
        self._source_loader = source_loader
        self._sources: dict[str, GoSourceIndex] = {}

    def normalize(self, record: dict) -> ComplexityFinding:
        path = str(record.get("path", ""))
        line = int(record.get("line", 0))
        column = int(record.get("column", 0))
        linter = str(record.get("linter", ""))
        message = str(record.get("message", ""))
        if linter not in COMPLEXITY_LINTERS:
            raise ComplexityIdentityError(f"unsupported complexity linter {linter!r}")
        if not path.endswith(".go") or line <= 0:
            raise ComplexityIdentityError(
                f"invalid Go complexity diagnostic position {path}:{line}"
            )

        source = self._sources.get(path)
        if source is None:
            if self._source_loader is None:
                source = GoSourceIndex.from_path(self.repo_root, path)
            else:
                source = GoSourceIndex(path, self._source_loader(path))
            self._sources[path] = source
        declaration, declaration_node = source.declaration_at(line)
        observed, limit, site = self._metrics(linter, message)
        if linter == "nestif":
            site = source.nestif_site(line, declaration_node)
        return ComplexityFinding(
            identity=ComplexityIdentity(path, declaration, linter, site),
            observed=observed,
            limit=limit,
            line=line,
            column=column,
            message=message,
        )

    def _metrics(self, linter: str, message: str) -> tuple[int, int, str]:
        if linter == "cyclop":
            return self._matched_metrics(
                _CYCLOP_PATTERN, message, self.limits.cyclop, ""
            )
        if linter == "gocognit":
            return self._matched_metrics(
                _GOCOGNIT_PATTERN, message, self.limits.gocognit, ""
            )
        if linter == "interfacebloat":
            return self._matched_metrics(
                _INTERFACEBLOAT_PATTERN,
                message,
                self.limits.interfacebloat,
                "",
            )
        if linter == "nestif":
            match = _NESTIF_PATTERN.fullmatch(message)
            if match is None:
                raise ComplexityIdentityError(f"unrecognized nestif message: {message}")
            return int(match.group("observed")), self.limits.nestif, ""
        if linter == "funlen":
            match = _FUNLEN_PATTERN.fullmatch(message)
            if match is None:
                raise ComplexityIdentityError(f"unrecognized funlen message: {message}")
            observed = int(match.group("observed"))
            limit = int(match.group("limit"))
            if limit == self.limits.funlen_statements:
                return observed, limit, "statements"
            if limit == self.limits.funlen_lines:
                return observed, limit, "lines"
            raise ComplexityIdentityError(
                f"funlen message limit {limit} does not match configured limits"
            )
        raise ComplexityIdentityError(f"unsupported complexity linter {linter!r}")

    @staticmethod
    def _matched_metrics(
        pattern: re.Pattern[str],
        message: str,
        configured_limit: int,
        site: str,
    ) -> tuple[int, int, str]:
        match = pattern.fullmatch(message)
        if match is None:
            raise ComplexityIdentityError(f"unrecognized complexity message: {message}")
        observed = int(match.group("observed"))
        message_limit = int(match.group("limit"))
        if message_limit != configured_limit:
            raise ComplexityIdentityError(
                f"diagnostic limit {message_limit} does not match configured "
                f"limit {configured_limit}"
            )
        return observed, configured_limit, site
