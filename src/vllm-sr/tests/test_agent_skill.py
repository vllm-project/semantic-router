"""The agent skill names only commands and options that exist.

An agent runs the skill's shell blocks as written, so every `vllm-sr` command
line in them parses with the CLI's own parser, and every installer option they
pass is one `install.sh` reads. Every command and option the prose names
exists too. The Router's config package loads the skill's YAML blocks
(src/semantic-router/pkg/config/agent_skill_examples_test.go).
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import click
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli.main import main  # noqa: E402

SKILL = REPO_ROOT / "tools" / "agent" / "skills" / "vllm-sr-agent-operations"
PAGES = [SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]
INSTALLER = REPO_ROOT / "install.sh"
FENCE = re.compile(r"^[ \t]*```(\w*)[^\n]*\n(.*?)^[ \t]*```", re.M | re.S)
SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)(\w+)\1")
CONTROL = {"|", "||", "&&", ";", "&", "(", ")", ";;"}
REDIRECTS = {">", ">>", "<", "<<", "<<<", ">&", "<&", "&>", ">|", "<>", "<<-"}
KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "!", "{", "}"}
WRAPPERS = {"nohup", "setsid", "exec", "time"}


def _rel(page: Path) -> str:
    return str(page.relative_to(REPO_ROOT))


def _shell_blocks() -> list[tuple[str, str]]:
    return [
        (_rel(page), match.group(2))
        for page in PAGES
        for match in FENCE.finditer(page.read_text(encoding="utf-8"))
        if match.group(1) in {"bash", "sh", "shell"}
    ]


def _prose_spans() -> list[tuple[str, str]]:
    spans = []
    for page in PAGES:
        prose = FENCE.sub("", page.read_text(encoding="utf-8"))
        spans.extend((_rel(page), span.strip()) for span in SPAN.findall(prose))
    return spans


def _without_heredocs(script: str) -> str:
    """Drop here-document bodies, which are another program's input."""
    lines, kept, terminator = script.splitlines(), [], None
    for line in lines:
        if terminator is not None:
            if line.strip() == terminator:
                terminator = None
            continue
        kept.append(line)
        if match := HEREDOC.search(line):
            terminator = match.group(2)
    return "\n".join(kept)


def _logical(script: str) -> str:
    """Join continuations, drop comments and redirected descriptors' numbers,
    and end each command at a newline."""
    out, quote, at_word_start, index = [], None, True, 0
    while index < len(script):
        char = script[index]
        if (
            not quote
            and at_word_start
            and (fd := re.match(r"\d+(?=[<>])", script[index:]))
        ):
            index += fd.end()
            continue
        if quote:
            out.append(char)
            if char == "\\" and quote == '"' and index + 1 < len(script):
                out.append(script[index + 1])
                index += 1
            elif char == quote:
                quote = None
        elif char == "\\" and script[index + 1 : index + 2] == "\n":
            index += 1
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char == "#" and at_word_start:
            while index < len(script) and script[index] != "\n":
                index += 1
            continue
        elif char == "\n":
            out.append(" ; ")
        else:
            out.append(char)
        at_word_start = char.isspace() or char in ";|&()"
        index += 1
    return "".join(out)


def _simple_commands(script: str) -> list[list[str]]:
    lexer = shlex.shlex(
        _logical(_without_heredocs(script)), posix=True, punctuation_chars=True
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    commands, current, tokens = [], [], list(lexer)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in CONTROL:
            commands.append(current)
            current = []
        elif token in REDIRECTS:
            index += 1
        else:
            current.append(token)
        index += 1
    commands.append(current)
    return [command for command in commands if command]


def _program(command: list[str]) -> tuple[str, list[str]]:
    """The program a simple command runs, past keywords, wrappers and assignments."""
    words = list(command)
    while words:
        word = words[0]
        if word in KEYWORDS or word in WRAPPERS or re.fullmatch(r"\w+=.*", word):
            words.pop(0)
        elif word == "timeout" and len(words) > 1:
            del words[:2]
        else:
            break
    return (words[0], words[1:]) if words else ("", [])


def _script_commands(script: str) -> list[tuple[str, list[str], list[str]]]:
    """Every (program, arguments, previous pipeline stage) of a script, nested `sh -c` included."""
    found = []
    previous: list[str] = []
    for command in _simple_commands(script):
        program, arguments = _program(command)
        if program in {"sh", "bash"} and arguments[:1] == ["-c"] and len(arguments) > 1:
            found.extend(_script_commands(arguments[1]))
        found.append((program, arguments, previous))
        previous = command
    return found


def _option_names(command: click.Command) -> set[str]:
    names = {"--help"}
    for param in command.params:
        if isinstance(param, click.Option):
            names.update(param.opts)
            names.update(param.secondary_opts)
    return names


def _all_option_names() -> set[str]:
    names, pending = set(), [main]
    while pending:
        command = pending.pop()
        names |= _option_names(command)
        if isinstance(command, click.Group):
            pending.extend(command.commands.values())
    return names


def _installer_options() -> set[str]:
    text = INSTALLER.read_text(encoding="utf-8")
    return set(re.findall(r"^\s+(--[a-z][a-z-]*)\)", text, re.M))


def _walk(arguments: list[str]) -> tuple[click.Command, click.Context, list[str]]:
    """Resolve the command a line names, consuming each group's own options."""
    command, rest = main, list(arguments)
    context = click.Context(main, info_name="vllm-sr")
    while isinstance(command, click.Group):
        while rest and rest[0].startswith("-"):
            word = rest.pop(0)
            name = word.split("=", 1)[0]
            if name == "--help":
                return command, context, ["--help"]
            option = next(
                (
                    param
                    for param in command.params
                    if isinstance(param, click.Option)
                    and name in (*param.opts, *param.secondary_opts)
                ),
                None,
            )
            if option is None:
                raise AssertionError(f"`{context.command_path}` has no option {name}")
            if not (option.is_flag or option.count or "=" in word) and rest:
                rest.pop(0)
        if not rest:
            if not command.invoke_without_command:
                raise AssertionError(f"`{context.command_path}` needs a command")
            return command, context, rest
        name = rest.pop(0)
        sub = command.get_command(context, name)
        if sub is None:
            raise AssertionError(f"`{context.command_path}` has no command {name!r}")
        command, context = sub, click.Context(sub, info_name=name, parent=context)
    return command, context, rest


@pytest.fixture
def paths_need_not_exist(monkeypatch):
    """The files a command names exist only once an agent runs the steps."""
    convert = click.Path.convert

    def lenient(self, value, param, ctx):
        exists, self.exists = self.exists, False
        try:
            return convert(self, value, param, ctx)
        finally:
            self.exists = exists

    monkeypatch.setattr(click.Path, "convert", lenient)


def test_shell_blocks_run_only_commands_the_cli_parses(paths_need_not_exist):
    checked = 0
    for where, script in _shell_blocks():
        for program, arguments, _ in _script_commands(script):
            if program != "vllm-sr":
                continue
            checked += 1
            line = " ".join(["vllm-sr", *arguments])
            try:
                command, context, rest = _walk(arguments)
                if "--help" in rest:
                    continue
                command.make_context(context.info_name, rest, parent=context.parent)
            except (click.ClickException, AssertionError) as error:
                raise AssertionError(f"{where}: {line}: {error}") from error
    assert checked >= 30


def test_installer_options_are_read_by_the_installer():
    options = _installer_options()
    assert {"--channel", "--mode", "--runtime", "--no-launch", "--pip-spec"} <= options
    checked = 0
    for where, script in _shell_blocks():
        for program, arguments, previous in _script_commands(script):
            if program != "bash" or not any(
                w.endswith("/install.sh") for w in previous
            ):
                continue
            checked += 1
            flags = [
                word for word in arguments if word.startswith("--") and word != "--"
            ]
            unknown = [flag for flag in flags if flag not in options]
            assert not unknown, f"{where}: install.sh reads no {unknown}"
    assert checked >= 2


def test_prose_names_only_commands_and_options_that_exist():
    options = _all_option_names() | _installer_options()
    checked = 0
    for where, span in _prose_spans():
        if span.startswith("vllm-sr "):
            checked += 1
            try:
                words = shlex.split(span)
            except ValueError as error:
                raise AssertionError(f"{where}: `{span}`: {error}") from error
            try:
                command, _, rest = _walk(words[1:])
            except AssertionError as error:
                raise AssertionError(f"{where}: `{span}`: {error}") from error
            known = _option_names(command)
            unknown = [
                word.split("=", 1)[0]
                for word in rest
                if word.startswith("-") and word.split("=", 1)[0] not in known
            ]
            assert not unknown, f"{where}: `{span}`: no option {unknown}"
        elif re.match(r"-{1,2}[a-z]", span):
            checked += 1
            name = span.split()[0].split("=", 1)[0]
            assert (
                name in options
            ), f"{where}: `{span}`: no vllm-sr or installer option {name}"
    assert checked >= 30
