"""Conservative, stdlib-only Router package selection (shadow by default)."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path, PurePosixPath

MODULE_PATH = "src/semantic-router"
# Audited repo-relative reads in *_test.go, including router-checked-docs in
# domains.yaml. Broad prefixes deliberately include future files in these trees.
INPUTS = {
    "config/": (
        "pkg/config",
        "pkg/dsl",
        "pkg/modeldownload",
        "pkg/systemonecodec",
        "pkg/configschema",
        "pkg/decision",
        "pkg/classification",
        "pkg/extproc",
        "pkg/cache",
        "pkg/k8s/configwriter",
    ),
    "website/": ("pkg/config", "pkg/apiserver", "pkg/catalog", "cmd", "pkg/extproc"),
    "dashboard/frontend/src/generated/": ("pkg/configschema",),
    "src/vllm-sr/cli/config_schema/": ("pkg/configschema",),
    "tools/test/services/provider-mocker/": ("pkg/protocolcodec",),
    "tools/agent/skills/vllm-sr-agent-operations/": ("pkg/config",),
    "tools/docker/riscv64/standalone.yaml": ("pkg/config",),
    "src/model-runtime/vllm_srun/api/openapi.yaml": ("pkg/modelservice",),
    "e2e/config/config.agent-smoke.cpu.yaml": ("cmd",),
    "e2e/profiles/dynamic-config/": ("pkg/k8s",),
    "deploy/": ("pkg/observability/metrics", "pkg/config", "pkg/modeldownload"),
    "e2e/": ("pkg/config",),
    "src/vllm-sr/cli/templates/": ("pkg/config",),
    # Cross-package fixture reads do not appear in Go's import graph.
    MODULE_PATH + "/pkg/config/testdata/": ("pkg/classification",),
    MODULE_PATH + "/pkg/codexreview/testdata/": ("pkg/extproc",),
    MODULE_PATH + "/pkg/protocolcodec/testdata/": ("pkg/extproc",),
    MODULE_PATH
    + "/pkg/routing/parity/testdata/": (
        "pkg/extproc",
        "internal/gatewayparity",
    ),
    MODULE_PATH + "/pkg/extproc/testdata/": ("internal/gatewayparity",),
}


# Packages whose tests find the repository root and then read paths the
# source does not spell as literals; any change outside the module runs them.
REPO_ROOT_READERS = (
    "internal/extensiontest",
    "pkg/config",
    "pkg/configschema",
    "pkg/systemonecodec",
)
# Tests that read every production Go file in the module, by package. Any
# production change runs them; their package's other tests run only when the
# package is selected for another reason.
SOURCE_SCANNERS = {
    "pkg/extproc": ("TestNoRouterPackageNamesARemovedClassifierModule",),
    "pkg/headers": (
        "TestResponsePathValuesAreSentAndDocumented",
        "TestOnlyThisPackageSpellsHeaderNames",
    ),
}
# Changes to how the lane runs, rather than to what it tests, run everything.
FULL_RUN_INPUTS = (
    ".github/workflows/test-and-build.yml",
    "tools/make/build-run-test.mk",
    "tools/ci/run_core_tests.py",
    "tools/ci/core_test_selection.py",
    "tools/ci/core_test_profiles.json",
    "tools/ci/storage_inventory.py",
)
# Non-Go sources the Go toolchain compiles into a package.
COMPILED_SUFFIXES = (".go", ".c", ".h", ".s", ".S", ".cc", ".cpp", ".syso")
_JOIN = re.compile(r"(?:filepath|path)\.Join\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
_FUNC = re.compile(r"^func (?:\([^)]*\) )?(\w+)\(", re.M)
_ASSIGN = re.compile(r"\b(\w+)\s*:?=\s*(.+)")


def _top_level(text: str) -> list[str]:
    """Split on commas outside parentheses."""
    parts, depth, current = [], 0, ""
    for character in text:
        depth += {"(": 1, ")": -1}.get(character, 0)
        if character == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += character
    return [*parts, current]


def _normalized(text: str) -> str:
    """Spell raw path strings as quoted ones and join expressions split over
    lines, so the line-oriented patterns below see Go as gofmt may write it."""
    text = re.sub(r"`([^`\"\\\n]*)`", r'"\1"', text)
    text = re.sub(r"\(\s*\n\s*", "(", text)
    text = re.sub(r",\s*\n\s*(?=[^\s/])", ", ", text)
    return re.sub(r",?\s*\n\s*\)", ")", text)


def _literals(arguments: str) -> list[str | None]:
    values = []
    for argument in _top_level(arguments):
        literal = _STRING.fullmatch(argument.strip())
        values.append(literal.group(1) if literal else None)
    return values


def _resolve(directory: PurePosixPath, value: str) -> PurePosixPath:
    target = directory
    for segment in value.split("/"):
        if segment == "..":
            target = target.parent
        elif segment not in {"", "."}:
            target = target / segment
    return target


def _root_expression(text: str, directory: PurePosixPath, roots: set[str]) -> bool:
    """Whether an expression is the repository root itself: a ".." chain that
    climbs to it from the file (optionally through filepath.Dir of the file,
    Clean or Abs), or a call to a function that returns it."""
    text = text.strip()
    wrapper = re.fullmatch(r"(?:filepath\.)?(?:Clean|Abs)\((.*)\)", text, re.S)
    if wrapper:
        return _root_expression(wrapper.group(1), directory, roots)
    join = re.fullmatch(r"(?:filepath|path)\.Join\((.*)\)", text, re.S)
    if join:
        arguments = [a.strip() for a in join.group(1).split(",")]
        if arguments and re.fullmatch(r"(?:filepath\.)?Dir\(.*\)", arguments[0]):
            arguments = arguments[1:]
        literals = _literals(",".join(arguments)) if arguments else []
        return (
            bool(literals)
            and all(literals)
            and (_resolve(directory, "/".join(literals)).as_posix() == ".")
        )
    literal = _STRING.fullmatch(text)
    if literal:
        return literal.group(1).startswith("..") and (
            _resolve(directory, literal.group(1)).as_posix() == "."
        )
    call = re.fullmatch(r"(\w+)\(.*\)", text, re.S)
    return bool(call) and call.group(1) in roots


def _root_functions(sources, directory: PurePosixPath) -> set[str]:
    """Functions whose return value is the repository root, to a fixed point."""
    bodies = {}
    for _, text in sources:
        starts = list(_FUNC.finditer(text))
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            bodies[match.group(1)] = text[match.end() : end]
    functions: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, body in bodies.items():
            if name in functions:
                continue
            local = _root_names(body, directory, functions)
            for returned in re.findall(r"\breturn\s+([^\n]+)", body):
                value = _top_level(returned)[0].strip()
                if value in local or _root_expression(value, directory, functions):
                    functions.add(name)
                    changed = True
                    break
    return functions


def _root_names(text: str, directory: PurePosixPath, functions: set[str]) -> set[str]:
    """Variables a file binds to the repository root."""
    names = set()
    for line in text.splitlines():
        assigned = re.search(r"\b(\w+)(?:\s*,\s*\w+)*\s*:?=\s*(.+)", line)
        if assigned and _root_expression(assigned.group(2), directory, functions):
            names.add(assigned.group(1))
    return names


def _range_paths(text: str) -> dict[str, list[str]]:
    """Loop variables ranging over a literal list of paths."""
    found = {}
    for match in re.finditer(
        r"for _, (\w+) := range \[\]string\{(.*?)\}\s*\{", text, re.S
    ):
        values = []
        for element in (e.strip() for e in _top_level(match.group(2))):
            if not element:
                continue
            literal = _STRING.fullmatch(element)
            join = re.fullmatch(r"filepath\.Join\((.*)\)", element, re.S)
            if literal:
                values.append(literal.group(1))
            elif join and all(_literals(join.group(1))):
                values.append("/".join(_literals(join.group(1))))
            else:
                values = None
                break
        if values:
            found[match.group(1)] = values
    return found


def _unresolved_root_use(text: str, names: set[str], binds_root) -> bool:
    """A root name used other than as the base of an all-literal Join, in its
    own definition, or to bind another root name, may build any path."""
    loops = _range_paths(text)
    literal_joins = [
        m.span()
        for m in _JOIN.finditer(text)
        if (args := _literals(m.group(1)))
        and args[0] is None
        and all(
            a is not None or b.strip() in loops
            for a, b in zip(args[1:], _top_level(m.group(1))[1:], strict=True)
        )
        and args[1:]
    ]
    for name in names:
        for use in re.finditer(rf"\b{re.escape(name)}\b(\(\))?", text):
            start = use.start()
            line_start = text.rfind("\n", 0, start) + 1
            line = text[line_start : text.find("\n", start)]
            if re.match(
                rf"\s*(?:func (?:\([^)]*\) )?)?{re.escape(name)}\b", line
            ) or re.match(rf"\s*(?:var\s+)?{re.escape(name)}\s*:?=", line):
                continue  # its definition
            if binds_root(line):
                continue  # binds another root name, tracked in its own right
            if not any(left <= start < right for left, right in literal_joins):
                return True
    return False


def repo_reads(root: Path) -> dict[str, set[str]]:
    """Repository paths Router test code reads, by reading package.

    A ".." chain resolves against the file's directory. A name bound to the
    repository root (a variable, or a function returning it) is followed:
    joined only with literals, it reads those paths; used any other way, the
    package may read anything, which the result records as ".". The guard test
    compares the result with INPUTS and REPO_ROOT_READERS."""
    top = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    files: dict[str, list[tuple[PurePosixPath, str]]] = {}
    for file in (root / MODULE_PATH).rglob("*.go"):
        relative = PurePosixPath(file.relative_to(root).as_posix())
        # Only tests and test-support packages read repository files.
        if not file.name.endswith("_test.go") and not any(
            "test" in part for part in relative.parent.parts
        ):
            continue
        package = relative.parent.relative_to(MODULE_PATH).as_posix()
        files.setdefault(package, []).append(
            (relative, _normalized(file.read_text(errors="ignore")))
        )
    found: dict[str, set[str]] = {}
    for package, sources in files.items():
        directory = sources[0][0].parent
        functions = _root_functions(sources, directory)
        unresolved = False
        for relative, text in sources:
            names = functions | _root_names(text, relative.parent, functions)
            unresolved = unresolved or _unresolved_root_use(
                text,
                names,
                lambda line, d=relative.parent, f=functions: bool(
                    (bound := re.search(r"\b\w+(?:\s*,\s*\w+)*\s*:?=\s*(.+)", line))
                    and _root_expression(bound.group(1), d, f)
                ),
            )
            # The root passed straight to a function (not bound to a name and
            # not joined with literals here) may be read anywhere below it.
            for line in text.splitlines():
                assigned = re.search(r"\b\w+(?:\s*,\s*\w+)*\s*:?=\s*(.+)", line)
                if "Join(" in line or (
                    assigned
                    and _root_expression(assigned.group(1), relative.parent, functions)
                ):
                    continue
                for literal in _STRING.finditer(line):
                    if (
                        literal.group(1).startswith("..")
                        and (
                            _resolve(relative.parent, literal.group(1)).as_posix()
                            == "."
                        )
                        and re.search(r"\w\(\s*\"" + re.escape(literal.group(1)), line)
                    ):
                        unresolved = True
            candidates = []
            loops = _range_paths(text)
            for match in _JOIN.finditer(text):
                arguments = [a.strip() for a in _top_level(match.group(1))]
                literals = _literals(match.group(1))
                pairs = list(zip(literals, arguments, strict=True))
                if literals and literals[0] and literals[0].startswith(".."):
                    base, rest = relative.parent, pairs
                elif literals[0] is None and (
                    arguments[0] in names or (len(literals) > 1 and literals[1] in top)
                ):
                    base, rest = PurePosixPath("."), pairs[1:]
                else:
                    continue
                chain = []
                for index, (literal, argument) in enumerate(rest):
                    if literal is not None:
                        chain.append(literal)
                        continue
                    prefix = _resolve(base, "/".join(chain)) if chain else base
                    tail = "/".join(x or "" for x, _ in rest[index + 1 :])
                    expanded = [
                        _resolve(prefix, value + ("/" + tail if tail else ""))
                        for value in loops.get(argument, ())
                    ]
                    # A loop elsewhere in the file can reuse the name, so trust
                    # the loop only when every path it names exists.
                    if (
                        expanded
                        and all(x is not None for x, _ in rest[index + 1 :])
                        and all((root / path).exists() for path in expanded)
                    ):
                        candidates += expanded
                    elif prefix.as_posix() == ".":
                        unresolved = True  # any path under the root
                    else:
                        candidates.append(prefix)  # a read under this directory
                    break
                else:
                    candidates.append(
                        _resolve(base, "/".join(chain)) if chain else base
                    )
            candidates += [
                _resolve(relative.parent, m.group(1))
                for m in _STRING.finditer(text)
                if m.group(1).startswith("../")
            ]
            for target in candidates:
                if (target.parts and target.parts[0] == "..") or not (
                    root / target
                ).exists():
                    continue  # fixtures for path checks, not files read
                if target.as_posix() == ".":
                    continue  # the root itself; its uses are judged above
                found.setdefault(target.as_posix(), set()).add(package)
        if unresolved:
            found.setdefault(".", set()).add(package)
    return found


def decode_graph(raw: str) -> list[dict]:
    """go list emits concatenated JSON objects, not a JSON array."""
    decoder = json.JSONDecoder()
    records = []
    while raw.strip():
        record, end = decoder.raw_decode(raw.lstrip())
        records.append(record)
        raw = raw.lstrip()[end:]
    if not records or any(
        p.get("Error") or p.get("DepsErrors") or p.get("Incomplete") for p in records
    ):
        raise ValueError("empty or incomplete go list graph")
    return records


def load_graph(root: Path, env: dict[str, str]) -> list[dict]:
    return decode_graph(
        subprocess.check_output(
            ["go", "list", "-mod=readonly", "-buildvcs=false", "-json", "./..."],
            cwd=root / MODULE_PATH,
            env=env,
            text=True,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    )


def changed_files(root: Path, env: dict[str, str]) -> list[str]:
    if "CORE_TEST_CHANGED_FILES" in env:
        paths = json.loads(env["CORE_TEST_CHANGED_FILES"])
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise ValueError("CORE_TEST_CHANGED_FILES must be a JSON string array")
        return paths
    base, head = env.get("CORE_TEST_BASE"), "HEAD"
    if not base and env.get("GITHUB_EVENT_PATH"):
        event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text())
        # A pull request checks out its merge commit: its first parent is the
        # base it applies to, as in the CI plan (ci-changes.yml).
        base = "HEAD^1" if "pull_request" in event else event.get("before")
    if not base:
        raise ValueError("no change base supplied")
    # No rename collapsing: both the old and new package must be considered.
    raw = subprocess.check_output(
        ["git", "diff", "--no-renames", "--name-only", "-z", base, head],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return [p for p in raw.split("\0") if p]


def select_packages(root: Path, graph: list[dict], paths: list[str]) -> dict:
    """Changed code spreads to every package that imports it; a file a test
    only reads selects just the packages that read it."""
    packages = {p["ImportPath"]: p for p in graph}
    directories = {
        Path(p["Dir"]).relative_to(root / MODULE_PATH).as_posix(): name
        for name, p in packages.items()
    }
    # go:embed files are compiled into their package, so they count as code.
    embedded = {
        f"{MODULE_PATH}/{directory}/{name}"
        for directory, package in directories.items()
        for name in packages[package].get("EmbedFiles") or []
    }
    reverse = {name: set() for name in packages}
    for name, package in packages.items():
        for field in ("Imports", "TestImports", "XTestImports"):
            for dependency in package.get(field, []):
                if dependency in reverse:
                    reverse[dependency].add(name)

    def owner(path: str) -> str | None:
        directory = PurePosixPath(path).parent.relative_to(MODULE_PATH)
        while directory.as_posix() not in directories and directory.as_posix() != ".":
            directory = directory.parent
        return directories.get(directory.as_posix())

    code, data, scanners, reasons = set(), set(), set(), []
    for path in paths:
        parts = PurePosixPath(path).parts
        if not parts or path.startswith("/") or ".." in parts:
            reasons.append(f"invalid path: {path}")
            continue
        for prefix, targets in INPUTS.items():
            if path.startswith(prefix) if prefix.endswith("/") else path == prefix:
                if set(targets) - directories.keys():
                    reasons.append(f"missing input owner: {path}")
                data.update(directories[t] for t in targets if t in directories)
        if path in FULL_RUN_INPUTS:
            reasons.append(f"core lane runner or toolchain: {path}")
            continue
        if not path.startswith(MODULE_PATH + "/"):
            # Router code reads nothing else outside the module;
            # test_router_reads_are_mapped keeps INPUTS and REPO_ROOT_READERS whole.
            data.update(directories[p] for p in REPO_ROOT_READERS if p in directories)
            continue
        if parts[-1] in {"go.mod", "go.sum"} or any(
            part in {"testutil", "testutils", "testhelper", "testhelpers"}
            for part in parts
        ):
            reasons.append(f"module metadata or shared test helper: {path}")
            continue
        package = owner(path)
        is_code = path.endswith(COMPILED_SUFFIXES) or path in embedded
        if package is None or (
            path.endswith(".go")
            and "testdata" not in parts
            and PurePosixPath(path).parent.relative_to(MODULE_PATH).as_posix()
            not in directories
        ):
            reasons.append(f"unknown/deleted Go package: {path}")
        elif path in embedded or (is_code and "testdata" not in parts):
            code.add(package)
            if not path.endswith("_test.go"):
                # These contracts scan production Go source across the module.
                scanners.update(
                    directories[p] for p in SOURCE_SCANNERS if p in directories
                )
        else:
            data.add(package)
    selected = set(packages) if reasons else set(code)
    pending = list(selected)
    while pending:
        for dependent in reverse[pending.pop()] - selected:
            selected.add(dependent)
            pending.append(dependent)
    filters = {}
    if not reasons:
        selected |= data
        # A package selected only to scan source runs just its scanning tests.
        for package in scanners - selected:
            relative = Path(packages[package]["Dir"]).relative_to(root / MODULE_PATH)
            filters[package] = list(SOURCE_SCANNERS[relative.as_posix()])
        selected |= scanners
    return {
        "all_packages": sorted(packages),
        "selected_packages": sorted(selected),
        "test_filters": filters,
        "changed_files": paths,
        "fallback_reasons": reasons,
    }


def plan_selection(root: Path, env: dict[str, str]) -> dict:
    try:
        graph = load_graph(root, env)
        try:
            return select_packages(root, graph, changed_files(root, env))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            return {
                "all_packages": sorted(p["ImportPath"] for p in graph),
                "selected_packages": sorted(p["ImportPath"] for p in graph),
                "fallback_reasons": [str(error)],
            }
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        # None is an explicit wildcard when graph discovery itself is unavailable.
        return {
            "all_packages": None,
            "selected_packages": None,
            "fallback_reasons": [f"graph unavailable: {error}"],
        }


def write_observation(
    output: Path, selection: dict, logs: list[Path], *, completed: bool
) -> dict:
    failed, failed_tests = set(), set()
    for log in logs:
        for line in log.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            package = event.get("Package") or event.get("ImportPath")
            if event.get("Action") in {"fail", "build-fail"} and package:
                failed.add(package)
                if event.get("Test"):
                    failed_tests.add((package, event["Test"].split("/", 1)[0]))
    selected = selection["selected_packages"]
    missed = sorted(failed - set(selected)) if selected is not None else []
    # A failure inside a filtered package, in a test the filter left out.
    filters = selection.get("test_filters") or {}
    missed += sorted(
        f"{package}/{test}"
        for package, test in failed_tests
        if package in filters and test not in filters[package]
    )
    record = {
        **selection,
        "completed": completed,
        "failing_packages": sorted(failed),
        "failing_packages_outside_selection": missed,
        "failure_outside_selection": bool(missed),
        "comparison_complete": completed,
    }
    (output / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def summary_line(record: dict) -> str:
    """One line for the job log and step summary, so a reviewer can compare
    the selection with the full run without downloading the artifact."""
    selected, every = record["selected_packages"], record["all_packages"]
    size = (
        f"{len(selected)} of {len(every)} packages"
        if selected is not None and every is not None
        else "all packages (no graph)"
    )
    fallback = "; ".join(record.get("fallback_reasons", [])[:3]) or "none"
    missed = ", ".join(record["failing_packages_outside_selection"]) or "none"
    return (
        f"Core test selection ({record['mode']}): {size} selected. "
        f"Fallback: {fallback}. Failures outside the selection: {missed}."
    )
