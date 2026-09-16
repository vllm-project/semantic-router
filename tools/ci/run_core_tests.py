#!/usr/bin/env python3
"""Run discovered Go contracts separately from required storage integrations."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from ci_results import actual_platform
from storage_inventory import ROOT, ginkgo_storage_suites, go_storage_tests

MODULE = ROOT / "src/semantic-router"
MODULE_NAME = "github.com/vllm-project/semantic-router/src/semantic-router"
SERVICES = ("milvus", "qdrant", "redis", "valkey", "postgres")


def package_id(package: str) -> str:
    return package.replace(MODULE_NAME + "/", "./")


def run_go(
    args: list[str], output: Path, env: dict[str, str], *, module: Path = MODULE
) -> list[dict]:
    command = ["go", "test", "-json", "-count=1", "-timeout=20m", *args]
    print("+ " + " ".join(command), flush=True)
    with output.open("w") as handle:
        result = subprocess.run(
            command, cwd=module, env=env, stdout=handle, stderr=None, check=False
        )
    events = [
        json.loads(line)
        for line in output.read_text().splitlines()
        if line.startswith("{")
    ]
    if result.returncode:
        diagnostics = [
            event.get("Output", "")
            for event in events
            if event.get("Action") == "build-output"
        ]
        print("".join(diagnostics[-80:]))
        failures = [
            event.get("Output", "")
            for event in events
            if event.get("Action") == "output"
        ]
        print("".join(failures[-80:]))
        raise ValueError(f"Go invocation failed; inspect {output}")
    return events


def terminal_cases(events: list[dict], *, roots_only: bool = False) -> list[dict]:
    cases = []
    for event in events:
        name = event.get("Test")
        action = event.get("Action")
        if not name or action not in {"pass", "fail", "skip"}:
            continue
        if roots_only and "/" in name:
            continue
        cases.append(
            {
                "id": f"{package_id(event['Package'])}/{name}",
                "status": {"pass": "passed", "fail": "failed", "skip": "skipped"}[
                    action
                ],
            }
        )
    return cases


def spec_name(spec: dict) -> str:
    return " ".join([*spec.get("ContainerHierarchyTexts", []), spec["LeafNodeText"]])


def spec_cases(report: Path, package: str, names: set[str] | None = None) -> list[dict]:
    suites = json.loads(report.read_text())
    cases = []
    for suite in suites:
        for spec in suite["SpecReports"]:
            if spec.get("LeafNodeType") != "It":
                continue
            name = spec_name(spec)
            if names is None:
                # A dry-run skips label-filtered specs. Pending selected specs
                # must remain in the inventory so they cannot qualify as passed.
                if spec["State"] not in {"passed", "pending"}:
                    continue
            elif name not in names:
                continue
            cases.append({"id": f"{package}/spec/{name}", "status": spec["State"]})
    return cases


def require_complete(cases: list[dict], expected: list[str]) -> None:
    actual = [case["id"] for case in cases]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("selected test inventory is empty or contains duplicates")
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(
            f"incomplete test inventory: missing={sorted(set(expected) - set(actual))}, extra={sorted(set(actual) - set(expected))}"
        )
    failed = [case for case in cases if case["status"] != "passed"]
    if failed:
        raise ValueError(f"required tests did not pass: {failed}")


def execute_storage(
    output: Path, env: dict[str, str], services: tuple[str, ...]
) -> dict:
    inventory = go_storage_tests()
    ginkgo = ginkgo_storage_suites()
    env = {**env, "VLLM_SR_REQUIRE_STORAGE_TESTS": "1"}
    cases, expected = [], []
    for backend in services:
        env[f"SKIP_{backend.upper()}_TESTS"] = "false"
        for package in sorted(set(inventory) | set(ginkgo)):
            names = inventory.get(package, {}).get(backend, [])
            suite = ginkgo.get(package)
            prefix = output / f"{backend}-{package.rsplit('/', 1)[-1]}"
            spec_expected = []
            flags = []
            if suite:
                dry = prefix.with_suffix(".dry.json")
                run_go(
                    [
                        "-run",
                        f"^{suite}$",
                        package,
                        "-ginkgo.dry-run",
                        f"-ginkgo.label-filter=storage:{backend}",
                        f"-ginkgo.json-report={dry}",
                    ],
                    prefix.with_suffix(".dry.jsonl"),
                    env,
                )
                spec_expected = spec_cases(dry, package)
                if spec_expected:
                    names = [*names, suite]
                    flags = [
                        f"-ginkgo.label-filter=storage:{backend}",
                        f"-ginkgo.json-report={prefix.with_suffix('.specs.json')}",
                    ]
            if not names:
                continue
            selected = [f"{package}/{name}" for name in names]
            events = run_go(
                [
                    "-tags=integration",
                    "-run",
                    "^(" + "|".join(names) + ")$",
                    package,
                    *flags,
                ],
                prefix.with_suffix(".jsonl"),
                env,
            )
            roots = terminal_cases(events, roots_only=True)
            require_complete(roots, selected)
            # Subtest skips also invalidate a selected storage integration.
            if any(case["status"] != "passed" for case in terminal_cases(events)):
                raise ValueError(
                    f"required storage subtest skipped or failed: {package}/{backend}"
                )
            cases.extend({**case, "id": f"{backend}:{case['id']}"} for case in roots)
            expected.extend(f"{backend}:{name}" for name in selected)
            if spec_expected:
                wanted = {case["id"].split("/spec/", 1)[1] for case in spec_expected}
                measured = spec_cases(
                    prefix.with_suffix(".specs.json"), package, wanted
                )
                require_complete(measured, [case["id"] for case in spec_expected])
                cases.extend(
                    {**case, "id": f"{backend}:{case['id']}"} for case in measured
                )
                expected.extend(f"{backend}:{case['id']}" for case in spec_expected)
    require_complete(cases, expected)
    return {"cases": cases, "expected_cases": expected}


def ginkgo_suites() -> dict[str, str]:
    suites = {}
    for path in MODULE.rglob("*_test.go"):
        current = None
        for line in path.read_text().splitlines():
            declaration = re.match(r"^func (\w+)\(", line)
            if declaration:
                current = declaration[1]
            if "RunSpecs(t," in line and current and current.startswith("Test"):
                package = "./" + str(path.parent.relative_to(MODULE))
                suites[package] = current
    return suites


def profile_exclusions() -> tuple[dict[str, set[str]], dict]:
    profiles = json.loads((ROOT / "tools/ci/core_test_profiles.json").read_text())
    excluded: dict[str, set[str]] = {}
    for record in profiles["excluded"]:
        source = ROOT / record["source"]
        if (
            not record["reason"]
            or not record["profile"]
            or not re.search(
                rf"^func {re.escape(record['test'])}\(", source.read_text(), re.M
            )
        ):
            raise ValueError(f"stale or unowned core exclusion: {record}")
        entries = excluded.setdefault(record["package"], set())
        if record["test"] in entries:
            raise ValueError(f"duplicate core exclusion: {record}")
        entries.add(record["test"])
    return excluded, profiles


def unit_exclusions() -> tuple[dict[str, set[str]], dict]:
    excluded, profiles = profile_exclusions()
    for package, backends in go_storage_tests().items():
        excluded.setdefault(package, set()).update(
            name for names in backends.values() for name in names
        )
    return excluded, profiles


def race_inventory(profiles: dict) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    for record in profiles["race"]:
        source = ROOT / record["source"]
        if not record["reason"] or not re.search(
            rf"^func {re.escape(record['test'])}\(", source.read_text(), re.M
        ):
            raise ValueError(f"stale or unowned race contract: {record}")
        entries = required.setdefault(record["package"], set())
        if record["test"] in entries:
            raise ValueError(f"duplicate race contract: {record}")
        entries.add(record["test"])
    return required


def unit_groups(
    inventory: dict[str, set[str]],
    excluded: dict[str, set[str]],
    suites: dict[str, str],
    race_required: dict[str, set[str]],
) -> tuple[dict, list[str]]:
    groups, expected = {}, []
    for package, names in race_required.items():
        if names - inventory.get(package, set()):
            raise ValueError(f"required race tests not collected: {package}")
    for package, names in inventory.items():
        selected = names - excluded.get(package, set()) - {suites.get(package)}
        race_names = race_required.get(package, set())
        if race_names - selected:
            raise ValueError(f"required race tests excluded from core: {package}")
        expected.extend(f"{package}/{name}" for name in sorted(selected))
        for race, partition in ((False, selected - race_names), (True, race_names)):
            if not partition:
                continue
            skipped = tuple(sorted(names - partition))
            groups.setdefault((skipped, race), []).append(package)
    return groups, expected


def collected_go_inventory(events: list[dict]) -> dict[str, set[str]]:
    inventory: dict[str, set[str]] = {}
    for event in events:
        name = event.get("Output", "").strip()
        if re.fullmatch(r"(?:Test|Fuzz|Example)\w*", name):
            inventory.setdefault(package_id(event["Package"]), set()).add(name)
    return inventory


def repository_go_tools() -> dict[str, tuple[list[str], list[str]]]:
    """Use the Make registry that builds external commands with this module."""
    lines = subprocess.check_output(
        ["make", "--no-print-directory", "-s", "go-tools-inventory"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    tools = {}
    for line in lines:
        name, flag_text, source_text = line.split("\t")
        flags, sources = shlex.split(flag_text), shlex.split(source_text)
        if not name or name in tools or flags not in ([], ["-race"]):
            raise ValueError(f"invalid repository Go tool: {line}")
        if (
            not sources
            or len(sources) != len(set(sources))
            or any(
                not (MODULE / source).resolve().is_relative_to(ROOT)
                or not (MODULE / source).is_file()
                or Path(source).suffix != ".go"
                for source in sources
            )
        ):
            raise ValueError(f"repository Go tool has invalid sources: {name}")
        tools[name] = (sources, flags)
    if not tools:
        raise ValueError("repository Go tool inventory is empty")
    return tools


def execute_tool_units(
    output: Path, env: dict[str, str]
) -> tuple[list[dict], list[str]]:
    cases, expected = [], []
    for name, (sources, flags) in repository_go_tools().items():
        prefix = output / ("tool-" + name)
        listing = run_go(
            [*flags, "-list", "^(Test|Fuzz|Example)", *sources],
            prefix.with_suffix(".inventory.jsonl"),
            env,
        )
        selected = [
            f"{package}/{test}"
            for package, names in collected_go_inventory(listing).items()
            for test in sorted(names)
        ]
        # A command without tests is still compiled by discovery. It contributes
        # no invented test case. Commands with tests must execute every root.
        if not selected:
            continue
        events = run_go([*flags, *sources], prefix.with_suffix(".jsonl"), env)
        roots = terminal_cases(events, roots_only=True)
        require_complete(roots, selected)
        if any(case["status"] != "passed" for case in terminal_cases(events)):
            raise ValueError(f"repository Go tool test skipped or failed: {name}")
        cases.extend({**case, "id": f"tool:{name}/{case['id']}"} for case in roots)
        expected.extend(f"tool:{name}/{identity}" for identity in selected)
    return cases, expected


def execute_unit(output: Path, env: dict[str, str]) -> dict:
    excluded, profiles = unit_exclusions()
    suites = ginkgo_suites()
    listed = run_go(
        ["-list", "^(Test|Fuzz|Example)", "./..."], output / "inventory.jsonl", env
    )
    inventory = collected_go_inventory(listed)
    cases = []
    race_required = race_inventory(profiles)
    groups, expected = unit_groups(inventory, excluded, suites, race_required)
    for index, ((skipped, race), packages) in enumerate(groups.items()):
        skip = (
            ["-skip", "^(" + "|".join(map(re.escape, skipped)) + ")$"]
            if skipped
            else []
        )
        if race:
            selected = set().union(*(race_required[package] for package in packages))
            flags = ["-race", "-run", "^(" + "|".join(sorted(selected)) + ")$"]
        else:
            flags = skip
        events = run_go(
            [*flags, *packages],
            output / f"unit-{index}.jsonl",
            env,
        )
        # Root cases express the framework's discovered inventory. A skipped
        # nested case still invalidates its mandatory parent.
        roots = terminal_cases(events, roots_only=True)
        for case in terminal_cases(events):
            if case["status"] != "passed":
                raise ValueError(f"unexpected mandatory Go result: {case}")
        cases.extend(roots)
    labels = " && ".join(
        ["!storage", *("!" + label for label in profiles["ginkgo_excluded_labels"])]
    )
    for package, suite in sorted(suites.items()):
        prefix = output / ("ginkgo-" + package.rsplit("/", 1)[-1])
        dry = prefix.with_suffix(".dry.json")
        flags = ["-run", f"^{suite}$", package, f"-ginkgo.label-filter={labels}"]
        run_go(
            [*flags, "-ginkgo.dry-run", f"-ginkgo.json-report={dry}"],
            prefix.with_suffix(".dry.jsonl"),
            env,
        )
        discovered = spec_cases(dry, package)
        selected_names = {case["id"].split("/spec/", 1)[1] for case in discovered}
        events = run_go(
            [*flags, f"-ginkgo.json-report={prefix.with_suffix('.specs.json')}"],
            prefix.with_suffix(".jsonl"),
            env,
        )
        roots = terminal_cases(events, roots_only=True)
        require_complete(roots, [f"{package}/{suite}"])
        measured = spec_cases(
            prefix.with_suffix(".specs.json"), package, selected_names
        )
        require_complete(measured, [case["id"] for case in discovered])
        cases.extend([*roots, *measured])
        expected.extend([f"{package}/{suite}", *(case["id"] for case in discovered)])
    tool_cases, tool_expected = execute_tool_units(output, env)
    cases.extend(tool_cases)
    expected.extend(tool_expected)
    require_complete(cases, expected)
    return {"cases": cases, "expected_cases": expected, "excluded_profiles": profiles}


def execute_owned(output: Path, env: dict[str, str]) -> dict:
    excluded, profiles = profile_exclusions()
    selections = (
        (
            ROOT / "candle-binding",
            ["."],
            "^Test(Owned.*|NewRegexProvider|RegexProvider_.*|UtilityFunctions)$",
            [],
        ),
        (ROOT / "onnx-binding", ["./instance"], "^Test", []),
        (
            MODULE,
            ["./pkg/modelruntime", "./pkg/modeldownload"],
            "^(TestOwnedImplicitORTEmbeddingAndExplicitCandleOverride|"
            "TestGlobalEmbeddingViewDoesNotChangeOtherModelFamilies|"
            "TestImplicitEmbeddingProvisioningFollowsBuildProvider)$",
            [
                "-ldflags=-X github.com/vllm-project/semantic-router/src/semantic-router/pkg/config.defaultModelProvider=ort"
            ],
        ),
    )
    cases, expected = [], []
    for index, (module, packages, pattern, flags) in enumerate(selections):
        # Binding modules use their repository-relative module name. Core package
        # exclusions belong to the unit invocation; these explicit alternate
        # provider-default cases are mandatory in this owned invocation.
        skipped = set().union(
            *(
                excluded.get(str((module / package).relative_to(ROOT)), set())
                for package in packages
            )
        )
        skip_flags = (
            ["-skip", "^(" + "|".join(map(re.escape, sorted(skipped))) + ")$"]
            if skipped
            else []
        )
        listing = run_go(
            [*flags, "-list", pattern, *packages],
            output / f"owned-{index}.inventory.jsonl",
            env,
            module=module,
        )
        selected = [
            f"{package_id(event['Package'])}/{event['Output'].strip()}"
            for event in listing
            if re.fullmatch(r"Test\w+", event.get("Output", "").strip())
            and event["Output"].strip() not in skipped
        ]
        events = run_go(
            [*flags, *skip_flags, "-race", "-run", pattern, *packages],
            output / f"owned-{index}.jsonl",
            env,
            module=module,
        )
        roots = terminal_cases(events, roots_only=True)
        require_complete(roots, selected)
        if any(case["status"] != "passed" for case in terminal_cases(events)):
            raise ValueError("required owned-native subtest skipped or failed")
        cases.extend(roots)
        expected.extend(selected)
    require_complete(cases, expected)
    return {"cases": cases, "expected_cases": expected, "excluded_profiles": profiles}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("unit", "storage", "owned"), default="storage"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--services", default=",".join(SERVICES))
    args = parser.parse_args()
    services = tuple(args.services.split(","))
    if (
        not services
        or len(services) != len(set(services))
        or set(services) - set(SERVICES)
    ):
        parser.error("choose distinct supported services: " + ",".join(SERVICES))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if args.mode == "unit":
        evidence = execute_unit(args.output, env)
    elif args.mode == "owned":
        evidence = execute_owned(args.output, env)
    else:
        evidence = execute_storage(args.output, env, services)
    evidence.update(
        runtime="none", device="none", platform=actual_platform(), artifacts=[]
    )
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
