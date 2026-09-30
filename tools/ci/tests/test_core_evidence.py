"""Negative coverage for collected Go/Ginkgo inventories and report adapters."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_core_tests import (
    collected_go_inventory,
    execute_owned,
    execute_tool_units,
    profile_exclusions,
    race_inventory,
    repository_go_tools,
    require_complete,
    spec_cases,
    terminal_cases,
    unit_groups,
)
from run_model_tests import OWNED_OMNI_TESTS
from workflow_evidence import go_cases, junit_cases


class RequiredInventoryTests(unittest.TestCase):
    def test_external_go_commands_keep_distinct_collected_case_identities(self) -> None:
        listed = [{"Package": "command-line-arguments", "Output": "TestOwned\n"}]
        passed = [
            {"Package": "command-line-arguments", "Test": "TestOwned", "Action": "pass"}
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "run_core_tests.repository_go_tools",
                return_value={"first": (["a.go"], []), "second": (["b.go"], [])},
            ),
            mock.patch(
                "run_core_tests.run_go", side_effect=[listed, passed, listed, passed]
            ),
        ):
            cases, expected = execute_tool_units(Path(directory), {})
        require_complete(cases, expected)
        self.assertEqual(len(cases), 2)
        self.assertEqual(
            {case["id"].split("/", 1)[0] for case in cases},
            {"tool:first", "tool:second"},
        )

    def test_external_go_command_cannot_skip_a_collected_case(self) -> None:
        listed = [{"Package": "command-line-arguments", "Output": "TestOwned\n"}]
        skipped = [
            {"Package": "command-line-arguments", "Test": "TestOwned", "Action": "skip"}
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "run_core_tests.repository_go_tools",
                return_value={"first": (["a.go"], [])},
            ),
            mock.patch("run_core_tests.run_go", side_effect=[listed, skipped]),
            self.assertRaisesRegex(ValueError, "did not pass"),
        ):
            execute_tool_units(Path(directory), {})

    def test_modelcompat_registry_separates_optional_checkpoint_and_keeps_race(
        self,
    ) -> None:
        sources, flags = repository_go_tools()["modelcompat"]
        self.assertEqual(flags, ["-race"])
        self.assertEqual(
            {Path(source).name for source in sources},
            {"main.go", "main_test.go", "make_test.go"},
        )

    def test_tool_flags_apply_to_discovery_and_execution(self) -> None:
        listed = [{"Package": "command-line-arguments", "Output": "TestOwned\n"}]
        passed = [
            {"Package": "command-line-arguments", "Test": "TestOwned", "Action": "pass"}
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "run_core_tests.repository_go_tools",
                return_value={"modelcompat": (["main.go"], ["-race"])},
            ),
            mock.patch("run_core_tests.run_go", side_effect=[listed, passed]) as run,
        ):
            cases, expected = execute_tool_units(Path(directory), {})
        require_complete(cases, expected)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["-race", "-list", "^(Test|Fuzz|Example)", "main.go"],
        )
        self.assertEqual(run.call_args_list[1].args[0], ["-race", "main.go"])

    def test_tool_inventory_rejects_filter_flags_and_invalid_sources(self) -> None:
        for flags, sources in (
            ("-run=^Nothing$", "../../tools/modelcompat/main.go"),
            ("", ""),
            ("", "/outside-repository.go"),
            ("", "../../tools/modelcompat/main.go ../../tools/modelcompat/main.go"),
        ):
            with (
                self.subTest(flags=flags, sources=sources),
                mock.patch(
                    "run_core_tests.subprocess.check_output",
                    return_value=f"modelcompat\t{flags}\t{sources}\n",
                ),
                self.assertRaises(ValueError),
            ):
                repository_go_tools()

    def test_go_inventory_includes_executed_examples_and_fuzz_seed_roots(self) -> None:
        package = (
            "github.com/vllm-project/semantic-router/src/semantic-router/pkg/example"
        )
        names = ["TestUnit", "FuzzCodec", "Example", "ExampleCodec"]
        listing = [
            {"Package": package, "Action": "output", "Output": name + "\n"}
            for name in [*names, "BenchmarkNotExecuted"]
        ]
        inventory = collected_go_inventory(listing)
        self.assertEqual(inventory, {"./pkg/example": set(names)})
        events = [
            {"Package": package, "Action": "pass", "Test": name}
            for name in [*names, "FuzzCodec/seed#0"]
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, rows in (("listed.jsonl", listing), ("actual.jsonl", events)):
                (root / filename).write_text("\n".join(json.dumps(row) for row in rows))
            cases, expected = go_cases(
                root / "actual.jsonl", root / "listed.jsonl", set()
            )
            require_complete(cases, expected)
            self.assertEqual(len(cases), len(names))
            events[-1]["Action"] = "skip"
            (root / "actual.jsonl").write_text(
                "\n".join(json.dumps(row) for row in events)
            )
            with self.assertRaisesRegex(ValueError, "subtest"):
                go_cases(root / "actual.jsonl", root / "listed.jsonl", set())

    def test_skip_missing_duplicate_and_empty_fail(self) -> None:
        for cases, expected in (
            ([{"id": "case", "status": "skipped"}], ["case"]),
            ([], ["case"]),
            ([{"id": "case", "status": "passed"}] * 2, ["case"]),
            ([], []),
        ):
            with (
                self.subTest(cases=cases, expected=expected),
                self.assertRaises(ValueError),
            ):
                require_complete(cases, expected)

    def test_ginkgo_discovery_excludes_filtered_specs_but_selected_skip_remains(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "SpecReports": [
                                {
                                    "ContainerHierarchyTexts": ["cache"],
                                    "LeafNodeText": "required",
                                    "LeafNodeType": "It",
                                    "State": "passed",
                                },
                                {
                                    "ContainerHierarchyTexts": ["cache"],
                                    "LeafNodeText": "unselected",
                                    "LeafNodeType": "It",
                                    "State": "skipped",
                                },
                            ]
                        }
                    ]
                )
            )
            selected = spec_cases(path, "./pkg/cache")
            self.assertEqual(
                [case["id"] for case in selected], ["./pkg/cache/spec/cache required"]
            )
            path.write_text(
                path.read_text().replace('"State": "passed"', '"State": "skipped"')
            )
            actual = spec_cases(path, "./pkg/cache", {"cache required"})
            with self.assertRaises(ValueError):
                require_complete(actual, [case["id"] for case in selected])

    def test_profile_exclusions_validate_binding_and_core_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "tools/ci/core_test_profiles.json"
            profile_path.parent.mkdir(parents=True)
            records = []
            for package, source in (
                ("./pkg/example", "src/semantic-router/pkg/example/external_test.go"),
                ("candle-binding", "candle-binding/external_test.go"),
            ):
                path = root / source
                path.parent.mkdir(parents=True)
                path.write_text("func TestExternalCheckpoint(t *testing.T) {}\n")
                records.append(
                    {
                        "package": package,
                        "test": "TestExternalCheckpoint",
                        "source": source,
                        "profile": "external-checkpoint",
                        "reason": "Requires a separately supplied compatibility checkpoint.",
                    }
                )
            profile_path.write_text(json.dumps({"excluded": records}))
            with mock.patch("run_core_tests.ROOT", root):
                excluded, _ = profile_exclusions()
                self.assertEqual(set(excluded), {"./pkg/example", "candle-binding"})
                self.assertEqual(excluded["candle-binding"], {"TestExternalCheckpoint"})
                (root / records[1]["source"]).write_text("// test removed\n")
                with self.assertRaisesRegex(ValueError, "stale or unowned"):
                    profile_exclusions()

    def test_owned_inventory_separates_explicit_references_and_rejects_other_skips(
        self,
    ):
        references = {
            "TestPublishedGroundedParity",
            "TestPublishedOmniParity",
            "TestPublishedOmniFullContext",
        }
        excluded, profiles = profile_exclusions()
        self.assertTrue(references <= excluded["onnx-binding/instance"])
        for record in profiles["excluded"]:
            if record["test"] in references:
                self.assertTrue(record["profile"].startswith("explicit-vela-"))

        def invoke(action):
            def run_go(args, _output, _env, *, module):
                names = {"TestOwned"}
                if module.name == "onnx-binding":
                    names |= references
                package = "test/" + module.name
                if "-list" in args:
                    return [
                        {"Package": package, "Output": name + "\n"}
                        for name in sorted(names)
                    ]
                if "-skip" in args:
                    skip = args[args.index("-skip") + 1]
                    names = {name for name in names if not re.fullmatch(skip, name)}
                self.assertEqual(names, {"TestOwned"})
                return [
                    {"Package": package, "Test": name, "Action": action}
                    for name in names
                ]

            with (
                tempfile.TemporaryDirectory() as directory,
                mock.patch("run_core_tests.run_go", side_effect=run_go),
            ):
                return execute_owned(Path(directory), {})

        evidence = invoke("pass")
        require_complete(evidence["cases"], evidence["expected_cases"])
        self.assertEqual(len(evidence["cases"]), 3)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            invoke("skip")

    def test_owned_omni_integration_exclusions_have_the_actual_artifact_lane(self):
        excluded, profiles = profile_exclusions()
        indexed = {(row["package"], row["test"]): row for row in profiles["excluded"]}
        for package, tests in OWNED_OMNI_TESTS.items():
            for name in tests:
                key = "./pkg/" + package
                self.assertIn(name, excluded[key])
                self.assertEqual(indexed[key, name]["profile"], "native.ort-cpu")
        self.assertEqual(
            indexed["./pkg/modelruntime/native", "TestPublishedVelaHalu"]["profile"],
            "native.candle-cpu",
        )
        self.assertEqual(
            indexed["./pkg/modelruntime/native", "TestPublishedOmniModels"]["profile"],
            "native.ort-cpu",
        )

    def test_race_and_ordinary_partitions_execute_each_selected_case_once(self) -> None:
        package = "./pkg/example"
        names = {"TestOwned", "TestOrdinary", "TestStorage", "TestSuite"}
        groups, expected = unit_groups(
            {package: names},
            {package: {"TestStorage"}},
            {package: "TestSuite"},
            {package: {"TestOwned"}},
        )
        measured = []
        for (skipped, race), packages in groups.items():
            self.assertEqual(packages, [package])
            selected = names - set(skipped)
            self.assertEqual(selected, {"TestOwned"} if race else {"TestOrdinary"})
            measured.extend(f"{package}/{name}" for name in selected)
        self.assertCountEqual(measured, expected)
        self.assertEqual(len(measured), len(set(measured)))
        for inventory, exclusions in (
            ({package: names - {"TestOwned"}}, {}),
            ({package: names}, {package: {"TestOwned"}}),
        ):
            with self.assertRaisesRegex(ValueError, "required race tests"):
                unit_groups(inventory, exclusions, {}, {package: {"TestOwned"}})

    def test_race_contracts_require_existing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture_test.go"
            source.write_text("func TestOwned(t *testing.T) {}\n")
            record = {
                "package": "./pkg/example",
                "test": "TestOwned",
                "source": source.name,
                "reason": "required concurrency contract",
            }
            with mock.patch("run_core_tests.ROOT", root):
                self.assertEqual(
                    race_inventory({"race": [record]}), {"./pkg/example": {"TestOwned"}}
                )
                with self.assertRaisesRegex(ValueError, "duplicate race"):
                    race_inventory({"race": [record, record]})
                source.write_text("// removed\n")
                with self.assertRaisesRegex(ValueError, "stale or unowned"):
                    race_inventory({"race": [record]})

    def test_pending_ginkgo_spec_remains_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "SpecReports": [
                                {
                                    "LeafNodeText": "unfinished required contract",
                                    "LeafNodeType": "It",
                                    "State": "pending",
                                }
                            ]
                        }
                    ]
                )
            )
            selected = spec_cases(path, "./pkg/cache")
            self.assertEqual(len(selected), 1)
            with self.assertRaisesRegex(ValueError, "did not pass"):
                require_complete(selected, [case["id"] for case in selected])

    def test_nested_go_skip_cannot_be_masked_by_passing_root(self) -> None:
        events = [
            {"Package": "pkg", "Test": "TestParent", "Action": "pass"},
            {"Package": "pkg", "Test": "TestParent/child", "Action": "skip"},
        ]
        self.assertEqual(len(terminal_cases(events)), 2)
        self.assertEqual(terminal_cases(events)[1]["status"], "skipped")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("\n".join(json.dumps(event) for event in events))
            with self.assertRaisesRegex(ValueError, "subtest"):
                go_cases(path, path, set())

    def test_junit_skip_is_preserved_and_empty_report_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            path.write_text(
                '<testsuite><testcase name="selected"><skipped/></testcase></testsuite>'
            )
            self.assertEqual(junit_cases(path)[0]["status"], "skipped")
            path.write_text("<testsuite/>")
            with self.assertRaisesRegex(ValueError, "empty"):
                junit_cases(path)


if __name__ == "__main__":
    unittest.main()
