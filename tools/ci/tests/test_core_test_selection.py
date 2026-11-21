"""Selection contracts use fake go list output; never compile Router packages."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core_test_selection as selection
import run_core_tests as runner

ROOT = Path("/repo")
PREFIX = "example/router"


def package(name, **fields):
    return {
        "ImportPath": f"{PREFIX}/{name}",
        "Dir": str(ROOT / selection.MODULE_PATH / name),
        **fields,
    }


def make_graph():
    return [
        package("pkg/a"),
        package("pkg/b", Imports=[f"{PREFIX}/pkg/a"]),
        package("pkg/c", XTestImports=[f"{PREFIX}/pkg/b"]),
        package("pkg/d", TestImports=[f"{PREFIX}/pkg/a"]),
        package("pkg/unrelated"),
    ]


def _write_repo(tmp_path, sources):
    for relative, text in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for existing in ("website/docs/a.md", "website/docs/b.md", "deploy/x.yaml"):
        (tmp_path / existing).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / existing).write_text("")
    return selection.repo_reads(tmp_path)


class CoreTestSelectionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_concatenated_go_list_and_reverse_test_dependencies(self):
        graph = make_graph()
        decoded = selection.decode_graph("\n".join(json.dumps(p) for p in graph))
        result = selection.select_packages(
            ROOT, decoded, ["src/semantic-router/pkg/a/a_test.go"]
        )
        assert result["selected_packages"] == [f"{PREFIX}/pkg/{p}" for p in "abcd"]
        assert not result["fallback_reasons"]

    def test_unknown_inputs_and_shared_helpers_select_all(self):
        for path in [
            "src/semantic-router/go.mod",
            "src/semantic-router/go.sum",
            "src/semantic-router/internal/testutil/storagetest/a.go",
            "src/semantic-router/pkg/deleted/a.go",
            "../escape",
            "/absolute",
        ]:
            graph = make_graph()
            with self.subTest(path=path):
                result = selection.select_packages(ROOT, graph, [path])
                assert result["selected_packages"] == result["all_packages"]
                assert result["fallback_reasons"]

    def test_testdata_nested_deleted_files_and_rename_sides(self):
        graph = make_graph()
        result = selection.select_packages(
            ROOT,
            graph,
            [
                "src/semantic-router/pkg/b/testdata/nested/deleted.json",
                "src/semantic-router/pkg/unrelated/renamed_test.go",
            ],
        )
        # Fixtures are read by their own package's tests; importers do not
        # re-read them, so only pkg/b runs for its testdata.
        assert result["selected_packages"] == [
            f"{PREFIX}/pkg/{p}" for p in ("b", "unrelated")
        ]

    def test_files_outside_the_module_that_no_router_test_reads_select_nothing(self):
        graph = make_graph()
        result = selection.select_packages(
            ROOT,
            graph,
            ["dashboard/backend/go.sum", "tools/agent/docs/x.md", "unknown/file"],
        )
        assert result["selected_packages"] == []
        assert not result["fallback_reasons"]

    def test_read_inputs_select_their_readers_but_embedded_files_spread(self):
        graph = make_graph()
        result = selection.select_packages(
            ROOT, graph, ["src/semantic-router/pkg/a/README.md"]
        )
        assert result["selected_packages"] == [f"{PREFIX}/pkg/a"]
        graph[0]["EmbedFiles"] = ["schema.json"]
        result = selection.select_packages(
            ROOT, graph, ["src/semantic-router/pkg/a/schema.json"]
        )
        assert result["selected_packages"] == [f"{PREFIX}/pkg/{p}" for p in "abcd"]

    def test_explicit_external_inputs_and_source_scanners(self):
        graph = make_graph()
        owners = {owner for targets in selection.INPUTS.values() for owner in targets}
        graph += [package(p) for p in owners | {"pkg/headers"}]
        for path in (
            "config/recipes/a/config.yaml",
            "website/docs/installation/storage-overview.md",
            "website/static/model-catalog/catalog.json",
        ):
            result = selection.select_packages(ROOT, graph, [path])
            assert not result["fallback_reasons"]
            assert f"{PREFIX}/pkg/config" in result["selected_packages"]
        result = selection.select_packages(
            ROOT, graph, ["src/semantic-router/pkg/a/a.go"]
        )
        assert {f"{PREFIX}/pkg/headers", f"{PREFIX}/pkg/extproc"} <= set(
            result["selected_packages"]
        )

    def test_missing_owner_and_graph_errors_fail_open(self):
        graph = make_graph()
        assert selection.select_packages(ROOT, graph, ["config/config.yaml"])[
            "fallback_reasons"
        ]
        with mock.patch.object(
            selection, "load_graph", side_effect=ValueError("broken graph")
        ):
            assert selection.plan_selection(ROOT, {})["selected_packages"] is None
        with mock.patch.object(selection, "load_graph", return_value=graph):
            result = selection.plan_selection(ROOT, {})
            assert result["selected_packages"] == result["all_packages"]
        for raw in ("", '{"Error": {"Err": "missing"}}', '{"DepsErrors": [{}]}'):
            with self.assertRaises(ValueError):
                selection.decode_graph(raw)

    def test_change_inputs_and_no_rename_collapsing(self):
        assert selection.changed_files(
            ROOT, {"CORE_TEST_CHANGED_FILES": '["a", "b"]'}
        ) == [
            "a",
            "b",
        ]
        with self.assertRaises(ValueError):
            selection.changed_files(ROOT, {"CORE_TEST_CHANGED_FILES": "{}"})
        with mock.patch.object(
            selection.subprocess, "check_output", return_value="old\0new\0"
        ) as go:
            assert selection.changed_files(ROOT, {"CORE_TEST_BASE": "base"}) == [
                "old",
                "new",
            ]
            assert "--no-renames" in go.call_args.args[0]

    def test_pull_request_diffs_the_merge_commit_against_its_base(self):
        event = self.tmp_path / "event.json"
        event.write_text(json.dumps({"pull_request": {"base": {"sha": "stale"}}}))
        with mock.patch.object(
            selection.subprocess, "check_output", return_value="a\0"
        ) as git:
            selection.changed_files(ROOT, {"GITHUB_EVENT_PATH": str(event)})
        assert git.call_args.args[0][-2:] == ["HEAD^1", "HEAD"]

    def test_hook_filters_discovery_race_and_ginkgo_only_when_enforced(self):
        for enforce in [False, True]:
            with self.subTest(enforce=enforce):
                selected = runner.MODULE_NAME + "/pkg/a"
                plan = {
                    "selected_packages": [selected],
                    "all_packages": [selected, "other"],
                }
                listed = [{"Package": selected, "Output": "TestA\n"}]
                with (
                    mock.patch.object(runner, "plan_selection", return_value=plan),
                    mock.patch.object(
                        runner,
                        "unit_exclusions",
                        return_value=({}, {"ginkgo_excluded_labels": []}),
                    ),
                    mock.patch.object(
                        runner,
                        "ginkgo_suites",
                        return_value={"./pkg/other": "TestSuite"} if enforce else {},
                    ),
                    mock.patch.object(
                        runner,
                        "race_inventory",
                        return_value={"./pkg/other": {"TestOther"}} if enforce else {},
                    ),
                    mock.patch.object(
                        runner, "execute_tool_units", return_value=([], [])
                    ),
                    mock.patch.object(
                        runner,
                        "run_go",
                        side_effect=[
                            listed,
                            [{"Package": selected, "Test": "TestA", "Action": "pass"}],
                        ],
                    ) as go,
                ):
                    runner.execute_unit(
                        self.tmp_path,
                        {"CORE_TEST_SELECTION_ENFORCE": "1"} if enforce else {},
                    )
                assert go.call_args_list[0].args[0][-1] == (
                    selected if enforce else "./..."
                )
                record = json.loads((self.tmp_path / "selection.json").read_text())
                assert record["completed"]
                assert record["mode"] == ("enforced" if enforce else "shadow")

    def test_shadow_failure_outside_selection_is_written_before_reraise(self):
        summary = self.tmp_path / "summary.md"
        plan = {
            "selected_packages": ["selected"],
            "all_packages": ["selected", "missed"],
        }

        def fail(output, env, selection):
            (output / "unit-0.jsonl").write_text(
                json.dumps({"Package": "missed", "Action": "fail"}) + "\n"
            )
            raise ValueError("Go invocation failed")

        with (
            mock.patch.object(runner, "plan_selection", return_value=plan),
            mock.patch.object(runner, "execute_selected_unit", side_effect=fail),
            self.assertRaises(ValueError),
        ):
            runner.execute_unit(self.tmp_path, {"GITHUB_STEP_SUMMARY": str(summary)})
        record = json.loads((self.tmp_path / "selection.json").read_text())
        assert summary.read_text() == (
            "Core test selection (shadow): 1 of 2 packages selected. "
            "Fallback: none. Failures outside the selection: missed.\n"
        )
        assert record["failure_outside_selection"]
        assert record["failing_packages_outside_selection"] == ["missed"]
        assert not record["completed"]
        assert not record["comparison_complete"]

    def test_empty_enforced_selection_does_not_invoke_implicit_current_package(self):
        plan = {"selected_packages": [], "mode": "enforced"}
        with (
            mock.patch.object(
                runner,
                "unit_exclusions",
                return_value=({}, {"ginkgo_excluded_labels": []}),
            ),
            mock.patch.object(runner, "ginkgo_suites", return_value={}),
            mock.patch.object(runner, "race_inventory", return_value={}),
            mock.patch.object(runner, "execute_tool_units", return_value=([], [])),
            mock.patch.object(runner, "run_go") as go,
        ):
            runner.execute_selected_unit(self.tmp_path, {}, plan)
        go.assert_not_called()

    def test_cross_package_testdata_consumers_without_import_edges(self):
        graph = [
            package(p)
            for p in ("pkg/protocolcodec", "pkg/extproc", "internal/gatewayparity")
        ]
        result = selection.select_packages(
            ROOT,
            graph,
            ["src/semantic-router/pkg/protocolcodec/testdata/clients/request.json"],
        )
        assert result["selected_packages"] == [
            f"{PREFIX}/pkg/{p}" for p in ("extproc", "protocolcodec")
        ]
        assert not result["fallback_reasons"]

    def test_go_list_load_is_readonly_and_does_not_compile(self):
        graph = make_graph()
        raw = "\n".join(json.dumps(p) for p in graph)
        with mock.patch.object(
            selection.subprocess, "check_output", return_value=raw
        ) as go:
            assert selection.load_graph(ROOT, {}) == graph
        assert go.call_args.args[0] == [
            "go",
            "list",
            "-mod=readonly",
            "-buildvcs=false",
            "-json",
            "./...",
        ]

    def test_failed_run_ignores_stale_logs_and_reports_no_comparison(self):
        (self.tmp_path / "unit-old.jsonl").write_text(
            json.dumps({"Package": "stale", "Action": "fail"})
        )
        plan = {"selected_packages": [], "all_packages": []}
        with (
            mock.patch.object(runner, "plan_selection", return_value=plan),
            mock.patch.object(
                runner,
                "execute_selected_unit",
                side_effect=ValueError("discovery failed"),
            ),
            self.assertRaises(ValueError),
        ):
            runner.execute_unit(self.tmp_path, {})
        record = json.loads((self.tmp_path / "selection.json").read_text())
        assert not record["failure_outside_selection"]
        assert not record["comparison_complete"]
        assert record["failing_packages"] == []

    def test_build_failure_import_path_is_observed(self):
        log = self.tmp_path / "inventory.jsonl"
        log.write_text(json.dumps({"ImportPath": "missed", "Action": "build-fail"}))
        selection.write_observation(
            self.tmp_path, {"selected_packages": ["selected"]}, [log], completed=False
        )
        assert json.loads((self.tmp_path / "selection.json").read_text())[
            "failure_outside_selection"
        ]

    def test_router_reads_are_mapped(self):
        """Every repository path Router code reads outside its own package must
        select the packages that read it, or a change there would go untested."""
        repo = Path(__file__).resolve().parents[3]
        top = {p.name for p in repo.iterdir() if p.is_dir()}
        module = repo / selection.MODULE_PATH
        packages = {
            p.relative_to(module).as_posix()
            for p in module.rglob("*")
            if p.is_dir() and any(p.glob("*.go"))
        }
        root_readers = set(selection.REPO_ROOT_READERS)
        scanners = set(selection.SOURCE_SCANNERS)

        def owners(target):
            return {
                owner
                for prefix, targets in selection.INPUTS.items()
                if (target + "/").startswith(prefix)
                or target == prefix
                or (prefix.endswith("/") and target == prefix[:-1])
                for owner in targets
            }

        unmapped = []
        for target, readers in selection.repo_reads(repo).items():
            parts = target.split("/")
            if target == ".":
                missing = readers - root_readers
            elif target == selection.MODULE_PATH:
                missing = readers - scanners
            elif target.startswith(selection.MODULE_PATH + "/"):
                inner = target[len(selection.MODULE_PATH) + 1 :]
                home = inner
                while home not in packages and "/" in home:
                    home = home.rsplit("/", 1)[0]
                missing = readers - {home} - owners(target)
            elif parts[0] not in top:
                continue  # path-traversal fixtures such as ../../etc/passwd
            else:
                missing = readers - root_readers - owners(target)
            if missing:
                unmapped.append(f"{target}: {sorted(missing)}")
        assert not unmapped, "map these reads in INPUTS:\n" + "\n".join(unmapped)

    def test_repo_reads_follow_root_names_and_literal_loops(self):
        reads = _write_repo(
            self.tmp_path,
            {
                "src/semantic-router/pkg/lit/a_test.go": (
                    "package lit\n"
                    "func repoRoot() string {\n"
                    '\treturn filepath.Join("..", "..", "..", "..")\n'
                    "}\n"
                    "func TestA(t *testing.T) {\n"
                    "\troot := repoRoot()\n"
                    '\tos.ReadFile(filepath.Join(root, "deploy", "x.yaml"))\n'
                    '\tfor _, page := range []string{"website/docs/a.md", filepath.Join("website", "docs", "b.md")} {\n'
                    '\t\tos.ReadFile(filepath.Join("..", "..", "..", "..", page))\n'
                    "\t}\n"
                    "}\n"
                ),
                "src/semantic-router/pkg/dyn/a_test.go": (
                    "package dyn\n"
                    "func TestB(t *testing.T) {\n"
                    '\troot := filepath.Join("..", "..", "..", "..")\n'
                    "\tos.ReadFile(filepath.Join(root, spec.Path))\n"
                    "}\n"
                ),
                "src/semantic-router/pkg/arg/a_test.go": (
                    "package arg\n"
                    "func TestC(t *testing.T) {\n"
                    '\tschema, err := generate("../../../..")\n'
                    "}\n"
                ),
            },
        )
        assert reads["deploy/x.yaml"] == {"pkg/lit"}
        assert reads["website/docs/a.md"] == {"pkg/lit"}
        assert reads["website/docs/b.md"] == {"pkg/lit"}
        # Joined with a variable, or handed to a function, the root reads anything.
        assert reads["."] == {"pkg/dyn", "pkg/arg"}

    def test_runner_and_compiled_inputs(self):
        graph = make_graph()
        for runner_input in (
            "tools/ci/run_core_tests.py",
            "tools/ci/core_test_profiles.json",
        ):
            result = selection.select_packages(ROOT, graph, [runner_input])
            assert result["fallback_reasons"]
        result = selection.select_packages(
            ROOT, graph, ["src/semantic-router/pkg/a/native.c"]
        )
        assert result["selected_packages"] == [f"{PREFIX}/pkg/{p}" for p in "abcd"]

    def test_source_scanner_tests_exist_where_named(self):
        repo = Path(__file__).resolve().parents[3]
        for package, names in selection.SOURCE_SCANNERS.items():
            source = "".join(
                p.read_text()
                for p in (repo / selection.MODULE_PATH / package).glob("*_test.go")
            )
            for name in names:
                assert f"func {name}(t *testing.T)" in source, (package, name)

    def test_production_changes_run_only_the_scanning_tests_of_other_packages(self):
        graph = make_graph()
        graph += [
            package("pkg/headers"),
            package("pkg/extproc", Imports=[f"{PREFIX}/pkg/b"]),
        ]
        result = selection.select_packages(
            ROOT, graph, ["src/semantic-router/pkg/d/d.go"]
        )
        # pkg/d's importers are none; the scanners run their scanning tests only.
        assert result["test_filters"] == {
            f"{PREFIX}/pkg/headers": list(selection.SOURCE_SCANNERS["pkg/headers"]),
            f"{PREFIX}/pkg/extproc": list(selection.SOURCE_SCANNERS["pkg/extproc"]),
        }
        # A change pkg/extproc imports runs all of pkg/extproc.
        result = selection.select_packages(
            ROOT, graph, ["src/semantic-router/pkg/a/a.go"]
        )
        assert f"{PREFIX}/pkg/extproc" not in result["test_filters"]

    def test_unit_groups_run_only_filtered_tests(self):
        groups, expected = runner.unit_groups(
            {"pkg/extproc": {"TestScan", "TestSlow", "TestSuite"}},
            {},
            {"pkg/extproc": "TestSuite"},
            {},
            {"pkg/extproc": {"TestScan"}},
        )
        assert expected == ["pkg/extproc/TestScan"]
        assert groups == {(("TestSlow", "TestSuite"), False): ["pkg/extproc"]}
        with self.assertRaises(ValueError):
            runner.unit_groups(
                {"pkg/x": {"TestA"}}, {}, {}, {}, {"pkg/x": {"TestGone"}}
            )

    def test_a_failure_the_filter_left_out_is_reported(self):
        log = self.tmp_path / "unit.jsonl"
        log.write_text(
            json.dumps(
                {
                    "Action": "fail",
                    "Package": f"{PREFIX}/pkg/extproc",
                    "Test": "TestSlow",
                }
            )
            + "\n"
        )
        selection.write_observation(
            self.tmp_path,
            {
                "selected_packages": [f"{PREFIX}/pkg/extproc"],
                "test_filters": {f"{PREFIX}/pkg/extproc": ["TestScan"]},
            },
            [log],
            completed=True,
        )
        record = json.loads((self.tmp_path / "selection.json").read_text())
        assert record["failing_packages_outside_selection"] == [
            f"{PREFIX}/pkg/extproc/TestSlow"
        ]

    def test_repo_reads_see_raw_strings_and_multiline_joins(self):
        reads = _write_repo(
            self.tmp_path,
            {
                "src/semantic-router/pkg/raw/a_test.go": (
                    "package raw\n"
                    "func TestA(t *testing.T) {\n"
                    "\tos.ReadFile(filepath.Join(`../../../..`, `deploy`, `x.yaml`))\n"
                    "\troot := filepath.Join(\n"
                    '\t\t"..",\n'
                    '\t\t"..", "..", "..",\n'
                    "\t)\n"
                    "\tos.ReadFile(filepath.Join(root, name))\n"
                    "}\n"
                ),
            },
        )
        assert reads["deploy/x.yaml"] == {"pkg/raw"}
        assert reads["."] == {"pkg/raw"}

    def test_embedded_testdata_spreads_to_importers(self):
        graph = make_graph()
        graph[0]["EmbedFiles"] = ["testdata/x.json"]
        result = selection.select_packages(
            ROOT, graph, ["src/semantic-router/pkg/a/testdata/x.json"]
        )
        assert result["selected_packages"] == [f"{PREFIX}/pkg/{p}" for p in "abcd"]

    def test_a_parameter_that_shares_a_loop_name_reads_under_its_prefix(self):
        # pkg/decision reads config/recipes/<name>/config.yaml through a parameter
        # while another test ranges over names that are not recipes.
        reads = _write_repo(
            self.tmp_path,
            {
                "src/semantic-router/pkg/reuse/a_test.go": (
                    "package reuse\n"
                    "func load(name string) {\n"
                    '\tos.ReadFile(filepath.Join("..", "..", "..", "..", "deploy", name))\n'
                    "}\n"
                    "func TestA(t *testing.T) {\n"
                    '\tfor _, name := range []string{"task", "clarify"} {\n'
                    "\t\tt.Log(name)\n"
                    "\t}\n"
                    "}\n"
                ),
            },
        )
        assert reads["deploy"] == {"pkg/reuse"}
