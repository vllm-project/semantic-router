import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

agent_support = importlib.import_module("agent_support")
baseline_contract = importlib.import_module("go_complexity_baseline")
config_contract = importlib.import_module("go_complexity_config")
identity = importlib.import_module("go_complexity_identity")
manifest_contract = importlib.import_module("go_complexity_manifest")
ratchet = importlib.import_module("go_complexity_ratchet")
source_policy = importlib.import_module("go_complexity_source_policy")
go_lint_gate = importlib.import_module("go_lint_gate")
go_lint_support = importlib.import_module("go_lint_support")
module_groups = importlib.import_module("module_file_groups")


def base_config() -> dict:
    return {
        "version": "2",
        "run": {"timeout": "5m"},
        "linters": {
            "default": "none",
            "enable": [
                "cyclop",
                "funlen",
                "gocognit",
                "interfacebloat",
                "nestif",
            ],
            "settings": {
                "cyclop": {"max-complexity": 12},
                "funlen": {"lines": 100, "statements": 40},
                "gocognit": {"min-complexity": 20},
                "interfacebloat": {"max": 5},
                "nestif": {"min-complexity": 4},
            },
            "exclusions": {"generated": "disable"},
        },
        "issues": {
            "max-issues-per-linter": 0,
            "max-same-issues": 0,
            "uniq-by-line": False,
        },
    }


def write_config(root: Path, config: dict | None = None) -> Path:
    path = root / baseline_contract.CONFIG_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config or base_config(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def finding(path: str = "pkg/service.go") -> identity.ComplexityFinding:
    return identity.ComplexityFinding(
        identity=identity.ComplexityIdentity(path, "Run", "cyclop"),
        observed=13,
        limit=12,
        line=3,
        column=1,
        message="fixture",
    )


def write_manifest_and_freeze(
    root: Path,
    findings: list[identity.ComplexityFinding],
) -> tuple[Path, Path]:
    config_path = root / baseline_contract.CONFIG_RELATIVE_PATH
    manifest_path = root / manifest_contract.MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = manifest_contract.render_manifest(
        findings,
        "2.5.0",
        config_path,
        "PL-0032",
        "TD048",
    )
    manifest_path.write_text(manifest_text, encoding="utf-8")
    parsed = manifest_contract.parse_manifest_text(manifest_text, str(manifest_path))
    marker_path = root / baseline_contract.FREEZE_RELATIVE_PATH
    marker_path.write_text(
        baseline_contract.render_freeze_marker(
            manifest_text.encode(), config_path.read_bytes(), parsed
        ),
        encoding="utf-8",
    )
    return manifest_path, marker_path


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


def init_repo(root: Path, branch: str = "feature") -> None:
    git(root, "init", "-b", branch)
    git(root, "config", "user.email", "ratchet@example.invalid")
    git(root, "config", "user.name", "Ratchet Test")


def commit_all(root: Path, message: str) -> str:
    git(root, "add", ".")
    git(root, "commit", "-m", message)
    return git(root, "rev-parse", "HEAD")


class GoComplexityConfigContractTests(unittest.TestCase):
    def parse(self, config: dict) -> config_contract.ComplexityConfigContract:
        return config_contract.parse_complexity_config(
            yaml.safe_dump(config).encode(), "fixture"
        )

    def test_disabling_linter_loosening_threshold_and_exclusion_fail(self) -> None:
        baseline = self.parse(base_config())
        disabled = base_config()
        disabled["linters"]["enable"].remove("nestif")
        loose = base_config()
        loose["linters"]["settings"]["cyclop"]["max-complexity"] = 13
        excluded = base_config()
        excluded["linters"]["exclusions"]["rules"] = [
            {"linters": ["cyclop"], "path": "pkg/.*"}
        ]

        self.assertTrue(
            config_contract.validate_not_looser(self.parse(disabled), baseline)
        )
        self.assertTrue(
            config_contract.validate_not_looser(self.parse(loose), baseline)
        )
        self.assertTrue(
            config_contract.validate_not_looser(self.parse(excluded), baseline)
        )

    def test_analysis_coverage_cannot_be_reduced(self) -> None:
        baseline = self.parse(base_config())
        current = base_config()
        current["run"]["tests"] = False

        errors = config_contract.validate_not_looser(self.parse(current), baseline)

        self.assertIn("analysis coverage", " ".join(errors))

    def test_stricter_threshold_is_allowed(self) -> None:
        baseline = self.parse(base_config())
        current = base_config()
        current["linters"]["settings"]["cyclop"]["max-complexity"] = 11

        self.assertEqual(
            config_contract.validate_not_looser(self.parse(current), baseline), []
        )


class GoComplexityFrozenBaselineTests(unittest.TestCase):
    def test_bootstrap_cannot_freeze_a_looser_target_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root)
            write_config(root)
            commit_all(root, "target config")
            git(root, "branch", "target")
            loose = base_config()
            loose["linters"]["settings"]["cyclop"]["max-complexity"] = 13
            config_path = write_config(root, loose)
            manifest_path, _ = write_manifest_and_freeze(root, [])
            commit_all(root, "attempt weak bootstrap")

            result = ratchet.run_complexity_ratchet(
                records=[],
                repo_root=root,
                config_path=config_path,
                manifest_path=manifest_path,
                changed_paths=set(),
                base_ref="target",
                tool_version="2.5.0",
            )

            self.assertFalse(result.passed)
            self.assertIn("threshold was loosened", " ".join(result.contract_errors))

    def test_first_committed_freeze_survives_later_pushes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root)
            write_config(root)
            commit_all(root, "base config")
            git(root, "branch", "target")
            write_manifest_and_freeze(root, [])
            frozen_commit = commit_all(root, "freeze complexity baseline")
            manifest_path = root / manifest_contract.MANIFEST_RELATIVE_PATH
            manifest_path.write_text(
                manifest_contract.render_manifest(
                    [finding()],
                    "2.5.0",
                    root / baseline_contract.CONFIG_RELATIVE_PATH,
                    "PL-0032",
                    "TD048",
                ),
                encoding="utf-8",
            )
            commit_all(root, "later push attempts to add debt")

            frozen = baseline_contract.resolve_frozen_baseline(root, "target")
            current = manifest_contract.load_manifest(manifest_path)
            errors, bootstrap = manifest_contract.validate_manifest_delta(
                current, frozen.manifest, {"pkg/service.go"}
            )

            self.assertEqual(frozen.commit, frozen_commit)
            self.assertTrue(frozen.bootstrap)
            self.assertTrue(bootstrap is False)
            self.assertEqual(len(errors), 1)
            self.assertIn("addition is frozen", errors[0])

    def test_freeze_marker_cannot_change_after_introduction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root)
            write_config(root)
            commit_all(root, "base config")
            git(root, "branch", "target")
            _, marker_path = write_manifest_and_freeze(root, [])
            commit_all(root, "freeze complexity baseline")
            marker_path.write_text("schema_version: 1\n", encoding="utf-8")

            with self.assertRaises(manifest_contract.ComplexityRatchetError):
                baseline_contract.resolve_frozen_baseline(root, "target")


class GoComplexityBypassTests(unittest.TestCase):
    def test_new_nolint_and_build_constraint_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root)
            source = root / "pkg/service.go"
            source.parent.mkdir(parents=True)
            source.write_text("package pkg\n\nfunc Run() {}\n", encoding="utf-8")
            commit_all(root, "base source")
            git(root, "branch", "target")
            source.write_text(
                "//go:build linux\n\npackage pkg\n\n//nolint:cyclop\nfunc Run() {}\n",
                encoding="utf-8",
            )

            errors = source_policy.validate_changed_source_policy(
                root, "target", {"pkg/service.go"}, {"pkg/service.go"}
            )

            self.assertEqual(len(errors), 2)
            self.assertIn("nolint", errors[0])
            self.assertIn("build constraint", errors[1])

    def test_new_file_cannot_use_build_constraint_to_escape_lint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root)
            (root / "go.mod").write_text(
                "module example.invalid/buildtag\n", encoding="utf-8"
            )
            (root / "base.go").write_text("package buildtag\n", encoding="utf-8")
            commit_all(root, "base source")
            git(root, "branch", "target")
            hidden = root / "hidden.go"
            hidden.write_text(
                "//go:build never\n\npackage buildtag\n", encoding="utf-8"
            )

            errors = source_policy.validate_changed_source_policy(
                root, "target", {"hidden.go"}
            )

            self.assertEqual(len(errors), 1)
            self.assertIn("was not loaded by lint", errors[0])

    def test_existing_inactive_file_requires_an_explicit_lint_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root)
            source = root / "pkg/service_windows.go"
            source.parent.mkdir(parents=True)
            source.write_text(
                "//go:build windows\n\npackage pkg\n\nfunc Run() {}\n",
                encoding="utf-8",
            )
            commit_all(root, "base windows source")
            git(root, "branch", "target")
            source.write_text(
                "//go:build windows\n\npackage pkg\n\nfunc Run() { if true {} }\n",
                encoding="utf-8",
            )

            context = source_policy.build_context_for_source(
                source.read_bytes(), source
            )
            uncovered = source_policy.validate_changed_source_policy(
                root, "target", {"pkg/service_windows.go"}
            )
            covered = source_policy.validate_changed_source_policy(
                root,
                "target",
                {"pkg/service_windows.go"},
                {"pkg/service_windows.go"},
            )

            self.assertIsNotNone(context)
            self.assertEqual(context.goos, "windows")
            self.assertEqual(len(uncovered), 1)
            self.assertEqual(covered, [])

    def test_nolint_all_and_generated_marker_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root)
            source = root / "pkg/service.go"
            source.parent.mkdir(parents=True)
            source.write_text("package pkg\n\nfunc Run() {}\n", encoding="utf-8")
            commit_all(root, "base source")
            git(root, "branch", "target")
            source.write_text(
                "// Code generated by fixture. DO NOT EDIT.\n"
                "package pkg\n\n//nolint:all\nfunc Run() {}\n",
                encoding="utf-8",
            )

            errors = source_policy.validate_changed_source_policy(
                root, "target", {"pkg/service.go"}, {"pkg/service.go"}
            )

            self.assertEqual(len(errors), 2)
            self.assertIn("nolint", errors[0])
            self.assertIn("generated-code marker", errors[1])

    def test_line_directives_are_rejected_without_string_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root)
            source = root / "pkg/service.go"
            source.parent.mkdir(parents=True)
            source.write_text("package pkg\n\nfunc Run() {}\n", encoding="utf-8")
            commit_all(root, "base source")
            git(root, "branch", "target")
            source.write_text(
                "package pkg\n\n"
                'const text = "/*line literal.go:1*/"\n'
                "//line unchanged.go:1\n"
                "func Run() {}\n"
                "/*line other.go:2*/ func Other() {}\n",
                encoding="utf-8",
            )

            errors = source_policy.validate_changed_source_policy(
                root, "target", {"pkg/service.go"}, {"pkg/service.go"}
            )

            self.assertEqual(len(errors), 1)
            self.assertIn("line directive", errors[0])

    def test_platform_test_suffix_resolves_an_actual_alternate_context(self) -> None:
        source = b"package pkg\n\nfunc Run() {}\n"
        context = source_policy.build_context_for_source(
            source, Path("service_windows_amd64_test.go")
        )

        self.assertIsNotNone(context)
        self.assertEqual((context.goos, context.goarch), ("windows", "amd64"))

    def test_alternate_context_proves_the_exact_file_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "go.mod").write_text(
                "module example.invalid/context\n", encoding="utf-8"
            )
            (root / "base.go").write_text("package context\n", encoding="utf-8")
            source = root / "service_windows_amd64_test.go"
            source.write_text("package context\n", encoding="utf-8")
            context = source_policy.BuildContext("windows", "amd64", False, ())

            covered, missing, errors = go_lint_gate._partition_context_files(
                [source], context
            )

            self.assertEqual(covered, [source])
            self.assertEqual(missing, [])
            self.assertEqual(errors, [])

    def test_ignored_filename_and_excess_custom_tags_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "permanently ignores"):
            source_policy.build_context_for_source(b"package pkg\n", Path("_hidden.go"))
        tags = " || ".join(f"custom{index}" for index in range(9))
        with self.assertRaisesRegex(ValueError, "too many custom tags"):
            source_policy.build_context_for_source(
                f"//go:build {tags}\n\npackage pkg\n".encode(),
                Path("hidden.go"),
            )


class GoComplexityLifecycleTests(unittest.TestCase):
    def test_existing_go_file_without_module_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "orphan.go"
            source.write_text("package orphan\n", encoding="utf-8")

            unresolved = go_lint_gate.unresolved_go_paths(root, {"orphan.go"}, {})
            existing = go_lint_gate.existing_unresolved_go_paths(root, unresolved)

            self.assertEqual(existing, {"orphan.go"})

    def test_gate_rejects_an_existing_go_file_without_a_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "orphan.go"
            source.write_text("package orphan\n", encoding="utf-8")
            config = root / "config.yml"
            config.write_text("version: '2'\n", encoding="utf-8")
            manifest = root / "debt.yml"
            with (
                mock.patch.object(
                    go_lint_gate, "resolve_golangci_lint", return_value="golangci-lint"
                ),
                mock.patch.object(go_lint_gate, "_run_ratchet", return_value=True),
            ):
                result = go_lint_gate.run_go_lint_gate(
                    ["orphan.go"], None, root, config, manifest, {}
                )

            self.assertEqual(result, 1)

    def test_complexity_scope_includes_module_override_in_mixed_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = root / "router"
            dashboard = root / "dashboard"
            router.mkdir()
            dashboard.mkdir()
            router_source = router / "service.go"
            dashboard_source = dashboard / "handler.go"
            router_source.write_text("package router\n", encoding="utf-8")
            dashboard_source.write_text("package dashboard\n", encoding="utf-8")
            grouped = {
                router: [router_source],
                dashboard: [dashboard_source],
            }
            scoped = go_lint_gate._complexity_changed_paths(
                grouped,
                set(),
                root,
            )

            self.assertEqual(
                scoped,
                {"dashboard/handler.go", "router/service.go"},
            )

    def test_module_override_cannot_shrink_full_complexity_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            router = root / "router"
            dashboard = root / "dashboard"
            for module in (router, dashboard):
                module.mkdir()
                (module / "go.mod").write_text(
                    f"module example.invalid/{module.name}\n",
                    encoding="utf-8",
                )
                (module / "service.go").write_text(
                    f"package {module.name}\n",
                    encoding="utf-8",
                )
            grouped = {router: [router / "service.go"]}

            with mock.patch.object(
                go_lint_gate,
                "_tracked_go_files",
                return_value=["router/service.go", "dashboard/service.go"],
            ):
                expanded, unowned = go_lint_gate._expanded_lint_groups(
                    root, grouped, True
                )

            self.assertEqual(set(expanded), {router, dashboard})
            self.assertEqual(unowned, set())

    def test_contract_scan_rejects_tracked_source_after_go_mod_removal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "module/service.go"
            source.parent.mkdir()
            source.write_text("package module\n", encoding="utf-8")

            with mock.patch.object(
                go_lint_gate,
                "_tracked_go_files",
                return_value=["module/service.go"],
            ):
                expanded, unowned = go_lint_gate._expanded_lint_groups(root, {}, True)

            self.assertEqual(expanded, {})
            self.assertEqual(unowned, {"module/service.go"})

    def test_golangci_warning_is_a_tool_error(self) -> None:
        payload = {
            "Report": {"Warnings": [{"Tag": "loader", "Text": "analysis incomplete"}]}
        }

        self.assertEqual(
            go_lint_gate._golangci_tool_errors(payload, ""),
            ["analysis incomplete"],
        )

    def test_golangci_issue_without_filename_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no filename"):
            go_lint_support.filter_go_issues(
                Path("/repo"),
                Path("/repo/module"),
                [{"Pos": {}, "FromLinter": "cyclop"}],
                {"module/service.go"},
            )

    def test_golangci_payload_requires_an_issue_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "no issue list"):
            go_lint_support.go_issues_from_payload({"Report": {}})

    def test_go_commands_force_module_mode(self) -> None:
        with mock.patch.dict(os.environ, {"GO111MODULE": "off"}):
            environment = go_lint_gate._context_environment(None)

        self.assertEqual(environment["GO111MODULE"], "on")

    def test_contract_paths_share_the_go_tool_bootstrap_predicate(self) -> None:
        root = agent_support.REPO_ROOT
        paths = [
            "tools/agent/requirements.txt",
            "tools/agent/scripts/agent_changed_files.py",
            "tools/agent/scripts/go_complexity_identity.py",
            "tools/agent/scripts/go_lint_support.py",
            "tools/make/agent.mk",
            "tools/linter/go/.golangci.agent.yml",
            "src/semantic-router/go.mod",
            "go.work",
        ]

        self.assertTrue(go_lint_gate.go_lint_tool_required(paths, root))
        self.assertTrue(
            all(go_lint_gate.is_complexity_contract_path(path, root) for path in paths)
        )
        self.assertFalse(
            go_lint_gate.go_lint_tool_required(["website/docs/index.md"], root)
        )

    def test_looser_config_cannot_be_hidden_by_synced_manifest_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root, "target")
            config_path = write_config(root)
            source = root / "pkg/service.go"
            source.parent.mkdir(parents=True)
            source.write_text("package pkg\n\nfunc Run() {}\n", encoding="utf-8")
            manifest_path, _ = write_manifest_and_freeze(root, [finding()])
            commit_all(root, "target complexity baseline")
            git(root, "checkout", "-b", "feature")
            loose = base_config()
            loose["linters"]["settings"]["cyclop"]["max-complexity"] = 14
            write_config(root, loose)
            source.write_text(
                "package pkg\n\nfunc Run() {}\n// changed with config\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                manifest_contract.render_manifest(
                    [], "2.5.0", config_path, "PL-0032", "TD048"
                ),
                encoding="utf-8",
            )

            result = ratchet.run_complexity_ratchet(
                records=[],
                repo_root=root,
                config_path=config_path,
                manifest_path=manifest_path,
                changed_paths={"pkg/service.go"},
                base_ref="target",
                tool_version="2.5.0",
            )

            self.assertFalse(result.passed)
            self.assertIn("threshold was loosened", " ".join(result.contract_errors))

    def test_tool_change_cannot_be_hidden_by_synced_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root, "target")
            config_path = write_config(root)
            source = root / "pkg/service.go"
            source.parent.mkdir(parents=True)
            source.write_text("package pkg\n\nfunc Run() {}\n", encoding="utf-8")
            manifest_path, _ = write_manifest_and_freeze(root, [finding()])
            commit_all(root, "target complexity baseline")
            git(root, "checkout", "-b", "feature")
            manifest_path.write_text(
                manifest_contract.render_manifest(
                    [finding()], "2.6.0", config_path, "PL-0032", "TD048"
                ),
                encoding="utf-8",
            )
            record = {
                "path": "pkg/service.go",
                "line": 3,
                "column": 1,
                "linter": "cyclop",
                "message": (
                    "calculated cyclomatic complexity for function Run is 13, max is 12"
                ),
            }

            result = ratchet.run_complexity_ratchet(
                records=[record],
                repo_root=root,
                config_path=config_path,
                manifest_path=manifest_path,
                changed_paths={"pkg/service.go"},
                base_ref="target",
                tool_version="2.6.0",
            )

            self.assertFalse(result.passed)
            self.assertIn("version differs", " ".join(result.contract_errors))

    def test_whole_module_deletion_still_reports_stale_debt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            init_repo(root, "target")
            config_path = write_config(root)
            module = root / "module"
            source = module / "pkg/old.go"
            source.parent.mkdir(parents=True)
            (module / "go.mod").write_text(
                "module example.invalid/old\n", encoding="utf-8"
            )
            source.write_text("package pkg\n\nfunc Run() {}\n", encoding="utf-8")
            manifest_path, _ = write_manifest_and_freeze(
                root, [finding("module/pkg/old.go")]
            )
            commit_all(root, "target complexity baseline")
            git(root, "checkout", "-b", "feature")
            source.unlink()
            (module / "go.mod").unlink()
            changed = {"module/pkg/old.go"}

            grouped = module_groups.group_files_by_module(
                root, list(changed), "go.mod", {".go"}
            )
            unresolved = go_lint_gate.unresolved_go_paths(root, changed, grouped)
            existing = go_lint_gate.existing_unresolved_go_paths(root, unresolved)
            result = ratchet.run_complexity_ratchet(
                records=[],
                repo_root=root,
                config_path=config_path,
                manifest_path=manifest_path,
                changed_paths=changed,
                base_ref="target",
                tool_version="2.5.0",
            )

            self.assertEqual(grouped, {})
            self.assertEqual(unresolved, changed)
            self.assertEqual(existing, set())
            self.assertFalse(result.bootstrap)
            self.assertEqual(len(result.stale), 1)
            self.assertFalse(result.passed)

    def test_manifest_reduction_requires_changed_source(self) -> None:
        entry = manifest_contract.DebtEntry(
            finding().identity, 14, 12, "PL-0032", "TD048"
        )
        lowered = replace(entry, observed=13)
        baseline = manifest_contract.DebtManifest("2.5.0", "sha256:fixture", (entry,))
        current = replace(baseline, entries=(lowered,))

        errors, _ = manifest_contract.validate_manifest_delta(current, baseline, set())

        self.assertEqual(len(errors), 1)
        self.assertIn("reduction requires a changed source", errors[0])

    def test_ordinary_issue_path_is_repository_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            module = root / "module"
            source = module / "pkg/service.go"
            source.parent.mkdir(parents=True)
            source.write_text("package pkg\n", encoding="utf-8")
            issue = {"Pos": {"Filename": str(source), "Line": 1, "Column": 1}}

            normalized = go_lint_support.repo_relative_go_issue(root, module, issue)

            self.assertEqual(normalized["Pos"]["Filename"], "module/pkg/service.go")

    def test_canonical_manifest_and_marker_match_active_contract(self) -> None:
        manifest = manifest_contract.load_manifest(
            agent_support.GO_COMPLEXITY_DEBT_MANIFEST
        )
        errors = manifest_contract.validate_manifest_contract(
            manifest, agent_support.GO_AGENT_CONFIG, tool_version="2.5.0"
        )
        marker = baseline_contract.parse_freeze_marker(
            (
                agent_support.REPO_ROOT / baseline_contract.FREEZE_RELATIVE_PATH
            ).read_bytes(),
            "canonical freeze marker",
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(manifest.entries), 2112)
        self.assertEqual(marker.tool_version, manifest.tool_version)
        self.assertTrue(all(entry.owner == "PL-0032" for entry in manifest.entries))
        self.assertTrue(all(entry.debt == "TD048" for entry in manifest.entries))
        manifest_paths = {entry.identity.path for entry in manifest.entries}
        self.assertTrue(
            {
                "dashboard/backend/handlers/log_spool_reader_unix.go",
                "onnx-binding/fa_test.go",
                "src/semantic-router/internal/nlgen/generate_live_test.go",
            }
            <= manifest_paths
        )

    def test_tracked_go_source_has_no_complexity_nolint(self) -> None:
        offenders = []
        for relative in go_lint_gate._tracked_go_files(agent_support.REPO_ROOT):
            source = (agent_support.REPO_ROOT / relative).read_bytes()
            if source_policy._complexity_nolint_sites(source):
                offenders.append(relative)

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
