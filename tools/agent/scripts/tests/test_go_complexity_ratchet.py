import importlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

agent_support = importlib.import_module("agent_support")
identity = importlib.import_module("go_complexity_identity")
manifest_contract = importlib.import_module("go_complexity_manifest")
ratchet = importlib.import_module("go_complexity_ratchet")
go_lint_gate = importlib.import_module("go_lint_gate")


GO_SOURCE = """package fixture

type Store struct{}

func (s *Store) Handle(enabled bool) {
    if enabled {
        if enabled {
            println("first")
        }
    }
    if enabled {
        if enabled {
            println("second")
        }
    }
}

type Repository interface {
    A()
    B()
    C()
    D()
    E()
    F()
}
"""


def write_config(root: Path) -> Path:
    config = {
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
    path = root / "golangci.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def source_line(source: str, fragment: str, occurrence: int = 1) -> int:
    seen = 0
    for index, line in enumerate(source.splitlines(), start=1):
        if fragment not in line:
            continue
        seen += 1
        if seen == occurrence:
            return index
    raise AssertionError(f"missing source fragment {fragment!r}")


class GoComplexityIdentityTests(unittest.TestCase):
    def test_method_identity_includes_receiver(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = write_config(root)
            normalizer = identity.ComplexityFindingNormalizer(
                root,
                config,
                source_loader=lambda _path: GO_SOURCE.encode(),
            )

            finding = normalizer.normalize(
                {
                    "path": "fixture.go",
                    "line": source_line(GO_SOURCE, "func (s *Store) Handle"),
                    "column": 1,
                    "linter": "cyclop",
                    "message": (
                        "calculated cyclomatic complexity for function Handle "
                        "is 13, max is 12"
                    ),
                }
            )

        self.assertEqual(finding.identity.declaration, "(*Store).Handle")
        self.assertEqual(finding.observed, 13)
        self.assertEqual(finding.limit, 12)

    def test_nestif_identity_uses_stable_ast_hash_and_occurrence(self) -> None:
        shifted_source = "\n\n" + GO_SOURCE
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = write_config(root)

            def normalize(source: str, occurrence: int):
                normalizer = identity.ComplexityFindingNormalizer(
                    root,
                    config,
                    source_loader=lambda _path: source.encode(),
                )
                return normalizer.normalize(
                    {
                        "path": "fixture.go",
                        "line": source_line(source, "if enabled", occurrence),
                        "column": 5,
                        "linter": "nestif",
                        "message": (
                            "`if enabled` has complex nested blocks (complexity: 4)"
                        ),
                    }
                )

            first = normalize(GO_SOURCE, 1)
            shifted = normalize(shifted_source, 1)
            second = normalize(GO_SOURCE, 3)

        self.assertEqual(first.identity.site, shifted.identity.site)
        self.assertTrue(first.identity.site.startswith("sha256:"))
        self.assertNotEqual(first.identity.site, second.identity.site)

    def test_interface_identity_and_limit_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = write_config(root)
            normalizer = identity.ComplexityFindingNormalizer(
                root,
                config,
                source_loader=lambda _path: GO_SOURCE.encode(),
            )
            finding = normalizer.normalize(
                {
                    "path": "fixture.go",
                    "line": source_line(GO_SOURCE, "type Repository interface"),
                    "column": 1,
                    "linter": "interfacebloat",
                    "message": "the interface has more than 5 methods: 6",
                }
            )

        self.assertEqual(finding.identity.declaration, "interface Repository")
        self.assertEqual((finding.observed, finding.limit), (6, 5))

    def test_message_limit_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = write_config(root)
            normalizer = identity.ComplexityFindingNormalizer(
                root,
                config,
                source_loader=lambda _path: GO_SOURCE.encode(),
            )

            with self.assertRaises(identity.ComplexityIdentityError):
                normalizer.normalize(
                    {
                        "path": "fixture.go",
                        "line": source_line(GO_SOURCE, "func (s *Store) Handle"),
                        "column": 1,
                        "linter": "cyclop",
                        "message": (
                            "calculated cyclomatic complexity for function Handle "
                            "is 13, max is 15"
                        ),
                    }
                )


def make_identity(site: str = "") -> identity.ComplexityIdentity:
    return identity.ComplexityIdentity(
        path="pkg/service.go",
        declaration="(*Service).Run",
        linter="nestif" if site else "cyclop",
        site=site,
    )


def make_finding(
    observed: int = 13,
    finding_identity: identity.ComplexityIdentity | None = None,
) -> identity.ComplexityFinding:
    return identity.ComplexityFinding(
        identity=finding_identity or make_identity(),
        observed=observed,
        limit=12,
        line=10,
        column=1,
        message="fixture",
    )


def make_entry(
    observed: int = 13,
    entry_identity: identity.ComplexityIdentity | None = None,
) -> manifest_contract.DebtEntry:
    return manifest_contract.DebtEntry(
        identity=entry_identity or make_identity(),
        observed=observed,
        limit=12,
        owner="PL-0032",
        debt="TD048",
    )


def make_manifest(
    *entries: manifest_contract.DebtEntry,
) -> manifest_contract.DebtManifest:
    return manifest_contract.DebtManifest(
        tool_version="2.5.0",
        config_sha256="sha256:fixture",
        entries=tuple(entries),
        identity_parser_version=manifest_contract.detect_identity_parser_version(),
    )


class GoComplexityRatchetTests(unittest.TestCase):
    def test_known_debt_passes(self) -> None:
        result = ratchet.evaluate_ratchet(
            [make_finding()],
            make_manifest(make_entry()),
            {"pkg/service.go"},
        )

        self.assertTrue(result.passed)
        self.assertEqual(len(result.known), 1)

    def test_new_and_worsened_debt_fail(self) -> None:
        new_result = ratchet.evaluate_ratchet(
            [make_finding()], make_manifest(), {"pkg/service.go"}
        )
        worsened_result = ratchet.evaluate_ratchet(
            [make_finding(observed=14)],
            make_manifest(make_entry()),
            {"pkg/service.go"},
        )

        self.assertFalse(new_result.passed)
        self.assertEqual(len(new_result.new), 1)
        self.assertFalse(worsened_result.passed)
        self.assertEqual(len(worsened_result.worsened), 1)

    def test_improvement_requires_manifest_tightening(self) -> None:
        result = ratchet.evaluate_ratchet(
            [make_finding(observed=13)],
            make_manifest(make_entry(observed=14)),
            {"pkg/service.go"},
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.improved), 1)

    def test_removed_finding_requires_stale_entry_removal(self) -> None:
        result = ratchet.evaluate_ratchet(
            [],
            make_manifest(make_entry()),
            {"pkg/service.go"},
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.stale, [make_entry()])

    def test_unchanged_file_does_not_trigger_stale_check(self) -> None:
        result = ratchet.evaluate_ratchet(
            [],
            make_manifest(make_entry()),
            {"pkg/other.go"},
        )

        self.assertTrue(result.passed)

    def test_full_evaluation_rejects_fabricated_unchanged_allowance(self) -> None:
        result = ratchet.evaluate_ratchet(
            [],
            make_manifest(make_entry()),
            set(),
            evaluated_paths={"pkg/service.go"},
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.stale, [make_entry()])

    def test_full_evaluation_rejects_manifest_path_outside_lint_scope(self) -> None:
        result = ratchet.evaluate_ratchet(
            [],
            make_manifest(make_entry()),
            set(),
            evaluated_paths={"pkg/other.go"},
            require_complete_manifest=True,
        )

        self.assertFalse(result.passed)
        self.assertIn("outside the evaluated scope", result.contract_errors[0])

    def test_baseline_allowance_cannot_be_widened(self) -> None:
        errors, _ = manifest_contract.validate_manifest_delta(
            make_manifest(make_entry(observed=14)),
            make_manifest(make_entry(observed=13)),
            {"pkg/service.go"},
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("allowance widened", errors[0])

    def test_changed_file_lint_does_not_use_line_only_new_issue_filter(self) -> None:
        source = inspect.getsource(go_lint_gate.run_go_lint_gate)
        config = yaml.safe_load(
            agent_support.GO_AGENT_CONFIG.read_text(encoding="utf-8")
        )

        self.assertNotIn("--new-from-rev", source)
        self.assertFalse(config["issues"]["uniq-by-line"])


if __name__ == "__main__":
    unittest.main()
