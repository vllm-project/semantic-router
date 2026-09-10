"""CI configuration tests for the canonical Dashboard gates (issue #2793).

The required Dashboard workflow runs ``make dashboard-check`` plus the dedicated
Evaluation browser target. Before #2793 the fast target gated lint, type-check,
frontend unit tests and go mod tidy, but never ran ``go test`` on
``dashboard/backend`` -- so a backend regression could pass the required gate
unnoticed. These tests assert both canonical entrypoints stay wired.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_MK = REPO_ROOT / "tools" / "make" / "dashboard.mk"
DASHBOARD_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "dashboard-test.yml"

ISSUE = "issue #2793"

# Matches "target: prereqs ## help", but not variable assignments ("VAR := value"),
# comments, or recipe lines (which are tab-indented).
_TARGET_RE = re.compile(r"^(?P<name>[^\t#=\s][^:=]*?):(?!=)(?P<rest>.*)$")
_VAR_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*[:?+]?=\s*(?P<value>.*)$")


class Target:
    """A single Makefile target: its prerequisites, ``##`` help text and recipe."""

    def __init__(self, name: str, prereqs: list[str], help_text: str) -> None:
        self.name = name
        self.prereqs = prereqs
        self.help_text = help_text
        self.recipe: list[str] = []


def _join_continuations(text: str) -> str:
    """Collapse backslash-continued Makefile lines into single logical lines."""
    return re.sub(r"\\\n\s*", " ", text)


def _parse_makefile(path: Path) -> tuple[dict[str, Target], dict[str, str]]:
    """Parse a Makefile into its targets and its variable assignments."""
    targets: dict[str, Target] = {}
    variables: dict[str, str] = {}
    current: Target | None = None

    for line in _join_continuations(path.read_text(encoding="utf-8")).splitlines():
        if line.startswith("\t"):
            if current is not None:
                current.recipe.append(line.strip())
            continue

        if not line.strip() or line.lstrip().startswith("#"):
            continue

        # _VAR_RE only matches when an assignment operator follows the name, so a
        # target line such as "dashboard-check: deps" can never be mistaken for one.
        var_match = _VAR_RE.match(line)
        if var_match:
            variables[var_match.group("name")] = var_match.group("value").strip()
            current = None
            continue

        target_match = _TARGET_RE.match(line)
        if target_match:
            name = target_match.group("name").strip()
            rest = target_match.group("rest")
            prereq_text, _, help_text = rest.partition("##")
            current = Target(name, prereq_text.split(), help_text.strip())
            # ".PHONY" and friends are recorded too; harmless, and keeps parsing simple.
            targets[name] = current
            continue

        current = None

    return targets, variables


def _expand(text: str, variables: dict[str, str]) -> str:
    """Expand ``$(VAR)`` references using the parsed variables (recursively)."""
    for _ in range(10):
        expanded = re.sub(
            r"\$\((?P<name>[A-Za-z_][A-Za-z0-9_]*)\)",
            lambda m: variables.get(m.group("name"), m.group(0)),
            text,
        )
        if expanded == text:
            return expanded
        text = expanded
    return text


TARGETS, VARIABLES = _parse_makefile(DASHBOARD_MK)


class DashboardGateTest(unittest.TestCase):
    """Guard the required Dashboard Make and workflow entrypoints."""

    def test_dashboard_check_requires_the_backend_test_target(self) -> None:
        check = TARGETS.get("dashboard-check")
        self.assertIsNotNone(
            check,
            f"{DASHBOARD_MK.name} must define a 'dashboard-check' target: it is what the "
            f"required Dashboard CI workflow and both agent harness domains invoke "
            f"({ISSUE}).",
        )
        self.assertIn(
            "dashboard-test-backend",
            check.prereqs,
            f"'dashboard-check' must depend on 'dashboard-test-backend' ({ISSUE}). "
            f"Without it the required Dashboard gate never runs 'go test' on "
            f"dashboard/backend, so a backend regression passes CI silently -- exactly "
            f"the gap {ISSUE} was filed to close. Found prerequisites: "
            f"{check.prereqs}.",
        )

    def test_dashboard_check_requires_fresh_evaluation_catalog_mirrors(self) -> None:
        check = TARGETS.get("dashboard-check")
        self.assertIsNotNone(check)
        self.assertIn("dashboard-evaluation-catalog-check", check.prereqs)

        catalog_check = TARGETS.get("dashboard-evaluation-catalog-check")
        self.assertIsNotNone(catalog_check)
        recipe = _expand(" ".join(catalog_check.recipe), VARIABLES)
        self.assertIn("tools/ci/sync_evaluation_catalogs.py --check", recipe)

    def test_dashboard_test_backend_runs_go_test_in_the_backend_directory(self) -> None:
        backend = TARGETS.get("dashboard-test-backend")
        self.assertIsNotNone(
            backend,
            f"{DASHBOARD_MK.name} must define a 'dashboard-test-backend' target; "
            f"'dashboard-check' depends on it to satisfy {ISSUE}.",
        )

        recipe = _expand(" ".join(backend.recipe), VARIABLES)
        self.assertIn(
            "go test",
            recipe,
            f"'dashboard-test-backend' must actually run 'go test' ({ISSUE}); "
            f"a target that no longer runs the tests would keep the gate green while "
            f"testing nothing. Recipe: {recipe!r}.",
        )
        self.assertIn(
            "dashboard/backend",
            recipe,
            f"'dashboard-test-backend' must run 'go test' inside dashboard/backend "
            f"({ISSUE}); the dashboard backend is a separate Go module, so running the "
            f"tests from any other working directory silently skips them. "
            f"Recipe: {recipe!r}.",
        )

    def test_dashboard_check_keeps_its_help_comment(self) -> None:
        check = TARGETS.get("dashboard-check")
        self.assertIsNotNone(
            check,
            f"{DASHBOARD_MK.name} must define a 'dashboard-check' target ({ISSUE}).",
        )
        self.assertTrue(
            check.help_text,
            f"'dashboard-check' must keep its '##' help comment ({ISSUE}, acceptance "
            f"criterion 5): 'make help' renders it, and it is how a contributor learns "
            f"what the gate covers.",
        )
        self.assertIn(
            "backend",
            check.help_text.lower(),
            f"the 'dashboard-check' help comment must mention the backend tests "
            f"({ISSUE}, acceptance criterion 5) so 'make help' describes the gate "
            f"accurately. Found: {check.help_text!r}.",
        )

    def test_evaluation_browser_target_owns_install_and_acceptance(self) -> None:
        browser = TARGETS.get("dashboard-test-e2e-evaluation")
        self.assertIsNotNone(
            browser,
            "dashboard.mk must define the repo-native Evaluation browser gate.",
        )
        recipe = _expand(" ".join(browser.recipe), VARIABLES)
        self.assertIn("playwright install --with-deps chromium", recipe)
        self.assertIn("npm run test:e2e:evaluation", recipe)

    def test_dashboard_workflow_reuses_the_browser_make_target(self) -> None:
        workflow = DASHBOARD_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("run: make dashboard-test-e2e-evaluation", workflow)
        self.assertNotIn("run: npm run test:e2e:evaluation", workflow)
        self.assertNotIn("run: npx playwright install", workflow)


if __name__ == "__main__":
    unittest.main()
