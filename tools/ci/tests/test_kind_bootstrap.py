import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_KIND = REPO_ROOT / ".github" / "actions" / "setup-kind" / "action.yml"
WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "integration-test-k8s.yml",
    REPO_ROOT / ".github" / "workflows" / "operator-ci.yml",
)
KIND_VERSION = "v0.31.0"
KIND_NODE_IMAGE = (
    "kindest/node:v1.33.7@sha256:"
    "d26ef333bdb2cbe9862a0f7c3803ecc7b4303d8cea8e814b481b09949d353040"
)


class KindBootstrapContractTests(unittest.TestCase):
    def test_one_pinned_bootstrap_is_used_by_both_workflows(self) -> None:
        setup_text = SETUP_KIND.read_text(encoding="utf-8")

        self.assertEqual(setup_text.count(f"KIND_VERSION: {KIND_VERSION}"), 1)
        self.assertEqual(setup_text.count(f"KIND_NODE_IMAGE: {KIND_NODE_IMAGE}"), 1)
        self.assertIn(
            'echo "KIND_NODE_IMAGE=${KIND_NODE_IMAGE}" >> "$GITHUB_ENV"',
            setup_text,
        )

        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow.name):
                self.assertEqual(text.count("uses: ./.github/actions/setup-kind"), 1)
                self.assertNotIn("v0.22.0", text)
                self.assertNotIn("v1.29.2", text)

        self.assertIn(
            '--image "${KIND_NODE_IMAGE}"',
            WORKFLOWS[1].read_text(encoding="utf-8"),
        )

    def test_e2e_keeps_docker_relocation_opt_in(self) -> None:
        e2e_text = WORKFLOWS[0].read_text(encoding="utf-8")
        operator_text = WORKFLOWS[1].read_text(encoding="utf-8")

        self.assertIn('relocate-docker: "true"', e2e_text)
        self.assertNotIn("relocate-docker:", operator_text)
        self.assertIn("Run Integration E2E tests", e2e_text)
        self.assertIn("Deploy Redis", operator_text)


if __name__ == "__main__":
    unittest.main()
