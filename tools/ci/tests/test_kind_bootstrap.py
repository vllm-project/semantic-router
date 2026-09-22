import unittest
from pathlib import Path

import yaml

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
STICKY_TOOL_SELECTION_BASELINE_TESTS = {
    "sticky-tool-selection",
    "sticky-tool-selection-recovery",
}


class KindBootstrapContractTests(unittest.TestCase):
    def test_one_pinned_bootstrap_is_used_by_both_workflows(self) -> None:
        setup_text = SETUP_KIND.read_text(encoding="utf-8")

        self.assertEqual(setup_text.count("uses: ./.github/actions/free-disk-space"), 1)
        self.assertNotIn("uses: $/.github/actions/free-disk-space", setup_text)
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
        bootstrap = yaml.safe_load(SETUP_KIND.read_text(encoding="utf-8"))
        self.assertEqual(bootstrap["inputs"]["relocate-docker"]["default"], "false")
        cleanup = bootstrap["runs"]["steps"][0]
        self.assertEqual(cleanup["uses"], "./.github/actions/free-disk-space")
        self.assertEqual(
            cleanup["with"]["relocate-docker"], "${{ inputs.relocate-docker }}"
        )

        e2e, operator = (
            yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"][
                "integration-test"
            ]["steps"]
            for path in WORKFLOWS
        )
        setup_action = "./.github/actions/setup-kind"
        setup = next(step for step in e2e if step.get("uses") == setup_action)
        self.assertEqual(setup["with"]["relocate-docker"], "true")
        load_images = next(
            step
            for step in e2e
            if step.get("uses") == "./.github/actions/load-ci-images"
        )
        build = next(step for step in e2e if step.get("run") == "make build-e2e")
        execute = next(
            step
            for step in e2e
            if "tools/ci/run_e2e_batch.py --batch" in step.get("run", "")
        )
        # Relocating Docker after importing images would discard the shared handoff.
        self.assertLess(e2e.index(setup), e2e.index(load_images))
        self.assertLess(e2e.index(load_images), e2e.index(build))
        self.assertLess(e2e.index(build), e2e.index(execute))

        operator_setup = next(
            step for step in operator if step.get("uses") == setup_action
        )
        self.assertNotIn("relocate-docker", operator_setup.get("with", {}))
        create = next(
            step for step in operator if "kind create cluster" in step.get("run", "")
        )
        redis = next(
            step
            for step in operator
            if "image: redis/redis-stack-server:" in step.get("run", "")
        )
        self.assertLess(operator.index(operator_setup), operator.index(create))
        self.assertLess(operator.index(create), operator.index(redis))
        self.assertEqual(redis["if"], "matrix.cache-backend == 'redis'")
        matrix = yaml.safe_load(WORKFLOWS[1].read_text())["jobs"]["integration-test"][
            "strategy"
        ]["matrix"]
        self.assertEqual(
            matrix,
            {
                "cache-backend": "${{ fromJSON(inputs.integration_matrix).*.cache-backend }}"
            },
        )

    def test_sticky_tool_selection_contract_runs_in_baseline_ci(self) -> None:
        e2e_text = WORKFLOWS[0].read_text(encoding="utf-8")
        prefix = 'ENVOY_AI_GATEWAY_CI_TESTS="'
        lines = [line.strip() for line in e2e_text.splitlines() if prefix in line]

        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].endswith('"'))
        selected = set(lines[0][len(prefix) : -1].split(","))
        self.assertTrue(STICKY_TOOL_SELECTION_BASELINE_TESTS.issubset(selected))
        self.assertNotIn("sticky-tool-selection-expiry", selected)


if __name__ == "__main__":
    unittest.main()
