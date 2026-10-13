"""Exercise the actual refresh workflow scripts with an offline GitHub client."""

import json
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
LEADERBOARD = "contributor-leaderboard.yml"
TRANSLATIONS = "check-translation-staleness.yml"
WORKFLOWS = (LEADERBOARD, TRANSLATIONS)

RUN_SCRIPT = r"""
const fs = require('node:fs');
const {script, existing, failure, operation} = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = {calls: [], failures: [], logs: []};
const call = async (name, params) => {
  result.calls.push({name, params});
  if (name === operation && failure) {
    throw Object.assign(new Error(failure.message), {status: failure.status});
  }
  return {data: name === 'list' ? existing : {number: 123}};
};
const github = {rest: {pulls: {
  list: params => call('list', params),
  create: params => call('create', params),
}}};
const core = {
  setFailed: message => result.failures.push(message),
  info: message => result.logs.push(message),
};
const context = {repo: {owner: 'test-owner', repo: 'test-repo'}};
const logger = {log: (...args) => result.logs.push(args.join(' '))};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
new AsyncFunction('github', 'core', 'context', 'console', script)(
  github, core, context, logger
).then(() => process.stdout.write(JSON.stringify(result))).catch(error => {
  console.error(error);
  process.exitCode = 1;
});
"""


def run_script(workflow, *, existing=None, failure=None, operation="create"):
    document = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
    step = next(
        step
        for job in document["jobs"].values()
        for step in job["steps"]
        if step.get("name") == "Create PR"
    )
    completed = subprocess.run(
        ["node", "-e", RUN_SCRIPT],
        input=json.dumps(
            {
                "script": step["with"]["script"],
                "existing": existing or [],
                "failure": failure,
                "operation": operation,
            }
        ),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(completed.stdout)


class RefreshPullRequestWorkflowTests(unittest.TestCase):
    def test_creates_refresh_pull_request(self):
        for workflow, branch in (
            (LEADERBOARD, "chore/contributor-leaderboard"),
            (TRANSLATIONS, "i18n/mark-outdated"),
        ):
            with self.subTest(workflow=workflow):
                result = run_script(workflow)
                self.assertEqual(result["failures"], [])
                creates = [
                    c["params"] for c in result["calls"] if c["name"] == "create"
                ]
                self.assertEqual(len(creates), 1)
                self.assertEqual(creates[0]["owner"], "test-owner")
                self.assertEqual(creates[0]["repo"], "test-repo")
                self.assertEqual(creates[0]["head"], branch)
                self.assertEqual(creates[0]["base"], "main")

    def test_creation_errors_fail_the_step(self):
        for workflow in WORKFLOWS:
            for status, message in (
                (403, "GitHub Actions is not permitted to create pull requests"),
                (422, "Validation Failed"),
                (500, "Internal Server Error"),
                (None, "Connection reset"),
            ):
                with self.subTest(workflow=workflow, status=status):
                    result = run_script(
                        workflow, failure={"status": status, "message": message}
                    )
                    self.assertEqual(len(result["failures"]), 1)
                    self.assertIn(message, result["failures"][0])
                    self.assertNotIn("PR created successfully", result["logs"])

    def test_leaderboard_reuses_open_refresh_pull_request(self):
        result = run_script(LEADERBOARD, existing=[{"number": 123}])
        self.assertEqual(result["failures"], [])
        self.assertEqual([c["name"] for c in result["calls"]], ["list"])
        query = result["calls"][0]["params"]
        self.assertEqual(query["owner"], "test-owner")
        self.assertEqual(query["repo"], "test-repo")
        self.assertEqual(query["state"], "open")
        self.assertEqual(query["head"], "test-owner:chore/contributor-leaderboard")
        self.assertEqual(query["base"], "main")
        self.assertTrue(any("#123" in line for line in result["logs"]))

    def test_leaderboard_lookup_failure_does_not_attempt_creation(self):
        result = run_script(
            LEADERBOARD,
            failure={"status": 403, "message": "Resource not accessible"},
            operation="list",
        )
        self.assertEqual([c["name"] for c in result["calls"]], ["list"])
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("Resource not accessible", result["failures"][0])


if __name__ == "__main__":
    unittest.main()
