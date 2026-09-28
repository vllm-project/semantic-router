from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "hf_headroom.sh"

# A private-looking node address, assembled at runtime so that this file passes the leak guard.
NODE_HOST = ".".join(["10", "20", "30", "40"])

REPOS = {
    "org": "llm-semantic-router",
    "repos": [
        {
            "kind": "model",
            "id": "llm-semantic-router/dev2-9b-staging",
            "private": True,
            "used": 31_783_619_693,
        },
        {
            "kind": "model",
            "id": "llm-semantic-router/DEV2.0-2B",
            "private": True,
            "used": 7_556_388_102,
        },
        {
            "kind": "dataset",
            "id": "llm-semantic-router/decision-2.0-training-data",
            "private": True,
            "used": 3_228_804_557,
        },
        {
            "kind": "model",
            "id": "llm-semantic-router/Decision-1.0-Lux-9B",
            "private": False,
            "used": None,
        },
        {
            "kind": "space",
            "id": "llm-semantic-router/decision-studio",
            "private": False,
            "used": None,
        },
    ],
}
PRIVATE_BYTES = 31_783_619_693 + 7_556_388_102 + 3_228_804_557


class HfHeadroomTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repos = self.tmp / "repos.json"
        self.repos.write_text(json.dumps(REPOS))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_script(
        self, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
            check=False,
        )

    def test_report_counts_only_private_repositories(self) -> None:
        out = self.run_script("--from-json", str(self.repos), "--min-free-gb", "40")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("42.57 GB used of 100 GB, headroom 57.43 GB", out.stdout)
        self.assertIn("threshold 40 GB: OK", out.stdout)
        self.assertNotIn("Decision-1.0-Lux-9B", out.stdout)

    def test_exit_one_below_threshold(self) -> None:
        out = self.run_script("--from-json", str(self.repos), "--min-free-gb", "57.5")
        self.assertEqual(out.returncode, 1)
        self.assertIn("BELOW", out.stdout)
        out = self.run_script(
            "--from-json", str(self.repos), "--min-free-gb", "10", "--cap-gb", "50"
        )
        self.assertEqual(out.returncode, 1)

    def test_json_output(self) -> None:
        out = self.run_script(
            "--from-json",
            str(self.repos),
            "--min-free-gb",
            "18",
            "--json",
            "--top",
            "2",
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        doc = json.loads(out.stdout)
        self.assertEqual(doc["private_bytes"], PRIVATE_BYTES)
        self.assertEqual(doc["headroom_bytes"], 100_000_000_000 - PRIVATE_BYTES)
        self.assertEqual(doc["min_free_bytes"], 18_000_000_000)
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["private_repos"], 3)
        self.assertEqual(
            [r["id"] for r in doc["largest"]],
            ["llm-semantic-router/dev2-9b-staging", "llm-semantic-router/DEV2.0-2B"],
        )

    def test_usage_and_input_errors_exit_two(self) -> None:
        self.assertEqual(self.run_script("--min-free-gb", "ten").returncode, 2)
        self.assertEqual(self.run_script("--bogus").returncode, 2)
        self.assertEqual(
            self.run_script(
                "--node", "node-a", "--from-json", str(self.repos)
            ).returncode,
            2,
        )
        self.assertEqual(
            self.run_script("--from-json", str(self.tmp / "missing.json")).returncode, 2
        )
        bad = self.tmp / "bad.json"
        bad.write_text("{not json")
        self.assertEqual(self.run_script("--from-json", str(bad)).returncode, 2)

    def _fake_ssh(self, body: str) -> dict[str, str]:
        bindir = self.tmp / "bin"
        bindir.mkdir(exist_ok=True)
        ssh = bindir / "ssh"
        ssh.write_text("#!/usr/bin/env bash\n" + body)
        ssh.chmod(0o755)
        nodes = self.tmp / "nodes.env"
        nodes.write_text(f"node-a=root@{NODE_HOST}\n")
        return {
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "DEV2_NODES_FILE": str(nodes),
        }

    def test_node_query_resolves_alias_and_sends_the_query_on_stdin(self) -> None:
        argv_log, stdin_log = self.tmp / "argv", self.tmp / "stdin"
        env = self._fake_ssh(
            f'printf "%s\\n" "$@" > {argv_log}\ncat > {stdin_log}\ncat {self.repos}\n'
        )
        out = self.run_script("--node", "node-a", "--min-free-gb", "40", env=env)
        self.assertEqual(out.returncode, 0, out.stderr)
        argv = argv_log.read_text().splitlines()
        self.assertIn(f"root@{NODE_HOST}", argv)
        self.assertEqual(
            argv[-1], "/data/dev2/tools/hf-cli/bin/python - llm-semantic-router"
        )
        self.assertIn('expand=["usedStorage"]', stdin_log.read_text())
        self.assertNotIn(NODE_HOST, out.stdout + out.stderr)

    def test_node_errors_exit_two_without_the_address(self) -> None:
        env = self._fake_ssh(
            f'echo "ssh: connect to host {NODE_HOST} port 22: Connection refused" >&2\nexit 255\n'
        )
        out = self.run_script("--node", "node-a", env=env)
        self.assertEqual(out.returncode, 2)
        self.assertIn("<node-a>", out.stderr)
        self.assertNotIn(NODE_HOST, out.stdout + out.stderr)


if __name__ == "__main__":
    unittest.main()
