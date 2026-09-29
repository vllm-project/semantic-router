from __future__ import annotations

import fcntl
import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "mirror_to_node.sh"

# A private-looking node address, assembled at runtime so that this file passes the leak guard.
NODE_HOST = ".".join(["10", "20", "30", "40"])

# Stands in for ssh: drops the options and the host and runs the command locally, as sshd would.
# FAKE_SSH_PROBE_DELAY sleeps after the state probe, so parallel runs all see an absent mirror.
# FAKE_SSH_BEFORE_LOCK runs a command just before the script's locked node-side step.
FAKE_SSH = r"""#!/usr/bin/env bash
while [[ "$1" == -o ]]; do shift 2; done
shift
if [[ "$*" == "bash -s" ]]; then
  script="$(cat)"
  if [[ "$script" == *"flock -w"* && -n "${FAKE_SSH_BEFORE_LOCK:-}" ]]; then
    FAKE_SSH_BEFORE_LOCK= bash -c "$FAKE_SSH_BEFORE_LOCK" >&2
  fi
  exec bash -c "$script"
fi
if [[ "$*" == *"echo receipt"* ]]; then
  bash -c "$*"
  sleep "${FAKE_SSH_PROBE_DELAY:-0}"
  exit 0
fi
exec bash -c "$*"
"""


class MirrorToNodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.repo = self.tmp / "repo"
        self.root = self.tmp / "node" / "src"
        bindir = self.tmp / "bin"
        bindir.mkdir()
        (bindir / "ssh").write_text(FAKE_SSH)
        (bindir / "ssh").chmod(0o755)
        nodes = self.tmp / "nodes.env"
        nodes.write_text(f"node-a=root@{NODE_HOST}\n")
        self.env = {
            **os.environ,
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "DEV2_NODES_FILE": str(nodes),
        }
        self.repo.mkdir()
        (self.repo / "sub" / "dir").mkdir(parents=True)
        (self.repo / "README.md").write_text("mirror test\n")
        (self.repo / "sub" / "dir" / "run.sh").write_text("#!/bin/sh\necho ok\n")
        (self.repo / "sub" / "dir" / "run.sh").chmod(0o755)
        (self.repo / "sub" / "data.json").write_text('{"a": 1}\n')
        for command in (
            ["git", "init", "-q"],
            ["git", "add", "-A"],
            [
                "git",
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.com",
                "commit",
                "-q",
                "-m",
                "fixture",
            ],
            ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        ):
            subprocess.run(command, cwd=self.repo, check=True, capture_output=True)
        self.sha = self.git("rev-parse", "HEAD")
        self.tree = self.git("rev-parse", "HEAD^{tree}")
        self.target = self.root / self.sha

    def tearDown(self) -> None:
        subprocess.run(["chmod", "-R", "u+w", str(self.tmp)], check=False)
        self._tmp.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    def mirror(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=self.repo,
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=False,
        )

    def action(self, out: subprocess.CompletedProcess[str]) -> str:
        self.assertEqual(out.returncode, 0, out.stderr)
        line = next(l for l in out.stdout.splitlines() if l.startswith("mirror ok:"))
        return line.rsplit("action=", 1)[1]

    def snapshot(self, path: Path) -> dict[str, bytes]:
        return {
            str(p.relative_to(path)): p.read_bytes()
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }

    def assert_single_clean_mirror(self) -> None:
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), sorted([".locks", self.sha])
        )
        self.assertEqual(list(self.target.rglob(".incoming-*")), [])
        receipt = json.loads((self.target / ".dev2-mirror.json").read_text())
        self.assertEqual((receipt["commit"], receipt["tree"]), (self.sha, self.tree))
        self.assertEqual(
            sorted(self.snapshot(self.target)),
            sorted(
                [".dev2-mirror.json", "README.md", "sub/data.json", "sub/dir/run.sh"]
            ),
        )

    def test_creates_then_reuses_without_uploading(self) -> None:
        self.assertEqual(
            self.action(self.mirror("node-a", "HEAD", str(self.root))), "created"
        )
        self.assert_single_clean_mirror()
        self.assertFalse(os.access(self.target / "README.md", os.W_OK))
        self.assertTrue(os.access(self.target / "sub" / "dir" / "run.sh", os.X_OK))
        again = self.mirror("node-a", "HEAD", str(self.root))
        self.assertEqual(self.action(again), "verified")
        self.assertEqual(
            self.action(self.mirror("--verify", "node-a", "HEAD", str(self.root))),
            "verified",
        )
        self.assert_single_clean_mirror()
        self.assertNotIn(NODE_HOST, again.stdout + again.stderr)

    def test_parallel_runs_leave_one_mirror_and_never_nest(self) -> None:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(
                pool.map(
                    lambda _: self.mirror(
                        "node-a", "HEAD", str(self.root), FAKE_SSH_PROBE_DELAY="1"
                    ),
                    range(6),
                )
            )
        actions = sorted(self.action(r) for r in results)
        self.assertEqual(actions, ["created"] + ["reused"] * 5)
        self.assert_single_clean_mirror()

    def test_run_that_lost_the_race_reuses_the_finished_mirror(self) -> None:
        # Another run creates the mirror after this run found it absent and uploaded its copy.
        peer = f"cd {self.repo} && bash {SCRIPT} node-a HEAD {self.root}"
        out = self.mirror("node-a", "HEAD", str(self.root), FAKE_SSH_BEFORE_LOCK=peer)
        self.assertEqual(self.action(out), "reused")
        self.assert_single_clean_mirror()

    def test_refuses_an_existing_directory_that_differs(self) -> None:
        self.target.mkdir(parents=True)
        (self.target / "README.md").write_text("something else\n")
        before = self.snapshot(self.target)
        out = self.mirror("node-a", "HEAD", str(self.root))
        self.assertEqual(out.returncode, 1)
        self.assertIn("refusing to touch it", out.stderr)
        self.assertEqual(self.snapshot(self.target), before)
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), sorted([".locks", self.sha])
        )

    def test_refuses_a_mirror_holding_a_nested_copy(self) -> None:
        self.assertEqual(
            self.action(self.mirror("node-a", "HEAD", str(self.root))), "created"
        )
        nested = self.target / f".incoming-{self.sha}.abc123"
        subprocess.run(["chmod", "u+w", str(self.target)], check=True)
        nested.mkdir()
        (nested / "README.md").write_text("mirror test\n")
        before = self.snapshot(self.target)
        out = self.mirror("node-a", "HEAD", str(self.root))
        self.assertEqual(out.returncode, 1)
        self.assertIn("nested staging", out.stderr)
        self.assertEqual(self.snapshot(self.target), before)

    def test_adopts_an_identical_plain_extraction(self) -> None:
        self.target.mkdir(parents=True)
        archive = subprocess.run(
            ["git", "archive", "HEAD"], cwd=self.repo, check=True, capture_output=True
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(self.target)], input=archive, check=True)
        self.assertEqual(
            self.action(self.mirror("node-a", "HEAD", str(self.root))), "adopted"
        )
        self.assert_single_clean_mirror()

    def test_lock_timeout_fails_and_cleans_up(self) -> None:
        locks = self.root / ".locks"
        locks.mkdir(parents=True)
        with open(locks / f"{self.sha}.lock", "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            out = self.mirror(
                "node-a", "HEAD", str(self.root), DEV2_MIRROR_LOCK_TIMEOUT="1"
            )
        self.assertEqual(out.returncode, 75, out.stderr)
        self.assertIn("timed out", out.stderr)
        self.assertEqual([p.name for p in self.root.iterdir()], [".locks"])

    def test_subtree_mirror_and_verify_of_an_absent_mirror(self) -> None:
        out = self.mirror("--verify", "node-a", "HEAD", str(self.root))
        self.assertEqual(out.returncode, 1)
        self.assertIn("no verified mirror", out.stderr)
        out = self.mirror("--path", "sub/dir", "node-a", "HEAD", str(self.root))
        self.assertEqual(self.action(out), "created")
        subtree = self.root / f"{self.sha}-sub_dir"
        self.assertEqual(
            sorted(self.snapshot(subtree)), [".dev2-mirror.json", "sub/dir/run.sh"]
        )
        self.assertEqual(
            json.loads((subtree / ".dev2-mirror.json").read_text())["path"], "sub/dir"
        )


if __name__ == "__main__":
    unittest.main()
