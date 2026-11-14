from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "ci"))

import latest_tags  # noqa: E402
from latest_tags import CURRENT, MOVE, NEWER, decide, published_names  # noqa: E402

REGISTRY = "ghcr.io/o"


class MainHistory:
    """Main is a <- b <- c; x branches from a and never reaches main."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q", "-b", "main")
        self.a, self.b, self.c = (self.commit(name) for name in "abc")
        self.git("checkout", "-q", "-b", "side", self.a)
        self.x = self.commit("x")
        self.git("checkout", "-q", "main")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "ci",
                "GIT_AUTHOR_EMAIL": "ci@example.com",
                "GIT_COMMITTER_NAME": "ci",
                "GIT_COMMITTER_EMAIL": "ci@example.com",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
            },
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(self, message: str) -> str:
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")


class DecisionTests(unittest.TestCase):
    def test_moves_unless_latest_holds_a_newer_commit(self) -> None:
        newer = [("c", b"C"), ("b", None)]
        self.assertEqual(decide(b"A", b"A", newer), (CURRENT, ""))
        self.assertEqual(decide(b"C", b"A", newer), (NEWER, "c"))
        # An older commit, a release, or no latest yet.
        for current in (b"Z", None):
            with self.subTest(current=current):
                self.assertEqual(decide(current, b"A", newer), (MOVE, ""))
        self.assertEqual(decide(b"C", b"C2", []), (MOVE, ""))

    def test_latest_names_include_aliases_but_not_fixtures(self) -> None:
        self.assertEqual(
            published_names(
                ["dashboard", "provider-mocker", "vllm-sr", "vllm-sr-rocm"]
            ),
            ["dashboard", "vllm-sr", "extproc", "vllm-sr-rocm", "extproc-rocm"],
        )


class MoveTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.history = MainHistory(Path(directory.name))
        self.registry: dict[str, bytes] = {}
        for name in ("vllm-sr", "extproc"):
            for commit in (self.history.a, self.history.b, self.history.c):
                self.registry[self.ref(name, commit)] = f"{name}@{commit}".encode()

    def ref(self, name: str, tag: str) -> str:
        return f"{REGISTRY}/semantic-router/{name}:{tag}"

    def latest(self, name: str = "vllm-sr") -> bytes | None:
        return self.registry.get(self.ref(name, "latest"))

    def point_latest(self, commit: str) -> None:
        for name in ("vllm-sr", "extproc"):
            self.registry[self.ref(name, "latest")] = self.registry[
                self.ref(name, commit)
            ]

    def run_move(self, source: str) -> tuple[list[str], str]:
        moved: list[str] = []

        def move(repository: str, commit: str) -> None:
            moved.append(repository.rsplit("/", 1)[1])
            self.registry[f"{repository}:latest"] = self.registry[
                f"{repository}:{commit}"
            ]

        argv = [
            "latest_tags.py",
            "--images",
            '["vllm-sr", "provider-mocker"]',
            "--source",
            source,
            "--main",
            "main",
            "--registry",
            REGISTRY,
        ]
        stdout = io.StringIO()
        with (
            contextlib.chdir(self.history.root),
            patch.object(sys, "argv", argv),
            patch.object(latest_tags, "registry_manifest", self.registry.get),
            patch.object(latest_tags, "move", move),
            contextlib.redirect_stdout(stdout),
        ):
            self.assertEqual(latest_tags.main(), 0)
        return moved, stdout.getvalue()

    def test_mains_head_takes_latest(self) -> None:
        self.point_latest(self.history.b)
        moved, _ = self.run_move(self.history.c)
        self.assertEqual(moved, ["vllm-sr", "extproc"])
        self.assertEqual(self.latest(), f"vllm-sr@{self.history.c}".encode())

    def test_an_older_commit_finishing_last_keeps_only_its_sha_tags(self) -> None:
        self.point_latest(self.history.c)
        moved, output = self.run_move(self.history.b)
        self.assertEqual(moved, [])
        self.assertEqual(self.latest("extproc"), f"extproc@{self.history.c}".encode())
        self.assertIn("A newer main commit already holds latest", output)

    def test_an_older_commit_moves_latest_past_an_even_older_one(self) -> None:
        # c is still building, or failed: b is the newest published commit.
        for name in ("vllm-sr", "extproc"):
            del self.registry[self.ref(name, self.history.c)]
        self.point_latest(self.history.a)
        moved, _ = self.run_move(self.history.b)
        self.assertEqual(moved, ["vllm-sr", "extproc"])
        self.registry[self.ref("vllm-sr", "latest")] = b"release v0.5.0"
        moved, _ = self.run_move(self.history.b)
        self.assertEqual(moved, ["vllm-sr"])

    def test_latest_already_at_this_commit_is_left_alone(self) -> None:
        self.point_latest(self.history.c)
        moved, output = self.run_move(self.history.c)
        self.assertEqual(moved, [])
        self.assertIn('"vllm-sr": "current"', output)

    def test_commits_off_main_or_far_behind_it_never_move_latest(self) -> None:
        self.point_latest(self.history.b)
        for name in ("vllm-sr", "extproc"):
            self.registry[self.ref(name, self.history.x)] = b"x"
        moved, output = self.run_move(self.history.x)
        self.assertEqual(moved, [])
        self.assertIn("is not on main", output)
        with patch.object(latest_tags, "MAX_NEWER_COMMITS", 1):
            moved, output = self.run_move(self.history.a)
        self.assertEqual(moved, [])
        self.assertIn("main is 2 commits past", output)

    def test_an_unpublished_commit_cannot_move_latest(self) -> None:
        del self.registry[self.ref("vllm-sr", self.history.c)]
        with self.assertRaisesRegex(RuntimeError, "is not published"):
            self.run_move(self.history.c)


class RegistryRetryTests(unittest.TestCase):
    def test_registry_calls_retry_a_transient_registry_error(self) -> None:
        # ghcr.io answered 500 to one layer upload of a Main publication.
        with patch.object(latest_tags.subprocess, "run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, b"{}", b"")
            latest_tags.move("ghcr.io/o/semantic-router/vllm-sr", "abc")
            latest_tags.registry_manifest("ghcr.io/o/semantic-router/vllm-sr:abc")
        for call in run.call_args_list:
            self.assertEqual(call.args[0][2:4], ["--retry-times", "3"])
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
