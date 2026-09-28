from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1] / "check_no_private.sh"

# Private-looking values are assembled at runtime so that this file passes the guard.
NODE_IP = ".".join(["10", "20", "30", "40"])
NODE_NAME = "gpu" + "box-17"
OTHER_IP = ".".join(["172", "16", "5", "9"])
SECRET = "s3cr3t" + "Value42xyz"
ENDPOINT_HOST = "mirror" + ".example-private.test"
HF_TOKEN = "hf" + "_" + "AbCdEf" * 5
APIKEY = "apikey" + "_a1b2c3d4e5f6g7"
JEV_LIVE = "jv" + "_live_Q9w8E7r6T5y4U3"
GH_TOKEN = "gh" + "p_Z" + "x9" * 12
PAT = "github" + "_pat_" + "A1" * 12
KEY_HEADER = "-" * 5 + "BEGIN OPENSSH PRIVATE KEY" + "-" * 5
PRIVATE = [
    NODE_IP,
    NODE_NAME,
    OTHER_IP,
    SECRET,
    ENDPOINT_HOST,
    HF_TOKEN,
    APIKEY,
    JEV_LIVE,
    GH_TOKEN,
    PAT,
    KEY_HEADER,
]


class CheckNoPrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        private = self.tmp / "private"
        private.mkdir()
        self.work = self.tmp / "work"
        self.work.mkdir()
        (private / "nodes.env").write_text(f"node-a=root@{NODE_IP}\n")
        (private / "node-names.env").write_text(f"# extra\nnode-a={NODE_NAME}\n")
        (private / "secrets.env").write_text(
            f"# private\nJEV_MIRROR_TOKEN={SECRET}\n"
            f"JEV_MIRROR_ENDPOINT=https://{ENDPOINT_HOST}/v1/decide\n"
        )
        self.env = {
            **os.environ,
            "HOME": str(self.tmp),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "DEV2_NODES_FILE": str(private / "nodes.env"),
            "DEV2_NODE_NAMES_FILE": str(private / "node-names.env"),
            "DEV2_SECRETS_FILE": str(private / "secrets.env"),
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def guard(self, *args: str, cwd: Path | None = None, **env: str):
        return subprocess.run(
            ["bash", str(GUARD), *args],
            cwd=cwd or self.work,
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=False,
        )

    def git(self, repo: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def make_repo(self) -> Path:
        repo = self.work / "repo"
        repo.mkdir()
        self.git(repo, "init", "-q", "-b", "main")
        return repo

    def assert_nothing_private_printed(self, result) -> None:
        printed = (result.stdout + result.stderr).lower()
        for value in PRIVATE:
            self.assertNotIn(value.lower(), printed)

    def test_paths_mode_reports_only_location_and_category(self) -> None:
        leaky = self.work / "leaky.md"
        lines = [
            "clean line",
            f"ssh root@{NODE_IP}",
            f"host ip-{NODE_IP.replace('.', '-')} is up",
            f"the box {NODE_NAME.upper()} rebooted",
            f"peer at {OTHER_IP}:8080.",
            f"token={SECRET}",
            f"see https://{ENDPOINT_HOST}/status",
            HF_TOKEN,
            f"{APIKEY} {JEV_LIVE}",
            f"{GH_TOKEN} {PAT}",
            KEY_HEADER,
            f"dev{OTHER_IP} and v{NODE_IP}",
        ]
        leaky.write_text("\n".join(lines) + "\n")
        result = self.guard(str(leaky))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                f"{leaky}:2: ipv4,node-address",
                f"{leaky}:3: node-address",
                f"{leaky}:4: node-address",
                f"{leaky}:5: ipv4",
                f"{leaky}:6: secret-value",
                f"{leaky}:7: secret-value",
                f"{leaky}:8: hf-token",
                f"{leaky}:9: apikey-token,jev-live-token",
                f"{leaky}:10: github-token",
                f"{leaky}:11: private-key",
                f"{leaky}:12: ipv4,node-address",
            ],
        )
        self.assert_nothing_private_printed(result)

    def test_allowed_addresses_versions_placeholders_and_mentions_are_clean(
        self,
    ) -> None:
        clean = self.work / "clean.md"
        clean.write_text(
            "bind 0.0.0.0:8000 or 127.0.0.1; docs use 192.0.2.10 and 203.0.113.7\n"
            "netmask 255.255.255.0; versions 1.2.3.4.5, 2.1.281.554, v0.2.1, v1.0.0.0\n"
            "scan for `hf_`, `apikey_`, `jv_live_`, `gho_` before committing\n"
            f"export HF_TOKEN=hf_{'x' * 24}\n"
            "say node A or node-a, never the address\n"
        )
        result = self.guard(str(clean))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, "")
        piped = subprocess.run(
            ["bash", str(GUARD), "-"],
            input=f"ssh root@{NODE_IP}\n",
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(piped.stdout.splitlines(), ["(stdin):1: ipv4,node-address"])

    def test_staged_mode_checks_added_lines_and_new_paths(self) -> None:
        repo = self.make_repo()
        notes = repo / "notes.md"
        notes.write_text(f"old line with {OTHER_IP}\nkeep\n")
        (repo / f"old-{NODE_IP}.txt").write_text("old\n")
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-qm", "base")
        self.assertEqual(self.guard(cwd=repo).returncode, 0)
        notes.write_text(f"old line with {OTHER_IP}\nkeep\nnew line\ntoken {SECRET}\n")
        (repo / "added.txt").write_text(f"\n\nssh root@{NODE_IP}\n")
        (repo / f"empty-{NODE_IP}.txt").write_text("")
        (repo / f"old-{NODE_IP}.txt").unlink()
        self.git(repo, "add", "-A")
        result = self.guard(cwd=repo)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                "added.txt:3: ipv4,node-address",
                "empty-[REDACTED].txt:(path): ipv4,node-address",
                "notes.md:4: secret-value",
            ],
        )
        self.assert_nothing_private_printed(result)

    def test_log_and_rev_modes_cover_messages_history_and_paths(self) -> None:
        repo = self.make_repo()
        (repo / "a.txt").write_text("clean\n")
        self.git(repo, "add", "a.txt")
        self.git(repo, "commit", "-qm", "base")
        base = self.git(repo, "rev-parse", "HEAD")
        (repo / "a.txt").write_text(f"clean\n{HF_TOKEN}\n")
        self.git(repo, "commit", "-qam", f"record run on {NODE_NAME}")
        leak = self.git(repo, "rev-parse", "HEAD")
        (repo / "a.txt").write_text("clean\n")
        run_dir = repo / f"runs-{NODE_IP}"
        run_dir.mkdir()
        (run_dir / "log.txt").write_text("ok\n")
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-qm", "scrub the token")
        head = self.git(repo, "rev-parse", "HEAD")

        log = self.guard("--log", f"{base}..HEAD", cwd=repo)
        self.assertEqual(log.returncode, 1, log.stderr)
        self.assertEqual(
            log.stdout.splitlines(),
            [
                f"{leak}:(message):1: node-address",
                f"{leak}:a.txt:2: hf-token",
                f"{head}:runs-[REDACTED]/log.txt:(path): ipv4,node-address",
            ],
        )
        rev = self.guard("--rev", "HEAD", cwd=repo)
        self.assertEqual(rev.returncode, 1, rev.stderr)
        self.assertEqual(
            rev.stdout.splitlines(),
            ["HEAD:runs-[REDACTED]/log.txt:(path): ipv4,node-address"],
        )
        self.assertEqual(self.guard("--rev", base, cwd=repo).returncode, 0)
        self.assert_nothing_private_printed(log)
        self.assert_nothing_private_printed(rev)

    def test_strict_requires_the_private_files(self) -> None:
        missing = str(self.tmp / "missing.env")
        result = self.guard("--strict", str(self.work), DEV2_NODES_FILE=missing)
        self.assertEqual(result.returncode, 2)
        self.assertIn("nodes file not found", result.stderr)
        relaxed = self.guard(str(self.work), DEV2_NODES_FILE=missing)
        self.assertEqual(relaxed.returncode, 0, relaxed.stdout)
        self.assertIn("check is disabled", relaxed.stderr)


if __name__ == "__main__":
    unittest.main()
