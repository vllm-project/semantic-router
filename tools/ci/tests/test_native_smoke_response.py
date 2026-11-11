"""A transport smoke must reject false-positive and truncated responses."""

import importlib.util
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

PATH = Path(__file__).resolve().parents[2] / "test/smoke/riscv64-standalone.py"


class NativeSmokeResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("native_smoke", PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_buffered_requires_selected_model_and_content(self):
        body = {"model": "code-model", "choices": [{"message": {"content": "ok"}}]}
        self.assertEqual(
            self.module.check_response(json.dumps(body), False, "code-model"), "ok"
        )
        with self.assertRaises(ValueError):
            self.module.check_response(json.dumps(body), False, "general-model")

    def test_fixture_is_readable_by_container_uid_under_private_umask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            previous = os.umask(0o077)
            try:
                self.module.write_configuration(path, "backend:8000", 19000)
            finally:
                os.umask(previous)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            self.assertEqual(
                json.loads(path.read_text())["listeners"][0]["port"], 19000
            )

    def test_stream_without_done_is_rejected(self):
        raw = 'data: {"model":"code-model","choices":[{"delta":{"content":"ok"}}]}\n\n'
        with self.assertRaisesRegex(ValueError, "DONE"):
            self.module.check_response(raw, True, "code-model")
        self.assertEqual(
            self.module.check_response(raw + "data: [DONE]\n\n", True, "code-model"),
            "ok",
        )

    def test_done_without_content_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.check_response("data: [DONE]\n\n", True, "code-model")

    def test_buffered_empty_choices_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module.check_response(
                '{"model":"code-model","choices":[]}', False, "code-model"
            )

    def test_stream_requires_event_delimiters(self):
        raw = 'data: {"model":"code-model","choices":[{"delta":{"content":"ok"}}]}\ndata: [DONE]\n'
        with self.assertRaisesRegex(ValueError, "event"):
            self.module.check_response(raw, True, "code-model")

    def test_wrong_media_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "media type"):
            self.module.check_media_type("application/json", True)
        self.module.check_media_type("text/event-stream; charset=utf-8", True)

    def test_cleanup_timeout_does_not_skip_remaining_actions_or_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            called = []

            def timeout():
                raise subprocess.TimeoutExpired("docker rm", 1)

            report = Path(directory) / "result.json"
            receipt = {"passed": True}
            self.module.finish_cleanup(
                [("stop", timeout), ("provider", lambda: called.append("provider"))],
                receipt,
                report,
            )
            self.assertEqual(called, ["provider"])
            saved = json.loads(report.read_text())
            self.assertFalse(saved["passed"])
            self.assertTrue(saved["cleanup_errors"])

    def test_router_removal_cleans_its_volume_even_if_log_collection_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            docker = Mock(side_effect=[RuntimeError("logs unavailable"), "removed"])
            with self.assertRaisesRegex(RuntimeError, "logs unavailable"):
                self.module.remove_router(docker, "owned-test", Path(directory))
            self.assertEqual(
                docker.call_args_list,
                [
                    call("logs", "owned-test"),
                    call("rm", "--force", "--volumes", "owned-test"),
                ],
            )

    def test_router_runs_unprivileged_with_only_explicit_writable_mounts(self):
        args = self.module.router_run_args(
            "router:test", "owned", "network", Path("/tmp/config.json")
        )
        self.assertEqual(args[args.index("--user") + 1], "65532:65532")
        self.assertIn("--read-only", args)
        self.assertEqual(args[args.index("--cap-drop") + 1], "ALL")
        self.assertIn("no-new-privileges", args)
        mounts = [args[i + 1] for i, value in enumerate(args) if value == "--tmpfs"]
        self.assertEqual(
            mounts,
            [
                "/tmp:rw,noexec,nosuid,mode=1777",
                "/app/config:rw,noexec,nosuid,uid=65532,gid=65532,mode=0700",
                "/app/models:rw,noexec,nosuid,uid=65532,gid=65532,mode=0700",
            ],
        )
        self.assertIn("/tmp/config.json:/app/config/config.yaml:ro", args)


if __name__ == "__main__":
    unittest.main()
