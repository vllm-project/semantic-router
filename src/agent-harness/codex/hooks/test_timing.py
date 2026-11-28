"""Tool timing: what the async hook writes beside Codex and sends to a stub management API."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import timing
import vsr

SESSION = "019a7c2e-0000-7000-8000-000000000005"
T0 = 1_794_900_000.0


class Management:
    """A stub management API that keeps every tool-events batch it gets."""

    def __init__(self):
        self.batches: list[dict] = []
        self.auth: list[str | None] = []
        self.redirect: str | None = None
        management = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                if management.redirect:
                    self.send_response(307)
                    self.send_header("location", management.redirect + self.path)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                if self.path == timing.PATH:
                    management.batches.append(body)
                    management.auth.append(self.headers.get("Authorization"))
                data = b'{"accepted":1}'
                self.send_response(200 if self.path == timing.PATH else 404)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


def event(name: str, call_id: str | None, tool: str = "Bash", **tool_input) -> dict:
    """A tool event as Codex 0.166 sends it (hooks crate, *.command.input)."""
    found = {
        "hook_event_name": name,
        "session_id": SESSION,
        "turn_id": "turn-1",
        "cwd": "/work",
        "model": "vllm-sr/auto",
        "permission_mode": "default",
        "transcript_path": None,
        "tool_name": tool,
        "tool_input": tool_input or {"command": "pytest -q"},
    }
    if call_id is not None:
        found["tool_use_id"] = call_id
    if name == "PostToolUse":
        found["tool_response"] = "3 passed, secret output"
    return found


class TimingTest(unittest.TestCase):
    def setUp(self):
        self.management = Management()
        self.addCleanup(self.management.server.shutdown)
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.home = Path(home.name)
        self.opts = {
            "managementUrl": self.management.url,
            "managementToken": "",
            "home": str(self.home),
        }

    def record(self, one: dict, at: float) -> None:
        timing.record(one, self.opts, vsr.NO_REDIRECTS, now=at)

    def test_a_call_sends_when_it_ran_and_nothing_of_its_input_or_output(self):
        self.record(event("PreToolUse", "call_1"), T0)
        self.record(event("PostToolUse", "call_1"), T0 + 1.25)

        self.assertEqual(
            self.management.batches,
            [
                {
                    "harness": "codex",
                    "session_id": SESSION,
                    "events": [
                        {
                            "tool_call_id": "call_1",
                            "tool": "Bash",
                            "status": "succeeded",
                            "started_at": timing.iso(T0),
                            "ended_at": timing.iso(T0 + 1.25),
                        }
                    ],
                }
            ],
        )
        self.assertNotIn("pytest", json.dumps(self.management.batches))
        self.assertNotIn("secret", json.dumps(self.management.batches))
        # The call's file is gone once it is sent, and it never held the command itself.
        self.assertEqual(list((self.home / timing.SPOOL).glob("*/*")), [])

    def test_an_approval_is_matched_by_its_command_and_sent_beside_the_whole_wait(self):
        self.record(event("PreToolUse", "call_a", command="rm -rf build"), T0)
        self.record(event("PreToolUse", "call_b", command="git push"), T0 + 0.01)
        # Codex 0.166 names no call in PermissionRequest and adds the reason it gives.
        self.record(
            event("PermissionRequest", None, command="git push", description="push"),
            T0 + 0.2,
        )
        self.record(event("PostToolUse", "call_a"), T0 + 0.5)
        self.record(event("PostToolUse", "call_b"), T0 + 9.0)

        sent = {
            b["events"][0]["tool_call_id"]: b["events"][0]
            for b in self.management.batches
        }
        self.assertNotIn("approval_requested_at", sent["call_a"])
        self.assertEqual(sent["call_b"]["approval_requested_at"], timing.iso(T0 + 0.2))
        self.assertEqual(sent["call_b"]["started_at"], timing.iso(T0 + 0.01))
        self.assertEqual(sent["call_b"]["ended_at"], timing.iso(T0 + 9.0))

    def test_a_tool_starts_after_the_guard_codex_waited_for(self):
        self.record(event("PreToolUse", "call_1"), T0)
        with mock.patch.object(timing.time, "time", return_value=T0 + 0.4):
            timing.checked(event("PreToolUse", "call_1"), self.opts)
        self.record(event("PostToolUse", "call_1"), T0 + 1.0)

        self.assertEqual(
            self.management.batches[0]["events"][0]["started_at"], timing.iso(T0 + 0.4)
        )

    def test_a_post_that_runs_before_its_pre_has_written_waits_for_it(self):
        later = threading.Timer(
            0.2, self.record, (event("PreToolUse", "call_1", tool="Read"), T0)
        )
        later.start()
        self.record(event("PostToolUse", "call_1", tool="Read"), T0 + 0.05)
        later.join()

        self.assertEqual(self.management.batches[0]["events"][0]["tool"], "Read")

    def test_the_token_and_the_pinned_session_go_and_a_redirect_is_refused(self):
        self.opts["managementToken"] = "mgmt-key"
        with mock.patch.dict(os.environ, {"VSR_SESSION_ID": "pinned"}):
            self.record(event("PreToolUse", "call_1"), T0)
            self.record(event("PostToolUse", "call_1"), T0 + 1)
            self.management.redirect = "http://127.0.0.1:9"
            self.record(event("PreToolUse", "call_2"), T0)
            self.record(event("PostToolUse", "call_2"), T0 + 1)

        self.assertEqual(self.management.auth, ["Bearer mgmt-key"])
        self.assertEqual([b["session_id"] for b in self.management.batches], ["pinned"])

    def transcript(self, *outputs: tuple[str, str]) -> str:
        """A session transcript holding the output Codex 0.166 gave the model for each call, in
        its rollout's shape."""
        path = self.home / "rollout.jsonl"
        with open(path, "a", encoding="utf-8") as rollout:
            for call_id, output in outputs:
                payload = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
                rollout.write(
                    json.dumps({"type": "response_item", "payload": payload}) + "\n"
                )
        return str(path)

    def test_a_command_that_exited_with_another_code_than_0_is_failed(self):
        exited = "Chunk ID: 7ca096\nWall time: 0.0000 seconds\nProcess exited with code {}\nOriginal token count: 12\nOutput:\n{}"
        path = self.transcript(
            ("call_tests", exited.format(1, "FAILED (failures=1)\n")),
            (
                "call_curl",
                "Exit code: 6\nWall time: 0.2 seconds\nOutput:\ncurl: (6) Could not resolve host\n",
            ),
            ("call_echo", exited.format(0, "Exit code: 3\n")),
            ("call_read", "def add(a, b):\n    return a + b\n"),
        )
        for call_id in (
            "call_tests",
            "call_curl",
            "call_echo",
            "call_read",
            "call_unwritten",
        ):
            self.record(event("PreToolUse", call_id), T0)
            post = event("PostToolUse", call_id)
            post["transcript_path"] = path
            # PostToolUse carries the output alone
            post["tool_response"] = "FAILED (failures=1)\n"
            with mock.patch.object(timing, "FIND_SECONDS", 0.2):
                self.record(post, T0 + 1)

        sent = {
            b["events"][0]["tool_call_id"]: b["events"][0]["status"]
            for b in self.management.batches
        }
        self.assertEqual(
            sent,
            {
                "call_tests": "failed",
                "call_curl": "failed",
                "call_echo": "succeeded",
                "call_read": "succeeded",
                "call_unwritten": "succeeded",
            },
        )
        self.assertNotIn("FAILED", json.dumps(self.management.batches))

    def test_a_post_waits_for_codex_to_write_the_output_it_reads(self):
        path = self.transcript()
        later = threading.Timer(
            0.3,
            self.transcript,
            (("call_1", "Exit code: 1\nWall time: 0.1 seconds\nOutput:\n"),),
        )
        self.record(event("PreToolUse", "call_1"), T0)
        post = event("PostToolUse", "call_1")
        post["transcript_path"] = path
        later.start()
        self.record(post, T0 + 1)
        later.join()

        self.assertEqual(self.management.batches[0]["events"][0]["status"], "failed")

    def test_no_management_api_writes_and_sends_nothing(self):
        self.opts["managementUrl"] = ""
        self.record(event("PreToolUse", "call_1"), T0)
        self.record(event("PostToolUse", "call_1"), T0 + 1)

        self.assertFalse((self.home / timing.SPOOL).exists())
        self.assertEqual(self.management.batches, [])

    def test_a_call_that_never_ends_is_swept_by_a_later_one(self):
        self.record(event("PreToolUse", "call_old"), T0)
        old = next((self.home / timing.SPOOL).glob("*/*"))
        os.utime(old, (T0, T0))
        self.record(event("PreToolUse", "call_new"), T0 + timing.KEEP_SECONDS + 1)

        self.assertEqual(
            [p.name for p in (self.home / timing.SPOOL).glob("*/*")], ["call_new"]
        )

    def test_codex_runs_the_timing_hook_async_for_every_tool_and_it_prints_at_once(
        self,
    ):
        hooks = json.loads(
            (Path(vsr.__file__).parent.parent / "hooks.json").read_text()
        )
        timed = {
            name: group
            for name, groups in hooks["hooks"].items()
            for group in groups
            if group["hooks"][0]["command"].endswith(" tool")
        }
        self.assertEqual(
            sorted(timed), ["PermissionRequest", "PostToolUse", "PreToolUse"]
        )
        for group in timed.values():
            self.assertEqual(group["matcher"], "*")
            self.assertIs(group["hooks"][0]["async"], True)
        out = io.StringIO()
        env = {"CODEX_HOME": str(self.home), "VSR_MANAGEMENT_URL": self.management.url}
        with mock.patch.dict(os.environ, env), mock.patch(
            "sys.stdin", io.StringIO(json.dumps(event("PreToolUse", "call_1")))
        ), redirect_stdout(out):
            started = time.monotonic()
            vsr.main(["vsr.py", "tool"])
        self.assertEqual(out.getvalue().strip(), "{}")
        self.assertLess(time.monotonic() - started, 0.5)


if __name__ == "__main__":
    unittest.main()
