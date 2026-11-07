import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import runtime
import settings


class RemoteWarning(unittest.TestCase):
    READY: ClassVar[dict] = {
        "url": "u",
        "ready": True,
        "model": "vllm-sr/vela3-lite",
        "health": {"status": "ready"},
        "healthMs": 5,
    }

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(
            runtime, "health", return_value=dict(self.READY)
        ), mock.patch.object(runtime, "run_probes", return_value=[]), mock.patch.dict(
            os.environ, {"SR_RUNTIME_TOKEN": ""}
        ), contextlib.redirect_stdout(
            out
        ), contextlib.redirect_stderr(
            err
        ):
            code = runtime.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_plain_http_to_another_host_warns(self):
        code, out, err = self.run_cli(
            "health", "--url", "http://10.0.0.5:8790/v1/decisions"
        )
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("ready"), out)
        self.assertIn("unencrypted", err)
        self.assertIn("https", err)
        self.assertNotIn("token", err)

    def test_the_warning_names_the_token_when_one_is_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = Path(tmp) / "token"
            token.write_text("s3cret\n")
            token.chmod(0o600)
            _, _, err = self.run_cli(
                "health",
                "--url",
                "http://sr.example.com/v1/decisions",
                "--token-file",
                str(token),
            )
        self.assertIn("unencrypted", err)
        self.assertIn("token", err)
        self.assertNotIn("s3cret", err)

    def test_probe_warns_too_and_json_stays_parseable(self):
        _, _, err = self.run_cli(
            "probe", "--smoke", "--url", "http://sr.example.com/v1/decisions"
        )
        self.assertIn("unencrypted", err)
        _, out, err = self.run_cli(
            "health", "--json", "--url", "http://sr.example.com/v1/decisions"
        )
        self.assertTrue(json.loads(out)["ready"])
        self.assertIn("unencrypted", err)

    def test_this_machine_and_https_do_not_warn(self):
        for url in (
            "http://127.0.0.1:8790/v1/decisions",
            "http://127.8.0.1:8790/v1/decisions",
            "http://localhost:8790/v1/decisions",
            "http://[::1]:8790/v1/decisions",
            "https://sr.example.com/v1/decisions",
        ):
            with self.subTest(url=url):
                code, out, err = self.run_cli("health", "--url", url)
                self.assertEqual((code, err), (0, ""))
                self.assertTrue(out.startswith("ready"), out)


class Token(unittest.TestCase):
    def test_a_token_file_others_can_read_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = Path(tmp) / "token"
            token.write_text("s3cret\n")
            token.chmod(0o644)
            with self.assertRaisesRegex(settings.SettingsError, "chmod 600"):
                runtime.read_token(str(token))
            token.chmod(0o600)
            self.assertEqual(runtime.read_token(str(token)), "s3cret")

    def test_the_cli_refuses_it_before_sending_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            token = Path(tmp) / "token"
            token.write_text("s3cret\n")
            token.chmod(0o640)
            err = io.StringIO()
            with mock.patch.object(
                runtime, "health", side_effect=AssertionError("sent")
            ), contextlib.redirect_stderr(err):
                code = runtime.main(["health", "--token-file", str(token)])
        self.assertEqual(code, 2)
        self.assertNotIn("s3cret", err.getvalue())

    def test_a_redirect_to_another_origin_drops_the_token(self):
        handler = runtime._SameOriginRedirect()
        request = runtime.urllib.request.Request(
            "https://sr.example.com/v1/decisions",
            b"{}",
            {"authorization": "Bearer s3cret"},
        )
        for url, kept in (
            ("https://sr.example.com/v2/decisions", True),
            ("https://elsewhere.example.com/v1/decisions", False),
            ("http://sr.example.com/v1/decisions", False),
            ("https://sr.example.com:8443/v1/decisions", False),
        ):
            with self.subTest(url=url):
                new = handler.redirect_request(request, None, 302, "", {}, url)
                self.assertEqual(new.has_header("Authorization"), kept)


class HealthDevice(unittest.TestCase):
    MODELS: ClassVar[dict] = {
        "object": "list",
        "data": [{"id": "vela3-lite", "device": "cuda", "accelerator": "cuda"}],
    }

    def health_with(self, models):
        replies = {
            "/health": {"status": "ready", "model": "vela3-lite"},
            "/v1/models": models,
        }

        def urlopen(request, timeout):
            body = json.dumps(replies[request.full_url.split("8790", 1)[1]]).encode()
            return contextlib.nullcontext(io.BytesIO(body))

        with mock.patch.object(runtime.OPENER, "open", urlopen):
            return runtime.health("http://127.0.0.1:8790/v1/decisions")

    def test_reports_the_device_the_runtime_runs_on(self):
        self.assertEqual(self.health_with(self.MODELS)["device"], "cuda")

    def test_falls_back_to_the_accelerator(self):
        self.assertEqual(
            self.health_with({"data": [{"id": "m", "accelerator": "rocm"}]})["device"],
            "rocm",
        )

    def test_an_older_runtime_has_no_device(self):
        self.assertIsNone(self.health_with({"data": [{"id": "m"}]})["device"])


if __name__ == "__main__":
    unittest.main()
