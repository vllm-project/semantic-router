import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import detect
import local
import runtime
import settings


class Merge(unittest.TestCase):
    def setUp(self):
        self.before = {
            "model": "opus",
            "enabledPlugins": {"vsr@vllm-sr": True},
            "pluginConfigs": {
                "other@market": {"options": {"x": 1}},
                "vsr@vllm-sr": {
                    "options": {"guard": False, "guardReview": 0.85},
                    "mcpServers": {"a": {"k": "v"}},
                },
            },
        }

    def test_keeps_every_other_key(self):
        after = settings.merge(
            self.before, "vsr@vllm-sr", {"guardReview": 0.8, "router": True}
        )
        self.assertEqual(after["model"], "opus")
        self.assertEqual(after["enabledPlugins"], {"vsr@vllm-sr": True})
        self.assertEqual(after["pluginConfigs"]["other@market"], {"options": {"x": 1}})
        self.assertEqual(
            after["pluginConfigs"]["vsr@vllm-sr"]["mcpServers"], {"a": {"k": "v"}}
        )
        self.assertEqual(
            after["pluginConfigs"]["vsr@vllm-sr"]["options"],
            {"guard": False, "guardReview": 0.8, "router": True},
        )
        self.assertEqual(
            self.before["pluginConfigs"]["vsr@vllm-sr"]["options"]["guardReview"],
            0.85,
            "input left unchanged",
        )

    def test_inline_and_installed_keys_are_separate(self):
        after = settings.merge(self.before, "vsr@inline", {"shieldInject": 0.95})
        self.assertEqual(
            after["pluginConfigs"]["vsr@inline"], {"options": {"shieldInject": 0.95}}
        )
        self.assertEqual(
            after["pluginConfigs"]["vsr@vllm-sr"]["options"]["guardReview"], 0.85
        )
        self.assertEqual(
            settings.effective_options(after, "vsr@inline")["shieldInject"], 0.95
        )
        self.assertEqual(
            settings.effective_options(after, "vsr@inline")["guardReview"],
            0.9,
            "defaults fill the rest",
        )
        self.assertEqual(settings.other_ids(after, "vsr@inline"), ["vsr@vllm-sr"])

    def test_empty_settings(self):
        self.assertEqual(
            settings.merge({}, "vsr@inline", {"router": False}),
            {"pluginConfigs": {"vsr@inline": {"options": {"router": False}}}},
        )

    def test_write_is_atomic_and_backs_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text('{"model": "opus"}\n')
            backup = settings.write(
                path,
                settings.render(
                    settings.merge(
                        json.loads(path.read_text()), "vsr@inline", {"guard": True}
                    )
                ),
            )
            self.assertEqual(json.loads(backup.read_text()), {"model": "opus"})
            self.assertEqual(
                json.loads(path.read_text())["pluginConfigs"]["vsr@inline"]["options"],
                {"guard": True},
            )

    def test_refuses_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text('{"model": ')
            with self.assertRaises(settings.SettingsError):
                settings.load(path)


class Values(unittest.TestCase):
    def test_types(self):
        values = settings.parse_values(
            [
                "guard=off",
                "guardReview=0.85",
                "calibrateEvery=300",
                "runtime=remote",
                "logPath=~/x.jsonl",
            ]
        )
        self.assertEqual(values["guard"], False)
        self.assertEqual(values["guardReview"], 0.85)
        self.assertEqual(values["calibrateEvery"], 300)
        self.assertEqual(values["runtime"], "remote")
        self.assertTrue(Path(values["logPath"]).is_absolute())

    def test_steps_lines(self):
        values = settings.parse_values(
            [
                "routerStepsLow=0.4",
                "routerStepsHigh=1.9",
                "routerStepsMinConfidence=0.25",
            ]
        )
        self.assertEqual(
            values,
            {
                "routerStepsLow": 0.4,
                "routerStepsHigh": 1.9,
                "routerStepsMinConfidence": 0.25,
            },
        )
        self.assertEqual(
            settings.parse_values(["routerStepsLow=1", "routerStepsHigh=1"]),
            {"routerStepsLow": 1.0, "routerStepsHigh": 1.0},
        )

    def test_rejects(self):
        for bad in (
            ["runtimeToken=abc"],
            ["guardReview=1.4"],
            ["runtime=gpu"],
            ["nope=1"],
            ["routerEasyBelow=1.6", "routerHardFrom=1.5"],
            ["runtimeUrl=localhost:8790"],
            ["routerStepsLow=2.1"],
            ["routerStepsMinConfidence=1.4"],
            ["routerStepsLow=1.3", "routerStepsHigh=0.8"],
        ):
            with self.subTest(bad=bad), self.assertRaises(settings.SettingsError):
                settings.parse_values(bad)


class PluginId(unittest.TestCase):
    listed: ClassVar[list] = [
        {"id": "pstack@pstack-claude", "installPath": "/c/pstack"},
        {
            "id": "vsr@vllm-sr",
            "installPath": "/c/sr/0.1.0",
            "readFromFolder": "/src/sr",
        },
    ]

    def test_running_copy_matches_marketplace_install(self):
        self.assertEqual(settings.resolve_id(self.listed, "/src/sr"), "vsr@vllm-sr")
        self.assertEqual(settings.resolve_id(self.listed, "/c/sr/0.1.0"), "vsr@vllm-sr")

    def test_running_copy_elsewhere_is_inline(self):
        self.assertEqual(
            settings.resolve_id(self.listed, "/home/me/sr-checkout"), "vsr@inline"
        )

    def test_without_root(self):
        self.assertEqual(settings.resolve_id(self.listed, None), "vsr@vllm-sr")
        self.assertEqual(settings.resolve_id([], None), "vsr@inline")


class Cli(unittest.TestCase):
    def test_plan_writes_nothing_and_apply_writes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ):
            os.environ["CLAUDE_CONFIG_DIR"] = tmp
            try:
                path = Path(tmp) / "settings.json"
                path.write_text('{"theme": "dark"}\n')
                self.assertEqual(
                    settings.main(
                        ["plan", "--id", "vsr@inline", "--set", "check=false"]
                    ),
                    0,
                )
                self.assertEqual(json.loads(path.read_text()), {"theme": "dark"})
                self.assertEqual(
                    settings.main(
                        ["apply", "--id", "vsr@inline", "--set", "check=false"]
                    ),
                    0,
                )
                self.assertEqual(
                    json.loads(path.read_text()),
                    {
                        "theme": "dark",
                        "pluginConfigs": {"vsr@inline": {"options": {"check": False}}},
                    },
                )
            finally:
                del os.environ["CLAUDE_CONFIG_DIR"]

    def test_apply_keeps_a_private_file_private(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(
            io.StringIO()
        ):
            os.environ["CLAUDE_CONFIG_DIR"] = tmp
            umask = os.umask(0o022)
            try:
                path = Path(tmp) / "settings.json"
                path.write_text('{"env": {"ANTHROPIC_API_KEY": "sk-test"}}\n')
                path.chmod(0o600)
                self.assertEqual(
                    settings.main(
                        ["apply", "--id", "vsr@inline", "--set", "check=false"]
                    ),
                    0,
                )
                self.assertEqual(
                    json.loads(path.read_text())["pluginConfigs"],
                    {"vsr@inline": {"options": {"check": False}}},
                )
                self.assertEqual(oct(stat.S_IMODE(path.stat().st_mode)), oct(0o600))
                backups = list(Path(tmp).glob("settings.json.sr-backup-*"))
                self.assertEqual(
                    [oct(stat.S_IMODE(b.stat().st_mode)) for b in backups], [oct(0o600)]
                )
            finally:
                os.umask(umask)
                del os.environ["CLAUDE_CONFIG_DIR"]


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


class Install(unittest.TestCase):
    def test_installs_the_runtime_from_a_pinned_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps = local.install_steps(Path(tmp) / "venv", "cpu", sys.executable)
        self.assertRegex(
            steps[-1][-1],
            r"^vllm-srun @ git\+https://github\.com/vllm-project/semantic-router\.git@[0-9a-f]{40}#subdirectory=src/model-runtime$",
        )


class Manifest(unittest.TestCase):
    def test_options_match_the_manifest(self):
        manifest = json.loads(
            (
                Path(__file__).resolve().parents[3] / ".claude-plugin" / "plugin.json"
            ).read_text()
        )
        declared = manifest.get("userConfig", {})
        self.assertEqual(set(declared), set(settings.OPTIONS))
        for name, (kind, default) in settings.OPTIONS.items():
            with self.subTest(option=name):
                self.assertEqual(declared[name].get("default", ""), default)
                self.assertEqual(
                    declared[name].get("sensitive", False), kind == "secret"
                )
                if kind == "choice":
                    self.assertEqual(
                        tuple(declared[name].get("options", ())), settings.CHOICES[name]
                    )

    def test_a_choice_takes_only_its_listed_values(self):
        self.assertEqual(settings.parse_value("logo", " smooth "), "smooth")
        with self.assertRaisesRegex(
            settings.SettingsError, "logo is one of auto, image, smooth, blocks, off"
        ):
            settings.parse_value("logo", "picture")
        with self.assertRaisesRegex(
            settings.SettingsError, "runtime is one of local-cpu, local-gpu, remote"
        ):
            settings.parse_value("runtime", "cloud")


class Recommend(unittest.TestCase):
    def base(self, **over):
        found = {
            "runtime": {"ready": False, "url": "u"},
            "pythons": [{"ok": True}],
            "memory": {"totalGiB": 16},
            "gpus": [],
        }
        found.update(over)
        return found

    def test_existing_runtime_wins(self):
        self.assertEqual(
            detect.recommend(self.base(runtime={"ready": True, "url": "u"}))["runtime"],
            "existing",
        )

    def test_gpu_with_free_memory(self):
        gpu = {"device": "cuda", "name": "g", "freeGiB": 30, "totalGiB": 48}
        rec = detect.recommend(self.base(gpus=[gpu]))
        self.assertEqual(
            (rec["runtime"], rec["device"], rec["model"]),
            ("local-gpu", "cuda", "vllm-sr/vela3-lite"),
        )
        self.assertEqual(rec["larger"], ["vllm-sr/vela3-mini", "vllm-sr/vela3-flash"])

    def test_busy_gpu_falls_back_to_cpu(self):
        gpu = {"device": "cuda", "name": "g", "freeGiB": 2, "totalGiB": 48}
        self.assertEqual(
            detect.recommend(self.base(gpus=[gpu]))["runtime"], "local-cpu"
        )

    def test_old_python_means_remote(self):
        self.assertEqual(
            detect.recommend(self.base(pythons=[{"ok": False}]))["runtime"], "remote"
        )


class KeepRunning(unittest.TestCase):
    """Keeping a ready runtime saves the `runtime` value that matches where it really runs."""

    def keep(self, url="http://127.0.0.1:8790/v1/decisions", device=None, saved=None):
        found = {
            "runtime": {"ready": True, "url": url, "device": device},
            "savedRuntime": saved,
            "pythons": [{"ok": True}],
            "memory": {"totalGiB": 16},
            "gpus": [],
        }
        return detect.recommend(found)

    def test_a_gpu_runtime_is_saved_as_local_gpu(self):
        for device in ("cuda", "rocm", "mps"):
            with self.subTest(device=device):
                rec = self.keep(device=device, saved="local-cpu")
                self.assertEqual(
                    (rec["runtime"], rec["setting"]), ("existing", "local-gpu")
                )
                self.assertIn(device, rec["why"])

    def test_a_cpu_runtime_is_saved_as_local_cpu(self):
        self.assertEqual(
            self.keep(device="cpu", saved="local-gpu")["setting"], "local-cpu"
        )

    def test_another_host_is_remote_whatever_its_device(self):
        self.assertEqual(
            self.keep(url="https://sr.example.com/v1/decisions", device="cuda")[
                "setting"
            ],
            "remote",
        )

    def test_an_unknown_device_keeps_the_saved_value_and_says_so(self):
        rec = self.keep(device=None, saved="local-gpu")
        self.assertEqual(rec["setting"], "local-gpu")
        self.assertIn("could not tell", rec["why"])
        rec = self.keep(device=None, saved=None)
        self.assertEqual(rec["setting"], "local-cpu")
        self.assertIn("could not tell", rec["why"])

    def test_the_summary_names_the_value_to_save(self):
        found = {
            "os": {"system": "Linux", "arch": "x86_64"},
            "cpus": 8,
            "memory": {"totalGiB": 16, "availableGiB": 8},
            "diskFreeGiB": 100,
            "gpus": [],
            "pythons": [],
            "git": True,
            "vllmSrun": None,
            "savedRuntime": None,
            "runtime": {
                "ready": True,
                "url": "http://127.0.0.1:8790/v1/decisions",
                "model": "vela3-lite",
                "device": "cuda",
                "portOpen": True,
            },
        }
        found["recommendation"] = detect.recommend(found)
        text = detect.summary(found)
        self.assertIn("ready, model vela3-lite on cuda", text)
        self.assertIn("runtime=local-gpu", text)


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

        with mock.patch.object(runtime.urllib.request, "urlopen", urlopen):
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
