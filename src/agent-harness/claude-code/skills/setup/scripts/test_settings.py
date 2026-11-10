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
            self.before, "vsr@vllm-sr", {"guardReview": 0.8, "router": False}
        )
        self.assertEqual(after["model"], "opus")
        self.assertEqual(after["enabledPlugins"], {"vsr@vllm-sr": True})
        self.assertEqual(after["pluginConfigs"]["other@market"], {"options": {"x": 1}})
        self.assertEqual(
            after["pluginConfigs"]["vsr@vllm-sr"]["mcpServers"], {"a": {"k": "v"}}
        )
        self.assertEqual(
            after["pluginConfigs"]["vsr@vllm-sr"]["options"],
            {"guard": False, "guardReview": 0.8, "router": False},
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

    def test_a_default_value_is_not_saved_and_clears_a_saved_one(self):
        after = settings.merge(
            self.before,
            "vsr@vllm-sr",
            {"guard": True, "guardReview": 0.9, "router": True, "check": False},
        )
        self.assertNotIn("options", after["pluginConfigs"]["vsr@vllm-sr"])
        self.assertEqual(
            after["pluginConfigs"]["vsr@vllm-sr"]["mcpServers"], {"a": {"k": "v"}}
        )
        self.assertEqual(
            settings.merge({"model": "opus"}, "vsr@inline", {"check": False}),
            {"model": "opus"},
            "nothing to save leaves no empty entry",
        )

    def test_setup_writes_only_the_switches_that_differ(self):
        apps = ("router", "escalate", "redact", "shield", "guard", "check")
        values = settings.parse_values(
            [f"{app}={app != 'shield'}" for app in apps]
            + ["logPath=~/.claude/vsr/decisions.jsonl"]
        )
        after = settings.merge({}, "vsr@inline", values)
        self.assertEqual(
            settings.saved_options(after, "vsr@inline"),
            {"shield": False, "check": True},
        )

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
                        json.loads(path.read_text()), "vsr@inline", {"guard": False}
                    )
                ),
            )
            self.assertEqual(json.loads(backup.read_text()), {"model": "opus"})
            self.assertEqual(
                json.loads(path.read_text())["pluginConfigs"]["vsr@inline"]["options"],
                {"guard": False},
            )

    def test_writes_through_a_symlink_and_keeps_every_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "dotfiles" / "settings.json"
            real.parent.mkdir()
            real.write_text("{}\n")
            link = Path(tmp) / "settings.json"
            link.symlink_to(real)
            first = settings.write(link, '{"a": 1}\n')
            second = settings.write(link, '{"a": 2}\n')
            self.assertTrue(link.is_symlink())
            self.assertEqual(json.loads(real.read_text()), {"a": 2})
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text()), {})
            self.assertEqual(json.loads(second.read_text()), {"a": 1})

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
        self.assertEqual(values["logPath"], "~/x.jsonl", "the mod places ~/ itself")

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
            ["runtimeUrl=localhost:8790"],
            ["routerStepsLow=2.1"],
            ["routerStepsMinConfidence=1.4"],
            ["calibrateEvery=-1"],
            ["calibrateEvery=1.5"],
        ):
            with self.subTest(bad=bad), self.assertRaises(settings.SettingsError):
                settings.parse_values(bad)

    def test_an_empty_log_path_turns_logging_off(self):
        self.assertEqual(settings.parse_value("logPath", ""), "")
        self.assertEqual(settings.parse_value("logPath", "  "), "")

    def test_a_relative_log_path_is_refused(self):
        with self.assertRaisesRegex(settings.SettingsError, "relative"):
            settings.parse_value("logPath", "logs/decisions.jsonl")
        self.assertEqual(
            settings.parse_value("logPath", "/var/x.jsonl"), "/var/x.jsonl"
        )

    def test_the_log_file_is_placed_as_the_mod_places_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"CLAUDE_CONFIG_DIR": tmp + "/", "HOME": "/home/me"}
            with mock.patch.dict(os.environ, env):
                self.assertEqual(
                    settings.log_file("~/.claude/vsr/decisions.jsonl"),
                    Path(tmp) / "vsr" / "decisions.jsonl",
                )
                self.assertEqual(
                    settings.log_file("~/logs/vsr.jsonl"),
                    Path("/home/me/logs/vsr.jsonl"),
                )
                self.assertEqual(
                    settings.log_file("/var/vsr.jsonl"), Path("/var/vsr.jsonl")
                )
                self.assertIsNone(settings.log_file("vsr.jsonl"))
                self.assertIsNone(settings.log_file(""))
            with mock.patch.dict(os.environ, {"HOME": "/home/me"}):
                os.environ.pop("CLAUDE_CONFIG_DIR", None)
                self.assertEqual(
                    settings.log_file("~/.claude/vsr/decisions.jsonl"),
                    Path("/home/me/.claude/vsr/decisions.jsonl"),
                )

    def test_zero_turns_the_calibration_reminder_off(self):
        self.assertEqual(settings.parse_value("calibrateEvery", "0"), 0)


class Order(unittest.TestCase):
    """The merged options keep each upper line at or above its lower line, as the mod does."""

    def test_equal_lines_are_accepted(self):
        for values in (
            {"routerEasyBelow": 1.0, "routerHardFrom": 1.0},
            {"routerStepsLow": 1.0, "routerStepsHigh": 1.0},
        ):
            with self.subTest(values=values):
                merged = settings.merge({}, "vsr@inline", values)
                self.assertEqual(settings.saved_options(merged, "vsr@inline"), values)

    def test_a_line_under_its_lower_line_is_rejected(self):
        for values in (
            {"routerEasyBelow": 1.6, "routerHardFrom": 1.5},
            {"routerStepsLow": 1.3, "routerStepsHigh": 0.8},
        ):
            with self.subTest(values=values), self.assertRaises(settings.SettingsError):
                settings.merge({}, "vsr@inline", values)

    def test_checks_against_saved_values_and_defaults(self):
        saved = settings.merge({}, "vsr@inline", {"routerHardFrom": 1.9})
        self.assertEqual(
            settings.effective_options(
                settings.merge(saved, "vsr@inline", {"routerEasyBelow": 1.7}),
                "vsr@inline",
            )["routerEasyBelow"],
            1.7,
        )
        with self.assertRaisesRegex(settings.SettingsError, "routerHardFrom"):
            settings.merge({}, "vsr@inline", {"routerEasyBelow": 1.7})


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

    def test_an_installed_copy_the_list_does_not_name_yet(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"CLAUDE_CONFIG_DIR": tmp}
        ):
            cache = Path(tmp) / "plugins" / "cache"
            listed = [
                {"id": "vsr@vllm-sr", "installPath": str(cache / "vllm-sr/vsr/0.1.1")}
            ]
            self.assertEqual(
                settings.resolve_id(listed, str(cache / "vllm-sr/vsr/0.1.2")),
                "vsr@vllm-sr",
                "an update moved the running copy",
            )
            self.assertEqual(
                settings.resolve_id([], str(cache / "other-market/vsr/0.2.0")),
                "vsr@other-market",
            )
            elsewhere = str(Path(tmp) / "plugins" / "synced" / "x")
            self.assertEqual(settings.resolve_id(listed, elsewhere), "vsr@vllm-sr")
            with self.assertRaisesRegex(settings.SettingsError, "pass --id"):
                settings.resolve_id(
                    [*listed, {"id": "vsr@fork", "installPath": "/f"}], elsewhere
                )
            self.assertEqual(
                settings.resolve_id(listed, "/home/me/sr-checkout"),
                "vsr@inline",
                "a --plugin-dir copy beside an installed one",
            )


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
                        ["plan", "--id", "vsr@inline", "--set", "check=true"]
                    ),
                    0,
                )
                self.assertEqual(json.loads(path.read_text()), {"theme": "dark"})
                self.assertEqual(
                    settings.main(
                        ["apply", "--id", "vsr@inline", "--set", "check=true"]
                    ),
                    0,
                )
                self.assertEqual(
                    json.loads(path.read_text()),
                    {
                        "theme": "dark",
                        "pluginConfigs": {"vsr@inline": {"options": {"check": True}}},
                    },
                )
            finally:
                del os.environ["CLAUDE_CONFIG_DIR"]

    def test_only_defaults_on_a_fresh_machine_writes_no_file(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {"CLAUDE_CONFIG_DIR": tmp}
        ):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = settings.main(
                    ["apply", "--id", "vsr@inline", "--set", "check=false"]
                )
            self.assertEqual(code, 0)
            self.assertIn("no change", out.getvalue())
            self.assertFalse((Path(tmp) / "settings.json").exists())

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
                        ["apply", "--id", "vsr@inline", "--set", "check=true"]
                    ),
                    0,
                )
                self.assertEqual(
                    json.loads(path.read_text())["pluginConfigs"],
                    {"vsr@inline": {"options": {"check": True}}},
                )
                self.assertEqual(oct(stat.S_IMODE(path.stat().st_mode)), oct(0o600))
                backups = list(Path(tmp).glob("settings.json.sr-backup-*"))
                self.assertEqual(
                    [oct(stat.S_IMODE(b.stat().st_mode)) for b in backups], [oct(0o600)]
                )
            finally:
                os.umask(umask)
                del os.environ["CLAUDE_CONFIG_DIR"]


class Manifest(unittest.TestCase):
    def test_the_ranges_name_manifest_options_of_the_right_type(self):
        config = settings.USER_CONFIG
        for name in (*settings.DIFFICULTY_LINES, *settings.WHOLE_NUMBERS):
            self.assertEqual(config[name]["type"], "number", name)
        for pair in settings.ORDERED:
            self.assertLessEqual(set(pair), set(settings.DIFFICULTY_LINES))
        for name in settings.PATHS:
            self.assertEqual(config[name]["type"], "string", name)

    def test_each_number_has_the_range_its_description_states(self):
        for name, spec in settings.USER_CONFIG.items():
            if spec["type"] != "number" or name in settings.WHOLE_NUMBERS:
                continue
            with self.subTest(option=name):
                upper = 2 if name in settings.DIFFICULTY_LINES else 1
                self.assertIn(f"(0 to {upper})", spec["description"])
                self.assertRaises(
                    settings.SettingsError, settings.parse_value, name, upper + 0.01
                )

    def test_every_default_is_a_value_the_parser_accepts(self):
        for name, default in settings.defaults().items():
            with self.subTest(option=name):
                parsed = settings.parse_value(name, default)
                if name not in settings.PATHS:
                    self.assertEqual(parsed, default)
        settings.check_order(settings.defaults())

    def test_the_token_is_never_saved_in_settings(self):
        self.assertNotIn("runtimeToken", settings.defaults())
        with self.assertRaisesRegex(settings.SettingsError, "/plugin configure"):
            settings.parse_value("runtimeToken", "abc")

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


if __name__ == "__main__":
    unittest.main()
