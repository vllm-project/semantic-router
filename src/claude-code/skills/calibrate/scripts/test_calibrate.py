import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import calibrate
import plan

GUARD_LABELS = [
    (0.80, "yes"),
    (0.82, "yes"),
    (0.85, "yes"),
    (0.92, "yes"),
    (0.95, "yes"),
    (0.97, "yes"),
    (0.10, "no"),
    (0.30, "no"),
    (0.76, "no"),
    (0.78, "no"),
]


class Sandbox(unittest.TestCase):
    """A settings file and a decision log in a temporary config directory; no network."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.settings = self.dir / "settings.json"
        self.log = self.dir / "vsr" / "decisions.jsonl"
        self.log.parent.mkdir()
        self.log.write_text(
            "".join(
                json.dumps(
                    {
                        "id": f"g{i}",
                        "at": 0,
                        "session": "s",
                        "app": "guard",
                        "verdict": "held" if s >= 0.9 else "allowed",
                        "score": s,
                        "threshold": 0.9,
                        "subject": f"cmd {i}",
                        "detail": {},
                        "learnable": True,
                    }
                )
                + "\n"
                for i, (s, _) in enumerate(GUARD_LABELS)
            )
        )
        self.save({"logPath": str(self.log)})
        for patch in (
            mock.patch.dict(
                os.environ, {"CLAUDE_CONFIG_DIR": tmp.name, "SR_RUNTIME_TOKEN": ""}
            ),
            mock.patch.object(plan, "runtime_model", return_value="vela3-lite"),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def save(self, options):
        self.settings.write_text(
            json.dumps(
                {"theme": "dark", "pluginConfigs": {"vsr@inline": {"options": options}}}
            )
        )

    def options(self):
        return json.loads(self.settings.read_text())["pluginConfigs"]["vsr@inline"][
            "options"
        ]

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = calibrate.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def write_plan(self, changes, plugin_id="vsr@inline"):
        path = self.dir / "plan.json"
        path.write_text(
            json.dumps({"pluginId": plugin_id, "goal": "balanced", "changes": changes})
        )
        return str(path)


class EndToEnd(Sandbox):
    def test_suggest_then_apply(self):
        pairs = [f"g{i}={label}" for i, (_, label) in enumerate(GUARD_LABELS)]
        self.assertEqual(self.run_cli("--id", "vsr@inline", "label", *pairs)[0], 0)
        plan_path = self.dir / "calibration-plan.json"
        self.assertEqual(
            self.run_cli("--id", "vsr@inline", "suggest", "--out", str(plan_path))[0],
            0,
        )
        saved_plan = json.loads(plan_path.read_text())
        self.assertEqual(
            saved_plan["changes"], {"guardReview": {"from": 0.9, "to": 0.8}}
        )
        self.assertNotIn("guardReview", self.options(), "suggest writes nothing")
        self.assertEqual(
            self.run_cli("--id", "vsr@inline", "apply", str(plan_path))[0], 0
        )
        saved = json.loads(self.settings.read_text())
        self.assertEqual(saved["theme"], "dark")
        self.assertEqual(self.options(), {"logPath": str(self.log), "guardReview": 0.8})
        record = json.loads(
            (self.log.parent / "calibrations.jsonl").read_text().splitlines()[-1]
        )
        self.assertEqual(record["changes"], {"guardReview": {"from": 0.9, "to": 0.8}})
        self.assertEqual(record["model"], "vela3-lite")

    def test_the_options_may_follow_the_subcommand(self):
        code, out, _ = self.run_cli("report", "--id", "vsr@inline")
        self.assertEqual(code, 0)
        self.assertIn("pluginConfigs['vsr@inline']", out)
        code, out, _ = self.run_cli("sample", "--id", "vsr@inline", "--apps", "guard")
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(out)), 4)

    def test_the_default_log_path_is_under_the_config_dir(self):
        self.save({})
        code, out, _ = self.run_cli("report", "--id", "vsr@inline")
        self.assertEqual(code, 0)
        self.assertIn(f"log {self.log}: 10 decisions", out)

    def test_label_writes_the_lines_the_pane_writes(self):
        self.assertEqual(
            self.run_cli("label", "--id", "vsr@inline", "g0=yes", "g6=no")[0], 0
        )
        lines = [
            json.loads(line)
            for line in (self.log.parent / "labels.jsonl").read_text().splitlines()
        ]
        self.assertEqual([sorted(x) for x in lines], [["app", "at", "id", "label"]] * 2)
        self.assertEqual(
            [(x["id"], x["app"], x["label"]) for x in lines],
            [("g0", "guard", "yes"), ("g6", "guard", "no")],
        )
        self.assertRegex(lines[0]["at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")

    def test_a_relative_log_path_is_logging_off(self):
        self.settings.write_text(
            json.dumps(
                {"pluginConfigs": {"vsr@inline": {"options": {"logPath": "x.jsonl"}}}}
            )
        )
        self.assertIn("keeps no log", self.run_cli("report", "--id", "vsr@inline")[2])

    def test_an_empty_log_path_is_logging_off(self):
        self.save({"logPath": ""})
        code, _, err = self.run_cli("--id", "vsr@inline", "report")
        self.assertEqual(code, 2)
        self.assertIn("keeps no log", err)


class Apply(Sandbox):
    """apply writes only what a calibration may change, however the plan file was made."""

    def assert_refused(self, changes, message, plugin_id="vsr@inline"):
        before = self.settings.read_text()
        code, _, err = self.run_cli(
            "--id", "vsr@inline", "apply", self.write_plan(changes, plugin_id)
        )
        self.assertEqual(code, 2)
        self.assertIn(message, err)
        self.assertEqual(self.settings.read_text(), before, "nothing written")

    def test_refuses_options_that_are_not_calibrated_thresholds(self):
        for name, value in (
            ("runtimeUrl", "https://elsewhere.example.com/v1/decisions"),
            ("shield", False),
            ("logPath", "/tmp/x.jsonl"),
        ):
            with self.subTest(option=name):
                self.assert_refused(
                    {name: {"from": None, "to": value}}, "a calibration plan sets only"
                )

    def test_refuses_a_move_larger_than_one_run_allows(self):
        self.assert_refused(
            {"guardReview": {"from": 0.9, "to": 0.01}}, "more than 0.15 in one run"
        )

    def test_refuses_a_plan_for_another_plugin_id(self):
        self.assert_refused(
            {"guardReview": {"from": 0.9, "to": 0.85}},
            "the plan is for 'vsr@vllm-sr'",
            plugin_id="vsr@vllm-sr",
        )

    def test_checks_the_line_order_of_the_merged_result(self):
        self.save({"logPath": str(self.log), "routerHardFrom": 0.6})
        self.assert_refused(
            {"routerEasyBelow": {"from": 0.5, "to": 0.65}},
            "routerHardFrom (0.6) must not be below routerEasyBelow (0.65)",
        )

    def test_a_value_back_at_its_default_clears_the_saved_one(self):
        self.save({"logPath": str(self.log), "guardReview": 0.95})
        code, _, _ = self.run_cli(
            "--id",
            "vsr@inline",
            "apply",
            self.write_plan({"guardReview": {"from": 0.95, "to": 0.9}}),
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.options(), {"logPath": str(self.log)})


class Probe(Sandbox):
    def test_sends_the_token_from_the_token_file(self):
        token = self.dir / "token"
        token.write_text("s3cret\n")
        token.chmod(0o600)
        with mock.patch.object(
            calibrate.runtime, "run_probes", return_value=[]
        ) as probes:
            code, _, _ = self.run_cli(
                "probe", "--id", "vsr@inline", "--token-file", str(token)
            )
        self.assertEqual(code, 0)
        self.assertEqual(probes.call_args.args[1], "s3cret")

    def test_refuses_a_token_file_others_can_read(self):
        token = self.dir / "token"
        token.write_text("s3cret\n")
        token.chmod(0o644)
        with mock.patch.object(calibrate.runtime, "run_probes") as probes:
            code, _, err = self.run_cli(
                "--id", "vsr@inline", "--token-file", str(token), "probe"
            )
        self.assertEqual(code, 2)
        self.assertIn("chmod 600", err)
        probes.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class WritesOnlyTheirOwnFiles(unittest.TestCase):
    def test_label_and_suggest_refuse_a_file_of_another_name(self):
        for given, default, name in [
            (
                "/home/me/.claude/settings.json",
                Path("/x/labels.jsonl"),
                calibrate.LABELS_NAME,
            ),
            ("/repo/src/app.py", Path("/x/calibration-plan.json"), calibrate.PLAN_NAME),
        ]:
            with self.assertRaises(calibrate.settings.SettingsError):
                calibrate.writable(given, default, name)

    def test_their_own_names_anywhere_are_fine(self):
        self.assertEqual(
            calibrate.writable(
                "/tmp/a/labels.jsonl", Path("/x/labels.jsonl"), calibrate.LABELS_NAME
            ),
            Path("/tmp/a/labels.jsonl"),
        )
        self.assertEqual(
            calibrate.writable(None, Path("/x/labels.jsonl"), calibrate.LABELS_NAME),
            Path("/x/labels.jsonl"),
        )
