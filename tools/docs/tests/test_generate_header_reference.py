"""The generated header reference follows the Router's header registry export."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_header_reference as reference


class HeaderReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.export = json.loads(reference.SOURCE.read_text(encoding="utf-8"))

    def render(self, export=None):
        return reference.render(export or self.export)

    def test_every_registered_header_and_trace_member_has_one_row(self):
        page = self.render()
        for header in self.export["headers"]:
            rows = re.findall(rf"^\| `{re.escape(header['name'])}` \|", page, re.M)
            self.assertEqual(len(rows), 1, header["name"])
        for member in self.export["trace_members"]:
            row = rf"^\| `{re.escape(member['key'])}` \| {member['owner']} \|"
            self.assertEqual(len(re.findall(row, page, re.M)), 1, member["key"])
        for code in self.export["warning_codes"]:
            self.assertIn(f"| `{code['code']}` | {code['doc']} |", page)

    def test_credentials_appear_by_name_only(self):
        page = self.render()
        secrets = [h for h in self.export["headers"] if h["sensitivity"] == "secret"]
        self.assertTrue(secrets)
        for header in secrets:
            self.assertNotIn(header["doc"], page, header["name"])
            self.assertIn(
                f"| `{header['name']}` | {', '.join(header['direction'])} |", page
            )

    def test_every_debug_mode_says_who_gets_the_trace(self):
        page = self.render()
        for mode in self.export["debug_modes"]:
            self.assertRegex(page, rf"(?m)^\| `{mode}`( \(default\))? \| ", mode)

    def test_unknown_contract_terms_fail_rendering(self):
        for change in (
            lambda export: export["headers"][0].update(surface="partner"),
            lambda export: export["headers"][-1].update(sensitivity="token"),
            lambda export: export["debug_modes"].append("operators"),
            lambda export: export["trace_members"].pop(0),
        ):
            export = copy.deepcopy(self.export)
            change(export)
            with self.assertRaises(reference.ReferenceError):
                self.render(export)

    def test_prose_names_only_registered_headers(self):
        export = copy.deepcopy(self.export)
        export["headers"] = [
            h
            for h in export["headers"]
            if h["name"] != "x-vsr-bench-expected-config-hash"
        ]
        with self.assertRaisesRegex(
            reference.ReferenceError, "x-vsr-bench-expected-config-hash"
        ):
            self.render(export)

    def test_stale_reference_fails_check_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "headers.md"
            output.write_text("stale\n", encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(
                io.StringIO()
            ):
                status = reference.main(["--check", "--output", str(output)])
            self.assertEqual(status, 1)
            self.assertIn("make headers-generate", stderr.getvalue())
            self.assertEqual(output.read_text(encoding="utf-8"), "stale\n")
            missing = Path(directory) / "missing.md"
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(
                    reference.main(["--check", "--output", str(missing)]), 1
                )
            self.assertFalse(missing.exists())

    def test_repository_reference_is_current(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reference.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
