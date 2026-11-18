"""A `claude plugin test` report counts as evidence only when it is whole."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow_evidence import plugin_test_cases

REPORT = """\
hooks/guard.test.tsx:
(pass) a risky command is held [12.30ms]
(skip) a held command waits for the person

hooks/view.test.tsx:
(pass) a risky command is held [8.00ms]
(fail) the file ran to its end [0.00ms]
  a rejection nothing handled: no implementation for tool.call

 2 pass
 1 fail
Ran 4 tests across 2 files. [1.20s]
"""


class PluginTestEvidenceTests(unittest.TestCase):
    def cases(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plugin-tests.log"
            path.write_text(text)
            return plugin_test_cases(path)

    def test_each_case_is_named_by_its_file_and_keeps_its_status(self):
        self.assertEqual(
            self.cases(REPORT),
            [
                {
                    "id": "hooks/guard.test.tsx::a risky command is held",
                    "status": "passed",
                },
                {
                    "id": "hooks/guard.test.tsx::a held command waits for the person",
                    "status": "skipped",
                },
                {
                    "id": "hooks/view.test.tsx::a risky command is held",
                    "status": "passed",
                },
                {
                    "id": "hooks/view.test.tsx::the file ran to its end",
                    "status": "failed",
                },
            ],
        )

    def test_colored_output_reads_the_same(self):
        colored = REPORT.replace("(pass)", "\x1b[32m(pass)\x1b[0m")
        self.assertEqual(self.cases(colored), self.cases(REPORT))

    def test_a_report_short_of_its_own_total_is_refused(self):
        with self.assertRaisesRegex(ValueError, "incomplete plugin test report"):
            self.cases(REPORT.replace("Ran 4 tests", "Ran 5 tests"))
        with self.assertRaisesRegex(ValueError, "incomplete plugin test report"):
            self.cases(REPORT.rsplit("Ran", 1)[0])


if __name__ == "__main__":
    unittest.main()
