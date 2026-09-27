from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from external_index021.__main__ import main, parser
from external_index021.selection import Selection


class CLITests(unittest.TestCase):
    def test_published_parity_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            main(["published-parity"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["rows"], 68)
        self.assertLess(result["max_display_delta"], 0.01)

    def test_verify_suite_does_not_require_output_path_or_gpu(self) -> None:
        args = parser().parse_args(["verify-suite", "--suite-dir", "/private/suite"])
        self.assertEqual(args.command, "verify-suite")
        selection = Selection(
            frozenset({"one"}),
            {
                "toolret": 315,
                "bright": 330,
                "home_dev_overlap": 48,
                "home_duplicate": 24,
            },
            151034,
            150317,
            "first",
            "provisional_home_copy",
            "a" * 64,
        )
        output = io.StringIO()
        with patch(
            "external_index021.score.verified_suite", return_value=object()
        ), patch(
            "external_index021.score.selected_rows", return_value=([], selection)
        ), redirect_stdout(
            output
        ):
            main(["verify-suite", "--suite-dir", "/private/suite"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["scoreable"], 150317)
        self.assertEqual(result["status"], "provisional_home_copy")
        self.assertNotIn("one", output.getvalue())


if __name__ == "__main__":
    unittest.main()
