#!/usr/bin/env python3
"""
Regression test for the scanner self-exclusion bug.

CI checks out the trusted scanner and the untrusted PR code into separate
roots (base/ vs pr-code/) so the scanner never executes fork-controlled
code. Before this fix, both scanners recognized "my own source" by an
absolute-path prefix derived from `__file__`, which only matches when the
scanner and the scanned tree share the same root. Run against a
differently-rooted checkout, the PR's own copy of these files tripped the
scanners' own credential/dropper detection rules — see #2628 follow-up.

Run directly: python3 tools/security/test_self_exclusion.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ast_security_scanner as ast_scanner
import scan_malicious_code as regex_scanner

# Roots a real checkout could plausibly use: the developer's own working
# copy, and the base/ and pr-code/ split CI uses under pull_request_target.
CANDIDATE_ROOTS = (
    "/repo",
    "/home/runner/work/semantic-router/semantic-router",
    "/home/runner/work/semantic-router/semantic-router/base",
    "/home/runner/work/semantic-router/semantic-router/pr-code",
)


class TestASTScannerSelfExclusion(unittest.TestCase):
    def test_own_source_files_excluded_under_any_root(self):
        for root in CANDIDATE_ROOTS:
            for name in ("ast_security_scanner.py", "scan_malicious_code.py"):
                path = f"{root}/tools/security/{name}"
                with self.subTest(path=path):
                    self.assertTrue(
                        ast_scanner._is_own_source(path),
                        f"expected {path} to be treated as the scanner's own source",
                    )

    def test_files_outside_tools_security_are_not_excluded(self):
        self.assertFalse(ast_scanner._is_own_source("/repo/pkg/extproc/recorder.go"))
        self.assertFalse(ast_scanner._is_own_source("/repo/some_module/bad.py"))

    def test_unknown_files_in_own_directory_are_not_excluded(self):
        for root in CANDIDATE_ROOTS:
            path = f"{root}/tools/security/backdoor.py"
            with self.subTest(path=path):
                self.assertFalse(
                    ast_scanner._is_own_source(path),
                    f"expected {path} to still be scanned, not treated as self",
                )


class TestRegexScannerSelfExclusion(unittest.TestCase):
    def test_own_source_files_excluded_under_any_root(self):
        for root in CANDIDATE_ROOTS:
            for name in ("ast_security_scanner.py", "scan_malicious_code.py"):
                path = Path(f"{root}/tools/security/{name}")
                with self.subTest(path=str(path)):
                    self.assertTrue(
                        regex_scanner._is_self_path(path),
                        f"expected {path} to be treated as the scanner's own source",
                    )

    def test_files_outside_tools_security_are_not_excluded(self):
        self.assertFalse(
            regex_scanner._is_self_path(Path("/repo/pkg/extproc/recorder.go"))
        )
        self.assertFalse(regex_scanner._is_self_path(Path("/repo/some_module/bad.py")))

    def test_unknown_files_in_own_directory_are_not_excluded(self):
        for root in CANDIDATE_ROOTS:
            path = Path(f"{root}/tools/security/backdoor.py")
            with self.subTest(path=str(path)):
                self.assertFalse(
                    regex_scanner._is_self_path(path),
                    f"expected {path} to still be scanned, not treated as self",
                )


if __name__ == "__main__":
    unittest.main()
