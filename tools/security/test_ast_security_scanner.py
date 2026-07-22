#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


SCANNER_PATH = Path(__file__).with_name("ast_security_scanner.py")
SPEC = importlib.util.spec_from_file_location("ast_security_scanner", SCANNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load scanner from {SCANNER_PATH}")
SCANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCANNER
SPEC.loader.exec_module(SCANNER)


class ASTSecurityScannerTests(unittest.TestCase):
    def test_own_source_detection_handles_mirrored_checkout(self):
        self.assertTrue(SCANNER._is_own_source(str(SCANNER_PATH)))
        self.assertTrue(
            SCANNER._is_own_source(
                "/tmp/pr-code/tools/security/ast_security_scanner.py"
            )
        )
        self.assertTrue(
            SCANNER._is_own_source("/tmp/pr-code/tools/security/scan_malicious_code.py")
        )
        self.assertFalse(SCANNER._is_own_source("/tmp/pr-code/src/application.py"))

    def test_metadata_service_patterns_keep_detection_values(self):
        self.assertEqual(SCANNER.METADATA_SERVICE_IP, "169.254.169.254")
        self.assertEqual(
            SCANNER.METADATA_SERVICE_HOST,
            "metadata.google.internal",
        )
        self.assertIn(
            SCANNER.METADATA_SERVICE_IP,
            SCANNER.HIGH_RISK_CREDENTIAL_STRINGS,
        )
        self.assertIn(
            SCANNER.METADATA_SERVICE_HOST,
            SCANNER.HIGH_RISK_CREDENTIAL_STRINGS,
        )

    def test_download_and_execute_is_still_high_severity(self):
        source = b"""
def install_payload(url):
    payload = requests.get(url)
    os.system(payload.text)
"""
        result = SCANNER.ScanResult()

        SCANNER.scan_file_ast("src/install_payload.py", source, result)

        findings = [
            finding
            for finding in result.findings
            if finding.category == "AST_DOWNLOAD_EXECUTE"
        ]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, SCANNER.Severity.HIGH)

    def test_download_without_execution_is_not_flagged_as_dropper(self):
        source = b"""
def fetch_payload(url):
    return requests.get(url)
"""
        result = SCANNER.ScanResult()

        SCANNER.scan_file_ast("src/fetch_payload.py", source, result)

        self.assertNotIn(
            "AST_DOWNLOAD_EXECUTE",
            {finding.category for finding in result.findings},
        )


if __name__ == "__main__":
    unittest.main()
