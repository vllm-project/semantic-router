from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from public_docs_paths import iter_public_document_paths  # noqa: E402
from public_docs_security import (  # noqa: E402
    find_credential_pairs,
    scan_public_documentation,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


class PublicDocumentationSecurityTests(unittest.TestCase):
    def test_repository_public_docs_do_not_publish_credential_pairs(self) -> None:
        self.assertEqual(scan_public_documentation(REPO_ROOT), [])

    def test_list_and_inline_aliases_are_rejected(self) -> None:
        cases = (
            "<details><summary>Playground access</summary>\n"
            "- **Username:** `regression-user`\n"
            "- **Password:** `regression-only-token-8391`\n"
            "</details>\n",
            "Email: qa-account@internal.local; Pass: regression-only-token-8391\n",
            "Login = regression-user\nPassword = regression-only-token-8391\n",
            "DASHBOARD_ADMIN_EMAIL=qa-account@internal.local\n"
            "DASHBOARD_ADMIN_PASSWORD=regression-only-token-8391\n",
        )
        for document in cases:
            with self.subTest(document=document):
                self.assertTrue(find_credential_pairs(document))

    def test_markdown_table_layouts_are_rejected(self) -> None:
        column_table = """\
| Login | Password |
| --- | --- |
| regression-user | regression-only-token-8391 |
"""
        key_value_table = """\
| Field | Value |
| --- | --- |
| Email | qa-account@internal.local |
| Pass | regression-only-token-8391 |
"""
        self.assertTrue(find_credential_pairs(column_table))
        self.assertTrue(find_credential_pairs(key_value_table))

    def test_html_layouts_are_rejected(self) -> None:
        inline_html = """\
<ul>
  <li><strong>Username:</strong> regression-user</li>
  <li><strong>Password:</strong> regression-only-token-8391</li>
</ul>
"""
        table_html = """\
<table><tr><th>Login</th><th>Pass</th></tr>
<tr><td>regression-user</td><td>regression-only-token-8391</td></tr></table>
"""
        definition_html = """\
<dl><dt>Email</dt><dd>qa-account@internal.local</dd>
<dt>Password</dt><dd>regression-only-token-8391</dd></dl>
"""
        for document in (inline_html, table_html, definition_html):
            with self.subTest(document=document):
                self.assertTrue(find_credential_pairs(document))

    def test_explicit_placeholders_and_security_guidance_are_allowed(self) -> None:
        documents = (
            "Username: <username>\nPassword: <password>\n",
            "Email: ${DASHBOARD_EMAIL}\nPass: ${DASHBOARD_PASSWORD}\n",
            'Username: "${DASHBOARD_USERNAME:?set DASHBOARD_USERNAME}"\n'
            'Password: "${DASHBOARD_PASSWORD:?set DASHBOARD_PASSWORD}"\n',
            "Login: YOUR_LOGIN\nPassword: replace-me\n",
            "Username: REDACTED\nPassword: MASKED\n",
            "Never publish a username/password pair. Use a password manager.",
            "Username: identify the account\nPassword: use a unique credential\n",
        )
        for document in documents:
            with self.subTest(document=document):
                self.assertEqual(find_credential_pairs(document), [])

    def test_distant_fields_and_separate_sections_are_not_paired(self) -> None:
        distant = (
            "Username: regression-user\n"
            + ("context\n" * 13)
            + ("Password: regression-only-token-8391\n")
        )
        separate_sections = """\
## Identity
Username: regression-user
## Password policy
Password: regression-only-token-8391
"""
        self.assertEqual(find_credential_pairs(distant), [])
        self.assertEqual(find_credential_pairs(separate_sections), [])

    def test_scans_source_docs_but_excludes_fixture_and_derived_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            disclosure = (
                "Username: regression-user\nPassword: regression-only-token-8391\n"
            )
            for relative_path in (
                "README.md",
                "docs/guide.mdx",
                "website/page.html",
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(disclosure, encoding="utf-8")
            for relative_path in (
                "tests/fixtures/example.md",
                "website/build/page.html",
                ".agent-harness/private.md",
                "private/operations.mdx",
            ):
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(disclosure, encoding="utf-8")

            paths = {
                path.relative_to(root) for path in iter_public_document_paths(root)
            }
            self.assertEqual(
                paths,
                {
                    Path("README.md"),
                    Path("docs/guide.mdx"),
                    Path("private/operations.mdx"),
                    Path("website/page.html"),
                },
            )
            findings = scan_public_documentation(root)
            self.assertEqual(len(findings), 4)
            self.assertEqual(
                {finding.path for finding in findings},
                paths,
            )

    def test_placeholder_words_inside_real_looking_values_do_not_bypass_gate(
        self,
    ) -> None:
        document = (
            "Login: customer-example-admin\n"
            "Password: not-sample-production-token-8391\n"
        )

        self.assertTrue(find_credential_pairs(document))

    def test_service_accepted_long_passwords_do_not_bypass_gate(self) -> None:
        for length in (129, 1024):
            with self.subTest(length=length):
                password = ("Ab3!" * ((length + 3) // 4))[:length]
                document = "Login: regression-user\nPassword: '" + password + "'\n"
                self.assertTrue(find_credential_pairs(document))

        self.assertTrue(
            find_credential_pairs(
                "Login: regression-user\nPassword: " + ("x" * 129) + "\n"
            )
        )
        self.assertTrue(
            find_credential_pairs(
                'Login: regression-user\nPassword: "' + ("*" * 16) + '"\n'
            )
        )

    def test_example_domain_and_placeholder_prefix_collisions_are_rejected(
        self,
    ) -> None:
        documents = (
            "Email: admin@example.com\nPassword: real-production-secret-8391\n",
            "Login: regression-user\nPassword: sample-real-production-secret-8391\n",
            "Login: regression-user\nPassword: your-real-production-secret-8391\n",
        )
        for document in documents:
            with self.subTest(document=document):
                self.assertTrue(find_credential_pairs(document))

    def test_schema_type_rows_are_not_treated_as_credentials(self) -> None:
        self.assertEqual(
            find_credential_pairs("username: string\npassword: string\n"),
            [],
        )

    def test_markdown_label_and_value_markup_is_parsed_without_losing_secrets(
        self,
    ) -> None:
        self.assertTrue(
            find_credential_pairs(
                "**Username**: `regression-user`\n"
                "**Password**: **real-production-secret-8391**\n"
            )
        )

    def test_dense_repeated_fields_collapse_to_location_pairs(self) -> None:
        document = (
            ("Login: user " * 2_000)
            + "\n"
            + ("Password: production-secret-8391 " * 2_000)
            + "\n"
        )
        self.assertEqual(find_credential_pairs(document), [(1, 2)])

    def test_findings_never_contain_credential_values(self) -> None:
        document = "Login: regression-user\nPassword: regression-only-token-8391\n"
        finding = find_credential_pairs(document)[0]

        self.assertEqual(finding, (1, 2))
        self.assertNotIn("regression", repr(finding))


if __name__ == "__main__":
    unittest.main()
