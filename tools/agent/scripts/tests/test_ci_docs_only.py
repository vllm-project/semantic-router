import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ci_docs_only import HEAVY_CHANGE_KEYS, LIGHT_CHANGE_KEYS, is_docs_only  # noqa: E402


def _changes(**overrides: bool) -> dict[str, bool]:
    base = {key: False for key in (*LIGHT_CHANGE_KEYS, *HEAVY_CHANGE_KEYS)}
    base.update(overrides)
    return base


class DocsOnlyClassificationTests(unittest.TestCase):
    def test_website_only_is_docs_only(self) -> None:
        self.assertTrue(is_docs_only(_changes(website=True)))

    def test_docs_only_markdown_without_core(self) -> None:
        self.assertTrue(is_docs_only(_changes(docs=True)))

    def test_agent_text_only_is_docs_only(self) -> None:
        self.assertTrue(is_docs_only(_changes(agent_text=True)))

    def test_mixed_website_and_core_is_not_docs_only(self) -> None:
        self.assertFalse(is_docs_only(_changes(website=True, core=True)))

    def test_operator_only_is_not_docs_only(self) -> None:
        self.assertFalse(is_docs_only(_changes(operator=True)))

    def test_no_matching_paths_is_not_docs_only(self) -> None:
        self.assertFalse(is_docs_only(_changes()))


if __name__ == "__main__":
    unittest.main()
