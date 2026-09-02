from __future__ import annotations

import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


class MainWorkflowContractTests(unittest.TestCase):
    def test_main_validation_runs_cannot_be_coalesced(self) -> None:
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github" / "workflows" / "main.yml").read_text(
                encoding="utf-8"
            )
        )

        self.assertNotIn(
            "concurrency",
            workflow,
            "main push runs must remain per-commit so change classification cannot skip a commit",
        )


if __name__ == "__main__":
    unittest.main()
