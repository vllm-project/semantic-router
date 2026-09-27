"""Post-key figures must compare reports with the frozen source map."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import generate_postkey_artifacts_v3 as postkey


class PostkeyArtifactTest(unittest.TestCase):
    def test_frozen_source_map_is_used_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rank = {"schema_version": "jevarena-ranking/3", "models": []}
            (root / "rank.json").write_text(json.dumps(rank))
            (root / "config.json").write_text(json.dumps({"arena_rank": "rank.json"}))
            (root / "diagnostic.json").write_text(
                json.dumps(
                    {
                        "schema_version": "jevarena-v3-postkey-answer-count-diagnostic/1",
                        "status": "postkey_diagnostic_not_preregistered_release_gate",
                        "ranked_result": rank,
                    }
                )
            )
            original = postkey.artifacts.SCORER_SOURCE_PATHS
            frozen_map = {"arena_v3": root / "frozen.py"}

            def observed(_config: Path, _output: Path) -> dict:
                self.assertEqual(postkey.artifacts.SCORER_SOURCE_PATHS, frozen_map)
                return {"publication_version": "v3"}

            with (
                patch.object(
                    postkey,
                    "_frozen_ranker",
                    return_value=SimpleNamespace(SCORER_SOURCE_PATHS=frozen_map),
                ),
                patch.object(postkey.artifacts, "generate", side_effect=observed),
            ):
                result = postkey.generate(
                    root, root / "config.json", root / "diagnostic.json", root / "out"
                )
            self.assertEqual(result["publication_version"], "v3")
            self.assertIs(postkey.artifacts.SCORER_SOURCE_PATHS, original)


if __name__ == "__main__":
    unittest.main()
