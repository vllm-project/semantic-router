import json
import tempfile
import unittest
from pathlib import Path

from evaluate import load_cases, ndcg


class EvaluationTest(unittest.TestCase):
    def test_perfect_reversed_and_no_relevant_documents(self):
        self.assertEqual(ndcg([3, 1, 0], [0, 1, 2], 2), 1)
        self.assertLess(ndcg([3, 1, 0], [2, 1, 0], 2), 0.1)
        self.assertIsNone(ndcg([0, 0], [0, 1], 2))

    def test_duplicate_candidate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            candidate = {"id": "same", "text": "text", "relevance": 1}
            path.write_text(
                json.dumps({"query": "q", "candidates": [candidate, candidate]})
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
