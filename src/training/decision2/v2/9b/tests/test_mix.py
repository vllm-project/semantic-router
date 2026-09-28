import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from lux9b.mix import content_sha256, substitute  # noqa: E402


def rows(prefix, groups, size, kind="choice"):
    return [
        {"id": f"{prefix}{g}-{i}", "group_id": f"{prefix}g{g}", "task_type": kind}
        for g in range(groups)
        for i in range(size)
    ]


class SubstituteTest(unittest.TestCase):
    def test_whole_groups_and_matched_rows(self):
        a0 = rows("a", 40, 2)
        arm = rows("x", 30, 3, "score")
        mixed, stats = substitute(a0, arm, 0.25, 7)
        self.assertEqual(stats["removed_rows"], 20)
        self.assertGreaterEqual(stats["added_rows"], 20)
        self.assertLess(stats["added_rows"] - stats["removed_rows"], 3)
        kept_groups = {r["group_id"] for r in mixed if r["id"].startswith("a")}
        for group in kept_groups:
            self.assertEqual(sum(r["group_id"] == group for r in mixed), 2)
        self.assertEqual(len({r["id"] for r in mixed}), len(mixed))

    def test_deterministic_and_rejects_shared_groups(self):
        a0, arm = rows("a", 20, 1), rows("x", 20, 1)
        self.assertEqual(substitute(a0, arm, 0.25, 3), substitute(a0, arm, 0.25, 3))
        with self.assertRaises(ValueError):
            substitute(a0, rows("a", 5, 1), 0.25, 3)

    def test_content_hash_ignores_order(self):
        data = rows("a", 3, 2)
        self.assertEqual(content_sha256(data), content_sha256(list(reversed(data))))


if __name__ == "__main__":
    unittest.main()
