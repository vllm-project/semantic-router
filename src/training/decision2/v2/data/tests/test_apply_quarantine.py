from __future__ import annotations

import unittest

from v2.data.apply_quarantine import apply, embed_groups, overlap_groups


class ApplyQuarantineTest(unittest.TestCase):
    def test_report_only_role_does_not_quarantine(self) -> None:
        receipt = {
            "groups": {
                "g1": {"roles": ["rights_clean_train"]},
                "g2": {"roles": ["rights_clean_train", "css15_goldfree"]},
            }
        }
        self.assertEqual(
            overlap_groups(receipt, {"rights_clean_train"}), {"g2": ["css15_goldfree"]}
        )
        self.assertEqual(set(overlap_groups(receipt, set())), {"g1", "g2"})

    def test_apply_removes_whole_groups(self) -> None:
        rows = [
            {"id": "a", "group_id": "g1", "source": "s"},
            {"id": "b", "group_id": "g1", "source": "s"},
            {"id": "c", "group_id": "g2", "source": "t"},
        ]
        quarantine = {"g1": ["typed_dev"], **embed_groups({"quarantined": []})}
        kept, report = apply(rows, quarantine)
        self.assertEqual([row["id"] for row in kept], ["c"])
        self.assertEqual(report["rows_removed"], 2)
        self.assertEqual(report["removed_groups_by_role"], {"typed_dev": 1})


if __name__ == "__main__":
    unittest.main()
