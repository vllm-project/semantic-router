import json
import unittest
from pathlib import Path

from report import render


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.data = (Path(__file__).parent / "testdata/report.mock.jsonl").read_bytes()
        self.records = [json.loads(line) for line in self.data.splitlines()]

    def encode(self):
        return "\n".join(json.dumps(r) for r in self.records).encode()

    def test_four_states(self):
        table = render(self.data, "mock")
        self.assertIn("MOCK 演示", table)
        self.assertIn(
            "| mock-001 | 可评分 | coding | coding | 通过 | 正确 | 1 |", table
        )
        self.assertIn("| mock-002 | 诊断 | — | coding | 通过 | 不评分 | 1 |", table)
        self.assertIn(
            "| mock-003 | 失败 | coding | coding | 未通过 | 不评分 | 1 |", table
        )
        self.assertIn(
            "| mock-004 | 未执行 | writing | — | 未检查 | 不评分 | 0 |", table
        )

    def test_valid_but_wrong(self):
        self.records[0].update(expected="writing", correct=False)
        self.assertIn(
            "| 可评分 | writing | coding | 通过 | 错误 |", render(self.encode(), "mock")
        )

    def test_reject_inconsistent_records(self):
        for index, change in [
            (0, {"schema": "unknown"}),
            (0, {"correct": False}),
            (1, {"correct": False}),
            (2, {"correct": True}),
            (3, {"contract_valid": False}),
            (3, {"attempts": 1}),
            (1, {"id": "mock-001"}),
            (1, {"revision": "different"}),
            (0, {"contract_valid": None}),
        ]:
            with self.subTest(change=change):
                records = [dict(r) for r in self.records]
                records[index].update(change)
                data = "\n".join(json.dumps(r) for r in records).encode()
                with self.assertRaises(ValueError):
                    render(data, "mock")

    def test_transport_failure_without_prediction(self):
        self.records[2].update(raw_response="not JSON", error_kind="status")
        self.assertIn(
            "| 失败 | coding | — | 未通过 | 不评分 |", render(self.encode(), "mock")
        )

    def test_escape_table_content(self):
        self.records[3]["reason"] = "<tag>|line\nnext"
        self.assertIn("&lt;tag&gt;&#124;line next", render(self.encode(), "mock"))

    def test_empty_or_malformed(self):
        for data in (b"", b"{", b"[]", b"null"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                render(data, "mock")


if __name__ == "__main__":
    unittest.main()
