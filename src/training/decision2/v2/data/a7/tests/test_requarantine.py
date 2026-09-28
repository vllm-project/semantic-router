from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.data.a7 import requarantine
from v2.data.freeze import canonical_jsonl


def _row(index: int, group: str, family: str = "natural_snli") -> dict:
    return {
        "id": f"a7:x:{index}",
        "group_id": group,
        "family": family,
        "task_type": "choice",
    }


class RequarantineTest(unittest.TestCase):
    def test_flagged_groups_merge_methods_and_skip_report_only(self) -> None:
        overlap = {
            "groups": {
                "g1": {"roles": ["v1_aho_a3", "rights_clean_train"]},
                "g2": {"roles": ["rights_clean_train"]},
            }
        }
        embed = {
            "quarantined": [{"group_id": "g3", "protected_role": "css15_goldfree"}]
        }
        flagged = requarantine.flagged_groups(
            [overlap], [embed], {"rights_clean_train"}
        )
        self.assertEqual(sorted(flagged), ["g1", "g3"])
        self.assertEqual(flagged["g1"]["lexical"], {"v1_aho_a3"})
        self.assertEqual(flagged["g3"]["embedding"], {"css15_goldfree"})

    def test_groups_leave_every_sub_arm(self) -> None:
        parts = {
            "A7h": {"train": [_row(1, "g1"), _row(2, "g2")], "aho": [_row(3, "g9")]},
            "A7m": {"train": [_row(4, "g1", "multinli"), _row(5, "g5", "multinli")]},
        }
        flagged = {"g1": {"lexical": {"v1_aho_a3"}, "embedding": set()}}
        kept, report = requarantine.requarantine(parts, flagged)
        self.assertEqual([r["id"] for r in kept["A7h"]["train"]], ["a7:x:2"])
        self.assertEqual([r["id"] for r in kept["A7m"]["train"]], ["a7:x:5"])
        self.assertEqual(report["A7h"]["groups_removed"], 1)
        self.assertEqual(
            report["A7m"]["groups_removed_by_method_role"], {"lexical|v1_aho_a3": 1}
        )
        self.assertEqual(report["A7h"]["rows_out"], {"train": 1, "aho": 1})

    def test_cli_writes_new_run_with_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp, "v2"), Path(tmp, "v3")
            (src / "final" / "views").mkdir(parents=True)
            (src / "build").mkdir()
            (src / "screens").mkdir()
            rows = [_row(1, "g1"), _row(2, "g2")]
            (src / "final" / "A7h.train.jsonl").write_bytes(canonical_jsonl(rows))
            (src / "final" / "admission.json").write_text('{"sub_arms": {}}\n')
            (src / "build" / "build-manifest.json").write_text("{}\n")
            (src / "screens" / "A7h.overlap.public.json").write_text("{}\n")
            (src / "screens" / "A7h.overlap.private.json").write_text("{}\n")
            view = {
                "view": "v",
                "covered": 2,
                "members": [
                    {"id": "a7:x:1", "part": "train", "sub_arm": "A7h"},
                    {"id": "a7:x:2", "part": "train", "sub_arm": "A7h"},
                ],
            }
            (src / "final" / "views" / "v.json").write_text(json.dumps(view))
            receipt = Path(tmp, "embed.private.json")
            receipt.write_text(
                json.dumps({"quarantined": [{"group_id": "g2", "protected_role": "r"}]})
            )
            requarantine.main(
                [
                    "--from-run", str(src), "--embed-receipt", str(receipt),
                    "--version", "a7-test", "--out-run", str(out),
                ]
            )  # fmt: skip
            kept = (out / "final" / "A7h.train.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["id"] for line in kept], ["a7:x:1"])
            new_view = json.loads((out / "final" / "views" / "v.json").read_text())
            self.assertEqual(new_view["covered"], 1)
            self.assertEqual(new_view["removed_by_requarantine"], 1)
            admission = json.loads((out / "final" / "admission.json").read_text())
            self.assertEqual(admission["version"], "a7-test")
            self.assertEqual(admission["requarantine"]["flagged_groups"], 1)
            self.assertTrue((out / "screens" / "A7h.overlap.public.json").exists())
            self.assertFalse((out / "screens" / "A7h.overlap.private.json").exists())


if __name__ == "__main__":
    unittest.main()
