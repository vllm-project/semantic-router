from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import normalized, validate
from v2.eval.sealed.sources import stance_it as source

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
HEADER = ["text_clean", "target", "topic", "likes", "label"]
GOV = ("Governo", "Emergenza Sicurezza")
OPP = ("Opposizione", "Emergenza Sud Italia")


def xlsx(path: Path, rows: list[list], shared: bool) -> None:
    """Minimal workbook: shared strings as two rich-text runs, or inline strings."""
    strings: list[str] = []

    def cell(ref: str, value) -> str:
        if isinstance(value, int):
            return f'<c r="{ref}"><v>{value}</v></c>'
        if shared:
            strings.append(value)
            return f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'

    body = "".join(
        f'<row r="{r}">'
        + "".join(
            cell(f"{chr(65 + c)}{r}", v) for c, v in enumerate(values) if v is not None
        )
        + "</row>"
        for r, values in enumerate([HEADER] + rows, start=1)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{MAIN}" xmlns:r="{DOC_REL}"><sheets>'
            '<sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{PKG_REL}"><Relationship Id="rId1" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{MAIN}"><sheetData>{body}</sheetData></worksheet>',
        )
        if shared:
            items = "".join(
                f"<si><r><t>{escape(s[: len(s) // 2])}</t></r>"
                f'<r><t xml:space="preserve">{escape(s[len(s) // 2 :])}</t></r></si>'
                for s in strings
            )
            archive.writestr(
                "xl/sharedStrings.xml", f'<sst xmlns="{MAIN}">{items}</sst>'
            )


def text_group(text: str) -> str:
    return "text:" + hashlib.sha256(normalized(text).encode()).hexdigest()[:16]


class StanceItTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        processed = self.root / "processed"
        xlsx(
            processed / "GOV_post_1_emergenza_sicurezza_READY.xlsx",
            [["Brava presidente", *GOV, 3, None], ["Applausi", *GOV, 0, None]]
            + [["Che vergogna", *GOV, 1, None]],
            shared=True,
        )
        xlsx(
            processed / "GOV_post_2_emergenza_sicurezza_READY.xlsx",
            [["Applausi", *GOV, 0, None], ["Dimettiti", *GOV, 5, None]],
            shared=False,
        )
        xlsx(
            processed / "OPP_post_1_emergenza_sud_italia_READY.xlsx",
            [["Forza presidente", *OPP, 2, None]],
            shared=True,
        )
        xlsx(
            self.root / "labeling" / "train_to_label.xlsx",
            [
                ["Brava presidente", *GOV, 3, "Consenso"],
                ["Applausi", *GOV, 0, "Consenso"],
                ["Che vergogna", *GOV, 1, "Dissenso"],
                ["  brava   Presidente ", *GOV, 0, "Consenso"],
                ["Commento senza etichetta", *GOV, 0, None],
            ],
            shared=True,
        )
        xlsx(
            self.root / "labeling" / "test_to_label.xlsx",
            [
                ["Forza presidente", *OPP, 2, "Altro"],
                ["Applausi", *OPP, 0, "Consenso"],
                ["Dimettiti", *GOV, 5, "Dissenso"],
            ],
            shared=False,
        )
        self.out = {c.source_item_id: c for c in source.candidates(self.root)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_rows_labels_and_filters(self):
        self.assertEqual(
            sorted(self.out),
            ["test-000", "test-001", "test-002", "train-000", "train-001", "train-002"],
        )
        gold = {k: c.gold for k, c in self.out.items()}
        self.assertEqual(
            gold,
            {
                "train-000": "agree",
                "train-001": "agree",
                "train-002": "disagree",
                "test-000": "other",
                "test-001": "agree",
                "test-002": "disagree",
            },
        )
        for c in self.out.values():
            self.assertEqual(validate(c, source.SPEC), [])
            self.assertEqual(c.balance_label, c.gold)
            self.assertEqual(
                list(c.question["criteria"]), ["agree", "disagree", "other"]
            )

    def test_state_is_label_blind(self):
        c = self.out["train-000"]
        self.assertEqual(
            c.state,
            {
                "target": "Governo",
                "topic": "Emergenza Sicurezza",
                "comment": "Brava presidente",
            },
        )
        self.assertEqual(c.overlap_texts, ["Brava presidente"])
        for c in self.out.values():
            blob = json.dumps([c.state, c.question], ensure_ascii=False)
            for word in ("Consenso", "Dissenso", "Altro", "likes"):
                self.assertNotIn(word, blob)

    def test_groups_follow_posts(self):
        self.assertEqual(
            self.out["train-000"].group_id, "post:GOV_post_1_emergenza_sicurezza"
        )
        self.assertEqual(
            self.out["train-002"].group_id, "post:GOV_post_1_emergenza_sicurezza"
        )
        self.assertEqual(
            self.out["test-002"].group_id, "post:GOV_post_2_emergenza_sicurezza"
        )
        self.assertEqual(
            self.out["test-000"].group_id, "post:OPP_post_1_emergenza_sud_italia"
        )
        self.assertEqual(self.out["train-001"].group_id, text_group("Applausi"))
        self.assertEqual(self.out["test-001"].group_id, text_group("Applausi"))

    def test_unknown_label_fails_loudly(self):
        xlsx(
            self.root / "labeling" / "test_to_label.xlsx",
            [["Nuovo commento", *GOV, 0, "Favorevole"]],
            shared=False,
        )
        with self.assertRaises(KeyError):
            list(source.candidates(self.root))

    def test_runner_counts(self):
        receipt = runner.run("stance_it", self.root, None)
        self.assertEqual(receipt["invalid"], {})
        task = receipt["tasks"]["stance_it/stance"]
        self.assertEqual(
            (task["label:agree"], task["label:disagree"], task["label:other"]),
            (3, 2, 1),
        )
        self.assertEqual(task["groups"], 4)


if __name__ == "__main__":
    unittest.main()
