import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.htdev_iso import manifest


class ManifestEvalOnlyTest(unittest.TestCase):
    def test_qualification_files_get_an_evaluation_only_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m3a"
            (root / "qual" / "out").mkdir(parents=True)
            (root / "waves").mkdir()
            (root / "qual" / "qual.prompts.jsonl").write_text('{"id": "a"}\n')
            (root / "qual" / "out" / "p1.jsonl").write_text('{"id": "a"}\n')
            (root / "waves" / "w.prompts.jsonl").write_text(
                '{"id": "b"}\n{"id": "c"}\n'
            )
            out, corpora = Path(tmp) / "T.json", Path(tmp) / "C.json"
            manifest.main(
                [
                    "--label",
                    f"local-m3a={root}",
                    "--output",
                    str(out),
                    "--corpora",
                    str(corpora),
                    "--workers",
                    "1",
                ]
            )
            labels = json.loads(corpora.read_text())["labels"]
            self.assertEqual(labels["local-m3a"]["kind"], "training")
            self.assertEqual(
                [Path(f["path"]).name for f in labels["local-m3a"]["files"]],
                ["w.prompts.jsonl"],
            )
            moved = labels["local-m3a:evaluation-only"]
            self.assertEqual(moved["kind"], "evaluation-only")
            self.assertEqual(len(moved["files"]), 2)
            training = json.loads(out.read_text())["labels"]
            self.assertEqual(training["local-m3a"]["rows"], 2)
            self.assertEqual(training["local-m3a:evaluation-only"]["file_count"], 2)


if __name__ == "__main__":
    unittest.main()
