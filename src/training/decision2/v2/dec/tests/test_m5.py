"""CPU tests for the decoder Milestone 5 block / MLX-DEV / label tools (synthetic fixtures)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.model.data import INPUT_FIELDS, digest

from v2.dec import m5_block, m5_labels, mlx_dev

LONG = "a passage sentence that is long enough to count"


def row(i, group, lang, kind, source="src", state=None, label=1):
    keys = {
        "noul": ["false", "true"],
        "score": ["0", "1", "2", "3"],
        "choice": ["a", "b"],
    }[kind]
    r = {
        "id": f"r{i}",
        "state": (
            state
            if state is not None
            else f"Unique content number {i} of group {group}\nTemplate header line shared"
        ),
        "instructions": "Rate the overall sentiment that the message expresses.",
        "options": [{"key": k, "description": f"level {k} description"} for k in keys],
        "label": label % len(keys),
        "task_type": kind,
        "family": "f",
        "group_id": group,
        "language": lang,
        "split": "train",
        "source": source,
        "evaluation_role": "train",
        "render_template": "t",
        "audit_metadata": {},
    }
    r["input_sha256"] = digest({f: r[f] for f in INPUT_FIELDS})
    return r


def line(r):
    return json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"


def mix(components):
    """[(name, [rows])] -> (lines with parsed rows, manifest)."""
    items, manifest = [], {"component_order": [], "components": {}}
    for name, rows in components:
        manifest["component_order"].append(name)
        manifest["components"][name] = {
            "rows": len(rows),
            "tokens": sum(len(r["state"]) for r in rows),
        }
        items += [(line(r), r) for r in rows]
    return items, manifest


def lengths(rows):
    return {r["id"]: len(r["state"]) for r in rows}


class SegmentTest(unittest.TestCase):
    def test_normalization_and_threshold(self):
        a = row(
            1, "g", "ja", "noul", state="Ｈｅｌｌｏ   World this is long enough\nshort"
        )
        b = row(2, "g", "ja", "noul", state="hello world THIS is long   enough")
        self.assertTrue(m5_block.segments(a) & m5_block.segments(b))
        self.assertEqual(
            len(m5_block.segments(row(3, "g", "ja", "noul", state="tiny"))), 3
        )


class BlockTest(unittest.TestCase):
    def build(self, template_max_groups):
        n4 = [
            ("A0s", [row(0, "e0", "en", "noul"), row(1, "e1", "en", "choice")]),
            ("A7k", [row(10, "k1", "ko", "score")]),
            (
                "H5",
                [
                    row(20 + i, f"h{i // 2}", "ja" if i < 8 else "de", "noul", label=i)
                    for i in range(12)
                ]
                + [row(40, "m1", "de", "choice")],
            ),
            (
                "H8",
                [
                    row(50, "j1", "ja", "choice", "jglue_jcommonsenseqa"),
                    row(51, "s1", "es-en", "score", "sentimix_spanglish"),
                    row(52, "t1", "en", "noul", "tydiqa"),
                ],
            ),
        ]
        items, manifest = mix(n4)
        extra = {
            "A7k": [row(100 + i, f"k{i + 2}", "ko", "score") for i in range(6)],
            "H5": [row(200 + i, f"H{i // 2}", "ja", "noul", label=i) for i in range(10)]
            + [row(260 + i, f"M{i}", "fr", "choice", "mtop") for i in range(6)],
            "H8": [
                row(300 + i, f"J{i}", "ja", "choice", "jglue_jcommonsenseqa")
                for i in range(6)
            ]
            + [
                row(320 + i, f"S{i}", "es-en", "score", "sentimix_spanglish")
                for i in range(6)
            ]
            + [
                row(340 + i, f"Q{i // 2}", "fa", "noul", "miracl_v1.0_train", label=i)
                for i in range(6)
            ],
            "A7q": [row(400 + i, f"q{i}", "es", "score") for i in range(8)],
            "A7s": [row(500 + i, f"a{i}", "hi-en", "score") for i in range(8)],
        }
        # A leaking group: shares a content line with an N4XF row.
        extra["H5"].append(
            row(
                290,
                "leak",
                "ja",
                "noul",
                state=n4[2][1][0]["state"].split("\n")[0] + "\nx",
            )
        )
        superset = []
        for name in m5_block.BLOCK_COMPONENTS:
            base = [r for n, rows in n4 if n == name for r in rows]
            superset.append((name, base + extra.get(name, [])))
        s_items, s_manifest = mix(superset)
        return m5_block.Block(
            items,
            manifest,
            s_items,
            s_manifest,
            {"excluded"},
            lengths,
            template_max_groups,
        )

    def test_literal_rule_drops_template_sharing_groups(self):
        block = self.build(None)
        selection = block.mlx_dev()
        self.assertEqual(sum(len(v) for v in selection.values()), 0)

    def test_template_rule_holdout_and_mixture(self):
        small = {k: v[:4] + (min(v[4], 2),) for k, v in m5_block.MLX_CELLS.items()}
        with mock.patch.dict(m5_block.MLX_CELLS, small):
            block = self.build(2)
            selection = block.mlx_dev()
        self.assertNotIn("leak", selection["h5-noul"])
        self.assertTrue(selection["h5-noul"])
        self.assertTrue(selection["h8-miracl-noul"])
        block.compose()
        lines, manifest = block.mixture()
        checks = block.checks(lines, manifest)
        self.assertTrue(checks["outside_block_identical"])
        self.assertEqual(checks["mlx_groups_in_n5b"], 0)
        self.assertEqual(checks["mlx_segment_conflicts_n5b_rows"], 0)
        self.assertEqual(
            checks["n4xf_block_rows_rendered_identically_in_superset"],
            checks["n4xf_block_rows"],
        )
        rows = [json.loads(x) for x in lines]
        ids = {r["id"] for r in rows}
        self.assertIn("r0", ids)
        self.assertIn("r52", ids)  # English H8 stays
        self.assertLess(
            block.stats["block"]["cells"]["noul"]["tokens"],
            block.stats["block"]["cells"]["noul"]["n4xf_tokens"],
        )
        self.assertGreater(block.stats["block"]["cells"]["a7q"]["added_groups"], 0)
        order = [
            name
            for name, _ in m5_block.component_slices(
                [(x, json.loads(x)) for x in lines], manifest
            )
        ]
        self.assertEqual(order, ["A0s", "A7k", "H5", "H8"])


class LabelBatchTest(unittest.TestCase):
    def test_whole_batch_sample_rebatches_identically(self):
        lengths_ = [((i * 7919) % 900) + 20 for i in range(400)]
        ids = [f"x{i}" for i in range(400)]
        chosen = m5_labels.whole_batch_sample(ids, lengths_, 60, "seed")
        self.assertGreaterEqual(len(chosen), 60)
        full = {tuple(b) for b in m5_labels.label_batches(lengths_)}
        sub = m5_labels.label_batches([lengths_[i] for i in chosen])
        self.assertTrue(all(tuple(chosen[j] for j in b) in full for b in sub))


class MlxScoreTest(unittest.TestCase):
    def index(self):
        out = []
        for i in range(40):
            cell = (
                mlx_dev.NOUL_CELLS[i % 2]
                if i < 20
                else (mlx_dev.CHOICE_CELLS + mlx_dev.SCORE_CELLS)[i % 6]
            )
            kind = (
                "noul"
                if i < 20
                else ("choice" if cell in mlx_dev.CHOICE_CELLS else "score")
            )
            gold = {"noul": ["true", "false"][i % 2], "choice": "a", "score": "2"}[kind]
            out.append(
                {
                    "id": f"p{i}",
                    "cell": cell,
                    "group_id": f"g{i // 2}",
                    "language": ["ja", "ko"][i % 4 // 2],
                    "source": cell,
                    "task_type": kind,
                    "gold_key": gold,
                }
            )
        return out

    def test_metrics_and_zero_paired_difference(self):
        index = self.index()
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "p.jsonl"
            with pred.open("w") as f:
                for e in index:
                    guess = {"noul": "true", "choice": "a", "score": "3"}[
                        e["task_type"]
                    ]
                    f.write(json.dumps({"id": e["id"], "prediction_key": guess}) + "\n")
            rows = mlx_dev.outcomes(index, pred)
            m = mlx_dev.metrics(mlx_dev.sum_stats(rows))
            self.assertAlmostEqual(m["noul_ml"], 0.5)
            self.assertAlmostEqual(m["noul_pred_yes_rate_macro"], 1.0)
            self.assertAlmostEqual(m["choice_ml"], 1.0)
            self.assertAlmostEqual(m["score_ml"], 0.0)
            self.assertAlmostEqual(m["score_ml_within1"], 1.0)
            boot = mlx_dev.paired_bootstrap(rows, rows, reps=50)
            self.assertEqual(boot["m_dev"]["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
