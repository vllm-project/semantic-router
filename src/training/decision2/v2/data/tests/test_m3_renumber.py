from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from training.model.data import INPUT_FIELDS, canonical, digest, validate_row
from v2.data.build_a0_variants import native_prompt
from v2.data.m3 import renumber
from v2.data.replay_targets import collector_digest
from v2.data.tests.test_build_a0_variants import _row

MODEL, REV = "org/lux", "rev1"


def _leaky(index: int) -> dict:
    row = _row(index, "choice")
    keys = ["result_2", "result_0", "result_3", "result_1"]
    row["options"] = [dict(o, key=k) for o, k in zip(row["options"], keys)]
    row["input_sha256"] = digest({f: row[f] for f in INPUT_FIELDS})
    return validate_row(row, "train")


def _write(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _receipts(path: Path, rows: list[dict], p: float) -> Path:
    out = []
    for row in rows:
        prompt = native_prompt(row)
        keys = [o["key"] for o in row["options"]]
        rest = (1 - p) / (len(keys) - 1)
        answer = (
            {"type": "noul", "noul": p}
            if row["task_type"] == "noul"
            else {
                "type": row["task_type"],
                "probabilities": {k: p if i == 0 else rest for i, k in enumerate(keys)},
            }
        )
        out.append(
            {
                "id": row["id"],
                "answers": {"decision": answer},
                "model_id": MODEL,
                "model_revision": REV,
                "revision_attested": True,
                "runtime_matches_validated": True,
                "source_input_sha256": collector_digest(prompt),
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    return path


class RenumberTest(unittest.TestCase):
    def test_rows_lux_and_r2_rederive_only_renumbered_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a0 = [
                _leaky(1),
                _row(2, "noul"),
                _leaky(3),
                _row(4, "score"),
                _row(5, "choice"),
            ]
            rp = [a0[0], a0[1], a0[4]]
            old_r2 = []
            for row in rp:
                keys = [o["key"] for o in row["options"]]
                old_r2.append(dict(row, teacher_probs={k: 1 / len(keys) for k in keys}))
            canonical_lux = [
                {
                    "id": r["id"],
                    "input_sha256": r["input_sha256"],
                    "teacher_probs": {
                        o["key"]: 1 / len(r["options"]) for o in r["options"]
                    },
                }
                for r in a0
            ]
            out = root / "pk1"
            args = argparse.Namespace(
                file=[
                    f"A0={_write(root / 'a0.jsonl', a0)}",
                    f"RP-v1q={_write(root / 'rp.jsonl', rp)}",
                ],
                r2=[f"kai={_write(root / 'kai.jsonl', old_r2)}"],
                control_a0=2,
                control_r2=1,
                out_dir=out,
            )
            summary = renumber.rows_command(args)
            self.assertEqual(summary["A0"]["renumbered"], 2)
            self.assertEqual(summary["RP-v1q"]["renumbered"], 1)
            plan = json.loads((out / "rederive.json").read_text())
            self.assertEqual(plan["lux_a0"]["renumbered"], ["r0001", "r0003"])
            self.assertEqual(len(plan["lux_a0"]["control"]), 2)
            receipt = json.loads((out / "A0" / "receipt.json").read_text())
            self.assertTrue(receipt["unchanged_rows_byte_identical"])
            pk1 = {
                r["id"]: r
                for r in map(
                    json.loads, (out / "A0" / "train.jsonl").read_text().splitlines()
                )
            }
            self.assertEqual(
                [o["key"] for o in pk1["r0001"]["options"]],
                [f"result_{i}" for i in range(4)],
            )
            lux_rows = [
                pk1[i]
                for i in sorted(
                    plan["lux_a0"]["renumbered"] + plan["lux_a0"]["control"]
                )
            ]
            lux_args = argparse.Namespace(
                out_dir=out,
                canonical=Path(_write(root / "lux.jsonl", canonical_lux)),
                output=_receipts(root / "lux.out.jsonl", lux_rows, 0.7),
                model_id=MODEL,
                revision=REV,
            )
            lux = renumber.lux_command(lux_args)
            self.assertEqual(
                (lux["rows"], lux["rederived"], lux["rederive_missing"]), (5, 2, 0)
            )
            new = {
                r["id"]: r
                for r in map(
                    json.loads,
                    (out / "lux1" / "A0-train.canonical.jsonl")
                    .read_text()
                    .splitlines(),
                )
            }
            self.assertEqual(new["r0001"]["input_sha256"], pk1["r0001"]["input_sha256"])
            self.assertAlmostEqual(new["r0001"]["teacher_probs"]["result_0"], 0.7)
            self.assertEqual(new["r0002"], canonical_lux[1])
            r2_rows = [
                r
                for r in map(
                    json.loads,
                    (out / "RP-v1q" / "train.jsonl").read_text().splitlines(),
                )
                if r["id"] in plan["r2"]["renumbered"] + plan["r2"]["control"]
            ]
            r2 = renumber.r2_command(
                argparse.Namespace(
                    out_dir=out,
                    tier=[
                        f"kai={root / 'kai.jsonl'}={_receipts(root / 'kai.out.jsonl', r2_rows, 0.6)}={MODEL}={REV}"
                    ],
                )
            )
            self.assertEqual((r2["kai"]["rows"], r2["kai"]["rederived"]), (3, 1))
            rows = [
                json.loads(line)
                for line in (out / "R2" / "kai" / "replay.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertAlmostEqual(rows[0]["teacher_probs"]["result_0"], 0.6)
            self.assertEqual(rows[1], old_r2[1])


if __name__ == "__main__":
    unittest.main()
