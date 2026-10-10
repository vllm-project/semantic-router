from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from d25.omni.eval.engine import VisionCodeReadoutModel
from d25.omni.model import checkpoint, tiny
from d25.omni.train import train
from d25.omni.train.model import init_vision_digests

ROOT = Path(__file__).resolve().parents[4]


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = tiny.fixtures()
        cls.tmp = Path(tempfile.mkdtemp(prefix="d25-omni-trainer-"))
        data = cls.paths["data"]
        cls.common = [
            "--rows",
            str(data / "mm.jsonl"),
            "--replay-rows",
            str(data / "text.jsonl"),
            "--dev-rows",
            str(data / "dev.jsonl"),
            "--replay-ratio",
            "0.5",
            "--effective-batch-size",
            "8",
            "--token-budget",
            "2048",
            "--max-rows-per-microbatch",
            "4",
            "--lr",
            "1e-3",
            "--warmup-ratio",
            "0.2",
            "--save-every",
            "3",
            "--resume-every",
            "2",
            "--eval-every",
            "3",
        ]
        for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
            os.environ.pop(name, None)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_dry_run_plans_without_a_model(self) -> None:
        out = self.tmp / "dry"
        train.main(
            [
                "--init",
                str(self.paths["init_vega"]),
                "--arm",
                "O-graft-frozen",
                "--out",
                str(out),
                "--dry-run",
                *self.common,
            ]
        )
        plan = json.loads((out / "plan.json").read_text())
        self.assertEqual(plan["rows"], {"main": 24, "replay": 40, "dev": 8})
        self.assertEqual(plan["schedule"]["steps"], 6)
        self.assertGreater(
            plan["tokens"]["image_token_share"], plan["tokens"]["image_row_share"] / 2
        )
        self.assertFalse((out / "checkpoints").exists())

    def test_wrong_init_kind_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            train.main(
                [
                    "--init",
                    str(self.paths["init_vega"]),
                    "--arm",
                    "O-fresh",
                    "--out",
                    str(self.tmp / "wrong"),
                    *self.common,
                ]
            )

    def test_single_process_run(self) -> None:
        out = self.tmp / "single"
        train.main(
            [
                "--init",
                str(self.paths["init_vega"]),
                "--arm",
                "O-graft-frozen",
                "--out",
                str(out),
                *self.common,
            ]
        )
        summary = json.loads((out / "summary.json").read_text())
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["checkpoints"], ["step-00003", "step-00006"])
        steps = lines(out / "training.jsonl")
        self.assertEqual([s["step"] for s in steps], list(range(1, 7)))
        self.assertTrue(all(s["replay_rows"] == 4 for s in steps))
        self.assertEqual(len(lines(out / "evaluations.jsonl")), 2)
        final = out / "checkpoints" / "step-00006"
        initial = init_vision_digests(self.paths["init_vega"])
        saved = init_vision_digests(final)
        encoder = [n for n in initial if checkpoint.component(n) == "vision_encoder"]
        self.assertTrue(all(saved[n] == initial[n] for n in encoder))
        provenance = checkpoint.read_decision_config(final)["provenance"]
        self.assertEqual(provenance["arm"], "O-graft-frozen")
        self.assertEqual(provenance["init"]["kind"], "vega")
        engine = VisionCodeReadoutModel(final, device="cpu")
        self.assertEqual(
            len(
                engine.predict(
                    [{"state": "s", "question": {"type": "noul"}, "images": []}]
                )[0]
            ),
            2,
        )
        train.main(
            [
                "--init",
                str(self.paths["init_vega"]),
                "--arm",
                "O-graft-frozen",
                "--out",
                str(out),
                *self.common,
            ]
        )
        self.assertEqual(len(lines(out / "training.jsonl")), 6)
        with self.assertRaises(SystemExit):
            train.main(
                [
                    "--init",
                    str(self.paths["init_vega"]),
                    "--arm",
                    "O-graft-frozen",
                    "--out",
                    str(out),
                    "--seed",
                    "1",
                    *self.common,
                ]
            )

    @unittest.skipUnless(
        shutil.which("torchrun") or Path(sys.executable).with_name("torchrun").exists(),
        "torchrun not available",
    )
    def test_two_rank_fsdp_run_matches_single_process(self) -> None:
        single = self.tmp / "fsdp-reference"
        train.main(
            [
                "--init",
                str(self.paths["init_stock"]),
                "--arm",
                "O-fresh",
                "--out",
                str(single),
                "--stop-after",
                "2",
                *self.common,
            ]
        )
        out = self.tmp / "fsdp"
        torchrun = shutil.which("torchrun") or str(
            Path(sys.executable).with_name("torchrun")
        )
        command = [
            torchrun,
            "--nproc_per_node",
            "2",
            "--master_port",
            "29561",
            "-m",
            "d25.omni.train.train",
            "--init",
            str(self.paths["init_stock"]),
            "--arm",
            "O-fresh",
            "--out",
            str(out),
            "--stop-after",
            "2",
            *self.common,
        ]
        environment = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2"}
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        sharded, reference = lines(out / "training.jsonl"), lines(
            single / "training.jsonl"
        )
        self.assertEqual([s["step"] for s in sharded], [1, 2])
        self.assertAlmostEqual(sharded[0]["loss"], reference[0]["loss"], delta=2e-3)
        self.assertEqual(sharded[0]["rows"], reference[0]["rows"])
        engine = VisionCodeReadoutModel(
            out / "checkpoints" / "step-00002", device="cpu"
        )
        self.assertTrue(engine.has_vision)


if __name__ == "__main__":
    unittest.main()
