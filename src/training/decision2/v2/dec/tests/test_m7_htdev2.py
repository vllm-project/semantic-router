"""Tests for the decoder M7 HT-DEV v2 diagnostic (ops/m7/m7_htdev2.py) and the M7 formal wrapper (ops/m7/m7-formal.sh:
slot filter, step order, stop on a failed GPU step, finished lease entries moved aside); bash syntax of the new scripts.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "ops" / "m7"
_spec = importlib.util.spec_from_file_location("m7_htdev2", OPS / "m7_htdev2.py")
m7h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m7h)


class FakeHTDev2:
    """Stands in for v2.eval.htdev2.score: every prediction row names its run in `who`."""

    REPLICATES = 10
    SEED = 1

    def __init__(self, f1: dict[str, dict[str, float]], invalid: dict[str, int]):
        self.f1, self.invalid = f1, invalid

    def read_jsonl(self, path):
        return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

    def report(self, gold, predictions, replicates=1):
        who = next(iter(predictions.values()))["who"]
        tasks = self.f1[who]
        values = sorted(tasks.values())
        return {
            "H_dev2": sum(values) / len(values),
            "H_dev2_median": values[len(values) // 2],
            "tasks": {t: {"macro_f1_all": v} for t, v in tasks.items()},
            "items": len(gold),
            "valid": len(gold) - self.invalid[who],
        }

    def outcomes(self, gold, predictions):
        return next(iter(predictions.values()))["who"]

    def bootstrap(self, tasks, replicates, seed, other=None):
        assert (tasks, other) == ("P", "I")
        return {"H_dev2": {"ci95": [-0.06, -0.01], "p_le_0": 0.99, "sd": 0.01}}


class HTDev2Test(unittest.TestCase):
    def test_verdict_band(self):
        self.assertEqual(m7h.verdict(-0.02), "FLAG")
        self.assertEqual(m7h.verdict(-0.0199), "TIE")
        self.assertEqual(m7h.verdict(0.0199), "TIE")
        self.assertEqual(m7h.verdict(0.02), "GAIN")

    def test_paired_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "gold.jsonl").write_text('{"id": "a"}\n{"id": "b"}\n')
            (t / "p.jsonl").write_text(
                '{"id": "a", "who": "P"}\n{"id": "b", "who": "P"}\n'
            )
            (t / "i.jsonl").write_text(
                '{"id": "a", "who": "I"}\n{"id": "b", "who": "I"}\n'
            )
            fake = FakeHTDev2(
                {"P": {"t1": 0.50, "t2": 0.40}, "I": {"t1": 0.55, "t2": 0.43}},
                {"P": 1, "I": 0},
            )
            out = m7h.evaluate(
                fake, t / "gold.jsonl", t / "p.jsonl", "P", t / "i.jsonl", "I"
            )
        self.assertAlmostEqual(out["delta"], 0.45 - 0.49)
        self.assertEqual(out["verdict"], "FLAG")
        self.assertEqual(out["ci95"], [-0.06, -0.01])
        self.assertEqual(out["tasks"]["t1"], {"P": 0.50, "I": 0.55})
        self.assertEqual(out["invalid_or_missing"], {"P": 1, "I": 0})
        self.assertEqual(set(out["files"]), {"gold", "P", "I"})
        self.assertIn("diagnostic", out["role"])


FAKE_M6_FORMAL = r"""#!/usr/bin/env bash
echo "formal $*" >> "$FAKE_LOG"
tier=$1 cmd=$2 point=${3:-}
lease=$M7_LEASES/gpu$M6_GPU.lock
case $cmd in
  smoke) [ "$point" = "$FAIL_SMOKE" ] && exit 1
         printf 'track=dec\nlast_job_end_utc=x\n' > "$lease/owner.m6-formal-smoke" ;;
  finalist) [ -e "$lease/owner.dec-formal" ] && { echo "owner.dec-formal exists" >> "$FAKE_LOG"; exit 1; }
            printf 'track=dec\nlast_job_end_utc=x\n' > "$lease/owner.dec-formal" ;;
esac
exit 0
"""


class FormalWrapperTest(unittest.TestCase):
    def run_wrapper(self, tier, gpu, slots, fail_smoke=""):
        tmp = Path(self.tmp.name)
        s = tmp / "decision2"
        (s / "v2/dec/ops/m6").mkdir(parents=True, exist_ok=True)
        for name in ("m6-formal.sh", "m6-score.sh"):
            (s / "v2/dec/ops/m6" / name).write_text(FAKE_M6_FORMAL)
        (tmp / "leases" / f"gpu{gpu}.lock").mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            M7_FORMAL_ROOT=str(tmp / "formal"),
            M7_ROOT=str(tmp / "m7"),
            M7_LEASES=str(tmp / "leases"),
            M7_DECISION2=str(s),
            M7_POLL_SECONDS="0",
            FAKE_LOG=str(tmp / "calls.log"),
            FAIL_SMOKE=fail_smoke,
            PATH=f"{tmp / 'bin'}:{os.environ['PATH']}",
        )
        (tmp / "bin").mkdir(exist_ok=True)
        (tmp / "bin" / "docker").write_text("#!/usr/bin/env bash\nexit 0\n")
        (tmp / "bin" / "docker").chmod(0o755)
        args = ["bash", str(OPS / "m7-formal.sh"), "run", "src", tier, str(gpu)]
        if slots:
            args.append(slots)
        r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return tmp

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        sel = Path(self.tmp.name) / "m7" / "select"
        sel.mkdir(parents=True)
        finalists = [
            {"slot": 2, "point": "4b-N7P-b1_2"},
            {"slot": 1, "point": "4b-N7C-b1"},
            {"slot": 3, "point": "4b-N7H-b1_3"},
        ]
        (sel / "4b-finalists.json").write_text(json.dumps({"finalists": finalists}))
        (sel / "2b-finalists.json").write_text(
            json.dumps({"finalists": [{"slot": 1, "point": "2b-S7H-b1"}]})
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_slots_in_order_and_stale_entries_moved(self):
        tmp = self.run_wrapper("4b", 3, "1,3")
        calls = (tmp / "calls.log").read_text().splitlines()
        self.assertEqual(
            calls,
            [
                "formal 4b smoke 4b-N7C-b1 8",
                "formal 4b finalist 4b-N7C-b1",
                "formal 4b smoke 4b-N7H-b1_3 8",
                "formal 4b finalist 4b-N7H-b1_3",
            ],
        )
        status = tmp / "formal" / "status"
        self.assertTrue((status / "m7-4b-N7H-b1_3.COLLECTED").is_file())
        self.assertFalse(list((tmp / "leases" / "gpu3.lock").iterdir()))
        self.assertEqual(
            len(list((tmp / "formal" / "logs" / "stale-leases").iterdir())), 2 * 2
        )

    def test_failed_smoke_stops_that_finalist_only(self):
        tmp = self.run_wrapper("4b", 4, "", fail_smoke="4b-N7C-b1")
        calls = (tmp / "calls.log").read_text().splitlines()
        self.assertNotIn("formal 4b finalist 4b-N7C-b1", calls)
        self.assertIn("formal 4b finalist 4b-N7P-b1_2", calls)
        self.assertTrue((tmp / "formal" / "status" / "m7-4b-N7C-b1.FAILED").is_file())

    def test_2b_scores_and_collects_mlx_on_node_a(self):
        tmp = self.run_wrapper("2b", 5, "")
        calls = (tmp / "calls.log").read_text().splitlines()
        self.assertEqual(
            calls,
            [
                "formal 2b smoke 2b-S7H-b1 8",
                "formal 2b finalist 2b-S7H-b1",
                "formal 2b m7-2b-S7H-b1",
                "formal 2b mlx m7-2b-S7H-b1",
                "formal 2b mlx m7-2b-S7H-b1",
            ],
        )

    def test_bash_syntax(self):
        for name in ("m7-formal.sh", "m7-htdev2.sh", "m7-relay.sh"):
            r = subprocess.run(
                ["bash", "-n", str(OPS / name)], capture_output=True, text=True
            )
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")


if __name__ == "__main__":
    unittest.main()
