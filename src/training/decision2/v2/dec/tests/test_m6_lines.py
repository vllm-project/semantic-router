"""CPU mechanics test for ops/m6/m6-lines.sh: fake mirror, fake launcher, fake rocm-smi, stub readout."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "ops" / "m6"

FAKE_LAUNCH = r"""#!/usr/bin/env bash
# fake launch.sh: <name> <sha> <out> [--cpu] -- -m <module> args...
name=$1 sha=$2 out=$3; shift 3
cpu=0; [ "$1" = --cpu ] && { cpu=1; shift; }; [ "$1" = -- ] && shift
shift  # -m
mod=$1; shift
mkdir -p "$out"
echo "$mod $*" >> "$FAKE_LOG"
start=$(date -u +%FT%TZ)
case $mod in
  v2.dec.soup)
    o=""; members=()
    while [ $# -gt 0 ]; do case $1 in --member) members+=("$2"); shift 2 ;; --output) o=$2; shift 2 ;; *) shift ;; esac; done
    d=$out/$(basename "$o"); mkdir -p "$d/backbone"
    echo '{"checkpoint_format": "full"}' > "$d/decision_config.json"
    printf '%s\n' "${members[@]}" > "$d/backbone/members.txt"
    python3 -c 'import json,sys; print(json.dumps({"members": sys.argv[1:], "model_sha256": "m", "head_sha256": "h"}))' "${members[@]}" > "$out.stdout.log" ;;
  v2.dec.infer_dec)
    while [ $# -gt 0 ]; do case $1 in --output) o=$2; shift 2 ;; --max-length) echo "$2" > "$out/max_length"; shift 2 ;; *) shift ;; esac; done
    echo '{"id": "x"}' > "$out/$(basename "$o")" ;;
esac
printf '{"job": "%s", "gpu": %s, "start_utc": "%s", "end_utc": "%s", "exit_status": 0}\n' "$name" \
  "$([ $cpu = 1 ] && echo null || echo "\"$DEC_GPU_LABEL\"")" "$start" "$(date -u +%FT%TZ)" > "$out.launch.json"
"""

FAKE_ROCM = """#!/usr/bin/env bash
echo "GPU[$2]		: VRAM Total Memory (B): ${FAKE_TOTAL:-206000000000}"
echo "GPU[$2]		: VRAM Total Used Memory (B): ${FAKE_USED:-10000000000}"
"""

STUB_READOUT = """import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--typed-gold"); p.add_argument("--css-gold"); p.add_argument("--output")
p.add_argument("--arm", action="append"); p.add_argument("--compare", action="append", default=[])
a = p.parse_args()
for spec in a.arm:
    for path in spec.split("=", 1)[1].split(","):
        open(path).read()
json.dump({"arms": {s.split("=", 1)[0]: {} for s in a.arm}, "compares": a.compare}, open(a.output, "w"))
"""


def ckpt(path: Path) -> Path:
    (path / "backbone").mkdir(parents=True)
    (path / "decision_config.json").write_text('{"checkpoint_format": "full"}')
    (path / "backbone" / "w.safetensors").write_text(path.name)
    return path


class LinesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.t = t
        s = t / "src" / "abc-src_training_decision2" / "src" / "training" / "decision2"
        (s / "v2" / "dec" / "ops" / "m6").mkdir(parents=True)
        (s.parents[2] / ".dev2-mirror.json").write_text(
            '{"commit": "abc", "tree": "t"}'
        )
        shutil.copy(
            OPS / "m6-lines.sh", s / "v2" / "dec" / "ops" / "m6" / "m6-lines.sh"
        )
        (s / "stub_readout.py").write_text(STUB_READOUT)
        self.script = s / "v2" / "dec" / "ops" / "m6" / "m6-lines.sh"
        bin_dir = t / "bin"
        bin_dir.mkdir()
        (bin_dir / "rocm-smi").write_text(FAKE_ROCM)
        (bin_dir / "rocm-smi").chmod(0o755)
        launch = t / "launch.sh"
        launch.write_text(FAKE_LAUNCH)
        self.root = t / "runs"
        ckpt(self.root / "m4/soup/N4XF/build/N4XF-soup")
        ckpt(self.root / "m4/arms/pre/m4-N4XF-s1-zero/checkpoint-0000000")
        ckpt(self.root / "m5/soup/N5BN/build/N5BN-soup")
        self.env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "M6_DEC_ROOT": str(self.root),
            "M6_LAUNCH": str(launch),
            "M6_GOLD": str(t / "gold"),
            "M6_LEASES": str(t / "leases"),
            "M6_DEV_READOUT": "stub_readout",
            "M6_GPU": "3",
            "FAKE_LOG": str(t / "launch.log"),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_lines(self, *args, env=None):
        return subprocess.run(
            ["bash", str(self.script), *args],
            env={**self.env, **(env or {})},
            capture_output=True,
            text=True,
        )

    def test_refs_then_arm_line_then_own_line(self):
        L = self.root / "m6/lines/4b"
        r = self.run_lines("4b", "line", "Nox")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("4b-I not read yet", r.stderr)
        r = self.run_lines("4b", "refs")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((L / "4b-I/dev/max_length").read_text().strip(), "16384")
        self.assertFalse((self.t / "leases/gpu3.lock/owner.dec-readout").exists())
        r = self.run_lines("4b", "line", "N6A")
        self.assertIn("arm soup not done yet", r.stderr)
        ckpt(self.root / "m6/soup/N6A/build/N6A-soup")
        (self.root / "m6/soup/N6A/DONE").write_text("soup=...\n")
        r = self.run_lines("4b", "line", "N6A")
        self.assertEqual(r.returncode, 0, r.stderr)
        w = json.loads((L / "4b-N6A-b2_3/weights.json").read_text())
        self.assertEqual(w["effective_weights"], {"A": "2/3", "I": "1/3"})
        self.assertEqual([m["name"] for m in w["members"]], ["I", "A", "A"])
        members = (
            (L / "4b-N6A-b2_3/build/4b-N6A-b2_3/backbone/members.txt")
            .read_text()
            .split()
        )
        self.assertEqual(
            members,
            ["/runs/m4/soup/N4XF/build/N4XF-soup"]
            + ["/runs/m6/soup/N6A/build/N6A-soup"] * 2,
        )
        b1 = json.loads((L / "4b-N6A-b1/weights.json").read_text())
        self.assertEqual(
            b1["checkpoint"], str(self.root / "m6/soup/N6A/build/N6A-soup")
        )
        self.assertIsNone(b1["soup_output"])
        spec = (L / "readout/L-N6A.line").read_text().strip()
        self.assertEqual(
            spec, "L-N6A:1/3=4b-N6A-b1_3,1/2=4b-N6A-b1_2,2/3=4b-N6A-b2_3,1=4b-N6A-b1"
        )
        doc = json.loads((L / "readout/L-N6A.json").read_text())
        self.assertEqual(
            sorted(doc["arms"]),
            ["4b-I", "4b-N6A-b1", "4b-N6A-b1_2", "4b-N6A-b1_3", "4b-N6A-b2_3"],
        )
        self.assertIn("4b-I:4b-N6A-b1_3", doc["compares"])
        r = self.run_lines("4b", "line", "Nox")
        self.assertEqual(r.returncode, 0, r.stderr)
        g = json.loads((L / "4b-Nox-g1_6/weights.json").read_text())
        self.assertEqual(g["effective_weights"], {"I": "5/6", "O": "1/6"})
        g3 = json.loads((L / "4b-Nox-g1_3/weights.json").read_text())
        self.assertEqual(g3["effective_weights"], {"I": "2/3", "O": "1/3"})
        r = self.run_lines("4b", "line", "N5BN")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(
            (L / "readout/L-N5BN.line").read_text().strip(),
            "L-N5BN:1/3=4b-N5BN-b1_3,1/2=4b-N5BN-b1_2",
        )
        jobs = [
            json.loads(x) for x in (L / "GPU-SECONDS.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(jobs), 2 * (2 + 4 + 2 + 2))
        self.assertTrue(all(j["gpu"] == "node B GPU3" for j in jobs))
        before = (self.t / "launch.log").read_text()
        r = self.run_lines("4b", "line", "N6A")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.t / "launch.log").read_text(), before)

    def test_low_vram_refuses_and_leaves_no_lease(self):
        r = self.run_lines("4b", "refs", env={"FAKE_USED": "160000000000"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("< 60", r.stderr)
        self.assertFalse((self.t / "leases/gpu3.lock/owner.dec-readout").exists())

    def test_wrong_gpu_and_unknown_line(self):
        r = self.run_lines("4b", "refs", env={"M6_GPU": "5"})
        self.assertIn("not a node-B decoder GPU", r.stderr)
        r = self.run_lines("2b", "line", "N5BN")
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
