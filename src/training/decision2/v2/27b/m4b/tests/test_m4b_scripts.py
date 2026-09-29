import subprocess
import unittest
from pathlib import Path

M4B = Path(__file__).resolve().parents[1]


class ScriptSyntaxTest(unittest.TestCase):
    def test_bash_n(self):
        scripts = sorted(M4B.glob("*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_drivers_start_strict_and_log_first(self):
        for name in (
            "run_readout.sh",
            "run_formal.sh",
            "run_mlx.sh",
            "run_gates.sh",
            "score_mlx_nodeA.sh",
            "mlx_pair_nodeA.sh",
        ):
            with self.subTest(script=name):
                lines = (M4B / name).read_text(encoding="utf-8").splitlines()
                code = [line for line in lines if line and not line.startswith("#")]
                self.assertEqual(code[0], "set -euo pipefail")
                self.assertTrue(code[1].startswith("echo "), code[1])

    def test_gpu_allocation(self):
        common = (M4B / "common3.sh").read_text(encoding="utf-8")
        self.assertIn('case "$GPU" in 0 | 1 | 2) ;;', common)
        self.assertIn("v2.27b.m4b.launch3 --name", common)
        self.assertIn('--gpus "$GPU"', common)
        for name in ("run_readout.sh", "run_formal.sh", "run_mlx.sh"):
            text = (M4B / name).read_text(encoding="utf-8")
            self.assertIn('source "$S/v2/27b/m4b/common3.sh"', text)
            self.assertNotIn("python3 -m v2.27b.launch ", text)


if __name__ == "__main__":
    unittest.main()
