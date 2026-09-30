import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

M5 = Path(__file__).resolve().parents[1] / "m5"


def lsoup_check() -> str:
    """The Python check embedded in m5-tail.sh's lsoup stage."""
    text = (M5 / "m5-tail.sh").read_text(encoding="utf-8")
    match = re.search(
        r"<<'EOF' \| tee \"\$OUT/soup/check.json\"\n(.*?)\nEOF\n", text, re.S
    )
    assert match, "lsoup check not found"
    return match.group(1)


class M5ScriptTest(unittest.TestCase):
    def test_bash_n(self):
        scripts = sorted(M5.glob("*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_l128_chain_constants(self):
        chain = (M5 / "m5-l128-chain.sh").read_text(encoding="utf-8")
        # pinned base text backbone + rank-256 adapter (4 x A20r's rank-64 466,911,232) + head
        self.assertIn(f"LOADED={25_624_600_064 + 4 * 466_911_232 + 5_263_872}", chain)
        self.assertIn("export CHECKPOINT_FORMAT=peft-lora/1", chain)
        self.assertIn("CAP=72", chain)
        self.assertIn("M5-L128.PUSHED", chain)
        watch = (M5 / "m5-mlx-watch.sh").read_text(encoding="utf-8")
        self.assertIn('"$X/$NAME.PUSHED"', watch)
        self.assertIn('"$X/$NAME.SKIP"', watch)

    def run_check(
        self, root: Path, relay_lines: list[str], manifest_rank=256, members_listed=None
    ):
        members = []
        for seed in ("s1", "s2"):
            member = root / seed / "checkpoint"
            member.mkdir(parents=True)
            (member / "decision_config.json").write_text(
                json.dumps({"lora": {"rank": 128, "alpha": 256}})
            )
            members.append(member)
        files = {
            "adapter/adapter_model.safetensors": "a" * 64,
            "decision_head.safetensors": "b" * 64,
        }
        manifest = {
            "lora": {"members": 2, "rank": manifest_rank, "alpha": 512},
            "verification": {
                "verify_adapter_config": True,
                "max_relative_diff": 1e-7,
                "tolerance_relative": 1e-6,
            },
            "members": [
                {"path": str(p), "files_sha256": files}
                for p in (members_listed or members)
            ],
            "output": {"model_sha256": "c" * 64},
        }
        (root / "soup_manifest.json").write_text(json.dumps(manifest))
        sums = root / "SHA256SUMS"
        sums.write_text("".join(line + "\n" for line in relay_lines))
        return subprocess.run(
            [
                sys.executable,
                "-",
                str(root / "soup_manifest.json"),
                f"{members[1]}={sums}",
                *map(str, members),
            ],
            input=lsoup_check(),
            capture_output=True,
            text=True,
        )

    def test_lsoup_check_accepts_the_relayed_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_check(
                Path(tmp),
                [
                    f"{'b' * 64}  ./decision_head.safetensors",
                    f"{'a' * 64}  ./adapter/adapter_model.safetensors",
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            out = json.loads(result.stdout)
            self.assertEqual((out["rank"], out["alpha"], out["members"]), (256, 512, 2))
            self.assertEqual(len(out["relay_checked"]), 1)

    def test_lsoup_check_refuses_a_changed_relay_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_check(
                Path(tmp),
                [
                    f"{'b' * 64}  ./decision_head.safetensors",
                    f"{'d' * 64}  ./adapter/adapter_model.safetensors",
                ],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("is not the relayed file", result.stderr)

    def test_lsoup_check_refuses_a_wrong_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_check(
                Path(tmp),
                [
                    f"{'b' * 64}  ./decision_head.safetensors",
                    f"{'a' * 64}  ./adapter/adapter_model.safetensors",
                ],
                manifest_rank=128,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "expected members / rank / alpha (2, 256, 512)", result.stderr
            )


if __name__ == "__main__":
    unittest.main()
