import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

launch = importlib.import_module("v2.27b.launch")
ROOT = Path(__file__).resolve().parents[1]


def fake_sysfs(root: Path, render: str, pci: str) -> Path:
    device = root / "devices" / pci
    device.mkdir(parents=True)
    node = root / "drm" / render
    node.mkdir(parents=True)
    (node / "device").symlink_to(device)
    return root / "drm"


class NodeGpuTest(unittest.TestCase):
    def test_node_a_accepts_only_gpu2_to_4(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = fake_sysfs(Path(tmp), "renderD145", "0000:93:00.0")
            with mock.patch.dict(os.environ, {"DEV2_NODE": "a"}):
                self.assertEqual(launch.render_node(2, drm).name, "renderD145")
                for gpu in (0, 1, 5, 6, 7):
                    with self.assertRaises(ValueError):
                        launch.render_node(gpu, drm)

    def test_default_node_is_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = fake_sysfs(Path(tmp), "renderD169", "0000:ab:00.0")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(launch.node_name(), "b")
                self.assertEqual(launch.render_node(5, drm).name, "renderD169")
                for gpu in (2, 3, 4):
                    with self.assertRaises(ValueError):
                        launch.render_node(gpu, drm)

    def test_unknown_node_and_explicit_node(self):
        with mock.patch.dict(os.environ, {"DEV2_NODE": "c"}):
            with self.assertRaises(ValueError):
                launch.node_name()
        self.assertEqual(sorted(launch.allowed_gpus("a")), [2, 3, 4])
        self.assertEqual(launch.ALLOWED_GPUS, launch.NODE_GPUS["b"])

    def test_lease_of_another_track_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            lease = Path(tmp) / "gpu2.lock"
            lease.mkdir()
            (lease / "owner").write_text(
                "track=9b\nstatus=released\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                launch.write_lease(2, {"status": "running"}, Path(tmp))

    def test_cap_limit_is_13_hours(self):
        self.assertEqual(launch.MAX_CAP_HOURS, 13.0)


class ArmScriptTest(unittest.TestCase):
    def test_rank_cache_and_cap_knobs(self):
        text = (ROOT / "run_lora_arm.sh").read_text(encoding="utf-8")
        self.assertIn('--lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"', text)
        self.assertIn("LORA_RANK=${LORA_RANK:-8}", text)
        self.assertIn("LORA_ALPHA=${LORA_ALPHA:-16}", text)
        self.assertIn("triton_cache copy", text)
        self.assertIn("triton_cache finish", text)
        self.assertIn('train_launch "$name" "$cap"', text)
        self.assertNotIn('train_launch "$name" "$FULL_CAP"', text)


if __name__ == "__main__":
    unittest.main()
