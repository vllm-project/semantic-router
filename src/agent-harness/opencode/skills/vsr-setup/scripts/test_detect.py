import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import detect


def found(**over) -> dict:
    base = {
        "os": {"system": "Linux", "arch": "x86_64", "arm": False},
        "cpus": 12,
        "runtime": {"ready": False, "url": "http://127.0.0.1:8790/v1/decisions"},
        "pythons": [{"ok": True}],
        "memory": {"totalGiB": 16},
        "gpus": [],
        "installed": None,
        "diskFreeGiB": 100,
        "modelCached": False,
    }
    base.update(over)
    return base


CURRENT = {"commit": detect.COMMIT, "multimodal": True}


class Recommend(unittest.TestCase):
    def test_a_current_runtime_that_answers_is_kept(self):
        ready = {"ready": True, "url": "http://127.0.0.1:8790/v1/decisions"}
        rec = detect.recommend(found(runtime=ready, installed=CURRENT))
        self.assertEqual(rec["runtime"], "existing")

    def test_an_older_or_imageless_runtime_here_is_upgraded(self):
        ready = {"ready": True, "url": "http://127.0.0.1:8790/v1/decisions"}
        for installed, why in (
            (
                {"commit": next(iter(detect.OLDER)), "multimodal": True},
                "installed from 579e0ca2a",
            ),
            ({"commit": detect.COMMIT, "multimodal": False}, "answers no image"),
        ):
            with self.subTest(why=why):
                rec = detect.recommend(found(runtime=ready, installed=installed))
                self.assertEqual(rec["runtime"], "upgrade")
                self.assertIn(why, rec["why"])

    def test_a_runtime_on_another_machine_is_kept_whatever_is_installed_here(self):
        ready = {"ready": True, "url": "https://app.vllm-sr.ai/v1/decisions"}
        rec = detect.recommend(
            found(runtime=ready, installed={"commit": "old", "multimodal": False})
        )
        self.assertEqual(rec["runtime"], "existing")

    def test_a_gpu_with_free_memory(self):
        gpu = {"device": "cuda", "name": "RTX 4090", "freeGiB": 10, "totalGiB": 24}
        rec = detect.recommend(found(gpus=[gpu]))
        self.assertEqual((rec["runtime"], rec["device"]), ("local-gpu", "cuda"))

    def test_a_busy_gpu_falls_back_to_the_cpu(self):
        gpu = {"device": "cuda", "name": "g", "freeGiB": 2, "totalGiB": 24}
        rec = detect.recommend(found(gpus=[gpu]))
        self.assertEqual(rec["runtime"], "local-cpu")
        self.assertIn("too little free memory", rec["why"])

    def test_an_arm_cpu_is_called_slow(self):
        rec = detect.recommend(
            found(os={"system": "Darwin", "arch": "arm64", "arm": True})
        )
        self.assertEqual(rec["runtime"], "local-cpu")
        self.assertIn("ARM CPU takes seconds", rec["why"])

    def test_a_full_disk_or_an_old_python_means_remote(self):
        self.assertEqual(detect.recommend(found(diskFreeGiB=1.0))["runtime"], "remote")
        self.assertEqual(
            detect.recommend(found(pythons=[{"ok": False}]))["runtime"], "remote"
        )

    def test_an_installed_runtime_and_cached_model_need_less_disk(self):
        self.assertEqual(detect.disk_need(found(), "gpu"), 8.8)
        self.assertEqual(
            detect.disk_need(found(installed=CURRENT, modelCached=True), "gpu"), 1.0
        )


class Device(unittest.TestCase):
    def test_the_reported_device_maps_to_what_local_py_serves_on(self):
        nvidia = [{"device": "cuda"}]
        amd = [{"device": "rocm"}]
        self.assertEqual(detect.serve_device("cuda:0", nvidia), "cuda")
        self.assertEqual(detect.serve_device("cuda:0", amd), "rocm")
        self.assertEqual(detect.serve_device("cpu", []), "cpu")
        self.assertIsNone(detect.serve_device(None, nvidia))

    def test_an_upgrade_names_the_device_to_restart_on(self):
        ready = {
            "ready": True,
            "url": "http://127.0.0.1:8790/v1/decisions",
            "device": "cuda:0",
        }
        rec = detect.recommend(
            found(
                runtime=ready,
                gpus=[{"device": "cuda", "name": "g", "freeGiB": 1, "totalGiB": 24}],
                installed={"commit": next(iter(detect.OLDER)), "multimodal": True},
            )
        )
        self.assertEqual((rec["runtime"], rec["device"]), ("upgrade", "cuda"))

    def test_a_commit_setup_does_not_know_is_never_moved_backwards(self):
        # The Claude Code plugin shares the venv and may pin a newer commit.
        ready = {"ready": True, "url": "http://127.0.0.1:8790/v1/decisions"}
        rec = detect.recommend(
            found(runtime=ready, installed={"commit": "f" * 40, "multimodal": True})
        )
        self.assertEqual(rec["runtime"], "existing")

    def test_a_local_build_is_never_upgraded(self):
        ready = {"ready": True, "url": "http://127.0.0.1:8790/v1/decisions"}
        rec = detect.recommend(
            found(runtime=ready, installed={"commit": None, "multimodal": False})
        )
        self.assertEqual(rec["runtime"], "existing")

    def test_a_slow_running_runtime_is_marked(self):
        ready = {
            "ready": True,
            "url": "http://127.0.0.1:8790/v1/decisions",
            "questionMs": 4000,
        }
        self.assertIn(
            "slow runtime, 4000 ms",
            detect.recommend(found(runtime=ready, installed=CURRENT))["slow"],
        )


class Installed(unittest.TestCase):
    def test_the_commit_comes_from_pips_direct_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp)
            info = venv / "lib/python3.12/site-packages/vllm_srun-0.2.0.dist-info"
            info.mkdir(parents=True)
            (info / "direct_url.json").write_text(
                json.dumps(
                    {
                        "url": "https://github.com/x",
                        "vcs_info": {"vcs": "git", "commit_id": detect.COMMIT},
                    }
                )
            )
            self.assertEqual(detect.installed_commit(venv), detect.COMMIT)
            self.assertIsNone(
                detect.installed(venv), "no vllm-srun script, not installed"
            )

    def test_a_gpu_whose_memory_reads_na_shares_the_systems(self):
        out = "NVIDIA A100, 81920, 1024\nNVIDIA GB10, [N/A], [N/A]\nRTX, Ti, 8, 24576, 0\n"
        self.assertEqual(
            [
                (g["name"], g["totalGiB"], g["freeGiB"], g.get("unified", False))
                for g in detect.nvidia_gpus(out)
            ],
            [
                ("NVIDIA A100", 80.0, 79.0, False),
                ("NVIDIA GB10", None, None, True),
                ("RTX, Ti, 8", 24.0, 24.0, False),
            ],
        )
        gb10 = {
            "device": "cuda",
            "name": "NVIDIA GB10",
            "unified": True,
            "freeGiB": 88.5,
            "totalGiB": 121.7,
        }
        self.assertEqual(detect.recommend(found(gpus=[gb10]))["runtime"], "local-gpu")


if __name__ == "__main__":
    unittest.main()
