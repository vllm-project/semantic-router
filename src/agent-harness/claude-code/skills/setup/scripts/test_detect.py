import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import detect
import runtime


class Recommend(unittest.TestCase):
    def base(self, **over):
        found = {
            "runtime": {"ready": False, "url": "u"},
            "pythons": [{"ok": True}],
            "memory": {"totalGiB": 16},
            "gpus": [],
        }
        found.update(over)
        return found

    def test_existing_runtime_wins(self):
        self.assertEqual(
            detect.recommend(self.base(runtime={"ready": True, "url": "u"}))["runtime"],
            "existing",
        )

    def test_gpu_with_free_memory(self):
        gpu = {"device": "cuda", "name": "g", "freeGiB": 30, "totalGiB": 48}
        rec = detect.recommend(self.base(gpus=[gpu]))
        self.assertEqual(
            (rec["runtime"], rec["device"], rec["model"]),
            ("local-gpu", "cuda", "vllm-sr/vela3-lite"),
        )
        self.assertEqual(rec["larger"], ["vllm-sr/vela3-mini", "vllm-sr/vela3-flash"])

    def test_busy_gpu_falls_back_to_cpu(self):
        gpu = {"device": "cuda", "name": "g", "freeGiB": 2, "totalGiB": 48}
        self.assertEqual(
            detect.recommend(self.base(gpus=[gpu]))["runtime"], "local-cpu"
        )

    def test_old_python_means_remote(self):
        self.assertEqual(
            detect.recommend(self.base(pythons=[{"ok": False}]))["runtime"], "remote"
        )


class Hardware(unittest.TestCase):
    def test_nvidia_fields_that_do_not_parse_skip_that_gpu(self):
        out = "NVIDIA A100, 81920, 1024\nNVIDIA GH200, [N/A], [N/A]\nRTX, Ti, 8, 24576, 0\n"
        self.assertEqual(
            [(g["name"], g["totalGiB"], g["freeGiB"]) for g in detect.nvidia_gpus(out)],
            [("NVIDIA A100", 80.0, 79.0), ("RTX, Ti, 8", 24.0, 24.0)],
        )

    def test_macos_available_memory_from_vm_stat(self):
        text = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                               10000.\n"
            "Pages active:                             90000.\n"
            "Pages inactive:                           50000.\n"
            "Pages speculative:                         2000.\n"
        )
        self.assertEqual(detect.vm_stat_available(text), 16384 * 62000)
        self.assertIsNone(detect.vm_stat_available(""))

    def test_unknown_available_memory_is_left_out(self):
        found = {
            "os": {"system": "Darwin", "name": "macOS 26.1", "arch": "arm64"},
            "cpus": 14,
            "memory": {"totalGiB": 24.0, "availableGiB": None},
            "diskFreeGiB": 100,
            "gpus": [],
            "pythons": [],
            "git": True,
            "vllmSrun": None,
            "runtime": {"ready": False, "url": "u", "portOpen": False},
            "recommendation": {"runtime": "remote", "why": "w"},
        }
        text = detect.summary(found)
        self.assertIn("24.0 GiB RAM · 100 GiB disk free", text)
        self.assertNotIn("None", text)


class Latency(unittest.TestCase):
    def found(self, **over):
        base = {
            "os": {"system": "Darwin", "appleSilicon": True},
            "runtime": {"ready": False, "url": "u"},
            "pythons": [{"ok": True}],
            "memory": {"totalGiB": 24},
            "gpus": [],
        }
        base.update(over)
        return base

    def test_apple_silicon_cpu_is_called_slow(self):
        rec = detect.recommend(self.found())
        self.assertEqual(rec["runtime"], "local-cpu")
        self.assertIn("about 2 s per question", rec["why"])
        self.assertIn("remote runtime is recommended", rec["why"])

    def test_a_slow_running_runtime_is_called_slow(self):
        ready = {"ready": True, "url": "http://127.0.0.1:8790/v1/decisions"}
        rec = detect.recommend(
            self.found(runtime={**ready, "device": "cpu", "questionMs": 2200})
        )
        self.assertIn("median 2200 ms per question", rec["why"])
        rec = detect.recommend(
            self.found(runtime={**ready, "device": "cpu", "questionMs": 180})
        )
        self.assertNotIn("slow", rec["why"])

    def test_the_probe_warns_when_the_median_is_over_a_second(self):
        def rows(*ms):
            return [{"got": {"score": 0.1, "ms": m}, "status": "ok"} for m in ms]

        for times, warned in (((900, 2100, 2300), True), ((120, 180, 2500), False)):
            err = io.StringIO()
            with self.subTest(times=times), mock.patch.object(
                runtime, "run_probes", return_value=rows(*times)
            ), mock.patch.object(runtime, "print_probes"), contextlib.redirect_stderr(
                err
            ):
                runtime.main(["probe", "--smoke"])
            self.assertEqual("slow runtime" in err.getvalue(), warned)


class Bf16(unittest.TestCase):
    def test_flags_name_native_bf16(self):
        tiger_lake = "processor\t: 0\nflags\t\t: fpu sse avx2 avx512f avx512_vnni\n"
        sapphire_rapids = "flags\t\t: avx512f avx512_bf16 amx_bf16\n"
        graviton3 = "Features\t: fp asimd sve bf16 i8mm\n"
        self.assertIs(detect.has_bf16(tiger_lake), False)
        self.assertIs(detect.has_bf16(sapphire_rapids), True)
        self.assertIs(detect.has_bf16(graviton3), True)
        self.assertIsNone(detect.has_bf16("processor\t: 0\n"))

    def test_a_cpu_without_bf16_is_said_to_run_float32(self):
        found = {
            "runtime": {"ready": False, "url": "u"},
            "pythons": [{"ok": True}],
            "memory": {"totalGiB": 16},
            "gpus": [],
            "cpuBf16": False,
        }
        rec = detect.recommend(found)
        self.assertEqual(rec["runtime"], "local-cpu")
        self.assertIn("runs in float32", rec["why"])
        self.assertNotIn("0.1 to 0.25", rec["why"])


class Disk(unittest.TestCase):
    def found(self, free, **over):
        base = {
            "runtime": {"ready": False, "url": "u"},
            "pythons": [{"ok": True}],
            "memory": {"totalGiB": 16},
            "gpus": [],
            "diskFreeGiB": free,
        }
        base.update(over)
        return base

    def test_too_little_disk_means_no_local_install(self):
        rec = detect.recommend(self.found(1.8))
        self.assertEqual(rec["runtime"], "remote")
        self.assertIn("only 1.8 GiB disk free", rec["why"])

    def test_enough_disk_installs_locally(self):
        self.assertEqual(detect.recommend(self.found(5.1))["runtime"], "local-cpu")

    def test_what_is_already_there_is_not_counted_again(self):
        found = self.found(1.8, vllmSrun={"path": "p"}, modelCached=True)
        self.assertEqual(detect.recommend(found)["runtime"], "local-cpu")

    def test_a_ready_runtime_needs_no_disk(self):
        ready = {
            "ready": True,
            "url": "http://127.0.0.1:8790/v1/decisions",
            "device": "cpu",
        }
        self.assertEqual(
            detect.recommend(self.found(0.2, runtime=ready))["runtime"], "existing"
        )

    def test_the_model_cache_follows_hf_home(self):
        with mock.patch.dict("os.environ", {"HF_HOME": "/data/hf", "HF_HUB_CACHE": ""}):
            self.assertEqual(detect.model_cache(), Path("/data/hf/hub"))


class KeepRunning(unittest.TestCase):
    """Keeping a ready runtime saves the `runtime` value that matches where it really runs."""

    def keep(self, url="http://127.0.0.1:8790/v1/decisions", device=None, saved=None):
        found = {
            "runtime": {"ready": True, "url": url, "device": device},
            "savedRuntime": saved,
            "pythons": [{"ok": True}],
            "memory": {"totalGiB": 16},
            "gpus": [],
        }
        return detect.recommend(found)

    def test_a_gpu_runtime_is_saved_as_local_gpu(self):
        for device in ("cuda", "rocm", "mps"):
            with self.subTest(device=device):
                rec = self.keep(device=device, saved="local-cpu")
                self.assertEqual(
                    (rec["runtime"], rec["setting"]), ("existing", "local-gpu")
                )
                self.assertIn(device, rec["why"])

    def test_a_cpu_runtime_is_saved_as_local_cpu(self):
        self.assertEqual(
            self.keep(device="cpu", saved="local-gpu")["setting"], "local-cpu"
        )

    def test_another_host_is_remote_whatever_its_device(self):
        self.assertEqual(
            self.keep(url="https://sr.example.com/v1/decisions", device="cuda")[
                "setting"
            ],
            "remote",
        )

    def test_an_unknown_device_keeps_the_saved_value_and_says_so(self):
        rec = self.keep(device=None, saved="local-gpu")
        self.assertEqual(rec["setting"], "local-gpu")
        self.assertIn("could not tell", rec["why"])
        rec = self.keep(device=None, saved=None)
        self.assertEqual(rec["setting"], "local-cpu")
        self.assertIn("could not tell", rec["why"])

    def test_the_summary_names_the_value_to_save(self):
        found = {
            "os": {"system": "Linux", "arch": "x86_64"},
            "cpus": 8,
            "memory": {"totalGiB": 16, "availableGiB": 8},
            "diskFreeGiB": 100,
            "gpus": [],
            "pythons": [],
            "git": True,
            "vllmSrun": None,
            "savedRuntime": None,
            "runtime": {
                "ready": True,
                "url": "http://127.0.0.1:8790/v1/decisions",
                "model": "vela3-lite",
                "device": "cuda",
                "portOpen": True,
            },
        }
        found["recommendation"] = detect.recommend(found)
        text = detect.summary(found)
        self.assertIn("ready, model vela3-lite on cuda", text)
        self.assertIn("runtime=local-gpu", text)


if __name__ == "__main__":
    unittest.main()
