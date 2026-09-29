import argparse
import importlib
import re
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

launch = importlib.import_module("v2.27b.launch")
launch3 = importlib.import_module("v2.27b.m4b.launch3")
SCRIPT = Path(launch3.__file__).with_name("run_ff_arm.sh")


def fake_sysfs(root: Path) -> Path:
    for pci, node in launch3.ALLOWED_GPUS.values():
        device = root / "devices" / pci
        device.mkdir(parents=True)
        (root / "drm" / node).mkdir(parents=True)
        (root / "drm" / node / "device").symlink_to(device)
    return root / "drm"


class Captured(Exception):
    pass


def parser_options(main, *args) -> dict[str, dict]:
    """Every action of the parser ``main`` builds, captured when it parses."""
    captured = {}

    def capture(self, *_, **__):
        captured.update(
            {
                action.option_strings[0] if action.option_strings else action.dest: {
                    "required": action.required,
                    "type": action.type,
                    "default": action.default,
                    "nargs": action.nargs,
                }
                for action in self._actions
                if action.dest != "help"
            }
        )
        raise Captured

    with mock.patch.object(
        argparse.ArgumentParser, "parse_args", capture
    ), mock.patch.object(sys, "argv", ["prog"]):
        try:
            main(*args)
        except Captured:
            pass
    return captured


class GpuTest(unittest.TestCase):
    def test_parse_gpus(self):
        self.assertEqual(launch3.parse_gpus("2,0,1"), [0, 1, 2])
        for bad in ("3", "0,0", "", "a", "0,4"):
            with self.assertRaises(ValueError):
                launch3.parse_gpus(bad)

    def test_render_node_pci_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            drm = fake_sysfs(Path(tmp))
            self.assertEqual(
                [launch3.render_node(gpu, drm).name for gpu in (0, 1, 2)],
                ["renderD129", "renderD137", "renderD145"],
            )
            (drm / "renderD137" / "device").unlink()
            (drm / "renderD137" / "device").symlink_to(
                Path(tmp) / "devices" / "0000:83:00.0"
            )
            with self.assertRaises(ValueError):
                launch3.render_node(1, drm)
            with self.assertRaises(ValueError):
                launch3.render_node(3, drm)

    def test_cli_matches_launch_except_gpus(self):
        single = parser_options(launch.main)
        multi = parser_options(launch3.main, [])
        self.assertIn("--gpu", single)
        self.assertEqual(single.pop("--gpu")["type"], int)
        self.assertTrue(multi.pop("--gpus")["required"])
        self.assertEqual(single, multi)


class CreateArgvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = dict(
            name="x",
            workdir="/code",
            command=["python3", "-V"],
            mount=[f"{self.tmp.name}:/code"],
        )
        self.devices = [
            Path("/dev/dri") / node for _, node in launch3.ALLOWED_GPUS.values()
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_three_gpus(self):
        argv = launch3.create_argv(Namespace(env=["A=1"], **self.base), self.devices)
        devices = [argv[i + 1] for i, item in enumerate(argv) if item == "--device"]
        self.assertEqual(
            devices,
            [
                "/dev/kfd",
                "/dev/dri/renderD129",
                "/dev/dri/renderD137",
                "/dev/dri/renderD145",
            ],
        )
        for name in ("ROCR", "HIP", "CUDA"):
            self.assertIn(f"{name}_VISIBLE_DEVICES=0,1,2", argv)
        for item in (
            "NCCL_SOCKET_IFNAME=lo",
            "GLOO_SOCKET_IFNAME=lo",
            "memlock=-1:-1",
            "A=1",
        ):
            self.assertIn(item, argv)
        self.assertEqual(argv[argv.index("--network") + 1], "none")
        self.assertEqual(argv[argv.index("--shm-size") + 1], launch3.MULTI_GPU_SHM)
        self.assertIn(
            "type=bind,src=%s,dst=/code,readonly" % Path(self.tmp.name).resolve(), argv
        )
        self.assertEqual(argv[-3:], [launch3.IMAGE_ID, "python3", "-V"])

    def test_one_gpu_matches_launch_plus_cuda_visible(self):
        args = Namespace(env=["A=1"], **self.base)
        argv = launch3.create_argv(args, self.devices[:1])
        at = argv.index("CUDA_VISIBLE_DEVICES=0")
        self.assertEqual(argv[at - 1 : at + 1], ["-e", "CUDA_VISIBLE_DEVICES=0"])
        self.assertEqual(
            argv[: at - 1] + argv[at + 1 :], launch.create_argv(args, self.devices[0])
        )

    def test_rejects_secret_env_and_bad_mount(self):
        with self.assertRaises(ValueError):
            launch3.create_argv(
                Namespace(env=["HF_TOKEN=abc"], **self.base), self.devices
            )
        bad = dict(self.base, mount=[f"{self.tmp.name}:/code:xx"])
        with self.assertRaises(ValueError):
            launch3.create_argv(Namespace(env=[], **bad), self.devices)


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lock = self.root / "gpu1.lock"
        self.lock.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def owner(self, name: str = "owner") -> dict:
        return launch.read_lease(self.lock / name)

    def test_take_over_idle_foreign_owner(self):
        (self.lock / "owner").write_text("track=9b-clm\nstatus=idle\n")
        result = launch3.take_lease(
            1,
            purpose="m4b A1-s1",
            expected_end=None,
            status="reserved-idle",
            root=self.root,
            now="2026-09-29T10:00:00Z",
        )
        self.assertEqual(
            result["moved_previous_owner"],
            str(self.lock / "owner.prev-20260929T100000Z"),
        )
        self.assertEqual(self.owner("owner.prev-20260929T100000Z")["track"], "9b-clm")
        text = (self.lock / "owner").read_text()
        self.assertIn("track=27b-m4b\n", text)
        self.assertEqual(self.owner()["status"], "reserved-idle")
        launch3.take_lease(
            1,
            purpose="m4b A1-s1 done",
            expected_end=None,
            status="idle",
            root=self.root,
        )
        self.assertEqual(self.owner()["status"], "idle")
        self.assertEqual(self.owner()["start_utc"], "2026-09-29T10:00:00Z")
        self.assertEqual(len(list(self.lock.glob("owner.prev-*"))), 1)

    def test_refuses_busy_owner(self):
        (self.lock / "owner").write_text("track=eval\nstatus=running\n")
        with self.assertRaises(ValueError):
            launch3.take_lease(
                1, purpose="p", expected_end=None, status="idle", root=self.root
            )
        (self.lock / "owner").write_text("track=27b-m4b\nstatus=running\ncontainer=c\n")
        with self.assertRaises(ValueError):
            launch3.take_lease(
                1, purpose="p", expected_end=None, status="idle", root=self.root
            )
        with self.assertRaises(ValueError):
            launch3.take_lease(
                4, purpose="p", expected_end=None, status="idle", root=self.root
            )

    def test_write_lease_and_cotenants(self):
        with self.assertRaises(ValueError):
            launch3.write_lease(1, {"status": "running"}, self.root)
        (self.lock / "owner").write_text("track=9b-clm\nstatus=idle\n")
        with self.assertRaises(ValueError):
            launch3.write_lease(1, {"status": "running"}, self.root)
        (self.lock / "owner").write_text("track=27b-m4b\nstatus=reserved-idle\n")
        (self.lock / "owner.eval").write_text("track=eval\nstatus=released\n")
        (self.lock / "owner.prev-20260929T001102Z").write_text(
            "track=x\nstatus=running\n"
        )
        launch3.check_cotenants(1, self.root)
        launch3.write_lease(1, {"status": "running", "container": "a"}, self.root)
        with self.assertRaises(ValueError):
            launch3.write_lease(1, {"status": "running", "container": "b"}, self.root)
        launch3.write_lease(
            1, {"status": "reserved-idle", "container": None}, self.root
        )
        self.assertNotIn("container", self.owner())
        launch3.write_lease(1, {"status": "running"}, self.root, shared="s1")
        launch3.write_lease(1, {"status": "running"}, self.root, shared="s2")
        self.assertEqual(self.owner()["shared_containers"], "s1,s2")
        (self.lock / "owner.eval").write_text("track=eval\nstatus=running\n")
        with self.assertRaises(ValueError):
            launch3.check_cotenants(1, self.root)


class ReceiptTest(unittest.TestCase):
    def test_gpu_hours_and_fields(self):
        self.assertEqual(launch3.gpu_hours(1800, 3), 1.5)
        args = Namespace(name="n", purpose="p", cap_hours=2.0, shared=False)
        devices = [Path("/dev/dri/renderD129"), Path("/dev/dri/renderD145")]
        receipt = launch3.build_receipt(
            args,
            [0, 2],
            devices,
            ["create"],
            cid="c",
            started_utc="t",
            elapsed=5400.0,
            timed_out=False,
            exit_code=0,
            log_sha256=None,
        )
        self.assertEqual(receipt["gpu_hours"], 3.0)
        self.assertEqual(receipt["gpu_indices"], [0, 2])
        self.assertEqual(receipt["pci_buses"], ["0000:83:00.0", "0000:93:00.0"])
        self.assertEqual(receipt["render_nodes"], ["renderD129", "renderD145"])
        self.assertEqual(receipt["track"], "27b-m4b")


class DriverScriptTest(unittest.TestCase):
    def test_bash_syntax_and_no_heredoc(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text()
        self.assertIsNone(re.search(r"(?<!<)<<(?!<)", text))
        self.assertLess(
            text.index('echo "$(date -u +%FT%TZ) start ARM='), text.index("CODE=")
        )
        self.assertIn("PYTHONPATH=/code:/opt/decision-fla", text)


if __name__ == "__main__":
    unittest.main()
