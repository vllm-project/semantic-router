"""Exercise the cross-architecture Make recipe's evidence and failure boundary."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

COMMAND = r"""
import json, os, pathlib, sys

root = pathlib.Path(os.environ["RISCV_CONTRACT_ROOT"])
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]

def option(key):
    return args[args.index(key) + 1]

if name == "go" and args[:2] == ["tool", "test2json"]:
    sys.stdout.write(sys.stdin.read())
    sys.exit(0)
if name == "go" and args[0] == "run":
    stage = "download"
elif name == "go" and args[0] == "build":
    stage = "router-build"
elif name == "go":
    stage = "go-cross-build" if "-c" in args else "host-parity"
elif name == "cargo" and "--list" in args:
    stage = "rust-list"
elif name == "cargo" and args[0] == "test":
    stage = "rust-" + option("--lib")
elif name == "cargo":
    stage = "cargo-cross-build" if "--target" in args else "cargo-host-build"
elif name == "qemu":
    if "-test.list" in args:
        stage = "binding-list"
    else:
        selection = option("-test.run")
        stage = ("ffi" if "FFI" in selection else
                 "parity" if "Parity" in selection else "binding")
else:
    stage = name
with (root / "calls.jsonl").open("a") as output:
    output.write(json.dumps({"stage": stage, "args": args}) + "\n")
if stage == os.environ.get("RISCV_CONTRACT_FAIL"):
    print("injected RISC-V failure: " + stage, file=sys.stderr)
    sys.exit(37)
if stage == "download":
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    manifest = pathlib.Path(option("--manifest"))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"models": [{"path": str(model)}]}))
elif stage == "rust-list":
    print("first: test\nsecond: test")
elif stage.startswith("rust-"):
    print("test " + stage[5:] + " ... ok\n")
    print("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured")
elif stage == "cargo-cross-build":
    lib = root / "lib" / "libcandle_semantic_router.so"
    lib.parent.mkdir(exist_ok=True)
    lib.touch()
elif stage == "binding-list":
    print("TestOwnedNativeSequenceLifecycle")
else:
    print(json.dumps({"Action": "pass", "Test": stage}))
"""


class RiscvMakeContractsTests(unittest.TestCase):
    def test_evidence_files_and_canonical_model_provisioning(self) -> None:
        result, calls, reports = self._run()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls[0]["stage"], "download")
        self.assertIn("riscv", calls[0]["args"])
        self.assertIn("--download", calls[0]["args"])
        self.assertEqual(calls[-1]["stage"], "router-smoke")
        self.assertEqual(
            set(reports),
            {
                "models.json",
                "host-parity.jsonl",
                "rust-list.txt",
                "rust-required.txt",
                "minimal-pattern.txt",
                "minimal-skip.txt",
                "rust-0.log",
                "rust-1.log",
                "ffi.log",
                "qemu-ffi.jsonl",
                "parity.log",
                "qemu-parity.jsonl",
                "binding-list.txt",
                "binding.log",
                "qemu-minimal.jsonl",
            },
        )
        self.assertEqual(reports["rust-required.txt"], "first\nsecond\n")
        self.assertIn("Owned.*", reports["minimal-pattern.txt"])
        self.assertEqual(
            reports["minimal-skip.txt"],
            "^TestOwnedNativeMaintainedHallucinationWithoutLabelMetadata$\n",
        )
        binding = next(call for call in calls if call["stage"] == "binding")
        skip_index = binding["args"].index("-test.skip")
        self.assertEqual(
            binding["args"][skip_index + 1], reports["minimal-skip.txt"].strip()
        )

    def test_each_execution_failure_stops_before_later_stages(self) -> None:
        for failure in (
            "download",
            "cargo-host-build",
            "host-parity",
            "cargo-cross-build",
            "rust-list",
            "rust-first",
            "rust-second",
            "go-cross-build",
            "ffi",
            "parity",
            "binding-list",
            "binding",
            "router-build",
            "router-smoke",
        ):
            with self.subTest(failure=failure):
                result, calls, _ = self._run(failure)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(calls[-1]["stage"], failure, calls)
                self.assertIn("injected RISC-V failure", result.stdout + result.stderr)

    def _run(self, failure: str = "") -> tuple:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in (
                "bin",
                "candle-binding",
                "src/semantic-router",
                "tools/ci",
                "tools/docker",
                "sysroot",
            ):
                (root / folder).mkdir(parents=True, exist_ok=True)
            (root / "Makefile").write_text(
                f"include {ROOT / 'tools/make/rust.mk'}\n"
                "LOG_TARGET = :\nNATIVE_ENV = RISCV_TEST_NATIVE=1\n"
            )
            command = f"#!{sys.executable}\n{COMMAND}"
            for name in ("go", "cargo", "qemu", "rustup", "cross-gcc"):
                executable = root / "bin" / name
                executable.write_text(command)
                executable.chmod(0o755)
            smoke = root / "tools/ci/riscv-qemu-router-smoke.sh"
            smoke.write_text(
                f'#!/bin/sh\n"{sys.executable}" "{root / "bin/router-smoke"}"\n'
            )
            (root / "bin/router-smoke").write_text(command)
            (root / "tools/docker/check-native-abi.sh").write_text(
                "#!/bin/sh\nexit 0\n"
            )
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in {"MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "MAKEFILES"}
            }
            environment.update(
                PATH=str(root / "bin") + os.pathsep + environment.get("PATH", ""),
                RISCV_CONTRACT_ROOT=str(root),
                RISCV_CONTRACT_FAIL=failure,
            )
            result = subprocess.run(
                [
                    "make",
                    "--no-print-directory",
                    "test-riscv-qemu",
                    "SHELL=/bin/sh",
                    "RISCV_GNU_CC=cross-gcc",
                    f"RISCV_QEMU={root / 'bin/qemu'}",
                    f"RISCV_SYSROOT={root / 'sysroot'}",
                    f"RISCV_CANDLE_LIBDIR={root / 'lib'}",
                    f"MODEL_TEST_REPORT_DIR={root / 'reports'}",
                    f"MODEL_TEST_MODELS_DIR={root / 'models'}",
                    "RISCV_QEMU_LIB_TESTS=first second",
                ],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl").read_text().splitlines()
            ]
            reports = {
                path.name: path.read_text() for path in (root / "reports").glob("*")
            }
            return result, calls, reports


if __name__ == "__main__":
    unittest.main()
