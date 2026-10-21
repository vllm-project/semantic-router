"""Deploy a standalone Router and sample CPU usage during the accompanying Go probe."""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import time
import uuid
from pathlib import Path

MIN_CPU_SAMPLES = 2
LISTENER_COUNT = 4


def source_identity(source):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    return {"source_revision": revision, "source_dirty": bool(status.strip())}


def docker(*args):
    return subprocess.check_output(
        ["docker", *args], text=True, stderr=subprocess.STDOUT, timeout=120
    )


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def process_sample(pids, cgroup):
    result = {"at": time.monotonic(), "cpu_seconds": {}, "rss_bytes": {}}
    for label, pid in pids.items():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            result["cpu_seconds"][label] = (
                int(fields[11]) + int(fields[12])
            ) / os.sysconf("SC_CLK_TCK")
            result["rss_bytes"][label] = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            result["cpu_seconds"][label] = None
            result["rss_bytes"][label] = None
    try:
        result["router_cpu_stat"] = {
            line.split()[0]: int(line.split()[1])
            for line in (cgroup / "cpu.stat").read_text().splitlines()
        }
    except (OSError, TypeError):
        result["router_cpu_stat"] = None
    return result


def annotate(report, samples):
    for phase in report["phases"]:
        phase["cpu_usage_approx"] = {}
        for label in ("client", "backend", "router"):
            available = [
                sample
                for sample in samples
                if sample["cpu_seconds"].get(label) is not None
            ]
            if len(available) < MIN_CPU_SAMPLES:
                phase["cpu_usage_approx"][label] = None
                continue
            left = min(
                available, key=lambda item: abs(item["at"] - phase["started_monotonic"])
            )
            right = min(
                available,
                key=lambda item: abs(item["at"] - phase["finished_monotonic"]),
            )
            elapsed = right["at"] - left["at"]
            phase["cpu_usage_approx"][label] = {
                "average_cores": (
                    (right["cpu_seconds"][label] - left["cpu_seconds"][label]) / elapsed
                    if elapsed > 0
                    else None
                ),
                "window_seconds": elapsed,
                "start_offset_seconds": left["at"] - phase["started_monotonic"],
                "end_offset_seconds": right["at"] - phase["finished_monotonic"],
            }
        left = min(
            samples, key=lambda item: abs(item["at"] - phase["started_monotonic"])
        )
        right = min(
            samples, key=lambda item: abs(item["at"] - phase["finished_monotonic"])
        )
        if left["router_cpu_stat"] is not None and right["router_cpu_stat"] is not None:
            phase["router_cgroup_delta"] = {
                key: right["router_cpu_stat"][key] - left["router_cpu_stat"][key]
                for key in ("usage_usec", "nr_throttled", "throttled_usec")
                if key in left["router_cpu_stat"] and key in right["router_cpu_stat"]
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrencies", default="1,4,16,64")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        "smoke", args.source / "tools/test/smoke/riscv64-standalone.py"
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    name = "router-go-probe-" + uuid.uuid4().hex[:10]
    processes, samples, pids = {}, [], {}
    attempted = False
    receipt = {
        "passed": False,
        **source_identity(args.source),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "router_cpus": 2,
        "router_memory": "1g",
        "router_gomaxprocs": 2,
        "router_log_level": "warn",
        "helper_gomaxprocs": 2,
        "seconds": args.seconds,
        "repeats": args.repeats,
        "concurrencies": args.concurrencies,
        "helper_sha256": hashlib.sha256(args.helper.read_bytes()).hexdigest(),
        "cpu_sample_interval": 0.5,
    }
    containers_before = set(docker("ps", "-a", "--format", "{{.Names}}").splitlines())
    volumes_before = set(docker("volume", "ls", "-q").splitlines())
    try:
        image = json.loads(docker("image", "inspect", args.image))[0]
        if (
            image["Architecture"]
            != {"riscv64": "riscv64", "x86_64": "amd64"}[platform.machine()]
        ):
            raise RuntimeError("wrong image architecture")
        receipt["image"] = {
            "name": args.image,
            "id": image["Id"],
            "architecture": image["Architecture"],
        }
        receipt["load_before"] = os.getloadavg()
        backend_port, listener, grpc, metrics = port(), port(), port(), port()
        if len({backend_port, listener, grpc, metrics}) != LISTENER_COUNT:
            raise RuntimeError("port collision")
        with (args.output / "backend.log").open("w") as log:
            processes["backend"] = subprocess.Popen(
                [
                    str(args.helper),
                    "serve",
                    "--listen",
                    f"127.0.0.1:{backend_port}",
                    "--procs",
                    "2",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        smoke.wait_ready(
            f"http://127.0.0.1:{backend_port}/health", processes["backend"]
        )
        config = args.output / "config.json"
        smoke.write_configuration(config, f"127.0.0.1:{backend_port}", listener)
        attempted = True
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "host",
            "--cpus",
            "2",
            "--memory",
            "1g",
            "--pids-limit",
            "256",
            "--user",
            "65532:65532",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,mode=1777",
            "--tmpfs",
            "/app/config:rw,noexec,nosuid,uid=65532,gid=65532,mode=0700",
            "--tmpfs",
            "/app/models:rw,noexec,nosuid,uid=65532,gid=65532,mode=0700",
            "-e",
            "GOMAXPROCS=2",
            "-e",
            "SR_LOG_LEVEL=warn",
            "-v",
            f"{config.resolve()}:/app/config/config.yaml:ro",
            "--entrypoint",
            "/usr/local/bin/router",
            args.image,
            "-config",
            "/app/config/config.yaml",
            "-gateway",
            "standalone",
            "-enable-api=false",
            "-listener-address=127.0.0.1",
            "-port",
            str(grpc),
            "-metrics-port",
            str(metrics),
        )
        smoke.wait_ready(f"http://127.0.0.1:{listener}/ready")
        try:
            with smoke.DIRECT.open(
                f"http://127.0.0.1:{metrics}/metrics", timeout=3
            ) as response:
                (args.output / "metrics-before.txt").write_bytes(response.read())
        except Exception as error:
            receipt["metrics_before_error"] = type(error).__name__
        pids["router"] = int(docker("inspect", "--format", "{{.State.Pid}}", name))
        pids["backend"] = processes["backend"].pid
        try:
            line = next(
                line
                for line in Path(f"/proc/{pids['router']}/cgroup")
                .read_text()
                .splitlines()
                if line.startswith("0::")
            )
            cgroup = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")
        except (OSError, StopIteration):
            cgroup = None
        command = [
            str(args.helper),
            "bench",
            "--endpoint",
            f"http://127.0.0.1:{listener}",
            "--backend-endpoint",
            f"http://127.0.0.1:{backend_port}",
            "--seconds",
            str(args.seconds),
            "--repeats",
            str(args.repeats),
            "--concurrencies",
            args.concurrencies,
            "--procs",
            "2",
            "--output",
            str((args.output / "http.json").resolve()),
        ]
        receipt["benchmark_command"] = command
        with (args.output / "client.log").open("w") as log:
            processes["client"] = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT
            )
        pids["client"] = processes["client"].pid
        deadline = time.monotonic() + max(
            600,
            args.seconds * args.repeats * len(args.concurrencies.split(",")) * 2 + 180,
        )
        while processes["client"].poll() is None:
            samples.append(process_sample(pids, cgroup))
            if processes["backend"].poll() is not None:
                raise RuntimeError("backend exited during load")
            if time.monotonic() > deadline:
                raise TimeoutError("Go client exceeded test deadline")
            time.sleep(0.5)
        samples.append(process_sample(pids, cgroup))
        if processes["client"].returncode:
            raise RuntimeError("Go client failed; see client.log")
        report = json.loads((args.output / "http.json").read_text())
        annotate(report, samples)
        (args.output / "http-with-cpu.json").write_text(json.dumps(report) + "\n")
        receipt["passed"] = report["passed"]
        receipt["load_after"] = os.getloadavg()
        try:
            with smoke.DIRECT.open(
                f"http://127.0.0.1:{metrics}/metrics", timeout=3
            ) as response:
                (args.output / "metrics-after.txt").write_bytes(response.read())
        except Exception as error:
            receipt["metrics_after_error"] = type(error).__name__
    finally:
        errors = []
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if attempted:
            try:
                smoke.remove_router(docker, name, args.output)
            except Exception as error:
                errors.append("router: " + type(error).__name__)
        receipt["cleanup_errors"] = errors
        receipt["resources_preserved"] = containers_before == set(
            docker("ps", "-a", "--format", "{{.Names}}").splitlines()
        ) and volumes_before == set(docker("volume", "ls", "-q").splitlines())
        receipt["passed"] = (
            receipt["passed"] and not errors and receipt["resources_preserved"]
        )
        (args.output / "cpu-samples.json").write_text(json.dumps(samples) + "\n")
        (args.output / "environment.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    if not receipt["passed"]:
        raise SystemExit(1)
    print(json.dumps({"passed": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
