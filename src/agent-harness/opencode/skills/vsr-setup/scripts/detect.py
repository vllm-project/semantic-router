#!/usr/bin/env python3
"""Look at this machine before asking anything: OS, CPU, memory, disk, GPU, Python, the installed
runtime (its commit and multimodal extra), how it is kept running, and whether one answers at the
URL vSR asks. Installs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
import runtime

HOME = Path.home() / ".local/share/vsr"
VENV = HOME / "venv"
# The runtime vSR was tested against; local.py installs it, so the two must name the same commit.
COMMIT = "97446cc6baa185f491e0d86f8f4c0133bf993354"
# Commits setup pinned before, which it upgrades. The Claude Code plugin installs into the same venv
# and may pin a newer commit, so any other commit is left as it is rather than moved backwards.
OLDER = {"579e0ca2a0b1b03580eada4213da84fe68db57c2"}
GIB = 1024**3
GPU_DEVICES = ("cuda", "rocm", "mps", "xpu", "npu")
MODEL = "vllm-sr/vela3-lite"
# vela3-lite's weights on disk, and the GPU memory it needs with room to run.
MODEL_GIB, MODEL_GPU_GIB = 1.8, 4
# The venv with CPU torch measured 1.1 GiB on Linux, with CUDA torch 5.5 GiB.
VENV_GIB = {"cpu": 1.1, "gpu": 6.0}
# Room for pip's downloads and temporary files while installing.
SPARE_GIB = 1.0
LABEL = "vsr-runtime"


def _run(command: list[str], timeout: float = 10) -> str | None:
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        return done.stdout.strip() if done.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def os_info() -> dict:
    system = platform.system()
    info = {"system": system, "arch": platform.machine()}
    if system == "Linux":
        os_release = Path("/etc/os-release")
        if os_release.exists():
            found = re.search(r'^PRETTY_NAME="?([^"\n]*)', os_release.read_text(), re.M)
            info["name"] = found.group(1) if found else "Linux"
        info["systemdUser"] = (
            shutil.which("systemctl") is not None
            and Path(f"/run/user/{os.getuid()}/systemd").exists()
        )
    elif system == "Darwin":
        info["name"] = f"macOS {platform.mac_ver()[0]}"
    info["arm"] = platform.machine().lower() in ("arm64", "aarch64")
    return info


def memory() -> dict:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = dict(re.findall(r"^(\w+):\s+(\d+) kB", meminfo.read_text(), re.M))
        total, available = values.get("MemTotal"), values.get("MemAvailable")
        return {
            "totalGiB": round(int(total) * 1024 / GIB, 1) if total else None,
            "availableGiB": (
                round(int(available) * 1024 / GIB, 1) if available else None
            ),
        }
    total = _run(["sysctl", "-n", "hw.memsize"])
    return {
        "totalGiB": round(int(total) / GIB, 1) if total else None,
        "availableGiB": None,
    }


def model_cache() -> Path:
    """Where the Hugging Face hub keeps downloaded models."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "huggingface/hub"


def free_gib(path: Path) -> float:
    """Free space on the disk that holds `path`, or would hold it once created."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return round(shutil.disk_usage(path).free / GIB, 1)


def model_cached(cache: Path) -> bool:
    # vela3-lite loads d3-lite as its backbone.
    return all(
        (cache / f"models--vllm-sr--{name}").exists()
        for name in ("vela3-lite", "d3-lite")
    )


def disk_need(found: dict, kind: str) -> float:
    """GiB a local install still needs: the venv, the model, and spare room."""
    venv = 0 if found.get("installed") else VENV_GIB[kind]
    model = 0 if found.get("modelCached") else MODEL_GIB
    return round(venv + model + SPARE_GIB, 1)


def disk_short(found: dict, kind: str) -> str | None:
    """Why the disk is too full for a local install, or None when it fits (or is unknown)."""
    free = found.get("diskFreeGiB")
    need = disk_need(found, kind)
    if free is None or free >= need:
        return None
    return f"only {free} GiB disk free, and a local runtime needs about {need} GiB"


def _whole(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def nvidia_gpus(out: str) -> list[dict]:
    """GPUs from nvidia-smi's CSV. One whose memory reads [N/A] shares the system's memory, as a
    GB10's does: its free memory is the system's available memory, filled in by `detect`.
    """
    found = []
    for line in out.splitlines():
        # The name may hold commas; the two numbers never do.
        parts = [part.strip() for part in line.rsplit(",", 2)]
        if len(parts) != 3 or not parts[0]:
            continue
        name, total, used = parts[0], _whole(parts[1]), _whole(parts[2])
        if total is None or used is None:
            found.append(
                {
                    "device": "cuda",
                    "name": name,
                    "unified": True,
                    "totalGiB": None,
                    "freeGiB": None,
                }
            )
            continue
        found.append(
            {
                "device": "cuda",
                "name": name,
                "totalGiB": round(total / 1024, 1),
                "freeGiB": round((total - used) / 1024, 1),
            }
        )
    return found


def rocm_gpus(out: str) -> list[dict]:
    try:
        cards = json.loads(out or "{}")
    except json.JSONDecodeError:
        return []
    found = []
    for card in cards.values() if isinstance(cards, dict) else []:
        if not isinstance(card, dict):
            continue
        total = _whole(card.get("VRAM Total Memory (B)"))
        used = _whole(card.get("VRAM Total Used Memory (B)"))
        if total is None or used is None:
            continue
        found.append(
            {
                "device": "rocm",
                "name": card.get("Card series") or card.get("Card SKU") or "AMD GPU",
                "totalGiB": round(total / GIB, 1),
                "freeGiB": round((total - used) / GIB, 1),
            }
        )
    return found


def gpus() -> list[dict]:
    found = []
    if shutil.which("nvidia-smi"):
        found += nvidia_gpus(
            _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ]
            )
            or ""
        )
    if shutil.which("rocm-smi"):
        found += rocm_gpus(
            _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
            or ""
        )
    # The Apple GPU (mps) is left out: the runtime cannot place vela3 on it yet.
    return found


def pythons() -> list[dict]:
    seen, found = set(), []
    for name in ("python3", "python3.13", "python3.12", "python3.11", "python3.10"):
        path = shutil.which(name)
        if not path or os.path.realpath(path) in seen:
            continue
        seen.add(os.path.realpath(path))
        version = _run(
            [path, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"]
        )
        if version:
            major, minor = (int(x) for x in version.split(".")[:2])
            found.append(
                {"path": path, "version": version, "ok": (major, minor) >= (3, 10)}
            )
    return found


def _imports(python: Path, module: str) -> bool:
    try:
        return (
            subprocess.run(
                [str(python), "-c", f"import {module}"],
                capture_output=True,
                timeout=60,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def installed_commit(venv: Path) -> str | None:
    """The commit the venv's vllm-srun was installed from, read from pip's direct_url.json; its
    version number does not tell commits apart."""
    for record in sorted(
        venv.glob("lib/python3*/site-packages/vllm_srun-*.dist-info/direct_url.json")
    ):
        try:
            found = json.loads(record.read_text())
        except (OSError, ValueError):
            continue
        commit = (found.get("vcs_info") or {}).get("commit_id")
        if commit:
            return str(commit)
    return None


def installed(venv: Path = VENV) -> dict | None:
    """The runtime installed in the venv: its commit and whether it reads images (Pillow, from the
    multimodal extra); None when it is not installed."""
    if not (venv / "bin/vllm-srun").exists():
        return None
    python = venv / "bin/python"
    return {
        "venv": str(venv),
        "commit": installed_commit(venv),
        "multimodal": python.exists() and _imports(python, "PIL"),
    }


def command_of(pid: int) -> str:
    """The command line of a process: from /proc on Linux, from `ps` elsewhere (macOS)."""
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            return cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return ""
    try:
        done = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def started_pid(home: Path = HOME, command=command_of) -> int | None:
    """The pid in runtime.pid while that process is a live vllm-srun; None otherwise, so a pid a
    reboot handed to another process never counts. A runtime binds its port only after it starts
    loading, so this is also what tells a second `start` that one is already on its way.
    """
    try:
        pid = int((home / "runtime.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if alive(pid) and "vllm-srun" in command(pid) else None


def kept_by() -> str | None:
    """How a runtime on this machine is kept running: the service setup wrote, or a live runtime
    setup started for this login."""
    if (Path.home() / f".config/systemd/user/{LABEL}.service").exists():
        return "systemd"
    if (Path.home() / f"Library/LaunchAgents/local.{LABEL}.plist").exists():
        return "launchd"
    if started_pid() is not None:
        return "pid"
    return None


def port_open(url: str) -> bool:
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", port), timeout=1
        ):
            return True
    except OSError:
        return False


def needs_upgrade(found: dict) -> str | None:
    """Why the installed runtime should be installed again, or None when it is current."""
    have = found.get("installed")
    if not have or not have.get("commit"):
        # Not installed, or installed from a local checkout or wheel: someone's own build, left as it is.
        return None
    if have["commit"] in OLDER:
        return f"installed from {have['commit'][:9]}, not {COMMIT[:9]}"
    if have["commit"] != COMMIT:
        return None
    if not have.get("multimodal"):
        return "installed without its multimodal extra, so it answers no image"
    return None


def serve_device(label: str | None, gpus_found: list[dict]) -> str | None:
    """The `--device` local.py serves on, from the device a runtime reports (`cuda:0`, `cpu`). ROCm
    torch reports `cuda` too, so an AMD GPU with no NVIDIA one means `rocm`."""
    if not label:
        return None
    kind = label.split(":")[0].lower()
    if kind == "cpu":
        return "cpu"
    if kind in ("cuda", "rocm", "hip"):
        devices = {g["device"] for g in gpus_found}
        return "rocm" if "rocm" in devices and "cuda" not in devices else "cuda"
    return None


def recommend(found: dict) -> dict:
    """Where the model should run, from what this machine has."""
    r = found["runtime"]
    upgrade = needs_upgrade(found)
    if r["ready"]:
        here = runtime.on_this_machine(r["url"])
        slow = runtime.slow_note(r.get("questionMs"))
        why = f"a runtime already answers at {r['url']}" + (
            f" on {r['device']}" if r.get("device") else ""
        )
        if here and upgrade:
            return {
                "runtime": "upgrade",
                "device": serve_device(r.get("device"), found["gpus"]),
                "why": f"{why}, but it is {upgrade}",
            }
        return {"runtime": "existing", "why": why, **({"slow": slow} if slow else {})}
    good_python = any(p["ok"] for p in found["pythons"])
    ram = found["memory"]["totalGiB"] or 0
    for gpu in found["gpus"]:
        if (
            gpu["device"] in ("cuda", "rocm")
            and (gpu["freeGiB"] or 0) >= MODEL_GPU_GIB
            and good_python
            and disk_short(found, "gpu") is None
        ):
            return {
                "runtime": "local-gpu",
                "device": gpu["device"],
                "why": f"{gpu['name']} has {gpu['freeGiB']} GiB free",
            }
    short = disk_short(found, "cpu")
    if good_python and ram >= 4 and short:
        return {
            "runtime": "remote",
            "why": f"{short}; free some space or use a remote runtime",
        }
    if good_python and ram >= 4:
        if found["os"].get("arm"):
            why = (
                f"{ram} GiB RAM fit vela3-lite, but an ARM CPU takes seconds a question (about 4 s "
                "on a GB10's), so the apps will often step aside: a remote runtime is recommended if "
                "you have one"
            )
        else:
            why = (
                f"{found['cpus']} cores and {ram} GiB RAM run vela3-lite in float32, about 1 s a short "
                "question on an idle desktop CPU"
            )
        if any(g["device"] in ("cuda", "rocm") for g in found["gpus"]):
            why += "; the GPU has too little free memory or disk right now"
        return {"runtime": "local-cpu", "device": "cpu", "why": why}
    missing = "Python 3.10 or later" if not good_python else "4 GiB of RAM"
    return {
        "runtime": "remote",
        "why": f"this machine lacks {missing} for a local runtime",
    }


def detect(url: str) -> dict:
    cache = model_cache()
    mem = memory()
    found = {
        "os": os_info(),
        "cpus": os.cpu_count(),
        "memory": mem,
        "gpus": [
            (
                {**g, "totalGiB": mem["totalGiB"], "freeGiB": mem["availableGiB"]}
                if g.get("unified")
                else g
            )
            for g in gpus()
        ],
        "pythons": pythons(),
        "git": shutil.which("git") is not None,
        "installed": installed(),
        "keptBy": kept_by(),
        "runtime": {**runtime.health(url), "portOpen": port_open(url)},
        # The venv and the model may sit on different disks; the fuller one decides.
        "diskFreeGiB": min(free_gib(VENV), free_gib(cache)),
        "modelCached": model_cached(cache),
    }
    if found["runtime"]["ready"]:
        models = found["runtime"]["models"]
        # A runtime with several models answers only a request that names one.
        found["runtime"]["questionMs"] = runtime.time_question(
            url, models[0]["id"] if len(models) > 1 else None
        )
    found["recommendation"] = recommend(found)
    return found


def summary(found: dict) -> str:
    o, m, r = found["os"], found["memory"], found["runtime"]
    gpu = (
        "; ".join(
            f"{g['name']} ({g['device']}, {g['freeGiB']} of {g['totalGiB']} GiB free"
            + (", shared with the CPU)" if g.get("unified") else ")")
            for g in found["gpus"]
        )
        or "none"
    )
    py = (
        ", ".join(
            f"{p['version']}{'' if p['ok'] else ' (too old)'}" for p in found["pythons"]
        )
        or "none"
    )
    have = found["installed"]
    srun = (
        "not installed"
        if not have
        else f"{have['venv']} · "
        + (
            f"commit {have['commit'][:9]}"
            + (
                ""
                if have["commit"] == COMMIT
                else (
                    f" (setup pins {COMMIT[:9]})"
                    if have["commit"] in OLDER
                    else " (not one setup knows, perhaps newer: left as it is)"
                )
            )
            if have["commit"]
            else "not installed from git (a local build, left as it is)"
        )
        + ("" if have["multimodal"] else " · no multimodal extra (reads no image)")
        + (f" · kept by {found['keptBy']}" if found["keptBy"] else "")
    )
    model = r["models"][0]["id"] if r.get("models") else None
    state = (
        (
            f"ready, model {model}"
            + (f" on {r['device']}" if r.get("device") else "")
            + (f", {r['questionMs']} ms a question" if r.get("questionMs") else "")
        )
        if r["ready"]
        else (
            "loading"
            if r.get("loading")
            else (
                "port open but not a ready runtime"
                if r["portOpen"]
                else "nothing listening"
            )
        )
    )
    rec = found["recommendation"]
    lines = [
        f"machine   {o.get('name', o['system'])} · {o['arch']} · {found['cpus']} cores · {m['totalGiB']} GiB RAM"
        + (f" ({m['availableGiB']} available)" if m["availableGiB"] is not None else "")
        + f" · {found['diskFreeGiB']} GiB disk free",
        f"gpu       {gpu}",
        f"python    {py}"
        + (
            "" if found["git"] else " · git missing (pip needs it to install vllm-srun)"
        ),
        f"vllm-srun {srun}",
        f"runtime   {r['url']}: {state}",
        f"suggest   {rec['runtime']}"
        + (f" · {rec['device']}" if rec.get("device") else "")
        + f" ({rec['why']})",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=runtime.LOCAL_URL,
        help="the runtime URL vSR asks, as vsr_settings says it",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    found = detect(args.url)
    print(json.dumps(found, indent=2) if args.json else summary(found))
    slow = found["recommendation"].get("slow")
    if slow:
        print(f"warning: {slow}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
