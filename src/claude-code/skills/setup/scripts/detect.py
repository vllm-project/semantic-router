#!/usr/bin/env python3
"""Look at this machine before asking anything: OS, CPU, memory, GPU, Python, an installed
vllm-srun, and whether a runtime already answers at the configured URL. Installs nothing.
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
import settings

VENV = Path.home() / ".local/share/vsr/venv"
GIB = 1024**3
GPU_DEVICES = ("cuda", "rocm", "mps", "xpu", "npu")
# Weights on disk, measured or published; accuracy beyond lite is unmeasured.
MODELS = [
    ("vllm-sr/vela3-lite", 1.8, 4),
    ("vllm-sr/vela3-mini", 8.7, 12),
    ("vllm-sr/vela3-flash", 16, 22),
    ("vllm-sr/vela3-turbo", 50, 64),
]


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
    info = {"system": system, "arch": platform.machine(), "release": platform.release()}
    if system == "Linux":
        os_release = Path("/etc/os-release")
        if os_release.exists():
            found = re.search(r'^PRETTY_NAME="?([^"\n]*)', os_release.read_text(), re.M)
            info["name"] = found.group(1) if found else "Linux"
        info["wsl"] = "microsoft" in platform.release().lower()
        info["systemdUser"] = (
            shutil.which("systemctl") is not None
            and Path(f"/run/user/{os.getuid()}/systemd").exists()
        )
    elif system == "Darwin":
        info["name"] = f"macOS {platform.mac_ver()[0]}"
        info["appleSilicon"] = platform.machine() == "arm64"
    return info


def memory() -> dict:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = dict(re.findall(r"^(\w+):\s+(\d+) kB", meminfo.read_text(), re.M))
        return {
            "totalGiB": round(int(values["MemTotal"]) * 1024 / GIB, 1),
            "availableGiB": round(int(values.get("MemAvailable", 0)) * 1024 / GIB, 1),
        }
    total = _run(["sysctl", "-n", "hw.memsize"])
    return {
        "totalGiB": round(int(total) / GIB, 1) if total else None,
        "availableGiB": None,
    }


def gpus() -> list[dict]:
    found = []
    if shutil.which("nvidia-smi"):
        out = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in (out or "").splitlines():
            name, total, used = [part.strip() for part in line.split(",")]
            found.append(
                {
                    "vendor": "nvidia",
                    "device": "cuda",
                    "name": name,
                    "totalGiB": round(int(total) / 1024, 1),
                    "freeGiB": round((int(total) - int(used)) / 1024, 1),
                }
            )
    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
        try:
            cards = json.loads(out or "{}")
        except json.JSONDecodeError:
            cards = {}
        for card in cards.values():
            total = int(card.get("VRAM Total Memory (B)", 0))
            used = int(card.get("VRAM Total Used Memory (B)", 0))
            found.append(
                {
                    "vendor": "amd",
                    "device": "rocm",
                    "name": card.get("Card series")
                    or card.get("Card SKU")
                    or "AMD GPU",
                    "totalGiB": round(total / GIB, 1),
                    "freeGiB": round((total - used) / GIB, 1),
                }
            )
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        found.append(
            {
                "vendor": "apple",
                "device": "mps",
                "name": _run(["sysctl", "-n", "machdep.cpu.brand_string"])
                or "Apple Silicon",
                "totalGiB": None,
                "freeGiB": None,
                "unifiedMemory": True,
            }
        )
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


def vllm_srun() -> dict | None:
    for path in (VENV / "bin/vllm-srun", shutil.which("vllm-srun")):
        if path and Path(path).exists():
            python = Path(path).parent / "python"
            version = (
                _run(
                    [
                        str(python),
                        "-c",
                        "import importlib.metadata as m; print(m.version('vllm-srun'))",
                    ]
                )
                if python.exists()
                else None
            )
            return {"path": str(path), "version": version}
    return None


def port_open(url: str) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=1
        ):
            return True
    except OSError:
        return False


def configured(plugin_id: str | None) -> tuple[str, str | None]:
    """The runtime URL vSR runs with, and the `runtime` value saved under `plugin_id` (None when unsaved)."""
    if not plugin_id:
        return runtime.DEFAULT_URL, None
    saved = settings.load(settings.settings_path())[1]
    url = str(settings.effective_options(saved, plugin_id)["runtimeUrl"])
    return url, settings.saved_options(saved, plugin_id).get("runtime")


def keep_setting(found: dict) -> tuple[str, str]:
    """The `runtime` value that describes the ready runtime, and why."""
    r = found["runtime"]
    if not runtime.on_this_machine(r["url"]):
        return "remote", "it runs on another machine"
    if r.get("device") == "cpu":
        return "local-cpu", "it runs on cpu"
    if r.get("device") in GPU_DEVICES:
        return "local-gpu", f"it runs on {r['device']}"
    kept = found.get("savedRuntime") or "local-cpu"
    return kept, f"could not tell its device, so {kept} is kept" + (
        " from the saved settings" if found.get("savedRuntime") else ""
    )


def recommend(found: dict) -> dict:
    """Where the classifier should run, from what this machine has."""
    if found["runtime"]["ready"]:
        setting, why = keep_setting(found)
        return {
            "runtime": "existing",
            "setting": setting,
            "why": f"a runtime already answers at {found['runtime']['url']}; {why}",
        }
    good_python = any(p["ok"] for p in found["pythons"])
    ram = found["memory"]["totalGiB"] or 0
    for gpu in found["gpus"]:
        if (
            gpu["device"] in ("cuda", "rocm")
            and (gpu["freeGiB"] or 0) >= MODELS[0][2]
            and good_python
        ):
            fits = [name for name, _, need in MODELS if (gpu["freeGiB"] or 0) >= need]
            return {
                "runtime": "local-gpu",
                "device": gpu["device"],
                "model": MODELS[0][0],
                "larger": fits[1:],
                "why": f"{gpu['name']} has {gpu['freeGiB']} GiB free",
            }
    if good_python and ram >= 4:
        why = f"{os.cpu_count()} cores and {ram} GiB RAM run vela3-lite in 0.1 to 0.25 s per question"
        if any(g["device"] == "mps" for g in found["gpus"]):
            why += "; Apple GPU (mps) also works but is unmeasured"
        elif any(g["device"] in ("cuda", "rocm") for g in found["gpus"]):
            why += "; the GPU has too little free memory right now"
        return {
            "runtime": "local-cpu",
            "device": "cpu",
            "model": MODELS[0][0],
            "why": why,
        }
    missing = "Python 3.10 or later" if not good_python else "4 GiB of RAM"
    return {
        "runtime": "remote",
        "why": f"this machine lacks {missing} for a local runtime",
    }


def detect(plugin_id: str | None) -> dict:
    url, saved_runtime = configured(plugin_id)
    found = {
        "os": os_info(),
        "cpus": os.cpu_count(),
        "memory": memory(),
        "gpus": gpus(),
        "pythons": pythons(),
        "git": shutil.which("git") is not None,
        "vllmSrun": vllm_srun(),
        "runtime": {**runtime.health(url), "portOpen": port_open(url)},
        "diskFreeGiB": round(shutil.disk_usage(Path.home()).free / GIB, 1),
        "savedRuntime": saved_runtime,
    }
    found["recommendation"] = recommend(found)
    return found


def summary(found: dict) -> str:
    o, m, r = found["os"], found["memory"], found["runtime"]
    gpu = (
        "; ".join(
            f"{g['name']} ({g['device']}"
            + (
                f", {g['freeGiB']} of {g['totalGiB']} GiB free)"
                if g["totalGiB"]
                else ")"
            )
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
    srun = found["vllmSrun"]
    rec = found["recommendation"]
    state = (
        (
            f"ready, model {r['model']}"
            + (f" on {r['device']}" if r.get("device") else "")
        )
        if r["ready"]
        else (
            "port open but not a ready runtime"
            if r["portOpen"]
            else "nothing listening"
        )
    )
    lines = [
        f"machine   {o.get('name', o['system'])} · {o['arch']} · {found['cpus']} cores · {m['totalGiB']} GiB RAM ({m['availableGiB']} available) · {found['diskFreeGiB']} GiB disk free",
        f"gpu       {gpu}",
        f"python    {py}"
        + (
            "" if found["git"] else " · git missing (pip needs it to install vllm-srun)"
        ),
        f"vllm-srun {srun['path'] + ' ' + str(srun['version']) if srun else 'not installed'}",
        f"runtime   {r['url']}: {state}",
        f"suggest   {rec['runtime']}"
        + (f" · {rec['device']} · {rec['model']}" if rec.get("model") else "")
        + (f" · keep runtime={rec['setting']}" if rec.get("setting") else "")
        + f" ({rec['why']})",
    ]
    if rec.get("larger"):
        lines.append(
            f"          also fits: {', '.join(rec['larger'])} (accuracy gain unmeasured)"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--id", help="read runtimeUrl from the settings saved under this plugin id"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    found = detect(args.id)
    print(json.dumps(found, indent=2) if args.json else summary(found))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
