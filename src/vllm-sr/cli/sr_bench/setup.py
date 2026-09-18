"""Explicit installation and inspection of pinned optional benchmark harnesses."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

# Harness and task versions are part of sr-bench 1.0, never a moving branch.
PACKAGES = {
    "tau3": {"env": "TAU3", "module": "tau2", "repo": "https://github.com/sierra-research/tau3-bench.git", "revision": "fc0055dc4e0a316c3f83133267fbd6faaa770992", "python": "3.12", "sparse": ["src", "data/tau2/domains/airline", "data/tau2/domains/retail", "data/tau2/domains/telecom"]},
    "livecodebench": {"env": "LCB", "module": "lcb_runner", "repo": "https://github.com/LiveCodeBench/LiveCodeBench.git", "revision": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24", "python": "3.12"},
    "scicode": {"env": "SCICODE", "module": "inspect_evals", "repo": "https://github.com/UKGovernmentBEIS/inspect_evals.git", "revision": "a3837b57f40805efb4064734fbad7f93f5291c38", "python": "3.12", "extra": "scicode"},
    "terminal-bench-2.1": {"env": "TERMINAL", "module": "harbor", "repo": "https://github.com/laude-institute/harbor.git", "revision": "eeab9f0843e6af3fea2488b308b0098d8474ca98", "python": "3.12"},
}
TASK_SOURCES = {
    "arc-agi-2": {"repo": "https://github.com/arcprize/ARC-AGI-2.git", "revision": "f3283f727488ad98fe575ea6a5ac981e4a188e49"},
    "terminal-bench-2.1": {"repo": "https://github.com/harbor-framework/terminal-bench-2-1.git", "revision": "7131e4375048a0e408a8fb404b5f499d726b695b"},
}


def home():
    return Path(os.environ.get("SR_BENCH_HOME", "~/.cache/vllm-sr/sr-bench-1.0")).expanduser().resolve()


def harness_paths(benchmark):
    spec = PACKAGES[benchmark]
    root = Path(os.environ.get("SR_BENCH_" + spec["env"] + "_ROOT", str(home() / "harnesses" / benchmark)))
    python = Path(os.environ.get("SR_BENCH_" + spec["env"] + "_PYTHON", str(root / ".venv" / "bin" / "python")))
    return root, python


def _run(args, cwd=None):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True)


def _checkout(spec, root):
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", root])
        _run(["git", "-C", root, "remote", "add", "origin", spec["repo"]])
        _run(["git", "-c", "http.version=HTTP/1.1", "-C", root, "fetch", "--depth=1", "--filter=blob:none", "origin", spec["revision"]])
        if spec.get("sparse"):
            _run(["git", "-C", root, "sparse-checkout", "set", *spec["sparse"]])
        _run(["git", "-C", root, "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != spec["revision"] or subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"]).returncode:
        raise ValueError("Existing harness differs from its pinned version; select a fresh SR_BENCH_HOME")


def setup(benchmark="all", install=False):
    from .sources import COUNTS
    names = list(COUNTS) if benchmark == "all" else [benchmark]
    if any(name not in COUNTS for name in names):
        raise ValueError("Unknown benchmark")
    result = []
    for name in names:
        item = {"benchmark": name, "model_requests": 0}
        if name in PACKAGES:
            spec = PACKAGES[name]
            root, python = harness_paths(name)
            if install:
                uv = shutil.which("uv")
                if not uv:
                    raise ValueError("Optional harness installation requires uv on PATH")
                _checkout(spec, root)
                if (root / "uv.lock").is_file() and name == "tau3":
                    _run([uv, "sync", "--frozen", "--no-dev", "--python", spec["python"]], cwd=root)
                else:
                    if not python.is_file():
                        _run([uv, "venv", "--python", spec["python"], root / ".venv"])
                    package = str(root) + ("[" + spec["extra"] + "]" if spec.get("extra") else "")
                    _run([uv, "pip", "install", "--python", python, "-e", package])
                freeze = subprocess.check_output([uv, "pip", "freeze", "--python", str(python)], text=True)
                (root / ".sr-bench-dependencies.txt").write_text(freeze)
            item.update({"source_root": str(root), "python": str(python), "source_revision": spec["revision"], "installed": python.is_file()})
            lock = root / ".sr-bench-dependencies.txt"
            item["dependencies_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
        else:
            item["installed"] = True
        if name in TASK_SOURCES:
            spec = TASK_SOURCES[name]
            task_root = home() / "sources" / name
            if install:
                _checkout(spec, task_root)
            item.update({"task_root": str(task_root), "task_revision": spec["revision"], "tasks_available": task_root.is_dir()})
        if name in {"livecodebench", "scicode"}:
            item["requirements"] = ["Docker", "digest-pinned grading image"]
        elif name == "terminal-bench-2.1":
            item["requirements"] = ["Docker Compose", "task images frozen by digest"]
        elif name in {"hle", "simpleqa-verified"}:
            item["requirements"] = ["registered fixed judge target"]
        elif name == "tau3":
            item["requirements"] = ["registered fixed simulator target"]
        result.append(item)
    return {"version": "sr-bench-1.0", "home": str(home()), "benchmarks": result}
