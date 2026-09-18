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
    "tau3": {
        "env": "TAU3",
        "module": "tau2",
        "repo": "https://github.com/sierra-research/tau2-bench.git",
        "revision": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
        "python": "3.12",
        "sparse": [
            "src",
            "data/tau2/domains/airline",
            "data/tau2/domains/retail",
            "data/tau2/domains/telecom",
        ],
    },
    "livecodebench": {
        "env": "LCB",
        "module": "lcb_runner",
        "repo": "https://github.com/LiveCodeBench/LiveCodeBench.git",
        "revision": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
        "python": "3.12",
    },
    "scicode": {
        "env": "SCICODE",
        "module": "inspect_evals",
        "repo": "https://github.com/UKGovernmentBEIS/inspect_evals.git",
        "revision": "a3837b57f40805efb4064734fbad7f93f5291c38",
        "python": "3.12",
        "extra": "scicode",
        "uv_version": "0.12.5",
    },
    "terminal-bench-2.1": {
        "env": "TERMINAL",
        "module": "harbor",
        "repo": "https://github.com/laude-institute/harbor.git",
        "revision": "eeab9f0843e6af3fea2488b308b0098d8474ca98",
        "python": "3.12",
    },
}
TASK_SOURCES = {
    "arc-agi-2": {
        "repo": "https://github.com/arcprize/ARC-AGI-2.git",
        "revision": "f3283f727488ad98fe575ea6a5ac981e4a188e49",
    },
    "terminal-bench-2.1": {
        "repo": "https://github.com/harbor-framework/terminal-bench-2-1.git",
        "revision": "7131e4375048a0e408a8fb404b5f499d726b695b",
    },
}


def home():
    return (
        Path(os.environ.get("SR_BENCH_HOME", "~/.cache/vllm-sr/sr-bench-1.0"))
        .expanduser()
        .resolve()
    )


def harness_paths(benchmark):
    spec = PACKAGES[benchmark]
    root = Path(
        os.environ.get(
            "SR_BENCH_" + spec["env"] + "_ROOT", str(home() / "harnesses" / benchmark)
        )
    )
    python = Path(
        os.environ.get(
            "SR_BENCH_" + spec["env"] + "_PYTHON",
            str(root / ".venv" / "bin" / "python"),
        )
    )
    return root, python


SCICODE_DATA_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"


def scicode_test_path():
    return (
        Path(
            os.environ.get(
                "SR_BENCH_SCICODE_TEST_DATA",
                str(home() / "assets" / "scicode" / "test_data.h5"),
            )
        )
        .expanduser()
        .resolve()
    )


def _scicode_data(python):
    path = scicode_test_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        _run(
            [
                python,
                "-c",
                "import gdown,sys; gdown.download(id='17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW', output=sys.argv[1], quiet=False)",
                temporary,
            ]
        )
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SCICODE_DATA_SHA256:
            raise ValueError(
                "Downloaded SciCode test data differs from the pinned digest"
            )
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SCICODE_DATA_SHA256:
        raise ValueError("Existing SciCode test data differs from the pinned digest")
    return path


def build_grading_image():
    """Build only the offline code grader; no subject code runs at setup time."""
    directory = home() / "sandbox"
    directory.mkdir(parents=True, exist_ok=True)
    base = "python:3.12-slim-bookworm"
    _run(["docker", "pull", base])
    images = json.loads(
        subprocess.check_output(["docker", "image", "inspect", base], text=True)
    )
    base_digest = next(ref for ref in images[0]["RepoDigests"] if "@sha256:" in ref)
    requirements = "numpy==1.26.4 scipy==1.13.1 sympy==1.12.1 h5py==3.11.0 datasets==3.6.0 tqdm==4.67.1"
    dockerfile = f"FROM {base_digest}\nRUN pip install --no-cache-dir {requirements}\n"
    (directory / "Dockerfile").write_text(dockerfile)
    _run(["docker", "build", "--iidfile", directory / "image-id", directory])
    image = (directory / "image-id").read_text().strip()
    receipt = {
        "sandbox_image": image,
        "base_image": base_digest,
        "requirements": requirements,
        "dockerfile_sha256": hashlib.sha256(dockerfile.encode()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _run(args, cwd=None):
    try:
        subprocess.run([str(a) for a in args], cwd=cwd, check=True, timeout=1800)
    except subprocess.SubprocessError as exc:
        raise ValueError(
            "Pinned dependency setup failed; inspect installer output and rerun setup after fixing the dependency"
        ) from exc


def _checkout(spec, root):
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", root])
        _run(["git", "-C", root, "remote", "add", "origin", spec["repo"]])
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode:
        origin = subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"], text=True
        ).strip()
        if origin != spec["repo"] or any(p.name != ".git" for p in root.iterdir()):
            raise ValueError("Refusing to install into an unrelated directory")
        _run(
            [
                "git",
                "-c",
                "http.version=HTTP/1.1",
                "-C",
                root,
                "fetch",
                "--depth=1",
                "--filter=blob:none",
                "origin",
                spec["revision"],
            ]
        )
        if spec.get("sparse"):
            _run(["git", "-C", root, "sparse-checkout", "set", *spec["sparse"]])
        _run(["git", "-C", root, "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        actual != spec["revision"]
        or subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=False
        ).returncode
    ):
        raise ValueError(
            "Existing harness differs from its pinned version; select a fresh SR_BENCH_HOME"
        )


def setup(benchmark="all", install=False, build_sandbox=False):
    # Source preparation also uses setup helpers.
    from .sources import (  # noqa: PLC0415
        COUNTS,
    )

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
                    raise ValueError(
                        "Optional harness installation requires uv on PATH"
                    )
                _checkout(spec, root)
                if (root / "uv.lock").is_file() and name != "livecodebench":
                    uv_command = [uv]
                    if spec.get("uv_version"):
                        uv_command += [
                            "tool",
                            "run",
                            "--from",
                            "uv==" + spec["uv_version"],
                            "uv",
                        ]
                    _run(
                        [
                            *uv_command,
                            "sync",
                            "--frozen",
                            "--no-dev",
                            "--python",
                            spec["python"],
                            *(["--extra", spec["extra"]] if spec.get("extra") else []),
                        ],
                        cwd=root,
                    )
                else:
                    if not python.is_file():
                        _run([uv, "venv", "--python", spec["python"], root / ".venv"])
                    package = str(root) + (
                        "[" + spec["extra"] + "]" if spec.get("extra") else ""
                    )
                    # LiveCodeBench's complete provider runner depends on GPU
                    # inference engines. We use only its offline grader.
                    flags = ["--no-deps"] if name == "livecodebench" else []
                    _run(
                        [
                            uv,
                            "pip",
                            "install",
                            "--python",
                            python,
                            *flags,
                            "-e",
                            package,
                        ]
                    )
                freeze = subprocess.check_output(
                    [uv, "pip", "freeze", "--python", str(python)], text=True
                )
                (root / ".sr-bench-dependencies.txt").write_text(freeze)
                if name == "scicode":
                    _scicode_data(python)
            item.update(
                {
                    "source_root": str(root),
                    "python": str(python),
                    "source_revision": spec["revision"],
                    "installed": python.is_file(),
                }
            )
            lock = root / ".sr-bench-dependencies.txt"
            item["dependencies_sha256"] = (
                hashlib.sha256(lock.read_bytes()).hexdigest()
                if lock.is_file()
                else None
            )
        else:
            item["installed"] = True
        if name in TASK_SOURCES:
            spec = TASK_SOURCES[name]
            task_root = home() / "sources" / name
            if install:
                _checkout(spec, task_root)
            item.update(
                {
                    "task_root": str(task_root),
                    "task_revision": spec["revision"],
                    "tasks_available": task_root.is_dir(),
                }
            )
        if name in {"livecodebench", "scicode"}:
            item["requirements"] = ["Docker", "digest-pinned grading image"]
            if name == "scicode":
                item["test_data"] = str(scicode_test_path())
                item["test_data_available"] = scicode_test_path().is_file()
        elif name == "terminal-bench-2.1":
            item["requirements"] = ["Docker Compose", "task images frozen by digest"]
        elif name in {"hle", "simpleqa-verified"}:
            item["requirements"] = ["registered fixed judge target"]
        elif name == "tau3":
            item["requirements"] = ["registered fixed simulator and judge targets"]
        result.append(item)
    response = {"version": "sr-bench-1.0", "home": str(home()), "benchmarks": result}
    sandbox_manifest = home() / "sandbox" / "manifest.json"
    if build_sandbox:
        response["sandbox"] = build_grading_image()
    elif sandbox_manifest.is_file():
        response["sandbox"] = json.loads(sandbox_manifest.read_text())
    return response
