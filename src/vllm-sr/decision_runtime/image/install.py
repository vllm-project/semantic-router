"""Construct and inspect the two family environments in a Decision OCI image."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path("/opt/vllm-sr")
FAMILIES = {"vela": "4.57.6", "qwen35": "5.17.0"}


def run(*arguments: str) -> None:
    subprocess.run(arguments, check=True)


def output(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True).strip()


def base_torch_site(backend: str) -> tuple[Path, Path]:
    import torch

    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if backend == "cpu" and (hip or cuda):
        raise RuntimeError("CPU image must contain a CPU-only Torch wheel")
    if backend == "cuda" and (hip or not cuda):
        raise RuntimeError("CUDA image requires a CUDA Torch wheel")
    if backend == "rocm" and not hip:
        raise RuntimeError("ROCm image requires a preinstalled HIP Torch build")
    if backend in {"cpu", "cuda"} and not str(torch.__version__).startswith("2.12.0+"):
        raise RuntimeError("CPU/CUDA Torch wheel differs from the pinned build")

    torch_file = Path(torch.__file__).resolve(strict=True)
    distribution_site = Path(
        importlib.metadata.distribution("torch").locate_file("")
    ).resolve(strict=True)
    source_site = torch_file.parent.parent
    if not source_site.is_dir() or not distribution_site.is_dir():
        raise RuntimeError("Torch installation has no stable import path")
    return source_site, distribution_site


def install_family(
    family: str,
    *,
    wheel: Path,
    backend: str,
    inherited_sites: tuple[Path, ...],
    expected_torch: Path,
) -> None:
    environment = ROOT / "venvs" / family
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"missing Decision family Python: {python}")

    # A ROCm base can keep Torch in /opt/venv rather than a system site. The
    # family environments inherit that exact installation without copying or
    # allowing pip to replace the hardware-qualified Torch wheel.
    site = Path(
        output(
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        )
    )
    site.mkdir(parents=True, exist_ok=True)
    (site / "decision-base-torch.pth").write_text(
        "".join(f"{path}\n" for path in inherited_sites), encoding="utf-8"
    )

    requirements = ROOT / f"requirements-{family}.txt"
    packages = [str(wheel), "-r", str(requirements)]
    if family == "qwen35" and backend in {"rocm", "cuda"}:
        packages.append(f"flash-linear-attention[{backend}]==0.5.2")
    run(str(python), "-m", "pip", "install", "--no-cache-dir", *packages)

    smoke = (
        "import pathlib, sys, torch, transformers, safetensors, fastapi, uvicorn; "
        "from decision_runtime.catalog_adapter import resolve_decision_runtime_model; "
        "from decision_runtime.entrypoint import parse_launch_args; "
        "resolve_decision_runtime_model('llm-semantic-router/Decision-1.0-Kai-0.6B'); "
        "resolve_decision_runtime_model('llm-semantic-router/Decision-1.0-Eos-0.8B'); "
        f"assert transformers.__version__ == {FAMILIES[family]!r}; "
        f"assert str(pathlib.Path(torch.__file__).resolve()) == {str(expected_torch)!r}; "
        "assert pathlib.Path(sys.executable).resolve().is_file()"
    )
    run(str(python), "-c", smoke)
    if family == "qwen35" and backend in {"rocm", "cuda"}:
        run(
            str(python),
            "-c",
            "import importlib.metadata, importlib.util; "
            "assert importlib.metadata.version('flash-linear-attention') == '0.5.2'; "
            "assert importlib.metadata.version('fla-core') == '0.5.2'; "
            "assert importlib.util.find_spec('fla') is not None",
        )
    run(str(python), "-m", "decision_runtime.entrypoint", "--help")


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Decision OCI image requires Python 3.12")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "rocm"), required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    arguments = parser.parse_args()

    wheels = sorted(arguments.wheel_dir.glob("vllm_sr-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            "Decision image requires exactly one locally built CLI wheel"
        )
    source_site, distribution_site = base_torch_site(arguments.backend)
    sites = tuple(dict.fromkeys((source_site, distribution_site)))
    torch_path = Path(
        output(sys.executable, "-c", "import torch; print(torch.__file__)")
    ).resolve(strict=True)
    for family in FAMILIES:
        install_family(
            family,
            wheel=wheels[0],
            backend=arguments.backend,
            inherited_sites=sites,
            expected_torch=torch_path,
        )


if __name__ == "__main__":
    main()
