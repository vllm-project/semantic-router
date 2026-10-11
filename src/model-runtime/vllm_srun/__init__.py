"""Built-in model runtime for vLLM Semantic Router.

A standalone process that serves decision, classification, embedding and
reranking models behind one HTTP contract (``api/openapi.yaml``). Model
families, execution engines, accelerators and numerics profiles are plugins
discovered through Python entry points; see ``docs/design.md``.
"""

import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# libgomp's spin count is not a package default: it depends on what the
# process serves (config.spin_count).
# PyTorch's CPU allocator reads this when it loads. Models copy their weights
# into process memory, and on 4 KiB pages where those land physically differs
# from process to process: the same CPU forward ran about a quarter slower in
# some processes. Allocations of 2 MiB or more then take transparent huge pages
# (where the kernel's madvise mode allows), and every process lays them out alike.
os.environ.setdefault("THP_MEM_ALLOC_ENABLE", "1")
# oneDNN reads this when it creates its first primitive. Packed CPU linears see a
# new row count for almost every coalesced batch; at the default 1,024 entries
# they recompiled kernels on most calls. A cached primitive costs about 26 KB.
os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "8192")
# OpenBLAS reads this when NumPy loads it. Its idle threads spin after each call
# and take the cores of the CPU device's OpenMP team: after Omni Mini's audio
# features (a few small NumPy products) its CLAP tower ran about seven times
# slower. The runtime's NumPy work is small, so one thread serves it. Only on
# x86_64, where PyTorch uses MKL: its aarch64 builds run their own matrix
# products on OpenBLAS, which this would make single-threaded.
if platform.machine() in ("x86_64", "AMD64"):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# MIOpen reads these at its first convolution. Its default find mode times the
# candidate solvers of each new shape and keeps the fastest, so cold processes
# started together pick different solvers and answer differently; FAST takes
# the find database or the heuristic and never times. It then warns once per
# new shape, so only errors are logged.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")


def _load_version() -> str:
    """The checkout's pyproject version, else the installed distribution's."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        match = re.search(
            r'^version = "([^"]+)"$', pyproject.read_text(encoding="utf-8"), re.M
        )
    except FileNotFoundError:
        match = None
    if match is not None:
        return match.group(1)
    try:
        return version("vllm-srun")
    except PackageNotFoundError:
        return "unknown"


__version__ = _load_version()
