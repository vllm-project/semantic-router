"""Choose a bounded CPU thread budget inside the Decision container."""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_CPU_THREAD_CAP = 8
MAX_CPU_THREADS = 256
_MAX_CPU_THREAD_DIGITS = len(str(MAX_CPU_THREADS))
_CGROUP_CPU_MAX_FIELDS = 2
CPU_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
_REQUESTED_CPU_THREADS = "DECISION_CPU_THREADS"
_POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")


def configure_cpu_threads(*, cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """Set Torch and BLAS limits before either framework is imported."""

    requested = os.environ.get(_REQUESTED_CPU_THREADS)
    if requested is None:
        threads = min(DEFAULT_CPU_THREAD_CAP, available_cpu_count(cgroup_root))
    else:
        if (
            len(requested) > _MAX_CPU_THREAD_DIGITS
            or _POSITIVE_DECIMAL.fullmatch(requested) is None
        ):
            raise ValueError("DECISION_CPU_THREADS must be an integer from 1 to 256")
        threads = int(requested)
        if threads > MAX_CPU_THREADS:
            raise ValueError("DECISION_CPU_THREADS must be an integer from 1 to 256")
    for key in CPU_THREAD_ENVIRONMENT:
        os.environ[key] = str(threads)
    return threads


def available_cpu_count(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """Count the CPUs allowed by affinity and cgroup quota, when visible."""

    limits: list[int] = []
    try:
        affinity = len(os.sched_getaffinity(0))
        if affinity > 0:
            limits.append(affinity)
    except (AttributeError, OSError):
        pass
    logical_cpus = os.cpu_count()
    if logical_cpus is not None and logical_cpus > 0:
        limits.append(logical_cpus)

    cpu_max = _read_words(cgroup_root / "cpu.max")
    if len(cpu_max) == _CGROUP_CPU_MAX_FIELDS and cpu_max[0] != "max":
        quota = _quota_cpu_count(cpu_max[0], cpu_max[1])
        if quota is not None:
            limits.append(quota)
    for directory in (cgroup_root, cgroup_root / "cpu", cgroup_root / "cpu,cpuacct"):
        quota_words = _read_words(directory / "cpu.cfs_quota_us")
        period_words = _read_words(directory / "cpu.cfs_period_us")
        if len(quota_words) == 1 and len(period_words) == 1:
            quota = _quota_cpu_count(quota_words[0], period_words[0])
            if quota is not None:
                limits.append(quota)
    return min(limits) if limits else 1


def _read_words(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="ascii").split()
    except (OSError, UnicodeError):
        return []


def _quota_cpu_count(quota_text: str, period_text: str) -> int | None:
    try:
        quota = int(quota_text)
        period = int(period_text)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, quota // period)
