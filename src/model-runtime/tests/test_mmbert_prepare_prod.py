"""Optional full-weight loads for issue #2440 on a developer machine.

These tests are skipped when checkpoints are not present under ``models/`` at
the repository root. Each load runs in its own child process so a hang can be
killed without wedging pytest. CI does not ship multi-gigabyte weights.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = "VLLM_SR_MMBERT_PACKAGE"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(__file__).resolve().parents[1]
PROD_DEADLINE_SECONDS = 900

LOAD = f"""
import os
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.runtime import Runtime

runtime = Runtime(
    ServeConfig(
        models=(ModelConfig(model=os.environ[{PACKAGE!r}], device="cpu"),),
        load_attempts=1,
    )
)
runtime.start(background=False)
runtime.stop()
"""

PRODUCTION_PACKAGES = [
    pytest.param(
        REPO_ROOT / "models/mmbert32k-intent-classifier-merged",
        id="mmbert32k-intent",
    ),
    pytest.param(
        REPO_ROOT / "models/mmbert32k-pii-detector-merged",
        id="mmbert32k-pii",
    ),
    pytest.param(
        REPO_ROOT / "models/mmbert32k-jailbreak-detector-merged",
        id="mmbert32k-jailbreak",
    ),
    pytest.param(
        REPO_ROOT / "models/Vela-1.0-Encoder-307M-Embedding",
        id="vela-1.0-encoder-307m-embedding",
    ),
    pytest.param(
        REPO_ROOT / "models/Vela-1.0-Encoder-307M-Domain",
        id="vela-1.0-encoder-307m-domain",
    ),
]


def _require_weights(root: Path) -> None:
    weights = root / "model.safetensors"
    if not weights.is_file():
        pytest.skip(f"production weights not present: {weights}")


def _child_env(package: Path) -> dict[str, str]:
    env = os.environ.copy()
    env[PACKAGE] = str(package)
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(RUNTIME_ROOT), env.get("PYTHONPATH", "")) if item
    )
    return env


def _run_bounded_load(package: Path, deadline: int) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", LOAD],
        env=_child_env(package),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = proc.communicate(timeout=deadline)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        pytest.fail(
            f"{package.name} load exceeded {deadline}s; parent killed it\n{output}"
        )
    assert proc.returncode == 0, output


@pytest.mark.prod_weights
@pytest.mark.parametrize("package", PRODUCTION_PACKAGES)
def test_production_checkpoint_load_is_bounded(package: Path) -> None:
    _require_weights(package)
    _run_bounded_load(package, PROD_DEADLINE_SECONDS)
