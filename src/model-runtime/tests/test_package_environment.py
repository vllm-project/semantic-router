"""The environment defaults the package sets on import, before anything loads PyTorch."""

import json
import os
import subprocess
import sys

DEFAULTS = {
    "GOMP_SPINCOUNT": "10000",
    "THP_MEM_ALLOC_ENABLE": "1",
    "ONEDNN_PRIMITIVE_CACHE_CAPACITY": "8192",
    "MIOPEN_FIND_MODE": "FAST",
    "MIOPEN_LOG_LEVEL": "3",
    "OPENBLAS_NUM_THREADS": "1",
}


def environment_after_import(env: dict[str, str]) -> dict[str, str | None]:
    probe = (
        "import json, os, sys; import vllm_srun; "
        f"print(json.dumps({{k: os.environ.get(k) for k in {sorted(DEFAULTS)!r}}})); "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert out[1] == "False", "importing the package must not load PyTorch"
    return json.loads(out[0])


def test_import_sets_every_default_before_pytorch_loads() -> None:
    env = {k: v for k, v in os.environ.items() if k not in DEFAULTS}
    assert environment_after_import(env) == DEFAULTS


def test_a_value_the_caller_set_is_kept() -> None:
    env = {k: v for k, v in os.environ.items() if k not in DEFAULTS}
    env["MIOPEN_FIND_MODE"] = "NORMAL"
    assert environment_after_import(env)["MIOPEN_FIND_MODE"] == "NORMAL"
