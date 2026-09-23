"""Fail-closed contract tests for the optional Eos ROCm convolution path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_runtime.eos_rocm_conv import install_eos_rocm_conv  # noqa: E402


def test_missing_owned_kernel_keeps_the_library_path_untouched() -> None:
    receipt = install_eos_rocm_conv(
        object(),
        torch=object(),
        transformers=object(),
        modeling_qwen35=object(),
        kernel=None,
    )

    assert receipt.installed is False
    assert receipt.reason == "optional_kernel_unavailable"


def test_unqualified_runtime_is_a_safe_noop_before_model_introspection() -> None:
    torch = SimpleNamespace(version=SimpleNamespace(hip=None, git_version=None))
    receipt = install_eos_rocm_conv(
        object(),
        torch=torch,
        transformers=SimpleNamespace(__version__="0"),
        modeling_qwen35=object(),
        kernel=lambda *args, **kwargs: None,
    )

    assert receipt.installed is False
    assert receipt.reason == "unqualified_runtime"
