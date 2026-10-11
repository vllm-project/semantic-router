"""Static contracts of the model runtime's CI install (tools/make/model-runtime.mk)."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_RUNTIME_MK = REPO_ROOT / "tools" / "make" / "model-runtime.mk"


def install_recipe() -> str:
    """The model-runtime-install target's recipe lines."""
    text = MODEL_RUNTIME_MK.read_text(encoding="utf-8")
    found = re.search(r"^model-runtime-install:.*\n((?:\t.*\n)+)", text, re.M)
    assert found, "model-runtime-install lost its recipe"
    return found.group(1)


def test_the_ci_install_constrains_the_runtime_with_its_lock() -> None:
    editable = next(
        line
        for line in install_recipe().splitlines()
        if "[test,reference,onnx,multimodal]" in line
    )

    # The images install against requirements-lock.txt (#4630); the CI path
    # resolved live PyPI without it, so a new upstream release changed what CI
    # tested without changing what shipped (#4846).
    assert '-c "$(MODEL_RUNTIME_DIR)/requirements-lock.txt"' in editable


def test_the_ci_install_keeps_pytorch_on_the_cpu_index() -> None:
    # The lock omits PyTorch, so its wheel keeps coming from the CPU index
    # rather than the constraint.
    assert (
        'pip install "$(MODEL_RUNTIME_TORCH)" --index-url "$(MODEL_RUNTIME_TORCH_INDEX)"'
        in install_recipe()
    )
