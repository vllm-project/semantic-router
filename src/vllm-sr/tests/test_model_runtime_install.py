"""Static contracts of the model runtime's CI install (tools/make/model-runtime.mk)."""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_RUNTIME_MK = REPO_ROOT / "tools" / "make" / "model-runtime.mk"
EDITABLE = "[test,reference,onnx,multimodal]"


def install_recipe() -> str:
    """The model-runtime-install target's recipe lines."""
    text = MODEL_RUNTIME_MK.read_text(encoding="utf-8")
    found = re.search(r"^model-runtime-install:.*\n((?:\t.*\n)+)", text, re.M)
    assert found, "model-runtime-install lost its recipe"
    return found.group(1)


def gate_code() -> str:
    """The Python the version gate runs before choosing its branch."""
    gate = re.search(r"-c '([^']*version_info[^']*)'", install_recipe())
    assert gate, "the version gate lost its condition"
    return gate.group(1)


def branch_editable(branch: str) -> list[str]:
    """The editable install lines of the gate's then or else branch."""
    head, separator, tail = install_recipe().partition("else \\")
    assert separator, "the version gate lost its else branch"
    segment = head if branch == "then" else tail
    return [line for line in segment.splitlines() if EDITABLE in line]


def test_the_ci_install_constrains_the_runtime_with_its_lock() -> None:
    editable = [line for line in install_recipe().splitlines() if EDITABLE in line]

    # The images install against requirements-lock.txt (#4630); the CI path
    # resolved live PyPI without it, so a new upstream release changed what CI
    # tested without changing what shipped (#4846). The lock is compiled for
    # the image's Python, and test-local.yml installs on 3.11 through
    # vllm-sr-test-integration, so the constraint is gated on that interpreter
    # and every other version installs without pins it cannot resolve.
    assert len(editable) == 2, editable
    locked = [line for line in editable if "-c" in line]
    assert len(locked) == 1, editable
    assert '-c "$(MODEL_RUNTIME_DIR)/requirements-lock.txt"' in locked[0]
    assert any("-c" not in line for line in editable)


def test_the_version_gate_selects_the_branch_by_exit_status() -> None:
    # The gate runs as a shell condition, so only its exit status picks the
    # branch: a comparison that evaluates false without exiting non-zero still
    # selects the locked one on every interpreter.
    code = gate_code()
    for version, branch in (
        ((3, 12, 4), "then"),
        ((3, 11, 9), "else"),
        ((3, 13, 0), "else"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.version_info = {version!r}; exec({code!r})",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == (
            0 if branch == "then" else 1
        ), f"{version} did not take the {branch} branch: {result.stderr}"


def test_the_lock_rides_the_branch_the_gate_selects() -> None:
    then_lines = branch_editable("then")
    else_lines = branch_editable("else")
    assert len(then_lines) == 1 and "-c" in then_lines[0], then_lines
    assert "requirements-lock.txt" in then_lines[0], then_lines
    assert len(else_lines) == 1 and "-c" not in else_lines[0], else_lines


def test_the_ci_install_keeps_pytorch_on_the_cpu_index() -> None:
    # The lock omits PyTorch, so its wheel keeps coming from the CPU index
    # rather than the constraint.
    assert (
        'pip install "$(MODEL_RUNTIME_TORCH)" --index-url "$(MODEL_RUNTIME_TORCH_INDEX)"'
        in install_recipe()
    )
