"""Check model placement against the selected frontend image platform."""

import re

from cli.consts import PLATFORM_AMD, PLATFORM_NVIDIA

BUILTIN_ACCELERATORS = {
    "cpu": frozenset({"", PLATFORM_AMD, PLATFORM_NVIDIA}),
    "rocm": frozenset({PLATFORM_AMD}),
    "cuda": frozenset({PLATFORM_NVIDIA}),
    "xpu": frozenset(),
    "mps": frozenset(),
}
_PLATFORM_FOR = {"rocm": PLATFORM_AMD, "cuda": PLATFORM_NVIDIA}
_DEVICE = re.compile(r"(?P<accelerator>[A-Za-z_][\w-]*)(?::\d+)?\Z")


def check_device(device: str, platform: str, where: str = "--device") -> None:
    """Refuse a built-in accelerator that the platform's image cannot run."""

    platform = "" if platform == "cpu" else platform
    match = _DEVICE.fullmatch(device)
    if device == "auto" or match is None:
        return
    accelerator = match.group("accelerator").lower()
    platforms = BUILTIN_ACCELERATORS.get(accelerator)
    if platforms is None or platform in platforms:
        return
    if accelerator == "mps":
        raise ValueError(
            f"{where} {device}: mps needs macOS's Metal, which no Linux container "
            "gets; engine mode runs on the CPU there "
            "(https://github.com/vllm-project/semantic-router/issues/4636)"
        )
    if not platforms:
        raise ValueError(
            f"{where} {device}: no router image runs {accelerator}; use cpu, or "
            "rocm or cuda with --platform amd or nvidia"
        )
    raise ValueError(
        f"{where} {device} needs --platform {_PLATFORM_FOR[accelerator]}, whose "
        f"image runs {accelerator}"
    )
