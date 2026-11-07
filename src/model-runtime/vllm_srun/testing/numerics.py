"""Numeric comparison for the reference-parity tests.

The native CPU path and the Transformers reference share every kernel except
``causal_conv1d`` and ``flash-linear-attention``, which fall back to their
reference PyTorch implementations whenever the compiled extension is absent.
Those fallbacks reassociate the same sums rather than computing something else,
so the two paths agree to a few float32 steps instead of bit for bit.
"""

from __future__ import annotations

import math
import struct

# Steps of a float32, not an absolute difference: the same reassociation costs a
# proportional number of steps whatever the magnitude of the score, so one
# constant covers scores near zero and scores near one. Measured against the
# reference on arm64/NEON, where both fallback kernels are active, the worst
# case is 8 steps out of 19 scores, while a runtime with no fallback kernel
# (qwen3) matches exactly. 32 leaves 4x headroom over that worst case, and a
# real regression (a different kernel, a changed readout, packing on a host that
# cannot pack) moves scores by about 1e-3, three orders of magnitude further.
PARITY_TOLERANCE_ULPS = 32

# The bit pattern of the most negative float32, which is where the ordered key
# of a negative value bottoms out.
_INT32_MIN = -(2**31)


def _bits(value: float) -> int:
    """The IEEE-754 binary32 bit pattern of ``value``, after rounding to float32."""
    return struct.unpack("<i", struct.pack("<f", value))[0]


def _ordered(bits: int) -> int:
    """Map a float32 bit pattern onto a monotonic integer line.

    Adjacent representable floats get adjacent keys and both zeros share one, so
    the difference between two keys counts the float32 values between them.
    """
    return bits if bits >= 0 else _INT32_MIN - bits


def _scalar_ulps(left: float, right: float) -> int:
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError(f"no ULP distance for non-finite {left!r} and {right!r}")
    return abs(_ordered(_bits(left)) - _ordered(_bits(right)))


def float32_ulps(left, right) -> int:
    """Largest float32 ULP distance between two equally shaped nested sequences.

    A shape that differs raises instead of comparing the wrong pair, so a score
    that appears, vanishes or changes row fails loudly.
    """
    left_is_sequence = isinstance(left, (list, tuple))
    right_is_sequence = isinstance(right, (list, tuple))
    if left_is_sequence != right_is_sequence:
        raise ValueError(
            f"shape mismatch: {type(left).__name__} against {type(right).__name__}"
        )
    if not left_is_sequence:
        return _scalar_ulps(left, right)
    if len(left) != len(right):
        raise ValueError(f"length mismatch: {len(left)} against {len(right)}")
    return max(
        (float32_ulps(one, other) for one, other in zip(left, right, strict=True)),
        default=0,
    )
