"""Per-head OLS layer scores and centered ridge. NumPy only for CPU CI."""

from __future__ import annotations

import numpy as np


def ols_r2_per_head(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Single-source affine OLS R² for each KV head, across tokens and dimensions."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 3 or source.shape[0] < 2:
        raise ValueError(f"OLS needs matching (tokens, heads, dim), got {source.shape} and {target.shape}")
    scores = np.empty(source.shape[1], dtype=np.float64)
    for head in range(source.shape[1]):
        x, y = source[:, head, :], target[:, head, :]
        x_centered = x - x.mean(axis=0)
        y_centered = y - y.mean(axis=0)
        weight, *_ = np.linalg.lstsq(x_centered, y_centered, rcond=None)
        residual = y_centered - x_centered @ weight
        total = np.square(y_centered).sum()
        scores[head] = 1.0 - np.square(residual).sum() / total if total > 0 else 0.0
    return scores


class RidgeAccumulator:
    """Accumulate X'X and X'Y for Y ≈ X W + b. X, Y are (n, dx), (n, dy)."""

    def __init__(self, dx: int, dy: int) -> None:
        self.dx = dx
        self.dy = dy
        self.xtx = np.zeros((dx, dx), dtype=np.float64)
        self.xty = np.zeros((dx, dy), dtype=np.float64)
        self.sx = np.zeros(dx, dtype=np.float64)
        self.sy = np.zeros(dy, dtype=np.float64)
        self.sw = 0.0
        self.n = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.ascontiguousarray(x, dtype=np.float64)
        y64 = np.ascontiguousarray(y, dtype=np.float64)
        if x64.ndim != 2 or y64.ndim != 2 or x64.shape != (y64.shape[0], self.dx) or y64.shape[1] != self.dy:
            raise ValueError(f"Ridge needs matching rows and widths ({self.dx}, {self.dy}), got {x64.shape} and {y64.shape}")
        n = x64.shape[0]
        self.xtx += x64.T @ x64
        self.xty += x64.T @ y64
        self.sx += x64.sum(0)
        self.sy += y64.sum(0)
        self.sw += float(n)
        self.n += n

    def solve_affine(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        if self.n == 0 or self.sw <= 0:
            raise RuntimeError(
                f"Ridge has no rows (dx={self.dx}, dy={self.dy})"
            )
        if alpha < 0:
            raise ValueError("ridge alpha must be nonnegative")
        sw = self.sw
        xtx_c = self.xtx - np.outer(self.sx, self.sx) / sw
        xty_c = self.xty - np.outer(self.sx, self.sy) / sw
        a = xtx_c + alpha * np.eye(self.dx)
        weight, *_ = np.linalg.lstsq(a, xty_c, rcond=None)
        bias = self.sy / sw - (self.sx / sw) @ weight
        return weight.astype(np.float32), bias.astype(np.float32)
