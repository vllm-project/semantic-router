"""Pearson top-k and centered ridge. NumPy only so CI can run without a GPU."""

from __future__ import annotations

import numpy as np


class PearsonAccumulator:
    """Scalar Pearson r between flattened K or V of each (src, tgt) layer pair."""

    def __init__(self, n_src: int, n_tgt: int) -> None:
        self.n_src = n_src
        self.n_tgt = n_tgt
        self.n = np.zeros((n_src, n_tgt), dtype=np.int64)
        self.sx = np.zeros((n_src, n_tgt), dtype=np.float64)
        self.sy = np.zeros((n_src, n_tgt), dtype=np.float64)
        self.sxx = np.zeros((n_src, n_tgt), dtype=np.float64)
        self.syy = np.zeros((n_src, n_tgt), dtype=np.float64)
        self.sxy = np.zeros((n_src, n_tgt), dtype=np.float64)

    def add(self, src_layers: list[np.ndarray], tgt_layers: list[np.ndarray]) -> None:
        src = src_layers[: self.n_src]
        tgt = tgt_layers[: self.n_tgt]
        widths = {tuple(t.shape[1:]) for t in src + tgt}
        if len(widths) != 1:
            raise ValueError(
                f"Pearson needs one (n_kv, head_dim) for both models, got {sorted(widths)}"
            )
        nseq = min(int(src[0].shape[0]), int(tgt[0].shape[0]))
        x = np.stack([t[:nseq].reshape(-1).astype(np.float64) for t in src])
        y = np.stack([t[:nseq].reshape(-1).astype(np.float64) for t in tgt])
        feat = x.shape[1]
        self.n += feat
        self.sx += x.sum(1)[:, None]
        self.sy += y.sum(1)[None, :]
        self.sxx += (x * x).sum(1)[:, None]
        self.syy += (y * y).sum(1)[None, :]
        self.sxy += x @ y.T

    def corr(self) -> np.ndarray:
        n = np.maximum(self.n, 1)
        mx = self.sx / n
        my = self.sy / n
        cov = self.sxy / n - mx * my
        vx = np.maximum(self.sxx / n - mx * mx, 1e-12)
        vy = np.maximum(self.syy / n - my * my, 1e-12)
        return cov / np.sqrt(vx * vy)

    def topk(self, k: int) -> list[list[int]]:
        r = np.abs(self.corr())
        out: list[list[int]] = []
        for j in range(self.n_tgt):
            order = np.argsort(-r[:, j])[:k]
            out.append([int(i) for i in order])
        return out


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
        sw = self.sw
        xtx_c = self.xtx - np.outer(self.sx, self.sx) / sw
        xty_c = self.xty - np.outer(self.sx, self.sy) / sw
        a = xtx_c + alpha * np.eye(self.dx)
        weight, *_ = np.linalg.lstsq(a, xty_c, rcond=None)
        bias = self.sy / sw - (self.sx / sw) @ weight
        return weight.astype(np.float32), bias.astype(np.float32)
