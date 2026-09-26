from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Torch is optional in the contract test environment.
# ruff: noqa: PLC0415

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


class HookLayoutTests(unittest.TestCase):
    def test_as_bshd_layouts(self) -> None:
        torch = self._torch()
        from src.training.kv_mapper.hooks import as_bshd

        bshd = torch.randn(1, 5, 8, 16)
        bhmd = bshd.transpose(1, 2)
        self.assertEqual(tuple(as_bshd(bshd, 8, 16).shape), (1, 5, 8, 16))
        self.assertEqual(tuple(as_bshd(bhmd, 8, 16).shape), (1, 5, 8, 16))
        flat = torch.randn(1, 5, 8 * 16)
        self.assertEqual(tuple(as_bshd(flat, 8, 16).shape), (1, 5, 8, 16))

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest("torch not installed") from exc
        return torch


if __name__ == "__main__":
    unittest.main()
