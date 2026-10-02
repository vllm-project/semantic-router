"""Attention masks for packed prefix trees (torch only; shared by the kernel and its benchmark)."""

from __future__ import annotations

from typing import Any


def pack_mask(mask: Any) -> Any:
    """Bool [B, N, N] (row sees key) -> int32 [B, N, ceil(N/32)] words; bit j of word w = key 32w+j."""
    import torch

    B, N, _ = mask.shape
    W = (N + 31) // 32
    padded = torch.zeros(B, N, W * 32, dtype=torch.int64, device=mask.device)
    padded[:, :, :N] = mask.to(torch.int64)
    weights = torch.bitwise_left_shift(
        torch.ones(32, dtype=torch.int64, device=mask.device),
        torch.arange(32, device=mask.device),
    )
    words = (padded.view(B, N, W, 32) * weights).sum(-1)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32).contiguous()


def unpack_mask(words: Any, N: int) -> Any:
    """Inverse of ``pack_mask``."""
    import torch

    bits = torch.arange(32, device=words.device)
    unpacked = (words.to(torch.int64)[..., None] >> bits) & 1
    return unpacked.flatten(-2)[..., :N].bool()


def tree_mask(torch: Any, N: int, dev: Any) -> Any:
    """Ancestor mask of a synthetic prefix tree packed into N rows (row sees key).

    A shared prefix (45% of the rows), three question segments and two or three candidate
    tails per question; every row sees its ancestors' rows and, causally, its own segment.
    """
    prefix = int(N * 0.45)
    seg = [(0, prefix, -1)]  # (start, end, parent segment index or -1)
    rest = N - prefix
    q_len = max(1, rest // 10)
    pos = prefix
    questions = []
    for _ in range(3):
        questions.append(len(seg))
        seg.append((pos, min(N, pos + q_len), 0))
        pos = min(N, pos + q_len)
    tails = [qi for qi, cands in zip(questions, (2, 3, 2)) for _ in range(cands)]
    per = max(1, (N - pos) // len(tails))
    for i, qi in enumerate(tails):
        end = N if i == len(tails) - 1 else min(N, pos + per)
        if pos < end:
            seg.append((pos, end, qi))
        pos = end
    mask = torch.zeros(N, N, dtype=torch.bool, device=dev)
    for si, (s, e, _) in enumerate(seg):
        p = seg[si][2]
        while p != -1:
            cs, ce, parent = seg[p]
            mask[s:e, cs:ce] = True
            p = parent
        mask[s:e, s:e] = torch.ones(e - s, e - s, dtype=torch.bool, device=dev).tril()
    return mask
