"""Focused checks for the v10 post-key seal verifier."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from jev_arena.authored_v10_postkey import sealed_review


def test_two_artifacts_must_match_the_seal(tmp_path):
    stem = "originals"
    review = tmp_path / f"{stem}-review.jsonl"
    summary = tmp_path / f"{stem}-summary.md"
    review.write_text('{"id":"opaque"}\n')
    summary.write_text("Gold-blind review\n")
    seal = tmp_path / f"{stem}-seal.sha256"
    seal.write_text(
        f"{hashlib.sha256(review.read_bytes()).hexdigest()}  {review.name}\n"
        f"{hashlib.sha256(summary.read_bytes()).hexdigest()}  {summary.name}\n"
    )
    (tmp_path / f"{stem}-seal.utc").write_text(
        datetime.now(timezone.utc).isoformat() + "\n"
    )
    assert (
        sealed_review(tmp_path, stem)[0]
        == hashlib.sha256(review.read_bytes()).hexdigest()
    )

    summary.write_text("Changed after the seal\n")
    with pytest.raises(ValueError, match="seal is invalid"):
        sealed_review(tmp_path, stem)
