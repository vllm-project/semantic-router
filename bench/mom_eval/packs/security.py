"""Security-first MoM evaluation pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bench.mom_eval.packs.base import BasePack


class SecurityPack(BasePack):
    pack_id = "security/v1"

    def plan(self, manifest: dict[str, Any], entrypoint: str, run_mode: str) -> list:
        return self._plan_from_manifest(run_mode)

    def collect(self, raw_dir: Path, steps: list) -> Any:
        return self.collect_placeholder(raw_dir, steps)


def create_pack() -> SecurityPack:
    return SecurityPack()
