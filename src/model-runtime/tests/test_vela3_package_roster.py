"""The Vela 3 package roster: router question IDs must resolve to exactly one task."""

from __future__ import annotations

from pathlib import Path

import pytest
from vllm_srun.errors import PackageError
from vllm_srun.families.vela3.package import roster


def config(domain_ids: list[str], factcheck_ids: list[str]) -> dict:
    return {
        "adapters": {"span": {"file": "span.st"}, "router": {"file": "router.st"}},
        "router": {
            "task_adapters": {
                "domain": {"ids": domain_ids, "file": "domain.st"},
                "factcheck": {"ids": factcheck_ids, "file": "factcheck.st"},
            }
        },
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for name in ("span.st", "router.st", "domain.st", "factcheck.st"):
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


def test_distinct_question_ids_map_to_their_own_task(root) -> None:
    built, _ = roster(root, config(["domain"], ["fact"]))
    assert built.router_tasks == {
        "domain": ("domain", ("router:domain",)),
        "fact": ("factcheck", ("router:factcheck",)),
    }


def test_a_question_id_shared_by_two_tasks_is_rejected(root) -> None:
    with pytest.raises(PackageError, match=r"'domain'.*'domain' and 'factcheck'"):
        roster(root, config(["domain"], ["domain"]))
