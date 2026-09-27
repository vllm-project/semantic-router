"""0.2.1 row selection on a verified 0.2 suite.

The Space describes the Home duplicate criterion but does not publish which
surface copy of each duplicate pair was retained. ``first`` and ``last`` are
therefore provisional policies. Only an upstream-authorized explicit keep
list can resolve row identity; aggregate-score agreement cannot prove it.
"""

from __future__ import annotations

import collections
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .protocol import spec


def canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class Selection:
    keep_run_ids: frozenset[str]
    dropped: dict[str, int]
    scheduled: int
    scoreable: int
    home_policy: str
    status: str
    keep_ids_sha256: str

    def keep(self, evaluation: dict) -> bool:
        return evaluation["run_id"] in self.keep_run_ids


def verify_kit() -> Path:
    """Refuse a modified 0.2 kit, even if its advertised edition is 0.2."""
    import decision_index

    root = Path(decision_index.__file__).resolve().parent
    for name, expected in spec()["kit_files_sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"public 0.2 kit file differs from pinned revision: {name}"
            )
    return root


def _home_reference() -> tuple[dict[str, dict], set[str]]:
    verify_kit()
    from decision_index.suite.build.home_appliance import make_row

    source = {
        row["id"]: row
        for h in range(8, 24)
        for offset in range(10)
        for row in [make_row(h, offset)]
    }
    dev_states = {
        canonical(make_row(h, offset)["state"])
        for h in range(8)
        for offset in range(10)
    }
    if len(source) != 160 or len(dev_states) != 80:
        raise ValueError("pinned Home generator changed its state multiplicity")
    return source, dev_states


def _answerable(group: list[dict]) -> bool:
    signatures = set()
    for row in group:
        scoring = row["scoring"]
        ids = scoring["retrieved_ids"]
        qrels = scoring["qrels"]
        signatures.add(canonical((ids, scoring["scorable_ids"], qrels)))
    if len(signatures) != 1:
        raise ValueError("retrieval chunks disagree on candidate IDs or qrels")
    first = group[0]["scoring"]
    # The 0.2.1 rule says a relevant item among the 32 *retrieved*
    # candidates. A few overlong retrieved candidates may be absent from
    # scorable_ids; the latter would implement a different filter.
    return any(first["qrels"].get(doc, 0) > 0 for doc in first["retrieved_ids"])


def select(
    rows: Iterable[dict],
    *,
    home_policy: str = "first",
    home_keep_run_ids: set[str] | None = None,
    require_full_counts: bool = True,
    home_reference: tuple[dict[str, dict], set[str]] | None = None,
) -> Selection:
    """Select scoreable requests after the kit's common exclusions.

    ``rows`` must be the verified kit Suite.rows(apply_exclusions=True) stream
    including its added-rows file. ``home_policy='explicit'`` needs the exact
    88 upstream-kept run IDs; other policies cannot be labeled exact 0.2.1.
    """
    if home_policy not in {"first", "last", "explicit"}:
        raise ValueError("home_policy must be first, last, or explicit")
    if home_policy == "explicit" and not home_keep_run_ids:
        raise ValueError("explicit Home policy requires a nonempty keep-run-id set")
    if home_policy != "explicit" and home_keep_run_ids is not None:
        raise ValueError("a Home keep-run-id set requires home_policy='explicit'")
    source, dev_states = home_reference or _home_reference()
    all_rows = list(rows)
    run_ids = [r["_evaluation"]["run_id"] for r in all_rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id in kit suite")
    groups = collections.defaultdict(list)
    home_rows = []
    for row in all_rows:
        e = row["_evaluation"]
        if e["catalog_id"] in (2, 36):
            groups[(e["catalog_id"], e["group_id"])].append(row)
        elif e["catalog_id"] == 9:
            home_rows.append(row)
    if require_full_counts and len(home_rows) != 160:
        raise ValueError(f"0.2 Home rows: expected 160, got {len(home_rows)}")
    dropped = {"toolret": 0, "bright": 0, "home_dev_overlap": 0, "home_duplicate": 0}
    remove = set()
    for (number, _), group in groups.items():
        if not _answerable(group):
            remove.update(r["_evaluation"]["run_id"] for r in group)
            dropped["toolret" if number == 2 else "bright"] += len(group)
    grouped_home = collections.defaultdict(list)
    for row in home_rows:
        rid = row["id"]
        if rid not in source or any(
            row[k] != source[rid][k] for k in ("state", "questions", "expected")
        ):
            raise ValueError(f"Home row does not match the pinned 0.2 generator: {rid}")
        key = canonical(row["state"])
        if key in dev_states:
            remove.add(row["_evaluation"]["run_id"])
            dropped["home_dev_overlap"] += 1
        else:
            grouped_home[key].append(row)
    for group in grouped_home.values():
        if home_policy == "explicit":
            chosen = [
                r for r in group if r["_evaluation"]["run_id"] in home_keep_run_ids
            ]
            if len(chosen) != 1:
                raise ValueError(
                    "explicit Home keep list must choose one row per remaining state"
                )
            keep = chosen[0]
        else:
            keep = group[0 if home_policy == "first" else -1]
        for row in group:
            if row is not keep:
                remove.add(row["_evaluation"]["run_id"])
                dropped["home_duplicate"] += 1
    if home_policy == "explicit" and set(home_keep_run_ids) != {
        r["_evaluation"]["run_id"]
        for r in home_rows
        if r["_evaluation"]["run_id"] not in remove
    }:
        raise ValueError("explicit Home keep list contains an unknown or excluded row")
    expected = spec()["expected_drops"]
    if require_full_counts and (
        dropped["toolret"],
        dropped["bright"],
        dropped["home_dev_overlap"] + dropped["home_duplicate"],
    ) != (expected["toolret"], expected["bright"], expected["home"]):
        raise ValueError(
            f"0.2.1 row-filter counts differ from published protocol: {dropped}"
        )
    kept = frozenset(set(run_ids) - remove)
    n = len(all_rows)
    if require_full_counts and (n, len(kept)) != (151034, 150317):
        raise ValueError(
            f"complete 0.2/0.2.1 scoreable rows: expected 151034/150317, got {n}/{len(kept)}"
        )
    digest = hashlib.sha256("\n".join(sorted(kept)).encode()).hexdigest()
    return Selection(
        keep_run_ids=kept,
        dropped=dropped,
        scheduled=n,
        scoreable=len(kept),
        home_policy=home_policy,
        status=(
            "explicit_row_identity_unverified"
            if home_policy == "explicit"
            else "provisional_home_copy"
        ),
        keep_ids_sha256=digest,
    )
