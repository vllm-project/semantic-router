"""Append-only evidence ledger (JSONL).

One record per evaluated candidate. Two tiers with isolated verdict types
(recipe-lab design §5.3):

- tier "replay": verdict must be RANKED (sort-only; an unverified number must
  never look like a conclusion);
- tier "live":   verdict must be PASS / FAIL / INCONCLUSIVE, and live records
  must additionally carry `gates` (guard-floor results) and `bench_refs`
  (sr-bench run/compare IDs) so that reports and provenance manifests can be
  replayed from the ledger alone.

The ledger is append-only: there is no update or delete API by design.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import LIVE_VERDICTS, REPLAY_VERDICTS, SPLIT_NAMES, TIERS, ContractError

COMMON_REQUIRED = (
    "round",
    "ts",
    "candidate",
    "tier",
    "baseline_ref",
    "metrics",
    "verdict",
    "dataset_hash",
)


@dataclass(frozen=True)
class LedgerRecord:
    round: int
    ts: str
    candidate: dict
    tier: str
    baseline_ref: str
    metrics: dict
    verdict: str
    dataset_hash: str
    reasons: list[str] = field(default_factory=list)
    gates: dict | None = None
    bench_refs: list[str] = field(default_factory=list)
    attribution_delta: dict | None = None
    attribution_ref: str | None = None
    coverage_warning: bool = False
    notes: str = ""
    # B: which frozen split the record's data came from (calib/dev/frozen_test).
    # Legacy single-dataset records omit the field (equivalent to "dev").
    split: str | None = None
    legacy_workload: bool = (
        False  # M1: legacy single-dataset workloads record split: null
    )
    # M5: which declared workload (spec.workloads[].name) the record's data
    # came from; single anonymous legacy workloads omit the field.
    workload: str | None = None
    # A5: present only when the trial produced no usable metric (e.g.
    # "timeout" when the stack exceeded budget.max_minutes_per_trial).
    # best_replay / top_replay exclude records carrying a status.
    status: str | None = None
    # E: economic constraint evidence, name -> {metric, op, bound, baseline,
    # candidate, bound_value, status}. Constraints are distinct from gates
    # (safety vetoes). Live records carry the measured sr-bench evidence;
    # E2 replay records carry the cost-proxy evidence (proxy:
    # routed_cost_index) whenever a cost_model is declared.
    constraints: dict | None = None
    # E2: replay-tier cost-proxy warning - a candidate whose routed_cost_index
    # violates a declared constraint. best_replay / top_replay exclude warned
    # candidates from the ranking hand-off.
    constraint_warning: bool = False
    # M2: paired comparison statistics in the sr-bench compare
    # `comparisons[]` shape (paired_cases / quality_delta / quality_delta_ci95
    # / quality_delta_bootstrap_ci95 / wins / losses / ties), recorded
    # verbatim for report replay; None when the payload carried none.
    paired: dict | None = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "round": self.round,
            "ts": self.ts,
            "candidate": self.candidate,
            "tier": self.tier,
            "baseline_ref": self.baseline_ref,
            "metrics": self.metrics,
            "verdict": self.verdict,
            "dataset_hash": self.dataset_hash,
            "reasons": list(self.reasons),
        }
        if self.gates is not None:
            out["gates"] = dict(self.gates)
        if self.bench_refs:
            out["bench_refs"] = list(self.bench_refs)
        if self.attribution_delta is not None:
            out["attribution_delta"] = dict(self.attribution_delta)
        if self.attribution_ref:
            out["attribution_ref"] = self.attribution_ref
        if self.coverage_warning:
            out["coverage_warning"] = True
        if self.notes:
            out["notes"] = self.notes
        if self.split:
            out["split"] = self.split
        if self.legacy_workload:
            out["legacy_workload"] = True
        if self.workload:
            out["workload"] = self.workload
        if self.status:
            out["status"] = self.status
        if self.constraints is not None:
            out["constraints"] = dict(self.constraints)
        if self.constraint_warning:
            out["constraint_warning"] = True
        if self.paired is not None:
            out["paired"] = dict(self.paired)
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> LedgerRecord:
        validate_record(raw)
        return cls(
            round=int(raw["round"]),
            ts=str(raw["ts"]),
            candidate=dict(raw["candidate"]),
            tier=str(raw["tier"]),
            baseline_ref=str(raw["baseline_ref"]),
            metrics=dict(raw["metrics"]),
            verdict=str(raw["verdict"]),
            dataset_hash=str(raw["dataset_hash"]),
            reasons=[str(r) for r in raw.get("reasons", [])],
            gates=dict(raw["gates"]) if raw.get("gates") is not None else None,
            bench_refs=[str(r) for r in raw.get("bench_refs", [])],
            attribution_delta=(
                dict(raw["attribution_delta"])
                if raw.get("attribution_delta") is not None
                else None
            ),
            attribution_ref=raw.get("attribution_ref"),
            coverage_warning=bool(raw.get("coverage_warning", False)),
            notes=str(raw.get("notes", "")),
            split=raw.get("split"),
            status=raw.get("status"),
            constraints=(
                dict(raw["constraints"]) if raw.get("constraints") is not None else None
            ),
            constraint_warning=bool(raw.get("constraint_warning", False)),
            legacy_workload=bool(raw.get("legacy_workload", False)),
            workload=raw.get("workload"),
            paired=dict(raw["paired"]) if raw.get("paired") is not None else None,
        )


def validate_record(raw: dict) -> None:
    """Validate one ledger record against the v1alpha1 ledger contract."""
    for name in COMMON_REQUIRED:
        _require_present(raw, name)
    if isinstance(raw["round"], bool) or not isinstance(raw["round"], int):
        raise ContractError("round must be an integer")
    tier = raw["tier"]
    if tier not in TIERS:
        raise ContractError(f"tier must be one of {sorted(TIERS)}, got {tier!r}")
    verdict = raw["verdict"]
    if tier == "replay" and verdict not in REPLAY_VERDICTS:
        raise ContractError(
            f"replay-tier records must use RANKED verdict (got {verdict!r}); "
            "unverified numbers must not look like conclusions"
        )
    if tier == "live" and verdict not in LIVE_VERDICTS:
        raise ContractError(
            f"live-tier records must use one of {sorted(LIVE_VERDICTS)} (got {verdict!r}); "
            "use RANKED only in the replay tier"
        )
    if not isinstance(raw["metrics"], dict):
        raise ContractError("metrics must be a mapping")
    if not isinstance(raw["candidate"], dict):
        raise ContractError("candidate must be a mapping")
    _require_nonempty_str(raw["dataset_hash"], "dataset_hash")
    if tier == "live":
        gates = raw.get("gates")
        if not isinstance(gates, dict):
            raise ContractError(
                "live-tier records must carry a 'gates' mapping (guard-floor results)"
            )
        bench_refs = raw.get("bench_refs")
        if not isinstance(bench_refs, list):
            raise ContractError(
                "live-tier records must carry a 'bench_refs' list (sr-bench run/compare IDs)"
            )
        if verdict != "INCONCLUSIVE" and not bench_refs:
            raise ContractError(
                "live records with a concluded verdict must reference at least one sr-bench run "
                "(bench_refs); INCONCLUSIVE records may have an empty list"
            )
    for optional_list in ("reasons", "bench_refs"):
        if optional_list in raw and not isinstance(raw[optional_list], list):
            raise ContractError(f"{optional_list} must be a list")
    if "attribution_ref" in raw and raw["attribution_ref"] is not None:
        _require_nonempty_str(raw["attribution_ref"], "attribution_ref")
    if "split" in raw and raw["split"] is not None and raw["split"] not in SPLIT_NAMES:
        raise ContractError(
            f"split must be one of {sorted(SPLIT_NAMES)}, got {raw['split']!r}"
        )
    if "workload" in raw and raw["workload"] is not None:
        _require_nonempty_str(raw["workload"], "workload")
    if "status" in raw and raw["status"] is not None:
        _require_nonempty_str(raw["status"], "status")
        # A5: "timeout" - the trial exceeded its budget. H: "incomplete" - the
        # hot-reload replay pull lost or duplicated records for the candidate's
        # request ids. Neither carries a metric and both are excluded from
        # top_replay, so a broken capture can never rank as a worse recipe.
        if raw["status"] not in ("timeout", "incomplete"):
            raise ContractError(
                f"status must be 'timeout' or 'incomplete', got {raw['status']!r}"
            )
    if "constraint_warning" in raw and not isinstance(raw["constraint_warning"], bool):
        raise ContractError("constraint_warning must be a boolean")


def _require_present(raw: dict, name: str) -> None:
    if name not in raw:
        raise ContractError(f"ledger record is missing required field: {name}")


def _require_nonempty_str(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")


def _best_replay(records: list[dict], metric_key: str) -> dict | None:
    """A2: pure best-replay selection over raw ledger records.

    Highest ``metrics[metric_key]`` among tier "replay" / verdict "RANKED"
    records; coverage_warning records, A5 status records (a trial that
    produced no usable metric) and E2 constraint_warning records (a
    candidate that violates a declared constraint under the replay cost
    proxy) are excluded. Ties keep the earliest record.
    Shared by ``Ledger.top_replay`` and the ``next`` context export so the
    hand-off to the live tier and the agent context always agree.
    """
    best: dict | None = None
    best_value: float | None = None
    for raw in records:
        if raw.get("tier") != "replay" or raw.get("verdict") != "RANKED":
            continue
        if raw.get("coverage_warning"):
            continue
        if raw.get("status"):  # A5: a trial that produced no metric never ranks
            continue
        if raw.get(
            "constraint_warning"
        ):  # E2: a constraint-violating candidate never ranks
            continue
        value = raw.get("metrics", {}).get(metric_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if best_value is None or value > best_value:
            best, best_value = raw, float(value)
    return best


def compute_record_id(raw: dict) -> str:
    """G1: stable content identity of a ledger record, 'rec-<12 hex>' over the
    canonical JSON of the record without its own record_id. Same content =>
    same id, so provenance can cite records across rounds."""
    body = {k: v for k, v in raw.items() if k != "record_id"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return "rec-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class Ledger:
    """Append-only JSONL ledger at a fixed path.

    Appends are serialized by a lock file next to the ledger, idempotent (the
    same content is stored once) and published atomically: a writer killed
    mid-append leaves the previous file intact, including its last line.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def _locked(self, *, create: bool):
        """Hold the writer lock. Readers lock an existing lock file only: a
        ledger without one has never been appended to, and a lock that
        cannot be taken does not block a read. An append without its lock
        would race a concurrent writer into a lost record, so a lock
        failure fails the append instead of proceeding unlocked."""
        from .loop import writer_lock  # noqa: PLC0415

        lock_path = self.path.with_suffix(".lock")
        with ExitStack() as stack:
            if create:
                stack.enter_context(writer_lock(lock_path))
            elif lock_path.exists():
                with suppress(OSError):
                    stack.enter_context(writer_lock(lock_path))
            yield

    def append(self, record: LedgerRecord | dict) -> dict:
        raw = record.to_dict() if isinstance(record, LedgerRecord) else dict(record)
        validate_record(raw)
        raw["record_id"] = compute_record_id(raw)
        line = json.dumps(raw, ensure_ascii=False, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(create=True):
            prefix = self.path.read_bytes() if self.path.is_file() else b""
            for existing in self._read_rows(prefix):
                if existing["record_id"] != raw["record_id"]:
                    continue
                if existing != raw:
                    raise ContractError(
                        f"ledger record ID collision: {raw['record_id']} already "
                        "names a different record"
                    )
                return existing  # a duplicate append stores nothing new
            if prefix and not prefix.endswith(b"\n"):
                prefix += b"\n"
            staged: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.path.parent, delete=False
                ) as tmp:
                    staged = Path(tmp.name)
                with staged.open("wb") as output:
                    output.write(prefix)
                    output.write(line.encode("utf-8") + b"\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(staged, self.path)
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
        return raw

    def read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        with self._locked(create=False):
            return self._read_rows(self.path.read_bytes())

    def _read_rows(self, data: bytes) -> list[dict]:
        records: list[dict] = []
        for lineno, raw_line in enumerate(data.split(b"\n"), 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"ledger {self.path} line {lineno} is not valid JSON: {exc}"
                ) from exc
            # G1: pre-record_id ledgers get the same id rule applied on read
            if "record_id" not in raw:
                raw["record_id"] = compute_record_id(raw)
            records.append(raw)
        return records

    def iter_records(self) -> Iterator[dict]:
        yield from self.read()

    def filter(
        self,
        *,
        tier: str | None = None,
        verdict: str | None = None,
        round_no: int | None = None,
        proposer: str | None = None,
    ) -> list[dict]:
        out = []
        for raw in self.read():
            if tier is not None and raw.get("tier") != tier:
                continue
            if verdict is not None and raw.get("verdict") != verdict:
                continue
            if round_no is not None and raw.get("round") != round_no:
                continue
            if (
                proposer is not None
                and raw.get("candidate", {}).get("provenance", {}).get("proposer")
                != proposer
            ):
                continue
            out.append(raw)
        return out

    def top_replay(self, metric_key: str) -> dict | None:
        """Best replay-tier candidate: highest metric, coverage_warning excluded.

        This is the hand-off contract to the live tier (Sub-issue 3); A2 routes
        it through the shared pure `best_replay` selection."""
        return _best_replay(self.read(), metric_key)
