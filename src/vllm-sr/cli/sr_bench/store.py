"""Crash durable run/case/call journal; a dispatched call is never retried."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .contracts import canonical, planned_cells
from .experiments import bind_created_run, run_roles

MAX_PAGE_SIZE = 500
MAX_REPLAY_OPTIONS = 25
MAX_REPLAY_CURSOR_DIGITS = 19

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class RecoveryClaimError(ValueError):
    """The transaction rolled back because another child already owns a cell."""


def now():
    return datetime.now(timezone.utc).isoformat()


def run_summary(run):
    """Project collection metadata without copying frozen question/source payloads."""
    source = run["manifest"]
    manifest = {
        key: source[key]
        for key in (
            "version",
            "name",
            "mode",
            "profile",
            "seed",
            "cost_policy",
            "targets",
            "limits",
            "sampling",
            "benchmark_weights",
            "adapter_versions",
            "experiment",
            "plan_sha256",
            "case_sha256",
        )
        if key in source
    }
    if dataset := source.get("dataset"):
        manifest["dataset"] = {
            key: dataset[key]
            for key in (
                "id",
                "name",
                "path",
                "sha256",
                "case_count",
                "profile",
                "split",
                "seed",
                "custom_subset",
                "benchmarks",
            )
            if key in dataset
        }
    if recovery := source.get("recovery"):
        manifest["recovery"] = {
            key: recovery[key]
            for key in (
                "parent_run_id",
                "mode",
                "selected_cells_sha256",
                "parent_snapshot",
                "recovery_subset",
                "new_attempt_acknowledged",
            )
            if key in recovery
        }
        manifest["recovery"]["selected_cell_count"] = len(
            recovery.get("selected_cells", [])
        )
    return {
        **run,
        "manifest": manifest,
        "manifest_summary": True,
        "experiment_roles": run_roles(source),
    }


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "journal.sqlite3"
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        os.chmod(self.path, 0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_key TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, manifest TEXT NOT NULL, error TEXT, UNIQUE(owner,request_key));
        CREATE TABLE IF NOT EXISTS results(run_id TEXT, case_id TEXT, target_id TEXT, status TEXT, data TEXT, PRIMARY KEY(run_id,case_id,target_id));
        CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,run_id TEXT,case_id TEXT,target_id TEXT,role TEXT,status TEXT,data TEXT);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,at TEXT,kind TEXT,data TEXT);
        CREATE TABLE IF NOT EXISTS run_provenance(run_id TEXT PRIMARY KEY,data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS accounting_corrections(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,created_at TEXT NOT NULL,evidence_sha256 TEXT NOT NULL,data TEXT NOT NULL,UNIQUE(run_id,evidence_sha256));
        CREATE TABLE IF NOT EXISTS recovery_claims(parent_run_id TEXT,case_id TEXT,target_id TEXT,child_run_id TEXT,PRIMARY KEY(parent_run_id,case_id,target_id));
        CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY,owner TEXT NOT NULL,name TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,request_key TEXT,UNIQUE(owner,request_key));
        CREATE TABLE IF NOT EXISTS experiment_runs(seq INTEGER PRIMARY KEY AUTOINCREMENT,experiment_id TEXT NOT NULL,run_id TEXT NOT NULL,role TEXT NOT NULL,hypothesis TEXT NOT NULL,linked_at TEXT NOT NULL,UNIQUE(experiment_id,run_id));
        """
        )
        self.db.commit()

    def event(self, run_id, kind, data):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO events(run_id,at,kind,data) VALUES(?,?,?,?)",
                (run_id, now(), kind, canonical(data)),
            )

    def create(
        self,
        manifest,
        owner="local",
        request_key=None,
        provenance=None,
        *,
        actor_role="local",
    ):
        with self.lock, self.db:
            if request_key:
                row = self.db.execute(
                    "SELECT id,manifest FROM runs WHERE owner=? AND request_key=?",
                    (owner, request_key),
                ).fetchone()
                if row:
                    if json.loads(row[1])["plan_sha256"] != manifest["plan_sha256"]:
                        raise ValueError(
                            "idempotency key is already bound to a different plan"
                        )
                    return self.get(row[0]), False
            run_id = "run-" + uuid.uuid4().hex[:20]
            at = now()
            self.db.execute(
                "INSERT INTO runs VALUES(?,?,?,?,?,?,?,NULL)",
                (run_id, owner, request_key, "queued", at, at, canonical(manifest)),
            )
            if provenance is not None:
                self.db.execute(
                    "INSERT INTO run_provenance VALUES(?,?)",
                    (run_id, canonical(provenance)),
                )
            if recovery := manifest.get("recovery"):
                try:
                    self.db.executemany(
                        "INSERT INTO recovery_claims VALUES(?,?,?,?)",
                        [
                            (
                                recovery["parent_run_id"],
                                cell["case_id"],
                                cell["target_id"],
                                run_id,
                            )
                            for cell in planned_cells(manifest)
                        ],
                    )
                except sqlite3.IntegrityError as exc:
                    raise RecoveryClaimError(
                        "Recovery cells were already claimed by another child"
                    ) from exc
            bind_created_run(self, run_id, manifest, owner, actor_role)
            self.event(run_id, "created", {"plan_sha256": manifest["plan_sha256"]})
            return self.get(run_id), True

    def provenance(self, run_id):
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM run_provenance WHERE run_id=?", (run_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def accounting_correction(self, run_id):
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM accounting_corrections WHERE run_id=? ORDER BY rowid DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def append_accounting_correction(self, run_id, artifact):
        """Preserve original call rows; identical evidence has one durable receipt."""
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT data FROM accounting_corrections WHERE run_id=? AND evidence_sha256=?",
                (run_id, artifact["evidence_sha256"]),
            ).fetchone()
            if row:
                return json.loads(row[0])
            artifact = {**artifact, "created_at": now()}
            self.db.execute(
                "INSERT INTO accounting_corrections VALUES(?,?,?,?,?)",
                (
                    artifact["id"],
                    run_id,
                    artifact["created_at"],
                    artifact["evidence_sha256"],
                    canonical(artifact),
                ),
            )
            self.event(
                run_id,
                "accounting_reconciled",
                {
                    "id": artifact["id"],
                    "evidence_sha256": artifact["evidence_sha256"],
                    "model_requests": 0,
                },
            )
            return artifact

    def first_failure(self, run_id):
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM events WHERE run_id=? AND kind='failure_observed' ORDER BY seq LIMIT 1",
                (run_id,),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def get(self, run_id, owner=None):
        with self.lock:
            row = self.db.execute(
                "SELECT id,owner,status,created_at,updated_at,manifest,error FROM runs WHERE id=?",
                (run_id,),
            ).fetchone()
            if row is None or (owner is not None and row[1] != owner):
                raise KeyError("run not found")
            m = json.loads(row[5])
            counts = dict(
                self.db.execute(
                    "SELECT status,COUNT(*) FROM results WHERE run_id=? GROUP BY status",
                    (run_id,),
                ).fetchall()
            )
            completed = counts.get("completed", 0)
            failed = sum(
                v for k, v in counts.items() if k not in {"completed", "running"}
            )
            return {
                "id": row[0],
                "owner": row[1],
                "status": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "manifest": m,
                "error": row[6],
                "progress": {
                    "total": len(planned_cells(m)),
                    "completed": completed,
                    "failed": failed,
                    "running": counts.get("running", 0),
                },
            }

    def list(self, owner=None, summary=False):
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM runs"
                + (" WHERE owner=?" if owner is not None else "")
                + " ORDER BY created_at DESC",
                (() if owner is None else (owner,)),
            ).fetchall()
            return [
                run_summary(self.get(row[0])) if summary else self.get(row[0])
                for row in rows
            ]

    def request(self, owner, request_key):
        with self.lock:
            row = self.db.execute(
                "SELECT id FROM runs WHERE owner=? AND request_key=?",
                (owner, request_key),
            ).fetchone()
            return self.get(row[0]) if row else None

    def preview_candidates(self, owner=None, after=None, limit=10):
        """Bounded, owner-filtered completed previews in stable insertion order."""
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_REPLAY_OPTIONS
        ):
            raise ValueError("Replay options limit must be between 1 and 25")
        if after is not None and (
            not isinstance(after, str)
            or len(after) > MAX_REPLAY_CURSOR_DIGITS
            or not after.isascii()
            or not after.isdecimal()
            or not 0 < int(after) <= 2**63 - 1
        ):
            raise ValueError("Invalid replay options cursor")
        conditions = ["status='completed'", "json_extract(manifest,'$.mode')='preview'"]
        values = []
        if owner is not None:
            conditions.append("owner=?")
            values.append(owner)
        if after is not None:
            conditions.append("rowid<?")
            values.append(int(after))
        with self.lock:
            rows = self.db.execute(
                "SELECT rowid,id FROM runs WHERE "
                + " AND ".join(conditions)
                + " ORDER BY rowid DESC LIMIT ?",
                (*values, limit + 1),
            ).fetchall()
            page = rows[:limit]
            return (
                [self.get(row[1]) for row in page],
                str(page[-1][0]) if len(rows) > limit else None,
            )

    def children(self, run_id):
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM runs WHERE json_extract(manifest,'$.recovery.parent_run_id')=? ORDER BY created_at,id",
                (run_id,),
            ).fetchall()
            return [self.get(row[0]) for row in rows]

    def recovery_claims(self, parent_id):
        with self.lock:
            return {
                (row[0], row[1]): row[2]
                for row in self.db.execute(
                    "SELECT case_id,target_id,child_run_id FROM recovery_claims WHERE parent_run_id=?",
                    (parent_id,),
                )
            }

    def status(self, run_id, status, error=None):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE runs SET status=?,updated_at=?,error=? WHERE id=?",
                (status, now(), error, run_id),
            )
            self.event(run_id, status, {"error": error} if error else {})

    def result(self, run_id, case_id, target_id, status, data):
        with self.lock, self.db:
            payload = {
                **data,
                "case_id": case_id,
                "target_id": target_id,
                "status": status,
            }
            self.db.execute(
                "INSERT INTO results VALUES(?,?,?,?,?) ON CONFLICT(run_id,case_id,target_id) DO UPDATE SET status=excluded.status,data=excluded.data",
                (run_id, case_id, target_id, status, canonical(payload)),
            )
            self.db.execute("UPDATE runs SET updated_at=? WHERE id=?", (now(), run_id))
            self.event(
                run_id, "case_" + status, {"case_id": case_id, "target_id": target_id}
            )

    def results(self, run_id):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    "SELECT data FROM results WHERE run_id=? ORDER BY case_id,target_id",
                    (run_id,),
                )
            ]

    def start_call(self, run_id, case_id, target_id, role, data):
        call_id = "call-" + uuid.uuid4().hex
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO calls VALUES(?,?,?,?,?,?,?)",
                (
                    call_id,
                    run_id,
                    case_id,
                    target_id,
                    role,
                    "sent",
                    canonical({**data, "started_at": now()}),
                ),
            )
            self.event(
                run_id,
                "call_sent",
                {
                    "call_id": call_id,
                    "case_id": case_id,
                    "target_id": target_id,
                    "role": role,
                },
            )
        return call_id

    def finish_call(self, call_id, status, data):
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT data FROM calls WHERE id=?", (call_id,)
            ).fetchone()
            old = json.loads(row[0])
            self.db.execute(
                "UPDATE calls SET status=?,data=? WHERE id=?",
                (status, canonical({**old, **data, "finished_at": now()}), call_id),
            )

    def cached_call(self, run_id, case_id, target_id, data):
        call_id = "cached-" + uuid.uuid4().hex
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO calls VALUES(?,?,?,?,?,?,?)",
                (
                    call_id,
                    run_id,
                    case_id,
                    target_id,
                    "subject",
                    "replayed",
                    canonical(data),
                ),
            )
            self.event(
                run_id,
                "call_replayed",
                {"call_id": call_id, "source_call_id": data["source_call_id"]},
            )

    @staticmethod
    def _call_record(row):
        return {
            "id": row[0],
            "case_id": row[1],
            "target_id": row[2],
            "role": row[3],
            "status": row[4],
            **json.loads(row[5]),
        }

    def calls(self, run_id, summary=False):
        data = (
            "json_remove(data,'$.request','$.final','$.reasoning','$.raw_usage','$.tool_calls')"
            if summary
            else "data"
        )
        with self.lock:
            return [
                self._call_record(row)
                for row in self.db.execute(
                    f"SELECT id,case_id,target_id,role,status,{data} FROM calls WHERE run_id=? ORDER BY rowid",
                    (run_id,),
                )
            ]

    def call(self, run_id, call_id):
        with self.lock:
            row = self.db.execute(
                "SELECT id,case_id,target_id,role,status,data FROM calls WHERE run_id=? AND id=?",
                (run_id, call_id),
            ).fetchone()
            if row is None:
                raise KeyError("call not found")
            return self._call_record(row)

    def page(self, run_id, kind, after=0, limit=100):
        if (
            kind not in {"calls", "results"}
            or not after >= 0
            or not 1 <= limit <= MAX_PAGE_SIZE
        ):
            raise ValueError(
                "Evidence pages require after>=0 and limit between 1 and 500"
            )
        with self.lock:
            total = self.db.execute(
                f"SELECT count(*) FROM {kind} WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            if kind == "calls":
                selection = "id,case_id,target_id,role,status,json_remove(data,'$.request','$.final','$.reasoning','$.raw_usage','$.tool_calls')"
            else:
                selection = "json_remove(data,'$.partial')"
            rows = self.db.execute(
                f"SELECT rowid,{selection} FROM {kind} WHERE run_id=? AND rowid>? ORDER BY rowid LIMIT ?",
                (run_id, after, limit + 1),
            ).fetchall()
            values = [
                self._call_record(row[1:]) if kind == "calls" else json.loads(row[1])
                for row in rows[:limit]
            ]
            return {
                kind: values,
                "total": total,
                "limit": limit,
                "next_cursor": rows[limit - 1][0] if len(rows) > limit else None,
            }

    def events(self, run_id, after=0):
        with self.lock:
            return [
                {"seq": r[0], "at": r[1], "kind": r[2], "data": json.loads(r[3])}
                for r in self.db.execute(
                    "SELECT seq,at,kind,data FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 1000",
                    (run_id, after),
                )
            ]

    def recover(self):
        with self.lock, self.db:
            active = self.db.execute(
                "SELECT id FROM runs WHERE status NOT IN ('completed','failed','cancelled','interrupted')"
            ).fetchall()
            for (run_id,) in active:
                self.db.execute(
                    "UPDATE calls SET status='sent_unknown' WHERE run_id=? AND status='sent'",
                    (run_id,),
                )
                for r in self.results(run_id):
                    if r["status"] == "running":
                        self.result(
                            run_id,
                            r["case_id"],
                            r["target_id"],
                            "sent_unknown",
                            {
                                **r,
                                "error": "Worker stopped; dispatched requests are not retried",
                            },
                        )
                self.status(
                    run_id,
                    "interrupted",
                    "Service restarted; inspect saved calls before a new run",
                )
