"""Crash durable run/case/call journal; a dispatched call is never retried."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .contracts import canonical

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def now():
    return datetime.now(timezone.utc).isoformat()


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
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_key TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, manifest TEXT NOT NULL, error TEXT, UNIQUE(owner,request_key));
        CREATE TABLE IF NOT EXISTS results(run_id TEXT, case_id TEXT, target_id TEXT, status TEXT, data TEXT, PRIMARY KEY(run_id,case_id,target_id));
        CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,run_id TEXT,case_id TEXT,target_id TEXT,role TEXT,status TEXT,data TEXT);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,at TEXT,kind TEXT,data TEXT);
        """)
        self.db.commit()

    def event(self, run_id, kind, data):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO events(run_id,at,kind,data) VALUES(?,?,?,?)",
                (run_id, now(), kind, canonical(data)),
            )

    def create(self, manifest, owner="local", request_key=None):
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
            self.event(run_id, "created", {"plan_sha256": manifest["plan_sha256"]})
            return self.get(run_id), True

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
            return dict(
                id=row[0],
                owner=row[1],
                status=row[2],
                created_at=row[3],
                updated_at=row[4],
                manifest=m,
                error=row[6],
                progress=dict(
                    total=len(m["cases"]) * len(m["targets"]),
                    completed=completed,
                    failed=failed,
                    running=counts.get("running", 0),
                ),
            )

    def list(self, owner=None):
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM runs"
                + (" WHERE owner=?" if owner is not None else "")
                + " ORDER BY created_at DESC",
                (() if owner is None else (owner,)),
            ).fetchall()
            return [self.get(row[0]) for row in rows]

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

    def calls(self, run_id):
        with self.lock:
            return [
                {
                    "id": r[0],
                    "case_id": r[1],
                    "target_id": r[2],
                    "role": r[3],
                    "status": r[4],
                    **json.loads(r[5]),
                }
                for r in self.db.execute(
                    "SELECT id,case_id,target_id,role,status,data FROM calls WHERE run_id=? ORDER BY rowid",
                    (run_id,),
                )
            ]

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
