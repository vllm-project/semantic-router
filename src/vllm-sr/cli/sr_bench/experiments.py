"""Durable experiment organization, separate from immutable run evidence."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone

EXPERIMENT_ID = re.compile(r"exp-[0-9a-f]{32}\Z")

MAX_HYPOTHESIS_CHARS = 2000
MAX_NAME_CHARS = 160
MAX_KEY_CHARS = 128
MAX_EXPERIMENT_PAGE = 50

ROLES = {
    "smoke",
    "baseline",
    "initial",
    "preview",
    "candidate",
    "validation",
    "estimate",
    "recovery",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_membership(value):
    if not isinstance(value, dict) or set(value) - {"id", "role", "hypothesis"}:
        raise ValueError("experiment requires id, role and an optional hypothesis")
    identifier = value.get("id")
    if not isinstance(identifier, str) or not EXPERIMENT_ID.fullmatch(identifier):
        raise ValueError("Invalid experiment identity")
    if value.get("role") not in ROLES:
        raise ValueError("Unknown experiment run role")
    hypothesis = value.get("hypothesis", "")
    if not isinstance(hypothesis, str) or len(hypothesis) > MAX_HYPOTHESIS_CHARS:
        raise ValueError("Experiment hypothesis must be at most 2000 characters")
    return {"id": identifier, "role": value["role"], "hypothesis": hypothesis.strip()}


def run_roles(manifest):
    if manifest.get("recovery"):
        return ["recovery"]
    mode, profile = manifest["mode"], manifest["profile"]
    kinds = {target["kind"] for target in manifest["targets"]}
    if mode == "preview":
        return ["preview"]
    if mode == "replay":
        return ["estimate"]
    roles = []
    if mode == "live":
        if "single" in kinds:
            roles.append("baseline")
        if profile == "standard":
            roles.append("validation")
        else:
            if "mom" in kinds:
                roles.extend(["initial", "candidate"])
            if profile == "smoke":
                roles.append("smoke")
    return roles


def validate_role(manifest, role):
    if role not in run_roles(manifest):
        raise ValueError(
            "Run mode, profile or targets do not match its experiment role"
        )


def bind_created_run(store, run_id, manifest, owner, actor_role):
    """Called inside Store.create's transaction, before dispatch can start."""
    if "experiment" not in manifest:
        return
    link = validate_membership(manifest["experiment"])
    experiment = store.db.execute(
        "SELECT owner FROM experiments WHERE id=?", (link["id"],)
    ).fetchone()
    if experiment is None or (actor_role != "admin" and experiment[0] != owner):
        raise PermissionError("The submitting actor cannot manage this experiment")
    validate_role(manifest, link["role"])
    _insert_link(store, link["id"], run_id, link["role"], link["hypothesis"])


def _insert_link(store, experiment_id, run_id, role, hypothesis):
    existing = store.db.execute(
        "SELECT role,hypothesis FROM experiment_runs WHERE experiment_id=? AND run_id=?",
        (experiment_id, run_id),
    ).fetchone()
    if existing:
        if existing != (role, hypothesis):
            raise ValueError(
                "This run is already linked with a different role or hypothesis"
            )
        return
    at = now()
    store.db.execute(
        "INSERT INTO experiment_runs(experiment_id,run_id,role,hypothesis,linked_at) VALUES(?,?,?,?,?)",
        (experiment_id, run_id, role, hypothesis, at),
    )
    store.db.execute(
        "UPDATE experiments SET updated_at=? WHERE id=?", (at, experiment_id)
    )


class Experiments:
    def __init__(self, store):
        self.store = store

    def create(self, name, owner, request_key=None):
        if not isinstance(name, str) or not name.strip() or len(name) > MAX_NAME_CHARS:
            raise ValueError("Experiment name must be between 1 and 160 characters")
        if request_key is not None and (
            not isinstance(request_key, str)
            or not request_key
            or len(request_key) > MAX_KEY_CHARS
        ):
            raise ValueError("Invalid experiment idempotency key")
        name = name.strip()
        with self.store.lock, self.store.db:
            if request_key:
                row = self.store.db.execute(
                    "SELECT id,name FROM experiments WHERE owner=? AND request_key=?",
                    (owner, request_key),
                ).fetchone()
                if row:
                    if row[1] != name:
                        raise ValueError(
                            "Experiment idempotency key is already bound to another name"
                        )
                    return self.get(row[0], owner)
            identifier = "exp-" + uuid.uuid4().hex
            at = now()
            self.store.db.execute(
                "INSERT INTO experiments VALUES(?,?,?,?,?,?)",
                (identifier, owner, name, at, at, request_key),
            )
            return self.get(identifier, owner)

    def list(self, owner=None, after=0, limit=20):
        after, limit = int(after), int(limit)
        if after < 0 or not 1 <= limit <= MAX_EXPERIMENT_PAGE:
            raise ValueError("Invalid experiment page")
        where, args = ("WHERE owner=?", [owner]) if owner is not None else ("", [])
        with self.store.lock:
            rows = self.store.db.execute(
                f"SELECT rowid,id,name,created_at,updated_at FROM experiments {where} {'AND' if where else 'WHERE'} rowid>? ORDER BY rowid LIMIT ?",
                [*args, after, limit + 1],
            ).fetchall()
        page = [
            {"id": row[1], "name": row[2], "created_at": row[3], "updated_at": row[4]}
            for row in rows[:limit]
        ]
        return {
            "experiments": page,
            "next_cursor": rows[limit - 1][0] if len(rows) > limit else None,
            "has_more": len(rows) > limit,
        }

    def get(self, identifier, owner=None):
        with self.store.lock:
            row = self.store.db.execute(
                "SELECT owner,name,created_at,updated_at FROM experiments WHERE id=?",
                (identifier,),
            ).fetchone()
            if row is None or (owner is not None and row[0] != owner):
                raise KeyError(identifier)
            count = self.store.db.execute(
                "SELECT COUNT(*) FROM experiment_runs WHERE experiment_id=?",
                (identifier,),
            ).fetchone()[0]
        return {
            "id": identifier,
            "name": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "run_count": count,
        }

    def runs(self, identifier, owner=None, after=0, limit=20):
        experiment = self.get(identifier, owner)
        after, limit = int(after), int(limit)
        if after < 0 or not 1 <= limit <= MAX_EXPERIMENT_PAGE:
            raise ValueError("Invalid experiment run page")
        with self.store.lock:
            rows = self.store.db.execute(
                "SELECT seq,run_id,role,hypothesis,linked_at FROM experiment_runs WHERE experiment_id=? AND seq>? ORDER BY seq LIMIT ?",
                (identifier, after, limit + 1),
            ).fetchall()
            members = [
                {
                    "run_id": run_id,
                    "role": role,
                    "hypothesis": hypothesis,
                    "linked_at": linked_at,
                }
                for _, run_id, role, hypothesis, linked_at in rows[:limit]
            ]
        return {
            "experiment": experiment,
            "members": members,
            "next_cursor": rows[limit - 1][0] if len(rows) > limit else None,
            "has_more": len(rows) > limit,
        }

    def attach(self, identifier, run_id, role, hypothesis="", owner=None):
        link = validate_membership(
            {"id": identifier, "role": role, "hypothesis": hypothesis}
        )
        with self.store.lock, self.store.db:
            self.get(identifier, owner)
            run = self.store.get(run_id, owner)
            validate_role(run["manifest"], role)
            _insert_link(self.store, identifier, run_id, role, link["hypothesis"])
        return self.get(identifier, owner)
