"""Bounded, owner-scoped discovery of runnable saved-evidence combinations."""

import base64
import binascii
import json
import time

from .contracts import digest
from .replay_validation import ReplayValidator, replay_summary
from .report import ComparisonValidator

MAX_LIMIT = 25
MAX_SCAN_STEPS = 64
MAX_SCAN_SECONDS = 5
MAX_EVIDENCE_BYTES = 128 * 1024 * 1024
MAX_CURSOR_CHARS = 2048
MAX_RUN_ID_CHARS = 128
MAX_ROWID = 2**63 - 1


def _choice(run):
    summary = replay_summary(run)
    return {key: summary[key] for key in ("run_id", "name", "profile", "case_count")}


def _cursor(value):
    return base64.urlsafe_b64encode(json.dumps(value, sort_keys=True).encode()).decode()


def _revision(store, owner):
    with store.lock:
        row = store.db.execute(
            "SELECT COUNT(*),COALESCE(MAX(r.rowid),0),MAX(r.updated_at),"
            "COALESCE((SELECT MAX(e.seq) FROM events e JOIN runs x ON x.id=e.run_id"
            " WHERE x.status='completed' AND json_extract(x.manifest,'$.mode') IN ('live','preview')"
            + (" AND x.owner=?" if owner is not None else "")
            + "),0) FROM runs r WHERE r.status='completed'"
            " AND json_extract(r.manifest,'$.mode') IN ('live','preview')"
            + (" AND r.owner=?" if owner is not None else ""),
            () if owner is None else (owner, owner),
        ).fetchone()
    return digest(row), row[1]


def _state(store, kind, baseline_id, owner, after):
    scope = digest([kind, baseline_id, owner])
    revision, ceiling = _revision(store, owner)
    if after is None:
        return {
            "scope": scope,
            "revision": revision,
            "ceiling": ceiling,
            "baseline_after": ceiling + 1,
            "active": None,
            "child_after": ceiling + 1,
            "unverified_pairs": 0,
            "unverified_baselines": 0,
        }
    try:
        if not isinstance(after, str) or len(after) > MAX_CURSOR_CHARS:
            raise ValueError
        state = json.loads(base64.b64decode(after, altchars=b"-_", validate=True))
        if not isinstance(state, dict) or set(state) != {
            "scope",
            "revision",
            "ceiling",
            "baseline_after",
            "active",
            "child_after",
            "unverified_pairs",
            "unverified_baselines",
        }:
            raise ValueError
        if state["scope"] != scope:
            raise ValueError
        if any(
            type(state[k]) is not int or not 0 <= state[k] <= MAX_ROWID
            for k in (
                "ceiling",
                "baseline_after",
                "child_after",
                "unverified_pairs",
                "unverified_baselines",
            )
        ):
            raise ValueError
        if state["active"] is not None and (
            not isinstance(state["active"], str)
            or len(state["active"]) > MAX_RUN_ID_CHARS
        ):
            raise ValueError
        if any(
            state[k] > state["ceiling"] + 1 for k in ("baseline_after", "child_after")
        ):
            raise ValueError
    except (ValueError, TypeError, KeyError, binascii.Error) as exc:
        raise ValueError("Invalid run options cursor") from exc
    if state["revision"] != revision:
        raise ValueError("Saved runs changed; refresh options from the first page")
    return state


def _next(store, owner, state, *, baseline=False, kind=None, source=None):
    conditions = ["status='completed'", "rowid<=?", "rowid<?"]
    values = [state["ceiling"], state["baseline_after" if baseline else "child_after"]]
    if owner is not None:
        conditions.append("owner=?")
        values.append(owner)
    mode = "live" if baseline or kind == "comparison" else "preview"
    conditions.append("json_extract(manifest,'$.mode')=?")
    values.append(mode)
    if baseline:
        conditions.append(
            "EXISTS(SELECT 1 FROM json_each(manifest,'$.targets') "
            "WHERE json_extract(value,'$.kind')='single')"
        )
    else:
        conditions.extend(["id<>?", "json_array_length(manifest,'$.cases')=?"])
        values.extend([source["id"], len(source["manifest"]["cases"])])
        if kind == "comparison":
            conditions.append("json_extract(manifest,'$.case_sha256')=?")
            values.append(source["manifest"]["case_sha256"])
    with store.lock:
        return store.db.execute(
            "SELECT rowid,id FROM runs WHERE "
            + " AND ".join(conditions)
            + " ORDER BY rowid DESC LIMIT 1",
            values,
        ).fetchone()


def _evidence_size(store, run_id, include_calls):
    with store.lock:
        size = store.db.execute(
            "SELECT length(CAST(manifest AS BLOB)) FROM runs WHERE id=?", (run_id,)
        ).fetchone()[0]
        tables = ("results", "calls") if include_calls else ("results",)
        for table in tables:
            size += store.db.execute(
                f"SELECT COALESCE(SUM(length(CAST(data AS BLOB))),0) FROM {table} WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
        if include_calls:
            correction = store.db.execute(
                "SELECT length(CAST(data AS BLOB)) FROM accounting_corrections "
                "WHERE run_id=? ORDER BY rowid DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            size += correction[0] if correction else 0
        return size


def _load_baseline(store, run_id, owner, kind):
    # Check visibility, role and size before decoding a potentially large manifest.
    with store.lock:
        row = store.db.execute(
            "SELECT status,json_extract(manifest,'$.mode'),"
            "EXISTS(SELECT 1 FROM json_each(manifest,'$.targets') "
            "WHERE json_extract(value,'$.kind')='single') FROM runs WHERE id=?"
            + (" AND owner=?" if owner is not None else ""),
            (run_id,) if owner is None else (run_id, owner),
        ).fetchone()
    if row is None:
        raise KeyError("run not found")
    if row != ("completed", "live", 1):
        return None, False
    if _evidence_size(store, run_id, kind == "replay") > MAX_EVIDENCE_BYTES:
        return None, True
    return store.get(run_id, owner), False


def _limits(page, state):
    return {
        **page,
        "scan_limited": bool(
            state["unverified_pairs"] or state["unverified_baselines"]
        ),
        "unverified_pairs": state["unverified_pairs"],
        "unverified_baselines": state["unverified_baselines"],
    }


def run_options(store, kind, baseline_id=None, owner=None, after=None, limit=10):
    """Only proven compatible options; empty resumable pages never mean no matches."""
    if kind not in {"replay", "comparison"}:
        raise ValueError("Unknown run options kind")
    if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
        raise ValueError("Run options limit must be between 1 and 25")
    state = _state(store, kind, baseline_id, owner, after)
    page = {
        "baseline": None,
        "baselines": [],
        "options": [],
        "next_cursor": None,
        "has_more": False,
        "scanned_pairs": 0,
        "model_requests": 0,
    }
    source = None
    if baseline_id:
        source, limited = _load_baseline(store, baseline_id, owner, kind)
        if source is None:
            state["unverified_baselines"] += int(limited)
            page["empty_reason"] = (
                "evidence_size_limit"
                if limited
                else "baseline_not_completed_live_single"
            )
            return _limits(page, state)
        page["baseline"] = _choice(source)
    validator = None
    evidence_bytes = 0
    steps = 0
    deadline = time.monotonic() + MAX_SCAN_SECONDS
    complete = False
    while len(page["options"] if baseline_id else page["baselines"]) < limit:
        if steps >= MAX_SCAN_STEPS or time.monotonic() >= deadline:
            break
        if source is None:
            if state["active"] is not None:
                source, limited = _load_baseline(store, state["active"], owner, kind)
                if source is None and not limited:
                    raise ValueError(
                        "Cursor baseline no longer qualifies; refresh options"
                    )
            else:
                row = _next(store, owner, state, baseline=True)
                if row is None:
                    complete = True
                    break
                state["baseline_after"], state["active"] = row
                state["child_after"] = state["ceiling"] + 1
                source, limited = _load_baseline(store, row[1], owner, kind)
            steps += 1
            if limited:
                state["unverified_baselines"] += 1
                state["active"] = None
                continue
            if source is None:
                raise ValueError(
                    "Saved runs changed; refresh options from the first page"
                )
            if steps >= MAX_SCAN_STEPS or time.monotonic() >= deadline:
                break
        row = _next(store, owner, state, kind=kind, source=source)
        if row is None:
            if baseline_id:
                complete = True
                break
            source, validator, state["active"] = None, None, None
            continue
        size = _evidence_size(store, row[1], False)
        if validator is None:
            size += _evidence_size(store, source["id"], kind == "replay")
        if size > MAX_EVIDENCE_BYTES:
            steps += 1
            state["unverified_pairs"] += 1
            state["child_after"] = row[0]
            continue
        if evidence_bytes + size > MAX_EVIDENCE_BYTES:
            break
        evidence_bytes += size
        steps += 1
        page["scanned_pairs"] += 1
        state["child_after"] = row[0]
        candidate = store.get(row[1], owner)
        try:
            if validator is None:
                validator = (
                    ReplayValidator(store, source)
                    if kind == "replay"
                    else ComparisonValidator(store, source)
                )
            result = validator.validate(candidate)
            eligible = result["eligible"] if kind == "replay" else True
        except ValueError:
            eligible = False
        if eligible:
            if baseline_id:
                page["options"].append(_choice(candidate))
            else:
                page["baselines"].append(_choice(source))
                source, validator, state["active"] = None, None, None
    if not complete:
        # Check existence only, never perform an extra compatibility evaluation.
        if source is not None:
            more = _next(store, owner, state, kind=kind, source=source) is not None
            if not more and not baseline_id:
                state["active"] = None
                more = _next(store, owner, state, baseline=True) is not None
        else:
            more = _next(store, owner, state, baseline=True) is not None
        if more:
            page.update(next_cursor=_cursor(state), has_more=True)
    if baseline_id and after is None and not page["options"] and not page["has_more"]:
        page.update(
            baseline=None,
            empty_reason=(
                "evidence_size_limit"
                if state["unverified_pairs"] or state["unverified_baselines"]
                else "no_compatible_saved_runs"
            ),
        )
    return _limits(page, state)
