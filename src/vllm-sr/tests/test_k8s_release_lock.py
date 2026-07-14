"""Release-scoped Kubernetes Lease tests for CLI mutation serialization."""

import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _k8s_test_helpers import k8s_backend  # noqa: E402
from cli.k8s_release_lock import (  # noqa: E402
    RELEASE_OPERATION_LEASE_SECONDS,
    K8sReleaseLockMixin,
)

LOCK_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def lease_payload(
    backend,
    *,
    holder: str | None,
    renewed_at: datetime = LOCK_NOW,
    resource_version: str = "7",
) -> dict:
    manifest = json.loads(
        backend._release_lease_manifest(
            holder=holder,
            now=renewed_at,
            resource_version=resource_version,
        )
    )
    return manifest


def test_release_lock_create_is_atomic_and_content_free(monkeypatch):
    backend = k8s_backend()
    calls = []
    monkeypatch.setattr(backend, "_release_lock_now", lambda: LOCK_NOW)
    monkeypatch.setattr(backend, "_get_release_operation_lease", lambda: None)
    monkeypatch.setattr(
        backend,
        "_run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or MagicMock(returncode=0),
    )

    backend._acquire_release_operation_lock("holder-a")

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[1:4] == ["create", "-f", "-"]
    manifest = json.loads(kwargs["input_text"])
    assert manifest["kind"] == "Lease"
    assert manifest["spec"]["holderIdentity"] == "holder-a"
    assert manifest["spec"]["leaseDurationSeconds"] == RELEASE_OPERATION_LEASE_SECONDS


def test_release_lock_rejects_an_active_holder(monkeypatch):
    backend = k8s_backend()
    payload = lease_payload(backend, holder="other-holder")
    monkeypatch.setattr(backend, "_release_lock_now", lambda: LOCK_NOW)
    monkeypatch.setattr(backend, "_get_release_operation_lease", lambda: payload)

    with pytest.raises(RuntimeError, match="already in progress"):
        backend._acquire_release_operation_lock("holder-a")


def test_release_lock_create_collision_reads_and_rejects_winner(monkeypatch):
    backend = k8s_backend()
    winner = lease_payload(backend, holder="winner")
    reads = iter([None, winner])
    monkeypatch.setattr(backend, "_release_lock_now", lambda: LOCK_NOW)
    monkeypatch.setattr(backend, "_get_release_operation_lease", lambda: next(reads))
    monkeypatch.setattr(
        backend,
        "_run",
        lambda *args, **kwargs: MagicMock(returncode=1),
    )

    with pytest.raises(RuntimeError, match="already in progress"):
        backend._acquire_release_operation_lock("holder-a")


def test_release_lock_steals_only_an_expired_version_with_cas(monkeypatch):
    backend = k8s_backend()
    expired = LOCK_NOW - timedelta(seconds=RELEASE_OPERATION_LEASE_SECONDS + 1)
    payload = lease_payload(backend, holder="stale-holder", renewed_at=expired)
    calls = []
    monkeypatch.setattr(backend, "_release_lock_now", lambda: LOCK_NOW)
    monkeypatch.setattr(backend, "_get_release_operation_lease", lambda: payload)
    monkeypatch.setattr(
        backend,
        "_run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or MagicMock(returncode=0),
    )

    backend._acquire_release_operation_lock("holder-a")

    manifest = json.loads(calls[0][1]["input_text"])
    assert calls[0][0][1] == "replace"
    assert manifest["metadata"]["resourceVersion"] == "7"
    assert manifest["spec"]["holderIdentity"] == "holder-a"


def test_release_lock_clear_preserves_resource_version(monkeypatch):
    backend = k8s_backend()
    payload = lease_payload(backend, holder="holder-a")
    calls = []
    monkeypatch.setattr(backend, "_release_lock_now", lambda: LOCK_NOW)
    monkeypatch.setattr(backend, "_get_release_operation_lease", lambda: payload)
    monkeypatch.setattr(
        backend,
        "_run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or MagicMock(returncode=0),
    )

    backend._release_release_operation_lock("holder-a")

    manifest = json.loads(calls[0][1]["input_text"])
    assert manifest["metadata"]["resourceVersion"] == "7"
    assert "holderIdentity" not in manifest["spec"]


def test_release_lock_renew_uses_cas_and_asserts_current_holder(monkeypatch):
    backend = k8s_backend()
    payload = lease_payload(backend, holder="holder-a")
    calls = []
    monkeypatch.setattr(backend, "_release_lock_now", lambda: LOCK_NOW)
    monkeypatch.setattr(backend, "_get_release_operation_lease", lambda: payload)
    monkeypatch.setattr(
        backend,
        "_run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or MagicMock(returncode=0),
    )
    backend._active_release_lock_holder = "holder-a"

    backend._renew_release_operation_lock("holder-a")
    backend._assert_release_operation_lock()

    manifest = json.loads(calls[0][1]["input_text"])
    assert calls[0][0][1] == "replace"
    assert manifest["metadata"]["resourceVersion"] == "7"
    assert manifest["spec"]["holderIdentity"] == "holder-a"


def test_release_lock_renew_conflict_and_holder_change_fail_closed(monkeypatch):
    backend = k8s_backend()
    current = [lease_payload(backend, holder="holder-a")]
    monkeypatch.setattr(backend, "_release_lock_now", lambda: LOCK_NOW)
    monkeypatch.setattr(backend, "_get_release_operation_lease", lambda: current[0])
    monkeypatch.setattr(
        backend,
        "_run",
        lambda *args, **kwargs: MagicMock(returncode=1),
    )

    with pytest.raises(RuntimeError, match="renewal failed"):
        backend._renew_release_operation_lock("holder-a")

    backend._active_release_lock_holder = "holder-a"
    backend._active_release_lock_renewal_failed = threading.Event()
    backend._active_release_lock_renewal_failed.set()
    with pytest.raises(RuntimeError, match="renewal failed"):
        K8sReleaseLockMixin._assert_release_operation_lock(backend)

    backend._active_release_lock_renewal_failed = None
    current[0] = lease_payload(backend, holder="winner", resource_version="8")
    with pytest.raises(RuntimeError, match="ownership changed"):
        K8sReleaseLockMixin._assert_release_operation_lock(backend)


def test_release_lock_context_releases_after_failure(monkeypatch):
    backend = k8s_backend()
    events = []
    monkeypatch.setattr(
        backend,
        "_acquire_release_operation_lock",
        lambda holder: events.append(("acquire", holder)),
    )
    monkeypatch.setattr(
        backend,
        "_release_release_operation_lock",
        lambda holder: events.append(("release", holder)),
    )

    with (
        pytest.raises(ValueError, match="boom"),
        K8sReleaseLockMixin._release_operation_lock(backend),
    ):
        raise ValueError("boom")

    assert [event[0] for event in events] == ["acquire", "release"]
    assert events[0][1] == events[1][1]


def test_release_lock_context_joins_renewer_before_release(monkeypatch):
    backend = k8s_backend()
    events = []

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            del target, name, daemon
            self.alive = False

        def start(self):
            self.alive = True
            events.append("start")

        def join(self, *, timeout):
            assert timeout > 10
            events.append("join")
            self.alive = False

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(threading, "Thread", FakeThread)
    monkeypatch.setattr(
        backend,
        "_acquire_release_operation_lock",
        lambda holder: events.append("acquire"),
    )

    def release(holder):
        del holder
        assert events[-1] == "join"
        events.append("release")

    monkeypatch.setattr(backend, "_release_release_operation_lock", release)

    with K8sReleaseLockMixin._release_operation_lock(backend):
        events.append("body")

    assert events == ["acquire", "start", "body", "join", "release"]


def test_release_lock_context_retains_holder_when_renewer_cannot_join(monkeypatch):
    backend = k8s_backend()

    class StuckThread:
        def __init__(self, *, target, name, daemon):
            del target, name, daemon

        def start(self):
            pass

        def join(self, *, timeout):
            assert timeout > 10

        def is_alive(self):
            return True

    monkeypatch.setattr(threading, "Thread", StuckThread)
    monkeypatch.setattr(backend, "_acquire_release_operation_lock", lambda _: None)
    monkeypatch.setattr(
        backend,
        "_release_release_operation_lock",
        lambda _: pytest.fail("a live renewer must not race Lease release"),
    )

    with (
        pytest.raises(RuntimeError, match="renewer did not stop"),
        K8sReleaseLockMixin._release_operation_lock(backend),
    ):
        pass


def test_release_lock_thread_start_failure_releases_holder(monkeypatch):
    backend = k8s_backend()
    released = []

    class FailingThread:
        def __init__(self, *, target, name, daemon):
            del target, name, daemon

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading, "Thread", FailingThread)
    monkeypatch.setattr(backend, "_acquire_release_operation_lock", lambda _: None)
    monkeypatch.setattr(
        backend,
        "_release_release_operation_lock",
        released.append,
    )

    with (
        pytest.raises(RuntimeError, match="thread start failed"),
        K8sReleaseLockMixin._release_operation_lock(backend),
    ):
        pass

    assert len(released) == 1
    assert backend._active_release_lock_holder is None


def test_release_lock_name_is_bounded_and_collision_resistant():
    first = k8s_backend("Release_" + "a" * 80)
    second = k8s_backend("Release_" + "a" * 79 + "b")

    assert len(first._release_lock_name()) <= 63
    assert first._release_lock_name() != second._release_lock_name()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["metadata"].pop("resourceVersion"),
        lambda payload: payload["spec"].update({"leaseDurationSeconds": True}),
        lambda payload: payload["spec"].update({"renewTime": "not-a-time"}),
        lambda payload: payload["spec"].update({"renewTime": "2099-01-01T00:00:00Z"}),
    ],
)
def test_release_lock_malformed_state_fails_closed(mutation):
    backend = k8s_backend()
    payload = lease_payload(backend, holder="holder-a")
    mutation(payload)

    with pytest.raises(RuntimeError, match="Lease"):
        backend._release_lease_state(payload, LOCK_NOW)
