"""Tests for K8s teardown and fail-closed Secret cleanup."""

import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _k8s_test_helpers import (  # noqa: E402
    k8s_backend,
    managed_secret_payload,
)
from cli.k8s_secret_quarantine import (  # noqa: E402
    MANAGED_SECRET_OPERATION_GRACE,
    ManagedSecretCandidate,
)

TEARDOWN_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
TEARDOWN_STALE = TEARDOWN_NOW - timedelta(minutes=16)


def teardown_candidates(
    *names: str,
    delete_not_before: datetime | None = None,
) -> dict[str, ManagedSecretCandidate]:
    return {
        name: ManagedSecretCandidate(
            created_at=TEARDOWN_STALE,
            uid=f"uid-{name}",
            resource_version=f"rv-{name}",
            delete_not_before=delete_not_before,
        )
        for name in names
    }


def expired_teardown_candidates(*names: str) -> dict[str, ManagedSecretCandidate]:
    return teardown_candidates(
        *names,
        delete_not_before=TEARDOWN_NOW - timedelta(seconds=1),
    )


class TestK8sBackend:
    def test_absent_namespace_returns_without_creating_a_release_lease(
        self, monkeypatch
    ):
        backend = k8s_backend()
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_namespace_exists", lambda: False)
        monkeypatch.setattr(
            backend,
            "_release_operation_lock",
            lambda: pytest.fail("an absent namespace must not be recreated for a lock"),
        )
        monkeypatch.setattr(
            backend,
            "_helm_release_exists",
            lambda: pytest.fail("an absent namespace has no Helm release"),
        )

        backend.teardown()

    def test_teardown_uninstalls_before_release_scoped_cleanup(
        self, monkeypatch, caplog
    ):
        backend = k8s_backend()
        events = []
        release_states = iter([True, False])
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_helm_release_exists", lambda: next(release_states)
        )
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: ["vllm-sr-env-secrets"]
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: teardown_candidates("alpha-vsr-env-old"),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: (
                events.append((cmd[1], cmd)) or MagicMock(returncode=0)
            ),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name, _candidate: events.append(("delete", name)) or True,
        )
        backend.teardown()

        assert [event[0] for event in events] == ["annotate", "uninstall", "delete"]
        assert "alpha-vsr-env-old" in events[0][1]
        assert "--wait" in events[1][1]
        assert "never deletes this unowned compatibility Secret" in caplog.text
        assert "all live workload references" in caplog.text
        assert "retained Helm revisions for every release" in caplog.text

    def test_teardown_waits_through_fresh_secret_grace_before_delete(self, monkeypatch):
        backend = k8s_backend()
        name = "alpha-vsr-env-concurrent-in-flight"
        commands = []
        deleted = []
        release_states = iter([True, False])
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_helm_release_exists", lambda: next(release_states)
        )
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: {
                name: ManagedSecretCandidate(
                    created_at=TEARDOWN_NOW - timedelta(minutes=1),
                    uid=f"uid-{name}",
                    resource_version=f"rv-{name}",
                )
            },
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: commands.append(cmd) or MagicMock(returncode=0),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        backend.teardown()

        assert [command[1] for command in commands] == ["annotate", "uninstall"]
        assert deleted == [name]

    def test_teardown_releases_lease_during_quarantine_and_reacquires_for_delete(
        self, monkeypatch
    ):
        backend = k8s_backend()
        name = "alpha-vsr-env-old"
        lock_state = {"held": False, "phases": 0}
        deleted = []
        release_states = iter([True, False])

        @contextmanager
        def release_lock():
            assert lock_state["held"] is False
            lock_state["held"] = True
            lock_state["phases"] += 1
            try:
                yield
            finally:
                lock_state["held"] = False

        def assert_lock():
            assert lock_state["held"] is True

        def wait_without_lock(deadline):
            assert lock_state["held"] is False
            return deadline

        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_release_operation_lock", release_lock)
        monkeypatch.setattr(backend, "_assert_release_operation_lock", assert_lock)
        monkeypatch.setattr(
            backend, "_helm_release_exists", lambda: next(release_states)
        )
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: expired_teardown_candidates(name),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(backend, "_wait_for_secret_quarantine", wait_without_lock)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: MagicMock(returncode=0),
        )

        def delete(value, _candidate):
            assert lock_state["held"] is True
            deleted.append(value)
            return True

        monkeypatch.setattr(backend, "_delete_secret_if_exists", delete)

        backend.teardown()

        assert lock_state == {"held": False, "phases": 2}
        assert deleted == [name]

    def test_redeploy_during_unlocked_quarantine_adopts_secret_and_defers_cleanup(
        self, monkeypatch, caplog
    ):
        backend = k8s_backend()
        name = "alpha-vsr-env-redeployed"
        lock_state = {"held": False, "phases": 0}
        current_refs: list[str] = []
        deleted = []

        @contextmanager
        def release_lock():
            assert lock_state["held"] is False
            lock_state["held"] = True
            lock_state["phases"] += 1
            try:
                yield
            finally:
                lock_state["held"] = False

        def wait_and_redeploy(deadline):
            # This is the exact interleaving the two-phase design permits:
            # phase 1 has released the Lease, so a deploy may complete before
            # cleanup reacquires it. The new Deployment adopts the unchanged
            # immutable generation.
            assert lock_state == {"held": False, "phases": 1}
            current_refs.append(name)
            return deadline

        def assert_lock():
            assert lock_state["held"] is True

        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_release_operation_lock", release_lock)
        monkeypatch.setattr(backend, "_assert_release_operation_lock", assert_lock)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: list(current_refs)
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: expired_teardown_candidates(name),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(backend, "_wait_for_secret_quarantine", wait_and_redeploy)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: MagicMock(returncode=0),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        backend.teardown()

        assert lock_state == {"held": False, "phases": 2}
        assert current_refs == [name]
        assert deleted == []
        assert "later deployment exists" in caplog.text

    def test_redeploy_new_generation_defers_old_generation_retained_by_history(
        self, monkeypatch, caplog
    ):
        backend = k8s_backend()
        old_name = "alpha-vsr-env-old-history"
        new_name = "alpha-vsr-env-new-live"
        current_refs = [old_name]
        retained_history = {old_name}
        deleted = []
        release_states = iter([True, True])

        def wait_and_redeploy(deadline):
            current_refs[:] = [new_name]
            return deadline

        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_helm_release_exists", lambda: next(release_states)
        )
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: list(current_refs)
        )
        monkeypatch.setattr(
            backend,
            "_helm_retained_secret_refs",
            lambda: set(retained_history),
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: expired_teardown_candidates(old_name),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(backend, "_wait_for_secret_quarantine", wait_and_redeploy)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: MagicMock(returncode=0),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        backend.teardown()

        assert current_refs == [new_name]
        assert retained_history == {old_name}
        assert deleted == []
        assert "later deployment exists" in caplog.text

    def test_teardown_quarantines_its_pre_uninstall_generation(self, monkeypatch):
        backend = k8s_backend()
        name = "alpha-vsr-env-current"
        current_refs = iter([[name], []])
        deleted = []
        release_states = iter([True, False])
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_helm_release_exists", lambda: next(release_states)
        )
        monkeypatch.setattr(
            backend,
            "_current_router_secret_refs",
            lambda: next(current_refs),
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: {
                name: ManagedSecretCandidate(
                    created_at=TEARDOWN_NOW - timedelta(minutes=1),
                    uid=f"uid-{name}",
                    resource_version=f"rv-{name}",
                )
            },
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: MagicMock(returncode=0),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        backend.teardown()

        assert deleted == [name]

    def test_failed_teardown_retains_all_secrets(self, monkeypatch):
        backend = k8s_backend()
        deleted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: ["vllm-sr-env-secrets"]
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: teardown_candidates("alpha-vsr-env-old"),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)

        def run(cmd, **kwargs):
            return MagicMock(returncode=1 if cmd[1] == "uninstall" else 0)

        monkeypatch.setattr(backend, "_run", run)
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name, _candidate: deleted.append(name),
        )

        with pytest.raises(RuntimeError, match="Helm uninstall failed"):
            backend.teardown()

        assert deleted == []

    @pytest.mark.parametrize(
        "candidate",
        [
            ManagedSecretCandidate(
                created_at=TEARDOWN_NOW + timedelta(days=1),
                uid="uid-unsafe-created-at",
                resource_version="rv-unsafe-created-at",
            ),
            ManagedSecretCandidate(
                created_at=TEARDOWN_STALE,
                uid="uid-unsafe-delete-not-before",
                resource_version="rv-unsafe-delete-not-before",
                delete_not_before=TEARDOWN_NOW + timedelta(days=1),
            ),
        ],
    )
    def test_teardown_rejects_far_future_lifecycle_metadata_before_uninstall(
        self, monkeypatch, candidate
    ):
        backend = k8s_backend()
        commands = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: {"alpha-vsr-env-unsafe": candidate},
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: commands.append(cmd) or MagicMock(returncode=0),
        )

        with pytest.raises(RuntimeError, match="unsafe lifecycle metadata"):
            backend.teardown()

        assert commands == []

    def test_teardown_early_wait_result_retains_protected_secret_and_fails(
        self, monkeypatch
    ):
        backend = k8s_backend()
        name = "alpha-vsr-env-protected"
        deadline = TEARDOWN_NOW + timedelta(minutes=10)
        candidates = teardown_candidates(name, delete_not_before=deadline)
        deleted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: False)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: candidates,
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_wait_for_secret_quarantine",
            lambda _: deadline - timedelta(seconds=1),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            backend.teardown()

        assert deleted == []

    def test_teardown_delete_failure_tries_every_secret_then_fails(self, monkeypatch):
        backend = k8s_backend()
        attempted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: False)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: expired_teardown_candidates(
                "alpha-vsr-env-fails",
                "alpha-vsr-env-succeeds",
            ),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend, "_run", lambda *args, **kwargs: MagicMock(returncode=0)
        )

        def delete(name, _candidate):
            attempted.append(name)
            return name.endswith("succeeds")

        monkeypatch.setattr(backend, "_delete_secret_if_exists", delete)

        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            backend.teardown()

        assert attempted == ["alpha-vsr-env-fails", "alpha-vsr-env-succeeds"]

    def test_second_stop_retries_cleanup_after_release_is_proven_absent(
        self, monkeypatch
    ):
        backend = k8s_backend()
        release_states = iter([True, False, False, False])
        delete_results = iter([False, True])
        attempted = []
        uninstall_calls = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_helm_release_exists", lambda: next(release_states)
        )
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: expired_teardown_candidates("alpha-vsr-env-stale"),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: (
                uninstall_calls.append(args[0]) or MagicMock(returncode=0)
            ),
        )

        def delete(name, _candidate):
            attempted.append(name)
            return next(delete_results)

        monkeypatch.setattr(backend, "_delete_secret_if_exists", delete)

        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            backend.teardown()
        backend.teardown()

        assert sum(command[1] == "uninstall" for command in uninstall_calls) == 1
        assert attempted == ["alpha-vsr-env-stale", "alpha-vsr-env-stale"]

    def test_absent_release_does_not_delete_a_still_referenced_secret(
        self, monkeypatch
    ):
        backend = k8s_backend()
        live_name = "alpha-vsr-env-live"
        deleted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: False)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [live_name])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: expired_teardown_candidates(live_name),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name, _candidate: deleted.append(name) or True,
        )

        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            backend.teardown()

        assert deleted == []

    def test_absent_release_first_quarantines_unmigrated_secrets(self, monkeypatch):
        backend = k8s_backend()
        name = "alpha-vsr-env-pre-quarantine"
        commands = []
        deleted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: False)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: teardown_candidates(name),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: commands.append(cmd) or MagicMock(returncode=0),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        backend.teardown()

        assert len(commands) == 1
        assert commands[0][1:4] == ["annotate", "secret", name]
        assert deleted == [name]

    def test_teardown_quarantine_blocks_late_atomic_rollback_adoption(
        self, monkeypatch
    ):
        backend = k8s_backend()
        name = "alpha-vsr-env-old"
        uninstall_started = threading.Event()
        rollback_adopted = threading.Event()
        current_refs: list[str] = []
        deleted = []

        def late_rollback():
            assert uninstall_started.wait(timeout=1)
            current_refs.append(name)
            rollback_adopted.set()

        rollback_thread = threading.Thread(target=late_rollback)
        rollback_thread.start()
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(
            backend, "_current_router_secret_refs", lambda: list(current_refs)
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: expired_teardown_candidates(name),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)

        def run(cmd, **kwargs):
            if cmd[1] == "uninstall":
                uninstall_started.set()
                assert rollback_adopted.wait(timeout=1)
            return MagicMock(returncode=0)

        monkeypatch.setattr(backend, "_run", run)
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        backend.teardown()
        rollback_thread.join(timeout=1)

        assert current_refs == [name]
        assert deleted == []

    def test_expired_quarantine_rechecks_late_rollback_reference(self, monkeypatch):
        backend = k8s_backend()
        name = "alpha-vsr-env-late-rollback"
        deleted = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: False)
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [name])
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: teardown_candidates(
                name,
                delete_not_before=TEARDOWN_NOW - MANAGED_SECRET_OPERATION_GRACE,
            ),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda value, _candidate: deleted.append(value) or True,
        )

        with pytest.raises(RuntimeError, match="cleanup was incomplete"):
            backend.teardown()

        assert deleted == []

    def test_unknown_helm_release_state_never_starts_cleanup(self, monkeypatch):
        backend = k8s_backend()
        events = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend,
            "_helm_release_exists",
            lambda: (_ for _ in ()).throw(RuntimeError("unknown state")),
        )
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: events.append("list") or {},
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name, _candidate: events.append(("delete", name)) or True,
        )

        with pytest.raises(RuntimeError, match="unknown state"):
            backend.teardown()

        assert events == []

    def test_unknown_initial_kubectl_state_never_uninstalls_or_deletes(
        self, monkeypatch
    ):
        backend = k8s_backend()
        events = []
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(backend, "_helm_release_exists", lambda: True)
        monkeypatch.setattr(
            backend,
            "_current_router_secret_refs",
            lambda: (_ for _ in ()).throw(RuntimeError("unknown kubectl state")),
        )
        monkeypatch.setattr(
            backend,
            "_run",
            lambda *args, **kwargs: events.append("uninstall"),
        )
        monkeypatch.setattr(
            backend,
            "_delete_secret_if_exists",
            lambda name, _candidate: events.append(("delete", name)) or True,
        )

        with pytest.raises(RuntimeError, match="unknown kubectl state"):
            backend.teardown()

        assert events == []

    def test_release_secret_listing_filters_other_releases(self, monkeypatch):
        backend = k8s_backend()
        alpha = managed_secret_payload("alpha-vsr-env-one", "alpha", {})
        beta = managed_secret_payload("beta-vsr-env-one", "beta", {})
        monkeypatch.setattr(
            backend,
            "_get_json",
            lambda cmd, **kwargs: {"items": [alpha, beta]},
        )

        assert backend._list_release_managed_secret_names() == ["alpha-vsr-env-one"]

    def test_teardown_never_deletes_namespace_global_legacy_secret(
        self, monkeypatch, caplog
    ):
        backend = k8s_backend()
        commands = []
        release_states = iter([True, False])
        monkeypatch.setattr(backend, "_require_tool", lambda _: None)
        monkeypatch.setattr(
            backend, "_helm_release_exists", lambda: next(release_states)
        )
        monkeypatch.setattr(backend, "_current_router_secret_refs", lambda: [])
        # Even a mistakenly release-labelled legacy Secret remains unowned and safe.
        monkeypatch.setattr(
            backend,
            "_list_release_managed_secret_candidates",
            lambda: teardown_candidates("vllm-sr-env-secrets"),
        )
        monkeypatch.setattr(backend, "_managed_secret_gc_now", lambda: TEARDOWN_NOW)
        monkeypatch.setattr(
            backend,
            "_helm_retained_secret_refs",
            lambda: pytest.fail("teardown must not infer global legacy ownership"),
        )
        monkeypatch.setattr(
            backend,
            "_run",
            lambda cmd, **kwargs: commands.append(cmd) or MagicMock(returncode=0),
        )

        backend.teardown()

        assert len(commands) == 1
        assert commands[0][1] == "uninstall"
        assert "namespace-global legacy runtime Secret" in caplog.text
        assert "vllm-sr-env-secrets" in caplog.text

    def test_kubectl_inspection_failure_is_fail_closed_without_output_leak(
        self, monkeypatch, caplog
    ):
        backend = k8s_backend()
        sensitive_output = "server-returned-sensitive-output"
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: MagicMock(
                returncode=1,
                stdout="",
                stderr=sensitive_output,
            ),
        )

        with pytest.raises(RuntimeError, match="kubectl query failed"):
            backend._current_router_secret_refs()

        assert sensitive_output not in caplog.text
