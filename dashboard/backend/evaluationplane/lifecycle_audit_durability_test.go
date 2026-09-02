package evaluationplane

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestLifecycleCreationBindingRetryClosesVisibleLinkSyncFailure(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest()); err != nil {
		t.Fatalf("create run for lifecycle binding checkpoint: %v", err)
	}
	faults := &faultingLifecycleAuditCommitWriter{
		delegate:        atomicLifecycleAuditWriter{},
		publishFailures: 1,
		syncFailures:    1,
	}
	service.store.lifecycleAuditWriter = faults
	checkpoint := func() error {
		service.store.lifecycle.mu.Lock()
		defer service.store.lifecycle.mu.Unlock()
		return service.store.checkpointLifecycleAuditUnlocked(time.Now().UTC())
	}
	if err := checkpoint(); err == nil {
		t.Fatal("visible lifecycle binding publication unexpectedly reported durable success")
	}
	if got := len(service.store.lifecycle.pendingLifecycleBindings); got != 1 {
		t.Fatalf("pending lifecycle binding count=%d, want 1", got)
	}
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); !errors.Is(err, ErrConflict) {
		t.Fatalf("peer open error=%v, want pending binding conflict", err)
	}
	if err := checkpoint(); err == nil {
		t.Fatal("checkpoint retry bypassed persistent binding directory sync failure")
	}
	if err := checkpoint(); err != nil {
		t.Fatalf("checkpoint retry did not close binding durability: %v", err)
	}
	if got := len(service.store.lifecycle.pendingLifecycleBindings); got != 0 {
		t.Fatalf("pending lifecycle binding count=%d after commit, want 0", got)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close lifecycle binding durability service: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("reopen lifecycle binding checkpoint: %v", err)
	}
}

type faultingLifecycleAuditCommitWriter struct {
	delegate        lifecycleAuditWriter
	publishFailures int
	syncFailures    int
}

func (writer *faultingLifecycleAuditCommitWriter) WriteExclusive(path string, value any) error {
	if writer.publishFailures == 0 {
		return writer.delegate.WriteExclusive(path, value)
	}
	writer.publishFailures--
	if err := publishJSONWithoutParentSync(path, value); err != nil {
		return err
	}
	return errors.New("injected lifecycle audit sync failure after visible publication")
}

func (writer *faultingLifecycleAuditCommitWriter) SyncDirectory(path, description string) error {
	if writer.syncFailures > 0 {
		writer.syncFailures--
		return errors.New("injected lifecycle audit retry sync failure")
	}
	return writer.delegate.SyncDirectory(path, description)
}

func TestLifecycleNotFoundDedupeRetryClosesVisibleAuditSyncFailure(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	actor := testLifecycleActor(t, "audit-dedupe-durability", false)
	service.store.lifecycleNow = func() time.Time {
		return time.Date(2026, 9, 2, 1, 2, 3, 0, time.UTC)
	}
	faults := &faultingLifecycleAuditCommitWriter{
		delegate:        atomicLifecycleAuditWriter{},
		publishFailures: 1,
		syncFailures:    1,
	}
	service.store.lifecycleAuditWriter = faults
	runID := newTestClientRequestID()

	appendMissing := func() error {
		service.store.lifecycle.mu.Lock()
		defer service.store.lifecycle.mu.Unlock()
		_, err := service.store.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "start", "denied", "not_found", runID, "",
		)
		return err
	}
	if err := appendMissing(); err == nil {
		t.Fatal("visible audit publication unexpectedly reported durable success")
	}
	if err := appendMissing(); err == nil {
		t.Fatal("deduplicated retry bypassed persistent audit directory sync failure")
	}
	if err := appendMissing(); err != nil {
		t.Fatalf("deduplicated retry did not close audit durability: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	sequence, active := service.store.lifecycle.sequence, service.store.lifecycle.activeCount
	service.store.lifecycle.mu.Unlock()
	if sequence != 1 || active != 1 {
		t.Fatalf("deduplicated retry changed audit chain: sequence=%d active=%d", sequence, active)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close audit durability service: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("reopen durable lifecycle audit: %v", err)
	}
}
