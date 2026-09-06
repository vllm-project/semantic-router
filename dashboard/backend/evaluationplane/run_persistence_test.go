package evaluationplane

import (
	"context"
	"errors"
	"testing"
)

type faultingRunNamespacePersistence struct {
	delegate      runNamespacePersistence
	syncFailures  int
	syncCalls     int
	removeAll     func(string) error
	syncDirectory func(string, string) error
}

func (p *faultingRunNamespacePersistence) Rename(source, destination string) error {
	return p.delegate.Rename(source, destination)
}

func (p *faultingRunNamespacePersistence) RemoveAll(path string) error {
	if p.removeAll != nil {
		return p.removeAll(path)
	}
	return p.delegate.RemoveAll(path)
}

func (p *faultingRunNamespacePersistence) SyncDirectory(path, description string) error {
	p.syncCalls++
	if p.syncDirectory != nil {
		return p.syncDirectory(path, description)
	}
	if p.syncFailures > 0 {
		p.syncFailures--
		return errors.New("injected run parent sync failure")
	}
	return p.delegate.SyncDirectory(path, description)
}

func TestPeerOpenCannotCommitAnotherServicesPendingRunPublication(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	stableRequest := validCreateRequest()
	stableRequest.ClientRequestID = newTestClientRequestID()
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), stableRequest); err != nil {
		t.Fatalf("create stable run before publication fault: %v", err)
	}
	request := validCreateRequest()
	request.ClientRequestID = stableClientRequestID
	faults := &faultingRunNamespacePersistence{
		delegate:     atomicRunNamespacePersistence{},
		syncFailures: 1,
	}
	service.store.runPersistence = faults

	if _, err := service.CreateRunAs(context.Background(), SystemActor(), request); err == nil {
		t.Fatal("run publication unexpectedly crossed the injected parent sync failure")
	}
	callsBeforePeer := faults.syncCalls
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); !errors.Is(err, ErrConflict) {
		t.Fatalf("peer open error=%v, want pending publication conflict", err)
	}
	if faults.syncCalls != callsBeforePeer {
		t.Fatalf("peer open consumed owner publication barrier: calls=%d want=%d", faults.syncCalls, callsBeforePeer)
	}
	if _, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10}); !errors.Is(err, ErrConflict) {
		t.Fatalf("ledger exposed pending run publication: %v", err)
	}
	if _, err := service.GetRunAs(SystemActor(), request.ClientRequestID); !errors.Is(err, ErrConflict) {
		t.Fatalf("generic read exposed pending run publication: %v", err)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), request.ClientRequestID); !errors.Is(err, ErrConflict) {
		t.Fatalf("start consumed pending run publication: %v", err)
	}
	hold := true
	if _, err := service.UpdateRunLifecycle(
		SystemActor(), request.ClientRequestID, UpdateLifecycleRequest{EvidenceHold: &hold},
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("lifecycle update consumed pending run publication: %v", err)
	}
	if err := service.DeleteRunAs(SystemActor(), request.ClientRequestID); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete consumed pending run publication: %v", err)
	}
	callsBeforeUnrelatedRetry := faults.syncCalls
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), stableRequest); !errors.Is(err, ErrConflict) {
		t.Fatalf("unrelated idempotent create retry error=%v, want pending publication conflict", err)
	}
	if faults.syncCalls != callsBeforeUnrelatedRetry {
		t.Fatalf("unrelated retry synced another run publication: calls=%d want=%d", faults.syncCalls, callsBeforeUnrelatedRetry)
	}
	administrator := testLifecycleActor(t, "pending-run-publication-admin", true)
	if _, err := service.CreateRunAs(context.Background(), administrator, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("different administrator adopted pending run publication: %v", err)
	}
	if faults.syncCalls != callsBeforeUnrelatedRetry {
		t.Fatalf("different administrator synced pending run publication: calls=%d want=%d", faults.syncCalls, callsBeforeUnrelatedRetry)
	}
	created, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil || created.ID != request.ClientRequestID {
		t.Fatalf("owner retry did not commit run publication: run=%s err=%v", created.ID, err)
	}
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); err != nil {
		t.Fatalf("peer open remained blocked after explicit create retry: %v", err)
	}
}

func TestRunCreationRetryClosesParentSyncUncertainty(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	request.ClientRequestID = stableClientRequestID
	faults := &faultingRunNamespacePersistence{
		delegate:     atomicRunNamespacePersistence{},
		syncFailures: 2,
	}
	service.store.runPersistence = faults

	if _, err := service.CreateRunAs(context.Background(), SystemActor(), request); err == nil {
		t.Fatal("run creation succeeded before its parent namespace was durable")
	}
	if _, err := service.store.GetRun(request.ClientRequestID); !errors.Is(err, ErrConflict) {
		t.Fatalf("generic read exposed visible failed publication: %v", err)
	}
	if run, err := service.store.getRunForCreateRetry(request.ClientRequestID); err != nil || run.ID != request.ClientRequestID {
		t.Fatalf("create-only lookup cannot reconcile visible publication: run=%s err=%v", run.ID, err)
	}
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), request); err == nil {
		t.Fatal("idempotent run retry bypassed persistent parent sync failure")
	}
	created, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil || created.ID != request.ClientRequestID {
		t.Fatalf("idempotent run retry did not close parent durability: run=%s err=%v", created.ID, err)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close service before restart: %v", err)
	}
	restarted := reopenTestService(t, root)
	if run, err := restarted.store.GetRun(request.ClientRequestID); err != nil || run.ID != request.ClientRequestID {
		t.Fatalf("durable run publication disappeared after restart: run=%s err=%v", run.ID, err)
	}
}
