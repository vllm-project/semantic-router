package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type failingRunDeletionGCAuditWriter struct {
	delegate lifecycleAuditWriter
}

func (writer failingRunDeletionGCAuditWriter) WriteExclusive(path string, value any) error {
	record, isRecord := value.(lifecycleAuditRecord)
	if isRecord && record.ResourceKind == lifecycleResourceStore &&
		record.Action == "gc" && record.ReasonCode == "delete_cascade" {
		return errors.New("injected run deletion GC audit failure")
	}
	return writer.delegate.WriteExclusive(path, value)
}

func (writer failingRunDeletionGCAuditWriter) SyncDirectory(path, description string) error {
	return writer.delegate.SyncDirectory(path, description)
}

func terminalRunDeletionFixture(t *testing.T, service *Service, owner Actor) Run {
	t.Helper()
	request := validCreateRequest()
	request.ClientRequestID = newTestClientRequestID()
	run, err := service.CreateRunAs(context.Background(), owner, request)
	if err != nil {
		t.Fatalf("create run deletion fixture: %v", err)
	}
	startedAt := run.CreatedAt.Add(time.Microsecond)
	completedAt := startedAt.Add(time.Microsecond)
	run.Status, run.StartedAt, run.CompletedAt = StatusCancelled, &startedAt, &completedAt
	run.Progress.Message = "Run cancelled"
	if err := service.store.updateRunFixture(run); err != nil {
		t.Fatalf("make run deletion fixture terminal: %v", err)
	}
	return run
}

func assertRunDeletionPersistsAfterRestart(t *testing.T, service *Service, root, runID string) {
	t.Helper()
	if err := service.Close(); err != nil {
		t.Fatalf("close run deletion durability service: %v", err)
	}
	reopened, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("reopen committed run deletion: %v", err)
	}
	if _, err := reopened.GetRun(runID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted run reappeared after restart: %v", err)
	}
}

func TestRunDeletionParentSyncFailureKeepsLedgerFailClosed(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-delete-durability-owner", false)
	other := testLifecycleActor(t, "run-delete-durability-other", false)
	run := terminalRunDeletionFixture(t, service, owner)
	casDigest := writeGCCASArtifact(
		t, service.store, run.ID, "metrics.json", []byte("run-delete-durability\n"),
	)
	subscriber, unsubscribe, subscribeErr := service.SubscribeAs(owner, run.ID)
	if subscribeErr != nil {
		t.Fatalf("subscribe before run deletion: %v", subscribeErr)
	}
	defer unsubscribe()
	faults := &faultingRunNamespacePersistence{
		delegate:     atomicRunNamespacePersistence{},
		syncFailures: 1,
	}
	service.store.runPersistence = faults

	if err := service.DeleteRunAs(owner, run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("run deletion sync error=%v, want ErrConflict", err)
	}
	intentPath := runDeletionPath(service.store.runsRoot, run.ID)
	if _, err := os.Lstat(filepath.Join(service.store.runsRoot, run.ID)); !os.IsNotExist(err) {
		t.Fatalf("live run remained after atomic hide: %v", err)
	}
	if err := requirePrivateDirectory(intentPath); err != nil {
		t.Fatalf("run deletion intent is unavailable after uncertain sync: %v", err)
	}
	assertCASObjectExists(t, service.store, casDigest, true)
	replacement := validCreateRequest()
	replacement.ClientRequestID = run.ID
	if _, err := service.CreateRunAs(context.Background(), other, replacement); !errors.Is(err, ErrConflict) {
		t.Fatalf("hidden run identity rebind error=%v, want ErrConflict", err)
	}
	if _, err := os.Lstat(filepath.Join(service.store.runsRoot, run.ID)); !os.IsNotExist(err) {
		t.Fatalf("hidden run identity acquired a new live bundle: %v", err)
	}

	// Generic reads and CAS scans must fail closed without consuming the sync
	// fault: only DeleteRun can finish the disk transaction and close this
	// Service's subscriber channel.
	faults.syncFailures = 1
	if _, err := service.store.unreferencedCASCandidatesUnlocked(); !errors.Is(err, ErrConflict) {
		t.Fatalf("uncertain run CAS scan error=%v, want ErrConflict", err)
	}
	assertCASObjectExists(t, service.store, casDigest, true)
	if _, err := service.store.ListRuns(); !errors.Is(err, ErrConflict) {
		t.Fatalf("uncertain run ledger scan error=%v, want ErrConflict", err)
	}
	if _, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10}); !errors.Is(err, ErrConflict) {
		t.Fatalf("uncertain paged run ledger error=%v, want ErrConflict", err)
	}
	if count, _, _ := subscriberRegistryCounts(service); count != 1 {
		t.Fatalf("generic read changed subscriber count=%d, want 1", count)
	}
	if err := service.DeleteRunAs(other, run.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner run deletion retry error=%v, want ErrForbidden", err)
	}
	if err := requirePrivateDirectory(intentPath); err != nil {
		t.Fatalf("cross-owner retry consumed run deletion intent: %v", err)
	}
	if err := service.DeleteRunAs(owner, run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("owner retry sync error=%v, want ErrConflict", err)
	}
	if _, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10}); !errors.Is(err, ErrConflict) {
		t.Fatalf("uncertain retry paged ledger error=%v, want ErrConflict", err)
	}
	if count, _, _ := subscriberRegistryCounts(service); count != 1 {
		t.Fatalf("uncertain owner retry changed subscriber count=%d, want 1", count)
	}
	if err := service.DeleteRunAs(owner, run.ID); err != nil {
		t.Fatalf("owner retry did not commit run deletion: %v", err)
	}
	assertCASObjectExists(t, service.store, casDigest, false)
	select {
	case _, open := <-subscriber:
		if open {
			t.Fatal("run deletion subscriber remained open")
		}
	default:
		t.Fatal("run deletion subscriber was not closed")
	}
	if count, _, _ := subscriberRegistryCounts(service); count != 0 {
		t.Fatalf("run deletion subscriber count=%d, want 0", count)
	}
	page, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10})
	if err != nil || !page.LedgerComplete || len(page.Runs) != 0 {
		t.Fatalf("committed deletion page=%+v err=%v", page, err)
	}

	assertRunDeletionPersistsAfterRestart(t, service, root, run.ID)
}

func TestRunDeletionGCAuditFailureDoesNotStrandCommittedIntent(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-delete-gc-audit-owner", false)
	run := terminalRunDeletionFixture(t, service, owner)
	casDigest := writeGCCASArtifact(
		t, service.store, run.ID, "metrics.json", []byte("run-delete-gc-audit\n"),
	)
	service.store.lifecycleAuditWriter = failingRunDeletionGCAuditWriter{
		delegate: atomicLifecycleAuditWriter{},
	}

	if err := service.DeleteRunAs(owner, run.ID); err != nil {
		t.Fatalf("committed deletion was blocked by deferred GC audit: %v", err)
	}
	if _, err := os.Lstat(runDeletionPath(service.store.runsRoot, run.ID)); !os.IsNotExist(err) {
		t.Fatalf("committed deletion retained a global-blocking intent: %v", err)
	}
	assertCASObjectExists(t, service.store, casDigest, true)
	page, pageErr := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10})
	if pageErr != nil || !page.LedgerComplete || len(page.Runs) != 0 {
		t.Fatalf("ledger remained blocked after deferred GC: page=%+v err=%v", page, pageErr)
	}
	if err := service.DeleteRunAs(owner, run.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("retry changed the committed deletion result: %v", err)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close run deletion GC audit service: %v", err)
	}
	reopened, reopenErr := newStandaloneStore(root)
	if reopenErr != nil {
		t.Fatalf("reopen after deferred run deletion GC: %v", reopenErr)
	}
	if _, err := reopened.GetRun(run.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted run reappeared after restart: %v", err)
	}
}

func TestPeerServiceCannotRecoverAnotherServiceRunDeletionIntent(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	t.Cleanup(func() { _ = service.Close() })
	owner := testLifecycleActor(t, "run-delete-peer-recovery-owner", false)
	run := terminalRunDeletionFixture(t, service, owner)
	subscriber, unsubscribe, err := service.SubscribeAs(owner, run.ID)
	if err != nil {
		t.Fatalf("subscribe before uncertain deletion: %v", err)
	}
	defer unsubscribe()
	service.store.runPersistence = &faultingRunNamespacePersistence{
		delegate: atomicRunNamespacePersistence{}, syncFailures: 1,
	}
	if err := service.DeleteRunAs(owner, run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("stage uncertain run deletion: %v", err)
	}

	peer, peerErr := NewService(Options{
		DataDir: service.store.Root(), PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		DeploymentsDir: service.registrySource.deploymentsDir, RouterAPIURL: service.registrySource.routerAPIURL, EnvoyURL: service.registrySource.envoyURL,
		CodeRevision: service.codeRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if peer != nil {
		_ = peer.Close()
	}
	if !errors.Is(peerErr, ErrConflict) {
		t.Fatalf("peer Service recovered an owned deletion intent: %v", peerErr)
	}
	select {
	case _, open := <-subscriber:
		if !open {
			t.Fatal("failed peer open closed the owner Service subscriber")
		}
	default:
	}
	if count, _, _ := subscriberRegistryCounts(service); count != 1 {
		t.Fatalf("failed peer open changed subscriber count=%d, want 1", count)
	}
	evaluationRootCoordinators.Lock()
	refs := evaluationRootCoordinators.byRoot[service.store.root].serviceRefs
	evaluationRootCoordinators.Unlock()
	if refs != 1 {
		t.Fatalf("failed peer open leaked a root coordinator reference: %d", refs)
	}
	if err := service.DeleteRunAs(owner, run.ID); err != nil {
		t.Fatalf("owner retry did not commit deletion: %v", err)
	}
	requireSubscriptionClosed(t, subscriber, "owner deletion retry subscriber")
}

func TestRunDeletionStartupRecoversVisibleIntent(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-delete-startup-owner", false)
	run := terminalRunDeletionFixture(t, service, owner)
	service.store.runPersistence = &faultingRunNamespacePersistence{
		delegate:     atomicRunNamespacePersistence{},
		syncFailures: 1,
	}
	if err := service.DeleteRunAs(owner, run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("run deletion sync error=%v, want ErrConflict", err)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close before run deletion startup recovery: %v", err)
	}
	reopened, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("startup did not recover visible run deletion intent: %v", err)
	}
	if _, err := reopened.GetRun(run.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("startup-recovered run lookup error=%v, want ErrNotFound", err)
	}
	if _, err := os.Lstat(runDeletionPath(reopened.runsRoot, run.ID)); !os.IsNotExist(err) {
		t.Fatalf("startup retained run deletion intent: %v", err)
	}
}

func TestRunDeletionRecoversPartialIntentCleanup(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-partial-delete-owner", false)
	run := terminalRunDeletionFixture(t, service, owner)
	intentPath := runDeletionPath(service.store.runsRoot, run.ID)
	faults := &faultingRunNamespacePersistence{
		delegate: atomicRunNamespacePersistence{},
		removeAll: func(path string) error {
			if path != intentPath {
				t.Fatalf("run cleanup path=%q, want %q", path, intentPath)
			}
			if err := os.Remove(filepath.Join(path, runFileName)); err != nil {
				return err
			}
			return errors.New("injected partial run deletion cleanup")
		},
	}
	service.store.runPersistence = faults
	if err := service.DeleteRunAs(owner, run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("partial run cleanup error=%v, want ErrConflict", err)
	}
	if err := requirePrivateDirectory(intentPath); err != nil {
		t.Fatalf("partial run deletion intent is unavailable: %v", err)
	}
	service.store.runPersistence = atomicRunNamespacePersistence{}
	if err := service.DeleteRunAs(owner, run.ID); err != nil {
		t.Fatalf("run deletion retry did not recover partial intent: %v", err)
	}
	if _, err := os.Lstat(intentPath); !os.IsNotExist(err) {
		t.Fatalf("partial run deletion intent survived recovery: %v", err)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close partial run deletion service: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("reopen after partial run deletion recovery: %v", err)
	}
}

func TestRunDeletionCorruptCompleteIntentCannotAdvanceOtherOwners(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	firstOwner := testLifecycleActor(t, "run-corrupt-delete-owner", false)
	secondOwner := testLifecycleActor(t, "run-other-delete-owner", false)
	first := terminalRunDeletionFixture(t, service, firstOwner)
	second := terminalRunDeletionFixture(t, service, secondOwner)
	for _, run := range []Run{first, second} {
		if err := os.Rename(
			filepath.Join(service.store.runsRoot, run.ID),
			runDeletionPath(service.store.runsRoot, run.ID),
		); err != nil {
			t.Fatalf("stage run deletion intent: %v", err)
		}
	}
	firstIntent := runDeletionPath(service.store.runsRoot, first.ID)
	if err := os.WriteFile(filepath.Join(firstIntent, runFileName), []byte("{corrupt\n"), 0o600); err != nil {
		t.Fatalf("corrupt complete deletion intent: %v", err)
	}
	if err := service.DeleteRunAs(secondOwner, first.ID); err == nil {
		t.Fatal("corrupt foreign deletion intent was recovered")
	}
	for _, runID := range []string{first.ID, second.ID} {
		if err := requirePrivateDirectory(runDeletionPath(service.store.runsRoot, runID)); err != nil {
			t.Fatalf("retry advanced deletion intent %s: %v", runID, err)
		}
	}
}

func TestRunDeletionStartupRejectsCorruptCompleteIntent(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-corrupt-startup-owner", false)
	run := terminalRunDeletionFixture(t, service, owner)
	intentPath := runDeletionPath(service.store.runsRoot, run.ID)
	if err := os.Rename(filepath.Join(service.store.runsRoot, run.ID), intentPath); err != nil {
		t.Fatalf("stage run deletion intent: %v", err)
	}
	if err := os.WriteFile(filepath.Join(intentPath, lifecycleFileName), []byte("{corrupt\n"), 0o600); err != nil {
		t.Fatalf("corrupt deletion lifecycle: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := newStandaloneStore(root); err == nil {
		t.Fatal("startup recovered corrupt complete deletion intent")
	}
	if err := requirePrivateDirectory(intentPath); err != nil {
		t.Fatalf("startup consumed corrupt deletion intent: %v", err)
	}
}

func TestRunDeletionRetryRecoversAttestationCleanupFailure(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-attestation-delete-owner", false)
	other := testLifecycleActor(t, "run-attestation-delete-other", false)
	run := terminalRunDeletionFixture(t, service, owner)
	attestationPath := filepath.Join(service.store.attestationRoot, run.ID+".json")
	if err := service.store.writeExecutionAttestation(validExecutionAttestation(t, run.ID)); err != nil {
		t.Fatalf("write attestation cleanup fixture: %v", err)
	}
	delegate := atomicRunNamespacePersistence{}
	service.store.runPersistence = &faultingRunNamespacePersistence{
		delegate: delegate,
		syncDirectory: func(path, description string) error {
			if path == service.store.attestationRoot {
				return errors.New("injected attestation directory sync failure")
			}
			return delegate.SyncDirectory(path, description)
		},
	}

	if err := service.DeleteRunAs(owner, run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("attestation cleanup sync error=%v, want ErrConflict", err)
	}
	intentPath := runDeletionPath(service.store.runsRoot, run.ID)
	if err := requirePrivateDirectory(intentPath); err != nil {
		t.Fatalf("attestation failure consumed run deletion intent: %v", err)
	}
	if _, err := os.Lstat(attestationPath); !os.IsNotExist(err) {
		t.Fatalf("fault did not expose the remove-before-sync crash window: %v", err)
	}
	if err := service.DeleteRunAs(other, run.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner attestation retry error=%v, want ErrForbidden", err)
	}
	service.store.runPersistence = delegate
	if err := service.DeleteRunAs(owner, run.ID); err != nil {
		t.Fatalf("owner retry did not finish attestation cleanup: %v", err)
	}
	if _, err := os.Lstat(attestationPath); !os.IsNotExist(err) {
		t.Fatalf("run deletion retained execution attestation: %v", err)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close attestation run deletion service: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("reopen after attestation run deletion recovery: %v", err)
	}
}
