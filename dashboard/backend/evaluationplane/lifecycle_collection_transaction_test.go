package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

type faultingLifecycleCollectionPersistence struct {
	delegate      lifecycleCollectionPersistence
	writeCalls    int
	failWriteAt   int
	visible       bool
	partial       bool
	resolveCalls  int
	failResolveAt int
}

func (p *faultingLifecycleCollectionPersistence) WriteHeader(
	path string,
	value lifecycleCollectionHeader,
) error {
	p.writeCalls++
	if p.writeCalls == p.failWriteAt && !p.visible {
		return errors.New("injected collection transaction write failure")
	}
	if err := p.delegate.WriteHeader(path, value); err != nil {
		return err
	}
	if p.writeCalls == p.failWriteAt {
		return errors.New("injected visible collection transaction write failure")
	}
	return nil
}

func (p *faultingLifecycleCollectionPersistence) AppendProgress(
	path string,
	value lifecycleCollectionProgress,
) error {
	p.writeCalls++
	if p.writeCalls == p.failWriteAt && !p.visible {
		if p.partial {
			encoded, err := json.Marshal(value)
			if err != nil {
				return err
			}
			file, err := openBundleFile(path, os.O_WRONLY|os.O_APPEND)
			if err != nil {
				return err
			}
			_, writeErr := file.Write(encoded[:len(encoded)/2])
			closeErr := file.Close()
			if err := errors.Join(writeErr, closeErr); err != nil {
				return err
			}
		}
		return errors.New("injected collection transaction progress failure")
	}
	if err := p.delegate.AppendProgress(path, value); err != nil {
		return err
	}
	if p.writeCalls == p.failWriteAt {
		return errors.New("injected visible collection transaction progress failure")
	}
	return nil
}

func (p *faultingLifecycleCollectionPersistence) Resolve(path, directory string) error {
	p.resolveCalls++
	if p.resolveCalls == p.failResolveAt {
		return errors.New("injected collection transaction resolve failure")
	}
	return p.delegate.Resolve(path, directory)
}

func expiredCollectionRuns(t *testing.T, service *Service, count int) []Run {
	t.Helper()
	runs := make([]Run, 0, count)
	for range count {
		run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
		if err != nil {
			t.Fatalf("create collection transaction run: %v", err)
		}
		runs = append(runs, completeTestRun(t, service, run))
	}
	service.store.lifecycleNow = func() time.Time { return time.Now().UTC().Add(90 * 24 * time.Hour) }
	return runs
}

func TestLifecycleCollectionTransactionResumesFrozenPlanAndReturnsReceipt(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	expiredCollectionRuns(t, service, 2)
	actor := SystemActor()
	dryRun, transactionErr := service.CollectLifecycle(actor, CollectionRequest{})
	if transactionErr != nil || len(dryRun.Plan.Candidates) != 2 {
		t.Fatalf("build collection transaction plan: candidates=%d err=%v", len(dryRun.Plan.Candidates), transactionErr)
	}

	faults := &faultingLifecycleCollectionPersistence{
		delegate: atomicLifecycleCollectionPersistence{}, failWriteAt: 2,
	}
	service.store.collectionPersistence = faults
	request := CollectionRequest{Apply: true, PlanDigest: dryRun.Plan.PlanDigest}
	if _, err := service.CollectLifecycle(actor, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("collection progress write failure error=%v, want ErrConflict", err)
	}
	service.store.lifecycle.mu.Lock()
	pending, exists, readErr := service.store.readLifecycleCollectionTransactionUnlocked()
	service.store.lifecycle.mu.Unlock()
	if readErr != nil || !exists {
		t.Fatalf("read pending collection transaction: exists=%t err=%v", exists, readErr)
	}
	if pending.State != collectionStateApplying || pending.Next != 0 {
		t.Fatalf("pending collection progress=%+v", pending)
	}
	first := dryRun.Plan.Candidates[0]
	if _, err := os.Lstat(filepath.Join(service.store.runsRoot, first.RunID)); !os.IsNotExist(err) {
		t.Fatalf("first independently durable deletion was not committed: %v", err)
	}

	other, transactionErr := NewActor("collection-transaction-other-admin", true)
	if transactionErr != nil {
		t.Fatal(transactionErr)
	}
	if _, err := service.CollectLifecycle(other, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("different actor resumed collection transaction: %v", err)
	}
	if _, err := service.CollectLifecycle(actor, CollectionRequest{
		Apply: true, PlanDigest: digestString("different collection plan"),
	}); !errors.Is(err, ErrConflict) {
		t.Fatalf("different digest replaced collection transaction: %v", err)
	}

	completed, transactionErr := service.CollectLifecycle(actor, request)
	if transactionErr != nil {
		t.Fatalf("resume collection transaction: %v", transactionErr)
	}
	if !completed.Applied || len(completed.DeletedRunIDs) != 2 {
		t.Fatalf("completed collection receipt=%+v", completed)
	}
	retried, transactionErr := service.CollectLifecycle(actor, request)
	if transactionErr != nil || !reflect.DeepEqual(retried, completed) {
		t.Fatalf("completed receipt retry=%+v err=%v, want %+v", retried, transactionErr, completed)
	}
}

func TestLifecycleCollectionStartupRecoversPendingTransactionAndPeerFailsClosed(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	expiredCollectionRuns(t, service, 2)
	actor := SystemActor()
	dryRun, transactionErr := service.CollectLifecycle(actor, CollectionRequest{})
	if transactionErr != nil {
		t.Fatalf("build restart collection plan: %v", transactionErr)
	}
	service.store.collectionPersistence = &faultingLifecycleCollectionPersistence{
		delegate: atomicLifecycleCollectionPersistence{}, failWriteAt: 2,
	}
	request := CollectionRequest{Apply: true, PlanDigest: dryRun.Plan.PlanDigest}
	if _, err := service.CollectLifecycle(actor, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("create pending restart transaction: %v", err)
	}

	peer, transactionErr := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		RouterAPIURL: service.registrySource.routerAPIURL, EnvoyURL: service.registrySource.envoyURL,
		CodeRevision: service.codeRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if peer != nil {
		_ = peer.Close()
	}
	if !errors.Is(transactionErr, ErrConflict) {
		t.Fatalf("peer opener advanced pending collection transaction: %v", transactionErr)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close collection transaction owner: %v", err)
	}
	restarted := reopenTestService(t, root)
	receipt, transactionErr := restarted.CollectLifecycle(actor, request)
	if transactionErr != nil || len(receipt.DeletedRunIDs) != 2 {
		t.Fatalf("startup collection recovery receipt=%+v err=%v", receipt, transactionErr)
	}
	restarted.store.lifecycle.mu.Lock()
	durable, exists, readErr := restarted.store.readLifecycleCollectionTransactionUnlocked()
	restarted.store.lifecycle.mu.Unlock()
	if readErr != nil || !exists ||
		durable.State != collectionStateCompleted || durable.Next != 2 {
		t.Fatalf("startup collection transaction=%+v exists=%t err=%v", durable, exists, readErr)
	}
}

func TestLifecycleCollectionVisibleTransactionWriteRequiresRetrySync(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	expiredCollectionRuns(t, service, 1)
	dryRun, transactionErr := service.CollectLifecycle(SystemActor(), CollectionRequest{})
	if transactionErr != nil {
		t.Fatal(transactionErr)
	}
	faults := &faultingLifecycleCollectionPersistence{
		delegate: atomicLifecycleCollectionPersistence{}, failWriteAt: 1, visible: true, failResolveAt: 1,
	}
	service.store.collectionPersistence = faults
	request := CollectionRequest{Apply: true, PlanDigest: dryRun.Plan.PlanDigest}
	if _, err := service.CollectLifecycle(SystemActor(), request); !errors.Is(err, ErrConflict) {
		t.Fatalf("visible transaction write error=%v, want ErrConflict", err)
	}
	if _, err := service.GetRunAs(SystemActor(), dryRun.Plan.Candidates[0].RunID); err != nil {
		t.Fatalf("candidate was deleted before transaction durability retry: %v", err)
	}
	if _, err := service.CollectLifecycle(SystemActor(), request); !errors.Is(err, ErrConflict) {
		t.Fatalf("failed transaction resolve error=%v, want ErrConflict", err)
	}
	if service.store.lifecycle.pendingCollection == nil {
		t.Fatal("failed transaction resolve cleared the pending projection")
	}
	if _, err := service.GetRunAs(SystemActor(), dryRun.Plan.Candidates[0].RunID); err != nil {
		t.Fatalf("candidate was deleted before a successful durability retry: %v", err)
	}
	result, transactionErr := service.CollectLifecycle(SystemActor(), request)
	if transactionErr != nil || len(result.DeletedRunIDs) != 1 {
		t.Fatalf("visible transaction retry result=%+v err=%v", result, transactionErr)
	}
}

func TestLifecycleCollectionCompletedHeaderUncertaintyBlocksPeerUntilMatchingRetry(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	actor := SystemActor()
	dryRun, transactionErr := service.CollectLifecycle(actor, CollectionRequest{})
	if transactionErr != nil || len(dryRun.Plan.Candidates) != 0 {
		t.Fatalf("build empty collection plan: candidates=%d err=%v", len(dryRun.Plan.Candidates), transactionErr)
	}
	service.store.collectionPersistence = &faultingLifecycleCollectionPersistence{
		delegate: atomicLifecycleCollectionPersistence{}, failWriteAt: 1, visible: true,
	}
	request := CollectionRequest{Apply: true, PlanDigest: dryRun.Plan.PlanDigest}
	if _, err := service.CollectLifecycle(actor, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("visible completed header error=%v, want ErrConflict", err)
	}
	if service.store.lifecycle.pendingCollection == nil {
		t.Fatal("visible completed header did not retain its pending projection")
	}

	peer, transactionErr := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		RouterAPIURL: service.registrySource.routerAPIURL, EnvoyURL: service.registrySource.envoyURL,
		CodeRevision: service.codeRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if peer != nil {
		_ = peer.Close()
	}
	if !errors.Is(transactionErr, ErrConflict) {
		t.Fatalf("peer accepted uncertain completed receipt: %v", transactionErr)
	}

	result, transactionErr := service.CollectLifecycle(actor, request)
	if transactionErr != nil || !result.Applied || len(result.Plan.Candidates) != 0 {
		t.Fatalf("matching completed receipt retry=%+v err=%v", result, transactionErr)
	}
	if service.store.lifecycle.pendingCollection != nil {
		t.Fatal("matching durability retry did not clear the pending projection")
	}
}

func TestLifecycleCollectionRetryRepairsPartialProgressTail(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	expiredCollectionRuns(t, service, 1)
	actor := SystemActor()
	dryRun, transactionErr := service.CollectLifecycle(actor, CollectionRequest{})
	if transactionErr != nil {
		t.Fatal(transactionErr)
	}
	service.store.collectionPersistence = &faultingLifecycleCollectionPersistence{
		delegate: atomicLifecycleCollectionPersistence{}, failWriteAt: 2, partial: true,
	}
	request := CollectionRequest{Apply: true, PlanDigest: dryRun.Plan.PlanDigest}
	if _, err := service.CollectLifecycle(actor, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("partial progress append error=%v, want ErrConflict", err)
	}
	data, transactionErr := os.ReadFile(service.store.lifecycleCollectionPath())
	if transactionErr != nil {
		t.Fatal(transactionErr)
	}
	if bytes.HasSuffix(data, []byte{'\n'}) {
		t.Fatal("partial progress fixture unexpectedly ended at a record boundary")
	}
	service.store.lifecycle.mu.Lock()
	pending, exists, readErr := service.store.readLifecycleCollectionTransactionUnlocked()
	service.store.lifecycle.mu.Unlock()
	if readErr != nil || !exists || pending.Next != 0 {
		t.Fatalf("read complete progress prefix: transaction=%+v exists=%t err=%v", pending, exists, readErr)
	}

	result, transactionErr := service.CollectLifecycle(actor, request)
	if transactionErr != nil || len(result.DeletedRunIDs) != 1 {
		t.Fatalf("repair partial progress tail: result=%+v err=%v", result, transactionErr)
	}
	data, transactionErr = os.ReadFile(service.store.lifecycleCollectionPath())
	if transactionErr != nil || !bytes.HasSuffix(data, []byte{'\n'}) {
		t.Fatalf("repaired collection log boundary: err=%v", transactionErr)
	}
}

func TestLifecycleCollectionTemporaryFileBlocksPeerAndStartupRemovesIt(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	temporaryPath := filepath.Join(
		service.store.collectionRoot,
		lifecycleCollectionTemporaryPrefix+"orphan",
	)
	if err := os.WriteFile(temporaryPath, []byte("incomplete header"), 0o600); err != nil {
		t.Fatal(err)
	}
	peer, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		RouterAPIURL: service.registrySource.routerAPIURL, EnvoyURL: service.registrySource.envoyURL,
		CodeRevision: service.codeRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if peer != nil {
		_ = peer.Close()
	}
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("peer accepted lifecycle collection temporary file: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close collection temporary file owner: %v", err)
	}
	restarted := reopenTestService(t, root)
	if _, err := os.Lstat(temporaryPath); !os.IsNotExist(err) {
		t.Fatalf("startup did not remove lifecycle collection temporary file: %v", err)
	}
	if restarted.store.lifecycle.pendingCollection != nil {
		t.Fatal("startup cleanup created a pending collection projection")
	}
}

func TestCollectionPathExistsFailsClosedOnStatError(t *testing.T) {
	parent := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(parent, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if exists, err := collectionPathExists(filepath.Join(parent, "child")); err == nil || exists {
		t.Fatalf("stat failure exists=%t err=%v, want fail closed", exists, err)
	}
}

func TestLifecycleCollectionPlanBoundsAreSelfReadable(t *testing.T) {
	plan := CollectionPlan{
		Candidates: make([]CollectionPlanItem, maxLifecycleCollectionCandidates),
		Skipped:    map[string]int{"batch_limit": 1},
	}
	identity := collectionPlanIdentity{
		Candidates: make([]collectionItemIdentity, maxLifecycleCollectionCandidates),
		Skipped:    map[string]int{"batch_limit": 1},
	}
	if err := validateLifecycleCollectionPlanBounds(plan, identity); err != nil {
		t.Fatalf("near-bound collection plan was not persistable: %v", err)
	}
	header := lifecycleCollectionHeader{
		RecordType: "plan", SchemaVersion: lifecycleCollectionSchemaVersion,
		ActorPrincipalDigest: digestString("collection-size-bound"), State: collectionStateApplying,
		Plan: plan, PlanIdentity: identity,
	}
	header.HeaderDigest = lifecycleCollectionHeaderDigest(header)
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	projectedBytes := int64(len(encoded) + 1 + len(plan.Candidates)*maxLifecycleCollectionProgressBytes)
	if err := validateLifecycleCollectionPlanBoundsWithin(plan, identity, projectedBytes); err != nil {
		t.Fatalf("transaction at serialized byte bound: %v", err)
	}
	if err := validateLifecycleCollectionPlanBoundsWithin(plan, identity, projectedBytes-1); !errors.Is(err, ErrConflict) {
		t.Fatalf("transaction over serialized byte bound error=%v, want ErrConflict", err)
	}
	plan.Candidates = append(plan.Candidates, CollectionPlanItem{})
	identity.Candidates = append(identity.Candidates, collectionItemIdentity{})
	if err := validateLifecycleCollectionPlanBounds(plan, identity); !errors.Is(err, ErrConflict) {
		t.Fatalf("over-bound collection plan error=%v, want ErrConflict", err)
	}
}
