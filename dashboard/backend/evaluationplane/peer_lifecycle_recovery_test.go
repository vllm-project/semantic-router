package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestConcurrentStoreRecoveryCannotDeleteLiveLifecycleAuditTemp(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	writer := &pausingLifecycleAuditWriter{entered: make(chan string, 1), release: make(chan struct{})}
	service.store.lifecycleAuditWriter = writer
	createDone := make(chan error, 1)
	go func() {
		_, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
		createDone <- err
	}()
	var temporaryPath string
	select {
	case temporaryPath = <-writer.entered:
	case <-time.After(time.Second):
		t.Fatal("lifecycle audit writer did not reach post-sync pause")
	}
	reopenDone := make(chan error, 1)
	reopenAttempted := make(chan struct{})
	go func() {
		close(reopenAttempted)
		_, err := openTestPeerStore(t, service.store, LifecycleLimits{})
		reopenDone <- err
	}()
	<-reopenAttempted
	select {
	case err := <-reopenDone:
		t.Fatalf("concurrent store crossed active lifecycle writer: %v", err)
	default:
	}
	if _, err := os.Lstat(temporaryPath); err != nil {
		t.Fatalf("concurrent recovery removed live lifecycle audit temp: %v", err)
	}
	close(writer.release)
	if err := <-createDone; err != nil {
		t.Fatalf("lifecycle mutation after recovery wait: %v", err)
	}
	select {
	case err := <-reopenDone:
		if err != nil {
			t.Fatalf("concurrent store after lifecycle mutation: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("concurrent store did not resume after lifecycle mutation")
	}
}

func TestPeerOpenDoesNotRecoverLifecycleAuditTemporaryFiles(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	temporary, err := os.CreateTemp(service.store.lifecycleAuditRoot, lifecycleAuditTempPrefix+"*")
	if err != nil {
		t.Fatalf("create staged lifecycle audit record: %v", err)
	}
	temporaryPath := temporary.Name()
	if err := temporary.Close(); err != nil {
		t.Fatalf("close staged lifecycle audit record: %v", err)
	}
	sequenceBefore := service.store.lifecycle.sequence

	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); !errors.Is(err, ErrConflict) {
		t.Fatalf("peer open error=%v, want staged audit conflict", err)
	}
	if _, err := os.Lstat(temporaryPath); err != nil {
		t.Fatalf("peer opener removed staged lifecycle audit record: %v", err)
	}
	if got := service.store.lifecycle.sequence; got != sequenceBefore {
		t.Fatalf("peer opener changed lifecycle audit sequence from %d to %d", sequenceBefore, got)
	}
}

func TestPeerOpenDoesNotCommitVisibleLifecyclePolicyExpansion(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	expanded := service.store.lifecyclePolicy.Limits
	expanded.MaxOwnerBytes *= 2
	expanded.MaxStoreBytes *= 2
	faults := &faultingLifecyclePersistence{
		delegate:      atomicLifecyclePolicyPersistence{},
		writeFailures: 1,
		syncFailures:  1,
	}
	service.store.lifecyclePersistence = faults

	service.store.lifecycle.mu.Lock()
	policyErr := service.store.initializeLifecyclePolicyUnlocked(expanded)
	service.store.lifecycle.mu.Unlock()
	if policyErr == nil {
		t.Fatal("visible lifecycle policy expansion unexpectedly reported durable success")
	}
	policyPath := filepath.Join(service.store.lifecycleRoot, lifecyclePolicyFileName)
	var visible lifecycleStorePolicy
	if err := readJSON(policyPath, &visible); err != nil || visible.Limits != expanded {
		t.Fatalf("visible policy expansion missing: policy=%+v err=%v", visible, err)
	}
	if service.store.lifecyclePolicy.Limits == expanded {
		t.Fatal("uncertain lifecycle policy expansion was installed in memory")
	}

	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); !errors.Is(err, ErrConflict) {
		t.Fatalf("peer open error=%v, want policy durability conflict", err)
	}
	if service.store.lifecyclePolicy.Limits == expanded {
		t.Fatal("peer opener installed another Service's uncertain policy expansion")
	}

	service.store.lifecycle.mu.Lock()
	policyErr = service.store.initializeLifecyclePolicyUnlocked(expanded)
	service.store.lifecycle.mu.Unlock()
	if policyErr == nil {
		t.Fatal("owner retry bypassed the injected lifecycle policy sync failure")
	}
	service.store.lifecycle.mu.Lock()
	policyErr = service.store.initializeLifecyclePolicyUnlocked(expanded)
	service.store.lifecycle.mu.Unlock()
	if policyErr != nil || service.store.lifecyclePolicy.Limits != expanded {
		t.Fatalf("owner retry did not commit lifecycle policy expansion: policy=%+v err=%v", service.store.lifecyclePolicy.Limits, policyErr)
	}
}
