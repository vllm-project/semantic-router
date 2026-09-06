package evaluationplane

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDeleteRunCollectsOnlyUnreferencedCASObjects(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	first := createPendingGCTestRun(t, service, "first")
	second := createPendingGCTestRun(t, service, "second")
	shared := writeGCCASArtifact(t, service.store, first.ID, "metrics.json", []byte("shared\n"))
	writeGCCASArtifact(t, service.store, second.ID, "metrics.json", []byte("shared\n"))
	unique := writeGCCASArtifact(t, service.store, first.ID, "gates.json", []byte("first-only\n"))

	if err := service.DeleteRunAs(SystemActor(), first.ID); err != nil {
		t.Fatalf("DeleteRun first: %v", err)
	}
	assertCASObjectExists(t, service.store, shared, true)
	assertCASObjectExists(t, service.store, unique, false)

	if err := service.DeleteRunAs(SystemActor(), second.ID); err != nil {
		t.Fatalf("DeleteRun second: %v", err)
	}
	assertCASObjectExists(t, service.store, shared, false)
}

func TestNewStoreFinishesCASCollectionAfterCommittedRunDeletion(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run := createPendingGCTestRun(t, service, "crash-window")
	digest := writeGCCASArtifact(t, service.store, run.ID, "metrics.json", []byte("orphan-after-delete\n"))
	if err := os.RemoveAll(filepath.Join(service.store.runsRoot, run.ID)); err != nil {
		t.Fatalf("simulate committed run removal: %v", err)
	}
	if err := syncEvaluationDirectory(service.store.runsRoot, "test run deletion"); err != nil {
		t.Fatalf("sync simulated run removal: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before CAS recovery restart: %v", err)
	}
	reopened, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("NewStore recovery: %v", err)
	}
	assertCASObjectExists(t, reopened, digest, false)
}

func TestPeerStoreOpenDoesNotCollectCASOrAppendStartupAudit(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	data := []byte("unreferenced-while-owner-is-live\n")
	digest := strings.TrimPrefix(digestBytes(data), "sha256:")
	if err := os.WriteFile(
		filepath.Join(service.store.root, "objects", "sha256", digest), data, 0o600,
	); err != nil {
		t.Fatalf("write unreferenced CAS object: %v", err)
	}
	sequenceBefore := service.store.lifecycle.sequence

	peer := newTestPeerStore(t, service.store)
	assertCASObjectExists(t, peer, digest, true)
	if got := service.store.lifecycle.sequence; got != sequenceBefore {
		t.Fatalf("peer open changed audit sequence from %d to %d", sequenceBefore, got)
	}
}

func TestDeleteRunDefersCASSweepWhenRemainingOwnershipIsInvalid(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	deleted := createPendingGCTestRun(t, service, "deleted")
	keeper := createPendingGCTestRun(t, service, "keeper")
	stale := writeGCCASArtifact(t, service.store, deleted.ID, "metrics.json", []byte("candidate-garbage\n"))
	live := writeGCCASArtifact(t, service.store, keeper.ID, "metrics.json", []byte("still-live\n"))
	if err := os.WriteFile(
		filepath.Join(service.store.runsRoot, keeper.ID, "lineage.json"),
		[]byte("{\"schema_version\":\"evaluation.v1\",\"unknown\":true}\n"),
		0o600,
	); err != nil {
		t.Fatalf("write invalid remaining lineage: %v", err)
	}

	if err := service.DeleteRunAs(SystemActor(), deleted.ID); err != nil {
		t.Fatalf("DeleteRun with deferred GC: %v", err)
	}
	assertCASObjectExists(t, service.store, live, true)
	assertCASObjectExists(t, service.store, stale, true)
}

func createPendingGCTestRun(t *testing.T, service *Service, name string) Run {
	t.Helper()
	request := validCreateRequest()
	request.Name = name
	run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun %s: %v", name, err)
	}
	return run
}

func writeGCCASArtifact(t *testing.T, store *Store, runID, name string, data []byte) string {
	t.Helper()
	hexDigest := strings.TrimPrefix(digestBytes(data), "sha256:")
	if err := os.WriteFile(filepath.Join(store.runsRoot, runID, name), data, 0o600); err != nil {
		t.Fatalf("write run artifact %s: %v", name, err)
	}
	objectPath := filepath.Join(store.root, "objects", "sha256", hexDigest)
	if err := os.WriteFile(objectPath, data, 0o600); err != nil {
		t.Fatalf("write CAS object %s: %v", hexDigest, err)
	}
	return hexDigest
}

func assertCASObjectExists(t *testing.T, store *Store, digest string, want bool) {
	t.Helper()
	_, err := os.Lstat(filepath.Join(store.root, "objects", "sha256", digest))
	if want && err != nil {
		t.Fatalf("CAS object %s missing: %v", digest, err)
	}
	if !want && !os.IsNotExist(err) {
		t.Fatalf("CAS object %s still exists: %v", digest, err)
	}
}
