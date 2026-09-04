package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCreateBundleConflictPreservesPublishedBundle(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	run, manifest := preparePendingRun(t, service, request)
	conflictDir := filepath.Join(root, "runs", run.ID)
	if err := os.Mkdir(conflictDir, 0o700); err != nil {
		t.Fatalf("create conflict directory: %v", err)
	}
	marker := filepath.Join(conflictDir, "owner")
	if err := os.WriteFile(marker, []byte("existing\n"), 0o600); err != nil {
		t.Fatalf("write conflict marker: %v", err)
	}

	if _, err := service.store.CreateBundleAs(SystemActor(), run, manifest); !errors.Is(err, ErrConflict) {
		t.Fatalf("CreateBundle conflict error=%v, want ErrConflict", err)
	}
	if data, err := os.ReadFile(marker); err != nil || string(data) != "existing\n" {
		t.Fatalf("conflict destination was replaced: data=%q err=%v", data, err)
	}
	assertNoStagedRunBundles(t, filepath.Join(root, "runs"))
}

func TestCreateBundlePublishesCompleteInitialSnapshot(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, manifest := preparePendingRun(t, service, validCreateRequest())
	manifestPath, err := service.store.CreateBundleAs(SystemActor(), run, manifest)
	if err != nil {
		t.Fatalf("CreateBundle: %v", err)
	}
	if manifestPath != filepath.Join(root, "runs", run.ID, manifestFileName) {
		t.Fatalf("manifest path=%q", manifestPath)
	}
	if stored, err := service.store.GetRun(run.ID); err != nil || stored.ID != run.ID {
		t.Fatalf("stored run=%+v err=%v", stored, err)
	}
	if durable, _, err := service.readDurableManifest(run.ID); err != nil || durable.ManifestDigest != manifest.ManifestDigest {
		t.Fatalf("durable manifest=%+v err=%v", durable, err)
	}
	assertInitialSnapshotEvent(t, service.store, run.ID)
	assertNoStagedRunBundles(t, filepath.Join(root, "runs"))
}

func TestNewStoreRecoversOnlyPrivateStagedBundleDirectories(t *testing.T) {
	root := t.TempDir()
	if err := os.Chmod(root, 0o700); err != nil {
		t.Fatalf("protect store root: %v", err)
	}
	store, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	orphan := filepath.Join(store.runsRoot, stagedRunBundlePrefix+"crash123")
	if err := os.Mkdir(orphan, 0o700); err != nil {
		t.Fatalf("create staged orphan: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("recover staged orphan: %v", err)
	}
	if _, err := os.Lstat(orphan); !os.IsNotExist(err) {
		t.Fatalf("staged orphan survived recovery: %v", err)
	}

	invalid := filepath.Join(store.runsRoot, stagedRunBundlePrefix+"file123")
	if err := os.WriteFile(invalid, []byte("not a staged directory\n"), 0o600); err != nil {
		t.Fatalf("create invalid staged entry: %v", err)
	}
	if _, err := newStandaloneStore(root); err == nil {
		t.Fatal("NewStore silently removed an invalid staged entry")
	}
}

func TestAppendEventRejectsNonMonotonicDurableHistoryAfterCacheLoss(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	eventsPath := filepath.Join(root, "runs", run.ID, eventsFileName)
	initial, err := os.ReadFile(eventsPath)
	if err != nil {
		t.Fatalf("read initial event history: %v", err)
	}
	if err := os.WriteFile(eventsPath, append(initial, initial...), 0o600); err != nil {
		t.Fatalf("duplicate initial event history: %v", err)
	}
	service.store.runIndex.mu.Lock()
	delete(service.store.runIndex.eventSequences, run.ID)
	service.store.runIndex.mu.Unlock()
	if _, err := service.store.AppendEvent(Event{RunID: run.ID, Type: "progress"}); err == nil || !strings.Contains(err.Error(), "strictly monotonic") {
		t.Fatalf("AppendEvent non-monotonic history error=%v", err)
	}
}

func preparePendingRun(t *testing.T, service *Service, request CreateRunRequest) (Run, RunManifest) {
	t.Helper()
	registry, err := service.registrySnapshot()
	if err != nil {
		t.Fatalf("registrySnapshot: %v", err)
	}
	validated, target, err := service.validateCreateRequest(registry, request)
	if err != nil {
		t.Fatalf("validateCreateRequest: %v", err)
	}
	evidenceLevel, err := selectedSuiteEvidenceLevel(registry, validated.SuiteIDs, validated.Mode)
	if err != nil {
		t.Fatalf("selectedSuiteEvidenceLevel: %v", err)
	}
	run, manifest, err := service.newPendingRunManifest(registry, validated, target, evidenceLevel)
	if err != nil {
		t.Fatalf("newPendingRunManifest: %v", err)
	}
	return run, manifest
}

func assertInitialSnapshotEvent(t *testing.T, store *Store, runID string) {
	t.Helper()
	events, err := store.EventsAfter(runID, 0)
	if err != nil {
		t.Fatalf("EventsAfter initial snapshot: %v", err)
	}
	if len(events) != 1 || events[0].ID != "1" || events[0].RunID != runID || events[0].Type != "snapshot" {
		t.Fatalf("initial events=%+v, want one canonical snapshot", events)
	}
}

func assertNoStagedRunBundles(t *testing.T, runsRoot string) {
	t.Helper()
	entries, err := os.ReadDir(runsRoot)
	if err != nil {
		t.Fatalf("read runs root: %v", err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), stagedRunBundlePrefix) {
			t.Fatalf("staged run bundle leaked after publication: %s", entry.Name())
		}
	}
}
