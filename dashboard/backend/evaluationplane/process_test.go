package evaluationplane

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestWorkerStagingCannotWriteSiblingOrControlBundles(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	realRun := filepath.Join(root, "runs", run.ID)
	realManifest := filepath.Join(realRun, manifestFileName)
	manifestBefore, err := os.ReadFile(realManifest)
	if err != nil {
		t.Fatalf("read real manifest: %v", err)
	}
	statusBefore, err := os.ReadFile(filepath.Join(realRun, runFileName))
	if err != nil {
		t.Fatalf("read real status: %v", err)
	}
	staging, err := prepareWorkerStaging(ProcessSpec{
		ManifestPath: realManifest, StorePath: root,
		executionContracts:  serviceExecutionContractsForTest(t, service),
		evidencePublication: service.store.withEvidenceSerialization,
	})
	if err != nil {
		t.Fatalf("prepareWorkerStaging: %v", err)
	}
	defer staging.cleanup()
	if staging.storePath == root || filepath.Dir(staging.manifestPath) == realRun {
		t.Fatalf("worker staging aliases the durable store: %+v", staging)
	}
	stagedRun := filepath.Join(staging.storePath, "runs", run.ID)
	for _, protected := range []string{runFileName, eventsFileName} {
		if _, err := os.Stat(filepath.Join(stagedRun, protected)); !os.IsNotExist(err) {
			t.Fatalf("worker staging exposes protected %s: %v", protected, err)
		}
	}
	for _, name := range workerRunArtifactNames {
		if name == manifestFileName {
			continue
		}
		if err := os.WriteFile(filepath.Join(stagedRun, name), []byte("{}\n"), 0o600); err != nil {
			t.Fatalf("write staged %s: %v", name, err)
		}
	}
	writeStagingTestLineage(t, filepath.Join(stagedRun, "lineage.json"))
	for _, directory := range []string{
		filepath.Join(staging.storePath, "objects"),
		filepath.Join(staging.storePath, "objects", "sha256"),
	} {
		if err := os.Mkdir(directory, 0o700); err != nil {
			t.Fatalf("create staged CAS: %v", err)
		}
	}
	objectBytes := []byte("{}\n")
	objectName := strings.TrimPrefix(digestBytes(objectBytes), "sha256:")
	for _, name := range workerRunArtifactNames {
		data, err := os.ReadFile(filepath.Join(stagedRun, name))
		if err != nil {
			t.Fatalf("read staged %s for CAS: %v", name, err)
		}
		digest := strings.TrimPrefix(digestBytes(data), "sha256:")
		objectPath := filepath.Join(staging.storePath, "objects", "sha256", digest)
		if _, err := os.Stat(objectPath); err == nil {
			continue
		}
		if err := os.WriteFile(objectPath, data, 0o600); err != nil {
			t.Fatalf("write staged CAS object: %v", err)
		}
	}
	writeStagingTestLineage(t, filepath.Join(stagedRun, "lineage.json"))
	if err := staging.importEvidence(); err != nil {
		t.Fatalf("importEvidence: %v", err)
	}
	manifestAfter, _ := os.ReadFile(realManifest)
	statusAfter, _ := os.ReadFile(filepath.Join(realRun, runFileName))
	if !bytes.Equal(manifestBefore, manifestAfter) || !bytes.Equal(statusBefore, statusAfter) {
		t.Fatal("worker evidence import changed a server-owned control file")
	}
	if got, err := os.ReadFile(filepath.Join(root, "objects", "sha256", objectName)); err != nil || !bytes.Equal(got, objectBytes) {
		t.Fatalf("CAS import bytes=%q err=%v", got, err)
	}
	if got, err := os.ReadFile(filepath.Join(realRun, reportFileName)); err != nil || string(got) != "{}\n" {
		t.Fatalf("report import bytes=%q err=%v", got, err)
	}
}

func TestWorkerEvidenceRejectsStandaloneEventLedger(t *testing.T) {
	stagedRun := t.TempDir()
	if err := os.WriteFile(filepath.Join(stagedRun, "events.jsonl"), []byte("{}\n"), 0o600); err != nil {
		t.Fatalf("write standalone event ledger: %v", err)
	}
	if _, err := discoverWorkerRunArtifacts(stagedRun); err == nil || !strings.Contains(err.Error(), "unsupported run artifact") {
		t.Fatalf("standalone event ledger staging error=%v, want unsupported artifact rejection", err)
	}
}

func writeStagingTestLineage(t *testing.T, path string) {
	t.Helper()
	if err := writeJSONAtomic(path, map[string]any{
		"schema_version": SchemaVersion,
		"resolved_snapshot": map[string]any{
			"schema_version": SchemaVersion, "manifest_digest": digestString("staging-manifest"),
			"workload": map[string]any{}, "policy": map[string]any{}, "binding": map[string]any{},
			"pool": map[string]any{}, "arms": []any{}, "environment": map[string]any{},
			"fixture_ref": nil, "discovered_entrypoints": []any{}, "executors": []any{},
		},
		"normalized_suite_identities": nil,
	}); err != nil {
		t.Fatalf("write staging lineage: %v", err)
	}
}

func TestWorkerStagingRejectsOversizedStructuredReportBeforeImport(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	realRun := filepath.Join(root, "runs", run.ID)
	staging, err := prepareWorkerStaging(ProcessSpec{
		ManifestPath: filepath.Join(realRun, manifestFileName), StorePath: root,
		executionContracts:  serviceExecutionContractsForTest(t, service),
		evidencePublication: service.store.withEvidenceSerialization,
	})
	if err != nil {
		t.Fatalf("prepareWorkerStaging: %v", err)
	}
	defer staging.cleanup()
	stagedRun := filepath.Join(staging.storePath, "runs", run.ID)
	for _, name := range workerRunArtifactNames {
		if name == manifestFileName {
			continue
		}
		path := filepath.Join(stagedRun, name)
		if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
			t.Fatalf("write staged %s: %v", name, err)
		}
	}
	if err := os.Truncate(filepath.Join(stagedRun, reportFileName), maxStructuredArtifactBytes+1); err != nil {
		t.Fatalf("expand staged report: %v", err)
	}
	if err := staging.importEvidence(); err == nil || !strings.Contains(err.Error(), "per-file limit") {
		t.Fatalf("oversized report import error=%v", err)
	}
	if _, err := os.Stat(filepath.Join(realRun, reportFileName)); !os.IsNotExist(err) {
		t.Fatalf("oversized report was published: %v", err)
	}
}

func TestDecodeWorkerEventRejectsProtocolDrift(t *testing.T) {
	event, decodeErr := decodeWorkerEvent([]byte(`{"type":"progress","message":"running","track_id":"routing","progress":{"percent":25,"completed":1,"total":4}}`))
	if decodeErr != nil || event.Type != "progress" || event.TrackID != "routing" {
		t.Fatalf("decodeWorkerEvent=%+v err=%v", event, decodeErr)
	}
	if event.Message != "Evaluation progress updated" || event.Progress == nil || event.Progress.Message != event.Message {
		t.Fatalf("worker-controlled messages were not redacted: %+v", event)
	}
	recordCount := 4
	track, trackErr := decodeWorkerEvent([]byte(`{"type":"track","message":"contains provider secret","track_id":"routing","payload":{"record_count":4}}`))
	if trackErr != nil || track.Payload == nil || track.Payload.RecordCount == nil || *track.Payload.RecordCount != recordCount ||
		track.Message != "Evaluation track evidence collected" {
		t.Fatalf("typed track event=%+v err=%v", track, trackErr)
	}
	for _, line := range []string{
		`{"type":"completed","message":"done","payload":{"verdict":"pass"}}`,
		`{"type":"completed","message":"done","payload":{"verdict":"fail"}}`,
		`{"type":"completed","message":"done","payload":{"verdict":"unavailable"}}`,
	} {
		if _, err := decodeWorkerEvent([]byte(line)); err != nil {
			t.Fatalf("valid completed worker event was rejected: %s: %v", line, err)
		}
	}
	for _, line := range []string{
		`{"type":"shell","message":"run command"}`,
		`{"type":"progress","message":"running","command":"sh"}`,
		`{"type":"progress","message":""}`,
		`{"type":"progress","message":"running"} {}`,
		`{"type":"progress","message":"running","payload":{"record_count":1}}`,
		`{"type":"track","message":"running","track_id":"routing"}`,
		`{"type":"track","message":"running","payload":{"record_count":-1}}`,
		`{"type":"track","message":"running","payload":{"record_count":1,"url":"https://private.invalid"}}`,
		`{"type":"completed","message":"done"}`,
		`{"type":"completed","message":"done","payload":{"verdict":"maybe"}}`,
		`{"type":"completed","message":"done","payload":{"verdict":"not_applicable"}}`,
		`{"type":"completed","message":"done","payload":{"verdict":"waived"}}`,
	} {
		if _, err := decodeWorkerEvent([]byte(line)); err == nil {
			t.Fatalf("unsafe worker event was accepted: %s", line)
		}
	}
	oversized := []byte(`{"type":"progress","message":"` + strings.Repeat("x", maxWorkerEventLineBytes) + `"}`)
	if _, err := decodeWorkerEvent(oversized); err == nil {
		t.Fatal("oversized worker event was accepted")
	}
}

func TestWorkerEventsPersistOnlyTypedRedactedData(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	run = stageRunningTestRun(t, service, run)
	recordCount := 3
	if err := service.recordWorkerEvent(run.ID, WorkerEvent{
		Type: "track", Message: "api_key=private-value", TrackID: "routing",
		Progress: &RunProgress{Percent: 50, Completed: 1, Total: 999, CurrentTrackID: "routing", Message: "https://private.invalid"},
		Payload:  &WorkerEventPayload{RecordCount: &recordCount},
	}); err == nil || !strings.Contains(err.Error(), "immutable run contract") {
		t.Fatalf("invalid worker progress error=%v", err)
	}
	if err := service.recordWorkerEvent(run.ID, WorkerEvent{
		Type: "track", Message: "api_key=private-value", TrackID: "routing",
		Progress: &RunProgress{Percent: 50, Completed: 1, Total: 1, CurrentTrackID: "routing", Message: "https://private.invalid"},
		Payload:  &WorkerEventPayload{RecordCount: &recordCount},
	}); err != nil {
		t.Fatalf("record valid worker event: %v", err)
	}
	events, eventsErr := service.EventsAfterAs(SystemActor(), run.ID, "1")
	if eventsErr != nil || len(events) != 1 {
		t.Fatalf("EventsAfter=%+v err=%v", events, eventsErr)
	}
	event := events[0]
	if strings.Contains(event.Message, "private") || event.Progress == nil || strings.Contains(event.Progress.Message, "private") ||
		event.Progress.Total != run.Progress.Total || event.Payload == nil || event.Payload.RecordCount == nil || *event.Payload.RecordCount != recordCount {
		t.Fatalf("unsafe worker fields reached durable event: %+v", event)
	}
	if err := service.recordWorkerEvent(run.ID, WorkerEvent{
		Type: "progress", Message: "progress",
		Progress: &RunProgress{CurrentTrackID: "not-selected"},
	}); err == nil {
		t.Fatal("unknown progress track was accepted")
	}
}

func TestWorkerEventFloodFailsClosedAtPerRunLimit(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	run = stageRunningTestRun(t, service, run)
	service.mu.Lock()
	service.workerEvents[run.ID] = maxWorkerEventsPerRun - 1
	service.mu.Unlock()
	if err := service.recordWorkerEvent(run.ID, WorkerEvent{Type: "progress", Message: "last accepted"}); err != nil {
		t.Fatalf("last bounded worker event was rejected: %v", err)
	}
	if err := service.recordWorkerEvent(run.ID, WorkerEvent{Type: "progress", Message: "must stop"}); err == nil || !strings.Contains(err.Error(), "event limit") {
		t.Fatalf("event flood error=%v, want event limit", err)
	}
	events, eventsErr := service.EventsAfterAs(SystemActor(), run.ID, "")
	if eventsErr != nil || len(events) != 2 {
		t.Fatalf("bounded durable events=%d err=%v, want snapshot plus one worker event", len(events), eventsErr)
	}
}

func TestBoundedConcurrencyAndSuccessfulExitRequiresReport(t *testing.T) {
	process := &controlledProcess{started: make(chan ProcessSpec, 2), release: make(chan struct{}), writeReport: true}
	service, _ := newTestService(t, process, 1)
	first, firstErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if firstErr != nil {
		t.Fatalf("create first: %v", firstErr)
	}
	secondRequest := validCreateRequest()
	secondRequest.Name = "second"
	second, secondErr := service.CreateRunAs(context.Background(), SystemActor(), secondRequest)
	if secondErr != nil {
		t.Fatalf("create second: %v", secondErr)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), first.ID); err != nil {
		t.Fatalf("start first: %v", err)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), second.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("start second error=%v, want bounded-capacity ErrConflict", err)
	}
	select {
	case <-process.started:
	case <-time.After(time.Second):
		t.Fatal("first worker did not enter process")
	}
	queued, queuedErr := service.GetRunAs(SystemActor(), second.ID)
	if queuedErr != nil || queued.Status != StatusPending || queued.StartedAt != nil {
		t.Fatalf("capacity-bound run=%+v err=%v, want durable pending without worker", queued, queuedErr)
	}
	close(process.release)
	waitForRunStatus(t, service, first.ID, StatusCompleted)
	startRunAfterCapacityRelease(t, service, second.ID)
	waitForRunStatus(t, service, second.ID, StatusCompleted)

	noReportRelease := make(chan struct{})
	close(noReportRelease)
	noReportProcess := &controlledProcess{release: noReportRelease}
	noReportService, _ := newTestService(t, noReportProcess, 1)
	noReportRun, noReportErr := noReportService.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if noReportErr != nil {
		t.Fatalf("create no-report run: %v", noReportErr)
	}
	if _, err := noReportService.StartRunAs(context.Background(), SystemActor(), noReportRun.ID); err != nil {
		t.Fatalf("start no-report run: %v", err)
	}
	failed := waitForRunStatus(t, noReportService, noReportRun.ID, StatusFailed)
	if failed.Error == "" {
		t.Fatalf("missing report failure did not persist a safe error: %+v", failed)
	}
}

func TestServiceCloseCancelsAndWaitsForWorkerExit(t *testing.T) {
	process := &delayedCancelProcess{started: make(chan struct{}), cancelled: make(chan struct{}), exit: make(chan struct{})}
	service, root := newTestService(t, process, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err != nil {
		t.Fatalf("StartRun: %v", err)
	}
	<-process.started

	closed := make(chan error, 1)
	go func() { closed <- service.Close() }()
	<-process.cancelled
	select {
	case err := <-closed:
		t.Fatalf("Close returned before worker exit: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	close(process.exit)
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not wait for worker completion")
	}
	restarted := reopenTestService(t, root)
	cancelled, err := restarted.GetRunAs(SystemActor(), run.ID)
	if err != nil || cancelled.Status != StatusCancelled {
		t.Fatalf("durable run after Close=%+v err=%v, want cancelled", cancelled, err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("idempotent Close: %v", err)
	}

	if _, err := service.GetRunAs(SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("GetRun after Close error=%v, want ErrConflict", err)
	}
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest()); !errors.Is(err, ErrConflict) {
		t.Fatalf("CreateRun after Close error=%v, want ErrConflict", err)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("StartRun after Close error=%v, want ErrConflict", err)
	}
	if _, _, err := service.SubscribeAs(SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("Subscribe after Close error=%v, want ErrConflict", err)
	}
}

func TestWorkerTimeoutFailsRunAndReleasesCapacity(t *testing.T) {
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create evaluation root: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	process := &controlledProcess{started: make(chan ProcessSpec, 1)}
	service, serviceErr := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		CodeRevision: testSourceRevision, MaxConcurrent: 1, WorkerTimeout: 25 * time.Millisecond, Process: process,
	})
	if serviceErr != nil {
		t.Fatalf("NewService: %v", serviceErr)
	}
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err != nil {
		t.Fatalf("StartRun: %v", err)
	}
	<-process.started
	failed := waitForRunStatus(t, service, run.ID, StatusFailed)
	if !strings.Contains(failed.Error, "time limit") {
		t.Fatalf("timed-out run did not retain a safe timeout reason: %+v", failed)
	}

	second, secondErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if secondErr != nil {
		t.Fatalf("create second: %v", secondErr)
	}
	startRunAfterCapacityRelease(t, service, second.ID)
	if err := service.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func startRunAfterCapacityRelease(t *testing.T, service *Service, runID string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		_, err := service.StartRunAs(context.Background(), SystemActor(), runID)
		if err == nil {
			return
		}
		if !errors.Is(err, ErrConflict) || time.Now().After(deadline) {
			t.Fatalf("start run after capacity release: %v", err)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

type delayedCancelProcess struct {
	started   chan struct{}
	cancelled chan struct{}
	exit      chan struct{}
	once      sync.Once
}

func (p *delayedCancelProcess) Run(ctx context.Context, _ ProcessSpec, _ func(WorkerEvent) error) (ProcessResult, error) {
	close(p.started)
	<-ctx.Done()
	p.once.Do(func() { close(p.cancelled) })
	<-p.exit
	return ProcessResult{}, ctx.Err()
}

func TestCancelKeepsRunUndeletableUntilWorkerExit(t *testing.T) {
	process := &delayedCancelProcess{started: make(chan struct{}), cancelled: make(chan struct{}), exit: make(chan struct{})}
	service, root := newTestService(t, process, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err != nil {
		t.Fatalf("StartRun: %v", err)
	}
	<-process.started
	if _, err := service.CancelRunAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("CancelRun: %v", err)
	}
	<-process.cancelled
	if err := service.DeleteRunAs(SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("DeleteRun while worker exits error=%v, want ErrConflict", err)
	}
	close(process.exit)
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	for {
		service.mu.Lock()
		_, active := service.active[run.ID]
		service.mu.Unlock()
		if !active {
			break
		}
		select {
		case <-deadline.C:
			t.Fatal("worker did not leave active set")
		case <-time.After(5 * time.Millisecond):
		}
	}
	if err := service.DeleteRunAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("DeleteRun after worker exit: %v", err)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	if _, err := os.Stat(runDir); !os.IsNotExist(err) {
		t.Fatalf("run directory resurrected after delete: err=%v", err)
	}
}

func TestCommandProcessShapeCannotAcceptArbitraryCommands(t *testing.T) {
	typeOfSpec := reflect.TypeOf(ProcessSpec{})
	if typeOfSpec.NumField() != 6 || typeOfSpec.Field(0).Name != "ManifestPath" ||
		typeOfSpec.Field(1).Name != "StorePath" || typeOfSpec.Field(2).Name != "SuiteStorePath" ||
		typeOfSpec.Field(3).Name != "executionContracts" || typeOfSpec.Field(3).IsExported() ||
		typeOfSpec.Field(4).Name != "controlledPair" || typeOfSpec.Field(4).IsExported() ||
		typeOfSpec.Field(5).Name != "evidencePublication" || typeOfSpec.Field(5).IsExported() {
		t.Fatalf("ProcessSpec grew an arbitrary execution surface: %v", typeOfSpec)
	}
	if workerSandboxScript != "cli/evaluation/sandbox_worker.py" {
		t.Fatalf("worker sandbox script=%q", workerSandboxScript)
	}
}

func TestManifestAndControlEventFilesRejectSymlinkReplacement(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	outside := filepath.Join(root, "outside-control")
	if err := os.WriteFile(outside, []byte("unchanged\n"), 0o600); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	eventsPath := filepath.Join(runDir, eventsFileName)
	if err := os.Remove(eventsPath); err != nil {
		t.Fatalf("remove control events: %v", err)
	}
	if err := os.Symlink(outside, eventsPath); err != nil {
		t.Fatalf("symlink control events: %v", err)
	}
	if _, err := service.store.AppendEvent(Event{RunID: run.ID, Type: "progress", Timestamp: time.Now(), Message: "unsafe"}); err == nil {
		t.Fatal("symlink control event log accepted an append")
	}
	contents, readErr := os.ReadFile(outside)
	if readErr != nil || string(contents) != "unchanged\n" {
		t.Fatalf("outside control file changed: %q err=%v", contents, readErr)
	}

	manifestPath := filepath.Join(runDir, manifestFileName)
	if err := os.Remove(manifestPath); err != nil {
		t.Fatalf("remove manifest: %v", err)
	}
	if err := os.Symlink(outside, manifestPath); err != nil {
		t.Fatalf("symlink manifest: %v", err)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err == nil {
		t.Fatal("symlink manifest was accepted for execution")
	}
}

func TestStoreRejectsSharedDirectoryPermissions(t *testing.T) {
	root := t.TempDir()
	if err := os.Chmod(root, 0o750); err != nil {
		t.Fatalf("chmod store root: %v", err)
	}
	if _, err := newStandaloneStore(root); err == nil {
		t.Fatal("group-accessible evaluation store was accepted")
	}

	nestedRoot := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(nestedRoot, 0o700); err != nil {
		t.Fatalf("create nested store root: %v", err)
	}
	if err := os.Mkdir(filepath.Join(nestedRoot, "objects"), 0o750); err != nil {
		t.Fatalf("create shared objects directory: %v", err)
	}
	if _, err := newStandaloneStore(nestedRoot); err == nil {
		t.Fatal("preexisting group-accessible canonical directory was accepted")
	}
}

func TestStoreCreatesCanonicalPrivateDirectoryLayout(t *testing.T) {
	root := filepath.Join(t.TempDir(), "evaluation")
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	for _, directory := range []string{
		root,
		filepath.Join(root, "objects"),
		filepath.Join(root, "objects", "sha256"),
		filepath.Join(root, "runs"),
	} {
		info, err := os.Stat(directory)
		if err != nil {
			t.Fatalf("stat %s: %v", directory, err)
		}
		if info.Mode().Perm() != 0o700 {
			t.Fatalf("directory %s mode=%04o, want 0700", directory, info.Mode().Perm())
		}
	}
}
