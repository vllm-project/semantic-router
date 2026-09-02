package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type controlledProcess struct {
	started     chan ProcessSpec
	release     chan struct{}
	returned    chan struct{}
	writeReport bool
	err         error
	calls       atomic.Int32
}

type failOnceTerminalStatusPersistence struct {
	delegate            runStatusPersistence
	commitBeforeFailure bool
	failed              atomic.Bool
}

type alwaysFailTerminalStatusPersistence struct {
	delegate  runStatusPersistence
	attempted chan struct{}
	signaled  atomic.Bool
}

func (p *failOnceTerminalStatusPersistence) Write(path string, run Run) error {
	if terminalStatus(run.Status) && p.failed.CompareAndSwap(false, true) {
		if p.commitBeforeFailure {
			if err := p.delegate.Write(path, run); err != nil {
				return err
			}
		}
		return errors.New("injected terminal status persistence failure")
	}
	return p.delegate.Write(path, run)
}

func (p *failOnceTerminalStatusPersistence) SyncDirectory(path, description string) error {
	return p.delegate.SyncDirectory(path, description)
}

func (p *alwaysFailTerminalStatusPersistence) Write(path string, run Run) error {
	if terminalStatus(run.Status) {
		if p.signaled.CompareAndSwap(false, true) {
			close(p.attempted)
		}
		return errors.New("injected permanent terminal status persistence failure")
	}
	return p.delegate.Write(path, run)
}

func (p *alwaysFailTerminalStatusPersistence) SyncDirectory(path, description string) error {
	return p.delegate.SyncDirectory(path, description)
}

const testSourceRevision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func (p *controlledProcess) Run(ctx context.Context, spec ProcessSpec, emit func(WorkerEvent) error) (ProcessResult, error) {
	p.calls.Add(1)
	if p.started != nil {
		p.started <- spec
	}
	if err := emit(WorkerEvent{
		Type: "progress", Message: "fixture running",
		Progress: &RunProgress{Percent: 50, Completed: 0, Total: 1, CurrentTrackID: "routing"},
	}); err != nil {
		return ProcessResult{}, err
	}
	if p.release != nil {
		select {
		case <-ctx.Done():
			return ProcessResult{}, ctx.Err()
		case <-p.release:
		}
	} else {
		<-ctx.Done()
		return ProcessResult{}, ctx.Err()
	}
	if p.err != nil {
		return ProcessResult{}, p.err
	}
	if p.writeReport {
		result := ProcessResult{publishEvidence: func() error {
			return spec.evidencePublication(func() error { return writeProcessReport(spec) })
		}}
		if p.returned != nil {
			close(p.returned)
		}
		return result, nil
	}
	return ProcessResult{}, nil
}

func TestControlledProcessReportBundlePassesServerValidation(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	run = stageSealingTestRun(t, service, run)
	spec := ProcessSpec{ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName), StorePath: root}
	if err := writeProcessReport(spec); err != nil {
		t.Fatalf("writeProcessReport: %v", err)
	}
	if err := service.store.withEvidencePublication(func() error {
		return service.validateAndAnchorReportDuringPublication(run.ID)
	}); err != nil {
		t.Fatalf("validateAndAnchorReport: %v", err)
	}
	sealedBytes, readErr := service.store.ReadReport(run.ID)
	if readErr != nil {
		t.Fatalf("read sealed report: %v", readErr)
	}
	sealed, decodeErr := decodeReportStrict(run.ID, sealedBytes)
	if decodeErr != nil {
		t.Fatalf("decode sealed report: %v", decodeErr)
	}
	if sealed.Run.Name != run.Name || sealed.Run.Description != run.Description || sealed.Run.StartedAt == nil || sealed.Run.CompletedAt == nil {
		t.Fatalf("sealed report run identity is not server-owned: %+v", sealed.Run)
	}
	if sealed.AttestationRevision != ServerAttestationRevision {
		t.Fatalf("sealed report attestation_revision=%q, want %q", sealed.AttestationRevision, ServerAttestationRevision)
	}
	anchor, anchorErr := service.store.readReportAnchor(run.ID)
	if anchorErr != nil {
		t.Fatalf("read report anchor: %v", anchorErr)
	}
	if anchor.AttestationRevision != ServerAttestationRevision {
		t.Fatalf("sealed anchor attestation_revision=%q, want %q", anchor.AttestationRevision, ServerAttestationRevision)
	}
}

func TestRecoverInterruptedRunCompletesFullySealedPublication(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	run = stageSealingTestRun(t, service, run)
	spec := ProcessSpec{ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName), StorePath: root}
	if writeErr := writeProcessReport(spec); writeErr != nil {
		t.Fatalf("writeProcessReport: %v", writeErr)
	}
	if validationErr := service.store.withEvidencePublication(func() error {
		return service.validateAndAnchorReportDuringPublication(run.ID)
	}); validationErr != nil {
		t.Fatalf("validateAndAnchorReport: %v", validationErr)
	}
	if staged, readErr := service.GetRunAs(SystemActor(), run.ID); readErr != nil || staged.Status != StatusSealing || staged.CompletedAt != nil {
		t.Fatalf("expected anchor-after/status-before crash window, run=%+v err=%v", staged, readErr)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before sealed publication recovery: %v", err)
	}

	restarted, restartErr := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: filepath.Join(root, "config.yaml"),
		RouterAPIURL: "http://router.invalid", EnvoyURL: "http://envoy.invalid", CodeRevision: testSourceRevision,
	})
	if restartErr != nil {
		t.Fatalf("restart NewService: %v", restartErr)
	}
	t.Cleanup(func() { _ = restarted.Close() })
	recovered, recoveryErr := restarted.GetRunAs(SystemActor(), run.ID)
	if recoveryErr != nil || recovered.Status != StatusCompleted || recovered.CompletedAt == nil || recovered.Error != "" {
		t.Fatalf("sealed run was not recovered as completed: run=%+v err=%v", recovered, recoveryErr)
	}
	if _, reportErr := restarted.ReportJSONAs(SystemActor(), run.ID); reportErr != nil {
		t.Fatalf("ReportJSON after sealed recovery: %v", reportErr)
	}
	if recoverErr := restarted.RecoverInterruptedRuns(); recoverErr != nil {
		t.Fatalf("repeat RecoverInterruptedRuns: %v", recoverErr)
	}
	events, eventsErr := restarted.EventsAfterAs(SystemActor(), run.ID, "")
	if eventsErr != nil {
		t.Fatalf("EventsAfter: %v", eventsErr)
	}
	terminal := 0
	for _, event := range events {
		if terminalWorkerEventType(event.Type) {
			terminal++
			if event.Type != "completed" {
				t.Fatalf("terminal event=%+v, want completed", event)
			}
		}
	}
	if terminal != 1 {
		t.Fatalf("terminal events=%d, want exactly one: %+v", terminal, events)
	}
}

func TestValidateAndAnchorReportRejectsNonCanonicalPublicArtifactMetadata(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Artifact)
	}{
		{name: "kind", mutate: func(artifact *Artifact) { artifact.Kind = "worker-defined" }},
		{name: "media type", mutate: func(artifact *Artifact) { artifact.MediaType = "text/html" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
			if err != nil {
				t.Fatalf("CreateRun: %v", err)
			}
			run = stageSealingTestRun(t, service, run)
			spec := ProcessSpec{ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName), StorePath: root}
			if err := writeProcessReport(spec); err != nil {
				t.Fatalf("writeProcessReport: %v", err)
			}
			reportBytes, readErr := service.store.ReadReport(run.ID)
			if readErr != nil {
				t.Fatalf("read report: %v", readErr)
			}
			report, decodeErr := decodeWorkerReportStrict(run.ID, reportBytes)
			if decodeErr != nil {
				t.Fatalf("decode report: %v", decodeErr)
			}
			if len(report.Artifacts) == 0 {
				t.Fatal("fixture report has no public artifacts")
			}
			test.mutate(&report.Artifacts[0])
			if err := service.store.WriteReport(run.ID, workerReportFromReport(report)); err != nil {
				t.Fatalf("rewrite report: %v", err)
			}
			if err := service.store.withEvidencePublication(func() error {
				return service.validateAndAnchorReportDuringPublication(run.ID)
			}); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "artifact metadata") {
				t.Fatalf("validateAndAnchorReport error=%v, want artifact metadata ErrInvalid", err)
			}
		})
	}
}

func TestQualifiedEvidenceRequiresImmutableCodeRevision(t *testing.T) {
	valid := []string{
		testSourceRevision,
		"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
	}
	for _, revision := range valid {
		for _, level := range []EvidenceLevel{"E0", "E1", "E2", "E3", "E4", "E5"} {
			if err := requireQualifiedCodeRevision(level, revision); err != nil {
				t.Fatalf("level %s immutable revision %q rejected: %v", level, revision, err)
			}
		}
	}
	invalid := []string{"", "unavailable", "main", "latest", "abc1234", "commit-abc123", testSourceRevision + "-dirty"}
	for _, revision := range invalid {
		for _, level := range []EvidenceLevel{"E0", "E5"} {
			if err := requireQualifiedCodeRevision(level, revision); !errors.Is(err, ErrInvalid) {
				t.Fatalf("level %s mutable revision %q error=%v, want ErrInvalid", level, revision, err)
			}
		}
	}
}

func TestCreateRunFailsClosedWithoutImmutableSourceRevision(t *testing.T) {
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create root: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if _, err := NewService(Options{
		DataDir: root, ConfigPath: configPath, CodeRevision: "main", Process: &controlledProcess{},
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("NewService mutable revision error=%v, want ErrInvalid", err)
	}
}

func TestPendingRunCannotCrossSourceRevisionUpgrade(t *testing.T) {
	serviceA, root := newTestService(t, &controlledProcess{}, 1)
	run, err := serviceA.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	serviceB, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		CodeRevision: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if err != nil {
		t.Fatalf("NewService revision B: %v", err)
	}
	if _, startErr := serviceB.StartRunAs(context.Background(), SystemActor(), run.ID); !errors.Is(startErr, ErrConflict) || !strings.Contains(startErr.Error(), "source revision") {
		t.Fatalf("StartRun across source upgrade error=%v, want revision ErrConflict", startErr)
	}
	persisted, err := serviceB.GetRunAs(SystemActor(), run.ID)
	if err != nil || persisted.Status != StatusPending || persisted.StartedAt != nil {
		t.Fatalf("rejected upgraded run mutated durable state: run=%+v err=%v", persisted, err)
	}
}

func TestStartRejectsTamperedServerManifestDigest(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	manifestPath := filepath.Join(root, "runs", run.ID, manifestFileName)
	var manifest RunManifest
	if readErr := readJSON(manifestPath, &manifest); readErr != nil {
		t.Fatalf("read manifest: %v", readErr)
	}
	manifest.SampleLimit++
	if writeErr := writeJSONAtomic(manifestPath, manifest); writeErr != nil {
		t.Fatalf("write tampered manifest: %v", writeErr)
	}
	if _, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID); !errors.Is(startErr, ErrInvalid) || !strings.Contains(startErr.Error(), "manifest_digest") {
		t.Fatalf("StartRun tampered manifest error=%v, want manifest digest ErrInvalid", startErr)
	}
	persisted, err := service.GetRunAs(SystemActor(), run.ID)
	if err != nil || persisted.Status != StatusPending || persisted.StartedAt != nil {
		t.Fatalf("rejected manifest mutated run: %+v err=%v", persisted, err)
	}
}

func TestStartAndIdempotentRetryRejectSelfConsistentManifestStatusDrift(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), request)
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	manifestPath := filepath.Join(root, "runs", run.ID, manifestFileName)
	var manifest RunManifest
	if readErr := readJSON(manifestPath, &manifest); readErr != nil {
		t.Fatalf("read manifest: %v", readErr)
	}
	manifest.SampleLimit++
	manifestDigest, digestErr := manifestSemanticDigest(manifest)
	if digestErr != nil {
		t.Fatalf("recompute manifest digest: %v", digestErr)
	}
	manifest.ManifestDigest = manifestDigest
	if writeErr := writeJSONAtomic(manifestPath, manifest); writeErr != nil {
		t.Fatalf("write self-consistent drifted manifest: %v", writeErr)
	}
	if _, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID); !errors.Is(startErr, ErrInvalid) || !strings.Contains(startErr.Error(), "sample_limit") {
		t.Fatalf("StartRun drift error=%v, want sample_limit ErrInvalid", startErr)
	}
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), request); !errors.Is(err, ErrConflict) {
		t.Fatalf("idempotent retry on drifted bundle error=%v, want ErrConflict", err)
	}
}

func TestPendingRunRejectsSuiteRevisionDriftAndNonCurrentGateContract(t *testing.T) {
	for _, mutate := range []struct {
		name      string
		fn        func(*RunManifest)
		wantError error
		match     string
	}{
		{name: "suite revision", fn: func(manifest *RunManifest) { manifest.SuiteRevisions["evaluation-smoke"] = "builtin-v2" }, wantError: ErrConflict, match: "contract revision"},
		{name: "suite executor", fn: func(manifest *RunManifest) { manifest.SuiteExecutors["evaluation-smoke"] = "retired-executor.v0" }, wantError: ErrInvalid, match: "executor"},
		{name: "gate contract", fn: func(manifest *RunManifest) { manifest.GateContractVersion = "evaluation-release-gates.v1" }, wantError: ErrInvalid, match: "gate contract"},
	} {
		t.Run(mutate.name, func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
			if err != nil {
				t.Fatalf("CreateRun: %v", err)
			}
			manifestPath := filepath.Join(root, "runs", run.ID, manifestFileName)
			var manifest RunManifest
			if readErr := readJSON(manifestPath, &manifest); readErr != nil {
				t.Fatalf("read manifest: %v", readErr)
			}
			mutate.fn(&manifest)
			manifest.ManifestDigest, err = manifestSemanticDigest(manifest)
			if err != nil {
				t.Fatalf("refresh drifted manifest digest: %v", err)
			}
			if writeErr := writeJSONAtomic(manifestPath, manifest); writeErr != nil {
				t.Fatalf("write drifted manifest: %v", writeErr)
			}
			if _, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID); !errors.Is(startErr, mutate.wantError) || !strings.Contains(startErr.Error(), mutate.match) {
				t.Fatalf("StartRun drift error=%v, want %v containing %q", startErr, mutate.wantError, mutate.match)
			}
			persisted, err := service.GetRunAs(SystemActor(), run.ID)
			if err != nil || persisted.Status != StatusPending || persisted.StartedAt != nil {
				t.Fatalf("rejected drift mutated run: %+v err=%v", persisted, err)
			}
		})
	}
}

func TestReportBenchmarkRevisionsMustMatchDurableManifest(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	manifest, _, err := service.readDurableManifest(run.ID)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	report := reportForRun(run, nil)
	report.Provenance.BenchmarkRevisions = map[string]string{"evaluation-smoke": "nonempty-but-wrong"}
	if err := validateReportFrozenFields(run, manifest, report); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "benchmark") {
		t.Fatalf("forged benchmark revision error=%v, want ErrInvalid", err)
	}
}

func waitForRunStatus(t *testing.T, service *Service, runID string, want RunStatus) Run {
	t.Helper()
	deadline := time.NewTimer(3 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-deadline.C:
			run, _ := service.GetRunAs(SystemActor(), runID)
			t.Fatalf("timed out waiting for status %s; last run=%+v", want, run)
		case <-ticker.C:
			run, err := service.GetRunAs(SystemActor(), runID)
			if err == nil && run.Status == want {
				return run
			}
		}
	}
}

func waitForWorkerExit(t *testing.T, service *Service, runID string) {
	t.Helper()
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	for {
		service.mu.Lock()
		_, active := service.active[runID]
		service.mu.Unlock()
		if !active {
			return
		}
		select {
		case <-deadline.C:
			t.Fatal("evaluation worker did not exit")
		case <-ticker.C:
		}
	}
}

func TestRunLifecycleStartIsIdempotentAndBundleBacked(t *testing.T) {
	process := &controlledProcess{started: make(chan ProcessSpec, 2), release: make(chan struct{}), writeReport: true}
	service, root := newTestService(t, process, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	for _, name := range []string{runFileName, manifestFileName, eventsFileName} {
		if _, err := os.Stat(filepath.Join(root, "runs", run.ID, name)); err != nil {
			t.Fatalf("expected canonical bundle file %s: %v", name, err)
		}
	}
	started, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID)
	if startErr != nil || started.Status != StatusRunning {
		t.Fatalf("StartRun = %+v, %v", started, startErr)
	}
	second, secondStartErr := service.StartRunAs(context.Background(), SystemActor(), run.ID)
	if secondStartErr != nil || second.Status != StatusRunning {
		t.Fatalf("second StartRun = %+v, %v", second, secondStartErr)
	}
	select {
	case <-process.started:
	case <-time.After(time.Second):
		t.Fatal("worker did not start")
	}
	if got := process.calls.Load(); got != 1 {
		t.Fatalf("idempotent start launched %d workers", got)
	}
	close(process.release)
	completed := waitForRunStatus(t, service, run.ID, StatusCompleted)
	if completed.Progress.Percent != 100 || completed.CompletedAt == nil {
		t.Fatalf("unexpected completed run: %+v", completed)
	}
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("ReportJSON: %v", err)
	}
	events, eventsErr := service.EventsAfterAs(SystemActor(), run.ID, "")
	if eventsErr != nil {
		t.Fatalf("EventsAfter: %v", eventsErr)
	}
	if len(events) < 4 || events[0].Type != "snapshot" || events[len(events)-1].Type != "completed" {
		t.Fatalf("unexpected durable events: %+v", events)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err != nil || process.calls.Load() != 1 {
		t.Fatalf("completed start should be idempotent: calls=%d err=%v", process.calls.Load(), err)
	}
	if err := service.DeleteRunAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("DeleteRun: %v", err)
	}
	if _, err := service.GetRunAs(SystemActor(), run.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetRun after delete error=%v, want ErrNotFound", err)
	}
}

func TestCancelWinsBeforeSealingPublishesNoReportOrAnchor(t *testing.T) {
	process := &controlledProcess{
		started:     make(chan ProcessSpec, 1),
		release:     make(chan struct{}),
		returned:    make(chan struct{}),
		writeReport: true,
	}
	service, root := newTestService(t, process, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	peer, peerErr := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: filepath.Join(root, "config.yaml"),
		RouterAPIURL: "http://router.invalid", EnvoyURL: "http://envoy.invalid",
		CodeRevision: testSourceRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if peerErr != nil {
		t.Fatalf("create peer evaluation service: %v", peerErr)
	}
	defer func() { _ = peer.Close() }()
	if _, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID); startErr != nil {
		t.Fatalf("StartRun: %v", startErr)
	}
	<-process.started
	deadline := time.Now().Add(2 * time.Second)
	for {
		running, readErr := service.GetRunAs(SystemActor(), run.ID)
		if readErr == nil && running.Progress.Percent == 50 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("worker did not reach the publication barrier: run=%+v err=%v", running, readErr)
		}
		time.Sleep(5 * time.Millisecond)
	}
	service.mu.Lock()
	serviceLocked := true
	defer func() {
		if serviceLocked {
			service.mu.Unlock()
		}
	}()
	close(process.release)
	<-process.returned
	cancelled, err := peer.CancelRunAs(SystemActor(), run.ID)
	if err != nil || cancelled.Status != StatusCancelled {
		t.Fatalf("CancelRun: run=%+v err=%v", cancelled, err)
	}
	service.mu.Unlock()
	serviceLocked = false
	waitForWorkerExit(t, service, run.ID)

	runDir := filepath.Join(root, "runs", run.ID)
	for _, name := range []string{reportFileName, reportAnchorFileName} {
		if _, statErr := os.Lstat(filepath.Join(runDir, name)); !os.IsNotExist(statErr) {
			t.Fatalf("cancelled run published %s: %v", name, statErr)
		}
	}
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("ReportJSON for cancelled run error=%v, want ErrConflict", err)
	}
}

func TestSealingWinsBeforeCancelAndCompletesPublication(t *testing.T) {
	process := &controlledProcess{
		started: make(chan ProcessSpec, 1), release: make(chan struct{}), writeReport: true,
	}
	service, root := newTestService(t, process, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err != nil {
		t.Fatalf("StartRun: %v", err)
	}
	<-process.started

	service.store.lifecycle.evidenceMu.Lock()
	publicationLocked := true
	defer func() {
		if publicationLocked {
			service.store.lifecycle.evidenceMu.Unlock()
		}
		_ = service.Close()
	}()
	close(process.release)
	sealing := waitForRunStatus(t, service, run.ID, StatusSealing)
	if sealing.CompletedAt != nil || sealing.Error != "" {
		t.Fatalf("sealing state is not active: %+v", sealing)
	}
	if _, err := service.CancelRunAs(SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("CancelRun during sealing error=%v, want ErrConflict", err)
	}
	if current, err := service.GetRunAs(SystemActor(), run.ID); err != nil || current.Status != StatusSealing {
		t.Fatalf("cancel changed sealing run: run=%+v err=%v", current, err)
	}
	for _, name := range []string{reportFileName, reportAnchorFileName} {
		if _, statErr := os.Lstat(filepath.Join(root, "runs", run.ID, name)); !os.IsNotExist(statErr) {
			t.Fatalf("evidence %s was published before the sealing barrier opened: %v", name, statErr)
		}
	}
	service.store.lifecycle.evidenceMu.Unlock()
	publicationLocked = false

	completed := waitForRunStatus(t, service, run.ID, StatusCompleted)
	if completed.CompletedAt == nil {
		t.Fatalf("sealing run did not complete: %+v", completed)
	}
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("ReportJSON after sealing completion: %v", err)
	}
	if _, err := service.store.readReportAnchor(run.ID); err != nil {
		t.Fatalf("read completed report anchor: %v", err)
	}
}

func TestRunCancellationAndRestartRecovery(t *testing.T) {
	process := &controlledProcess{started: make(chan ProcessSpec, 1)}
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
	waitForRunStatus(t, service, run.ID, StatusCancelled)

	interrupted, interruptedErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if interruptedErr != nil {
		t.Fatalf("CreateRun interrupted: %v", interruptedErr)
	}
	interrupted = stageRunningTestRun(t, service, interrupted)
	if err := service.Close(); err != nil {
		t.Fatalf("close before interrupted run recovery: %v", err)
	}
	restarted, restartErr := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: filepath.Join(root, "config.yaml"),
		CodeRevision: testSourceRevision,
	})
	if restartErr != nil {
		t.Fatalf("restart NewService: %v", restartErr)
	}
	t.Cleanup(func() { _ = restarted.Close() })
	recovered := waitForRunStatus(t, restarted, interrupted.ID, StatusFailed)
	if recovered.CompletedAt == nil || recovered.Error == "" {
		t.Fatalf("recovery did not persist failure evidence: %+v", recovered)
	}
	events, eventsErr := restarted.EventsAfterAs(SystemActor(), interrupted.ID, "1")
	if eventsErr != nil || len(events) != 1 || events[0].Type != "failed" {
		t.Fatalf("recovery events=%+v err=%v", events, eventsErr)
	}
}

func TestFinalizeRetriesTransientTerminalStatusFailureInProcess(t *testing.T) {
	process := &controlledProcess{started: make(chan ProcessSpec, 1), release: make(chan struct{}), writeReport: true}
	service, _ := newTestService(t, process, 1)
	persistence := &failOnceTerminalStatusPersistence{
		delegate: service.store.statusPersistence, commitBeforeFailure: true,
	}
	service.store.statusPersistence = persistence
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	if _, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID); startErr != nil {
		t.Fatalf("StartRun: %v", startErr)
	}
	<-process.started
	close(process.release)
	completed := waitForRunStatus(t, service, run.ID, StatusCompleted)
	if !persistence.failed.Load() || completed.CompletedAt == nil {
		t.Fatalf("terminal retry was not exercised: run=%+v failed=%v", completed, persistence.failed.Load())
	}
	events, err := service.EventsAfterAs(SystemActor(), run.ID, "")
	if err != nil {
		t.Fatalf("EventsAfter: %v", err)
	}
	assertSingleStableTerminalEvent(t, service, run.ID, events, "completed")
}

func TestCancelRetryUsesDerivedStableTerminalEvent(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	persistence := &failOnceTerminalStatusPersistence{delegate: service.store.statusPersistence}
	service.store.statusPersistence = persistence
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	run = stageRunningTestRun(t, service, run)
	if _, appendErr := service.store.AppendEvent(Event{
		RunID: run.ID, Type: "cancelled", Timestamp: time.Now().UTC(), Message: "must not persist",
	}); !errors.Is(appendErr, ErrInvalid) {
		t.Fatalf("terminal control-event append error=%v, want ErrInvalid", appendErr)
	}
	if _, cancelErr := service.CancelRunAs(SystemActor(), run.ID); cancelErr == nil || !strings.Contains(cancelErr.Error(), "injected terminal") {
		t.Fatalf("first CancelRun error=%v, want injected persistence failure", cancelErr)
	}
	if running, readErr := service.GetRunAs(SystemActor(), run.ID); readErr != nil || running.Status != StatusRunning {
		t.Fatalf("failed cancel changed durable state: run=%+v err=%v", running, readErr)
	}
	cancelled, cancelErr := service.CancelRunAs(SystemActor(), run.ID)
	if cancelErr != nil || cancelled.Status != StatusCancelled {
		t.Fatalf("retry CancelRun: run=%+v err=%v", cancelled, cancelErr)
	}
	events, eventsErr := service.EventsAfterAs(SystemActor(), run.ID, "")
	if eventsErr != nil {
		t.Fatalf("EventsAfter: %v", eventsErr)
	}
	assertSingleStableTerminalEvent(t, service, run.ID, events, "cancelled")
}

func TestCloseStopsTerminalPersistenceRetryWithoutLeakingWorker(t *testing.T) {
	process := &controlledProcess{started: make(chan ProcessSpec, 1), release: make(chan struct{}), writeReport: true}
	service, _ := newTestService(t, process, 1)
	persistence := &alwaysFailTerminalStatusPersistence{
		delegate: service.store.statusPersistence, attempted: make(chan struct{}),
	}
	service.store.statusPersistence = persistence
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	if _, err := service.StartRunAs(context.Background(), SystemActor(), run.ID); err != nil {
		t.Fatalf("StartRun: %v", err)
	}
	<-process.started
	close(process.release)
	select {
	case <-persistence.attempted:
	case <-time.After(time.Second):
		t.Fatal("terminal persistence was not attempted")
	}

	closed := make(chan error, 1)
	go func() { closed <- service.Close() }()
	select {
	case closeErr := <-closed:
		if closeErr == nil || !strings.Contains(closeErr.Error(), "injected permanent terminal status persistence failure") {
			t.Fatalf("Close error=%v, want terminal persistence failure", closeErr)
		}
	case <-time.After(time.Second):
		t.Fatal("Close leaked a worker blocked in terminal persistence retry")
	}
	service.mu.Lock()
	_, active := service.active[run.ID]
	service.mu.Unlock()
	if active {
		t.Fatal("terminal persistence retry left an active worker")
	}
}

func assertSingleStableTerminalEvent(t *testing.T, service *Service, runID string, events []Event, eventType string) {
	t.Helper()
	terminal := make([]Event, 0, 1)
	for _, event := range events {
		if terminalWorkerEventType(event.Type) {
			terminal = append(terminal, event)
		}
	}
	if len(terminal) != 1 || terminal[0].Type != eventType || terminal[0].ID != events[len(events)-1].ID {
		t.Fatalf("terminal events=%+v, want one stable %s event at tail", terminal, eventType)
	}
	replay, err := service.EventsAfterAs(SystemActor(), runID, terminal[0].ID)
	if err != nil || len(replay) != 0 {
		t.Fatalf("terminal replay after id %s = %+v, err=%v", terminal[0].ID, replay, err)
	}
	again, err := service.EventsAfterAs(SystemActor(), runID, "")
	if err != nil || len(again) != len(events) || again[len(again)-1].ID != terminal[0].ID {
		t.Fatalf("repeated replay is unstable: events=%+v err=%v", again, err)
	}
}
