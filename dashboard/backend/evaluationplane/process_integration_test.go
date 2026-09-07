package evaluationplane

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestCommandProcessFixtureEndToEnd(t *testing.T) {
	python := os.Getenv("VLLM_SR_EVALUATION_TEST_PYTHON")
	if python == "" {
		t.Skip("set VLLM_SR_EVALUATION_TEST_PYTHON to run the real Python worker")
	}
	pythonRoot, err := filepath.Abs("../../../src/vllm-sr")
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("PYTHONPATH", pythonRoot)
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create evaluation store: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	service, serviceErr := NewService(Options{
		DataDir: root, PythonPath: python, ConfigPath: configPath,
		CodeRevision: testSourceRevision, MaxConcurrent: 1,
	})
	if serviceErr != nil {
		t.Fatalf("NewService: %v", serviceErr)
	}
	var workerDiagnostics bytes.Buffer
	service.process.(*CommandProcess).diagnosticSink = &workerDiagnostics
	processCapture := &capturedProcess{Process: service.process}
	service.process = processCapture
	t.Cleanup(func() {
		if err := service.Close(); err != nil {
			t.Errorf("close evaluation service: %v", err)
		}
	})
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), CreateRunRequest{
		ClientRequestID: newTestClientRequestID(),
		Name:            "real fixture worker", SuiteIDs: []string{"evaluation-smoke"},
		TrackIDs: append([]TrackID(nil), allTrackIDs...), Mode: ModeReplay, TargetID: "fixture",
		ChangeProfile: "schema_adapter", SampleLimit: 4, Concurrency: 2, Seed: 17,
	})
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	manifestPath := filepath.Join(root, "runs", run.ID, manifestFileName)
	manifestBefore, beforeErr := os.ReadFile(manifestPath)
	if beforeErr != nil {
		t.Fatalf("read staged manifest: %v", beforeErr)
	}
	// Exercise the production lifecycle. In particular, do not make the
	// server-owned StartedAt equal CreatedAt as that hides worker/report clock
	// authority bugs.
	time.Sleep(time.Millisecond)
	started, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID)
	if startErr != nil || started.Status != StatusRunning || started.StartedAt == nil || !started.StartedAt.After(run.CreatedAt) {
		t.Fatalf("StartRun=%+v err=%v", started, startErr)
	}
	waitForCompletedFixtureRun(t, service, run.ID, processCapture, &workerDiagnostics)
	manifestAfter, afterErr := os.ReadFile(manifestPath)
	if afterErr != nil {
		t.Fatalf("read completed manifest: %v", afterErr)
	}
	if !bytes.Equal(manifestBefore, manifestAfter) {
		t.Fatal("Python worker rewrote the server-owned run manifest")
	}
	reportBytes, reportErr := service.ReportJSONAs(SystemActor(), run.ID)
	if reportErr != nil {
		t.Fatalf("strict report validation: %v", reportErr)
	}
	report, decodeErr := decodeReportStrict(run.ID, reportBytes)
	if decodeErr != nil {
		t.Fatalf("decode server-sealed report: %v", decodeErr)
	}
	if report.AttestationRevision != ServerAttestationRevision {
		t.Fatalf("server-sealed report attestation_revision=%q, want %q", report.AttestationRevision, ServerAttestationRevision)
	}
	anchor, anchorErr := service.store.readReportAnchor(run.ID)
	if anchorErr != nil {
		t.Fatalf("read server-owned report anchor: %v", anchorErr)
	}
	if anchor.AttestationRevision != ServerAttestationRevision {
		t.Fatalf("server-owned anchor attestation_revision=%q, want %q", anchor.AttestationRevision, ServerAttestationRevision)
	}
	for _, name := range []string{eventsFileName, "records.jsonl", reportFileName} {
		if _, err := os.Stat(filepath.Join(root, "runs", run.ID, name)); err != nil {
			t.Fatalf("expected end-to-end bundle file %s: %v", name, err)
		}
	}
}

func waitForCompletedFixtureRun(
	t *testing.T,
	service *Service,
	runID string,
	processCapture *capturedProcess,
	diagnostics *bytes.Buffer,
) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for {
		completed, err := service.GetRunAs(SystemActor(), runID)
		if err == nil && terminalStatus(completed.Status) {
			if completed.Status != StatusCompleted {
				t.Fatalf("real worker did not complete: %+v process_error=%v diagnostics=%s", completed, processCapture.Err(), diagnostics.String())
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for real worker: run=%+v err=%v", completed, err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

type capturedProcess struct {
	Process
	mu  sync.Mutex
	err error
}

func (process *capturedProcess) Run(ctx context.Context, spec ProcessSpec, emit func(WorkerEvent) error) (ProcessResult, error) {
	result, err := process.Process.Run(ctx, spec, emit)
	process.mu.Lock()
	process.err = err
	process.mu.Unlock()
	return result, err
}

func (process *capturedProcess) Err() error {
	process.mu.Lock()
	defer process.mu.Unlock()
	return process.err
}

func TestSealRejectsWorkerClaimedAttestationRevision(t *testing.T) {
	assertSafetyBundleSealRejected(t, safetyReportOptions{
		mutateReport: func(report *Report) {
			report.AttestationRevision = ServerAttestationRevision
		},
	}, "unknown field \"attestation_revision\"")
}

func TestSealRejectsSelfConsistentForgedGateMetricBundleAgainstRecords(t *testing.T) {
	assertSafetyBundleSealRejected(t, safetyReportOptions{
		mutateMetrics: func(metrics []Metric) { metrics[0].Value = floatPointer(1) },
	}, "safety.violation_rate value does not match records")
}

func TestSealRejectsForgedMetricIntervalAndSingleRunComparison(t *testing.T) {
	t.Run("confidence interval", func(t *testing.T) {
		assertSafetyBundleSealRejected(t, safetyReportOptions{
			mutateMetrics: func(metrics []Metric) { metrics[1].ConfidenceInterval = []float64{0.99, 1} },
		}, "safety.block_accuracy confidence_interval does not match records")
	})
	t.Run("baseline and delta", func(t *testing.T) {
		assertSafetyBundleSealRejected(t, safetyReportOptions{
			mutateMetrics: func(metrics []Metric) {
				metrics[0].BaselineValue = floatPointer(0)
				metrics[0].Delta = floatPointer(0)
			},
		}, "cannot publish worker-owned baseline_value or delta")
	})
}

func TestSealRejectsUnavailableRecordsPresentedAsFullCoverage(t *testing.T) {
	assertSafetyBundleSealRejected(t, safetyReportOptions{
		unavailable: true,
		mutateReport: func(report *Report) {
			report.Summary.Coverage = serverCoverage(1, 1)
			report.Tracks[0].Coverage = serverCoverage(1, 1)
		},
	}, "report summary coverage does not match records")
}

func TestSealRejectsForgedCostLedgerAgainstRecords(t *testing.T) {
	assertSafetyBundleSealRejected(t, safetyReportOptions{
		mutateReport: func(report *Report) {
			report.Costs.Runtime.Amount = floatPointer(1)
		},
	}, "runtime cost amount does not match records")
}

func TestSealRejectsForgedTrackPresentation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*TrackReport)
		match  string
	}{
		{name: "status", mutate: func(track *TrackReport) { track.Status = "unavailable" }, match: "presentation does not match records"},
		{name: "evidence level", mutate: func(track *TrackReport) { track.EvidenceLevel = "E5" }, match: "does not match server-sealed case evidence"},
		{name: "summary", mutate: func(track *TrackReport) { track.Summary = "Perfect benchmark." }, match: "presentation does not match records"},
		{name: "error", mutate: func(track *TrackReport) { track.Error = "forged conclusion" }, match: "presentation does not match records"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertSafetyBundleSealRejected(t, safetyReportOptions{
				mutateReport: func(report *Report) { test.mutate(&report.Tracks[0]) },
			}, test.match)
		})
	}
}

type safetyReportOptions struct {
	unavailable   bool
	mutateMetrics func([]Metric)
	mutateReport  func(*Report)
}

func assertSafetyBundleSealRejected(t *testing.T, options safetyReportOptions, match string) {
	t.Helper()
	service, root := newTestService(t, &controlledProcess{}, 1)
	request := validCreateRequest()
	request.TrackIDs = []TrackID{"safety"}
	run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := time.Now().UTC()
	run.Status = StatusSealing
	run.StartedAt = &now
	if updateErr := service.store.updateRunFixture(run); updateErr != nil {
		t.Fatalf("stage sealing run: %v", updateErr)
	}
	spec := ProcessSpec{
		ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName),
		StorePath:    root,
	}
	if writeErr := writeSafetyReportBundle(spec, options); writeErr != nil {
		t.Fatalf("write forged report: %v", writeErr)
	}
	err = service.store.withEvidencePublication(func() error {
		return service.validateAndAnchorReportDuringPublication(run.ID)
	})
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), match) {
		t.Fatalf("forged bundle error=%v, want ErrInvalid containing %q", err, match)
	}
}

func writeSafetyReportBundle(spec ProcessSpec, options safetyReportOptions) error {
	fixture, prepareErr := prepareProcessReportFixture(spec)
	if prepareErr != nil {
		return prepareErr
	}
	status := "succeeded"
	if options.unavailable {
		status = "unavailable"
	}
	records := []byte(fmt.Sprintf("{\"schema_version\":\"evaluation.v1\",\"id\":\"safety-case-1\",\"track_id\":\"safety\",\"case_id\":\"case-1\",\"attempt_id\":\"attempt-case-1\",\"status\":%q,\"safety_violations\":0,\"should_block\":true,\"blocked\":true}\n", status))
	if err := os.WriteFile(filepath.Join(fixture.runDir, "records.jsonl"), records, 0o600); err != nil {
		return err
	}
	completedAt := time.Now().UTC()
	provenance := Provenance{
		SchemaVersion: SchemaVersion, GeneratedAt: completedAt, CodeRevision: fixture.manifest.CodeRevision,
		BenchmarkRevisions:        map[string]string{"evaluation-smoke": "builtin-v1"},
		WorkloadSnapshotDigest:    mustTestCanonicalDigest(fixture.workload),
		PolicySnapshotDigest:      mustTestCanonicalDigest(fixture.policy),
		BindingSnapshotDigest:     mustTestCanonicalDigest(fixture.binding),
		PoolSnapshotDigest:        mustTestCanonicalDigest(map[string]any{"pool": fixture.pool, "arms": fixture.arms}),
		EnvironmentSnapshotDigest: mustTestCanonicalDigest(fixture.environment),
		TargetID:                  fixture.manifest.Target.ID,
		Seed:                      fixture.manifest.Seed,
		RedactionPolicy:           fixture.manifest.RedactionPolicy,
	}
	metrics := []Metric{
		canonicalReducedMetric("safety.violation_rate", "Safety violation rate", "safety", "violations/case", "lower_is_better", 0, 1),
		canonicalReducedMetric("safety.block_accuracy", "Blocking decision accuracy", "safety", "fraction", "higher_is_better", 1, 1),
		{
			ID: "safety.hard_policy_static_passed", Name: "Runtime hard-policy static proof result", TrackID: "safety",
			Unit: "boolean", Direction: "higher_is_better", SampleCount: 0,
			AnalysisProvenance: validMetricAnalysisProvenanceFor("safety.hard_policy_static_passed", 0),
		},
		{
			ID: "safety.hard_policy_observation_count", Name: "Hard-policy dynamic observation count", TrackID: "safety",
			Unit: "observations", Direction: "higher_is_better", SampleCount: 0,
			AnalysisProvenance: validMetricAnalysisProvenanceFor("safety.hard_policy_observation_count", 0),
		},
	}
	metrics[1].ConfidenceInterval = serverWilsonInterval(1, 1)
	evaluated := 1
	succeeded := 1
	unavailable := 0
	trackStatus := "completed"
	trackSummary := "Collected 1 evidence records."
	if options.unavailable {
		evaluated = 0
		succeeded = 0
		unavailable = 1
		trackStatus = "unavailable"
		trackSummary = "No qualified evidence was produced."
		for index := range metrics {
			metrics[index].Value = nil
			metrics[index].ConfidenceInterval = nil
			metrics[index].SampleCount = 0
		}
	}
	if options.mutateMetrics != nil {
		options.mutateMetrics(metrics)
	}
	gates := testReleaseGates(fixture.run.ChangeProfile, completedAt)
	setTestGatePlanCoverage(gates, "safety", evaluated, 1)
	artifacts, artifactErr := writeSafetyEvidenceArtifacts(fixture.runDir, metrics, gates, provenance, succeeded, unavailable)
	if artifactErr != nil {
		return artifactErr
	}
	reportRun := fixture.run
	reportRun.Status = StatusCompleted
	reportRun.CompletedAt = &completedAt
	reportRun.Progress = RunProgress{Percent: 100, Completed: 1, Total: 1, Message: "Evaluation completed"}
	report := Report{
		SchemaVersion: SchemaVersion,
		Run:           reportRun,
		Summary: ReportSummary{
			Verdict: "unavailable", Coverage: serverCoverage(evaluated, 1),
			PassedGates: 2, UnavailableGates: 5,
		},
		Tracks: []TrackReport{{
			TrackID: "safety", Status: trackStatus, EvidenceLevel: "E0", Summary: trackSummary,
			Coverage: serverCoverage(evaluated, 1), Metrics: metrics, Gates: []Gate{gates[2]},
		}},
		Metrics: metrics, Gates: gates,
		Costs: CostLedgers{
			Runtime: CostAmount{Currency: "USD"}, EvaluationOverhead: CostAmount{Currency: "USD"}, CapacityTCO: CostAmount{Currency: "USD"},
		},
		Recommendations: []string{"Inspect the server-reduced safety diagnostic."},
		Provenance:      provenance,
		Artifacts:       artifacts,
	}
	if options.mutateReport != nil {
		options.mutateReport(&report)
	}
	if report.AttestationRevision != "" {
		return writeJSONAtomic(filepath.Join(fixture.runDir, reportFileName), report)
	}
	return writeJSONAtomic(filepath.Join(fixture.runDir, reportFileName), workerReportFromReport(report))
}

func writeSafetyEvidenceArtifacts(
	runDir string,
	metrics []Metric,
	gates []Gate,
	provenance Provenance,
	succeeded, unavailable int,
) ([]Artifact, error) {
	if err := writeJSONAtomic(filepath.Join(runDir, "metrics.json"), map[string]any{"schema_version": SchemaVersion, "metrics": metrics}); err != nil {
		return nil, err
	}
	if err := writeJSONAtomic(filepath.Join(runDir, "gates.json"), map[string]any{"schema_version": SchemaVersion, "gates": gates}); err != nil {
		return nil, err
	}
	if err := writeJSONAtomic(filepath.Join(runDir, "provenance.json"), provenance); err != nil {
		return nil, err
	}
	if err := writeJSONAtomic(filepath.Join(runDir, "failure-summary.json"), map[string]any{
		"schema_version": SchemaVersion, "total_records": 1, "failed": 0, "unavailable": unavailable,
		"by_track": []map[string]any{{"track_id": "safety", "succeeded": succeeded, "failed": 0, "unavailable": unavailable}},
	}); err != nil {
		return nil, err
	}
	publicNames := []string{"metrics.json", "gates.json", "provenance.json", "failure-summary.json"}
	artifacts := make([]Artifact, 0, len(publicNames)+1)
	var publicReceipt strings.Builder
	for _, name := range publicNames {
		data, readErr := os.ReadFile(filepath.Join(runDir, name))
		if readErr != nil {
			return nil, readErr
		}
		artifacts = append(artifacts, testArtifact(name, data))
		publicReceipt.WriteString(strings.TrimPrefix(digestBytes(data), "sha256:"))
		publicReceipt.WriteString("  " + name + "\n")
	}
	publicReceiptBytes := []byte(publicReceipt.String())
	if err := os.WriteFile(filepath.Join(runDir, publicChecksumArtifactName), publicReceiptBytes, 0o600); err != nil {
		return nil, err
	}
	artifacts = append(artifacts, testArtifact(publicChecksumArtifactName, publicReceiptBytes))
	if err := writeTestPrivateReceiptWithoutTesting(runDir); err != nil {
		return nil, err
	}
	return artifacts, nil
}
