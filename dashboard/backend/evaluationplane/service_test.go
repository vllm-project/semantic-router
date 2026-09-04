package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
	writeReport bool
	err         error
	calls       atomic.Int32
}

const testSourceRevision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func (p *controlledProcess) Run(ctx context.Context, spec ProcessSpec, emit func(WorkerEvent) error) error {
	p.calls.Add(1)
	if p.started != nil {
		p.started <- spec
	}
	if err := emit(WorkerEvent{
		Type: "progress", Message: "fixture running",
		Progress: &RunProgress{Percent: 50, Completed: 0, Total: 1, CurrentTrackID: "routing"},
	}); err != nil {
		return err
	}
	if p.release != nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-p.release:
		}
	} else {
		<-ctx.Done()
		return ctx.Err()
	}
	if p.err != nil {
		return p.err
	}
	if p.writeReport {
		return writeProcessReport(spec)
	}
	return nil
}

type processReportFixture struct {
	run         Run
	manifest    RunManifest
	runDir      string
	workload    map[string]any
	policy      map[string]any
	binding     map[string]any
	pool        map[string]any
	arms        []any
	environment map[string]any
}

func prepareProcessReportFixture(spec ProcessSpec) (processReportFixture, error) {
	var run Run
	if err := readJSON(filepath.Join(filepath.Dir(spec.ManifestPath), runFileName), &run); err != nil {
		return processReportFixture{}, err
	}
	var manifest RunManifest
	if err := readJSON(spec.ManifestPath, &manifest); err != nil {
		return processReportFixture{}, err
	}
	runDir := filepath.Dir(spec.ManifestPath)
	core := map[string][]byte{
		"cases.jsonl":         []byte("{\"schema_version\":\"evaluation.v1\",\"id\":\"case-1\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"modality\":\"text\",\"tags\":[]}\n"),
		"grading-cases.jsonl": []byte("{\"case_id\":\"case-1\"}\n"),
		"records.jsonl":       []byte("{\"schema_version\":\"evaluation.v1\",\"id\":\"routing-case-1\",\"track_id\":\"routing\",\"case_id\":\"case-1\",\"attempt_id\":\"attempt-case-1\",\"status\":\"succeeded\"}\n"),
		"failure-cases.jsonl": {},
	}
	for name, data := range core {
		if err := os.WriteFile(filepath.Join(runDir, name), data, 0o600); err != nil {
			return processReportFixture{}, err
		}
	}
	visibleSnapshot, _ := json.Marshal(map[string]any{
		"schema_version": SchemaVersion, "cases": []json.RawMessage{bytes.TrimSpace(core["cases.jsonl"])},
	})
	gradingSnapshot, _ := json.Marshal(map[string]any{
		"schema_version": SchemaVersion, "cases": []json.RawMessage{bytes.TrimSpace(core["grading-cases.jsonl"])},
	})
	visibleRef := testArtifactRef(visibleSnapshot)
	gradingRef := testArtifactRef(gradingSnapshot)
	fixtureRef := testArtifactRef(core["records.jsonl"])
	casValues := make([][]byte, 0, len(core)+2)
	for _, data := range core {
		casValues = append(casValues, data)
	}
	casValues = append(casValues, visibleSnapshot, gradingSnapshot)
	for _, data := range casValues {
		hex := strings.TrimPrefix(digestBytes(data), "sha256:")
		if err := os.WriteFile(filepath.Join(spec.StorePath, "objects", "sha256", hex), data, 0o600); err != nil {
			return processReportFixture{}, err
		}
	}
	workloadDigest, err := canonicalValueDigest(map[string]any{
		"visible_cases": visibleRef["digest"], "grading_cases": gradingRef["digest"],
	})
	if err != nil {
		return processReportFixture{}, err
	}
	workload := map[string]any{
		"schema_version": SchemaVersion, "id": "workload-" + strings.TrimPrefix(workloadDigest, "sha256:")[:16],
		"visible_cases": visibleRef, "grading_cases": gradingRef,
	}
	policy := map[string]any{
		"schema_version": SchemaVersion, "id": "fixture-policy", "entrypoint_model": "fixture-entrypoint",
		"recipe_digest": manifest.PolicySnapshotDigest,
	}
	fixtureArms, err := builtinFixtureModelArms()
	if err != nil {
		return processReportFixture{}, err
	}
	pool := map[string]any{"schema_version": SchemaVersion, "id": "fixture-pool", "arm_ids": []string{"arm-fast", "arm-strong"}}
	arms, err := modelArmsCanonicalValue(fixtureArms)
	if err != nil {
		return processReportFixture{}, err
	}
	binding := map[string]any{
		"schema_version": SchemaVersion, "id": "fixture-binding", "policy_id": "fixture-policy", "pool_id": "fixture-pool",
	}
	environment := map[string]any{
		"schema_version": SchemaVersion, "id": "fixture-environment", "target_id": "fixture",
		"platform": "local-replay", "hardware_class": "recorded", "currency": "USD",
	}
	manifestDigest, err := manifestSemanticDigest(manifest)
	if err != nil {
		return processReportFixture{}, err
	}
	lineage := map[string]any{
		"schema_version": SchemaVersion, "manifest_digest": manifestDigest,
		"workload": workload, "policy": policy, "binding": binding, "pool": pool, "arms": arms,
		"environment": environment, "fixture_ref": fixtureRef, "discovered_entrypoints": []string{}, "executors": []any{},
	}
	if err := writeJSONAtomic(filepath.Join(runDir, "lineage.json"), lineage); err != nil {
		return processReportFixture{}, err
	}
	return processReportFixture{
		run: run, manifest: manifest, runDir: runDir, workload: workload, policy: policy,
		binding: binding, pool: pool, arms: arms, environment: environment,
	}, nil
}

func mustTestCanonicalDigest(value any) string {
	encoded, err := canonicalValueDigest(value)
	if err != nil {
		panic(err)
	}
	return encoded
}

func writeProcessReportEvidence(
	fixture processReportFixture,
	provenance Provenance,
	metrics []Metric,
	gates []Gate,
) ([]Artifact, error) {
	runDir := fixture.runDir
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
		"schema_version": SchemaVersion, "total_records": 1, "failed": 0, "unavailable": 0,
		"by_track": []map[string]any{{"track_id": "routing", "succeeded": 1, "failed": 0, "unavailable": 0}},
	}); err != nil {
		return nil, err
	}
	for _, name := range []string{"report.md", "report.html"} {
		if err := os.WriteFile(filepath.Join(runDir, name), []byte("private rendered report\n"), 0o600); err != nil {
			return nil, err
		}
	}
	publicNames := []string{"metrics.json", "gates.json", "provenance.json", "failure-summary.json"}
	artifacts := make([]Artifact, 0, len(publicNames)+1)
	var publicReceipt strings.Builder
	for _, name := range publicNames {
		data, err := os.ReadFile(filepath.Join(runDir, name))
		if err != nil {
			return nil, err
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

func writeProcessReport(spec ProcessSpec) error {
	fixture, err := prepareProcessReportFixture(spec)
	if err != nil {
		return err
	}
	completedAt := time.Now().UTC()
	provenance := Provenance{
		SchemaVersion: SchemaVersion, GeneratedAt: completedAt, CodeRevision: fixture.manifest.CodeRevision,
		BenchmarkRevisions:     map[string]string{"evaluation-smoke": "builtin-v1"},
		WorkloadSnapshotDigest: mustTestCanonicalDigest(fixture.workload), PolicySnapshotDigest: mustTestCanonicalDigest(fixture.policy),
		BindingSnapshotDigest: mustTestCanonicalDigest(fixture.binding), PoolSnapshotDigest: mustTestCanonicalDigest(map[string]any{"pool": fixture.pool, "arms": fixture.arms}),
		EnvironmentSnapshotDigest: mustTestCanonicalDigest(fixture.environment), TargetID: fixture.manifest.Target.ID, Seed: fixture.manifest.Seed,
		RedactionPolicy: fixture.manifest.RedactionPolicy,
	}
	metrics := []Metric{}
	gates := testReleaseGates(fixture.run.ChangeProfile, completedAt)
	artifacts, err := writeProcessReportEvidence(fixture, provenance, metrics, gates)
	if err != nil {
		return err
	}
	reportRun := fixture.run
	reportRun.Name = fixture.run.ID
	reportRun.Description = "Evaluation suites: " + strings.Join(fixture.run.SuiteIDs, ", ")
	reportRun.Status = StatusCompleted
	reportRun.CompletedAt = &completedAt
	reportRun.Progress = RunProgress{Percent: 100, Completed: len(fixture.run.TrackIDs), Total: len(fixture.run.TrackIDs), Message: "Evaluation completed"}
	report := Report{
		SchemaVersion: SchemaVersion,
		Run:           reportRun,
		Summary: ReportSummary{
			Verdict: "unavailable", Coverage: Coverage{Evaluated: 1, Total: 1, Fraction: 1, Unavailable: 0},
			PassedGates: 2, UnavailableGates: 5,
		},
		Tracks: []TrackReport{{
			TrackID: "routing", Status: "completed", EvidenceLevel: "E0", Summary: "Collected 1 evidence records.",
			Coverage: Coverage{Evaluated: 1, Total: 1, Fraction: 1}, Metrics: []Metric{}, Gates: []Gate{gates[4]},
		}},
		Metrics:         metrics,
		Gates:           gates,
		Costs:           CostLedgers{Runtime: CostAmount{Currency: "USD"}, EvaluationOverhead: CostAmount{Currency: "USD"}, CapacityTCO: CostAmount{Currency: "USD"}},
		Recommendations: []string{"Resolve unavailable evidence."},
		Provenance:      provenance,
		Artifacts:       artifacts,
	}
	return writeJSONAtomic(filepath.Join(fixture.runDir, reportFileName), report)
}

func testArtifactRef(data []byte) map[string]any {
	return map[string]any{
		"schema_version": SchemaVersion,
		"digest":         digestBytes(data), "media_type": "application/x-ndjson", "size_bytes": len(data),
	}
}

func testArtifact(name string, data []byte) Artifact {
	return Artifact{
		ID: strings.ReplaceAll(name, ".", "-"), Name: name, Kind: filepath.Ext(name), URI: name,
		Digest: digestBytes(data), MediaType: "application/json", SizeBytes: int64(len(data)),
	}
}

func testReleaseGates(profile ChangeProfile, evaluatedAt time.Time) []Gate {
	dispositions := []string{"required", "required", "advisory", "advisory", "required", "advisory", "not_applicable", "advisory", "not_applicable", "not_applicable"}
	gates := make([]Gate, 0, 10)
	for index := range 10 {
		verdict := GateVerdict("unavailable")
		if index < 2 {
			verdict = "pass"
		} else if dispositions[index] == "not_applicable" {
			verdict = "not_applicable"
		}
		count := 1
		coverage := Coverage{Evaluated: 1, Total: 1, Fraction: 1}
		gate := Gate{
			ID: fmt.Sprintf("G%d", index), Name: gateNames[index], TrackID: gateTracks[index],
			Disposition: dispositions[index], Verdict: verdict, ChangeProfile: profile,
			ContractVersion: GateContractVersion, EvidenceRefs: gateEvidenceRefs[index], EvidenceLevel: gateEvidenceLevels[index],
			SampleCount: &count, Coverage: &coverage, Owner: gateOwners[index], EvaluatedAt: &evaluatedAt,
		}
		switch index {
		case 0:
			observed := 1.0
			gate.Observed = &observed
			gate.Threshold = &GateThreshold{Operator: ">=", Value: 1, Unit: "fraction"}
		case 1:
			observed := 1.0
			gate.Observed = &observed
			gate.Threshold = &GateThreshold{Operator: ">=", Value: 1, Unit: "boolean"}
		}
		gates = append(gates, gate)
	}
	return gates
}

func writeTestPrivateReceiptWithoutTesting(runDir string) error {
	var receipt bytes.Buffer
	for _, name := range workerRunArtifactNames {
		if name == "events.jsonl" || name == privateChecksumArtifactName || name == reportFileName {
			continue
		}
		data, err := os.ReadFile(filepath.Join(runDir, name))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return err
		}
		_, _ = fmt.Fprintf(&receipt, "%s  %s\n", strings.TrimPrefix(digestBytes(data), "sha256:"), name)
	}
	return os.WriteFile(filepath.Join(runDir, privateChecksumArtifactName), receipt.Bytes(), 0o600)
}

func newTestService(t *testing.T, process Process, maxConcurrent int) (*Service, string) {
	t.Helper()
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create private evaluation root: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	service, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		RouterAPIURL: "http://router.invalid", EnvoyURL: "http://envoy.invalid",
		CodeRevision: testSourceRevision, MaxConcurrent: maxConcurrent, Process: process,
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	return service, root
}

func validCreateRequest() CreateRunRequest {
	return CreateRunRequest{
		Name: "routing fixture", Description: "test", SuiteIDs: []string{"evaluation-smoke"},
		TrackIDs: []TrackID{"routing"}, Mode: ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
}

func TestControlledProcessReportBundlePassesServerValidation(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := time.Now().UTC()
	run.Status = StatusRunning
	run.StartedAt = &now
	if err := service.store.UpdateRun(run); err != nil {
		t.Fatalf("stage running run: %v", err)
	}
	spec := ProcessSpec{ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName), StorePath: root}
	if err := writeProcessReport(spec); err != nil {
		t.Fatalf("writeProcessReport: %v", err)
	}
	if err := service.validateAndAnchorReport(run.ID); err != nil {
		t.Fatalf("validateAndAnchorReport: %v", err)
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
	service, err := NewService(Options{DataDir: root, ConfigPath: configPath, CodeRevision: "main", Process: &controlledProcess{}})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	if _, err := service.CreateRun(context.Background(), validCreateRequest()); !errors.Is(err, ErrInvalid) {
		t.Fatalf("CreateRun mutable revision error=%v, want ErrInvalid", err)
	}
}

func TestPendingRunCannotCrossSourceRevisionUpgrade(t *testing.T) {
	serviceA, root := newTestService(t, &controlledProcess{}, 1)
	run, err := serviceA.CreateRun(context.Background(), validCreateRequest())
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
	if _, startErr := serviceB.StartRun(context.Background(), run.ID); !errors.Is(startErr, ErrConflict) || !strings.Contains(startErr.Error(), "source revision") {
		t.Fatalf("StartRun across source upgrade error=%v, want revision ErrConflict", startErr)
	}
	persisted, err := serviceB.GetRun(run.ID)
	if err != nil || persisted.Status != StatusPending || persisted.StartedAt != nil {
		t.Fatalf("rejected upgraded run mutated durable state: run=%+v err=%v", persisted, err)
	}
}

func TestStartRejectsTamperedServerManifestDigest(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
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
	if _, startErr := service.StartRun(context.Background(), run.ID); !errors.Is(startErr, ErrInvalid) || !strings.Contains(startErr.Error(), "manifest_digest") {
		t.Fatalf("StartRun tampered manifest error=%v, want manifest digest ErrInvalid", startErr)
	}
	persisted, err := service.GetRun(run.ID)
	if err != nil || persisted.Status != StatusPending || persisted.StartedAt != nil {
		t.Fatalf("rejected manifest mutated run: %+v err=%v", persisted, err)
	}
}

func TestPendingRunCannotCrossSuiteOrProfileContractUpgrade(t *testing.T) {
	for _, mutate := range []struct {
		name string
		fn   func(*RunManifest)
	}{
		{name: "suite revision", fn: func(manifest *RunManifest) { manifest.SuiteRevisions["evaluation-smoke"] = "builtin-v2" }},
		{name: "gate contract", fn: func(manifest *RunManifest) { manifest.GateContractVersion = "evaluation-release-gates.v2" }},
	} {
		t.Run(mutate.name, func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			run, err := service.CreateRun(context.Background(), validCreateRequest())
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
			if _, startErr := service.StartRun(context.Background(), run.ID); !errors.Is(startErr, ErrConflict) || !strings.Contains(startErr.Error(), "contract revision") {
				t.Fatalf("StartRun drift error=%v, want contract ErrConflict", startErr)
			}
			persisted, err := service.GetRun(run.ID)
			if err != nil || persisted.Status != StatusPending || persisted.StartedAt != nil {
				t.Fatalf("rejected drift mutated run: %+v err=%v", persisted, err)
			}
		})
	}
}

func TestReportBenchmarkRevisionsMustMatchDurableManifest(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
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
			run, _ := service.GetRun(runID)
			t.Fatalf("timed out waiting for status %s; last run=%+v", want, run)
		case <-ticker.C:
			run, err := service.GetRun(runID)
			if err == nil && run.Status == want {
				return run
			}
		}
	}
}

func TestRunLifecycleStartIsIdempotentAndBundleBacked(t *testing.T) {
	process := &controlledProcess{started: make(chan ProcessSpec, 2), release: make(chan struct{}), writeReport: true}
	service, root := newTestService(t, process, 1)
	run, createErr := service.CreateRun(context.Background(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	for _, name := range []string{runFileName, manifestFileName, eventsFileName} {
		if _, err := os.Stat(filepath.Join(root, "runs", run.ID, name)); err != nil {
			t.Fatalf("expected canonical bundle file %s: %v", name, err)
		}
	}
	started, startErr := service.StartRun(context.Background(), run.ID)
	if startErr != nil || started.Status != StatusRunning {
		t.Fatalf("StartRun = %+v, %v", started, startErr)
	}
	second, secondStartErr := service.StartRun(context.Background(), run.ID)
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
	if _, err := service.ReportJSON(run.ID); err != nil {
		t.Fatalf("ReportJSON: %v", err)
	}
	events, eventsErr := service.EventsAfter(run.ID, "")
	if eventsErr != nil {
		t.Fatalf("EventsAfter: %v", eventsErr)
	}
	if len(events) < 4 || events[0].Type != "snapshot" || events[len(events)-1].Type != "completed" {
		t.Fatalf("unexpected durable events: %+v", events)
	}
	if _, err := service.StartRun(context.Background(), run.ID); err != nil || process.calls.Load() != 1 {
		t.Fatalf("completed start should be idempotent: calls=%d err=%v", process.calls.Load(), err)
	}
	if err := service.DeleteRun(run.ID); err != nil {
		t.Fatalf("DeleteRun: %v", err)
	}
	if _, err := service.GetRun(run.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetRun after delete error=%v, want ErrNotFound", err)
	}
}

func TestRunCancellationAndRestartRecovery(t *testing.T) {
	process := &controlledProcess{started: make(chan ProcessSpec, 1)}
	service, root := newTestService(t, process, 1)
	run, createErr := service.CreateRun(context.Background(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	if _, err := service.StartRun(context.Background(), run.ID); err != nil {
		t.Fatalf("StartRun: %v", err)
	}
	<-process.started
	if _, err := service.CancelRun(run.ID); err != nil {
		t.Fatalf("CancelRun: %v", err)
	}
	waitForRunStatus(t, service, run.ID, StatusCancelled)

	interrupted, interruptedErr := service.CreateRun(context.Background(), validCreateRequest())
	if interruptedErr != nil {
		t.Fatalf("CreateRun interrupted: %v", interruptedErr)
	}
	interrupted.Status = StatusRunning
	if err := service.store.UpdateRun(interrupted); err != nil {
		t.Fatalf("stage interrupted status: %v", err)
	}
	restarted, restartErr := NewService(Options{DataDir: root, PythonPath: "python3", ConfigPath: filepath.Join(root, "config.yaml")})
	if restartErr != nil {
		t.Fatalf("restart NewService: %v", restartErr)
	}
	recovered := waitForRunStatus(t, restarted, interrupted.ID, StatusFailed)
	if recovered.CompletedAt == nil || recovered.Error == "" {
		t.Fatalf("recovery did not persist failure evidence: %+v", recovered)
	}
	events, eventsErr := restarted.EventsAfter(interrupted.ID, "1")
	if eventsErr != nil || len(events) != 1 || events[0].Type != "failed" {
		t.Fatalf("recovery events=%+v err=%v", events, eventsErr)
	}
}
