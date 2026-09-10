package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
)

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

type processReportWorkloadFixture struct {
	workload   map[string]any
	fixtureRef map[string]any
}

func prepareProcessReportWorkload(
	spec ProcessSpec,
	manifest RunManifest,
	runDir string,
) (processReportWorkloadFixture, error) {
	caseTrackIDs := make([]TrackID, 0, len(manifest.TrackIDs))
	for _, trackID := range manifest.TrackIDs {
		if trackID != "multimodal" {
			caseTrackIDs = append(caseTrackIDs, trackID)
		}
	}
	visibleCase, err := json.Marshal(map[string]any{
		"schema_version": SchemaVersion,
		"id":             "case-1",
		"track_ids":      caseTrackIDs,
		"messages":       []map[string]any{{"role": "user", "content": "test"}},
		"modality":       "text",
		"tags":           []string{},
	})
	if err != nil {
		return processReportWorkloadFixture{}, err
	}
	core := map[string][]byte{
		"cases.jsonl":         append(visibleCase, '\n'),
		"grading-cases.jsonl": []byte("{\"case_id\":\"case-1\"}\n"),
		"records.jsonl":       []byte("{\"schema_version\":\"evaluation.v1\",\"id\":\"routing-case-1\",\"track_id\":\"routing\",\"case_id\":\"case-1\",\"attempt_id\":\"attempt-case-1\",\"status\":\"succeeded\"}\n"),
	}
	for name, data := range core {
		if writeCoreErr := os.WriteFile(filepath.Join(runDir, name), data, 0o600); writeCoreErr != nil {
			return processReportWorkloadFixture{}, writeCoreErr
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
	casValues := make([][]byte, 0, len(core)+2)
	for _, data := range core {
		casValues = append(casValues, data)
	}
	casValues = append(casValues, visibleSnapshot, gradingSnapshot)
	for _, data := range casValues {
		hex := strings.TrimPrefix(digestBytes(data), "sha256:")
		if writeObjectErr := os.WriteFile(filepath.Join(spec.StorePath, "objects", "sha256", hex), data, 0o600); writeObjectErr != nil {
			return processReportWorkloadFixture{}, writeObjectErr
		}
	}
	workloadDigest, err := canonicalValueDigest(map[string]any{
		"visible_cases": visibleRef["digest"], "grading_cases": gradingRef["digest"],
	})
	if err != nil {
		return processReportWorkloadFixture{}, err
	}
	return processReportWorkloadFixture{
		workload: map[string]any{
			"schema_version": SchemaVersion, "id": "workload-" + strings.TrimPrefix(workloadDigest, "sha256:")[:16],
			"visible_cases": visibleRef, "grading_cases": gradingRef,
		},
		fixtureRef: testArtifactRef(core["records.jsonl"]),
	}, nil
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
	workloadFixture, err := prepareProcessReportWorkload(spec, manifest, runDir)
	if err != nil {
		return processReportFixture{}, err
	}
	workload := workloadFixture.workload
	policy := map[string]any{
		"schema_version": SchemaVersion, "id": "fixture-policy", "entrypoint_model": "fixture-entrypoint",
		"recipe_digest": manifest.PolicySnapshotDigest,
	}
	fixtureArms, fixtureArmsErr := builtinFixtureModelArms()
	if fixtureArmsErr != nil {
		return processReportFixture{}, fixtureArmsErr
	}
	pool := map[string]any{"schema_version": SchemaVersion, "id": "fixture-pool", "arm_ids": []string{"arm-fast", "arm-strong"}}
	arms, armsErr := modelArmsCanonicalValue(fixtureArms)
	if armsErr != nil {
		return processReportFixture{}, armsErr
	}
	binding := map[string]any{
		"schema_version": SchemaVersion, "id": "fixture-binding", "policy_id": "fixture-policy", "pool_id": "fixture-pool",
	}
	environment := map[string]any{
		"schema_version": SchemaVersion, "id": "fixture-environment", "target_id": "fixture",
		"platform": "local-replay", "hardware_class": "recorded", "currency": "USD",
	}
	manifestDigest, manifestDigestErr := manifestSemanticDigest(manifest)
	if manifestDigestErr != nil {
		return processReportFixture{}, manifestDigestErr
	}
	executorID, ok := manifestExecutorIdentity(manifest)
	if !ok {
		return processReportFixture{}, fmt.Errorf("manifest executor identity is invalid")
	}
	executors := make([]map[string]any, 0, len(manifest.TrackIDs))
	for _, trackID := range manifest.TrackIDs {
		executors = append(executors, map[string]any{
			"schema_version": SchemaVersion, "track_id": trackID,
			"executor_id": executorID, "mode": manifest.Mode,
		})
	}
	resolvedLineage := map[string]any{
		"schema_version": SchemaVersion, "manifest_digest": manifestDigest,
		"workload": workload, "policy": policy, "binding": binding, "pool": pool, "arms": arms,
		"environment": environment, "fixture_ref": workloadFixture.fixtureRef, "discovered_entrypoints": []string{}, "executors": executors,
	}
	lineage := map[string]any{
		"schema_version": SchemaVersion, "resolved_snapshot": resolvedLineage,
		"normalized_suite_identities": nil,
	}
	if writeErr := writeJSONAtomic(filepath.Join(runDir, "lineage.json"), lineage); writeErr != nil {
		return processReportFixture{}, writeErr
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
	metrics := []Metric{
		{
			ID: "routing.accuracy", Name: "Routing accuracy", TrackID: "routing",
			Unit: "fraction", Direction: "higher_is_better", SampleCount: 0,
			AnalysisProvenance: validMetricAnalysisProvenance(0),
		},
		{
			ID: "routing.robustness_pass_rate", Name: "Pinned declared-shift relation pass rate", TrackID: "routing",
			Unit: "fraction", Direction: "higher_is_better", SampleCount: 0,
			AnalysisProvenance: validMetricAnalysisProvenanceFor("routing.robustness_pass_rate", 0),
		},
		{
			ID: "routing.robustness_worst_slice_pass_rate", Name: "Worst declared robustness-slice pass rate", TrackID: "routing",
			Unit: "fraction", Direction: "higher_is_better", SampleCount: 0,
			AnalysisProvenance: validMetricAnalysisProvenanceFor("routing.robustness_worst_slice_pass_rate", 0),
		},
	}
	gates := testReleaseGates(fixture.run.ChangeProfile, completedAt)
	setTestGatePlanCoverage(gates, "routing", 1, 1)
	artifacts, err := writeProcessReportEvidence(fixture, provenance, metrics, gates)
	if err != nil {
		return err
	}
	reportRun := fixture.run
	reportRun.Name = fixture.manifest.Name
	reportRun.Description = fixture.manifest.Description
	reportRun.Status = StatusCompleted
	reportRun.CompletedAt = &completedAt
	reportRun.Progress = RunProgress{Percent: 100, Completed: len(fixture.run.TrackIDs), Total: len(fixture.run.TrackIDs), Message: "Evaluation completed"}
	report := Report{
		SchemaVersion: SchemaVersion,
		Run:           reportRun,
		Summary: ReportSummary{
			Verdict: "unavailable", Coverage: serverCoverage(1, 1),
			PassedGates: 2, UnavailableGates: 5,
		},
		Tracks: []TrackReport{{
			TrackID: "routing", Status: "completed", EvidenceLevel: "E0", Summary: "Collected 1 evidence records.",
			Coverage: serverCoverage(1, 1), Metrics: metrics, Gates: []Gate{gates[4]},
		}},
		Metrics:         metrics,
		Gates:           gates,
		Costs:           CostLedgers{Runtime: CostAmount{Currency: "USD"}, EvaluationOverhead: CostAmount{Currency: "USD"}, CapacityTCO: CostAmount{Currency: "USD"}},
		Recommendations: []string{"Resolve unavailable evidence."},
		Provenance:      provenance,
		Artifacts:       artifacts,
	}
	return writeJSONAtomic(filepath.Join(fixture.runDir, reportFileName), workerReportFromReport(report))
}

func testArtifactRef(data []byte) map[string]any {
	return map[string]any{
		"schema_version": SchemaVersion,
		"digest":         digestBytes(data), "media_type": "application/x-ndjson", "size_bytes": len(data),
	}
}

func testArtifact(name string, data []byte) Artifact {
	contract := publicArtifactContracts[name]
	return Artifact{
		ID: strings.ReplaceAll(name, ".", "-"), Name: name, Kind: contract.Kind, URI: name,
		Digest: digestBytes(data), MediaType: contract.MediaType, SizeBytes: int64(len(data)),
	}
}

func testReleaseGates(profile ChangeProfile, evaluatedAt time.Time) []Gate {
	definitions := releaseGateDefinitions()
	gates := make([]Gate, 0, len(definitions))
	for index, definition := range definitions {
		disposition, _ := releaseProfileDisposition(profile, definition.ID)
		verdict := GateVerdict("unavailable")
		if index < 2 {
			verdict = "pass"
		} else if disposition == GateDispositionNotApplicable {
			verdict = "not_applicable"
		}
		count := 1
		coverage := Coverage{Evaluated: 1, Total: 1, Fraction: 1}
		gate := Gate{
			ID: definition.ID, Name: definition.Name, TrackID: definition.TrackID,
			Disposition: disposition, Verdict: verdict, ChangeProfile: profile,
			ContractVersion: GateContractVersion, EvidenceRefs: definition.EvidenceRefs, EvidenceLevel: definition.EvidenceLevel,
			SampleCount: &count, Coverage: &coverage, Owner: definition.Owner, EvaluatedAt: &evaluatedAt,
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

func setTestGatePlanCoverage(gates []Gate, selectedTrack TrackID, evaluated, total int) {
	for index := range gates {
		gateEvaluated := evaluated
		gateTotal := total
		if gates[index].TrackID != "" && gates[index].TrackID != selectedTrack {
			gateEvaluated = 0
			gateTotal = 0
		}
		count := gateEvaluated
		coverage := Coverage{Evaluated: gateEvaluated, Total: gateTotal, Unavailable: gateTotal - gateEvaluated}
		if gateTotal > 0 {
			coverage.Fraction = float64(gateEvaluated) / float64(gateTotal)
		}
		gates[index].SampleCount = &count
		gates[index].Coverage = &coverage
	}
}

func writeTestPrivateReceiptWithoutTesting(runDir string) error {
	var receipt bytes.Buffer
	for _, name := range workerRunArtifactNames {
		if name == privateChecksumArtifactName || name == reportFileName {
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
	t.Cleanup(func() { _ = service.Close() })
	return service, root
}

func validCreateRequest() CreateRunRequest {
	return CreateRunRequest{
		ClientRequestID: newTestClientRequestID(),
		Name:            "routing fixture", Description: "test", SuiteIDs: []string{"evaluation-smoke"},
		TrackIDs: []TrackID{"routing"}, Mode: ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
}

func newTestClientRequestID() string { return uuid.NewString() }

// updateRunFixture is deliberately test-only. Production status writers expose
// one narrow transition each and never accept a caller-owned Run replacement.
func (s *Store) updateRunFixture(run Run) error {
	if err := validateStoredRun(run.ID, run); err != nil {
		return fmt.Errorf("%w: evaluation run status fixture is invalid: %w", ErrInvalid, err)
	}
	paired, err := s.acquireControlledPairMutationBarrier(run.ID)
	if err != nil {
		return err
	}
	defer s.releaseControlledPairMutationBarrier(paired)

	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	runDir, err := s.checkedRunDir(run.ID)
	if err != nil {
		return err
	}
	if paired {
		current, readErr := s.getRunPhysical(run.ID)
		if readErr != nil {
			return readErr
		}
		if transitionErr := validateFixtureStatusMutation(current, run); transitionErr != nil {
			return transitionErr
		}
	}
	return s.persistRunStatusProjectionLocked(runDir, run)
}

func validateFixtureStatusMutation(current, next Run) error {
	if current.Status == next.Status {
		if terminalStatus(current.Status) && !reflect.DeepEqual(current, next) {
			return fmt.Errorf("%w: controlled pair terminal member is immutable", ErrConflict)
		}
		return nil
	}
	switch current.Status {
	case StatusRunning:
		if next.Status == StatusSealing || terminalStatus(next.Status) {
			return nil
		}
	case StatusSealing:
		if terminalStatus(next.Status) {
			return nil
		}
	}
	return fmt.Errorf(
		"%w: controlled pair member cannot transition from %s to %s",
		ErrConflict, current.Status, next.Status,
	)
}

func stageRunningTestRun(t *testing.T, service *Service, run Run) Run {
	t.Helper()
	now := time.Now().UTC()
	run.Status = StatusRunning
	run.StartedAt = &now
	run.CompletedAt = nil
	run.Error = ""
	if err := service.store.updateRunFixture(run); err != nil {
		t.Fatalf("stage running test run: %v", err)
	}
	return run
}

func stageSealingTestRun(t *testing.T, service *Service, run Run) Run {
	t.Helper()
	run = stageRunningTestRun(t, service, run)
	sealing, err := service.store.commitRunSealing(run.ID)
	if err != nil {
		t.Fatalf("stage sealing test run: %v", err)
	}
	return sealing
}

func completeTestRun(t *testing.T, service *Service, run Run) Run {
	t.Helper()
	now := time.Now().UTC()
	if run.StartedAt == nil {
		run.StartedAt = &now
	}
	run.Status = StatusCompleted
	run.CompletedAt = &now
	run.Error = ""
	run.Progress = RunProgress{
		Percent: 100, Completed: len(run.TrackIDs), Total: len(run.TrackIDs), Message: "Evaluation completed",
	}
	if err := service.store.updateRunFixture(run); err != nil {
		t.Fatalf("complete test run: %v", err)
	}
	return run
}
