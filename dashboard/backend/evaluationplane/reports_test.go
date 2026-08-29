package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func float64Pointer(value float64) *float64 { return &value }

func TestReportJSONIsStrictVersionedIdentityCheckedAndRaw(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRun(context.Background(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	reportPath := filepath.Join(root, "runs", run.ID, reportFileName)
	valid := reportForRun(run, []Artifact{})
	valid.Gates = []Gate{{
		ID: "G0", Name: "Reproducibility", Disposition: "required", Verdict: "pass",
		ChangeProfile: run.ChangeProfile, ContractVersion: GateContractVersion,
		EvidenceRefs: []string{"provenance.json"},
	}}
	raw, marshalErr := json.MarshalIndent(valid, "", "    ")
	if marshalErr != nil {
		t.Fatalf("Marshal report: %v", marshalErr)
	}
	if err := os.WriteFile(reportPath, raw, 0o600); err != nil {
		t.Fatalf("write report: %v", err)
	}
	sealTestReport(t, service, run.ID)
	got, reportErr := service.ReportJSON(run.ID)
	if reportErr != nil || !bytes.Equal(got, raw) {
		t.Fatalf("ReportJSON did not preserve raw bundle bytes: equal=%v err=%v", bytes.Equal(got, raw), reportErr)
	}

	tests := []struct {
		name   string
		mutate func(map[string]any)
		match  string
	}{
		{name: "unknown field", mutate: func(value map[string]any) { value["engine_extension"] = true }, match: "unknown field"},
		{name: "wrong schema", mutate: func(value map[string]any) { value["schema_version"] = "evaluation.v2" }, match: "schema_version"},
		{name: "wrong nested schema", mutate: func(value map[string]any) { value["provenance"].(map[string]any)["schema_version"] = "evaluation.v2" }, match: "nested schema_version"},
		{name: "wrong identity", mutate: func(value map[string]any) { value["run"].(map[string]any)["id"] = "other-run" }, match: "identity mismatch"},
		{name: "null collection", mutate: func(value map[string]any) { value["artifacts"] = nil }, match: "cannot be null"},
		{name: "gate profile mismatch", mutate: func(value map[string]any) {
			value["gates"].([]any)[0].(map[string]any)["change_profile"] = "recipe"
		}, match: "change_profile"},
		{name: "gate contract mismatch", mutate: func(value map[string]any) {
			value["gates"].([]any)[0].(map[string]any)["contract_version"] = "old"
		}, match: "contract_version"},
		{name: "gate evidence missing", mutate: func(value map[string]any) {
			value["gates"].([]any)[0].(map[string]any)["evidence_refs"] = []any{}
		}, match: "evidence_refs"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var value map[string]any
			if err := json.Unmarshal(raw, &value); err != nil {
				t.Fatalf("Unmarshal report: %v", err)
			}
			test.mutate(value)
			if err := writeJSONAtomic(reportPath, value); err != nil {
				t.Fatalf("write mutated report: %v", err)
			}
			if _, err := service.ReportJSON(run.ID); err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("ReportJSON error=%v, want match %q", err, test.match)
			}
		})
	}
}

type pairedComparisonFixture struct {
	service      *Service
	baselineRun  Run
	candidateRun Run
	baseline     Report
	candidate    Report
}

func newPairedComparisonFixture(t *testing.T) *pairedComparisonFixture {
	t.Helper()
	service, _ := newTestService(t, &controlledProcess{}, 1)
	baselineRequest := validCreateRequest()
	baselineRequest.ChangeProfile = "recipe"
	baselineRun, baselineErr := service.CreateRun(context.Background(), baselineRequest)
	if baselineErr != nil {
		t.Fatalf("create baseline: %v", baselineErr)
	}
	baselineRun.Status = StatusCompleted
	if err := service.store.UpdateRun(baselineRun); err != nil {
		t.Fatalf("complete baseline: %v", err)
	}
	candidateRequest := validCreateRequest()
	candidateRequest.Name = "candidate"
	candidateRequest.ChangeProfile = baselineRun.ChangeProfile
	candidateRequest.BaselineRunID = baselineRun.ID
	candidateRun, candidateErr := service.CreateRun(context.Background(), candidateRequest)
	if candidateErr != nil {
		t.Fatalf("create candidate: %v", candidateErr)
	}
	baseline := reportForRun(baselineRun, nil)
	baseline.Metrics = []Metric{
		{ID: "routing.accuracy", Name: "Quality", Value: float64Pointer(0.8), Unit: "score", Direction: "higher_is_better"},
		{ID: "routing.latency_p95_ms", Name: "Latency", Value: float64Pointer(100), Unit: "ms", Direction: "lower_is_better"},
		{ID: "missing", Name: "Missing", Value: nil, Unit: "score", Direction: "higher_is_better"},
	}
	candidate := reportForRun(candidateRun, nil)
	candidate.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"
	candidate.Provenance.BindingSnapshotDigest = "sha256:candidate-binding"
	candidate.Metrics = []Metric{
		{ID: "routing.accuracy", Name: "Quality", Value: float64Pointer(0.9), Unit: "score", Direction: "higher_is_better"},
		{ID: "routing.latency_p95_ms", Name: "Latency", Value: float64Pointer(104), Unit: "ms", Direction: "lower_is_better"},
		{ID: "missing", Name: "Missing", Value: float64Pointer(1), Unit: "score", Direction: "higher_is_better"},
		{ID: "candidate-only", Name: "Candidate only", Value: float64Pointer(7), Unit: "count", Direction: "higher_is_better"},
	}
	return &pairedComparisonFixture{
		service: service, baselineRun: baselineRun, candidateRun: candidateRun,
		baseline: baseline, candidate: candidate,
	}
}

func assertInitialPairedComparison(t *testing.T, fixture *pairedComparisonFixture) {
	t.Helper()
	writeAnchoredTestReport(t, fixture.service, fixture.baselineRun.ID, fixture.baseline)
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	comparison, comparisonErr := fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID)
	if comparisonErr != nil {
		t.Fatalf("Compare: %v", comparisonErr)
	}
	if comparison.Verdict != "unavailable" || !strings.Contains(comparison.Summary, "1 improved, 1 regressed") ||
		!strings.Contains(comparison.Summary, "case-level paired") {
		t.Fatalf("unexpected comparison: %+v", comparison)
	}
	if comparison.Metrics[0].Delta == nil || *comparison.Metrics[0].Delta <= 0 {
		t.Fatalf("quality improvement delta missing: %+v", comparison.Metrics[0])
	}
	if comparison.Metrics[1].Delta == nil || *comparison.Metrics[1].Delta <= 0 {
		t.Fatalf("latency raw delta missing: %+v", comparison.Metrics[1])
	}
	if comparison.Metrics[2].Delta != nil || comparison.Metrics[3].Delta != nil {
		t.Fatalf("missing evidence must not produce deltas: %+v", comparison.Metrics)
	}
}

func assertPairedPromotionFailures(t *testing.T, fixture *pairedComparisonFixture) {
	t.Helper()
	fixture.candidate.Metrics[1].Value = float64Pointer(106)
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	comparison, comparisonErr := fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID)
	if comparisonErr != nil || comparison.Verdict != "fail" || !strings.Contains(comparison.Summary, "5%") {
		t.Fatalf("latency budget comparison=%+v err=%v", comparison, comparisonErr)
	}

	fixture.candidate.Metrics[1].Value = float64Pointer(100)
	fixture.candidate.Metrics[0].Value = float64Pointer(0.7)
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	comparison, comparisonErr = fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID)
	if comparisonErr != nil || comparison.Verdict != "fail" || !strings.Contains(comparison.Summary, "primary") {
		t.Fatalf("primary regression comparison=%+v err=%v", comparison, comparisonErr)
	}

	fixture.candidate.Summary.Verdict = "unavailable"
	fixture.candidate.Metrics = []Metric{}
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	comparison, comparisonErr = fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID)
	if comparisonErr != nil || comparison.Verdict != "unavailable" {
		t.Fatalf("unavailable comparison=%+v err=%v", comparison, comparisonErr)
	}
}

func assertPairedComparisonRejectsCohortMismatches(t *testing.T, fixture *pairedComparisonFixture) {
	t.Helper()
	fixture.candidate.Provenance.WorkloadSnapshotDigest = "sha256:different-workload"
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("workload mismatch error=%v, want ErrInvalid", err)
	}
	fixture.candidate.Provenance.WorkloadSnapshotDigest = fixture.baseline.Provenance.WorkloadSnapshotDigest
	fixture.candidate.Provenance.PoolSnapshotDigest = "sha256:different-pool"
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("pool mismatch error=%v, want ErrInvalid", err)
	}
	fixture.candidate.Provenance.PoolSnapshotDigest = fixture.baseline.Provenance.PoolSnapshotDigest
	fixture.candidate.Provenance.BenchmarkRevisions = map[string]string{"fixture": "different"}
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("benchmark mismatch error=%v, want ErrInvalid", err)
	}
	fixture.candidate.Provenance.BenchmarkRevisions = fixture.baseline.Provenance.BenchmarkRevisions
	fixture.candidate.Run.Concurrency++
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("concurrency mismatch error=%v, want ErrInvalid", err)
	}
	fixture.candidate.Run.Concurrency = fixture.baseline.Run.Concurrency
	fixture.candidate.Run.Seed++
	fixture.candidate.Provenance.Seed = fixture.candidate.Run.Seed
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.Compare(fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("cohort mismatch error=%v, want ErrInvalid", err)
	}
}

func TestCompareUsesPairedPromotionGatesInsteadOfMetricMajority(t *testing.T) {
	fixture := newPairedComparisonFixture(t)
	assertInitialPairedComparison(t, fixture)
	assertPairedPromotionFailures(t, fixture)
	assertPairedComparisonRejectsCohortMismatches(t, fixture)
}

func TestCompareRequiredGateAndPairingAvailabilityPrecedence(t *testing.T) {
	baselineRun := Run{
		SchemaVersion: SchemaVersion, ID: "baseline", Status: StatusCompleted, Mode: ModeReplay,
		TargetID: "fixture", ChangeProfile: "recipe",
		SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []TrackID{"routing"},
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	candidateRun := baselineRun
	candidateRun.ID = "candidate"
	candidateRun.BaselineRunID = baselineRun.ID
	baseline := reportForRun(baselineRun, nil)
	candidate := reportForRun(candidateRun, nil)
	baseline.Metrics = []Metric{{ID: "advisory.score", Name: "Advisory", Value: float64Pointer(1), Unit: "score", Direction: "higher_is_better"}}
	candidate.Metrics = []Metric{{ID: "advisory.score", Name: "Advisory", Value: float64Pointer(2), Unit: "score", Direction: "higher_is_better"}}
	candidate.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"
	candidate.Provenance.BindingSnapshotDigest = "sha256:candidate-binding"

	candidate.Gates = []Gate{{
		ID: "G3", Name: "Required", Disposition: "required", Verdict: "fail",
		ChangeProfile: candidateRun.ChangeProfile, ContractVersion: GateContractVersion, EvidenceRefs: []string{"metrics.json"},
	}}
	comparison, err := comparePairedReports(baseline, candidate)
	if err != nil || comparison.Verdict != "fail" {
		t.Fatalf("required failure comparison=%+v err=%v", comparison, err)
	}
	candidate.Gates[0].Verdict = "unavailable"
	comparison, err = comparePairedReports(baseline, candidate)
	if err != nil || comparison.Verdict != "unavailable" {
		t.Fatalf("required unavailable comparison=%+v err=%v", comparison, err)
	}
	candidate.Gates = []Gate{}
	candidate.Metrics = []Metric{{ID: "candidate-only", Name: "Candidate", Value: float64Pointer(2), Unit: "score", Direction: "higher_is_better"}}
	comparison, err = comparePairedReports(baseline, candidate)
	if err != nil || comparison.Verdict != "unavailable" || !strings.Contains(comparison.Summary, "No matched direction-aware aggregate") {
		t.Fatalf("unpaired comparison=%+v err=%v", comparison, err)
	}

	candidate.Metrics = baseline.Metrics
	candidate.Provenance.PolicySnapshotDigest = ""
	if _, err := comparePairedReports(baseline, candidate); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing policy digest error=%v, want ErrInvalid", err)
	}
}

func TestCompareFreezesSnapshotFactorsByChangeProfile(t *testing.T) {
	profiles := []struct {
		profile ChangeProfile
		allowed map[string]bool
	}{
		{profile: "schema_adapter", allowed: map[string]bool{}},
		{profile: "recipe", allowed: map[string]bool{"policy": true, "binding": true}},
		{profile: "selector", allowed: map[string]bool{"policy": true, "binding": true}},
		{profile: "model_pool", allowed: map[string]bool{"pool": true, "binding": true}},
		{profile: "runtime_capacity", allowed: map[string]bool{"environment": true}},
		{profile: "agent_multimodal", allowed: map[string]bool{"policy": true, "binding": true}},
		{profile: "online_adaptation", allowed: map[string]bool{"policy": true, "binding": true}},
	}
	for _, profile := range profiles {
		t.Run(string(profile.profile), func(t *testing.T) {
			baselineRun := Run{
				SchemaVersion: SchemaVersion, ID: "baseline", Status: StatusCompleted,
				Mode: ModeReplay, TargetID: "fixture", ChangeProfile: profile.profile,
				SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []TrackID{"routing"},
				SampleLimit: 4, Concurrency: 1, Seed: 17,
			}
			candidateRun := baselineRun
			candidateRun.ID, candidateRun.BaselineRunID = "candidate", "baseline"
			baseline := reportForRun(baselineRun, nil)
			candidate := reportForRun(candidateRun, nil)
			baseline.Metrics = []Metric{{ID: "routing.accuracy", Name: "Accuracy", Value: float64Pointer(0.8), Unit: "score", Direction: "higher_is_better"}}
			candidate.Metrics = []Metric{{ID: "routing.accuracy", Name: "Accuracy", Value: float64Pointer(0.9), Unit: "score", Direction: "higher_is_better"}}

			for _, factor := range []string{"policy", "binding", "pool", "environment"} {
				candidateFactor := candidate
				switch factor {
				case "policy":
					candidateFactor.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"
				case "binding":
					candidateFactor.Provenance.BindingSnapshotDigest = "sha256:candidate-binding"
				case "pool":
					candidateFactor.Provenance.PoolSnapshotDigest = "sha256:candidate-pool"
				case "environment":
					candidateFactor.Provenance.EnvironmentSnapshotDigest = "sha256:candidate-environment"
				}
				_, err := comparePairedReports(baseline, candidateFactor)
				if profile.allowed[factor] && err != nil {
					t.Fatalf("allowed %s treatment rejected: %v", factor, err)
				}
				if !profile.allowed[factor] && !errors.Is(err, ErrInvalid) {
					t.Fatalf("frozen %s treatment error=%v, want ErrInvalid", factor, err)
				}
			}
		})
	}
}

func TestCompareRejectsSelfWrongBaselineLinkAndUnchangedTreatment(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	baselineRun := Run{
		SchemaVersion: SchemaVersion, ID: "baseline", Status: StatusCompleted, Mode: ModeReplay,
		TargetID: "fixture", ChangeProfile: "model_pool", SuiteIDs: []string{"evaluation-smoke"},
		TrackIDs: []TrackID{"model_pool"}, SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	candidateRun := baselineRun
	candidateRun.ID = "candidate"
	candidateRun.BaselineRunID = baselineRun.ID
	baseline := reportForRun(baselineRun, nil)
	candidate := reportForRun(candidateRun, nil)
	baseline.Metrics = []Metric{{ID: "model_pool.oracle_quality", Name: "Oracle", Value: float64Pointer(0.8), Unit: "score", Direction: "higher_is_better"}}
	candidate.Metrics = []Metric{{ID: "model_pool.oracle_quality", Name: "Oracle", Value: float64Pointer(0.9), Unit: "score", Direction: "higher_is_better"}}

	if _, err := service.Compare("same", "same"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("self-comparison error=%v, want ErrInvalid", err)
	}
	if _, err := comparePairedReports(baseline, baseline); !errors.Is(err, ErrInvalid) {
		t.Fatalf("same report comparison error=%v, want ErrInvalid", err)
	}

	wrongLink := candidate
	wrongLink.Run.BaselineRunID = "other-baseline"
	if _, err := comparePairedReports(baseline, wrongLink); !errors.Is(err, ErrInvalid) {
		t.Fatalf("wrong baseline lineage error=%v, want ErrInvalid", err)
	}
	if _, err := comparePairedReports(baseline, candidate); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "treatment factor") {
		t.Fatalf("unchanged treatment error=%v, want treatment ErrInvalid", err)
	}

	candidate.Provenance.PoolSnapshotDigest = "sha256:candidate-pool"
	if _, err := comparePairedReports(baseline, candidate); err != nil {
		t.Fatalf("declared model-pool treatment was rejected: %v", err)
	}

	schemaBaselineRun := baselineRun
	schemaBaselineRun.ChangeProfile = "schema_adapter"
	schemaCandidateRun := schemaBaselineRun
	schemaCandidateRun.ID, schemaCandidateRun.BaselineRunID = "schema-candidate", schemaBaselineRun.ID
	schemaBaseline := reportForRun(schemaBaselineRun, nil)
	schemaCandidate := reportForRun(schemaCandidateRun, nil)
	if _, err := comparePairedReports(schemaBaseline, schemaCandidate); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "source code revision") {
		t.Fatalf("unchanged schema-adapter revision error=%v, want treatment ErrInvalid", err)
	}
	schemaCandidate.Provenance.CodeRevision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	if _, err := comparePairedReports(schemaBaseline, schemaCandidate); err != nil {
		t.Fatalf("changed schema-adapter source revision was rejected: %v", err)
	}
}
