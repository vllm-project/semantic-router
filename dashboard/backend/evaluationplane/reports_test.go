package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReportJSONIsStrictVersionedIdentityCheckedAndRaw(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	reportPath := filepath.Join(root, "runs", run.ID, reportFileName)
	valid := reportForRun(run, []Artifact{})
	valid.Metrics = []Metric{{
		ID: "routing.accuracy", Name: "Routing accuracy", TrackID: "routing",
		Value: float64Pointer(0.75), Unit: "fraction", Direction: "higher_is_better",
		ConfidenceInterval: []float64{0.5, 0.9}, SampleCount: 4,
		AnalysisProvenance: validMetricAnalysisProvenance(0),
	}}
	valid.Gates = []Gate{{
		ID: "G0", Name: "Reproducibility", Disposition: "required", Verdict: "pass",
		ChangeProfile: run.ChangeProfile, ContractVersion: GateContractVersion,
		EvidenceRefs: []string{"provenance.json"},
	}}
	raw, marshalErr := json.MarshalIndent(workerReportFromReport(valid), "", "    ")
	if marshalErr != nil {
		t.Fatalf("Marshal report: %v", marshalErr)
	}
	if err := os.WriteFile(reportPath, raw, 0o600); err != nil {
		t.Fatalf("write report: %v", err)
	}
	sealTestReport(t, service, run.ID)
	sealedRaw, readErr := service.store.ReadReport(run.ID)
	if readErr != nil {
		t.Fatalf("read canonical sealed report: %v", readErr)
	}
	got, reportErr := service.ReportJSONAs(SystemActor(), run.ID)
	if reportErr != nil || !bytes.Equal(got, sealedRaw) {
		t.Fatalf("ReportJSON did not preserve sealed bundle bytes: equal=%v err=%v", bytes.Equal(got, sealedRaw), reportErr)
	}
	raw = sealedRaw

	tests := []struct {
		name   string
		mutate func(map[string]any)
		match  string
	}{
		{name: "unknown field", mutate: func(value map[string]any) { value["engine_extension"] = true }, match: "unknown field"},
		{name: "wrong schema", mutate: func(value map[string]any) { value["schema_version"] = "evaluation.v2" }, match: "schema_version"},
		{name: "wrong nested schema", mutate: func(value map[string]any) { value["provenance"].(map[string]any)["schema_version"] = "evaluation.v2" }, match: "nested schema_version"},
		{name: "wrong identity", mutate: func(value map[string]any) { value["run"].(map[string]any)["id"] = "other-run" }, match: "identity mismatch"},
		{name: "gate-only summary verdict", mutate: func(value map[string]any) { value["summary"].(map[string]any)["verdict"] = "not_applicable" }, match: "summary verdict"},
		{name: "waived summary verdict", mutate: func(value map[string]any) { value["summary"].(map[string]any)["verdict"] = "waived" }, match: "summary verdict"},
		{name: "null collection", mutate: func(value map[string]any) { value["artifacts"] = nil }, match: "cannot be null"},
		{name: "missing routing recipe field", mutate: func(value map[string]any) { delete(value, "routing_recipe_report") }, match: "server-owned routing_recipe_report field"},
		{name: "gate profile mismatch", mutate: func(value map[string]any) {
			value["gates"].([]any)[0].(map[string]any)["change_profile"] = "recipe"
		}, match: "change_profile"},
		{name: "gate contract mismatch", mutate: func(value map[string]any) {
			value["gates"].([]any)[0].(map[string]any)["contract_version"] = "old"
		}, match: "contract_version"},
		{name: "gate evidence missing", mutate: func(value map[string]any) {
			value["gates"].([]any)[0].(map[string]any)["evidence_refs"] = []any{}
		}, match: "evidence_refs"},
		{name: "blank metric name", mutate: func(value map[string]any) {
			value["metrics"].([]any)[0].(map[string]any)["name"] = "  "
		}, match: "blank name"},
		{name: "blank metric unit", mutate: func(value map[string]any) {
			value["metrics"].([]any)[0].(map[string]any)["unit"] = ""
		}, match: "blank unit"},
		{name: "metric outside selected tracks", mutate: func(value map[string]any) {
			value["metrics"].([]any)[0].(map[string]any)["track_id"] = "joint"
		}, match: "not selected by the run"},
		{name: "negative metric sample count", mutate: func(value map[string]any) {
			value["metrics"].([]any)[0].(map[string]any)["sample_count"] = -1
		}, match: "sample_count cannot be negative"},
		{name: "missing metric analysis provenance", mutate: func(value map[string]any) {
			delete(value["metrics"].([]any)[0].(map[string]any), "analysis_provenance")
		}, match: "analysis_provenance"},
		{name: "illegal metric analysis provenance", mutate: func(value map[string]any) {
			value["metrics"].([]any)[0].(map[string]any)["analysis_provenance"].(map[string]any)["missingness"] = "impute"
		}, match: "registered estimator"},
		{name: "malformed metric confidence interval", mutate: func(value map[string]any) {
			value["metrics"].([]any)[0].(map[string]any)["confidence_interval"] = []any{0.5}
		}, match: "exactly two bounds"},
		{name: "reversed metric confidence interval", mutate: func(value map[string]any) {
			value["metrics"].([]any)[0].(map[string]any)["confidence_interval"] = []any{0.9, 0.5}
		}, match: "bounds are reversed"},
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
			if _, err := service.ReportJSONAs(SystemActor(), run.ID); err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("ReportJSON error=%v, want match %q", err, test.match)
			}
		})
	}
}

func TestValidateReportMetricsRejectsMisleadingNumericEvidence(t *testing.T) {
	validMetric := validRoutingReportMetric
	if err := validateReportMetrics([]Metric{validMetric()}, []TrackID{"routing"}); err != nil {
		t.Fatalf("valid metric rejected: %v", err)
	}
	systemMetric := validMetric()
	systemMetric.ID = "system.total_cost"
	systemMetric.Name = "Total cost"
	systemMetric.TrackID = ""
	if err := validateReportMetrics([]Metric{systemMetric}, []TrackID{"routing"}); err == nil || !strings.Contains(err.Error(), "metric id is not registered") {
		t.Fatalf("unknown system-level metric error=%v, want fail-closed catalog rejection", err)
	}
	unavailable := validMetric()
	unavailable.Value = nil
	unavailable.ConfidenceInterval = nil
	unavailable.SampleCount = 0
	if err := validateReportMetrics([]Metric{unavailable}, []TrackID{"routing"}); err != nil {
		t.Fatalf("unavailable metric without statistical claims rejected: %v", err)
	}
	compared := validMetric()
	compared.BaselineValue = float64Pointer(0.7)
	compared.Delta = float64Pointer(*compared.Value - *compared.BaselineValue)
	if err := validateReportMetrics([]Metric{compared}, []TrackID{"routing"}); err != nil {
		t.Fatalf("consistent comparison metric rejected: %v", err)
	}
	roundedComparison := validMetric()
	roundedComparison.ID = "routing.latency_p95_ms"
	roundedComparison.Name = "Latency p95"
	roundedComparison.AnalysisProvenance = validMetricAnalysisProvenanceFor(roundedComparison.ID, 0)
	roundedComparison.Unit = "ms"
	roundedComparison.Value = float64Pointer(1000.1)
	roundedComparison.BaselineValue = float64Pointer(1000)
	roundedComparison.Delta = float64Pointer(0.1)
	if err := validateReportMetrics([]Metric{roundedComparison}, []TrackID{"routing"}); err != nil {
		t.Fatalf("JSON-rounded comparison metric rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*Metric)
		match  string
	}{
		{name: "blank id", mutate: func(metric *Metric) { metric.ID = " " }, match: "blank metric id"},
		{name: "blank name", mutate: func(metric *Metric) { metric.Name = "\t" }, match: "blank name"},
		{name: "blank unit", mutate: func(metric *Metric) { metric.Unit = " " }, match: "blank unit"},
		{name: "unselected track", mutate: func(metric *Metric) { metric.TrackID = "joint" }, match: "not selected by the run"},
		{name: "invalid direction", mutate: func(metric *Metric) { metric.Direction = "maximize" }, match: "invalid direction"},
		{name: "negative sample count", mutate: func(metric *Metric) { metric.SampleCount = -1 }, match: "sample_count cannot be negative"},
		{name: "empty confidence interval", mutate: func(metric *Metric) { metric.ConfidenceInterval = []float64{} }, match: "exactly two bounds"},
		{name: "short confidence interval", mutate: func(metric *Metric) { metric.ConfidenceInterval = []float64{0.7} }, match: "exactly two bounds"},
		{name: "long confidence interval", mutate: func(metric *Metric) { metric.ConfidenceInterval = []float64{0.6, 0.8, 0.9} }, match: "exactly two bounds"},
		{name: "reversed confidence interval", mutate: func(metric *Metric) { metric.ConfidenceInterval = []float64{0.9, 0.7} }, match: "bounds are reversed"},
		{name: "non-finite confidence lower bound", mutate: func(metric *Metric) { metric.ConfidenceInterval = []float64{math.NaN(), 0.9} }, match: "bounds must be finite"},
		{name: "non-finite confidence upper bound", mutate: func(metric *Metric) { metric.ConfidenceInterval = []float64{0.7, math.Inf(1)} }, match: "bounds must be finite"},
		{name: "confidence interval without value", mutate: func(metric *Metric) { metric.Value = nil }, match: "requires an estimate and samples"},
		{name: "confidence interval without samples", mutate: func(metric *Metric) { metric.SampleCount = 0 }, match: "requires an estimate and samples"},
		{name: "non-finite value", mutate: func(metric *Metric) { metric.Value = float64Pointer(math.NaN()) }, match: "value must be finite"},
		{name: "non-finite baseline", mutate: func(metric *Metric) {
			metric.BaselineValue = float64Pointer(math.Inf(1))
			metric.Delta = float64Pointer(0)
		}, match: "baseline_value must be finite"},
		{name: "non-finite delta", mutate: func(metric *Metric) {
			metric.BaselineValue = float64Pointer(0.7)
			metric.Delta = float64Pointer(math.Inf(-1))
		}, match: "delta must be finite"},
		{name: "baseline without delta", mutate: func(metric *Metric) { metric.BaselineValue = float64Pointer(0.7) }, match: "must be published together"},
		{name: "delta without baseline", mutate: func(metric *Metric) { metric.Delta = float64Pointer(0.1) }, match: "must be published together"},
		{name: "comparison without candidate value", mutate: func(metric *Metric) {
			metric.Value = nil
			metric.ConfidenceInterval = nil
			metric.BaselineValue = float64Pointer(0.7)
			metric.Delta = float64Pointer(0.1)
		}, match: "requires a candidate value"},
		{name: "inconsistent delta", mutate: func(metric *Metric) {
			metric.BaselineValue = float64Pointer(0.7)
			metric.Delta = float64Pointer(0.2)
		}, match: "does not match value minus baseline_value"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metric := validMetric()
			test.mutate(&metric)
			err := validateReportMetrics([]Metric{metric}, []TrackID{"routing"})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("validateReportMetrics error=%v, want match %q", err, test.match)
			}
		})
	}

	first, second := validMetric(), validMetric()
	if err := validateReportMetrics([]Metric{first, second}, []TrackID{"routing"}); err == nil || !strings.Contains(err.Error(), "duplicate metric id") {
		t.Fatalf("duplicate metric validation error=%v", err)
	}
}

func TestDecodeReportStrictRequiresCurrentAttestationRevision(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	report := reportForRun(run, nil)
	for _, revision := range []string{"", "evaluation-server-attestation.v999"} {
		report.AttestationRevision = revision
		data, marshalErr := json.Marshal(report)
		if marshalErr != nil {
			t.Fatalf("encode report: %v", marshalErr)
		}
		if _, err := decodeReportStrict(run.ID, data); err == nil || !strings.Contains(err.Error(), "attestation_revision must be") {
			t.Fatalf("revision %q error=%v, want exact current contract rejection", revision, err)
		}
	}
	report.AttestationRevision = ServerAttestationRevision
	report.Run.ClientRequestID = ""
	data, marshalErr := json.Marshal(report)
	if marshalErr != nil {
		t.Fatalf("encode report without client identity: %v", marshalErr)
	}
	if _, decodeErr := decodeReportStrict(run.ID, data); !errors.Is(decodeErr, ErrInvalid) || !strings.Contains(decodeErr.Error(), "identity") {
		t.Fatalf("server report missing client identity error=%v", decodeErr)
	}
	workerData, workerMarshalErr := json.Marshal(workerReportFromReport(report))
	if workerMarshalErr != nil {
		t.Fatalf("encode worker report without client identity: %v", workerMarshalErr)
	}
	if _, err := decodeWorkerReportStrict(run.ID, workerData); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "identity") {
		t.Fatalf("worker report missing client identity error=%v", err)
	}
}

func TestWorkerReportEnvelopeRejectsEveryServerOwnedPublicationField(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	draft, marshalErr := json.Marshal(workerReportFromReport(reportForRun(run, nil)))
	if marshalErr != nil {
		t.Fatalf("encode worker draft: %v", marshalErr)
	}
	tests := []struct {
		name   string
		mutate func(map[string]any)
		field  string
	}{
		{name: "attestation revision", field: "attestation_revision", mutate: func(value map[string]any) {
			value["attestation_revision"] = ServerAttestationRevision
		}},
		{name: "method reports", field: "method_reports", mutate: func(value map[string]any) {
			value["method_reports"] = []any{}
		}},
		{name: "sealed track evidence levels", field: "track_evidence_levels", mutate: func(value map[string]any) {
			value["run"].(map[string]any)["track_evidence_levels"] = map[string]any{}
		}},
		{name: "controlled pair membership", field: "controlled_pair", mutate: func(value map[string]any) {
			value["run"].(map[string]any)["controlled_pair"] = map[string]any{"pair_id": "forged", "role": "candidate"}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var forged map[string]any
			if err := json.Unmarshal(draft, &forged); err != nil {
				t.Fatalf("decode worker draft: %v", err)
			}
			test.mutate(forged)
			encoded, err := json.Marshal(forged)
			if err != nil {
				t.Fatalf("encode forged worker draft: %v", err)
			}
			if _, err := decodeWorkerReportStrict(run.ID, encoded); err == nil || !strings.Contains(err.Error(), "unknown field \""+test.field+"\"") {
				t.Fatalf("forged %s error=%v, want strict unknown-field rejection", test.field, err)
			}
		})
	}
}

func TestControlledPairReportSealRestoresDurableMembership(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish controlled pair: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	started, err := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if err != nil {
		t.Fatalf("start controlled pair: %v", err)
	}
	baseline := started.Baseline
	baseline, err = service.store.GetRun(baseline.ID)
	if err != nil {
		t.Fatalf("read controlled-pair baseline: %v", err)
	}
	if baseline.ControlledPair == nil {
		t.Fatal("durable controlled-pair baseline lacks membership")
	}
	expected := *baseline.ControlledPair

	sealExistingControlledPairMember(t, service, baseline)
	sealed, err := service.decodedReport(baseline.ID)
	if err != nil {
		t.Fatalf("read sealed controlled-pair report: %v", err)
	}
	if sealed.Run.ControlledPair == nil || *sealed.Run.ControlledPair != expected {
		t.Fatalf("sealed membership=%+v, want %+v", sealed.Run.ControlledPair, expected)
	}
}

type pairedComparisonFixture struct {
	service      *Service
	baselineRun  Run
	candidateRun Run
	baseline     Report
	candidate    Report
}

func writePairedPrivateRecord(t *testing.T, service *Service, runID string, quality float64) {
	t.Helper()
	record := fmt.Sprintf(
		"{\"schema_version\":%q,\"id\":\"routing-case-1\",\"track_id\":\"routing\",\"case_id\":\"case-1\",\"attempt_id\":\"attempt-case-1\",\"status\":\"succeeded\",\"quality\":%.6f}\n",
		SchemaVersion,
		quality,
	)
	path := filepath.Join(service.store.runsRoot, runID, "records.jsonl")
	if err := os.WriteFile(path, []byte(record), 0o600); err != nil {
		t.Fatalf("write paired private record: %v", err)
	}
}

func newPairedComparisonFixture(t *testing.T) *pairedComparisonFixture {
	t.Helper()
	metric := func(id, name, unit, direction string, value *float64, samples int) Metric {
		return Metric{
			ID: id, Name: name, TrackID: "routing", Value: value, Unit: unit,
			Direction: direction, SampleCount: samples,
			AnalysisProvenance: validMetricAnalysisProvenanceFor(id, 0),
		}
	}
	service, _ := newTestService(t, &controlledProcess{}, 1)
	baselineRequest := validCreateRequest()
	baselineRequest.ChangeProfile = "recipe"
	baselineRun, baselineErr := service.CreateRunAs(context.Background(), SystemActor(), baselineRequest)
	if baselineErr != nil {
		t.Fatalf("create baseline: %v", baselineErr)
	}
	baselineRun = completeTestRun(t, service, baselineRun)
	candidateRequest := validCreateRequest()
	candidateRequest.Name = "candidate"
	candidateRequest.ChangeProfile = baselineRun.ChangeProfile
	candidateRequest.BaselineRunID = baselineRun.ID
	candidateRun, candidateErr := service.CreateRunAs(context.Background(), SystemActor(), candidateRequest)
	if candidateErr != nil {
		t.Fatalf("create candidate: %v", candidateErr)
	}
	writePairedPrivateRecord(t, service, baselineRun.ID, 0.8)
	writePairedPrivateRecord(t, service, candidateRun.ID, 0.9)
	baseline := reportForRun(baselineRun, nil)
	baseline.AttestationRevision = ServerAttestationRevision
	baseline.Metrics = []Metric{
		metric("routing.accuracy", "Quality", "score", "higher_is_better", float64Pointer(0.8), 1),
		metric("routing.latency_p95_ms", "Latency", "ms", "lower_is_better", float64Pointer(100), 1),
		metric("routing.coverage", "Missing", "fraction", "higher_is_better", nil, 0),
	}
	candidate := reportForRun(candidateRun, nil)
	candidate.AttestationRevision = ServerAttestationRevision
	candidate.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"
	candidate.Metrics = []Metric{
		metric("routing.accuracy", "Quality", "score", "higher_is_better", float64Pointer(0.9), 1),
		metric("routing.latency_p95_ms", "Latency", "ms", "lower_is_better", float64Pointer(104), 1),
		metric("routing.coverage", "Missing", "fraction", "higher_is_better", float64Pointer(1), 1),
		metric("routing.fallback_rate", "Candidate only", "fraction", "lower_is_better", float64Pointer(0), 1),
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
	comparison, comparisonErr := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID)
	if comparisonErr != nil {
		t.Fatalf("Compare: %v", comparisonErr)
	}
	if comparison.Verdict != "unavailable" || !strings.Contains(comparison.Summary, "1 improved, 1 regressed") ||
		!strings.Contains(comparison.Summary, "complete qualified evidence") {
		t.Fatalf("unexpected comparison: %+v", comparison)
	}
	if comparison.AttestationRevision != ServerAttestationRevision {
		t.Fatalf("comparison attestation=%q, want %q", comparison.AttestationRevision, ServerAttestationRevision)
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
	comparison, comparisonErr := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID)
	if comparisonErr != nil || comparison.Verdict != "fail" || !strings.Contains(comparison.Summary, "latency") {
		t.Fatalf("latency budget comparison=%+v err=%v", comparison, comparisonErr)
	}

	fixture.candidate.Metrics[1].Value = float64Pointer(100)
	fixture.candidate.Metrics[0].Value = float64Pointer(0.7)
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	comparison, comparisonErr = fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID)
	if comparisonErr != nil || comparison.Verdict != "unavailable" || !strings.Contains(comparison.Summary, "complete qualified evidence") {
		t.Fatalf("primary regression comparison=%+v err=%v", comparison, comparisonErr)
	}

	fixture.candidate.Summary.Verdict = "unavailable"
	fixture.candidate.Metrics = []Metric{}
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	comparison, comparisonErr = fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID)
	if comparisonErr != nil || comparison.Verdict != "unavailable" {
		t.Fatalf("unavailable comparison=%+v err=%v", comparison, comparisonErr)
	}
}

func assertPairedComparisonRejectsCohortMismatches(t *testing.T, fixture *pairedComparisonFixture) {
	t.Helper()
	fixture.candidate.Provenance.WorkloadSnapshotDigest = "sha256:different-workload"
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("workload mismatch error=%v, want ErrInvalid", err)
	}
	fixture.candidate.Provenance.WorkloadSnapshotDigest = fixture.baseline.Provenance.WorkloadSnapshotDigest
	fixture.candidate.Provenance.PoolSnapshotDigest = "sha256:different-pool"
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("pool mismatch error=%v, want ErrInvalid", err)
	}
	fixture.candidate.Provenance.PoolSnapshotDigest = fixture.baseline.Provenance.PoolSnapshotDigest
	fixture.candidate.Provenance.BenchmarkRevisions = map[string]string{"fixture": "different"}
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("benchmark mismatch error=%v, want ErrInvalid", err)
	}
	fixture.candidate.Provenance.BenchmarkRevisions = fixture.baseline.Provenance.BenchmarkRevisions
	fixture.candidate.Run.Concurrency++
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("concurrency mismatch error=%v, want ErrInvalid", err)
	}
	fixture.candidate.Run.Concurrency = fixture.baseline.Run.Concurrency
	fixture.candidate.Run.Seed++
	fixture.candidate.Provenance.Seed = fixture.candidate.Run.Seed
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("cohort mismatch error=%v, want ErrInvalid", err)
	}
}

func TestCompareUsesPairedPromotionGatesInsteadOfMetricMajority(t *testing.T) {
	fixture := newPairedComparisonFixture(t)
	assertInitialPairedComparison(t, fixture)
	assertPairedPromotionFailures(t, fixture)
	assertPairedComparisonRejectsCohortMismatches(t, fixture)
}

func TestCompareRejectsUnpairedAndTamperedPrivateRecords(t *testing.T) {
	fixture := newPairedComparisonFixture(t)
	writeAnchoredTestReport(t, fixture.service, fixture.baselineRun.ID, fixture.baseline)
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	candidatePath := filepath.Join(fixture.service.store.runsRoot, fixture.candidateRun.ID, "records.jsonl")
	unpaired := fmt.Sprintf(
		"{\"schema_version\":%q,\"id\":\"different-record\",\"track_id\":\"routing\",\"case_id\":\"case-1\",\"attempt_id\":\"attempt-case-1\",\"status\":\"succeeded\",\"quality\":0.9}\n",
		SchemaVersion,
	)
	if err := os.WriteFile(candidatePath, []byte(unpaired), 0o600); err != nil {
		t.Fatal(err)
	}
	writeAnchoredTestReport(t, fixture.service, fixture.candidateRun.ID, fixture.candidate)
	if _, err := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "case-aligned") {
		t.Fatalf("unpaired private records error=%v", err)
	}
	if err := os.WriteFile(candidatePath, []byte("tampered\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.service.CompareAs(SystemActor(), fixture.baselineRun.ID, fixture.candidateRun.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("tampered private records error=%v, want ErrInvalid", err)
	}
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
	baseline.AttestationRevision = ServerAttestationRevision
	candidate.AttestationRevision = ServerAttestationRevision
	baseline.Metrics = []Metric{{ID: "advisory.score", Name: "Advisory", Value: float64Pointer(1), Unit: "score", Direction: "higher_is_better"}}
	candidate.Metrics = []Metric{{ID: "advisory.score", Name: "Advisory", Value: float64Pointer(2), Unit: "score", Direction: "higher_is_better"}}
	candidate.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"

	candidate.Gates = []Gate{{
		ID: "G4", Name: "Required", Disposition: "required", Verdict: "fail",
		ChangeProfile: candidateRun.ChangeProfile, ContractVersion: GateContractVersion, EvidenceRefs: []string{"metrics.json"},
	}}
	comparison, err := comparePairedReports(baseline, candidate, nil, nil)
	if err != nil || comparison.Verdict != "fail" {
		t.Fatalf("required failure comparison=%+v err=%v", comparison, err)
	}
	candidate.Gates[0].Verdict = "unavailable"
	comparison, err = comparePairedReports(baseline, candidate, nil, nil)
	if err != nil || comparison.Verdict != "unavailable" {
		t.Fatalf("required unavailable comparison=%+v err=%v", comparison, err)
	}
	candidate.Gates = []Gate{}
	candidate.Metrics = []Metric{{ID: "candidate-only", Name: "Candidate", Value: float64Pointer(2), Unit: "score", Direction: "higher_is_better"}}
	comparison, err = comparePairedReports(baseline, candidate, nil, nil)
	if err != nil || comparison.Verdict != "unavailable" || !strings.Contains(comparison.Summary, "complete qualified evidence") {
		t.Fatalf("unpaired comparison=%+v err=%v", comparison, err)
	}

	candidate.Metrics = baseline.Metrics
	candidate.Provenance.PolicySnapshotDigest = ""
	if _, err := comparePairedReports(baseline, candidate, nil, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("missing policy digest error=%v, want ErrInvalid", err)
	}
}

func TestCompareRequiresCurrentAttestationAndAlwaysPublishesIt(t *testing.T) {
	baselineRun := Run{
		SchemaVersion: SchemaVersion, ID: "baseline", Status: StatusCompleted,
		Mode: ModeReplay, TargetID: "fixture", ChangeProfile: "recipe",
		SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []TrackID{"routing"},
		SampleLimit: 4, Concurrency: 1, Seed: 17, EvidenceLevel: "E0",
	}
	candidateRun := baselineRun
	candidateRun.ID, candidateRun.BaselineRunID = "candidate", "baseline"
	baseline := reportForRun(baselineRun, nil)
	candidate := reportForRun(candidateRun, nil)
	baseline.AttestationRevision = ServerAttestationRevision
	candidate.AttestationRevision = ServerAttestationRevision
	candidate.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"

	comparison, err := comparePairedReports(baseline, candidate, nil, nil)
	if err != nil || comparison.AttestationRevision != ServerAttestationRevision {
		t.Fatalf("current comparison=%+v err=%v", comparison, err)
	}
	candidate.AttestationRevision = ""
	if _, err = comparePairedReports(baseline, candidate, nil, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("unattested comparison error=%v, want ErrInvalid", err)
	}
}

func TestPairedMetricEvidenceRequiresMatchingMetricSchema(t *testing.T) {
	value := float64Pointer(0.8)
	baseline := Metric{
		ID: "routing.accuracy", Name: "Accuracy", TrackID: "routing",
		Value: value, Unit: "ratio", Direction: "higher_is_better",
	}
	tests := []struct {
		name   string
		mutate func(*Metric)
	}{
		{name: "unit", mutate: func(metric *Metric) { metric.Unit = "percent" }},
		{name: "track", mutate: func(metric *Metric) { metric.TrackID = "joint" }},
		{name: "direction", mutate: func(metric *Metric) { metric.Direction = "" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := baseline
			candidate.Value = float64Pointer(0.9)
			candidate.BaselineValue = float64Pointer(0.7)
			candidate.Delta = float64Pointer(0.2)
			test.mutate(&candidate)

			metrics, evidence := pairedMetricEvidence([]Metric{baseline}, []Metric{candidate}, nil)
			if evidence.matched != 0 || evidence.improvements != 0 || evidence.regressions != 0 {
				t.Fatalf("schema-mismatched metric produced comparison evidence: %+v", evidence)
			}
			if len(metrics) != 1 || metrics[0].BaselineValue != nil || metrics[0].Delta != nil {
				t.Fatalf("schema-mismatched metric produced a paired delta: %+v", metrics)
			}
		})
	}

	candidate := baseline
	candidate.Value = float64Pointer(0.9)
	metrics, evidence := pairedMetricEvidence([]Metric{baseline}, []Metric{candidate}, nil)
	if evidence.matched != 1 || evidence.improvements != 1 || evidence.regressions != 0 {
		t.Fatalf("matching metric schema did not produce comparison evidence: %+v", evidence)
	}
	if len(metrics) != 1 || metrics[0].BaselineValue == nil || metrics[0].Delta == nil {
		t.Fatalf("matching metric schema did not produce a paired delta: %+v", metrics)
	}
}

func TestCompareFreezesSnapshotFactorsByChangeProfile(t *testing.T) {
	selectorPolicy := digestString("comparison-selector-policy")
	mixture := &CatalogMixture{
		ID: "mom-comparison", RecipeName: "comparison",
		SelectorPolicyDigest: selectorPolicy,
		SelectorDigest:       selectorSnapshotDigest(selectorPolicy, []SupportModel{}),
		AdaptationDigest:     digestString("comparison-adaptation"),
	}
	type profileContract struct {
		profile ChangeProfile
		primary string
		allowed map[string]bool
	}
	profiles := []profileContract{
		{profile: "schema_adapter", primary: "code", allowed: map[string]bool{"code": true}},
		{profile: "recipe", primary: "policy", allowed: map[string]bool{"policy": true}},
		{profile: "selector", primary: "selector", allowed: map[string]bool{"selector": true}},
		{profile: "model_pool", primary: "pool", allowed: map[string]bool{"pool": true, "binding": true, "environment": true}},
		{profile: "runtime_capacity", primary: "environment", allowed: map[string]bool{"environment": true}},
		{profile: "online_adaptation", primary: "adaptation", allowed: map[string]bool{"adaptation": true}},
	}
	mutate := func(report Report, factor string) Report {
		if report.Run.Mixture != nil {
			copy := *report.Run.Mixture
			report.Run.Mixture = &copy
		}
		switch factor {
		case "code":
			report.Provenance.CodeRevision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
		case "policy":
			report.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"
		case "selector":
			report.Run.Mixture.SelectorDigest = digestString("candidate-selector")
		case "adaptation":
			report.Run.Mixture.AdaptationDigest = digestString("candidate-adaptation")
		case "binding":
			report.Provenance.BindingSnapshotDigest = "sha256:candidate-binding"
		case "pool":
			report.Provenance.PoolSnapshotDigest = "sha256:candidate-pool"
		case "environment":
			report.Provenance.EnvironmentSnapshotDigest = "sha256:candidate-environment"
		}
		return report
	}
	allFactors := []string{"code", "policy", "selector", "adaptation", "binding", "pool", "environment"}
	for _, profile := range profiles {
		t.Run(string(profile.profile), func(t *testing.T) {
			baselineRun := Run{
				SchemaVersion: SchemaVersion, ID: "baseline", Status: StatusCompleted,
				Mode: ModeLive, TargetID: "mom-comparison", Mixture: mixture, ChangeProfile: profile.profile,
				SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []TrackID{"routing"},
				SampleLimit: 4, Concurrency: 1, Seed: 17,
			}
			candidateRun := baselineRun
			candidateRun.ID, candidateRun.BaselineRunID = "candidate", "baseline"
			baseline := reportForRun(baselineRun, nil)
			candidate := reportForRun(candidateRun, nil)
			baseline.Metrics = []Metric{{ID: "routing.accuracy", Name: "Accuracy", Value: float64Pointer(0.8), Unit: "score", Direction: "higher_is_better"}}
			candidate.Metrics = []Metric{{ID: "routing.accuracy", Name: "Accuracy", Value: float64Pointer(0.9), Unit: "score", Direction: "higher_is_better"}}

			valid := mutate(candidate, profile.primary)
			if _, err := comparePairedReports(baseline, valid, nil, nil); err != nil {
				t.Fatalf("primary %s treatment rejected: %v", profile.primary, err)
			}
			for _, factor := range allFactors {
				if factor == profile.primary {
					continue
				}
				mixed := mutate(valid, factor)
				_, err := comparePairedReports(baseline, mixed, nil, nil)
				if profile.allowed[factor] && err != nil {
					t.Fatalf("explicit dependent %s treatment rejected: %v", factor, err)
				}
				if !profile.allowed[factor] && !errors.Is(err, ErrInvalid) {
					t.Fatalf("mixed %s treatment error=%v, want ErrInvalid", factor, err)
				}
			}
		})
	}

	agentRun := Run{
		SchemaVersion: SchemaVersion, ID: "baseline", Status: StatusCompleted,
		Mode: ModeLive, TargetID: "mom-comparison", Mixture: mixture, ChangeProfile: "agent_multimodal",
		SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []TrackID{"routing"}, SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	agentCandidateRun := agentRun
	agentCandidateRun.ID, agentCandidateRun.BaselineRunID = "candidate", "baseline"
	agentBaseline := reportForRun(agentRun, nil)
	agentCandidate := mutate(reportForRun(agentCandidateRun, nil), "policy")
	if _, err := comparePairedReports(agentBaseline, agentCandidate, nil, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "no independent") {
		t.Fatalf("agent/multimodal profile error=%v, want fail-closed factor error", err)
	}
}

func TestCompareRejectsSelfWrongBaselineLinkAndUnchangedTreatment(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	persisted, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("create self-comparison run: %v", createErr)
	}
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

	if _, err := service.CompareAs(SystemActor(), persisted.ID, persisted.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("self-comparison error=%v, want ErrInvalid", err)
	}
	if _, err := comparePairedReports(baseline, baseline, nil, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("same report comparison error=%v, want ErrInvalid", err)
	}

	wrongLink := candidate
	wrongLink.Run.BaselineRunID = "other-baseline"
	if _, err := comparePairedReports(baseline, wrongLink, nil, nil); !errors.Is(err, ErrInvalid) {
		t.Fatalf("wrong baseline lineage error=%v, want ErrInvalid", err)
	}
	if _, err := comparePairedReports(baseline, candidate, nil, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "treatment factor") {
		t.Fatalf("unchanged treatment error=%v, want treatment ErrInvalid", err)
	}

	candidate.Provenance.PoolSnapshotDigest = "sha256:candidate-pool"
	if _, err := comparePairedReports(baseline, candidate, nil, nil); err != nil {
		t.Fatalf("declared model-pool treatment was rejected: %v", err)
	}

	schemaBaselineRun := baselineRun
	schemaBaselineRun.ChangeProfile = "schema_adapter"
	schemaCandidateRun := schemaBaselineRun
	schemaCandidateRun.ID, schemaCandidateRun.BaselineRunID = "schema-candidate", schemaBaselineRun.ID
	schemaBaseline := reportForRun(schemaBaselineRun, nil)
	schemaCandidate := reportForRun(schemaCandidateRun, nil)
	if _, err := comparePairedReports(schemaBaseline, schemaCandidate, nil, nil); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "code treatment factor") {
		t.Fatalf("unchanged schema-adapter revision error=%v, want treatment ErrInvalid", err)
	}
	schemaCandidate.Provenance.CodeRevision = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	if _, err := comparePairedReports(schemaBaseline, schemaCandidate, nil, nil); err != nil {
		t.Fatalf("changed schema-adapter source revision was rejected: %v", err)
	}
}
