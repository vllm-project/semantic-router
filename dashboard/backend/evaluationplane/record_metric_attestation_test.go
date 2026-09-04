package evaluationplane

import (
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func boolPointer(value bool) *bool        { return &value }
func floatPointer(value float64) *float64 { return &value }
func int64Pointer(value int64) *int64     { return &value }

func TestRecordMetricReducerMatchesPythonGateMetricSemantics(t *testing.T) {
	reducer := newRecordMetricReducer()
	records := []executionRecordEvidence{
		{TrackID: "routing", Status: "succeeded", Quality: floatPointer(1)},
		{TrackID: "routing", Status: "failed", Quality: floatPointer(0)},
		{TrackID: "safety", Status: "succeeded", SafetyViolations: int64Pointer(2), ShouldBlock: boolPointer(true), Blocked: boolPointer(true)},
		{TrackID: "safety", Status: "failed", ShouldBlock: boolPointer(true), Blocked: boolPointer(false)},
		{TrackID: "safety", Status: "unavailable", SafetyViolations: int64Pointer(100), ShouldBlock: boolPointer(true), Blocked: boolPointer(false)},
		{TrackID: "model_pool", CaseID: "case-1", Status: "succeeded", Success: boolPointer(true), Quality: floatPointer(0.8)},
		{TrackID: "model_pool", CaseID: "case-1", Status: "failed", Success: boolPointer(false), Quality: floatPointer(1)},
		{TrackID: "model_pool", CaseID: "case-1", Status: "unavailable", Success: boolPointer(true), Quality: floatPointer(1)},
		{TrackID: "model_pool", CaseID: "case-2", Status: "succeeded", Success: boolPointer(true), Quality: floatPointer(0)},
		{TrackID: "model_pool", CaseID: "case-3", Status: "succeeded", Success: boolPointer(true), Quality: floatPointer(0.5)},
		{TrackID: "joint", CaseID: "case-1", Status: "succeeded", Success: boolPointer(true), Quality: floatPointer(0.4)},
		{TrackID: "joint", CaseID: "case-1", Status: "failed", Success: boolPointer(false), Quality: floatPointer(0.8)},
		{TrackID: "joint", CaseID: "case-2", Status: "succeeded", Success: boolPointer(true), Quality: floatPointer(0)},
		{TrackID: "joint", CaseID: "case-3", Status: "succeeded", Success: boolPointer(true), Quality: floatPointer(0.75)},
		{TrackID: "joint", CaseID: "case-1", Status: "unavailable", Quality: floatPointer(0)},
		{TrackID: "agentic", Status: "succeeded", Success: boolPointer(true)},
		{TrackID: "agentic", Status: "failed", Success: boolPointer(false)},
		{TrackID: "preference", Status: "succeeded", PreferenceMatch: boolPointer(true), BehaviorPropensity: floatPointer(0.5)},
		{TrackID: "preference", Status: "succeeded", PreferenceMatch: boolPointer(false), BehaviorPropensity: floatPointer(0.25)},
		{TrackID: "preference", Status: "succeeded", PreferenceMatch: boolPointer(true)},
		{TrackID: "preference", Status: "unavailable", PreferenceMatch: boolPointer(true), BehaviorPropensity: floatPointer(1)},
	}
	for _, record := range records {
		if err := reducer.observe(record); err != nil {
			t.Fatalf("observe(%+v): %v", record, err)
		}
	}
	attestation, err := reducer.finalize()
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	assertReducedMetric(t, attestation.RoutingAccuracy, 0.5, 2)
	assertReducedMetric(t, attestation.SafetyViolationRate, 1, 2)
	assertReducedMetric(t, attestation.SafetyBlockAccuracy, 0.5, 2)
	assertReducedMetric(t, attestation.JointNormalizedRegret, 0.5, 3)
	assertReducedMetric(t, attestation.AgenticSuccessRate, 0.5, 2)
	assertReducedMetric(t, attestation.PreferenceAgreement, 2.0/3.0, 3)
	assertReducedMetric(t, attestation.PreferencePropensity, 2.0/3.0, 3)
	assertReducedMetric(t, attestation.PreferenceEffectiveN, 1.8, 2)
	assertReducedMetric(t, attestation.PreferenceEffectiveRatio, 0.9, 2)
	assertReducedMetric(t, attestation.PreferenceIPSAgreement, 1.0/3.0, 2)
}

func TestCapacityMetricReducerCountsOnlyTypedMeasurementRows(t *testing.T) {
	reducer := newRecordMetricReducer()
	warmup, measurement := "warmup", "measurement"
	concurrency := int64(1)
	for _, record := range []executionRecordEvidence{
		{TrackID: "capacity", CaseID: "case-1", Status: "failed", Success: boolPointer(false), Concurrency: &concurrency, LoadPhase: &warmup},
		{TrackID: "capacity", CaseID: "case-1", Status: "succeeded", Success: boolPointer(true), Concurrency: &concurrency, LoadPhase: &measurement},
		{TrackID: "capacity", CaseID: "case-1", Status: "failed", Success: boolPointer(false), Concurrency: &concurrency, LoadPhase: &measurement},
	} {
		if err := reducer.observe(record); err != nil {
			t.Fatal(err)
		}
	}
	attestation, err := reducer.finalize()
	if err != nil {
		t.Fatal(err)
	}
	if attestation.CapacityRowsByCase["case-1"] != 2 ||
		len(attestation.CapacityLevelsByCase["case-1"]) != 1 {
		t.Fatalf("capacity typed measurement coverage = %#v / %#v", attestation.CapacityRowsByCase, attestation.CapacityLevelsByCase)
	}
}

func TestRecordMetricReducerRejectsUnboundedInversePropensity(t *testing.T) {
	reducer := newRecordMetricReducer()
	tiny := math.SmallestNonzeroFloat64
	err := reducer.observe(executionRecordEvidence{
		TrackID: "preference", Status: "succeeded",
		PreferenceMatch: boolPointer(true), BehaviorPropensity: &tiny,
	})
	if err == nil || !strings.Contains(err.Error(), "inverse-propensity aggregate") {
		t.Fatalf("tiny propensity error=%v, want finite aggregate rejection", err)
	}
}

func TestRecordMetricReducerMatchesPythonCanonicalOrderedRegret(t *testing.T) {
	python := os.Getenv("VLLM_SR_EVALUATION_TEST_PYTHON")
	if python == "" {
		t.Skip("set VLLM_SR_EVALUATION_TEST_PYTHON to run the cross-runtime reducer test")
	}

	reducer := newRecordMetricReducer()
	oracle := 1e-16
	largeQuality := 1.0
	zeroQuality := 0.0
	poolRecord := executionRecordEvidence{
		SchemaVersion: SchemaVersion, ID: "pool-case", AttemptID: "attempt-pool",
		TrackID: "model_pool", CaseID: "case", Status: "succeeded",
		Success: boolPointer(true), Quality: &oracle,
	}
	largeRegretRecord := executionRecordEvidence{
		SchemaVersion: SchemaVersion, ID: "joint-case-0", AttemptID: "attempt-joint-0",
		TrackID: "joint", CaseID: "case", Status: "succeeded", Quality: &largeQuality,
	}
	selectedTracks := map[TrackID]bool{"model_pool": true, "joint": true}
	caseIDs := map[string]struct{}{"case": {}}
	executor := builtinExecutorContractForTest(t, fixtureReplayExecutorID)
	if err := validateExecutionRecord(poolRecord, selectedTracks, caseIDs, executor); err != nil {
		t.Fatalf("pool boundary record is not legal: %v", err)
	}
	if err := validateExecutionRecord(largeRegretRecord, selectedTracks, caseIDs, executor); err != nil {
		t.Fatalf("joint boundary record is not legal: %v", err)
	}
	if err := reducer.observe(poolRecord); err != nil {
		t.Fatal(err)
	}
	if err := reducer.observe(largeRegretRecord); err != nil {
		t.Fatal(err)
	}
	for index := 1; index <= 99_999; index++ {
		record := executionRecordEvidence{
			SchemaVersion: SchemaVersion, ID: fmt.Sprintf("joint-case-%d", index), AttemptID: fmt.Sprintf("attempt-joint-%d", index),
			TrackID: "joint", CaseID: "case", Status: "succeeded", Quality: &zeroQuality,
		}
		if err := validateExecutionRecord(record, selectedTracks, caseIDs, executor); err != nil {
			t.Fatalf("joint record %d is not legal: %v", index, err)
		}
		if err := reducer.observe(record); err != nil {
			t.Fatal(err)
		}
	}
	attestation, err := reducer.finalize()
	if err != nil {
		t.Fatal(err)
	}
	if attestation.JointNormalizedRegret.Value == nil {
		t.Fatal("Go reducer returned an unavailable normalized regret")
	}

	pythonRoot, err := filepath.Abs("../../../src/vllm-sr")
	if err != nil {
		t.Fatal(err)
	}
	command := exec.Command(python, "-c", `
from cli.evaluation.metric_core import canonical_ordered_float_sum
oracle = 1e-16
values = [max(0.0, oracle - 1.0) / oracle]
values.extend(max(0.0, oracle - 0.0) / oracle for _ in range(99_999))
print(repr(canonical_ordered_float_sum(values) / len(values)))
`)
	command.Dir = pythonRoot
	output, err := command.Output()
	if err != nil {
		t.Fatalf("run Python canonical reducer: %v", err)
	}
	pythonValue, err := strconv.ParseFloat(strings.TrimSpace(string(output)), 64)
	if err != nil {
		t.Fatalf("parse Python canonical reducer output %q: %v", output, err)
	}
	if math.Float64bits(*attestation.JointNormalizedRegret.Value) != math.Float64bits(pythonValue) {
		t.Fatalf("Go normalized regret=%v, Python normalized regret=%v", *attestation.JointNormalizedRegret.Value, pythonValue)
	}
}

func assertReducedMetric(t *testing.T, metric reducedMetricEvidence, wantValue float64, wantSamples int) {
	t.Helper()
	if metric.Value == nil || !reducedFloatsEqual(*metric.Value, wantValue) || metric.SampleCount != wantSamples {
		t.Fatalf("metric=%+v, want value=%v samples=%d", metric, wantValue, wantSamples)
	}
}

func TestRecordMetricReducerChecksSafetyAggregateOverflow(t *testing.T) {
	reducer := newRecordMetricReducer()
	maximum := int64(^uint64(0) >> 1)
	record := executionRecordEvidence{TrackID: "safety", Status: "succeeded", SafetyViolations: &maximum}
	if err := reducer.observe(record); err != nil {
		t.Fatal(err)
	}
	if err := reducer.observe(record); err != nil {
		t.Fatal(err)
	}
	if err := reducer.observe(record); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("overflow error=%v", err)
	}
}

func TestValidateServerReducedMetricsRejectsForgedValueCountAndMetadata(t *testing.T) {
	attestation := recordMetricAttestation{
		SafetyViolationRate:   reducedMetricEvidence{Value: floatPointer(0.5), SampleCount: 2},
		SafetyBlockAccuracy:   reducedMetricEvidence{Value: floatPointer(1), ConfidenceInterval: serverWilsonInterval(2, 2), SampleCount: 2},
		JointNormalizedRegret: reducedMetricEvidence{Value: floatPointer(0.2), SampleCount: 3},
	}
	report := Report{
		Run: Run{TrackIDs: []TrackID{"safety", "joint"}},
		Metrics: []Metric{
			canonicalReducedMetric("safety.violation_rate", "Safety violation rate", "safety", "violations/case", "lower_is_better", 0.5, 2),
			canonicalReducedMetric("safety.block_accuracy", "Blocking decision accuracy", "safety", "fraction", "higher_is_better", 1, 2),
			canonicalReducedMetric("joint.normalized_regret", "Normalized pool-oracle regret", "joint", "fraction", "lower_is_better", 0.2, 3),
		},
	}
	report.Metrics[1].ConfidenceInterval = append([]float64(nil), attestation.SafetyBlockAccuracy.ConfidenceInterval...)
	if err := validateServerReducedMetrics(report, attestation); err != nil {
		t.Fatalf("canonical reduced metrics rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*Metric)
		match  string
	}{
		{name: "value", mutate: func(metric *Metric) { metric.Value = floatPointer(0.6) }, match: "value does not match"},
		{name: "availability", mutate: func(metric *Metric) { metric.Value = nil }, match: "availability"},
		{name: "sample count", mutate: func(metric *Metric) { metric.SampleCount++ }, match: "sample_count"},
		{name: "name", mutate: func(metric *Metric) { metric.Name = "Trust me" }, match: "metadata"},
		{name: "track", mutate: func(metric *Metric) { metric.TrackID = "joint" }, match: "metadata"},
		{name: "unit", mutate: func(metric *Metric) { metric.Unit = "percent" }, match: "metadata"},
		{name: "direction", mutate: func(metric *Metric) { metric.Direction = "target" }, match: "metadata"},
		{name: "confidence interval", mutate: func(metric *Metric) { metric.ConfidenceInterval = []float64{0.99, 1} }, match: "confidence_interval"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			forged := report
			forged.Metrics = append([]Metric(nil), report.Metrics...)
			test.mutate(&forged.Metrics[0])
			err := validateServerReducedMetrics(forged, attestation)
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("error=%v, want ErrInvalid containing %q", err, test.match)
			}
		})
	}
}

func TestServerOwnsG4G6G9DiagnosticMetricReduction(t *testing.T) {
	routingAccuracy := 0.75
	agenticSuccess := 0.5
	preferenceAgreement := 0.6
	propensityCoverage := 0.8
	effectiveN := 3.2
	effectiveRatio := 0.8
	ipsAgreement := 0.55
	attestation := recordMetricAttestation{
		RoutingAccuracy: reducedMetricEvidence{
			Value: &routingAccuracy, SampleCount: 4,
			ConfidenceInterval: serverWilsonInterval(3, 4),
		},
		AgenticSuccessRate: reducedMetricEvidence{
			Value: &agenticSuccess, SampleCount: 4,
			ConfidenceInterval: serverWilsonInterval(2, 4),
		},
		PreferenceAgreement: reducedMetricEvidence{
			Value: &preferenceAgreement, SampleCount: 5,
			ConfidenceInterval: serverWilsonInterval(3, 5),
		},
		PreferencePropensity: reducedMetricEvidence{
			Value: &propensityCoverage, SampleCount: 5,
			ConfidenceInterval: serverWilsonInterval(4, 5),
		},
		PreferenceEffectiveN:     reducedMetricEvidence{Value: &effectiveN, SampleCount: 4},
		PreferenceEffectiveRatio: reducedMetricEvidence{Value: &effectiveRatio, SampleCount: 4},
		PreferenceIPSAgreement:   reducedMetricEvidence{Value: &ipsAgreement, SampleCount: 4},
	}
	report := Report{
		Run: Run{TrackIDs: []TrackID{"routing", "agentic", "preference"}},
		Metrics: []Metric{
			canonicalReducedMetric("routing.accuracy", "Routing accuracy", "routing", "fraction", "higher_is_better", routingAccuracy, 4),
			canonicalReducedMetric("agentic.success_rate", "Trajectory success rate", "agentic", "fraction", "higher_is_better", agenticSuccess, 4),
			canonicalReducedMetric("preference.agreement", "Offline preference agreement", "preference", "fraction", "higher_is_better", preferenceAgreement, 5),
			canonicalReducedMetric("preference.propensity_coverage", "Behavior propensity coverage", "preference", "fraction", "higher_is_better", propensityCoverage, 5),
			canonicalReducedMetric("preference.effective_sample_size", "Inverse-propensity effective sample size", "preference", "effective samples", "higher_is_better", effectiveN, 4),
			canonicalReducedMetric("preference.effective_sample_ratio", "Effective-sample ratio", "preference", "fraction", "higher_is_better", effectiveRatio, 4),
			canonicalReducedMetric("preference.self_normalized_ips_agreement", "Self-normalized IPS agreement", "preference", "fraction", "higher_is_better", ipsAgreement, 4),
		},
	}
	report.Metrics[0].ConfidenceInterval = append([]float64(nil), attestation.RoutingAccuracy.ConfidenceInterval...)
	report.Metrics[1].ConfidenceInterval = append([]float64(nil), attestation.AgenticSuccessRate.ConfidenceInterval...)
	report.Metrics[2].ConfidenceInterval = append([]float64(nil), attestation.PreferenceAgreement.ConfidenceInterval...)
	report.Metrics[3].ConfidenceInterval = append([]float64(nil), attestation.PreferencePropensity.ConfidenceInterval...)
	if err := validateServerReducedMetrics(report, attestation); err != nil {
		t.Fatalf("canonical G4/G6/G9 diagnostics rejected: %v", err)
	}

	for index, metricID := range []string{
		"routing.accuracy", "agentic.success_rate", "preference.agreement",
		"preference.propensity_coverage", "preference.effective_sample_size",
		"preference.effective_sample_ratio", "preference.self_normalized_ips_agreement",
	} {
		t.Run(metricID, func(t *testing.T) {
			forged := report
			forged.Metrics = append([]Metric(nil), report.Metrics...)
			forgedValue := *forged.Metrics[index].Value + 0.01
			forged.Metrics[index].Value = &forgedValue
			err := validateServerReducedMetrics(forged, attestation)
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "value does not match") {
				t.Fatalf("forged %s error=%v", metricID, err)
			}
		})
	}
}

func TestValidateWorkerSingleRunMetricOwnershipRejectsComparisons(t *testing.T) {
	baseline := 0.4
	delta := 0.1
	err := validateWorkerSingleRunMetricOwnership([]Metric{{ID: "routing.accuracy", BaselineValue: &baseline, Delta: &delta}})
	if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "baseline_value or delta") {
		t.Fatalf("comparison ownership error=%v", err)
	}
}

func canonicalReducedMetric(id, name string, track TrackID, unit, direction string, value float64, samples int) Metric {
	return Metric{ID: id, Name: name, TrackID: track, Value: floatPointer(value), Unit: unit, Direction: direction, SampleCount: samples, AnalysisProvenance: validMetricAnalysisProvenanceFor(id, 0)}
}

func TestReducedFloatComparisonIsTightAndFinite(t *testing.T) {
	base := 0.2
	within := base
	for range maxReducedFloatULPs {
		within = math.Nextafter(within, math.Inf(1))
	}
	outside := math.Nextafter(within, math.Inf(1))
	if !reducedFloatsEqual(base, within) {
		t.Fatal("eight ULP difference was rejected")
	}
	if reducedFloatsEqual(base, outside) {
		t.Fatal("nine ULP difference was accepted")
	}
	if reducedFloatsEqual(math.Inf(1), math.Inf(1)) || reducedFloatsEqual(math.NaN(), math.NaN()) {
		t.Fatal("non-finite values were accepted")
	}
}
