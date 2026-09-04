package evaluationplane

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"testing"
)

type modelPoolParityFixture struct {
	SchemaVersion  string   `json:"schema_version"`
	FrozenArmIDs   []string `json:"frozen_arm_ids"`
	PlannedCaseIDs []string `json:"planned_case_ids"`
	PoolRecords    []struct {
		CaseID      string   `json:"case_id"`
		ArmID       string   `json:"arm_id"`
		Status      string   `json:"status"`
		Success     *bool    `json:"success"`
		Quality     *float64 `json:"quality"`
		RuntimeCost *float64 `json:"runtime_cost"`
	} `json:"pool_records"`
	JointRecords []struct {
		CaseID        string `json:"case_id"`
		SelectedArmID string `json:"selected_arm_id"`
		Status        string `json:"status"`
	} `json:"joint_records"`
	ExpectedMetrics []struct {
		ID                 string   `json:"id"`
		Value              *float64 `json:"value"`
		SampleCount        int      `json:"sample_count"`
		ObservedExclusions int      `json:"observed_exclusions"`
	} `json:"expected_metrics"`
}

func TestModelPoolMetricReducerMatchesGoPythonGoldenFixture(t *testing.T) {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve model-pool parity test location")
	}
	fixturePath := filepath.Join(filepath.Dir(currentFile), "../../../src/vllm-sr/tests/fixtures/model_pool_metric_parity.v1.json")
	payload, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read parity fixture: %v", err)
	}
	var fixture modelPoolParityFixture
	if decodeErr := json.Unmarshal(payload, &fixture); decodeErr != nil {
		t.Fatalf("decode parity fixture: %v", decodeErr)
	}
	if fixture.SchemaVersion != "model-pool-metric-parity.v1" {
		t.Fatalf("unexpected parity fixture schema %q", fixture.SchemaVersion)
	}
	input := modelPoolReductionInput{
		FrozenArmIDs: fixture.FrozenArmIDs, PlannedCaseIDs: fixture.PlannedCaseIDs, Authoritative: true,
	}
	for _, row := range fixture.PoolRecords {
		armID := row.ArmID
		input.PoolRecords = append(input.PoolRecords, executionRecordEvidence{
			TrackID: "model_pool", CaseID: row.CaseID, ArmID: &armID, Status: row.Status,
			Success: row.Success, Quality: row.Quality, RuntimeCost: row.RuntimeCost,
		})
	}
	for _, row := range fixture.JointRecords {
		selectedArmID := row.SelectedArmID
		input.JointRecords = append(input.JointRecords, executionRecordEvidence{
			TrackID: "joint", CaseID: row.CaseID, Status: row.Status, SelectedArmID: &selectedArmID,
		})
	}
	metrics, err := reduceAuthoritativeModelPoolMetrics(input)
	if err != nil {
		t.Fatalf("reduce parity fixture: %v", err)
	}
	if len(metrics) != len(fixture.ExpectedMetrics) {
		t.Fatalf("metric count=%d, want %d", len(metrics), len(fixture.ExpectedMetrics))
	}
	for index, want := range fixture.ExpectedMetrics {
		got := metrics[index]
		if got.ID != want.ID || got.SampleCount != want.SampleCount || (got.Value == nil) != (want.Value == nil) {
			t.Fatalf("metric[%d]=%#v, want %#v", index, got, want)
		}
		if got.Value != nil && math.Abs(*got.Value-*want.Value) > 1e-12 {
			t.Fatalf("metric[%d] value=%v, want %v", index, *got.Value, *want.Value)
		}
		exclusions := 0
		for _, count := range got.MissingReasonCounts {
			exclusions += count
		}
		if exclusions != want.ObservedExclusions {
			t.Fatalf("metric[%d] exclusions=%d, want %d", index, exclusions, want.ObservedExclusions)
		}
	}
}

func TestReduceAuthoritativeModelPoolMetricsFullDenseMatrix(t *testing.T) {
	input := modelPoolReductionInput{
		FrozenArmIDs:   []string{"bravo", "alpha"},
		PlannedCaseIDs: []string{"case-2", "case-1"},
		Authoritative:  true,
		PoolRecords: []executionRecordEvidence{
			modelPoolSucceededRecord("case-1", "alpha", true, 0.9, 2),
			modelPoolFailedRecord("case-1", "bravo", 1),
			modelPoolSucceededRecord("case-2", "alpha", true, 0.4, 2),
			modelPoolSucceededRecord("case-2", "bravo", true, 0.8, 3),
		},
		JointRecords: []executionRecordEvidence{
			modelPoolJointRecord("case-2", "bravo"),
			modelPoolJointRecord("case-1", "alpha"),
		},
	}
	metrics, err := reduceAuthoritativeModelPoolMetrics(input)
	if err != nil {
		t.Fatalf("reduce: %v", err)
	}
	assertModelPoolValue(t, metrics, "model_pool.arm_count", 2, 2)
	assertModelPoolValue(t, metrics, modelPoolArmMetricID("alpha", "quality"), 0.65, 2)
	assertModelPoolValue(t, metrics, modelPoolArmMetricID("bravo", "quality"), 0.4, 2)
	assertModelPoolValue(t, metrics, modelPoolArmMetricID("alpha", "marginal_contribution"), 0.45, 2)
	assertModelPoolValue(t, metrics, modelPoolArmMetricID("bravo", "marginal_contribution"), 0.2, 2)
	assertModelPoolValue(t, metrics, "model_pool.best_single_quality", 0.65, 2)
	assertModelPoolValue(t, metrics, "model_pool.oracle_quality", 0.85, 2)
	assertModelPoolValue(t, metrics, "model_pool.oracle_gain", 0.2, 2)
	assertModelPoolValue(t, metrics, "model_pool.unique_wins", 2, 2)
	assertModelPoolValue(t, metrics, "model_pool.unique_win_rate", 1, 2)
	assertModelPoolValue(t, metrics, "model_pool.quality_dominated_arm_count", 0, 2)
	assertModelPoolValue(t, metrics, modelPoolArmMetricID("alpha", "success_rate"), 1, 2)
	assertModelPoolValue(t, metrics, modelPoolArmMetricID("bravo", "success_rate"), 0.5, 2)
	assertModelPoolValue(t, metrics, "model_pool.worst_arm_reliability", 0.5, 2)
	assertModelPoolValue(t, metrics, "model_pool.all_arm_failure_rate", 0, 2)
	assertModelPoolValue(t, metrics, "model_pool.mean_pairwise_failure_jaccard", 0, 2)
	assertModelPoolValue(t, metrics, "model_pool.pareto_evaluable_arm_count", 2, 2)
	assertModelPoolValue(t, metrics, "model_pool.pareto_dominated_arm_count", 1, 2)
	assertModelPoolValue(t, metrics, "model_pool.selection_entropy_bits", 1, 2)
	assertModelPoolValue(t, metrics, "model_pool.selection_arm_coverage", 1, 2)
	assertModelPoolValue(t, metrics, "model_pool.quality_shared_support_cases", 2, 2)
	assertModelPoolValue(t, metrics, "model_pool.quality_cost_shared_support_fraction", 1, 2)
	assertModelPoolMetricIDsSorted(t, metrics)

	input.PoolRecords[0], input.PoolRecords[3] = input.PoolRecords[3], input.PoolRecords[0]
	input.JointRecords[0], input.JointRecords[1] = input.JointRecords[1], input.JointRecords[0]
	reordered, err := reduceAuthoritativeModelPoolMetrics(input)
	if err != nil {
		t.Fatalf("reduce reordered: %v", err)
	}
	if !reflect.DeepEqual(metrics, reordered) {
		t.Fatalf("reducer output changed with record ordering:\n%#v\n%#v", metrics, reordered)
	}
}

func TestReduceAuthoritativeModelPoolMetricsFailsClosedOnIncompleteEvidence(t *testing.T) {
	base := modelPoolReductionInput{
		FrozenArmIDs:   []string{"alpha", "bravo"},
		PlannedCaseIDs: []string{"case-1", "case-2"},
		Authoritative:  true,
		PoolRecords: []executionRecordEvidence{
			modelPoolSucceededRecord("case-1", "alpha", true, 0.7, 1),
			modelPoolSucceededRecord("case-1", "bravo", true, 0.7, 1),
			modelPoolSucceededRecord("case-2", "alpha", true, 0.7, 1),
			modelPoolSucceededRecord("case-2", "bravo", true, 0.7, 1),
		},
		JointRecords: []executionRecordEvidence{modelPoolJointRecord("case-1", "alpha"), modelPoolJointRecord("case-2", "bravo")},
	}

	t.Run("ties retain zero unique and marginal contribution", func(t *testing.T) {
		metrics, err := reduceAuthoritativeModelPoolMetrics(base)
		if err != nil {
			t.Fatalf("reduce: %v", err)
		}
		assertModelPoolValue(t, metrics, "model_pool.unique_wins", 0, 2)
		assertModelPoolValue(t, metrics, modelPoolArmMetricID("alpha", "marginal_contribution"), 0, 2)
		assertModelPoolValue(t, metrics, modelPoolArmMetricID("bravo", "marginal_contribution"), 0, 2)
	})

	t.Run("ungraded quality only removes quality and pareto metrics", func(t *testing.T) {
		input := base
		input.PoolRecords = append([]executionRecordEvidence(nil), base.PoolRecords...)
		input.PoolRecords[3].Quality = nil
		metrics, err := reduceAuthoritativeModelPoolMetrics(input)
		if err != nil {
			t.Fatalf("reduce: %v", err)
		}
		assertModelPoolUnavailable(t, metrics, "model_pool.oracle_quality", 1, map[modelPoolMissingReason]int{modelPoolUngradedSuccess: 1})
		assertModelPoolUnavailable(t, metrics, "model_pool.pareto_dominated_arm_count", 1, map[modelPoolMissingReason]int{modelPoolUngradedSuccess: 1})
		assertModelPoolValue(t, metrics, "model_pool.worst_arm_reliability", 1, 2)
		assertModelPoolValue(t, metrics, "model_pool.selection_entropy_bits", 1, 2)
	})

	t.Run("missing cost only removes cost-dependent metrics", func(t *testing.T) {
		input := base
		input.PoolRecords = append([]executionRecordEvidence(nil), base.PoolRecords...)
		input.PoolRecords[1].RuntimeCost = nil
		metrics, err := reduceAuthoritativeModelPoolMetrics(input)
		if err != nil {
			t.Fatalf("reduce: %v", err)
		}
		assertModelPoolValue(t, metrics, "model_pool.oracle_quality", 0.7, 2)
		assertModelPoolUnavailable(t, metrics, "model_pool.pareto_evaluable_arm_count", 1, map[modelPoolMissingReason]int{modelPoolMissingRuntimeCost: 1})
	})

	t.Run("missing joint selection only removes selection metrics", func(t *testing.T) {
		input := base
		input.JointRecords = input.JointRecords[:1]
		metrics, err := reduceAuthoritativeModelPoolMetrics(input)
		if err != nil {
			t.Fatalf("reduce: %v", err)
		}
		assertModelPoolUnavailable(t, metrics, "model_pool.selection_entropy_bits", 1, map[modelPoolMissingReason]int{modelPoolMissingSelection: 1})
		assertModelPoolValue(t, metrics, "model_pool.oracle_quality", 0.7, 2)
	})

	t.Run("missing arm cell fail closes every dense matrix axis", func(t *testing.T) {
		input := base
		input.PoolRecords = input.PoolRecords[:3]
		metrics, err := reduceAuthoritativeModelPoolMetrics(input)
		if err != nil {
			t.Fatalf("reduce: %v", err)
		}
		assertModelPoolUnavailable(t, metrics, "model_pool.oracle_quality", 1, map[modelPoolMissingReason]int{modelPoolMissingArmCell: 1})
		assertModelPoolUnavailable(t, metrics, "model_pool.worst_arm_reliability", 0, map[modelPoolMissingReason]int{modelPoolMissingArmCell: 1})
	})
}

func TestReduceAuthoritativeModelPoolMetricsRejectsAmbiguousEvidence(t *testing.T) {
	base := modelPoolReductionInput{
		FrozenArmIDs:   []string{"alpha", "bravo"},
		PlannedCaseIDs: []string{"case-1"},
		Authoritative:  true,
		PoolRecords: []executionRecordEvidence{
			modelPoolSucceededRecord("case-1", "alpha", true, 1, 1),
			modelPoolSucceededRecord("case-1", "bravo", true, 1, 1),
		},
		JointRecords: []executionRecordEvidence{modelPoolJointRecord("case-1", "alpha")},
	}
	duplicate := base
	duplicate.PoolRecords = append(duplicate.PoolRecords, duplicate.PoolRecords[0])
	if _, err := reduceAuthoritativeModelPoolMetrics(duplicate); err == nil {
		t.Fatal("duplicate case-arm coordinate was accepted")
	}
	duplicateJoint := base
	duplicateJoint.JointRecords = append(duplicateJoint.JointRecords, duplicateJoint.JointRecords[0])
	if _, err := reduceAuthoritativeModelPoolMetrics(duplicateJoint); err == nil {
		t.Fatal("duplicate joint evidence was accepted")
	}
	outOfScopeJoint := base
	outOfScopeJoint.JointRecords = append(outOfScopeJoint.JointRecords, executionRecordEvidence{TrackID: "joint", CaseID: "case-outside", Status: "succeeded"})
	if _, err := reduceAuthoritativeModelPoolMetrics(outOfScopeJoint); err == nil {
		t.Fatal("out-of-scope joint evidence was accepted")
	}
	nonAuthoritative := base
	nonAuthoritative.Authoritative = false
	metrics, err := reduceAuthoritativeModelPoolMetrics(nonAuthoritative)
	if err != nil {
		t.Fatalf("non-authoritative reduce: %v", err)
	}
	for _, metric := range metrics {
		if metric.Value != nil || !reflect.DeepEqual(metric.MissingReasonCounts, map[modelPoolMissingReason]int{modelPoolNonAuthoritative: 1}) {
			t.Fatalf("non-authoritative metric was not unavailable: %#v", metric)
		}
	}
}

func TestCanonicalModelPoolMetricIDsAreExact(t *testing.T) {
	for _, id := range modelPoolStaticMetricIDs {
		if !isCanonicalModelPoolMetricID(id) {
			t.Fatalf("static ID is not recognized: %q", id)
		}
	}
	for _, id := range []string{
		modelPoolArmMetricID("alpha", "quality"),
		modelPoolArmMetricID("alpha", "success_rate"),
		modelPoolArmMetricID("alpha", "marginal_contribution"),
	} {
		if !isCanonicalModelPoolMetricID(id) {
			t.Fatalf("dynamic ID is not recognized: %q", id)
		}
	}
	dotted := modelPoolArmMetricID("team.a", "quality")
	armID, measure, ok := parseCanonicalModelPoolArmMetricID(dotted)
	if !ok || armID != "team.a" || measure != "quality" {
		t.Fatalf("arm dimension did not round trip: id=%q arm=%q measure=%q ok=%t", dotted, armID, measure, ok)
	}
	for _, id := range []string{
		"model_pool.arm.alpha.oracle_quality",
		"prefix." + modelPoolArmMetricID("alpha", "quality"),
		"model_pool.arm..quality",
		"model_pool.arm.team.a.quality",
		modelPoolArmMetricID("alpha", "quality") + ".extra",
	} {
		if isCanonicalModelPoolMetricID(id) {
			t.Fatalf("non-canonical ID was accepted: %q", id)
		}
	}
}

func TestValidateServerReducedModelPoolMetricsRequiresExactEvidenceSet(t *testing.T) {
	evidence, err := reduceAuthoritativeModelPoolMetrics(modelPoolReductionInput{
		FrozenArmIDs:   []string{"alpha", "team.a"},
		PlannedCaseIDs: []string{"case-1"},
		Authoritative:  true,
		PoolRecords: []executionRecordEvidence{
			modelPoolSucceededRecord("case-1", "alpha", true, 0.8, 1),
			modelPoolSucceededRecord("case-1", "team.a", true, 0.6, 2),
		},
		JointRecords: []executionRecordEvidence{modelPoolJointRecord("case-1", "alpha")},
	})
	if err != nil {
		t.Fatalf("reduce: %v", err)
	}
	report := Report{}
	for _, metric := range evidence {
		exclusions := 0
		for _, count := range metric.MissingReasonCounts {
			exclusions += count
		}
		report.Metrics = append(report.Metrics, Metric{
			ID: metric.ID, TrackID: "model_pool", Value: metric.Value, SampleCount: metric.SampleCount,
			AnalysisProvenance: MetricAnalysisProvenance{ObservedExclusions: &exclusions},
		})
	}
	if err := validateServerReducedModelPoolMetrics(report, evidence); err != nil {
		t.Fatalf("exact server evidence rejected: %v", err)
	}
	report.Metrics = report.Metrics[1:]
	if err := validateServerReducedModelPoolMetrics(report, evidence); err == nil {
		t.Fatal("missing canonical model-pool metric was accepted")
	}
}

func modelPoolSucceededRecord(caseID, armID string, success bool, quality, cost float64) executionRecordEvidence {
	return executionRecordEvidence{TrackID: "model_pool", CaseID: caseID, Status: "succeeded", ArmID: &armID, Success: &success, Quality: &quality, RuntimeCost: &cost}
}

func modelPoolFailedRecord(caseID, armID string, cost float64) executionRecordEvidence {
	return executionRecordEvidence{TrackID: "model_pool", CaseID: caseID, Status: "failed", ArmID: &armID, RuntimeCost: &cost}
}

func modelPoolJointRecord(caseID, selectedArmID string) executionRecordEvidence {
	return executionRecordEvidence{TrackID: "joint", CaseID: caseID, Status: "succeeded", SelectedArmID: &selectedArmID}
}

func assertModelPoolValue(t *testing.T, metrics []modelPoolMetricEvidence, id string, want float64, sampleCount int) {
	t.Helper()
	metric := findModelPoolMetric(t, metrics, id)
	if metric.Value == nil || math.Abs(*metric.Value-want) > 1e-12 || metric.SampleCount != sampleCount || len(metric.MissingReasonCounts) != 0 {
		t.Fatalf("%s = %#v, want value %v, samples %d, no reasons", id, metric, want, sampleCount)
	}
}

func assertModelPoolUnavailable(t *testing.T, metrics []modelPoolMetricEvidence, id string, sampleCount int, reasons map[modelPoolMissingReason]int) {
	t.Helper()
	metric := findModelPoolMetric(t, metrics, id)
	if metric.Value != nil || metric.SampleCount != sampleCount || !reflect.DeepEqual(metric.MissingReasonCounts, reasons) {
		t.Fatalf("%s = %#v, want unavailable samples %d reasons %#v", id, metric, sampleCount, reasons)
	}
}

func findModelPoolMetric(t *testing.T, metrics []modelPoolMetricEvidence, id string) modelPoolMetricEvidence {
	t.Helper()
	for _, metric := range metrics {
		if metric.ID == id {
			return metric
		}
	}
	t.Fatalf("metric %q was not emitted", id)
	return modelPoolMetricEvidence{}
}

func assertModelPoolMetricIDsSorted(t *testing.T, metrics []modelPoolMetricEvidence) {
	t.Helper()
	ids := make([]string, len(metrics))
	for index, metric := range metrics {
		ids[index] = metric.ID
	}
	if !sort.StringsAreSorted(ids) {
		t.Fatalf("metric IDs are not sorted: %v", ids)
	}
}
