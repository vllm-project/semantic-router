package evaluationplane

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func pairedRecord(id string, track TrackID, caseID, attemptID string, armID *string, quality float64, success bool) executionRecordEvidence {
	return executionRecordEvidence{
		SchemaVersion: SchemaVersion, ID: id, TrackID: track, CaseID: caseID,
		AttemptID: attemptID, Status: "succeeded", ArmID: armID,
		Quality: float64Pointer(quality), Success: &success,
	}
}

func pairedStatisticByID(t *testing.T, results []ComparisonStatistic, metricID string) ComparisonStatistic {
	t.Helper()
	for _, result := range results {
		if result.ID == metricID {
			return result
		}
	}
	t.Fatalf("paired statistic %q is absent", metricID)
	return ComparisonStatistic{}
}

func TestPairedOracleUsesPerCaseMaximumInsteadOfArmMean(t *testing.T) {
	baseline := make([]executionRecordEvidence, 0, 200)
	candidate := make([]executionRecordEvidence, 0, 200)
	for index := range 100 {
		caseID := fmt.Sprintf("case-%03d", index)
		for armIndex, armID := range []string{"arm-a", "arm-b"} {
			id := caseID + "-" + armID
			oldQuality := []float64{0.9, 0.1}[armIndex]
			newQuality := []float64{0.8, 0.8}[armIndex]
			baseline = append(baseline, pairedRecord(id, "model_pool", caseID, "attempt-1", &armID, oldQuality, true))
			candidate = append(candidate, pairedRecord(id, "model_pool", caseID, "attempt-1", &armID, newQuality, true))
		}
	}
	results, err := computePairedStatistics(baseline, candidate, 17)
	if err != nil {
		t.Fatalf("computePairedStatistics: %v", err)
	}
	oracle := pairedStatisticByID(t, results, "model_pool.oracle_quality")
	if oracle.SampleCount != 100 || math.Abs(oracle.Delta+0.1) > 1e-12 || len(oracle.DeltaConfidenceInterval) != 2 || oracle.DeltaConfidenceInterval[1] >= 0 {
		t.Fatalf("oracle statistic used the wrong analysis unit: %+v", oracle)
	}
}

func TestPairedModelPoolAllowsChangedArmSetOnAlignedCases(t *testing.T) {
	baseline := make([]executionRecordEvidence, 0, comparisonMinimumAnalysisUnits)
	candidate := make([]executionRecordEvidence, 0, comparisonMinimumAnalysisUnits*2)
	baselineArm, candidateArmA, candidateArmB := "incumbent", "specialist-a", "specialist-b"
	for index := range comparisonMinimumAnalysisUnits {
		caseID := fmt.Sprintf("case-%03d", index)
		baseline = append(baseline,
			pairedRecord("baseline-"+caseID, "model_pool", caseID, "pool", &baselineArm, 0.7, true),
		)
		candidate = append(candidate,
			pairedRecord("candidate-a-"+caseID, "model_pool", caseID, "pool", &candidateArmA, 0.6, true),
			pairedRecord("candidate-b-"+caseID, "model_pool", caseID, "pool", &candidateArmB, 0.9, true),
		)
	}
	statistics, err := computePairedStatistics(baseline, candidate, 17)
	if err != nil {
		t.Fatalf("changed model-pool arm set rejected: %v", err)
	}
	oracle := pairedStatisticByID(t, statistics, "model_pool.oracle_quality")
	if oracle.SampleCount != comparisonMinimumAnalysisUnits || math.Abs(oracle.BaselineValue-0.7) > 1e-12 ||
		math.Abs(oracle.CandidateValue-0.9) > 1e-12 || oracle.Verdict != "pass" {
		t.Fatalf("changed-arm feasible oracle was reduced incorrectly: %+v", oracle)
	}
}

func TestPairedNormalizedRegretRegressionFails(t *testing.T) {
	baseline := make([]executionRecordEvidence, 0, 200)
	candidate := make([]executionRecordEvidence, 0, 200)
	armID := "arm-a"
	for index := range 100 {
		caseID := fmt.Sprintf("case-%03d", index)
		baseline = append(baseline,
			pairedRecord(caseID+"-pool", "model_pool", caseID, "pool", &armID, 0.5, true),
			pairedRecord(caseID+"-joint", "joint", caseID, "joint", nil, 0.5, true),
		)
		candidate = append(candidate,
			pairedRecord(caseID+"-pool", "model_pool", caseID, "pool", &armID, 1.0, true),
			pairedRecord(caseID+"-joint", "joint", caseID, "joint", nil, 0.8, true),
		)
	}
	results, err := computePairedStatistics(baseline, candidate, 17)
	if err != nil {
		t.Fatalf("computePairedStatistics: %v", err)
	}
	regret := pairedStatisticByID(t, results, "joint.normalized_regret")
	if math.Abs(regret.Delta-0.2) > 1e-12 || len(regret.DeltaConfidenceInterval) != 2 || regret.DeltaConfidenceInterval[0] <= 0 {
		t.Fatalf("normalized regret regression was not detected: %+v", regret)
	}
}

func TestPairedPoolAndJointQualityTreatFailuresAsZero(t *testing.T) {
	baseline := make([]executionRecordEvidence, 0, comparisonMinimumAnalysisUnits*3)
	candidate := make([]executionRecordEvidence, 0, comparisonMinimumAnalysisUnits*3)
	armA, armB := "arm-a", "arm-b"
	failed := false
	for index := range comparisonMinimumAnalysisUnits {
		caseID := fmt.Sprintf("case-%03d", index)
		baseline = append(baseline,
			pairedRecord(caseID+"-a", "model_pool", caseID, "pool-a", &armA, 0.5, true),
			pairedRecord(caseID+"-b", "model_pool", caseID, "pool-b", &armB, 0.4, true),
			pairedRecord(caseID+"-joint", "joint", caseID, "joint", nil, 0.5, true),
		)
		candidate = append(candidate,
			executionRecordEvidence{
				SchemaVersion: SchemaVersion, ID: caseID + "-a", TrackID: "model_pool", CaseID: caseID,
				AttemptID: "pool-a", Status: "failed", ArmID: &armA, Success: &failed,
			},
			pairedRecord(caseID+"-b", "model_pool", caseID, "pool-b", &armB, 0.4, true),
			executionRecordEvidence{
				SchemaVersion: SchemaVersion, ID: caseID + "-joint", TrackID: "joint", CaseID: caseID,
				AttemptID: "joint", Status: "failed", Success: &failed,
			},
		)
	}
	statistics, err := computePairedStatistics(baseline, candidate, 17)
	if err != nil {
		t.Fatal(err)
	}
	oracle := pairedStatisticByID(t, statistics, "model_pool.oracle_quality")
	realized := pairedStatisticByID(t, statistics, "joint.realized_quality")
	regret := pairedStatisticByID(t, statistics, "joint.normalized_regret")
	if oracle.SampleCount != comparisonMinimumAnalysisUnits || math.Abs(oracle.CandidateValue-0.4) > 1e-12 ||
		math.Abs(oracle.Delta+0.1) > 1e-12 || math.Abs(realized.CandidateValue) > 1e-12 ||
		math.Abs(realized.Delta+0.5) > 1e-12 || math.Abs(regret.CandidateValue-1) > 1e-12 ||
		math.Abs(regret.Delta-1) > 1e-12 {
		t.Fatalf("failed outcomes escaped paired quality cohort: oracle=%+v realized=%+v regret=%+v", oracle, realized, regret)
	}
}

func TestPairedRegretNeverTreatsSampledOracleNoiseAsNegativeShortfall(t *testing.T) {
	armID := "arm-a"
	baseline := []executionRecordEvidence{
		pairedRecord("pool", "model_pool", "case", "pool", &armID, 0.8, true),
		pairedRecord("joint", "joint", "case", "joint", nil, 0.9, true),
	}
	candidate := append([]executionRecordEvidence(nil), baseline...)
	statistics, err := computePairedStatistics(baseline, candidate, 17)
	if err != nil {
		t.Fatal(err)
	}
	regret := pairedStatisticByID(t, statistics, "joint.normalized_regret")
	if regret.BaselineValue != 0 || regret.CandidateValue != 0 || regret.Delta != 0 {
		t.Fatalf("sampled oracle noise created negative normalized regret: %+v", regret)
	}
}

func TestPairedRecordsRejectIdentityMismatch(t *testing.T) {
	baseline := []executionRecordEvidence{pairedRecord("record-1", "routing", "case-1", "attempt-1", nil, 0.5, true)}
	candidate := []executionRecordEvidence{pairedRecord("record-1", "routing", "case-2", "attempt-1", nil, 0.6, true)}
	if _, err := computePairedStatistics(baseline, candidate, 17); err == nil || !strings.Contains(err.Error(), "identities") {
		t.Fatalf("identity mismatch error=%v", err)
	}
}

func TestQualifiedPairedIntervalsProducePassAndFail(t *testing.T) {
	baselineRun := Run{
		SchemaVersion: SchemaVersion, ID: "baseline", Status: StatusCompleted, Mode: ModeReplay,
		TargetID: "fixture", ChangeProfile: "recipe", SuiteIDs: []string{"suite"},
		TrackIDs: []TrackID{"routing"}, SampleLimit: 40, Concurrency: 1, Seed: 17, EvidenceLevel: "E3",
	}
	candidateRun := baselineRun
	candidateRun.ID, candidateRun.BaselineRunID = "candidate", baselineRun.ID
	baselineReport, candidateReport := reportForRun(baselineRun, nil), reportForRun(candidateRun, nil)
	baselineReport.Provenance.BenchmarkRevisions = map[string]string{"suite": "revision"}
	candidateReport.Provenance.BenchmarkRevisions = baselineReport.Provenance.BenchmarkRevisions
	candidateReport.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"
	for _, report := range []*Report{&baselineReport, &candidateReport} {
		report.Tracks = []TrackReport{{
			TrackID: "routing", Status: "completed", EvidenceLevel: "E3",
			Coverage: Coverage{Evaluated: 40, Total: 40, Fraction: 1}, Metrics: []Metric{}, Gates: []Gate{},
		}}
		report.Metrics = []Metric{{
			ID: "routing.accuracy", Name: "Accuracy", TrackID: "routing",
			Value: float64Pointer(0), Unit: "fraction", Direction: "higher_is_better",
		}}
	}
	baselineRecords := make([]executionRecordEvidence, 0, 40)
	candidateRecords := make([]executionRecordEvidence, 0, 40)
	for index := range 40 {
		caseID := fmt.Sprintf("case-%03d", index)
		baselineRecords = append(baselineRecords, pairedRecord(caseID, "routing", caseID, "attempt", nil, 0.5, true))
		candidateRecords = append(candidateRecords, pairedRecord(caseID, "routing", caseID, "attempt", nil, 0.7, true))
	}
	baselineReport.Metrics[0].Value = float64Pointer(0.5)
	candidateReport.Metrics[0].Value = float64Pointer(0.7)
	comparison, err := comparePairedReports(baselineReport, candidateReport, baselineRecords, candidateRecords)
	if err != nil || comparison.Verdict != "pass" {
		t.Fatalf("qualified paired improvement comparison=%+v err=%v", comparison, err)
	}
	for index := range candidateRecords {
		candidateRecords[index].Quality = float64Pointer(0.3)
	}
	candidateReport.Metrics[0].Value = float64Pointer(0.3)
	comparison, err = comparePairedReports(baselineReport, candidateReport, baselineRecords, candidateRecords)
	if err != nil || comparison.Verdict != "fail" {
		t.Fatalf("qualified paired regression comparison=%+v err=%v", comparison, err)
	}
}

func comparativeG3Fixture(caseCount int, baselineRealized, candidateRealized float64) (Report, Report, []executionRecordEvidence, []executionRecordEvidence) {
	baselineRun := Run{
		SchemaVersion: SchemaVersion, ID: "baseline", Status: StatusCompleted, Mode: ModeReplay,
		TargetID: "fixture", ChangeProfile: "recipe", SuiteIDs: []string{"suite"},
		TrackIDs: []TrackID{"model_pool", "joint"}, SampleLimit: caseCount, Concurrency: 1, Seed: 17, EvidenceLevel: "E4",
	}
	candidateRun := baselineRun
	candidateRun.ID, candidateRun.BaselineRunID = "candidate", baselineRun.ID
	baseline, candidate := reportForRun(baselineRun, nil), reportForRun(candidateRun, nil)
	for _, report := range []*Report{&baseline, &candidate} {
		report.Provenance.BenchmarkRevisions = map[string]string{"suite": "revision"}
		report.Tracks = []TrackReport{
			{TrackID: "model_pool", Status: "completed", EvidenceLevel: "E4", Coverage: Coverage{Evaluated: caseCount, Total: caseCount, Fraction: 1}, Metrics: []Metric{}, Gates: []Gate{}},
			{TrackID: "joint", Status: "completed", EvidenceLevel: "E4", Coverage: Coverage{Evaluated: caseCount, Total: caseCount, Fraction: 1}, Metrics: []Metric{}, Gates: []Gate{}},
		}
	}
	candidate.Provenance.PolicySnapshotDigest = "sha256:candidate-policy"
	forgedObserved, forgedSamples := 0.0, caseCount
	candidate.Gates = []Gate{{
		ID: "G3", Name: "Offline value", TrackID: "joint", Disposition: "required", Verdict: "pass",
		ChangeProfile: "recipe", ContractVersion: GateContractVersion,
		EvidenceRefs: []string{"worker-forged:g3"}, EvidenceLevel: "E4", Observed: &forgedObserved,
		Threshold:   &GateThreshold{Operator: "<=", Value: defaultNormalizedRegretMaximum, Unit: "fraction"},
		SampleCount: &forgedSamples, Owner: "recipe-and-model-pool", Rationale: "Worker claim must be ignored.",
	}}
	armID := "arm-a"
	baselineRecords := make([]executionRecordEvidence, 0, caseCount*2)
	candidateRecords := make([]executionRecordEvidence, 0, caseCount*2)
	for index := range caseCount {
		caseID := fmt.Sprintf("case-%03d", index)
		baselineRecords = append(baselineRecords,
			pairedRecord(caseID+"-pool", "model_pool", caseID, "pool", &armID, 1, true),
			pairedRecord(caseID+"-joint", "joint", caseID, "joint", nil, baselineRealized, true),
		)
		candidateRecords = append(candidateRecords,
			pairedRecord(caseID+"-pool", "model_pool", caseID, "pool", &armID, 1, true),
			pairedRecord(caseID+"-joint", "joint", caseID, "joint", nil, candidateRealized, true),
		)
	}
	return baseline, candidate, baselineRecords, candidateRecords
}

func TestComparativeG3IsServerReducedAndSmallSamplesCannotPass(t *testing.T) {
	baseline, candidate, baselineRecords, candidateRecords := comparativeG3Fixture(2, 0.8, 0.8)
	comparison, err := comparePairedReports(baseline, candidate, baselineRecords, candidateRecords)
	if err != nil {
		t.Fatal(err)
	}
	g3, found := reportGateFromSlice(comparison.Gates, "G3")
	if !found || g3.Verdict != "unavailable" || g3.Observed != nil || g3.Threshold != nil ||
		g3.SampleCount == nil || *g3.SampleCount != 2 || g3.EvidenceRefs[0] != comparativeG3ReductionRef ||
		comparison.Verdict != "unavailable" {
		t.Fatalf("small comparative G3=%+v comparison=%+v", g3, comparison)
	}
}

func TestComparativeG3KeepsSyntheticReplayDiagnosticAcrossRegretBoundaries(t *testing.T) {
	tests := []struct {
		name                                string
		baselineRealized, candidateRealized float64
	}{
		{name: "qualified improvement", baselineRealized: 0.8, candidateRealized: 0.9},
		{name: "absolute regret failure", baselineRealized: 0.8, candidateRealized: 0.5},
		{name: "relative frontier failure", baselineRealized: 0.9, candidateRealized: 0.8},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			baseline, candidate, baselineRecords, candidateRecords := comparativeG3Fixture(
				comparisonMinimumAnalysisUnits, test.baselineRealized, test.candidateRealized,
			)
			comparison, err := comparePairedReports(baseline, candidate, baselineRecords, candidateRecords)
			if err != nil {
				t.Fatal(err)
			}
			g3, found := reportGateFromSlice(comparison.Gates, "G3")
			if !found || g3.Verdict != "unavailable" || g3.EvidenceLevel != "E0" ||
				g3.Observed != nil || g3.Threshold != nil || g3.SampleCount == nil ||
				*g3.SampleCount != comparisonMinimumAnalysisUnits {
				t.Fatalf("G3=%+v comparison=%+v", g3, comparison)
			}
		})
	}
}
