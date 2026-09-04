package evaluationplane

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"time"
)

const aggregateLatencyRegressionBudget = 0.05

func comparePairedReports(baseline, candidate Report) (Comparison, error) {
	if err := validatePairedReportCohort(baseline, candidate); err != nil {
		return Comparison{}, err
	}
	metrics, evidence := pairedMetricEvidence(baseline.Metrics, candidate.Metrics)
	verdict, reason := comparisonVerdict(candidate, evidence, baseline.Summary, candidate.Summary)
	recommendations := comparisonRecommendations(verdict, evidence)
	return Comparison{
		SchemaVersion: SchemaVersion, BaselineRunID: baseline.Run.ID, CandidateRunID: candidate.Run.ID,
		Verdict: verdict,
		Summary: fmt.Sprintf(
			"Compared %d metrics (%d matched aggregate deltas): %d improved, %d regressed. %s",
			len(metrics), evidence.matched, evidence.improvements, evidence.regressions, reason,
		),
		Metrics: metrics, Gates: candidate.Gates, Recommendations: recommendations,
		CreatedAt: time.Now().UTC(),
	}, nil
}

func validatePairedReportCohort(baseline, candidate Report) error {
	if baseline.Run.ID == candidate.Run.ID {
		return fmt.Errorf("%w: baseline and candidate runs must be distinct", ErrInvalid)
	}
	if candidate.Run.BaselineRunID != baseline.Run.ID {
		return fmt.Errorf("%w: candidate baseline_run_id must identify the compared baseline", ErrInvalid)
	}
	if baseline.Run.Mode != candidate.Run.Mode || baseline.Run.TargetID != candidate.Run.TargetID ||
		baseline.Run.ChangeProfile != candidate.Run.ChangeProfile ||
		baseline.Run.SampleLimit != candidate.Run.SampleLimit || baseline.Run.Concurrency != candidate.Run.Concurrency || baseline.Run.Seed != candidate.Run.Seed ||
		!sameStringSet(baseline.Run.SuiteIDs, candidate.Run.SuiteIDs) || !sameTrackSet(baseline.Run.TrackIDs, candidate.Run.TrackIDs) {
		return fmt.Errorf("%w: baseline and candidate report cohorts do not match", ErrInvalid)
	}
	if !validChangeProfile(baseline.Run.ChangeProfile) {
		return fmt.Errorf("%w: baseline and candidate change_profile is invalid", ErrInvalid)
	}
	if baseline.Provenance.WorkloadSnapshotDigest == "" ||
		baseline.Provenance.WorkloadSnapshotDigest != candidate.Provenance.WorkloadSnapshotDigest {
		return fmt.Errorf("%w: baseline and candidate workload snapshots must match", ErrInvalid)
	}
	if err := validateTreatmentFactors(baseline, candidate); err != nil {
		return err
	}
	if len(baseline.Provenance.BenchmarkRevisions) == 0 ||
		!reflect.DeepEqual(candidate.Provenance.BenchmarkRevisions, baseline.Provenance.BenchmarkRevisions) {
		return fmt.Errorf("%w: baseline and candidate benchmark revisions must match", ErrInvalid)
	}
	return nil
}

type treatmentFactors struct {
	policy      bool
	binding     bool
	pool        bool
	environment bool
}

// comparisonTreatment returns the only snapshot factors that may change for
// each v1 profile. All other factors are frozen and must match exactly.
func comparisonTreatment(profile ChangeProfile) treatmentFactors {
	switch profile {
	case "recipe", "selector", "agent_multimodal", "online_adaptation":
		return treatmentFactors{policy: true, binding: true}
	case "model_pool":
		return treatmentFactors{binding: true, pool: true}
	case "runtime_capacity":
		return treatmentFactors{environment: true}
	default:
		return treatmentFactors{}
	}
}

func validateTreatmentFactors(baseline, candidate Report) error {
	allowed := comparisonTreatment(baseline.Run.ChangeProfile)
	if baseline.Run.ChangeProfile == "schema_adapter" {
		if baseline.Provenance.CodeRevision == "" || candidate.Provenance.CodeRevision == "" ||
			baseline.Provenance.CodeRevision == candidate.Provenance.CodeRevision {
			return fmt.Errorf("%w: change_profile %q requires the source code revision treatment to change", ErrInvalid, baseline.Run.ChangeProfile)
		}
	}
	factors := []struct {
		name                string
		baseline, candidate string
		mayChange           bool
	}{
		{"policy", baseline.Provenance.PolicySnapshotDigest, candidate.Provenance.PolicySnapshotDigest, allowed.policy},
		{"binding", baseline.Provenance.BindingSnapshotDigest, candidate.Provenance.BindingSnapshotDigest, allowed.binding},
		{"pool", baseline.Provenance.PoolSnapshotDigest, candidate.Provenance.PoolSnapshotDigest, allowed.pool},
		{"environment", baseline.Provenance.EnvironmentSnapshotDigest, candidate.Provenance.EnvironmentSnapshotDigest, allowed.environment},
	}
	changedTreatment := false
	hasTreatment := false
	for _, factor := range factors {
		if factor.baseline == "" || factor.candidate == "" {
			return fmt.Errorf("%w: baseline and candidate %s snapshot identities are required", ErrInvalid, factor.name)
		}
		if !factor.mayChange && factor.candidate != factor.baseline {
			return fmt.Errorf("%w: baseline and candidate %s snapshots must match for change_profile %q", ErrInvalid, factor.name, baseline.Run.ChangeProfile)
		}
		if factor.mayChange {
			hasTreatment = true
			changedTreatment = changedTreatment || factor.candidate != factor.baseline
		}
	}
	if hasTreatment && !changedTreatment {
		return fmt.Errorf("%w: change_profile %q requires at least one declared treatment factor to change", ErrInvalid, baseline.Run.ChangeProfile)
	}
	return nil
}

type comparisonEvidence struct {
	matched           int
	improvements      int
	regressions       int
	primaryRegression bool
	latencyOverBudget bool
}

func pairedMetricEvidence(baseline, candidate []Metric) ([]Metric, comparisonEvidence) {
	baselineByID := make(map[string]Metric, len(baseline))
	for _, metric := range baseline {
		baselineByID[metric.ID] = metric
	}
	metrics := make([]Metric, 0, len(candidate))
	evidence := comparisonEvidence{}
	for _, metric := range candidate {
		old, ok := baselineByID[metric.ID]
		if !ok || old.Value == nil || metric.Value == nil ||
			(old.Direction != "" && metric.Direction != "" && old.Direction != metric.Direction) {
			metrics = append(metrics, metric)
			continue
		}
		baselineValue := *old.Value
		delta := *metric.Value - baselineValue
		metric.BaselineValue, metric.Delta = &baselineValue, &delta
		evidence.matched++
		if delta != 0 && metric.Direction != "target" && metric.Direction != "" {
			if metricImproved(metric.Direction, delta) {
				evidence.improvements++
			} else {
				evidence.regressions++
				if primaryPromotionMetric(metric.ID) {
					evidence.primaryRegression = true
				}
			}
		}
		if latencyMetric(metric.ID) && exceedsRegressionBudget(baselineValue, *metric.Value, aggregateLatencyRegressionBudget) {
			evidence.latencyOverBudget = true
		}
		metrics = append(metrics, metric)
	}
	return metrics, evidence
}

func comparisonVerdict(candidate Report, evidence comparisonEvidence, baseline, current ReportSummary) (GateVerdict, string) {
	requiredUnavailable := false
	for _, gate := range candidate.Gates {
		if gate.Disposition != "required" {
			continue
		}
		if gate.Verdict == "fail" {
			return "fail", "A required candidate gate failed."
		}
		if gate.Verdict == "unavailable" {
			requiredUnavailable = true
		}
	}
	if current.Verdict == "fail" {
		return "fail", "The candidate report failed."
	}
	if evidence.primaryRegression {
		return "fail", "A primary quality or joint-system metric regressed."
	}
	if summaryQualityRegressed(baseline, current) {
		return "fail", "A primary quality or joint-system metric regressed."
	}
	if evidence.latencyOverBudget || summaryLatencyOverBudget(baseline, current) {
		return "fail", "Tail latency exceeded the 5% aggregate regression budget."
	}
	if current.Verdict == "unavailable" || requiredUnavailable {
		return "unavailable", "Required candidate evidence is unavailable."
	}
	if evidence.matched == 0 {
		return "unavailable", "No matched direction-aware aggregate metric evidence is available."
	}
	return "unavailable", "Aggregate point deltas are descriptive only; case-level paired deltas and their confidence interval are unavailable."
}

func metricImproved(direction string, delta float64) bool {
	if direction == "lower_is_better" {
		return delta < 0
	}
	return delta > 0
}

func primaryPromotionMetric(id string) bool {
	switch id {
	case "routing.accuracy", "model_pool.oracle_quality", "joint.realized_quality",
		"joint.oracle_regret", "joint.normalized_regret", "joint.reliability":
		return true
	default:
		return false
	}
}

func latencyMetric(id string) bool {
	return strings.Contains(strings.ToLower(id), "latency")
}

func exceedsRegressionBudget(baseline, candidate, budget float64) bool {
	if math.IsNaN(baseline) || math.IsNaN(candidate) || math.IsInf(baseline, 0) || math.IsInf(candidate, 0) {
		return true
	}
	if baseline <= 0 {
		return candidate > baseline
	}
	return candidate > baseline*(1+budget)
}

func summaryQualityRegressed(baseline, candidate ReportSummary) bool {
	return baseline.QualityScore != nil && candidate.QualityScore != nil && *candidate.QualityScore < *baseline.QualityScore
}

func summaryLatencyOverBudget(baseline, candidate ReportSummary) bool {
	return baseline.LatencyP95MS != nil && candidate.LatencyP95MS != nil &&
		exceedsRegressionBudget(*baseline.LatencyP95MS, *candidate.LatencyP95MS, aggregateLatencyRegressionBudget)
}

func comparisonRecommendations(verdict GateVerdict, evidence comparisonEvidence) []string {
	switch verdict {
	case "fail":
		return []string{"Do not promote until required gates and promotion-critical regressions are resolved."}
	case "unavailable":
		return []string{"Collect complete profile-qualified workload, treatment-factor, and benchmark snapshots before promotion."}
	default:
		if evidence.regressions > 0 {
			return []string{"Promotion-critical budgets passed; review advisory metric regressions before rollout."}
		}
		return []string{"Paired promotion-critical evidence passed without a detected regression."}
	}
}
