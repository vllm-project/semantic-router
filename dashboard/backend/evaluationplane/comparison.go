package evaluationplane

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"time"
)

const aggregateLatencyRegressionBudget = 0.05

func comparePairedReports(
	baseline, candidate Report,
	baselineRecords, candidateRecords []executionRecordEvidence,
) (Comparison, error) {
	if err := validatePairedReportCohort(baseline, candidate); err != nil {
		return Comparison{}, err
	}
	statistics, err := computePairedStatistics(baselineRecords, candidateRecords, candidate.Run.Seed)
	if err != nil {
		return Comparison{}, err
	}
	metrics, evidence := pairedMetricEvidence(baseline.Metrics, candidate.Metrics, statistics)
	gates := comparisonGates(baseline, candidate, statistics)
	verdict, reason := comparisonVerdict(baseline, candidate, gates, evidence)
	recommendations := comparisonRecommendations(verdict, evidence)
	return Comparison{
		SchemaVersion: SchemaVersion, AttestationRevision: ServerAttestationRevision,
		BaselineRunID: baseline.Run.ID, CandidateRunID: candidate.Run.ID,
		Verdict: verdict,
		Summary: fmt.Sprintf(
			"Compared %d metrics (%d matched aggregate deltas): %d improved, %d regressed. %s",
			len(metrics), evidence.matched, evidence.improvements, evidence.regressions, reason,
		),
		Metrics: metrics, Statistics: statistics, Gates: gates, Recommendations: recommendations,
		CreatedAt: time.Now().UTC(),
	}, nil
}

func compareControlledPairReports(
	baseline, candidate campaignRunEvidence,
) (Comparison, error) {
	// This is intentionally stricter than generic report comparison: distinct
	// deployment targets are comparable only when the server-owned AB/BA
	// protocol, manifests, attestations, and observation receipts all validate.
	if _, err := buildCampaignPairedLiveEvidence(baseline, candidate); err != nil {
		return Comparison{}, err
	}
	normalizedCandidate := normalizeControlledPairCandidate(baseline.report, candidate.report)
	return comparePairedReports(
		baseline.report,
		normalizedCandidate,
		baseline.records,
		candidate.records,
	)
}

func validatePairedReportCohort(baseline, candidate Report) error {
	if baseline.AttestationRevision != ServerAttestationRevision ||
		candidate.AttestationRevision != ServerAttestationRevision {
		return fmt.Errorf("%w: both reports must use the current server attestation contract", ErrInvalid)
	}
	if baseline.Run.ID == candidate.Run.ID {
		return fmt.Errorf("%w: baseline and candidate runs must be distinct", ErrInvalid)
	}
	if candidate.Run.BaselineRunID != baseline.Run.ID {
		return fmt.Errorf("%w: candidate baseline_run_id must identify the compared baseline", ErrInvalid)
	}
	if baseline.Run.Mode != candidate.Run.Mode || baseline.Run.TargetID != candidate.Run.TargetID ||
		!sameRunMixtureIdentity(baseline.Run.Mixture, candidate.Run.Mixture) ||
		baseline.Run.ChangeProfile != candidate.Run.ChangeProfile ||
		baseline.Run.SampleLimit != candidate.Run.SampleLimit || baseline.Run.Concurrency != candidate.Run.Concurrency || baseline.Run.Seed != candidate.Run.Seed ||
		!reflect.DeepEqual(baseline.Run.CapacitySLO, candidate.Run.CapacitySLO) ||
		!reflect.DeepEqual(baseline.Run.CapacityLoadProtocol, candidate.Run.CapacityLoadProtocol) ||
		!reflect.DeepEqual(baseline.Run.SuiteIDs, candidate.Run.SuiteIDs) || !reflect.DeepEqual(baseline.Run.TrackIDs, candidate.Run.TrackIDs) {
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

func sameRunMixtureIdentity(left, right *CatalogMixture) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return left.ID == right.ID && left.RecipeName == right.RecipeName
}

type treatmentFactors struct {
	supported   bool
	primary     string
	code        bool
	policy      bool
	selector    bool
	adaptation  bool
	binding     bool
	pool        bool
	environment bool
}

// comparisonTreatment defines an identifiable treatment, not a loose allowlist.
// Primary is the factor that must change; every false factor must remain frozen.
// model_pool is the sole explicit composite because pool membership can also
// change its candidate binding and candidate-serving topology.
func comparisonTreatment(profile ChangeProfile) treatmentFactors {
	switch profile {
	case "schema_adapter":
		return treatmentFactors{supported: true, primary: "code", code: true}
	case "recipe":
		return treatmentFactors{supported: true, primary: "policy", policy: true}
	case "selector":
		return treatmentFactors{supported: true, primary: "selector", selector: true}
	case "model_pool":
		// Executable arm composition and its serving topology are one treatment:
		// adding or removing an arm necessarily changes the target-scoped backend
		// topology. Create-time validation still freezes the Router/Envoy origins,
		// credentials, and every production ledger so an unrelated environment
		// change cannot hide inside this profile.
		return treatmentFactors{
			supported: true, primary: "pool", binding: true, pool: true, environment: true,
		}
	case "runtime_capacity":
		return treatmentFactors{supported: true, primary: "environment", environment: true}
	case "online_adaptation":
		return treatmentFactors{supported: true, primary: "adaptation", adaptation: true}
	case "agent_multimodal":
		// Agent/trajectory and multimodal behavior do not yet have a server-owned
		// independent factor identity. Accepting policy or binding drift here would
		// let the same delta bypass the recipe/selector treatment contract.
		return treatmentFactors{}
	default:
		return treatmentFactors{}
	}
}

func validateTreatmentFactors(baseline, candidate Report) error {
	allowed := comparisonTreatment(baseline.Run.ChangeProfile)
	if !allowed.supported {
		return fmt.Errorf(
			"%w: change_profile %q has no independent server-owned treatment factor and cannot be paired",
			ErrInvalid, baseline.Run.ChangeProfile,
		)
	}
	if baseline.Provenance.CodeRevision == "" || candidate.Provenance.CodeRevision == "" {
		return fmt.Errorf("%w: baseline and candidate code revision identities are required", ErrInvalid)
	}
	codeChanged := baseline.Provenance.CodeRevision != candidate.Provenance.CodeRevision
	if codeChanged && !allowed.code {
		return fmt.Errorf(
			"%w: baseline and candidate code revisions must match for change_profile %q",
			ErrInvalid, baseline.Run.ChangeProfile,
		)
	}

	baselineSelector, candidateSelector, selectorAvailable := comparisonMixtureFactor(
		baseline.Run.Mixture, candidate.Run.Mixture, func(mixture *CatalogMixture) string { return mixture.SelectorDigest },
	)
	baselineAdaptation, candidateAdaptation, adaptationAvailable := comparisonMixtureFactor(
		baseline.Run.Mixture, candidate.Run.Mixture, func(mixture *CatalogMixture) string { return mixture.AdaptationDigest },
	)
	factors := []struct {
		name                 string
		baseline, candidate  string
		available, mayChange bool
	}{
		{"policy", baseline.Provenance.PolicySnapshotDigest, candidate.Provenance.PolicySnapshotDigest, true, allowed.policy},
		{"selector", baselineSelector, candidateSelector, selectorAvailable, allowed.selector},
		{"adaptation", baselineAdaptation, candidateAdaptation, adaptationAvailable, allowed.adaptation},
		{"binding", baseline.Provenance.BindingSnapshotDigest, candidate.Provenance.BindingSnapshotDigest, true, allowed.binding},
		{"pool", baseline.Provenance.PoolSnapshotDigest, candidate.Provenance.PoolSnapshotDigest, true, allowed.pool},
		{"environment", baseline.Provenance.EnvironmentSnapshotDigest, candidate.Provenance.EnvironmentSnapshotDigest, true, allowed.environment},
	}
	primaryChanged := allowed.primary == "code" && codeChanged
	for _, factor := range factors {
		if !factor.available {
			if factor.name == allowed.primary {
				return fmt.Errorf(
					"%w: change_profile %q requires a server-owned %s snapshot",
					ErrInvalid, baseline.Run.ChangeProfile, factor.name,
				)
			}
			continue
		}
		if factor.baseline == "" || factor.candidate == "" {
			return fmt.Errorf("%w: baseline and candidate %s snapshot identities are required", ErrInvalid, factor.name)
		}
		if !factor.mayChange && factor.candidate != factor.baseline {
			return fmt.Errorf("%w: baseline and candidate %s snapshots must match for change_profile %q", ErrInvalid, factor.name, baseline.Run.ChangeProfile)
		}
		if factor.name == allowed.primary {
			primaryChanged = factor.candidate != factor.baseline
		}
	}
	if !primaryChanged {
		return fmt.Errorf(
			"%w: change_profile %q requires the %s treatment factor to change",
			ErrInvalid, baseline.Run.ChangeProfile, allowed.primary,
		)
	}
	return nil
}

func comparisonMixtureFactor(
	baseline, candidate *CatalogMixture,
	value func(*CatalogMixture) string,
) (string, string, bool) {
	if baseline == nil || candidate == nil {
		return "", "", false
	}
	return value(baseline), value(candidate), true
}

type comparisonEvidence struct {
	matched             int
	improvements        int
	regressions         int
	latencyOverBudget   bool
	intervalCount       int
	intervalFailed      bool
	intervalPassed      bool
	intervalUnavailable bool
}

func pairedMetricEvidence(
	baseline, candidate []Metric,
	statistics []ComparisonStatistic,
) ([]Metric, comparisonEvidence) {
	baselineByID := make(map[string]Metric, len(baseline))
	for _, metric := range baseline {
		baselineByID[metric.ID] = metric
	}
	evidence := comparisonEvidence{}
	statisticsByID := make(map[string]ComparisonStatistic, len(statistics))
	allIntervalsPassed := true
	for _, statistic := range statistics {
		statisticsByID[statistic.ID] = statistic
		switch statistic.Verdict {
		case "pass":
		case "fail":
			evidence.intervalFailed = true
			allIntervalsPassed = false
		default:
			evidence.intervalUnavailable = true
			allIntervalsPassed = false
		}
		evidence.intervalCount++
	}
	metrics := make([]Metric, 0, len(candidate))
	evidence.intervalPassed = evidence.intervalCount > 0 && allIntervalsPassed
	for _, metric := range candidate {
		old, ok := baselineByID[metric.ID]
		if !ok || old.Value == nil || metric.Value == nil ||
			old.Unit != metric.Unit || old.TrackID != metric.TrackID || old.Direction != metric.Direction {
			metric.BaselineValue = nil
			metric.Delta = nil
			metrics = append(metrics, metric)
			continue
		}
		baselineValue := *old.Value
		delta := *metric.Value - baselineValue
		metric.BaselineValue, metric.Delta = &baselineValue, &delta
		if statistic, registered := statisticsByID[metric.ID]; registered {
			metric.BaselineValue = float64Reference(statistic.BaselineValue)
			metric.Delta = float64Reference(statistic.Delta)
			metric.ConfidenceInterval = append([]float64(nil), statistic.DeltaConfidenceInterval...)
			metric.SampleCount = statistic.SampleCount
		}
		evidence.matched++
		if delta != 0 && metric.Direction != "target" && metric.Direction != "" {
			if metricImproved(metric.Direction, delta) {
				evidence.improvements++
			} else {
				evidence.regressions++
			}
		}
		if latencyMetric(metric.ID) && exceedsRegressionBudget(baselineValue, *metric.Value, aggregateLatencyRegressionBudget) {
			evidence.latencyOverBudget = true
		}
		metrics = append(metrics, metric)
	}
	return metrics, evidence
}

func float64Reference(value float64) *float64 { return &value }

func comparisonVerdict(baseline, candidate Report, gates []Gate, evidence comparisonEvidence) (DecisionVerdict, string) {
	baselineSummary, current := baseline.Summary, candidate.Summary
	requiredUnavailable := false
	for _, gate := range gates {
		if gate.Disposition != GateDispositionRequired {
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
	if evidence.intervalFailed {
		return "fail", "A registered paired statistic regressed with 95% confidence."
	}
	if evidence.latencyOverBudget || summaryLatencyOverBudget(baselineSummary, current) {
		return "fail", "Tail latency exceeded the 5% aggregate regression budget."
	}
	if !completeQualifiedTrackVector(baseline) || !completeQualifiedTrackVector(candidate) {
		return "unavailable", "Every selected track needs complete qualified evidence for paired promotion."
	}
	if requiredUnavailable {
		return "unavailable", "Required candidate evidence is unavailable."
	}
	if evidence.intervalUnavailable {
		return "unavailable", fmt.Sprintf(
			"Every registered statistic needs at least %d independent case units and a decisive 95%% non-inferiority interval.",
			comparisonMinimumAnalysisUnits,
		)
	}
	if evidence.intervalPassed {
		return "pass", "Registered case-aligned paired confidence intervals passed."
	}
	if evidence.intervalCount == 0 {
		return "unavailable", "No registered paired statistic has a valid interval."
	}
	return "unavailable", "A registered paired confidence interval crosses the promotion boundary."
}

func completeQualifiedTrackVector(report Report) bool {
	if len(report.Tracks) != len(report.Run.TrackIDs) {
		return false
	}
	for index, trackID := range report.Run.TrackIDs {
		track := report.Tracks[index]
		if track.TrackID != trackID || track.Status != "completed" || track.EvidenceLevel == "E0" || track.Coverage.Unavailable != 0 {
			return false
		}
	}
	return true
}

func metricImproved(direction string, delta float64) bool {
	if direction == "lower_is_better" {
		return delta < 0
	}
	return delta > 0
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

func summaryLatencyOverBudget(baseline, candidate ReportSummary) bool {
	return baseline.LatencyP95MS != nil && candidate.LatencyP95MS != nil &&
		exceedsRegressionBudget(*baseline.LatencyP95MS, *candidate.LatencyP95MS, aggregateLatencyRegressionBudget)
}

func comparisonRecommendations(verdict DecisionVerdict, evidence comparisonEvidence) []string {
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
