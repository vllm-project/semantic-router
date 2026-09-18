package evaluationplane

import (
	"fmt"
	"math"
	"strings"
)

const maxReducedFloatULPs = 8

type reducedMetricEvidence struct {
	Value              *float64
	ConfidenceInterval []float64
	SampleCount        int
}

type recordMetricAttestation struct {
	RoutingAccuracy          reducedMetricEvidence
	SafetyViolationRate      reducedMetricEvidence
	SafetyBlockAccuracy      reducedMetricEvidence
	JointNormalizedRegret    reducedMetricEvidence
	AgenticSuccessRate       reducedMetricEvidence
	PreferenceAgreement      reducedMetricEvidence
	PreferencePropensity     reducedMetricEvidence
	PreferenceEffectiveN     reducedMetricEvidence
	PreferenceEffectiveRatio reducedMetricEvidence
	PreferenceIPSAgreement   reducedMetricEvidence
	ModelPool                []modelPoolMetricEvidence
	SafetyTypedRowsByCase    map[string]int
	CapacityRowsByCase       map[string]int
	CapacityLevelsByCase     map[string]map[int64]struct{}
}

type jointMetricRow struct {
	caseID  string
	quality float64
}

type recordMetricReducer struct {
	routingQualityRows       int
	routingQualityTotal      float64
	safetyRows               int
	safetyViolationTotal     uint64
	safetyBlockRows          int
	safetyBlockCorrect       int
	poolOracleByCase         map[string]float64
	jointRows                []jointMetricRow
	agenticSuccessRows       int
	agenticSucceededRows     int
	preferenceRows           int
	preferenceMatchRows      int
	preferenceMatches        int
	preferencePropensityRows int
	preferenceWeightRows     int
	preferenceWeightTotal    float64
	preferenceWeightSq       float64
	preferenceMatchWeight    float64
	safetyTypedRowsByCase    map[string]int
	capacityRowsByCase       map[string]int
	capacityLevelsByCase     map[string]map[int64]struct{}
}

func newRecordMetricReducer() *recordMetricReducer {
	return &recordMetricReducer{
		poolOracleByCase:      make(map[string]float64),
		safetyTypedRowsByCase: make(map[string]int),
		capacityRowsByCase:    make(map[string]int),
		capacityLevelsByCase:  make(map[string]map[int64]struct{}),
	}
}

func (reducer *recordMetricReducer) observe(record executionRecordEvidence) error {
	if record.Status == "unavailable" {
		return nil
	}
	switch record.TrackID {
	case "routing":
		return reducer.observeRouting(record)
	case "safety":
		return reducer.observeSafety(record)
	case "model_pool":
		reducer.observeModelPool(record)
	case "joint":
		reducer.observeJoint(record)
	case "agentic":
		reducer.observeAgentic(record)
	case "preference":
		return reducer.observePreference(record)
	case "capacity":
		return reducer.observeCapacity(record)
	}
	return nil
}

func (reducer *recordMetricReducer) observeRouting(record executionRecordEvidence) error {
	if record.Quality == nil {
		return nil
	}
	reducer.routingQualityRows++
	reducer.routingQualityTotal += *record.Quality
	if !finiteFloat(reducer.routingQualityTotal) {
		return fmt.Errorf("routing.accuracy aggregate is not finite")
	}
	return nil
}

func (reducer *recordMetricReducer) observeSafety(record executionRecordEvidence) error {
	reducer.safetyRows++
	if record.SafetyViolations != nil {
		// #nosec G115 -- strict record validation rejects negative counters before reduction.
		violations := uint64(*record.SafetyViolations)
		if reducer.safetyViolationTotal > ^uint64(0)-violations {
			return fmt.Errorf("safety_violations aggregate overflows the reducer")
		}
		reducer.safetyViolationTotal += violations
	}
	if record.ShouldBlock != nil && record.Blocked != nil {
		reducer.safetyBlockRows++
		if *record.ShouldBlock == *record.Blocked {
			reducer.safetyBlockCorrect++
		}
	}
	if record.SafetyViolations != nil && record.ShouldBlock != nil && record.Blocked != nil {
		reducer.safetyTypedRowsByCase[record.CaseID]++
	}
	return nil
}

func (reducer *recordMetricReducer) observeModelPool(record executionRecordEvidence) {
	quality, present := failedOutcomeQuality(record)
	if !present {
		return
	}
	current, observed := reducer.poolOracleByCase[record.CaseID]
	if !observed || quality > current {
		reducer.poolOracleByCase[record.CaseID] = quality
	}
}

func (reducer *recordMetricReducer) observeJoint(record executionRecordEvidence) {
	if quality, present := failedOutcomeQuality(record); present {
		reducer.jointRows = append(reducer.jointRows, jointMetricRow{
			caseID: record.CaseID, quality: quality,
		})
	}
}

func (reducer *recordMetricReducer) observeAgentic(record executionRecordEvidence) {
	// Fault-recovery continuity is an independent G6 method, not a task
	// trajectory outcome. Its rows must not dilute task success metrics.
	if record.Recovery != nil {
		return
	}
	if record.Success == nil {
		return
	}
	reducer.agenticSuccessRows++
	if *record.Success {
		reducer.agenticSucceededRows++
	}
}

func (reducer *recordMetricReducer) observePreference(record executionRecordEvidence) error {
	reducer.preferenceRows++
	if record.PreferenceMatch != nil {
		reducer.preferenceMatchRows++
		if *record.PreferenceMatch {
			reducer.preferenceMatches++
		}
	}
	if record.BehaviorPropensity != nil {
		reducer.preferencePropensityRows++
	}
	if record.BehaviorPropensity == nil || record.PreferenceMatch == nil {
		return nil
	}
	weight := 1 / *record.BehaviorPropensity
	reducer.preferenceWeightRows++
	reducer.preferenceWeightTotal += weight
	reducer.preferenceWeightSq += weight * weight
	if *record.PreferenceMatch {
		reducer.preferenceMatchWeight += weight
	}
	if !finiteFloat(weight) || !finiteFloat(reducer.preferenceWeightTotal) ||
		!finiteFloat(reducer.preferenceWeightSq) || !finiteFloat(reducer.preferenceMatchWeight) {
		return fmt.Errorf("preference inverse-propensity aggregate is not finite")
	}
	return nil
}

func (reducer *recordMetricReducer) observeCapacity(record executionRecordEvidence) error {
	// Warmup is part of the sealed load process but never part of a
	// measurement claim. Recorded-source capacity rows have no load phase
	// and remain ordinary replay observations.
	if record.LoadPhase != nil && *record.LoadPhase == "warmup" {
		return nil
	}
	if record.Success != nil && record.Concurrency != nil {
		reducer.capacityRowsByCase[record.CaseID]++
		levels := reducer.capacityLevelsByCase[record.CaseID]
		if levels == nil {
			levels = make(map[int64]struct{})
			reducer.capacityLevelsByCase[record.CaseID] = levels
		}
		levels[*record.Concurrency] = struct{}{}
	}
	return nil
}

// failedOutcomeQuality mirrors the report reducer: an attempted failure is a
// measured zero-quality outcome, not a row to discard. Successful but ungraded
// observations remain unavailable.
func failedOutcomeQuality(record executionRecordEvidence) (float64, bool) {
	if record.Status == "failed" || (record.Success != nil && !*record.Success) {
		return 0, true
	}
	if record.Quality == nil {
		return 0, false
	}
	return *record.Quality, true
}

func (reducer *recordMetricReducer) finalize() (recordMetricAttestation, error) {
	attestation := reducer.emptyAttestation()
	reducer.finalizeObservedRates(&attestation)
	if err := reducer.finalizePreferenceMetrics(&attestation); err != nil {
		return recordMetricAttestation{}, err
	}
	jointRegret, err := reducer.finalizeJointRegret()
	if err != nil {
		return recordMetricAttestation{}, err
	}
	attestation.JointNormalizedRegret = jointRegret
	return attestation, nil
}

func (reducer *recordMetricReducer) emptyAttestation() recordMetricAttestation {
	return recordMetricAttestation{
		RoutingAccuracy:          reducedMetricEvidence{SampleCount: reducer.routingQualityRows},
		SafetyViolationRate:      reducedMetricEvidence{SampleCount: reducer.safetyRows},
		SafetyBlockAccuracy:      reducedMetricEvidence{SampleCount: reducer.safetyBlockRows},
		AgenticSuccessRate:       reducedMetricEvidence{SampleCount: reducer.agenticSuccessRows},
		PreferenceAgreement:      reducedMetricEvidence{SampleCount: reducer.preferenceMatchRows},
		PreferencePropensity:     reducedMetricEvidence{SampleCount: reducer.preferenceRows},
		PreferenceEffectiveN:     reducedMetricEvidence{SampleCount: reducer.preferenceWeightRows},
		PreferenceEffectiveRatio: reducedMetricEvidence{SampleCount: reducer.preferenceWeightRows},
		PreferenceIPSAgreement:   reducedMetricEvidence{SampleCount: reducer.preferenceWeightRows},
		SafetyTypedRowsByCase:    reducer.safetyTypedRowsByCase,
		CapacityRowsByCase:       reducer.capacityRowsByCase,
		CapacityLevelsByCase:     reducer.capacityLevelsByCase,
	}
}

func (reducer *recordMetricReducer) finalizeObservedRates(attestation *recordMetricAttestation) {
	if reducer.routingQualityRows > 0 {
		value := reducer.routingQualityTotal / float64(reducer.routingQualityRows)
		attestation.RoutingAccuracy.Value = &value
		successes := int(math.RoundToEven(value * float64(reducer.routingQualityRows)))
		if successes < 0 {
			successes = 0
		} else if successes > reducer.routingQualityRows {
			successes = reducer.routingQualityRows
		}
		attestation.RoutingAccuracy.ConfidenceInterval = serverWilsonInterval(successes, reducer.routingQualityRows)
	}
	if reducer.safetyRows > 0 {
		value := float64(reducer.safetyViolationTotal) / float64(reducer.safetyRows)
		attestation.SafetyViolationRate.Value = &value
	}
	if reducer.safetyBlockRows > 0 {
		value := float64(reducer.safetyBlockCorrect) / float64(reducer.safetyBlockRows)
		attestation.SafetyBlockAccuracy.Value = &value
		attestation.SafetyBlockAccuracy.ConfidenceInterval = serverWilsonInterval(reducer.safetyBlockCorrect, reducer.safetyBlockRows)
	}
	if reducer.agenticSuccessRows > 0 {
		value := float64(reducer.agenticSucceededRows) / float64(reducer.agenticSuccessRows)
		attestation.AgenticSuccessRate.Value = &value
		attestation.AgenticSuccessRate.ConfidenceInterval = serverWilsonInterval(reducer.agenticSucceededRows, reducer.agenticSuccessRows)
	}
	if reducer.preferenceMatchRows > 0 {
		value := float64(reducer.preferenceMatches) / float64(reducer.preferenceMatchRows)
		attestation.PreferenceAgreement.Value = &value
		attestation.PreferenceAgreement.ConfidenceInterval = serverWilsonInterval(reducer.preferenceMatches, reducer.preferenceMatchRows)
	}
	if reducer.preferenceRows > 0 {
		value := float64(reducer.preferencePropensityRows) / float64(reducer.preferenceRows)
		attestation.PreferencePropensity.Value = &value
		attestation.PreferencePropensity.ConfidenceInterval = serverWilsonInterval(reducer.preferencePropensityRows, reducer.preferenceRows)
	}
}

func (reducer *recordMetricReducer) finalizePreferenceMetrics(attestation *recordMetricAttestation) error {
	if reducer.preferenceWeightRows > 0 && reducer.preferenceWeightSq > 0 {
		effective := reducer.preferenceWeightTotal * reducer.preferenceWeightTotal / reducer.preferenceWeightSq
		ratio := effective / float64(reducer.preferenceWeightRows)
		agreement := reducer.preferenceMatchWeight / reducer.preferenceWeightTotal
		if !finiteFloat(effective) || !finiteFloat(ratio) || !finiteFloat(agreement) {
			return fmt.Errorf("preference inverse-propensity metrics are not finite")
		}
		attestation.PreferenceEffectiveN.Value = &effective
		attestation.PreferenceEffectiveRatio.Value = &ratio
		attestation.PreferenceIPSAgreement.Value = &agreement
	}
	return nil
}

func (reducer *recordMetricReducer) finalizeJointRegret() (reducedMetricEvidence, error) {
	normalizedRegretTotal := 0.0
	normalizedRegretCount := 0
	for _, row := range reducer.jointRows {
		oracle, present := reducer.poolOracleByCase[row.caseID]
		if !present || oracle <= 0 {
			continue
		}
		shortfall := oracle - row.quality
		if shortfall < 0 {
			shortfall = 0
		}
		normalizedRegretTotal += shortfall / oracle
		if !finiteFloat(normalizedRegretTotal) {
			return reducedMetricEvidence{}, fmt.Errorf("joint.normalized_regret aggregate is not finite")
		}
		normalizedRegretCount++
	}
	metric := reducedMetricEvidence{SampleCount: normalizedRegretCount}
	if normalizedRegretCount > 0 {
		value := normalizedRegretTotal / float64(normalizedRegretCount)
		if !finiteFloat(value) {
			return reducedMetricEvidence{}, fmt.Errorf("joint.normalized_regret is not finite")
		}
		metric.Value = &value
	}
	return metric, nil
}

type reducedMetricContract struct {
	ID        string
	Name      string
	TrackID   TrackID
	Unit      string
	Direction string
	Expected  func(recordMetricAttestation) reducedMetricEvidence
}

var reducedMetricContracts = []reducedMetricContract{
	{
		ID: "routing.accuracy", Name: "Routing accuracy", TrackID: "routing",
		Unit: "fraction", Direction: "higher_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.RoutingAccuracy },
	},
	{
		ID: "safety.violation_rate", Name: "Safety violation rate", TrackID: "safety",
		Unit: "violations/case", Direction: "lower_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.SafetyViolationRate },
	},
	{
		ID: "safety.block_accuracy", Name: "Blocking decision accuracy", TrackID: "safety",
		Unit: "fraction", Direction: "higher_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.SafetyBlockAccuracy },
	},
	{
		ID: "joint.normalized_regret", Name: "Normalized pool-oracle regret", TrackID: "joint",
		Unit: "fraction", Direction: "lower_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.JointNormalizedRegret },
	},
	{
		ID: "agentic.success_rate", Name: "Trajectory success rate", TrackID: "agentic",
		Unit: "fraction", Direction: "higher_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.AgenticSuccessRate },
	},
	{
		ID: "preference.agreement", Name: "Offline preference agreement", TrackID: "preference",
		Unit: "fraction", Direction: "higher_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.PreferenceAgreement },
	},
	{
		ID: "preference.propensity_coverage", Name: "Behavior propensity coverage", TrackID: "preference",
		Unit: "fraction", Direction: "higher_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.PreferencePropensity },
	},
	{
		ID: "preference.effective_sample_size", Name: "Inverse-propensity effective sample size", TrackID: "preference",
		Unit: "effective samples", Direction: "higher_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.PreferenceEffectiveN },
	},
	{
		ID: "preference.effective_sample_ratio", Name: "Effective-sample ratio", TrackID: "preference",
		Unit: "fraction", Direction: "higher_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.PreferenceEffectiveRatio },
	},
	{
		ID: "preference.self_normalized_ips_agreement", Name: "Self-normalized IPS agreement", TrackID: "preference",
		Unit: "fraction", Direction: "higher_is_better",
		Expected: func(value recordMetricAttestation) reducedMetricEvidence { return value.PreferenceIPSAgreement },
	},
}

func validateServerReducedMetrics(report Report, attestation recordMetricAttestation) error {
	metrics := make(map[string]Metric, len(report.Metrics))
	for _, metric := range report.Metrics {
		metrics[metric.ID] = metric
	}
	for _, contract := range reducedMetricContracts {
		selected := containsTrack(report.Run.TrackIDs, contract.TrackID)
		actual, present := metrics[contract.ID]
		if !selected {
			if present {
				return fmt.Errorf("%w: server-reduced metric %s is published for an unselected track", ErrInvalid, contract.ID)
			}
			continue
		}
		if !present {
			return fmt.Errorf("%w: server-reduced metric %s is missing", ErrInvalid, contract.ID)
		}
		if actual.Name != contract.Name || actual.TrackID != contract.TrackID || actual.Unit != contract.Unit || actual.Direction != contract.Direction {
			return fmt.Errorf("%w: server-reduced metric %s metadata is not canonical", ErrInvalid, contract.ID)
		}
		expected := contract.Expected(attestation)
		if actual.SampleCount != expected.SampleCount {
			return fmt.Errorf("%w: server-reduced metric %s sample_count does not match records", ErrInvalid, contract.ID)
		}
		if (actual.Value == nil) != (expected.Value == nil) {
			return fmt.Errorf("%w: server-reduced metric %s availability does not match records", ErrInvalid, contract.ID)
		}
		if actual.Value != nil && !reducedFloatsEqual(*actual.Value, *expected.Value) {
			return fmt.Errorf("%w: server-reduced metric %s value does not match records", ErrInvalid, contract.ID)
		}
		if !reducedIntervalsEqual(actual.ConfidenceInterval, expected.ConfidenceInterval) {
			return fmt.Errorf("%w: server-reduced metric %s confidence_interval does not match records", ErrInvalid, contract.ID)
		}
	}
	return validateServerReducedModelPoolMetrics(report, attestation.ModelPool)
}

func validateServerReducedModelPoolMetrics(report Report, expected []modelPoolMetricEvidence) error {
	if expected == nil {
		return nil
	}
	actual := make(map[string]Metric, len(expected))
	for _, metric := range report.Metrics {
		if metric.TrackID != "model_pool" && !strings.HasPrefix(metric.ID, "model_pool.") {
			continue
		}
		if metric.TrackID != "model_pool" || !isCanonicalModelPoolMetricID(metric.ID) {
			return fmt.Errorf("%w: model-pool metric %s is not canonical", ErrInvalid, metric.ID)
		}
		actual[metric.ID] = metric
	}
	if len(actual) != len(expected) {
		return fmt.Errorf("%w: model-pool metric set is missing or contains extra metrics", ErrInvalid)
	}
	for _, want := range expected {
		got, present := actual[want.ID]
		if !present || got.SampleCount != want.SampleCount || (got.Value == nil) != (want.Value == nil) || got.ConfidenceInterval != nil {
			return fmt.Errorf("%w: model-pool metric %s does not match server evidence", ErrInvalid, want.ID)
		}
		if got.Value != nil && !reducedFloatsEqual(*got.Value, *want.Value) {
			return fmt.Errorf("%w: model-pool metric %s value does not match server evidence", ErrInvalid, want.ID)
		}
		exclusions := 0
		for _, count := range want.MissingReasonCounts {
			exclusions += count
		}
		if got.AnalysisProvenance.ObservedExclusions == nil || *got.AnalysisProvenance.ObservedExclusions != exclusions {
			return fmt.Errorf("%w: model-pool metric %s provenance exclusions do not match server evidence", ErrInvalid, want.ID)
		}
	}
	return nil
}

func serverWilsonInterval(successes, total int) []float64 {
	if total <= 0 {
		return nil
	}
	z := 1.959963984540054
	numerator := float64(successes) / float64(total)
	denominator := 1 + z*z/float64(total)
	center := (numerator + z*z/(2*float64(total))) / denominator
	margin := z * math.Sqrt((numerator*(1-numerator)+z*z/(4*float64(total)))/float64(total)) / denominator
	return []float64{math.Max(0, center-margin), math.Min(1, center+margin)}
}

func reducedIntervalsEqual(left, right []float64) bool {
	if (left == nil) != (right == nil) || len(left) != len(right) {
		return false
	}
	for index := range left {
		if !reducedFloatsEqual(left[index], right[index]) {
			return false
		}
	}
	return true
}

func reducedFloatsEqual(left, right float64) bool {
	if !finiteFloat(left) || !finiteFloat(right) {
		return false
	}
	if left == right {
		return true
	}
	spacing := math.Max(
		math.Abs(math.Nextafter(left, math.Inf(1))-left),
		math.Abs(math.Nextafter(right, math.Inf(1))-right),
	)
	return math.Abs(left-right) <= maxReducedFloatULPs*spacing
}
