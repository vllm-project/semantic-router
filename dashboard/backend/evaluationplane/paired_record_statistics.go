package evaluationplane

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math/rand"
	"sort"
)

type pairedDirection string

const (
	pairedHigher pairedDirection = "higher_is_better"
	pairedLower  pairedDirection = "lower_is_better"
)

type pairedAnalysisUnit string

const (
	pairedCaseMean         pairedAnalysisUnit = "case_mean"
	pairedCaseMax          pairedAnalysisUnit = "case_max"
	pairedOracleRegret     pairedAnalysisUnit = "case_oracle_regret"
	pairedNormalizedRegret pairedAnalysisUnit = "case_normalized_regret"
)

const (
	comparisonConfidenceLevel      = 0.95
	comparisonMinimumAnalysisUnits = 20
	comparisonBootstrapSamples     = 1000
	comparisonG3RelativeMargin     = 0.05
)

type pairedValue func(executionRecordEvidence) (float64, bool)

type pairedStatisticDefinition struct {
	metricID             string
	trackID              TrackID
	direction            pairedDirection
	analysisUnit         pairedAnalysisUnit
	nonInferiorityMargin float64
	value                pairedValue
}

type pairedRecordPair struct {
	baseline  executionRecordEvidence
	candidate executionRecordEvidence
}

type pairedValuePair struct {
	baseline  float64
	candidate float64
}

var pairedStatisticRegistry = []pairedStatisticDefinition{
	{metricID: "routing.accuracy", trackID: "routing", direction: pairedHigher, analysisUnit: pairedCaseMean, nonInferiorityMargin: 0.02, value: recordQuality},
	{metricID: "model_pool.oracle_quality", trackID: "model_pool", direction: pairedHigher, analysisUnit: pairedCaseMax, nonInferiorityMargin: 0.02, value: outcomeRecordQuality},
	{metricID: "joint.realized_quality", trackID: "joint", direction: pairedHigher, analysisUnit: pairedCaseMean, nonInferiorityMargin: 0.02, value: recordQuality},
	{metricID: "joint.reliability", trackID: "joint", direction: pairedHigher, analysisUnit: pairedCaseMean, nonInferiorityMargin: 0.01, value: recordSuccess},
	{metricID: "joint.oracle_regret", trackID: "joint", direction: pairedLower, analysisUnit: pairedOracleRegret, nonInferiorityMargin: 0.02, value: recordQuality},
	{metricID: "joint.normalized_regret", trackID: "joint", direction: pairedLower, analysisUnit: pairedNormalizedRegret, nonInferiorityMargin: comparisonG3RelativeMargin, value: recordQuality},
	{metricID: "agentic.task_score", trackID: "agentic", direction: pairedHigher, analysisUnit: pairedCaseMean, nonInferiorityMargin: 0.05, value: recordQuality},
	{metricID: "agentic.success_rate", trackID: "agentic", direction: pairedHigher, analysisUnit: pairedCaseMean, nonInferiorityMargin: 0.02, value: recordSuccess},
	{metricID: "multimodal.quality", trackID: "multimodal", direction: pairedHigher, analysisUnit: pairedCaseMean, nonInferiorityMargin: 0.05, value: recordQuality},
	{metricID: "preference.agreement", trackID: "preference", direction: pairedHigher, analysisUnit: pairedCaseMean, nonInferiorityMargin: 0.02, value: recordPreference},
	{metricID: "safety.violation_case_rate", trackID: "safety", direction: pairedLower, analysisUnit: pairedCaseMean, nonInferiorityMargin: 0, value: recordViolation},
}

func recordQuality(record executionRecordEvidence) (float64, bool) {
	return outcomeRecordQuality(record)
}

func outcomeRecordQuality(record executionRecordEvidence) (float64, bool) {
	if record.Status == "unavailable" {
		return 0, false
	}
	return failedOutcomeQuality(record)
}

func recordSuccess(record executionRecordEvidence) (float64, bool) {
	if record.Status == "unavailable" || record.Success == nil {
		return 0, false
	}
	if *record.Success {
		return 1, true
	}
	return 0, true
}

func recordPreference(record executionRecordEvidence) (float64, bool) {
	if record.Status == "unavailable" || record.PreferenceMatch == nil {
		return 0, false
	}
	if *record.PreferenceMatch {
		return 1, true
	}
	return 0, true
}

func recordViolation(record executionRecordEvidence) (float64, bool) {
	if record.Status == "unavailable" || record.SafetyViolations == nil {
		return 0, false
	}
	if *record.SafetyViolations > 0 {
		return 1, true
	}
	return 0, true
}

func alignPrivateRecords(baseline, candidate []executionRecordEvidence) ([]pairedRecordPair, error) {
	baselineByID := make(map[string]executionRecordEvidence, len(baseline))
	candidateByID := make(map[string]executionRecordEvidence, len(candidate))
	baselinePoolByCase := make(map[string][]executionRecordEvidence)
	candidatePoolByCase := make(map[string][]executionRecordEvidence)
	for _, record := range baseline {
		if _, duplicate := baselineByID[record.ID]; duplicate {
			return nil, fmt.Errorf("%w: baseline private records contain duplicate ids", ErrInvalid)
		}
		baselineByID[record.ID] = record
		if record.TrackID == "model_pool" {
			baselinePoolByCase[record.CaseID] = append(baselinePoolByCase[record.CaseID], record)
		}
	}
	for _, record := range candidate {
		if _, duplicate := candidateByID[record.ID]; duplicate {
			return nil, fmt.Errorf("%w: candidate private records contain duplicate ids", ErrInvalid)
		}
		candidateByID[record.ID] = record
		if record.TrackID == "model_pool" {
			candidatePoolByCase[record.CaseID] = append(candidatePoolByCase[record.CaseID], record)
		}
	}
	ids := make([]string, 0, len(baselineByID))
	for id, record := range baselineByID {
		if record.TrackID == "model_pool" {
			continue
		}
		if _, ok := candidateByID[id]; !ok {
			return nil, fmt.Errorf("%w: private records are not case-aligned", ErrInvalid)
		}
		ids = append(ids, id)
	}
	for id, record := range candidateByID {
		if record.TrackID != "model_pool" {
			if _, ok := baselineByID[id]; !ok {
				return nil, fmt.Errorf("%w: private records are not case-aligned", ErrInvalid)
			}
		}
	}
	sort.Strings(ids)
	pairs := make([]pairedRecordPair, 0, len(ids)+len(baselinePoolByCase))
	for _, id := range ids {
		old, current := baselineByID[id], candidateByID[id]
		if old.TrackID != current.TrackID || old.CaseID != current.CaseID || old.AttemptID != current.AttemptID || stringValue(old.ArmID) != stringValue(current.ArmID) {
			return nil, fmt.Errorf("%w: private record analysis identities do not match", ErrInvalid)
		}
		pairs = append(pairs, pairedRecordPair{baseline: old, candidate: current})
	}
	if len(baselinePoolByCase) != len(candidatePoolByCase) {
		return nil, fmt.Errorf("%w: model-pool private records are not case-aligned", ErrInvalid)
	}
	for _, caseID := range sortedMapKeys(baselinePoolByCase) {
		oldRecords := baselinePoolByCase[caseID]
		currentRecords, found := candidatePoolByCase[caseID]
		if !found || len(oldRecords) == 0 || len(currentRecords) == 0 {
			return nil, fmt.Errorf("%w: model-pool private records are not case-aligned", ErrInvalid)
		}
		// A model-pool treatment may add or remove arms. A deterministic
		// case-local cycle covers every arm on both sides in O(max(B, C))
		// space; reducers subsequently take one case-level oracle, so arm rows
		// never become independent observations.
		for index := range max(len(oldRecords), len(currentRecords)) {
			pairs = append(pairs, pairedRecordPair{
				baseline: oldRecords[index%len(oldRecords)], candidate: currentRecords[index%len(currentRecords)],
			})
		}
	}
	return pairs, nil
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func computePairedStatistics(baseline, candidate []executionRecordEvidence, seed int64) ([]ComparisonStatistic, error) {
	pairs, err := alignPrivateRecords(baseline, candidate)
	if err != nil {
		return nil, err
	}
	results := make([]ComparisonStatistic, 0, len(pairedStatisticRegistry))
	for _, definition := range pairedStatisticRegistry {
		values, valueErr := pairedAnalysisValues(definition, pairs)
		if valueErr != nil {
			return nil, valueErr
		}
		if len(values) == 0 {
			continue
		}
		deltas := make([]float64, len(values))
		baselineValues := make([]float64, len(values))
		candidateValues := make([]float64, len(values))
		for index, pair := range values {
			baselineValues[index], candidateValues[index] = pair.baseline, pair.candidate
			deltas[index] = pair.candidate - pair.baseline
		}
		baselineMean, candidateMean := meanFloat64(baselineValues), meanFloat64(candidateValues)
		statistic := ComparisonStatistic{
			ID: definition.metricID, TrackID: definition.trackID,
			EstimatorID: "paired-bootstrap-case-clustered-delta", EstimatorVersion: "v1",
			AnalysisUnit: string(definition.analysisUnit), Direction: string(definition.direction),
			NonInferiorityMargin: definition.nonInferiorityMargin,
			BaselineValue:        baselineMean, CandidateValue: candidateMean,
			Delta: candidateMean - baselineMean, ConfidenceLevel: comparisonConfidenceLevel,
			DeltaConfidenceInterval: []float64{}, CandidateConfidenceInterval: []float64{},
			SampleCount: len(values), Verdict: "unavailable",
		}
		if len(values) >= comparisonMinimumAnalysisUnits {
			statistic.DeltaConfidenceInterval = pairedBootstrapInterval(deltas, metricSeed(seed, definition.metricID+".delta"))
			statistic.CandidateConfidenceInterval = pairedBootstrapInterval(candidateValues, metricSeed(seed, definition.metricID+".candidate"))
			statistic.Verdict = comparisonStatisticVerdict(statistic)
		}
		results = append(results, statistic)
	}
	return results, nil
}

func comparisonStatisticVerdict(statistic ComparisonStatistic) GateVerdict {
	if statistic.SampleCount < comparisonMinimumAnalysisUnits || len(statistic.DeltaConfidenceInterval) != 2 ||
		len(statistic.CandidateConfidenceInterval) != 2 {
		return "unavailable"
	}
	lower, upper := statistic.DeltaConfidenceInterval[0], statistic.DeltaConfidenceInterval[1]
	if statistic.Direction == string(pairedHigher) {
		if lower >= -statistic.NonInferiorityMargin {
			return "pass"
		}
		if upper < -statistic.NonInferiorityMargin {
			return "fail"
		}
		return "unavailable"
	}
	if upper <= statistic.NonInferiorityMargin {
		return "pass"
	}
	if lower > statistic.NonInferiorityMargin {
		return "fail"
	}
	return "unavailable"
}

func pairedAnalysisValues(definition pairedStatisticDefinition, pairs []pairedRecordPair) ([]pairedValuePair, error) {
	if definition.analysisUnit == pairedOracleRegret || definition.analysisUnit == pairedNormalizedRegret {
		return pairedRegretValues(pairs, definition.analysisUnit == pairedNormalizedRegret)
	}
	if definition.analysisUnit == pairedCaseMax {
		return pairedCaseMaximumValues(definition, pairs)
	}
	eligible := make([]struct {
		caseID string
		value  pairedValuePair
	}, 0)
	for _, pair := range pairs {
		if pair.baseline.TrackID != definition.trackID {
			continue
		}
		oldValue, oldOK := definition.value(pair.baseline)
		newValue, newOK := definition.value(pair.candidate)
		if oldOK && newOK {
			eligible = append(eligible, struct {
				caseID string
				value  pairedValuePair
			}{caseID: pair.baseline.CaseID, value: pairedValuePair{baseline: oldValue, candidate: newValue}})
		}
	}
	byCase := make(map[string][]pairedValuePair)
	for _, row := range eligible {
		byCase[row.caseID] = append(byCase[row.caseID], row.value)
	}
	caseIDs := sortedMapKeys(byCase)
	values := make([]pairedValuePair, 0, len(caseIDs))
	for _, caseID := range caseIDs {
		oldValue, newValue := byCase[caseID][0].baseline, byCase[caseID][0].candidate
		if definition.analysisUnit == pairedCaseMax {
			for _, value := range byCase[caseID][1:] {
				oldValue = max(oldValue, value.baseline)
				newValue = max(newValue, value.candidate)
			}
		} else {
			oldValues, newValues := make([]float64, 0, len(byCase[caseID])), make([]float64, 0, len(byCase[caseID]))
			for _, value := range byCase[caseID] {
				oldValues, newValues = append(oldValues, value.baseline), append(newValues, value.candidate)
			}
			oldValue, newValue = meanFloat64(oldValues), meanFloat64(newValues)
		}
		values = append(values, pairedValuePair{baseline: oldValue, candidate: newValue})
	}
	return values, nil
}

func pairedCaseMaximumValues(
	definition pairedStatisticDefinition,
	pairs []pairedRecordPair,
) ([]pairedValuePair, error) {
	baselineByCase := make(map[string][]float64)
	candidateByCase := make(map[string][]float64)
	for _, pair := range pairs {
		if pair.baseline.TrackID != definition.trackID {
			continue
		}
		if value, present := definition.value(pair.baseline); present {
			baselineByCase[pair.baseline.CaseID] = append(baselineByCase[pair.baseline.CaseID], value)
		}
		if value, present := definition.value(pair.candidate); present {
			candidateByCase[pair.candidate.CaseID] = append(candidateByCase[pair.candidate.CaseID], value)
		}
	}
	if len(baselineByCase) != len(candidateByCase) {
		return nil, fmt.Errorf("%w: model-pool feasible-oracle cases are incomplete", ErrInvalid)
	}
	values := make([]pairedValuePair, 0, len(baselineByCase))
	for _, caseID := range sortedMapKeys(baselineByCase) {
		current, found := candidateByCase[caseID]
		if !found || len(baselineByCase[caseID]) == 0 || len(current) == 0 {
			return nil, fmt.Errorf("%w: model-pool feasible-oracle cases are incomplete", ErrInvalid)
		}
		values = append(values, pairedValuePair{
			baseline: maxFloat64(baselineByCase[caseID]), candidate: maxFloat64(current),
		})
	}
	return values, nil
}

func pairedRegretValues(pairs []pairedRecordPair, normalized bool) ([]pairedValuePair, error) {
	byCase := make(map[string][]pairedRecordPair)
	for _, pair := range pairs {
		if pair.baseline.TrackID == "model_pool" || pair.baseline.TrackID == "joint" {
			byCase[pair.baseline.CaseID] = append(byCase[pair.baseline.CaseID], pair)
		}
	}
	values := make([]pairedValuePair, 0, len(byCase))
	for _, caseID := range sortedMapKeys(byCase) {
		var oldOracle, newOracle []float64
		var oldRealized, newRealized []float64
		for _, pair := range byCase[caseID] {
			if pair.baseline.TrackID == "model_pool" {
				if value, ok := outcomeRecordQuality(pair.baseline); ok {
					oldOracle = append(oldOracle, value)
				}
				if value, ok := outcomeRecordQuality(pair.candidate); ok {
					newOracle = append(newOracle, value)
				}
			} else {
				if value, ok := recordQuality(pair.baseline); ok {
					oldRealized = append(oldRealized, value)
				}
				if value, ok := recordQuality(pair.candidate); ok {
					newRealized = append(newRealized, value)
				}
			}
		}
		if len(oldOracle) == 0 || len(newOracle) == 0 {
			continue
		}
		if len(oldRealized) != 1 || len(newRealized) != 1 {
			if len(oldRealized) != 0 || len(newRealized) != 0 {
				return nil, fmt.Errorf("%w: joint regret requires one realized record per case", ErrInvalid)
			}
			continue
		}
		oldBest, newBest := maxFloat64(oldOracle), maxFloat64(newOracle)
		if normalized && (oldBest <= 0 || newBest <= 0) {
			continue
		}
		// Finite-pool oracle regret is a shortfall. Stochastic reruns can make
		// realized quality exceed the sampled arm maximum; that is zero
		// shortfall, never negative evidence that could game the upper bound.
		oldRegret, newRegret := max(0, oldBest-oldRealized[0]), max(0, newBest-newRealized[0])
		if normalized {
			oldRegret, newRegret = oldRegret/oldBest, newRegret/newBest
		}
		values = append(values, pairedValuePair{baseline: oldRegret, candidate: newRegret})
	}
	return values, nil
}

func sortedMapKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func maxFloat64(values []float64) float64 {
	current := values[0]
	for _, value := range values[1:] {
		current = max(current, value)
	}
	return current
}

func meanFloat64(values []float64) float64 {
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func metricSeed(seed int64, metricID string) int64 {
	digest := sha256.Sum256([]byte(metricID))
	return seed + int64(binary.BigEndian.Uint32(digest[:4]))
}

func pairedBootstrapInterval(deltas []float64, seed int64) []float64 {
	if len(deltas) < 2 {
		return nil
	}
	random := rand.New(rand.NewSource(seed))
	estimates := make([]float64, comparisonBootstrapSamples)
	resample := make([]float64, len(deltas))
	for index := range estimates {
		for row := range resample {
			resample[row] = deltas[random.Intn(len(deltas))]
		}
		estimates[index] = meanFloat64(resample)
	}
	sort.Float64s(estimates)
	return []float64{bootstrapPercentile(estimates, 0.025), bootstrapPercentile(estimates, 0.975)}
}

func bootstrapPercentile(values []float64, quantile float64) float64 {
	position := quantile * float64(len(values)-1)
	lower := int(position)
	upper := lower + 1
	if upper >= len(values) {
		return values[lower]
	}
	weight := position - float64(lower)
	return values[lower]*(1-weight) + values[upper]*weight
}
