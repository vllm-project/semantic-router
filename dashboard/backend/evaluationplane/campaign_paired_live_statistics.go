package evaluationplane

import (
	"math"
	"math/rand"
	"sort"
)

func campaignPairedStatistics(
	cohort campaignPairedCohort,
	trackIDs []TrackID,
	seed int64,
) []CampaignPairedStatistic {
	byCase := make(map[string][]campaignObservationPair)
	for _, pair := range cohort.exactPairs {
		key := string(pair.baseline.trackID) + "\x00" + pair.baseline.caseID
		byCase[key] = append(byCase[key], pair)
	}
	aggregates := make([]*campaignCasePair, 0, len(byCase)+len(cohort.poolCases))
	for _, pairs := range byCase {
		aggregates = append(aggregates, campaignExactCaseAggregate(pairs))
	}
	for _, poolCase := range cohort.poolCases {
		aggregates = append(aggregates, campaignPoolCaseAggregate(poolCase))
	}
	statistics := make([]CampaignPairedStatistic, 0, len(trackIDs)*3+1)
	for _, trackID := range trackIDs {
		cases := campaignCasesForTrack(aggregates, trackID)
		if campaignTrackHasQualityStatistic(trackID) {
			statistics = append(statistics, campaignQualityStatistic(trackID, cases, seed))
		}
		if trackID == "model_pool" {
			statistics = append(statistics, campaignPoolWorstArmReliabilityStatistic(cohort.poolCases, seed))
		}
		statistics = append(
			statistics,
			campaignFailureStatistic(trackID, cases, seed),
			campaignLatencyStatistic(trackID, cases, seed),
		)
	}
	return statistics
}

func campaignPoolWorstArmReliabilityStatistic(
	poolCases []campaignPoolCasePair,
	seed int64,
) CampaignPairedStatistic {
	const statisticID = "campaign.g3.model_pool.worst_arm_reliability_non_inferiority"
	statistic := CampaignPairedStatistic{
		ID: statisticID, GateID: "G3", TrackID: "model_pool",
		AnalysisUnit: campaignPoolWorstArmReliabilityUnit, Direction: "higher_is_better",
		Margin: campaignFailureRiskMargin, ConfidenceLevel: campaignPairedConfidenceLevel,
		ConfidenceInterval: []float64{}, SampleCount: len(poolCases), Verdict: "unavailable",
	}
	if len(poolCases) == 0 {
		return statistic
	}
	baselineValue := campaignWorstArmReliability(poolCases, false, nil)
	candidateValue := campaignWorstArmReliability(poolCases, true, nil)
	statistic.BaselineValue = float64Reference(baselineValue)
	statistic.CandidateValue = float64Reference(candidateValue)
	statistic.Delta = float64Reference(candidateValue - baselineValue)
	if len(poolCases) < campaignPairedMinimumCases {
		return statistic
	}
	statistic.ConfidenceInterval = campaignWorstArmReliabilityInterval(
		poolCases,
		metricSeed(seed, statisticID),
	)
	statistic.CandidateConfidenceInterval = campaignWorstArmReliabilityCandidateInterval(
		poolCases,
		metricSeed(seed, statisticID+".candidate"),
	)
	statistic.Verdict = campaignWorstArmReliabilityVerdict(statistic)
	return statistic
}

func campaignWorstArmReliability(
	poolCases []campaignPoolCasePair,
	candidate bool,
	resample []int,
) float64 {
	armSuccesses := make(map[string]float64)
	indices := resample
	if indices == nil {
		indices = make([]int, len(poolCases))
		for index := range poolCases {
			indices[index] = index
		}
	}
	for _, index := range indices {
		observations := poolCases[index].baseline
		if candidate {
			observations = poolCases[index].candidate
		}
		for _, observation := range observations {
			if _, present := armSuccesses[observation.armID]; !present {
				armSuccesses[observation.armID] = 0
			}
			if observation.success {
				armSuccesses[observation.armID]++
			}
		}
	}
	worst := 1.0
	for _, successes := range armSuccesses {
		worst = math.Min(worst, successes/float64(len(indices)))
	}
	return worst
}

func campaignWorstArmReliabilityInterval(
	poolCases []campaignPoolCasePair,
	seed int64,
) []float64 {
	random := rand.New(rand.NewSource(seed))
	estimates := make([]float64, campaignPairedBootstrapSamples)
	resample := make([]int, len(poolCases))
	for index := range estimates {
		for row := range resample {
			resample[row] = random.Intn(len(poolCases))
		}
		estimates[index] = campaignWorstArmReliability(poolCases, true, resample) -
			campaignWorstArmReliability(poolCases, false, resample)
	}
	sort.Float64s(estimates)
	return []float64{
		bootstrapPercentile(estimates, 0.025),
		bootstrapPercentile(estimates, 0.975),
	}
}

func campaignWorstArmReliabilityCandidateInterval(
	poolCases []campaignPoolCasePair,
	seed int64,
) []float64 {
	random := rand.New(rand.NewSource(seed))
	estimates := make([]float64, campaignPairedBootstrapSamples)
	resample := make([]int, len(poolCases))
	for index := range estimates {
		for row := range resample {
			resample[row] = random.Intn(len(poolCases))
		}
		estimates[index] = campaignWorstArmReliability(poolCases, true, resample)
	}
	sort.Float64s(estimates)
	return []float64{
		bootstrapPercentile(estimates, 0.025),
		bootstrapPercentile(estimates, 0.975),
	}
}

func campaignArmReliabilityStatistics(
	poolCases []campaignPoolCasePair,
	seed int64,
) []CampaignArmReliabilityStatistic {
	if len(poolCases) == 0 {
		return []CampaignArmReliabilityStatistic{}
	}
	baseline := campaignArmFailureVectors(poolCases, false)
	candidate := campaignArmFailureVectors(poolCases, true)
	armIDs := make(map[string]struct{}, len(baseline)+len(candidate))
	for armID := range baseline {
		armIDs[armID] = struct{}{}
	}
	for armID := range candidate {
		armIDs[armID] = struct{}{}
	}
	statistics := make([]CampaignArmReliabilityStatistic, 0, len(armIDs))
	for _, armID := range sortedMapKeys(armIDs) {
		oldValues, oldPresent := baseline[armID]
		newValues, newPresent := candidate[armID]
		statistic := CampaignArmReliabilityStatistic{
			ArmID: armID, Direction: "lower_is_better", Margin: campaignFailureRiskMargin,
			ConfidenceLevel: campaignPairedConfidenceLevel, ConfidenceInterval: []float64{},
			CandidateConfidenceInterval: []float64{},
			BaselineSampleCount:         len(oldValues), CandidateSampleCount: len(newValues), Verdict: "unavailable",
		}
		if oldPresent {
			statistic.BaselineFailureRate = float64Reference(meanFloat64(oldValues))
		}
		if newPresent {
			statistic.CandidateFailureRate = float64Reference(meanFloat64(newValues))
		}
		switch {
		case oldPresent && newPresent:
			statistic.Cohort = campaignArmCohortPaired
			deltas := make([]float64, len(oldValues))
			for index := range oldValues {
				deltas[index] = newValues[index] - oldValues[index]
			}
			statistic.Delta = float64Reference(
				*statistic.CandidateFailureRate - *statistic.BaselineFailureRate,
			)
			if len(deltas) >= campaignPairedMinimumCases {
				statistic.ConfidenceInterval = pairedBootstrapInterval(
					deltas,
					metricSeed(seed, "campaign.g3.model_pool.arm."+armID+".failure_rate"),
				)
				statistic.CandidateConfidenceInterval = pairedBootstrapInterval(
					newValues,
					metricSeed(seed, "campaign.g3.model_pool.arm."+armID+".absolute_failure_rate"),
				)
				statistic.Verdict = campaignArmReliabilityVerdict(statistic)
			}
		case oldPresent:
			statistic.Cohort = campaignArmCohortBaselineOnly
		default:
			statistic.Cohort = campaignArmCohortCandidateOnly
			statistic.Margin = 1 - frozenCampaignG3PromotionPolicy.MinimumCandidateArmReliability
			if len(newValues) >= campaignPairedMinimumCases {
				statistic.CandidateConfidenceInterval = pairedBootstrapInterval(
					newValues,
					metricSeed(seed, "campaign.g3.model_pool.arm."+armID+".absolute_failure_rate"),
				)
				statistic.Verdict = campaignArmReliabilityVerdict(statistic)
			}
		}
		statistics = append(statistics, statistic)
	}
	return statistics
}

func campaignArmFailureVectors(
	poolCases []campaignPoolCasePair,
	candidate bool,
) map[string][]float64 {
	values := make(map[string][]float64)
	for _, poolCase := range poolCases {
		observations := poolCase.baseline
		if candidate {
			observations = poolCase.candidate
		}
		for _, observation := range observations {
			failure := 0.0
			if !observation.success {
				failure = 1
			}
			values[observation.armID] = append(values[observation.armID], failure)
		}
	}
	return values
}

func campaignArmReliabilityVerdict(statistic CampaignArmReliabilityStatistic) GateVerdict {
	if statistic.CandidateSampleCount < campaignPairedMinimumCases ||
		len(statistic.CandidateConfidenceInterval) != 2 {
		return "unavailable"
	}
	candidateLower, candidateUpper := statistic.CandidateConfidenceInterval[0], statistic.CandidateConfidenceInterval[1]
	if statistic.Cohort == campaignArmCohortCandidateOnly {
		if candidateUpper <= statistic.Margin {
			return "pass"
		}
		if candidateLower > statistic.Margin {
			return "fail"
		}
		return "unavailable"
	}
	if statistic.Cohort != campaignArmCohortPaired ||
		statistic.BaselineSampleCount != statistic.CandidateSampleCount ||
		len(statistic.ConfidenceInterval) != 2 {
		return "unavailable"
	}
	lower, upper := statistic.ConfidenceInterval[0], statistic.ConfidenceInterval[1]
	absoluteMaximum := 1 - frozenCampaignG3PromotionPolicy.MinimumCandidateArmReliability
	if upper <= statistic.Margin && candidateUpper <= absoluteMaximum {
		return "pass"
	}
	if lower > statistic.Margin || candidateLower > absoluteMaximum {
		return "fail"
	}
	return "unavailable"
}

func campaignExactCaseAggregate(pairs []campaignObservationPair) *campaignCasePair {
	row := &campaignCasePair{trackID: pairs[0].baseline.trackID, caseID: pairs[0].baseline.caseID}
	baselineQuality := make([]float64, 0, len(pairs))
	candidateQuality := make([]float64, 0, len(pairs))
	qualityComplete := true
	baselineFailures, candidateFailures := 0, 0
	for _, pair := range pairs {
		oldQuality, oldPresent := campaignObservationQuality(pair.baseline)
		newQuality, newPresent := campaignObservationQuality(pair.candidate)
		if oldPresent && newPresent {
			baselineQuality = append(baselineQuality, oldQuality)
			candidateQuality = append(candidateQuality, newQuality)
		} else {
			qualityComplete = false
		}
		if !pair.baseline.success {
			baselineFailures++
		}
		if !pair.candidate.success {
			candidateFailures++
		}
		row.baselineMaxLatency = math.Max(row.baselineMaxLatency, pair.baseline.latencyMS)
		row.candidateMaxLatency = math.Max(row.candidateMaxLatency, pair.candidate.latencyMS)
	}
	if qualityComplete && len(baselineQuality) > 0 {
		row.baselineQuality = float64Reference(meanFloat64(baselineQuality))
		row.candidateQuality = float64Reference(meanFloat64(candidateQuality))
	}
	row.baselineFailure = float64(baselineFailures) / float64(len(pairs))
	row.candidateFailure = float64(candidateFailures) / float64(len(pairs))
	return row
}

func campaignPoolCaseAggregate(poolCase campaignPoolCasePair) *campaignCasePair {
	row := &campaignCasePair{trackID: "model_pool", caseID: poolCase.caseID}
	if quality, present := campaignPoolOracleQuality(poolCase.baseline); present {
		row.baselineQuality = float64Reference(quality)
	}
	if quality, present := campaignPoolOracleQuality(poolCase.candidate); present {
		row.candidateQuality = float64Reference(quality)
	}
	baselineFailures, candidateFailures := 0, 0
	for _, observation := range poolCase.baseline {
		if !observation.success {
			baselineFailures++
		}
		row.baselineMaxLatency = math.Max(row.baselineMaxLatency, observation.latencyMS)
	}
	for _, observation := range poolCase.candidate {
		if !observation.success {
			candidateFailures++
		}
		row.candidateMaxLatency = math.Max(row.candidateMaxLatency, observation.latencyMS)
	}
	if baselineFailures == len(poolCase.baseline) {
		row.baselineFailure = 1
	}
	if candidateFailures == len(poolCase.candidate) {
		row.candidateFailure = 1
	}
	return row
}

func campaignPoolOracleQuality(observations []campaignAttestedObservation) (float64, bool) {
	qualities := make([]float64, 0, len(observations))
	for _, observation := range observations {
		quality, present := campaignObservationQuality(observation)
		if !present {
			return 0, false
		}
		qualities = append(qualities, quality)
	}
	if len(qualities) == 0 {
		return 0, false
	}
	return maxFloat64(qualities), true
}

func campaignCasesForTrack(values []*campaignCasePair, trackID TrackID) []*campaignCasePair {
	rows := make([]*campaignCasePair, 0)
	for _, row := range values {
		if row.trackID == trackID {
			rows = append(rows, row)
		}
	}
	sort.Slice(rows, func(left, right int) bool { return rows[left].caseID < rows[right].caseID })
	return rows
}

func campaignTrackHasQualityStatistic(trackID TrackID) bool {
	return trackID == "routing" || trackID == "model_pool" || trackID == "joint" || trackID == "multimodal"
}

func campaignObservationQuality(record campaignAttestedObservation) (float64, bool) {
	if !record.success {
		return 0, true
	}
	if record.quality == nil {
		return 0, false
	}
	return *record.quality, true
}

func campaignQualityStatistic(trackID TrackID, cases []*campaignCasePair, seed int64) CampaignPairedStatistic {
	baseline, candidate, deltas := make([]float64, 0, len(cases)), make([]float64, 0, len(cases)), make([]float64, 0, len(cases))
	missing := 0
	for _, row := range cases {
		if row.baselineQuality == nil || row.candidateQuality == nil {
			missing++
			continue
		}
		oldValue, newValue := *row.baselineQuality, *row.candidateQuality
		baseline, candidate, deltas = append(baseline, oldValue), append(candidate, newValue), append(deltas, newValue-oldValue)
	}
	return campaignStatistic(
		"campaign.g3."+string(trackID)+".quality_non_inferiority", "G3", trackID,
		campaignQualityAnalysisUnit(trackID), "higher_is_better", campaignQualityMargin,
		baseline, candidate, deltas, missing, seed,
	)
}

func campaignQualityAnalysisUnit(trackID TrackID) string {
	if trackID == "model_pool" {
		return campaignPoolQualityUnit
	}
	return campaignQualityUnit
}

func campaignFailureStatistic(trackID TrackID, cases []*campaignCasePair, seed int64) CampaignPairedStatistic {
	baseline, candidate, deltas := make([]float64, 0, len(cases)), make([]float64, 0, len(cases)), make([]float64, 0, len(cases))
	for _, row := range cases {
		oldValue, newValue := row.baselineFailure, row.candidateFailure
		baseline, candidate, deltas = append(baseline, oldValue), append(candidate, newValue), append(deltas, newValue-oldValue)
	}
	return campaignStatistic(
		"campaign.g8."+string(trackID)+".failure_risk", "G8", trackID,
		campaignFailureAnalysisUnit(trackID), "lower_is_better", campaignFailureRiskMargin,
		baseline, candidate, deltas, 0, seed,
	)
}

func campaignFailureAnalysisUnit(trackID TrackID) string {
	if trackID == "model_pool" {
		return campaignPoolFailureUnit
	}
	return campaignFailureUnit
}

func campaignLatencyStatistic(trackID TrackID, cases []*campaignCasePair, seed int64) CampaignPairedStatistic {
	baseline, candidate, deltas := make([]float64, 0, len(cases)), make([]float64, 0, len(cases)), make([]float64, 0, len(cases))
	for _, row := range cases {
		denominator := math.Max(row.baselineMaxLatency, 1)
		baselineRatio := row.baselineMaxLatency / denominator
		candidateRatio := row.candidateMaxLatency / denominator
		baseline, candidate = append(baseline, baselineRatio), append(candidate, candidateRatio)
		deltas = append(deltas, candidateRatio-baselineRatio)
	}
	return campaignStatistic(
		"campaign.g8."+string(trackID)+".latency_risk", "G8", trackID,
		campaignLatencyUnit, "lower_is_better", campaignLatencyRiskMargin,
		baseline, candidate, deltas, 0, seed,
	)
}

func campaignStatistic(
	id, gateID string,
	trackID TrackID,
	analysisUnit, direction string,
	margin float64,
	baseline, candidate, deltas []float64,
	missing int,
	seed int64,
) CampaignPairedStatistic {
	statistic := CampaignPairedStatistic{
		ID: id, GateID: gateID, TrackID: trackID, AnalysisUnit: analysisUnit,
		Direction: direction, Margin: margin, ConfidenceLevel: campaignPairedConfidenceLevel,
		ConfidenceInterval: []float64{}, SampleCount: len(deltas), MissingPairs: missing,
		Verdict: "unavailable",
	}
	if len(deltas) > 0 && missing == 0 {
		baselineMean, candidateMean := meanFloat64(baseline), meanFloat64(candidate)
		statistic.BaselineValue = float64Reference(baselineMean)
		statistic.CandidateValue = float64Reference(candidateMean)
		statistic.Delta = float64Reference(candidateMean - baselineMean)
	}
	if len(deltas) < campaignPairedMinimumCases || missing != 0 {
		return statistic
	}
	statistic.ConfidenceInterval = pairedBootstrapInterval(deltas, metricSeed(seed, id))
	statistic.Verdict = campaignStatisticVerdict(statistic)
	return statistic
}

func campaignStatisticVerdict(statistic CampaignPairedStatistic) GateVerdict {
	if statistic.MissingPairs != 0 || statistic.SampleCount < campaignPairedMinimumCases || len(statistic.ConfidenceInterval) != 2 {
		return "unavailable"
	}
	lower, upper := statistic.ConfidenceInterval[0], statistic.ConfidenceInterval[1]
	if statistic.Direction == "higher_is_better" {
		if lower >= -statistic.Margin {
			return "pass"
		}
		if upper < -statistic.Margin {
			return "fail"
		}
		return "unavailable"
	}
	if upper <= statistic.Margin {
		return "pass"
	}
	if lower > statistic.Margin {
		return "fail"
	}
	return "unavailable"
}

func campaignWorstArmReliabilityVerdict(statistic CampaignPairedStatistic) GateVerdict {
	if statistic.SampleCount < campaignPairedMinimumCases || statistic.MissingPairs != 0 ||
		len(statistic.ConfidenceInterval) != 2 || len(statistic.CandidateConfidenceInterval) != 2 {
		return "unavailable"
	}
	deltaLower, deltaUpper := statistic.ConfidenceInterval[0], statistic.ConfidenceInterval[1]
	candidateLower, candidateUpper := statistic.CandidateConfidenceInterval[0], statistic.CandidateConfidenceInterval[1]
	minimum := frozenCampaignG3PromotionPolicy.MinimumCandidateArmReliability
	if deltaLower >= -statistic.Margin && candidateLower >= minimum {
		return "pass"
	}
	if deltaUpper < -statistic.Margin || candidateUpper < minimum {
		return "fail"
	}
	return "unavailable"
}
