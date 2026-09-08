package evaluationplane

const (
	campaignG3CandidateNormalizedRegretID = "campaign.g3.candidate_normalized_regret"
	campaignG3PairedNormalizedRegretID    = "campaign.g3.paired_normalized_regret_delta"
	campaignG3NoInformationFrontierID     = "campaign.g3.no_information_frontier_lift"
	campaignG3JointReliabilityID          = "campaign.g3.joint_reliability"
	campaignG3AllArmFailureID             = "campaign.g3.all_arm_failure_rate"
)

var frozenCampaignG3PromotionPolicy = CampaignG3PromotionPolicy{
	CandidateNormalizedRegretMaximum: defaultNormalizedRegretMaximum,
	PairedNormalizedRegretMargin:     comparisonG3RelativeMargin,
	MinimumNoInformationFrontierLift: 0.05,
	MinimumJointReliability:          0.80,
	MaximumAllArmFailureRate:         0.20,
	MinimumCandidateArmReliability:   0.80,
}

type campaignG3PromotionStatisticContract struct {
	id        string
	direction string
	threshold GateThreshold
}

func campaignG3PromotionStatisticContracts() []campaignG3PromotionStatisticContract {
	policy := frozenCampaignG3PromotionPolicy
	return []campaignG3PromotionStatisticContract{
		{id: campaignG3CandidateNormalizedRegretID, direction: "lower_is_better", threshold: GateThreshold{Operator: "<=", Value: policy.CandidateNormalizedRegretMaximum, Unit: "fraction"}},
		{id: campaignG3PairedNormalizedRegretID, direction: "lower_is_better", threshold: GateThreshold{Operator: "<=", Value: policy.PairedNormalizedRegretMargin, Unit: "fraction"}},
		{id: campaignG3NoInformationFrontierID, direction: "higher_is_better", threshold: GateThreshold{Operator: ">=", Value: policy.MinimumNoInformationFrontierLift, Unit: "quality"}},
		{id: campaignG3JointReliabilityID, direction: "higher_is_better", threshold: GateThreshold{Operator: ">=", Value: policy.MinimumJointReliability, Unit: "fraction"}},
		{id: campaignG3AllArmFailureID, direction: "lower_is_better", threshold: GateThreshold{Operator: "<=", Value: policy.MaximumAllArmFailureRate, Unit: "fraction"}},
	}
}

type campaignG3CaseVectors struct {
	candidateRegret  []float64
	regretDelta      []float64
	frontierLift     []float64
	jointReliability []float64
	allArmFailure    []float64
	missing          int
}

func campaignG3PromotionStatistics(
	cohort campaignPairedCohort,
	seed int64,
) []CampaignG3PromotionStatistic {
	vectors := campaignG3PromotionCaseVectors(cohort)
	contracts := campaignG3PromotionStatisticContracts()
	values := [][]float64{
		vectors.candidateRegret,
		vectors.regretDelta,
		vectors.frontierLift,
		vectors.jointReliability,
		vectors.allArmFailure,
	}
	statistics := make([]CampaignG3PromotionStatistic, 0, len(contracts))
	for index, contract := range contracts {
		statistic := CampaignG3PromotionStatistic{
			ID: contract.id, Direction: contract.direction, ConfidenceLevel: campaignPairedConfidenceLevel,
			ConfidenceInterval: []float64{}, Threshold: contract.threshold,
			SampleCount: len(values[index]), MissingCases: vectors.missing, Verdict: "unavailable",
		}
		if len(values[index]) > 0 {
			statistic.Estimate = meanFloat64(values[index])
		}
		if statistic.SampleCount >= campaignPairedMinimumCases && statistic.MissingCases == 0 {
			statistic.ConfidenceInterval = pairedBootstrapInterval(values[index], metricSeed(seed, contract.id))
		}
		statistic.Verdict = campaignG3PromotionStatisticVerdict(statistic)
		statistics = append(statistics, statistic)
	}
	return statistics
}

func campaignG3PromotionCaseVectors(cohort campaignPairedCohort) campaignG3CaseVectors {
	result := campaignG3CaseVectors{}
	jointByCase := make(map[string][]campaignObservationPair)
	for _, pair := range cohort.exactPairs {
		if pair.baseline.trackID == "joint" {
			jointByCase[pair.baseline.caseID] = append(jointByCase[pair.baseline.caseID], pair)
		}
	}
	// The no-information frontier is not zero. It is the strongest static
	// candidate-pool policy: one fixed arm, chosen by mean quality over the
	// complete dense case x arm cohort. Per-case lift is then measured against
	// that same predeclared arm so routing must add information value rather
	// than merely produce positive absolute quality.
	staticQualityByCase, staticFrontierComplete := campaignCandidateStaticFrontier(cohort.poolCases)
	seen := make(map[string]bool, len(cohort.poolCases))
	for _, poolCase := range cohort.poolCases {
		seen[poolCase.caseID] = true
		joint := jointByCase[poolCase.caseID]
		baselineOracle, baselineOK := campaignCompleteOracleQuality(poolCase.baseline)
		candidateOracle, candidateOK := campaignCompleteOracleQuality(poolCase.candidate)
		baselineRealized, candidateRealized, jointReliability, jointOK := campaignCompleteJointCase(joint)
		staticQuality, staticCaseOK := staticQualityByCase[poolCase.caseID]
		missing := !baselineOK || !candidateOK || !jointOK || !staticFrontierComplete || !staticCaseOK ||
			len(poolCase.baseline) == 0 || len(poolCase.candidate) == 0
		if missing {
			result.missing++
		}
		baselineRegret := campaignNormalizedRegretOrWorst(baselineOracle, baselineRealized, baselineOK && jointOK)
		candidateRegret := campaignNormalizedRegretOrWorst(candidateOracle, candidateRealized, candidateOK && jointOK)
		result.candidateRegret = append(result.candidateRegret, candidateRegret)
		result.regretDelta = append(result.regretDelta, candidateRegret-baselineRegret)
		frontierLift := 0.0
		if jointOK && staticFrontierComplete && staticCaseOK {
			frontierLift = candidateRealized - staticQuality
		}
		result.frontierLift = append(result.frontierLift, frontierLift)
		result.jointReliability = append(result.jointReliability, jointReliability)
		allFailed := 1.0
		if candidateOK && candidateOracle > 0 && campaignAnySuccessfulArm(poolCase.candidate) {
			allFailed = 0
		}
		result.allArmFailure = append(result.allArmFailure, allFailed)
	}
	for caseID := range jointByCase {
		if !seen[caseID] {
			result.missing++
		}
	}
	return result
}

// campaignCandidateStaticFrontier returns the per-case quality of the one
// candidate arm with the highest cohort mean. It independently rechecks the
// dense matrix instead of assuming an earlier alignment step will always be
// the only caller of the promotion reducer.
func campaignCandidateStaticFrontier(
	poolCases []campaignPoolCasePair,
) (map[string]float64, bool) {
	if len(poolCases) == 0 {
		return nil, false
	}
	qualityByCase := make(map[string]map[string]float64, len(poolCases))
	qualityByArm := make(map[string][]float64)
	seenCases := make(map[string]bool, len(poolCases))
	var expectedArms map[string]bool
	for _, poolCase := range poolCases {
		if poolCase.caseID == "" || seenCases[poolCase.caseID] || len(poolCase.candidate) == 0 {
			return nil, false
		}
		seenCases[poolCase.caseID] = true
		caseQualities := make(map[string]float64, len(poolCase.candidate))
		caseArms := make(map[string]bool, len(poolCase.candidate))
		caseComplete := true
		for _, observation := range poolCase.candidate {
			quality, present := campaignObservationQuality(observation)
			if observation.armID == "" || caseArms[observation.armID] {
				return nil, false
			}
			caseArms[observation.armID] = true
			if !present {
				caseComplete = false
				continue
			}
			caseQualities[observation.armID] = quality
		}
		if expectedArms == nil {
			expectedArms = make(map[string]bool, len(caseArms))
			for armID := range caseArms {
				expectedArms[armID] = true
			}
		} else if len(caseArms) != len(expectedArms) {
			return nil, false
		}
		for armID := range expectedArms {
			if !caseArms[armID] {
				return nil, false
			}
		}
		if !caseComplete {
			continue
		}
		qualityByCase[poolCase.caseID] = caseQualities
		for armID, quality := range caseQualities {
			qualityByArm[armID] = append(qualityByArm[armID], quality)
		}
	}
	if len(qualityByCase) == 0 {
		return nil, false
	}
	bestArm, bestMean := "", -1.0
	for _, armID := range sortedMapKeys(expectedArms) {
		values := qualityByArm[armID]
		if len(values) != len(qualityByCase) {
			return nil, false
		}
		mean := meanFloat64(values)
		if bestArm == "" || mean > bestMean {
			bestArm, bestMean = armID, mean
		}
	}
	if bestArm == "" {
		return nil, false
	}
	frontier := make(map[string]float64, len(qualityByCase))
	for caseID, qualities := range qualityByCase {
		frontier[caseID] = qualities[bestArm]
	}
	return frontier, true
}

func campaignCompleteOracleQuality(observations []campaignAttestedObservation) (float64, bool) {
	if len(observations) == 0 {
		return 0, false
	}
	values := make([]float64, 0, len(observations))
	for _, observation := range observations {
		value, ok := campaignObservationQuality(observation)
		if !ok {
			return 0, false
		}
		values = append(values, value)
	}
	return maxFloat64(values), true
}

func campaignCompleteJointCase(pairs []campaignObservationPair) (float64, float64, float64, bool) {
	if len(pairs) == 0 {
		return 0, 0, 0, false
	}
	baseline, candidate, reliability := make([]float64, 0, len(pairs)), make([]float64, 0, len(pairs)), make([]float64, 0, len(pairs))
	for _, pair := range pairs {
		left, leftOK := campaignObservationQuality(pair.baseline)
		right, rightOK := campaignObservationQuality(pair.candidate)
		if !leftOK || !rightOK {
			return 0, 0, 0, false
		}
		baseline, candidate = append(baseline, left), append(candidate, right)
		if pair.candidate.success {
			reliability = append(reliability, 1)
		} else {
			reliability = append(reliability, 0)
		}
	}
	return meanFloat64(baseline), meanFloat64(candidate), meanFloat64(reliability), true
}

func campaignNormalizedRegretOrWorst(oracle, realized float64, complete bool) float64 {
	if !complete || oracle <= 0 {
		return 1
	}
	return max(0, oracle-realized) / oracle
}

func campaignAnySuccessfulArm(observations []campaignAttestedObservation) bool {
	for _, observation := range observations {
		if observation.success {
			return true
		}
	}
	return false
}

func campaignG3PromotionStatisticVerdict(statistic CampaignG3PromotionStatistic) GateVerdict {
	if statistic.SampleCount < campaignPairedMinimumCases || statistic.MissingCases != 0 ||
		len(statistic.ConfidenceInterval) != 2 {
		return "unavailable"
	}
	lower, upper := statistic.ConfidenceInterval[0], statistic.ConfidenceInterval[1]
	if statistic.Direction == "higher_is_better" {
		if lower >= statistic.Threshold.Value {
			return "pass"
		}
		if upper < statistic.Threshold.Value {
			return "fail"
		}
		return "unavailable"
	}
	if upper <= statistic.Threshold.Value {
		return "pass"
	}
	if lower > statistic.Threshold.Value {
		return "fail"
	}
	return "unavailable"
}
