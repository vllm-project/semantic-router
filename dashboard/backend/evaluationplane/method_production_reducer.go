package evaluationplane

import (
	"fmt"
	"math"
	"reflect"
	"time"
)

type productionMethodAttestation struct {
	AssignmentCount               int
	LedgerTotalAssignmentCount    int
	OutcomeCount                  int
	LedgerTotalOutcomeCount       int
	AssignmentSupport             *float64
	AssignmentBalancePValue       *float64
	RiskEventRate                 *float64
	RiskEventUpper95              *float64
	RiskBudgetMaxRate             *float64
	MinimumAssignmentCount        int
	ControlsOperational           *bool
	CandidateSafe                 *bool
	OutcomeCoverage               *float64
	CausalEligible                bool
	TargetIPSReward               *float64
	TargetSNIPSReward             *float64
	TargetSNIPSLower95            *float64
	TargetSNIPSUpper95            *float64
	ReferenceSNIPSReward          *float64
	TargetEffectiveSampleSize     *float64
	ReferenceEffectiveSampleSize  *float64
	TargetEffectiveSampleRatio    *float64
	ReferenceEffectiveSampleRatio *float64
	MinimumEffectiveSampleSize    float64
	MinimumEffectiveSampleRatio   float64
	SegmentCoverage               *float64
	SegmentCount                  int
	RewardLift                    *float64
	RewardLiftLower95             *float64
	RewardLiftUpper95             *float64
	MinimumRewardLift             float64
	PreferencePassed              *bool
	PolicySnapshotDigest          string
	ConfigDigest                  string
	TargetID                      string
	BackendTopologyDigest         string
	MixtureSnapshotDigest         string
}

func productionContractEqual(left, right productionExperimentMethodEvidence) bool {
	return left.ContractVersion == right.ContractVersion && left.ExperimentID == right.ExperimentID && left.LedgerID == right.LedgerID &&
		left.LedgerTotalAssignmentCount == right.LedgerTotalAssignmentCount && left.LedgerTotalOutcomeCount == right.LedgerTotalOutcomeCount &&
		left.SourceID == right.SourceID && left.PolicySnapshotDigest == right.PolicySnapshotDigest && left.ConfigDigest == right.ConfigDigest &&
		left.TargetID == right.TargetID && left.BackendTopologyDigest == right.BackendTopologyDigest &&
		left.MixtureSnapshotDigest == right.MixtureSnapshotDigest &&
		left.Environment == right.Environment && left.AssignmentScheme == right.AssignmentScheme && reflect.DeepEqual(left.PolicyArms, right.PolicyArms) &&
		reducedFloatsEqual(left.MinimumEffectiveSampleSize, right.MinimumEffectiveSampleSize) &&
		reducedFloatsEqual(left.MinimumEffectiveSampleRatio, right.MinimumEffectiveSampleRatio) &&
		left.MinimumSegmentSampleSize == right.MinimumSegmentSampleSize && left.MinimumAssignmentCount == right.MinimumAssignmentCount &&
		reducedFloatsEqual(left.MinimumRewardLift, right.MinimumRewardLift) && reducedFloatsEqual(left.ConfidenceLevel, right.ConfidenceLevel) &&
		reducedFloatsEqual(left.RiskBudgetMaxRate, right.RiskBudgetMaxRate) && left.StopRuleID == right.StopRuleID &&
		left.StopRuleEvaluatedAt.Equal(right.StopRuleEvaluatedAt) && left.StopTriggered == right.StopTriggered &&
		left.RollbackReceiptID == right.RollbackReceiptID && left.RollbackValidatedAt.Equal(right.RollbackValidatedAt) &&
		left.RollbackReady == right.RollbackReady && optionalTimesEqual(left.RollbackExecutedAt, right.RollbackExecutedAt) &&
		optionalBoolsEqual(left.RollbackSucceeded, right.RollbackSucceeded) && left.LedgerSealedAt.Equal(right.LedgerSealedAt)
}

func optionalTimesEqual(left, right *time.Time) bool {
	return (left == nil && right == nil) || (left != nil && right != nil && left.Equal(*right))
}

func optionalBoolsEqual(left, right *bool) bool {
	return (left == nil && right == nil) || (left != nil && right != nil && *left == *right)
}

func oneSidedWilsonUpper(events, total int) *float64 {
	if total <= 0 {
		return nil
	}
	const z = 1.6448536269514722
	rate := float64(events) / float64(total)
	zSquared := z * z
	denominator := 1 + zSquared/float64(total)
	center := rate + zSquared/(2*float64(total))
	spread := z * math.Sqrt(rate*(1-rate)/float64(total)+zSquared/(4*float64(total*total)))
	value := math.Min(1, (center+spread)/denominator)
	return &value
}

func effectiveSampleSize(weights []float64) *float64 {
	total, squares := 0.0, 0.0
	for _, weight := range weights {
		total += weight
		squares += weight * weight
	}
	if squares <= 0 {
		return nil
	}
	value := total * total / squares
	return &value
}

func selfNormalizedReward(weights, rewards []float64) *float64 {
	total, weighted := 0.0, 0.0
	for index, weight := range weights {
		total += weight
		weighted += weight * rewards[index]
	}
	if total <= 0 {
		return nil
	}
	value := weighted / total
	return &value
}

func reduceProductionMethod(records []executionRecordEvidence) (productionMethodAttestation, error) {
	rows := productionExperimentRows(records)
	if len(rows) == 0 {
		return productionMethodAttestation{}, nil
	}
	state := newProductionReductionState(*rows[0].ProductionExperiment, len(rows))
	for _, record := range rows {
		if err := state.observe(record); err != nil {
			return productionMethodAttestation{}, err
		}
	}
	attestation, completeOutcomes, err := state.baseAttestation()
	if err != nil || !completeOutcomes {
		return attestation, err
	}
	return state.completeOutcomeAttestation(attestation)
}

func productionExperimentRows(records []executionRecordEvidence) []executionRecordEvidence {
	rows := make([]executionRecordEvidence, 0)
	for _, record := range records {
		if record.TrackID == "preference" && record.ProductionExperiment != nil {
			rows = append(rows, record)
		}
	}
	return rows
}

type productionReductionState struct {
	contract         productionExperimentMethodEvidence
	rowCount         int
	assignments      map[string]struct{}
	exposures        map[string]struct{}
	participants     map[string]struct{}
	armCounts        map[string]int
	expectedCounts   map[string]float64
	riskEvents       int
	targetWeights    []float64
	referenceWeights []float64
	rewards          []float64
	segmentOutcomes  map[string]int
	allSegments      map[string]struct{}
}

func newProductionReductionState(
	contract productionExperimentMethodEvidence,
	rowCount int,
) *productionReductionState {
	return &productionReductionState{
		contract: contract, rowCount: rowCount,
		assignments: make(map[string]struct{}), exposures: make(map[string]struct{}),
		participants: make(map[string]struct{}), armCounts: make(map[string]int, len(contract.PolicyArms)),
		expectedCounts:  make(map[string]float64, len(contract.PolicyArms)),
		segmentOutcomes: make(map[string]int), allSegments: make(map[string]struct{}),
	}
}

func (state *productionReductionState) observe(record executionRecordEvidence) error {
	experiment := *record.ProductionExperiment
	if !productionContractEqual(state.contract, experiment) {
		return fmt.Errorf("production rows mix sealed experiment contracts")
	}
	if _, duplicate := state.assignments[experiment.AssignmentID]; duplicate {
		return fmt.Errorf("production assignment identities must be unique")
	}
	if _, duplicate := state.exposures[experiment.ExposureID]; duplicate {
		return fmt.Errorf("production exposure identities must be unique")
	}
	if _, duplicate := state.participants[experiment.ParticipantDigest]; duplicate {
		return fmt.Errorf("production participant identities must be unique")
	}
	state.assignments[experiment.AssignmentID] = struct{}{}
	state.exposures[experiment.ExposureID] = struct{}{}
	state.participants[experiment.ParticipantDigest] = struct{}{}
	state.armCounts[experiment.AssignedPolicyArmID]++
	state.allSegments[experiment.SegmentID] = struct{}{}
	if experiment.RiskEvent {
		state.riskEvents++
	}
	for _, arm := range experiment.PolicyArms {
		state.expectedCounts[arm.ID] += arm.AssignmentProbability
	}
	if record.OnlinePreference == nil {
		return nil
	}
	assigned := assignedProductionPolicyArm(experiment)
	state.targetWeights = append(state.targetWeights, assigned.TargetPolicyProbability/experiment.BehaviorPropensity)
	state.referenceWeights = append(state.referenceWeights, assigned.ReferencePolicyProbability/experiment.BehaviorPropensity)
	state.rewards = append(state.rewards, record.OnlinePreference.Outcome.Reward)
	state.segmentOutcomes[experiment.SegmentID]++
	return nil
}

func assignedProductionPolicyArm(experiment productionExperimentMethodEvidence) experimentPolicyArmEvidence {
	for _, arm := range experiment.PolicyArms {
		if arm.ID == experiment.AssignedPolicyArmID {
			return arm
		}
	}
	return experimentPolicyArmEvidence{}
}

func (state *productionReductionState) baseAttestation() (productionMethodAttestation, bool, error) {
	first := state.contract
	supported := 0
	chiSquare := 0.0
	for _, arm := range first.PolicyArms {
		if state.armCounts[arm.ID] > 0 {
			supported++
		}
		expected := state.expectedCounts[arm.ID]
		if expected <= 0 {
			return productionMethodAttestation{}, false, fmt.Errorf("production policy arm has zero expected support")
		}
		difference := float64(state.armCounts[arm.ID]) - expected
		chiSquare += difference * difference / expected
	}
	support := float64(supported) / float64(len(first.PolicyArms))
	balanceP := math.Erfc(math.Sqrt(chiSquare / 2))
	riskRate := float64(state.riskEvents) / float64(state.rowCount)
	riskUpper := oneSidedWilsonUpper(state.riskEvents, state.rowCount)
	riskBudget := first.RiskBudgetMaxRate
	sealedAssignments := state.rowCount == first.LedgerTotalAssignmentCount
	controlsOperational := first.RollbackReady && (!first.StopTriggered ||
		(first.RollbackExecutedAt != nil && first.RollbackSucceeded != nil && *first.RollbackSucceeded))
	candidateSafe := sealedAssignments && state.rowCount >= first.MinimumAssignmentCount && balanceP >= 0.01 &&
		riskUpper != nil && *riskUpper <= first.RiskBudgetMaxRate && first.RollbackReady && !first.StopTriggered && support == 1
	outcomeCoverage := float64(len(state.rewards)) / float64(state.rowCount)
	completeOutcomes := sealedAssignments && first.LedgerTotalOutcomeCount == first.LedgerTotalAssignmentCount &&
		len(state.rewards) == first.LedgerTotalOutcomeCount && outcomeCoverage == 1
	targetEffective := effectiveSampleSize(state.targetWeights)
	referenceEffective := effectiveSampleSize(state.referenceWeights)
	targetRatio := effectiveSampleRatio(targetEffective, state.rowCount)
	referenceRatio := effectiveSampleRatio(referenceEffective, state.rowCount)
	coveredSegments := 0
	for segment := range state.allSegments {
		if state.segmentOutcomes[segment] >= first.MinimumSegmentSampleSize {
			coveredSegments++
		}
	}
	segmentCoverage := float64(coveredSegments) / float64(len(state.allSegments))
	causalEligible := completeOutcomes && candidateSafe && targetEffective != nil && referenceEffective != nil &&
		*targetEffective >= first.MinimumEffectiveSampleSize && *referenceEffective >= first.MinimumEffectiveSampleSize &&
		targetRatio != nil && referenceRatio != nil && *targetRatio >= first.MinimumEffectiveSampleRatio &&
		*referenceRatio >= first.MinimumEffectiveSampleRatio && segmentCoverage == 1
	attestation := productionMethodAttestation{
		AssignmentCount: state.rowCount, LedgerTotalAssignmentCount: first.LedgerTotalAssignmentCount,
		OutcomeCount: len(state.rewards), LedgerTotalOutcomeCount: first.LedgerTotalOutcomeCount,
		AssignmentSupport: &support, AssignmentBalancePValue: &balanceP, RiskEventRate: &riskRate,
		RiskEventUpper95: riskUpper, RiskBudgetMaxRate: &riskBudget, MinimumAssignmentCount: first.MinimumAssignmentCount,
		ControlsOperational: &controlsOperational, CandidateSafe: &candidateSafe, OutcomeCoverage: &outcomeCoverage,
		CausalEligible: causalEligible, TargetEffectiveSampleSize: targetEffective, ReferenceEffectiveSampleSize: referenceEffective,
		TargetEffectiveSampleRatio: targetRatio, ReferenceEffectiveSampleRatio: referenceRatio,
		MinimumEffectiveSampleSize: first.MinimumEffectiveSampleSize, MinimumEffectiveSampleRatio: first.MinimumEffectiveSampleRatio,
		SegmentCoverage: &segmentCoverage, MinimumRewardLift: first.MinimumRewardLift,
		SegmentCount:         len(state.segmentOutcomes),
		PolicySnapshotDigest: first.PolicySnapshotDigest, ConfigDigest: first.ConfigDigest,
		TargetID: first.TargetID, BackendTopologyDigest: first.BackendTopologyDigest,
		MixtureSnapshotDigest: first.MixtureSnapshotDigest,
	}
	return attestation, completeOutcomes, nil
}

func effectiveSampleRatio(effective *float64, total int) *float64 {
	if effective == nil {
		return nil
	}
	value := *effective / float64(total)
	return &value
}

func (state *productionReductionState) completeOutcomeAttestation(
	attestation productionMethodAttestation,
) (productionMethodAttestation, error) {
	first := state.contract
	preferencePassed := false
	if attestation.CausalEligible {
		targetReward := selfNormalizedReward(state.targetWeights, state.rewards)
		referenceReward := selfNormalizedReward(state.referenceWeights, state.rewards)
		attestation.TargetSNIPSReward = targetReward
		attestation.ReferenceSNIPSReward = referenceReward
		if targetReward == nil || referenceReward == nil {
			return productionMethodAttestation{}, fmt.Errorf("causally eligible production window lost a policy reward")
		}
		ipsTotal := 0.0
		for index, reward := range state.rewards {
			ipsTotal += state.targetWeights[index] * reward
		}
		ipsReward := ipsTotal / float64(state.rowCount)
		attestation.TargetIPSReward = &ipsReward
		lift := *targetReward - *referenceReward
		count := len(state.rewards)
		targetWeightTotal, referenceWeightTotal := 0.0, 0.0
		for index := range state.rewards {
			targetWeightTotal += state.targetWeights[index]
			referenceWeightTotal += state.referenceWeights[index]
		}
		targetVariance := 0.0
		for index, reward := range state.rewards {
			difference := reward - *targetReward
			targetVariance += state.targetWeights[index] * state.targetWeights[index] * difference * difference
		}
		targetHalfWidth := 1.959963984540054 * math.Sqrt(math.Max(0, targetVariance/(targetWeightTotal*targetWeightTotal)))
		targetLower := math.Max(0, *targetReward-targetHalfWidth)
		targetUpper := math.Min(1, *targetReward+targetHalfWidth)
		attestation.TargetSNIPSLower95 = &targetLower
		attestation.TargetSNIPSUpper95 = &targetUpper
		influenceSquares := 0.0
		for index, reward := range state.rewards {
			influence := float64(count) * (state.targetWeights[index]*(reward-*targetReward)/targetWeightTotal -
				state.referenceWeights[index]*(reward-*referenceReward)/referenceWeightTotal)
			influenceSquares += influence * influence
		}
		standardError := math.Sqrt(influenceSquares / float64(count*(count-1)))
		halfWidth := 1.959963984540054 * standardError
		lower := math.Max(-1, lift-halfWidth)
		upper := math.Min(1, lift+halfWidth)
		attestation.RewardLift = &lift
		attestation.RewardLiftLower95 = &lower
		attestation.RewardLiftUpper95 = &upper
		preferencePassed = lower >= first.MinimumRewardLift
	}
	attestation.PreferencePassed = &preferencePassed
	return attestation, nil
}
