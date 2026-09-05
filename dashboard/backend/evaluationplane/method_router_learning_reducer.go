package evaluationplane

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

const routerLearningZ95 = 1.959963984540054

var routerLearningPolicyIDs = []string{"static-base", "routing-sampling", "beta-bernoulli"}

const routerLearningTrialCount = 8

type routerLearningPolicyAttestation struct {
	PolicyID                    string
	TrialCount                  int
	RoundCount                  int
	ProtectedRoundCount         int
	SolveRate                   *float64
	SolveRateInterval           []float64
	LifecycleCostMeanUSD        *float64
	LifecycleCostInterval       []float64
	LatencyMeanMS               *float64
	LatencyInterval             []float64
	ModelCallMean               *float64
	ModelCallInterval           []float64
	ProtectionViolationRate     *float64
	ProtectionViolationInterval []float64
	HardConstraintViolationRate *float64
	HardConstraintInterval      []float64
}

type routerLearningMethodAttestation struct {
	Policies map[string]routerLearningPolicyAttestation
}

type routerLearningTrialRows struct {
	seed int64
	rows []executionRecordEvidence
}

func reduceRouterLearningMethod(records []executionRecordEvidence) (routerLearningMethodAttestation, error) {
	byPolicy := make(map[string]map[string]*routerLearningTrialRows)
	candidateIDs := ""
	for _, record := range records {
		method := record.RouterLearning
		if method == nil {
			continue
		}
		signature := fmt.Sprint(method.CandidateArmIDs)
		if candidateIDs == "" {
			candidateIDs = signature
		} else if signature != candidateIDs {
			return routerLearningMethodAttestation{}, fmt.Errorf("router learning candidate set drifted")
		}
		trials := byPolicy[method.PolicyID]
		if trials == nil {
			trials = make(map[string]*routerLearningTrialRows)
			byPolicy[method.PolicyID] = trials
		}
		trial := trials[method.TrialID]
		if trial == nil {
			trial = &routerLearningTrialRows{seed: method.TrialSeed}
			trials[method.TrialID] = trial
		} else if trial.seed != method.TrialSeed {
			return routerLearningMethodAttestation{}, fmt.Errorf("router learning trial seed drifted")
		}
		for _, prior := range trial.rows {
			if prior.RouterLearning.RoundIndex == method.RoundIndex {
				return routerLearningMethodAttestation{}, fmt.Errorf("router learning round is duplicated")
			}
		}
		trial.rows = append(trial.rows, record)
	}
	if len(byPolicy) == 0 {
		return routerLearningMethodAttestation{}, nil
	}
	if len(byPolicy) != len(routerLearningPolicyIDs) {
		return routerLearningMethodAttestation{}, fmt.Errorf("router learning replay requires every policy")
	}
	var paired map[string]routerLearningTrialRows
	result := routerLearningMethodAttestation{Policies: make(map[string]routerLearningPolicyAttestation)}
	for _, policyID := range routerLearningPolicyIDs {
		trials := byPolicy[policyID]
		if len(trials) != routerLearningTrialCount {
			return routerLearningMethodAttestation{}, fmt.Errorf("router learning replay requires eight paired trials")
		}
		if paired == nil {
			paired = make(map[string]routerLearningTrialRows, len(trials))
			for trialID, trial := range trials {
				paired[trialID] = *trial
			}
		} else if err := validatePairedRouterLearningTrials(paired, trials); err != nil {
			return routerLearningMethodAttestation{}, err
		}
		attestation, err := reduceRouterLearningPolicy(policyID, trials)
		if err != nil {
			return routerLearningMethodAttestation{}, err
		}
		result.Policies[policyID] = attestation
	}
	return result, nil
}

func validatePairedRouterLearningTrials(reference map[string]routerLearningTrialRows, candidate map[string]*routerLearningTrialRows) error {
	if len(reference) != len(candidate) {
		return fmt.Errorf("router learning policy trial sets are not paired")
	}
	for trialID, baseline := range reference {
		trial := candidate[trialID]
		if trial == nil || trial.seed != baseline.seed || len(trial.rows) != len(baseline.rows) {
			return fmt.Errorf("router learning policy trials are not paired")
		}
		baselineByRound := make(map[int64]executionRecordEvidence, len(baseline.rows))
		for _, row := range baseline.rows {
			baselineByRound[row.RouterLearning.RoundIndex] = row
		}
		for _, row := range trial.rows {
			method := row.RouterLearning
			referenceRow, present := baselineByRound[method.RoundIndex]
			if !present || referenceRow.CaseID != row.CaseID ||
				!slices.Equal(referenceRow.RouterLearning.EligibleArmIDs, method.EligibleArmIDs) ||
				(referenceRow.RouterLearning.ProtectedArmID == nil) != (method.ProtectedArmID == nil) ||
				referenceRow.RouterLearning.FeedbackDelayRounds != method.FeedbackDelayRounds ||
				referenceRow.RouterLearning.FeedbackObserved != method.FeedbackObserved {
				return fmt.Errorf("router learning policy round protocols are not paired")
			}
			if method.ProtectedArmID != nil && *referenceRow.RouterLearning.ProtectedArmID != *method.ProtectedArmID {
				return fmt.Errorf("router learning policy protections are not paired")
			}
		}
	}
	return nil
}

func reduceRouterLearningPolicy(policyID string, trials map[string]*routerLearningTrialRows) (routerLearningPolicyAttestation, error) {
	trialIDs := make([]string, 0, len(trials))
	for trialID := range trials {
		trialIDs = append(trialIDs, trialID)
	}
	sort.Strings(trialIDs)
	solve, costs, latency, calls, protection, hard := []float64{}, []float64{}, []float64{}, []float64{}, []float64{}, []float64{}
	roundCount, protectedCount := 0, 0
	for _, trialID := range trialIDs {
		rows := trials[trialID].rows
		sort.Slice(rows, func(left, right int) bool {
			return rows[left].RouterLearning.RoundIndex < rows[right].RouterLearning.RoundIndex
		})
		if len(rows) == 0 {
			return routerLearningPolicyAttestation{}, fmt.Errorf("router learning trial is empty")
		}
		successTotal, costTotal, latencyTotal, callTotal, hardTotal := 0.0, 0.0, 0.0, 0.0, 0.0
		protectionTotal, trialProtected := 0.0, 0
		for index, row := range rows {
			method := row.RouterLearning
			if method.RoundIndex != int64(index) || row.LatencyMS == nil {
				return routerLearningPolicyAttestation{}, fmt.Errorf("router learning trial rounds are incomplete")
			}
			if method.OutcomeSuccess {
				successTotal++
			}
			costTotal += method.LifecycleCostUSD
			latencyTotal += *row.LatencyMS
			callTotal += float64(method.CallCount)
			if method.HardConstraintViolation {
				hardTotal++
			}
			if method.ProtectionRequired {
				trialProtected++
				if method.ProtectionViolation {
					protectionTotal++
				}
			}
		}
		count := float64(len(rows))
		solve = append(solve, successTotal/count)
		costs = append(costs, costTotal/count)
		latency = append(latency, latencyTotal/count)
		calls = append(calls, callTotal/count)
		hard = append(hard, hardTotal/count)
		if trialProtected > 0 {
			protection = append(protection, protectionTotal/float64(trialProtected))
		}
		roundCount += len(rows)
		protectedCount += trialProtected
	}
	return routerLearningPolicyAttestation{
		PolicyID: policyID, TrialCount: len(trials), RoundCount: roundCount, ProtectedRoundCount: protectedCount,
		SolveRate: methodFloatPointer(routerLearningMean(solve)), SolveRateInterval: routerLearningInterval(solve, true),
		LifecycleCostMeanUSD: methodFloatPointer(routerLearningMean(costs)), LifecycleCostInterval: routerLearningInterval(costs, false),
		LatencyMeanMS: methodFloatPointer(routerLearningMean(latency)), LatencyInterval: routerLearningInterval(latency, false),
		ModelCallMean: methodFloatPointer(routerLearningMean(calls)), ModelCallInterval: routerLearningInterval(calls, false),
		ProtectionViolationRate: optionalRouterLearningMean(protection), ProtectionViolationInterval: routerLearningInterval(protection, true),
		HardConstraintViolationRate: methodFloatPointer(routerLearningMean(hard)), HardConstraintInterval: routerLearningInterval(hard, true),
	}, nil
}

func routerLearningMean(values []float64) float64 {
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func optionalRouterLearningMean(values []float64) *float64 {
	if len(values) == 0 {
		return nil
	}
	return methodFloatPointer(routerLearningMean(values))
}

func routerLearningInterval(values []float64, bounded bool) []float64 {
	if len(values) < 2 {
		return nil
	}
	center := routerLearningMean(values)
	variance := 0.0
	for _, value := range values {
		delta := value - center
		variance += delta * delta
	}
	variance /= float64(len(values) - 1)
	margin := routerLearningZ95 * math.Sqrt(variance/float64(len(values)))
	lower, upper := center-margin, center+margin
	if bounded {
		lower = math.Max(0, lower)
		upper = math.Min(1, upper)
	}
	return []float64{lower, upper}
}
