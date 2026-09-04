package evaluationplane

import (
	"fmt"
	"reflect"
)

// validateCampaignControlledPairCohort rejects any post-hoc pairing. Every
// shared observation must prove one server-owned AB/BA block, and one-sided
// model-pool arms must carry the explicit treatment cohort assigned at run time.
func validateCampaignControlledPairCohort(cohort campaignPairedCohort) (string, error) {
	sessionID := ""
	pairedCount := 0
	bindSession := func(observation campaignAttestedObservation) error {
		pair := observation.controlledPair
		if pair == nil || pair.Protocol != controlledPairInterleaveABBA ||
			pair.ContractVersion != controlledPairProtocolVersion {
			return fmt.Errorf("%w: live observation lacks server-owned controlled pairing provenance", ErrInvalid)
		}
		if sessionID == "" {
			sessionID = pair.SessionID
		} else if pair.SessionID != sessionID {
			return fmt.Errorf("%w: live observations mix controlled pair sessions", ErrInvalid)
		}
		return nil
	}
	for _, pair := range cohort.exactPairs {
		if err := bindSession(pair.baseline); err != nil {
			return "", err
		}
		if err := bindSession(pair.candidate); err != nil {
			return "", err
		}
		if err := validateCampaignControlledObservationPair(pair.baseline, pair.candidate); err != nil {
			return "", err
		}
		pairedCount++
	}
	for _, poolCase := range cohort.poolCases {
		baselineByArm := make(map[string]campaignAttestedObservation, len(poolCase.baseline))
		candidateByArm := make(map[string]campaignAttestedObservation, len(poolCase.candidate))
		for _, observation := range poolCase.baseline {
			if err := bindSession(observation); err != nil {
				return "", err
			}
			baselineByArm[observation.armID] = observation
		}
		for _, observation := range poolCase.candidate {
			if err := bindSession(observation); err != nil {
				return "", err
			}
			candidateByArm[observation.armID] = observation
		}
		for armID, baseline := range baselineByArm {
			candidate, shared := candidateByArm[armID]
			if !shared {
				if baseline.controlledPair.Cohort != campaignArmCohortBaselineOnly {
					return "", fmt.Errorf("%w: removed arm %q is not attested as baseline-only", ErrInvalid, armID)
				}
				continue
			}
			if err := validateCampaignControlledObservationPair(baseline, candidate); err != nil {
				return "", err
			}
			pairedCount++
		}
		for armID, candidate := range candidateByArm {
			if _, shared := baselineByArm[armID]; !shared &&
				candidate.controlledPair.Cohort != campaignArmCohortCandidateOnly {
				return "", fmt.Errorf("%w: added arm %q is not attested as candidate-only", ErrInvalid, armID)
			}
		}
	}
	if sessionID == "" || pairedCount == 0 {
		return "", fmt.Errorf("%w: controlled pair produced no shared paired observations", ErrInvalid)
	}
	return sessionID, nil
}

func validateCampaignControlledObservationPair(
	baseline campaignAttestedObservation,
	candidate campaignAttestedObservation,
) error {
	left, right := baseline.controlledPair, candidate.controlledPair
	if left == nil || right == nil || left.Cohort != campaignArmCohortPaired ||
		right.Cohort != campaignArmCohortPaired || left.Role != controlledPairRoleBaseline ||
		right.Role != controlledPairRoleCandidate || left.SessionID != right.SessionID ||
		left.Protocol != right.Protocol || left.BlockID != right.BlockID ||
		left.CoordinateDigest != right.CoordinateDigest || left.Order != right.Order ||
		left.AttemptID != right.AttemptID || !reflect.DeepEqual(left.Load, right.Load) ||
		left.Position == right.Position || left.Position+right.Position != 3 {
		return fmt.Errorf("%w: paired observations do not share one exact controlled block", ErrInvalid)
	}
	first, second := left, right
	if right.Position == 1 {
		first, second = right, left
	}
	if first.CompletedAt.IsZero() || second.ObservedAt.IsZero() ||
		first.CompletedAt.After(second.ObservedAt) {
		return fmt.Errorf("%w: controlled pair timestamps do not prove first completion before second admission", ErrInvalid)
	}
	return nil
}
