package evaluationplane

import "fmt"

func (s *Service) CompareAs(actor Actor, baselineRunID, candidateRunID string) (Comparison, error) {
	releaseOperation, operationErr := s.beginOperation()
	if operationErr != nil {
		return Comparison{}, operationErr
	}
	defer releaseOperation()
	s.store.lifecycle.mu.Lock()
	_, baselineAuthorizationErr := s.store.runForActorUnlocked(actor, baselineRunID)
	if baselineAuthorizationErr != nil {
		s.store.lifecycle.mu.Unlock()
		return Comparison{}, baselineAuthorizationErr
	}
	_, candidateAuthorizationErr := s.store.runForActorUnlocked(actor, candidateRunID)
	s.store.lifecycle.mu.Unlock()
	if candidateAuthorizationErr != nil {
		return Comparison{}, candidateAuthorizationErr
	}
	if ledgerErr := s.requireCompleteRunLedger(); ledgerErr != nil {
		return Comparison{}, ledgerErr
	}
	if baselineRunID == candidateRunID {
		return Comparison{}, fmt.Errorf("%w: baseline and candidate runs must be distinct", ErrInvalid)
	}
	// The complete-ledger refresh takes the evidence write lock. Reauthorize
	// after that refresh and transfer directly from both lifecycle identities to
	// one evidence read lease so deletion cannot race the comparison snapshot.
	release, acquireErr := s.acquireAuthorizedEvidenceRead(actor, baselineRunID, candidateRunID)
	if acquireErr != nil {
		return Comparison{}, acquireErr
	}
	defer release()
	baseline, baselineErr := s.decodedReport(baselineRunID)
	if baselineErr != nil {
		return Comparison{}, baselineErr
	}
	candidate, candidateErr := s.decodedReport(candidateRunID)
	if candidateErr != nil {
		return Comparison{}, candidateErr
	}
	if baseline.Run.TargetID != candidate.Run.TargetID {
		baselineEvidence, err := s.loadCampaignRunEvidence(campaignEvidenceBinding{
			slotID: "g3", gateID: "G3", bindingRole: "baseline", runID: baselineRunID,
		}, nil)
		if err != nil {
			return Comparison{}, err
		}
		candidateEvidence, err := s.loadCampaignRunEvidence(campaignEvidenceBinding{
			slotID: "g3", gateID: "G3", bindingRole: "candidate", runID: candidateRunID,
			candidate: true,
		}, nil)
		if err != nil {
			return Comparison{}, err
		}
		return compareControlledPairReports(baselineEvidence, candidateEvidence)
	}
	if cohortErr := validatePairedReportCohort(baseline, candidate); cohortErr != nil {
		return Comparison{}, cohortErr
	}
	baselineRecords, baselineRecordsErr := s.loadPrivateComparisonRecords(baselineRunID)
	if baselineRecordsErr != nil {
		return Comparison{}, baselineRecordsErr
	}
	candidateRecords, candidateRecordsErr := s.loadPrivateComparisonRecords(candidateRunID)
	if candidateRecordsErr != nil {
		return Comparison{}, candidateRecordsErr
	}
	return comparePairedReports(baseline, candidate, baselineRecords, candidateRecords)
}
