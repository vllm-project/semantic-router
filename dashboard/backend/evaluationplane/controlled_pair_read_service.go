package evaluationplane

import "fmt"

func (s *Service) GetControlledPairExecutionAs(actor Actor, pairID string) (ControlledPairExecution, error) {
	release, err := s.beginOperation()
	if err != nil {
		return ControlledPairExecution{}, err
	}
	defer release()
	return s.getControlledPairExecutionAs(actor, pairID)
}

func (s *Service) getControlledPairExecutionAs(actor Actor, pairID string) (ControlledPairExecution, error) {
	if err := validateActor(actor); err != nil {
		return ControlledPairExecution{}, err
	}
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.store.lifecycle.evidenceMu.Lock()
	defer s.store.lifecycle.evidenceMu.Unlock()
	s.store.runIndex.coordinator.Lock()
	defer s.store.runIndex.coordinator.Unlock()
	s.store.mu.Lock()
	defer s.store.mu.Unlock()
	return s.controlledPairExecutionAsUnlocked(actor, pairID)
}

// controlledPairExecutionAsUnlocked builds the public response from one
// physical, policy, reference, and active-worker snapshot. Callers hold the
// lifecycle, service, evidence, index, and store locks in that order.
func (s *Service) controlledPairExecutionAsUnlocked(actor Actor, pairID string) (ControlledPairExecution, error) {
	pair, err := s.store.readControlledPair(pairID)
	if err != nil {
		return ControlledPairExecution{}, err
	}
	if pair.State == controlledPairStateDeleted || pair.State == controlledPairStateDeleting {
		return ControlledPairExecution{}, fmt.Errorf("%w: controlled pair %s", ErrNotFound, pairID)
	}
	if pair.OwnerPrincipalDigest != actor.principalDigest && !actor.administrator {
		return ControlledPairExecution{}, fmt.Errorf("%w: controlled pair belongs to another evaluation principal", ErrForbidden)
	}
	baseline, err := s.store.getRunPhysical(pair.BaselineRunID)
	if err != nil {
		return ControlledPairExecution{}, err
	}
	candidate, err := s.store.getRunPhysical(pair.CandidateRunID)
	if err != nil {
		return ControlledPairExecution{}, err
	}
	baselineActive := s.active[pair.BaselineRunID] != nil || s.activity.contains(pair.BaselineRunID)
	candidateActive := s.active[pair.CandidateRunID] != nil || s.activity.contains(pair.CandidateRunID)
	active := baselineActive || candidateActive
	capabilities := s.store.controlledPairPreflightCapabilitiesUnlocked(pair, baseline, candidate, active)
	return ControlledPairExecution{
		SchemaVersion: SchemaVersion, ContractVersion: controlledPairProtocolVersion,
		ID: pair.PairID, Protocol: pair.Protocol,
		BaselineSourceRunID: pair.BaselineSourceRunID, CandidateSourceRunID: pair.CandidateSourceRunID,
		BaselineRun: baseline, CandidateRun: candidate, State: pair.State, Capabilities: capabilities,
	}, nil
}
