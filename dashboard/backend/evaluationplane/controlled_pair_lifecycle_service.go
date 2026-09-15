package evaluationplane

import "fmt"

func (s *Service) CancelControlledPairExecutionAs(actor Actor, pairID string) (ControlledPairExecution, error) {
	release, operationErr := s.beginOperation()
	if operationErr != nil {
		return ControlledPairExecution{}, operationErr
	}
	defer release()
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	pair, err := s.store.cancelControlledPairAs(actor, pairID)
	if err != nil {
		return ControlledPairExecution{}, err
	}
	s.activity.requestCancel(pair.BaselineRunID, pair.CandidateRunID)
	s.store.lifecycle.evidenceMu.Lock()
	defer s.store.lifecycle.evidenceMu.Unlock()
	s.store.runIndex.coordinator.Lock()
	defer s.store.runIndex.coordinator.Unlock()
	s.store.mu.Lock()
	defer s.store.mu.Unlock()
	return s.controlledPairExecutionAsUnlocked(actor, pair.PairID)
}

func (s *Service) DeleteControlledPairExecutionAs(actor Actor, pairID string) error {
	release, operationErr := s.beginOperation()
	if operationErr != nil {
		return operationErr
	}
	defer release()
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	pair, err := s.store.authorizeControlledPairServiceAction(actor, pairID, "delete")
	if err != nil {
		return err
	}
	baselineActive := s.active[pair.BaselineRunID] != nil || s.activity.contains(pair.BaselineRunID)
	candidateActive := s.active[pair.CandidateRunID] != nil || s.activity.contains(pair.CandidateRunID)
	if baselineActive || candidateActive {
		return fmt.Errorf("%w: controlled pair workers are still exiting", ErrConflict)
	}
	if err := s.store.deleteControlledPairAs(actor, pairID); err != nil {
		return err
	}
	s.cleanupDeletedRunSubscribersLocked(pair.BaselineRunID)
	s.cleanupDeletedRunSubscribersLocked(pair.CandidateRunID)
	return nil
}
