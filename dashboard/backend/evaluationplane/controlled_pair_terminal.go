package evaluationplane

import (
	"fmt"
	"path/filepath"
)

func (s *Store) refreshControlledPairTerminalState(runID string) error {
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.refreshControlledPairTerminalStateUnlocked(runID)
}

func (s *Store) refreshControlledPairTerminalStateUnlocked(runID string) error {
	runDir, err := s.checkedRunDirPhysical(runID)
	if err != nil {
		return err
	}
	pair, paired, err := s.controlledPairForRun(runID, runDir)
	if err != nil || !paired || pair.State == controlledPairStateTerminal {
		return err
	}
	if pair.State != controlledPairStateRunning {
		return fmt.Errorf("%w: controlled pair terminal reconciliation requires a running aggregate", ErrConflict)
	}
	baseline, err := s.getRunPhysical(pair.BaselineRunID)
	if err != nil {
		return err
	}
	candidate, err := s.getRunPhysical(pair.CandidateRunID)
	if err != nil {
		return err
	}
	if !terminalStatus(baseline.Status) || !terminalStatus(candidate.Status) {
		return nil
	}
	pair.State = controlledPairStateTerminal
	pair.BaselineRun, pair.CandidateRun = baseline, candidate
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		if err := s.syncRunStatusDirectory(
			filepath.Join(s.runsRoot, runID), "controlled pair terminal reconciliation",
		); err != nil {
			return err
		}
	}
	return s.writeControlledPairDurably(pair)
}

func (s *Store) getRunPhysical(id string) (Run, error) {
	runDir, err := s.checkedRunDirPhysical(id)
	if err != nil {
		return Run{}, err
	}
	var run Run
	if err := readJSON(filepath.Join(runDir, runFileName), &run); err != nil {
		return Run{}, err
	}
	if err := validateStoredRun(id, run); err != nil {
		return Run{}, fmt.Errorf("validate evaluation run status: %w", err)
	}
	return run, nil
}
