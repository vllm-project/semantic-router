package evaluationplane

import (
	"fmt"
	"os"
	"path/filepath"
)

func (s *Store) controlledPairForRun(runID, runDir string) (controlledPairManifest, bool, error) {
	path := filepath.Join(runDir, controlledPairMembershipFile)
	if _, err := os.Lstat(path); os.IsNotExist(err) {
		return controlledPairManifest{}, false, nil
	} else if err != nil {
		return controlledPairManifest{}, false, err
	}
	var membership controlledPairMembership
	if err := readJSON(path, &membership); err != nil {
		return controlledPairManifest{}, false, err
	}
	if membership.SchemaVersion != SchemaVersion || membership.RunID != runID ||
		!validClientRequestID(membership.PairID) ||
		(membership.Role != controlledPairRoleBaseline && membership.Role != controlledPairRoleCandidate) {
		return controlledPairManifest{}, false, fmt.Errorf("%w: controlled pair membership is invalid", ErrInvalid)
	}
	pair, err := s.readControlledPair(membership.PairID)
	if err != nil {
		return controlledPairManifest{}, false, err
	}
	expectedID := pair.BaselineRunID
	if membership.Role == controlledPairRoleCandidate {
		expectedID = pair.CandidateRunID
	}
	if expectedID != runID {
		return controlledPairManifest{}, false, fmt.Errorf("%w: controlled pair membership does not match aggregate", ErrInvalid)
	}
	expectedRun := pair.BaselineRun
	if membership.Role == controlledPairRoleCandidate {
		expectedRun = pair.CandidateRun
	}
	if !controlledPairRunMembershipMatches(expectedRun, pair.PairID, membership.Role) {
		return controlledPairManifest{}, false, fmt.Errorf("%w: controlled pair run membership projection is invalid", ErrInvalid)
	}
	return pair, true, nil
}

func (s *Store) controlledPairRunSnapshot(runID, runDir string) (Run, bool, error) {
	pair, paired, err := s.controlledPairForRun(runID, runDir)
	if err != nil || !paired {
		return Run{}, false, err
	}
	if pair.State == controlledPairStatePublishing || pair.State == controlledPairStateDeleting ||
		pair.State == controlledPairStateDeleted {
		return Run{}, false, fmt.Errorf("%w: %w", ErrNotFound, errControlledPairNotCommitted)
	}
	if pair.State != controlledPairStateStarting && pair.State != controlledPairStateCancelling {
		return Run{}, false, nil
	}
	if runID == pair.BaselineRunID {
		return pair.BaselineRun, true, nil
	}
	return pair.CandidateRun, true, nil
}

func (s *Store) controlledPairEventLimit(runID, runDir string) (uint64, bool, error) {
	pair, paired, err := s.controlledPairForRun(runID, runDir)
	if err != nil || !paired || pair.State != controlledPairStateStarting {
		return 0, false, err
	}
	return 1, true, nil
}

func (s *Store) ensureRunNotControlledPairReferencedUnlocked(runID string) error {
	entries, err := os.ReadDir(s.controlledPairRoot)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return fmt.Errorf("%w: controlled pair reference ledger is invalid", ErrConflict)
		}
		pair, err := s.readControlledPair(entry.Name())
		if err != nil {
			return err
		}
		if pair.State == controlledPairStateDeleting || pair.State == controlledPairStateDeleted {
			continue
		}
		if runID == pair.BaselineSourceRunID || runID == pair.CandidateSourceRunID ||
			runID == pair.BaselineRunID || runID == pair.CandidateRunID {
			return fmt.Errorf("%w: controlled pair evidence requires aggregate retention", ErrConflict)
		}
	}
	return nil
}
