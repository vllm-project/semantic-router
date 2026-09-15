package evaluationplane

import (
	"errors"
	"fmt"
	"os"
)

const unavailableCandidateBaselineMessage = "candidate baseline is unavailable to this evaluation principal"

// validateRunReferenceIntegrity rejects a store whose durable run graph cannot
// be reconstructed exactly. Baseline links are scientific cohort identities,
// not best-effort UI metadata.
func (s *Store) validateRunReferenceIntegrity() error {
	// Corrupt bundles remain quarantined by the run ledger. Valid candidates
	// must still resolve their baseline within the valid projection; otherwise
	// startup would publish a dangling scientific comparison graph.
	runs := s.runIndex.allRuns()
	byID := make(map[string]Run, len(runs))
	for _, run := range runs {
		byID[run.ID] = run
	}
	for _, run := range runs {
		if run.BaselineRunID == "" {
			continue
		}
		baseline, found := byID[run.BaselineRunID]
		if !found || baseline.ID == run.ID || !baseline.CreatedAt.Before(run.CreatedAt) {
			return fmt.Errorf("%w: run %s has a dangling or non-causal baseline reference", ErrInvalid, run.ID)
		}
	}
	return nil
}

func (s *Store) loadCompleteRunReferenceLedgerUnlocked() ([]Run, error) {
	if err := s.requireNoPendingRunPublications(); err != nil {
		return nil, err
	}
	if err := s.requireNoRunDeletionIntentsUnlocked(); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(s.runsRoot)
	if err != nil {
		return nil, fmt.Errorf("list evaluation runs: %w", err)
	}
	runs := make([]Run, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return nil, fmt.Errorf("evaluation run ledger contains an invalid entry")
		}
		run, readErr := s.getRunUnlocked(entry.Name())
		if readErr != nil {
			return nil, readErr
		}
		runs = append(runs, run)
	}
	return runs, nil
}

func (s *Store) ensureRunNotBaselineReferencedUnlocked(runID string) error {
	runs, err := s.loadCompleteRunReferenceLedgerUnlocked()
	if err != nil {
		return fmt.Errorf("%w: run reference ledger cannot be verified: %w", ErrConflict, err)
	}
	for _, run := range runs {
		if run.BaselineRunID == runID {
			return fmt.Errorf("%w: run is the baseline of run %s", ErrConflict, run.ID)
		}
	}
	return nil
}

// baselineSnapshotForCreateAs applies the ownership gate before returning any
// status or comparability fact. The Store publication path revalidates the same
// owner and reference while holding this lock order through the final rename.
func (s *Store) baselineSnapshotForCreateAs(
	actor Actor,
	baselineRunID string,
	candidateRunID string,
) (Run, error) {
	if err := validateActor(actor); err != nil {
		return Run{}, err
	}
	if !validClientRequestID(baselineRunID) || !validClientRequestID(candidateRunID) {
		return Run{}, fmt.Errorf("%w: candidate baseline identity is invalid", ErrInvalid)
	}
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	baseline, err := s.getRunUnlocked(baselineRunID)
	if err != nil {
		if !actor.administrator {
			reason := "invalid_evidence"
			if errors.Is(err, ErrNotFound) {
				reason = "not_found"
			}
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceRun, "create", "denied", reason,
				candidateRunID, "",
			); auditErr != nil {
				return Run{}, auditErr
			}
			return Run{}, fmt.Errorf("%w: %s", ErrForbidden, unavailableCandidateBaselineMessage)
		}
		return Run{}, fmt.Errorf("%w: baseline run is unavailable", ErrInvalid)
	}
	lifecycle, err := s.readRunLifecycle(baseline)
	if err != nil {
		if !actor.administrator {
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceRun, "create", "denied", "invalid_evidence",
				candidateRunID, "",
			); auditErr != nil {
				return Run{}, auditErr
			}
			return Run{}, fmt.Errorf("%w: %s", ErrForbidden, unavailableCandidateBaselineMessage)
		}
		return Run{}, fmt.Errorf("%w: baseline ownership is invalid", ErrInvalid)
	}
	if !actor.administrator && lifecycle.OwnerPrincipalDigest != actor.principalDigest {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "create", "denied", "not_owner",
			candidateRunID, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return Run{}, auditErr
		}
		return Run{}, fmt.Errorf("%w: %s", ErrForbidden, unavailableCandidateBaselineMessage)
	}
	return baseline, nil
}

func (s *Store) validateNewRunReferenceUnlocked(actor Actor, run Run) error {
	if run.BaselineRunID == "" {
		return nil
	}
	runs, ledgerErr := s.loadCompleteRunReferenceLedgerUnlocked()
	if ledgerErr != nil {
		return fmt.Errorf("%w: run reference ledger cannot be verified before candidate publication: %w", ErrConflict, ledgerErr)
	}
	var baseline Run
	found := false
	for _, stored := range runs {
		if stored.ID == run.BaselineRunID {
			baseline, found = stored, true
			break
		}
	}
	if !found {
		if !actor.administrator {
			if err := s.appendLifecycleDenialsUnlocked(
				actor, "create", "not_found", "", run.ID,
			); err != nil {
				return err
			}
			return fmt.Errorf("%w: %s", ErrForbidden, unavailableCandidateBaselineMessage)
		}
		return fmt.Errorf("%w: baseline is no longer a completed causal predecessor", ErrConflict)
	}
	lifecycle, err := s.readRunLifecycle(baseline)
	if err != nil {
		return fmt.Errorf("%w: baseline ownership cannot be verified", ErrConflict)
	}
	if !actor.administrator && actor.principalDigest != lifecycle.OwnerPrincipalDigest {
		if err := s.appendLifecycleDenialsUnlocked(
			actor, "create", "not_owner", lifecycle.OwnerPrincipalDigest, run.ID,
		); err != nil {
			return err
		}
		return fmt.Errorf("%w: %s", ErrForbidden, unavailableCandidateBaselineMessage)
	}
	if baseline.Status != StatusCompleted || !baseline.CreatedAt.Before(run.CreatedAt) {
		return fmt.Errorf("%w: baseline is no longer a completed causal predecessor", ErrConflict)
	}
	return nil
}
