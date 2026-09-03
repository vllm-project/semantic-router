package evaluationplane

import (
	"fmt"
	"os"
)

// requireUnreservedControlledPairMemberIDsUnlocked treats every aggregate,
// including a deleted tombstone, as the permanent source of truth for its two
// member Run identities. The caller owns lifecycle, evidence-publication,
// index, and Store locks, so creation cannot race aggregate publication.
func (s *Store) requireUnreservedControlledPairMemberIDsUnlocked(
	actor Actor,
	runIDs ...string,
) error {
	if err := validateActor(actor); err != nil {
		return err
	}
	requested := make(map[string]bool, len(runIDs))
	for _, runID := range runIDs {
		if !validClientRequestID(runID) {
			return fmt.Errorf("%w: run identity must be a canonical UUID", ErrInvalid)
		}
		requested[runID] = true
	}
	entries, err := os.ReadDir(s.controlledPairRoot)
	if err != nil {
		return fmt.Errorf("inspect controlled pair identity reservations: %w", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return fmt.Errorf("%w: controlled pair identity reservation ledger is invalid", ErrInvalid)
		}
		pair, err := s.readControlledPair(entry.Name())
		if err != nil {
			return fmt.Errorf("%w: controlled pair identity reservation ledger is invalid", ErrInvalid)
		}
		collisions := make([]string, 0, 2)
		for _, reserved := range []string{pair.BaselineRunID, pair.CandidateRunID} {
			if requested[reserved] {
				collisions = append(collisions, reserved)
			}
		}
		if len(collisions) == 0 {
			continue
		}
		reason := "conflict"
		result := fmt.Errorf("%w: run identity is permanently reserved", ErrConflict)
		if !actor.administrator && actor.principalDigest != pair.OwnerPrincipalDigest {
			reason = "not_owner"
			result = fmt.Errorf("%w: run identity is reserved by another evaluation principal", ErrForbidden)
		}
		if err := s.appendLifecycleDenialsUnlocked(
			actor, "create", reason, pair.OwnerPrincipalDigest, collisions...,
		); err != nil {
			return err
		}
		return result
	}
	return nil
}
