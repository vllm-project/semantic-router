package evaluationplane

import (
	"fmt"
	"os"
	"path/filepath"
)

// acquireControlledPairMutationBarrier detects the immutable membership
// sidecar before taking lifecycle.mu. Pair deletion cannot race this check into
// an unsafe write: deletion holds lifecycle.mu and the mutation revalidates the
// physical bundle after acquiring it. Ordinary runs avoid the shared barrier.
func (s *Store) acquireControlledPairMutationBarrier(runID string) (bool, error) {
	runDir, err := s.checkedRunDirPhysical(runID)
	if err != nil {
		return false, err
	}
	path := filepath.Join(runDir, controlledPairMembershipFile)
	if _, err := os.Lstat(path); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	var membership controlledPairMembership
	if err := readJSON(path, &membership); err != nil {
		return false, err
	}
	if membership.SchemaVersion != SchemaVersion || membership.RunID != runID ||
		!validClientRequestID(membership.PairID) ||
		(membership.Role != controlledPairRoleBaseline && membership.Role != controlledPairRoleCandidate) {
		return false, fmt.Errorf("%w: controlled pair mutation membership is invalid", ErrInvalid)
	}
	s.lifecycle.mu.Lock()
	return true, nil
}

func (s *Store) releaseControlledPairMutationBarrier(acquired bool) {
	if acquired {
		s.lifecycle.mu.Unlock()
	}
}
