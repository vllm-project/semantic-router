package evaluationplane

import (
	"errors"
	"fmt"
)

// runForActorUnlocked is the single ownership check used by actor-aware run
// reads. Callers hold lifecycle.mu so authorization and evidence lookup cannot
// race a lifecycle mutation or deletion.
func (s *Store) runForActorUnlocked(actor Actor, runID string) (Run, error) {
	if err := validateActor(actor); err != nil {
		return Run{}, err
	}
	run, lifecycle, err := s.getRunWithLifecycleUnlocked(runID)
	if err != nil {
		return Run{}, err
	}
	projectedOwner, exists := s.runIndex.ownerDigest(runID)
	if !exists || projectedOwner != lifecycle.OwnerPrincipalDigest {
		return Run{}, fmt.Errorf("%w: run ownership projection is unavailable or inconsistent", ErrInvalid)
	}
	if !actor.administrator && actor.principalDigest != lifecycle.OwnerPrincipalDigest {
		return Run{}, fmt.Errorf("%w: run belongs to another evaluation principal", ErrForbidden)
	}
	return run, nil
}

// acquireAuthorizedEvidenceRead transfers protection from the lifecycle
// identity lock directly to the shared evidence read lease. A run therefore
// cannot be deleted and rebound to another principal between authorization and
// the report, artifact, or event read that follows.
func (s *Service) acquireAuthorizedEvidenceRead(
	actor Actor,
	runID string,
	additionalRunIDs ...string,
) (func(), error) {
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	if _, err := s.store.runForActorUnlocked(actor, runID); err != nil {
		return nil, err
	}
	for _, additionalRunID := range additionalRunIDs {
		if _, err := s.store.runForActorUnlocked(actor, additionalRunID); err != nil {
			return nil, err
		}
	}
	return s.acquireEvidenceRead()
}

func (s *Store) appendLifecycleDenialsUnlocked(
	actor Actor,
	action, reason, ownerDigest string,
	runIDs ...string,
) error {
	for _, runID := range runIDs {
		if _, err := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, action, "denied", reason, runID, ownerDigest,
		); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) auditExistingCreateUnlocked(actor Actor, run Run) error {
	if err := validateActor(actor); err != nil {
		return err
	}
	lifecycle, err := s.readRunLifecycle(run)
	if err != nil {
		return err
	}
	if !actor.administrator && actor.principalDigest != lifecycle.OwnerPrincipalDigest {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "create", "denied", "not_owner", run.ID, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return auditErr
		}
		return fmt.Errorf("%w: client_request_id belongs to another evaluation principal", ErrForbidden)
	}
	_, err = s.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceRun, "create", "allowed", lifecycleOwnerAuthorizationReason(actor, lifecycle.OwnerPrincipalDigest), run.ID, lifecycle.OwnerPrincipalDigest,
	)
	return err
}

func (s *Store) authorizeRunActionUnlocked(actor Actor, runID, action string) error {
	if err := validateActor(actor); err != nil {
		return err
	}
	if !validClientRequestID(runID) || !validLifecycleAction(action) || action == "create" || action == "gc" {
		return fmt.Errorf("%w: lifecycle action identity is invalid", ErrInvalid)
	}
	run, authorizationErr := s.getRunUnlocked(runID)
	if authorizationErr != nil {
		if errors.Is(authorizationErr, ErrNotFound) {
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceRun, action, "denied", "not_found", runID, "",
			); auditErr != nil {
				return auditErr
			}
		}
		return authorizationErr
	}
	lifecycle, authorizationErr := s.readRunLifecycle(run)
	if authorizationErr != nil {
		return authorizationErr
	}
	if !actor.administrator && actor.principalDigest != lifecycle.OwnerPrincipalDigest {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, action, "denied", "not_owner", runID, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return auditErr
		}
		return fmt.Errorf("%w: run belongs to another evaluation principal", ErrForbidden)
	}
	if err := s.requireRunPublicationDurable(runID); err != nil {
		return err
	}
	if action == "start" || action == "cancel" || action == "delete" {
		runDir, pathErr := s.checkedRunDirPhysical(runID)
		if pathErr != nil {
			return pathErr
		}
		if _, paired, pairErr := s.controlledPairForRun(runID, runDir); pairErr != nil {
			return pairErr
		} else if paired {
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceRun, action, "denied", "conflict", runID, lifecycle.OwnerPrincipalDigest,
			); auditErr != nil {
				return auditErr
			}
			return fmt.Errorf(
				"%w: controlled pair members require the aggregate lifecycle", ErrConflict,
			)
		}
	}
	if action == "delete" {
		if lifecycle.EvidenceHold {
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceRun, action, "denied", "evidence_hold", runID, lifecycle.OwnerPrincipalDigest,
			); auditErr != nil {
				return auditErr
			}
			return fmt.Errorf("%w: held evidence cannot be deleted", ErrConflict)
		}
		if lifecycle.RetentionClass == RetentionProtected {
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceRun, action, "denied", "protected_retention", runID, lifecycle.OwnerPrincipalDigest,
			); auditErr != nil {
				return auditErr
			}
			return fmt.Errorf("%w: protected evidence cannot be deleted", ErrConflict)
		}
		return nil
	}
	_, authorizationErr = s.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceRun, action, "allowed", lifecycleOwnerAuthorizationReason(actor, lifecycle.OwnerPrincipalDigest), runID, lifecycle.OwnerPrincipalDigest,
	)
	return authorizationErr
}

func (s *Store) authorizeAdministratorActionUnlocked(actor Actor, action, reason string) error {
	if err := validateActor(actor); err != nil {
		return err
	}
	if !actor.administrator {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceStore, action, "denied", "not_administrator", "", "",
		); auditErr != nil {
			return auditErr
		}
		return fmt.Errorf("%w: administrator authority is required", ErrForbidden)
	}
	_, err := s.appendLifecycleAuditUnlocked(actor, lifecycleResourceStore, action, "allowed", reason, "", "")
	return err
}
