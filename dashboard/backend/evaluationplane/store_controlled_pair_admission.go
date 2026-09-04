package evaluationplane

import (
	"errors"
	"fmt"
)

func (s *Store) reconcileControlledPairCreateUnlocked(
	actor Actor,
	existing controlledPairManifest,
	requested controlledPairManifest,
) (controlledPairManifest, error) {
	if existing.OwnerPrincipalDigest != actor.principalDigest && !actor.administrator {
		if err := s.appendLifecycleDenialsUnlocked(
			actor, "create", "not_owner", existing.OwnerPrincipalDigest,
			requested.BaselineRunID, requested.CandidateRunID,
		); err != nil {
			return controlledPairManifest{}, err
		}
		return controlledPairManifest{}, fmt.Errorf(
			"%w: client_request_id belongs to another evaluation principal", ErrForbidden,
		)
	}
	if !sameControlledPairIdentity(existing, requested) {
		if err := s.appendLifecycleDenialsUnlocked(
			actor, "create", "conflict", existing.OwnerPrincipalDigest,
			requested.BaselineRunID, requested.CandidateRunID,
		); err != nil {
			return controlledPairManifest{}, err
		}
		return controlledPairManifest{}, fmt.Errorf(
			"%w: client_request_id is bound to another controlled pair", ErrConflict,
		)
	}
	switch existing.State {
	case controlledPairStatePublishing:
		if err := s.recoverControlledPairPublication(existing); err != nil {
			return controlledPairManifest{}, err
		}
	case controlledPairStateStarting:
		if err := s.recoverControlledPairStart(existing); err != nil {
			return controlledPairManifest{}, err
		}
	}
	reconciled, err := s.readControlledPair(existing.PairID)
	if err != nil {
		return controlledPairManifest{}, err
	}
	if reconciled.State == controlledPairStateRunning || reconciled.State == controlledPairStateTerminal {
		if err := s.syncControlledPairCommitCut(reconciled, "controlled pair create retry"); err != nil {
			return controlledPairManifest{}, err
		}
	} else if err := s.syncControlledPairDirectory(reconciled.PairID, "controlled pair create retry"); err != nil {
		return controlledPairManifest{}, err
	}
	return reconciled, nil
}

func sameControlledPairIdentity(left, right controlledPairManifest) bool {
	return left.PairID == right.PairID && left.ClientRequestID == right.ClientRequestID &&
		left.Protocol == right.Protocol && left.OwnerPrincipalDigest == right.OwnerPrincipalDigest &&
		left.BaselineSourceRunID == right.BaselineSourceRunID &&
		left.CandidateSourceRunID == right.CandidateSourceRunID &&
		left.BaselineRunID == right.BaselineRunID && left.CandidateRunID == right.CandidateRunID &&
		left.BaselineRole == right.BaselineRole && left.CandidateRole == right.CandidateRole &&
		left.BaselineSourceManifestSemanticDigest == right.BaselineSourceManifestSemanticDigest &&
		left.CandidateSourceManifestSemanticDigest == right.CandidateSourceManifestSemanticDigest &&
		left.BaselineSourceManifestArtifactDigest == right.BaselineSourceManifestArtifactDigest &&
		left.CandidateSourceManifestArtifactDigest == right.CandidateSourceManifestArtifactDigest &&
		left.BaselineSourceAnchorDigest == right.BaselineSourceAnchorDigest &&
		left.CandidateSourceAnchorDigest == right.CandidateSourceAnchorDigest &&
		left.BaselineSourceAttestationDigest == right.BaselineSourceAttestationDigest &&
		left.CandidateSourceAttestationDigest == right.CandidateSourceAttestationDigest &&
		left.BaselineMemberManifestDigest == right.BaselineMemberManifestDigest &&
		left.CandidateMemberManifestDigest == right.CandidateMemberManifestDigest &&
		left.CohortDigest == right.CohortDigest && left.TreatmentDigest == right.TreatmentDigest
}

// prepareControlledPairRequestUnlocked authorizes both immutable source
// owners and resumes any durable transaction before the Service reads source
// evidence or freezes credentials. The caller owns lifecycle.mu.
func (s *Store) prepareControlledPairRequestUnlocked(
	actor Actor,
	request CreateControlledPairRequest,
) (controlledPairManifest, bool, error) {
	if err := validateActor(actor); err != nil {
		return controlledPairManifest{}, false, err
	}
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	existing, err := s.readControlledPair(request.ClientRequestID)
	if err == nil {
		if existing.OwnerPrincipalDigest != actor.principalDigest && !actor.administrator {
			return controlledPairManifest{}, false, s.auditControlledPairRequestDeniedUnlocked(
				actor, request, "not_owner", existing.OwnerPrincipalDigest, ErrForbidden,
			)
		}
		if existing.BaselineSourceRunID != request.BaselineSourceRunID ||
			existing.CandidateSourceRunID != request.CandidateSourceRunID ||
			existing.BaselineRunID != request.BaselineRunID || existing.CandidateRunID != request.CandidateRunID {
			return controlledPairManifest{}, false, s.auditControlledPairRequestDeniedUnlocked(
				actor, request, "conflict", existing.OwnerPrincipalDigest, ErrConflict,
			)
		}
		if existing.State == controlledPairStateDeleted || existing.State == controlledPairStateDeleting {
			return existing, true, nil
		}
	} else if !errors.Is(err, ErrNotFound) {
		return controlledPairManifest{}, false, err
	}

	if authorizationErr := s.authorizeControlledPairSourcesUnlocked(actor, request); authorizationErr != nil {
		return controlledPairManifest{}, false, authorizationErr
	}
	if errors.Is(err, ErrNotFound) {
		return controlledPairManifest{}, false, nil
	}
	switch existing.State {
	case controlledPairStatePublishing:
		if recoveryErr := s.recoverControlledPairPublication(existing); recoveryErr != nil {
			return controlledPairManifest{}, false, recoveryErr
		}
	case controlledPairStateStarting:
		if recoveryErr := s.recoverControlledPairStart(existing); recoveryErr != nil {
			return controlledPairManifest{}, false, recoveryErr
		}
	}
	resumed, err := s.readControlledPair(existing.PairID)
	if errors.Is(err, ErrNotFound) {
		return controlledPairManifest{}, false, nil
	}
	if err != nil {
		return controlledPairManifest{}, false, err
	}
	if resumed.State == controlledPairStateRunning || resumed.State == controlledPairStateTerminal {
		if err := s.syncControlledPairCommitCut(resumed, "controlled pair request retry"); err != nil {
			return controlledPairManifest{}, false, err
		}
	} else if err := s.syncControlledPairDirectory(resumed.PairID, "controlled pair request retry"); err != nil {
		return controlledPairManifest{}, false, err
	}
	return resumed, true, nil
}

func (s *Store) authorizeControlledPairSourcesUnlocked(actor Actor, request CreateControlledPairRequest) error {
	for _, sourceID := range []string{request.BaselineSourceRunID, request.CandidateSourceRunID} {
		sourceRun, err := s.getRunPhysical(sourceID)
		if err != nil {
			return err
		}
		lifecycle, err := s.readRunLifecycle(sourceRun)
		if err != nil {
			return err
		}
		if lifecycle.OwnerPrincipalDigest != actor.principalDigest && !actor.administrator {
			return s.auditControlledPairRequestDeniedUnlocked(
				actor, request, "not_owner", lifecycle.OwnerPrincipalDigest, ErrForbidden,
			)
		}
	}
	return nil
}

func (s *Store) auditControlledPairRequestDeniedUnlocked(
	actor Actor,
	request CreateControlledPairRequest,
	reason, owner string,
	domainErr error,
) error {
	for _, runID := range []string{request.BaselineRunID, request.CandidateRunID} {
		if _, err := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "create", "denied", reason, runID, owner,
		); err != nil {
			return err
		}
	}
	if errors.Is(domainErr, ErrForbidden) {
		return fmt.Errorf("%w: controlled pair source belongs to another principal", domainErr)
	}
	return fmt.Errorf("%w: client_request_id is bound to another controlled pair", domainErr)
}
