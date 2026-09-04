package evaluationplane

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// cancelControlledPairAs commits cancellation as one aggregate transition.
// While the durable intent is in progress, readers continue to see both
// aggregate-owned running snapshots; physical member writes are never exposed.
func (s *Store) cancelControlledPairAs(actor Actor, pairID string) (controlledPairManifest, error) {
	if err := validateActor(actor); err != nil {
		return controlledPairManifest{}, err
	}
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	pair, err := s.readControlledPair(pairID)
	if err != nil {
		return controlledPairManifest{}, err
	}
	if err := s.authorizeControlledPairLifecycleUnlocked(actor, pair, "cancel"); err != nil {
		return controlledPairManifest{}, err
	}
	if pair.State == controlledPairStateTerminal {
		if err := s.syncControlledPairCommitCut(pair, "controlled pair cancellation retry"); err != nil {
			return controlledPairManifest{}, err
		}
		return pair, nil
	}
	if pair.State == controlledPairStateCancelling {
		if err := s.recoverControlledPairCancellation(pair); err != nil {
			return controlledPairManifest{}, err
		}
		return s.readControlledPair(pairID)
	}
	if pair.State != controlledPairStateRunning {
		return controlledPairManifest{}, fmt.Errorf(
			"%w: controlled pair cannot be cancelled from %s", ErrConflict, pair.State,
		)
	}
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		run, err := s.getRunPhysical(runID)
		if err != nil {
			return controlledPairManifest{}, err
		}
		if run.Status == StatusSealing {
			return controlledPairManifest{}, fmt.Errorf(
				"%w: controlled pair cancellation waits for sealing evidence", ErrConflict,
			)
		}
		if run.Status != StatusRunning && !terminalStatus(run.Status) {
			return controlledPairManifest{}, fmt.Errorf(
				"%w: controlled pair member cannot be cancelled from %s", ErrConflict, run.Status,
			)
		}
	}
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		if _, err := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "cancel", "allowed", lifecycleAuthorizationReasonForPair(actor, pair),
			runID, pair.OwnerPrincipalDigest,
		); err != nil {
			return controlledPairManifest{}, err
		}
	}
	pair.State = controlledPairStateCancelling
	if err := s.writeControlledPairDurably(pair); err != nil {
		return controlledPairManifest{}, err
	}
	if err := s.finishControlledPairCancellation(pair); err != nil {
		return controlledPairManifest{}, err
	}
	return s.readControlledPair(pairID)
}

func (s *Store) finishControlledPairCancellation(pair controlledPairManifest) error {
	completedAt := time.Now().UTC().Truncate(time.Microsecond)
	terminal := make([]Run, 0, 2)
	for _, snapshot := range []Run{pair.BaselineRun, pair.CandidateRun} {
		run, err := s.getRunPhysical(snapshot.ID)
		if err != nil {
			return err
		}
		switch {
		case terminalStatus(run.Status):
			// A completed arm is immutable evidence. Cancellation only stops the
			// counterpart and must never replace this snapshot with start-time data.
			if err := s.syncRunStatusDirectory(
				filepath.Join(s.runsRoot, run.ID), "controlled pair terminal member retry",
			); err != nil {
				return err
			}
		case run.Status == StatusSealing:
			return fmt.Errorf("%w: controlled pair cancellation waits for sealing evidence", ErrConflict)
		case run.Status == StatusRunning:
			run.Status = StatusCancelled
			run.CompletedAt = &completedAt
			run.Error = ""
			run.Progress.Message = "Controlled pair cancelled"
			if err := s.writeRunStatusDurably(filepath.Join(s.runsRoot, run.ID), run); err != nil {
				return err
			}
		default:
			return fmt.Errorf("%w: controlled pair member cannot be cancelled from %s", ErrConflict, run.Status)
		}
		terminal = append(terminal, run)
	}
	pair.State = controlledPairStateTerminal
	pair.BaselineRun, pair.CandidateRun = terminal[0], terminal[1]
	if err := s.validatePublishedControlledPair(pair); err != nil {
		return err
	}
	if err := s.writeControlledPairDurably(pair); err != nil {
		return err
	}
	s.runIndex.upsertBatch(terminal, nil)
	return nil
}

func (s *Store) recoverControlledPairCancellation(pair controlledPairManifest) error {
	if err := s.validatePublishedControlledPair(pair); err != nil {
		return err
	}
	return s.finishControlledPairCancellation(pair)
}

// deleteControlledPairAs durably hides both members before reclaiming either
// directory. The deleted aggregate remains as a small identity tombstone so a
// client request ID can never be rebound to different evidence.
func (s *Store) deleteControlledPairAs(actor Actor, pairID string) error {
	if err := validateActor(actor); err != nil {
		return err
	}
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	pair, err := s.readControlledPair(pairID)
	if err != nil {
		return err
	}
	if err := s.authorizeControlledPairLifecycleUnlocked(actor, pair, "delete"); err != nil {
		return err
	}
	if s.lifecycle.contains(pair.BaselineRunID) || s.lifecycle.contains(pair.CandidateRunID) {
		return fmt.Errorf("%w: controlled pair workers are still exiting", ErrConflict)
	}
	if pair.State == controlledPairStateDeleted {
		return s.syncControlledPairDirectory(pair.PairID, "controlled pair deletion retry")
	}
	if pair.State == controlledPairStateDeleting {
		projectionIdentity := controlledPairDeletionProjectionIdentity(pair.PairID)
		s.runIndex.markPendingChange(projectionIdentity)
		if err := s.syncControlledPairDirectory(pair.PairID, "controlled pair deletion retry"); err != nil {
			return err
		}
		s.runIndex.removeBatch(pair.BaselineRunID, pair.CandidateRunID)
		s.runIndex.clearPendingChange(projectionIdentity)
		return s.finishControlledPairDeletion(pair)
	}
	if pair.State != controlledPairStatePending && pair.State != controlledPairStateTerminal {
		return fmt.Errorf("%w: controlled pair cannot be deleted from %s", ErrConflict, pair.State)
	}
	if err := s.ensureControlledPairNotExternallyReferencedUnlocked(pair); err != nil {
		if auditErr := s.appendLifecycleDenialsUnlocked(
			actor, "delete", "referenced", pair.OwnerPrincipalDigest,
			pair.BaselineRunID, pair.CandidateRunID,
		); auditErr != nil {
			return auditErr
		}
		return err
	}
	if err := s.validateControlledPairMembersForDeletionUnlocked(actor, pair); err != nil {
		return err
	}
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		if _, err := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "delete", "allowed", lifecycleAuthorizationReasonForPair(actor, pair), runID, pair.OwnerPrincipalDigest,
		); err != nil {
			return err
		}
	}
	if _, err := s.appendLifecycleAuditUnlocked(
		SystemActor(), lifecycleResourceStore, "gc", "allowed", "delete_cascade", "", "",
	); err != nil {
		return err
	}
	deletedAt := controlledPairDeletionTime(time.Now())
	pair.State, pair.DeletedAt = controlledPairStateDeleting, &deletedAt
	pair.DeleteReceiptDigest = controlledPairDeleteReceipt(pair)
	projectionIdentity := controlledPairDeletionProjectionIdentity(pair.PairID)
	s.runIndex.markPendingChange(projectionIdentity)
	if err := s.writeControlledPairDurably(pair); err != nil {
		if visible, readErr := s.readControlledPair(pair.PairID); readErr == nil &&
			visible.State != controlledPairStateDeleting {
			s.runIndex.clearPendingChange(projectionIdentity)
		}
		return err
	}
	s.runIndex.removeBatch(pair.BaselineRunID, pair.CandidateRunID)
	s.runIndex.clearPendingChange(projectionIdentity)
	return s.finishControlledPairDeletion(pair)
}

func (s *Store) validateControlledPairMembersForDeletionUnlocked(
	actor Actor,
	pair controlledPairManifest,
) error {
	for _, run := range []Run{pair.BaselineRun, pair.CandidateRun} {
		lifecycle, err := s.readRunLifecycle(run)
		if err != nil {
			return err
		}
		if lifecycle.EvidenceHold || lifecycle.RetentionClass == RetentionProtected {
			reason := "evidence_hold"
			if lifecycle.RetentionClass == RetentionProtected {
				reason = "protected_retention"
			}
			if err := s.appendLifecycleDenialsUnlocked(
				actor, "delete", reason, pair.OwnerPrincipalDigest, run.ID,
			); err != nil {
				return err
			}
			return fmt.Errorf("%w: controlled pair member retention prevents aggregate deletion", ErrConflict)
		}
		if err := s.ensureRunNotCampaignReferencedUnlocked(run.ID); err != nil {
			if auditErr := s.appendLifecycleDenialsUnlocked(
				actor, "delete", "referenced", pair.OwnerPrincipalDigest, run.ID,
			); auditErr != nil {
				return auditErr
			}
			return err
		}
	}
	return nil
}

func controlledPairDeletionProjectionIdentity(pairID string) string { return "pair-delete:" + pairID }

func (s *Store) ensureControlledPairNotExternallyReferencedUnlocked(pair controlledPairManifest) error {
	targets := map[string]bool{pair.BaselineRunID: true, pair.CandidateRunID: true}
	runs, err := s.loadCompleteRunReferenceLedgerUnlocked()
	if err != nil {
		return fmt.Errorf("%w: controlled pair reference ledger cannot be verified: %w", ErrConflict, err)
	}
	for _, run := range runs {
		if run.ID == pair.CandidateRunID && run.BaselineRunID == pair.BaselineRunID {
			continue // the aggregate's sole internal comparison edge
		}
		if targets[run.BaselineRunID] {
			return fmt.Errorf("%w: controlled pair member is referenced by run %s", ErrConflict, run.ID)
		}
	}
	entries, err := os.ReadDir(s.controlledPairRoot)
	if err != nil {
		return fmt.Errorf("%w: controlled pair reference graph cannot be verified", ErrConflict)
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return fmt.Errorf("%w: controlled pair reference graph is invalid", ErrConflict)
		}
		other, err := s.readControlledPair(entry.Name())
		if err != nil {
			return fmt.Errorf("%w: controlled pair reference graph cannot be verified", ErrConflict)
		}
		if other.PairID == pair.PairID || other.State == controlledPairStateDeleting || other.State == controlledPairStateDeleted {
			continue
		}
		for _, referencedID := range []string{
			other.BaselineSourceRunID, other.CandidateSourceRunID,
			other.BaselineRunID, other.CandidateRunID,
		} {
			if targets[referencedID] {
				return fmt.Errorf("%w: controlled pair member is referenced by aggregate %s", ErrConflict, other.PairID)
			}
		}
	}
	return nil
}

func (s *Store) controlledPairPreflightCapabilitiesUnlocked(
	pair controlledPairManifest,
	baseline Run,
	candidate Run,
	activeWorkers bool,
) ControlledPairCapabilities {
	capabilities := ControlledPairCapabilities{}
	if pair.State == controlledPairStateRunning {
		capabilities.CanCancel = baseline.Status != StatusSealing && candidate.Status != StatusSealing &&
			(baseline.Status == StatusRunning || terminalStatus(baseline.Status)) &&
			(candidate.Status == StatusRunning || terminalStatus(candidate.Status))
	}
	if pair.State != controlledPairStatePending && pair.State != controlledPairStateTerminal {
		return capabilities
	}
	if activeWorkers {
		return capabilities
	}
	for _, run := range []Run{baseline, candidate} {
		lifecycle, err := s.readRunLifecycle(run)
		if err != nil || lifecycle.EvidenceHold || lifecycle.RetentionClass == RetentionProtected {
			return capabilities
		}
		if err := s.ensureRunNotCampaignReferencedUnlocked(run.ID); err != nil {
			return capabilities
		}
	}
	if err := s.ensureControlledPairNotExternallyReferencedUnlocked(pair); err != nil {
		return capabilities
	}
	capabilities.CanDelete = true
	return capabilities
}

func (s *Store) finishControlledPairDeletion(pair controlledPairManifest) error {
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		runDir := filepath.Join(s.runsRoot, runID)
		if _, statErr := os.Lstat(runDir); statErr == nil {
			// Deleting is already a durable aggregate commit and the tombstone
			// permanently reserves both UUIDs. A partially removed member is
			// therefore transaction garbage, not a new identity to revalidate.
			if err := requirePrivateDirectory(runDir); err != nil {
				return err
			}
			if removeErr := s.pairPersistence.RemoveAll(runDir); removeErr != nil {
				return removeErr
			}
		} else if !os.IsNotExist(statErr) {
			return statErr
		}
		if _, err := s.removeExecutionAttestationIfPresent(runID); err != nil {
			return err
		}
	}
	if err := s.pairPersistence.SyncDirectory(s.runsRoot, "controlled pair deletion"); err != nil {
		return err
	}
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: runID})
	}
	if err := syncEvaluationDirectory(s.attestationRoot, "controlled pair attestation deletion"); err != nil {
		return err
	}
	pair.State = controlledPairStateDeleted
	if err := s.writeControlledPairTombstoneDurably(pair); err != nil {
		return err
	}
	s.runIndex.removeBatch(pair.BaselineRunID, pair.CandidateRunID)
	if err := s.sweepUnreferencedCASUnlocked(); err != nil {
		log.Printf("evaluationplane: CAS collection deferred after controlled pair deletion: %v", err)
	}
	return nil
}

func (s *Store) authorizeControlledPairLifecycleUnlocked(actor Actor, pair controlledPairManifest, action string) error {
	if err := validateActor(actor); err != nil {
		return err
	}
	if !validLifecycleAction(action) || action == "create" || action == "gc" {
		return fmt.Errorf("%w: controlled pair lifecycle action is invalid", ErrInvalid)
	}
	if pair.OwnerPrincipalDigest == actor.principalDigest || actor.administrator {
		return nil
	}
	if err := s.appendLifecycleDenialsUnlocked(
		actor, action, "not_owner", pair.OwnerPrincipalDigest,
		pair.BaselineRunID, pair.CandidateRunID,
	); err != nil {
		return err
	}
	return fmt.Errorf("%w: controlled pair belongs to another evaluation principal", ErrForbidden)
}

// authorizeControlledPairServiceAction runs before service-owned state checks
// such as active worker handles. The lifecycle and service mutexes are already
// held, so owner denial is audited without exposing process state.
func (s *Store) authorizeControlledPairServiceAction(
	actor Actor,
	pairID string,
	action string,
) (controlledPairManifest, error) {
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	pair, err := s.readControlledPair(pairID)
	if err != nil {
		return controlledPairManifest{}, err
	}
	if err := s.authorizeControlledPairLifecycleUnlocked(actor, pair, action); err != nil {
		return controlledPairManifest{}, err
	}
	return pair, nil
}
