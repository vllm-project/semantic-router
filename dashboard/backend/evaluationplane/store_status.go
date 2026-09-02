package evaluationplane

import (
	"fmt"
	"path/filepath"
	"reflect"
)

// runStatusPersistence is the narrow atomic-publication seam for the one
// authoritative lifecycle fact. Tests can exercise transient storage failures
// without weakening the production filesystem contract.
type runStatusPersistence interface {
	Write(path string, run Run) error
	SyncDirectory(path, description string) error
}

type atomicRunStatusPersistence struct{}

func (atomicRunStatusPersistence) Write(path string, run Run) error {
	return writeJSONAtomic(path, run)
}

func (atomicRunStatusPersistence) SyncDirectory(path, description string) error {
	return syncEvaluationDirectory(path, description)
}

func (s *Store) writeRunStatusDurably(runDir string, run Run) error {
	if err := s.requireLifecycleResourceDurable(
		lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID},
	); err != nil {
		return err
	}
	// Write owns the healthy-path atomic rename and containing-directory sync.
	// SyncDirectory is reserved for idempotent retries and startup recovery, so
	// every status update pays for exactly one commit barrier.
	return s.statusPersistence.Write(filepath.Join(runDir, runFileName), run)
}

func (s *Store) syncRunStatusDirectory(runDir, description string) error {
	runID := filepath.Base(filepath.Clean(runDir))
	if err := s.requireLifecycleResourceDurable(
		lifecycleResourceRef{Kind: lifecycleResourceRun, ID: runID},
	); err != nil {
		return err
	}
	if err := s.statusPersistence.SyncDirectory(runDir, description); err != nil {
		return fmt.Errorf("evaluation run status durability is uncertain: %w", err)
	}
	return nil
}

// commitOrdinaryRunStart publishes the only mutable transition accepted from
// an ordinary launch request. The caller owns lifecycle.mu; this method owns
// the shared root/index cut and rejects a stale full-Run replacement.
func (s *Store) commitOrdinaryRunStart(starting Run) error {
	if err := validateStoredRun(starting.ID, starting); err != nil {
		return fmt.Errorf("%w: ordinary run start status is invalid: %w", ErrInvalid, err)
	}
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	runDir, err := s.checkedRunDir(starting.ID)
	if err != nil {
		return err
	}
	current, err := s.getRunUnlocked(starting.ID)
	if err != nil {
		return err
	}
	if current.ControlledPair != nil || current.Status != StatusPending ||
		starting.Status != StatusRunning || starting.StartedAt == nil {
		return fmt.Errorf("%w: ordinary run cannot start from %s", ErrConflict, current.Status)
	}
	expected := current
	expected.Status = StatusRunning
	startedAt := *starting.StartedAt
	expected.StartedAt = &startedAt
	expected.Error = ""
	expected.Progress.Message = starting.Progress.Message
	if !reflect.DeepEqual(expected, starting) {
		return fmt.Errorf("%w: ordinary run start changed immutable state", ErrConflict)
	}
	return s.persistRunStatusProjectionLocked(runDir, starting)
}

// commitWorkerProgress applies only the untrusted worker's validated progress
// payload to the current durable Run. Re-reading after the shared coordinator
// lock is the cross-Service cancellation barrier: a stale worker snapshot can
// never replace a peer's terminal state.
func (s *Store) commitWorkerProgress(runID string, progress RunProgress) error {
	if err := validateResourceID(runID); err != nil {
		return err
	}
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	runDir, err := s.checkedRunDir(runID)
	if err != nil {
		return err
	}
	current, err := s.getRunUnlocked(runID)
	if err != nil {
		return err
	}
	if current.Status != StatusRunning {
		return fmt.Errorf("%w: worker progress cannot update run from %s", ErrConflict, current.Status)
	}
	updated := current
	updated.Progress = progress
	if err := validateStoredRun(updated.ID, updated); err != nil {
		return fmt.Errorf("%w: worker progress status is invalid: %w", ErrInvalid, err)
	}
	return s.persistRunStatusProjectionLocked(runDir, updated)
}

func (s *Store) persistRunStatusProjectionLocked(runDir string, run Run) error {
	if err := s.writeRunStatusDurably(runDir, run); err != nil {
		// Atomic publication may have completed before a directory sync error.
		// Re-read the canonical fact so the in-memory projection never guesses.
		if durable, readErr := s.getRunUnlocked(run.ID); readErr == nil {
			s.runIndex.upsert(durable)
		}
		return err
	}
	s.runIndex.upsert(run)
	return nil
}

// commitRunSealing is the atomic cut between cancellable execution and
// server-owned evidence publication. No canonical worker evidence may be
// published before this transition commits.
func (s *Store) commitRunSealing(id string) (Run, error) {
	paired, err := s.acquireControlledPairMutationBarrier(id)
	if err != nil {
		return Run{}, err
	}
	defer s.releaseControlledPairMutationBarrier(paired)
	return s.commitRunSealingWithinLifecycle(id)
}

func (s *Store) commitRunSealingWithinLifecycle(id string) (Run, error) {
	if err := validateResourceID(id); err != nil {
		return Run{}, err
	}
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return Run{}, err
	}
	current, err := s.getRunUnlocked(id)
	if err != nil {
		return Run{}, err
	}
	if current.Status != StatusRunning {
		return current, fmt.Errorf("%w: run cannot transition from %s to %s", ErrConflict, current.Status, StatusSealing)
	}
	sealing := current
	sealing.Status = StatusSealing
	sealing.Progress.Message = "Sealing evaluation evidence"
	if err := validateStoredRun(sealing.ID, sealing); err != nil {
		return current, fmt.Errorf("%w: sealing run status is invalid: %w", ErrInvalid, err)
	}
	if err := s.writeRunStatusDurably(runDir, sealing); err != nil {
		if durable, readErr := s.getRunUnlocked(id); readErr == nil {
			s.runIndex.upsert(durable)
			return durable, err
		}
		return current, err
	}
	s.runIndex.upsert(sealing)
	return sealing, nil
}

// commitSealedEvidenceLevelsWithinLifecycle persists the server-derived run
// headline and per-track evidence strengths while the caller holds the paired
// lifecycle mutation barrier.
func (s *Store) commitSealedEvidenceLevelsWithinLifecycle(id string, levels sealedEvidenceLevels) (Run, error) {
	if err := validateResourceID(id); err != nil {
		return Run{}, err
	}
	if evidenceLevelRank(levels.Run) < 0 {
		return Run{}, fmt.Errorf("%w: sealed evidence level is invalid", ErrInvalid)
	}
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	runDir, err := s.checkedRunDir(id)
	if err != nil {
		return Run{}, err
	}
	current, err := s.getRunUnlocked(id)
	if err != nil {
		return Run{}, err
	}
	if current.Status != StatusSealing {
		return current, fmt.Errorf("%w: run cannot seal evidence from %s", ErrConflict, current.Status)
	}
	sealed := current
	sealed.EvidenceLevel = levels.Run
	sealed.TrackEvidenceLevels = copyTrackEvidenceLevels(levels.ByTrack)
	if err := validateStoredRun(sealed.ID, sealed); err != nil {
		return current, fmt.Errorf("%w: sealed evidence status is invalid: %w", ErrInvalid, err)
	}
	if err := s.writeRunStatusDurably(runDir, sealed); err != nil {
		if durable, readErr := s.getRunUnlocked(id); readErr == nil {
			s.runIndex.upsert(durable)
			return durable, err
		}
		return current, err
	}
	s.runIndex.upsert(sealed)
	return sealed, nil
}

// commitTerminalRun atomically orders the final status publication after every
// control event across all Store instances sharing this root. The returned SSE
// event is derived from that committed status and the immutable log tail.
func (s *Store) commitTerminalRun(run Run) (Event, error) {
	paired, err := s.acquireControlledPairMutationBarrier(run.ID)
	if err != nil {
		return Event{}, err
	}
	defer s.releaseControlledPairMutationBarrier(paired)
	return s.commitTerminalRunWithinLifecycle(run)
}

func (s *Store) commitTerminalRunWithinLifecycle(run Run) (Event, error) {
	if err := validateStoredRun(run.ID, run); err != nil {
		return Event{}, fmt.Errorf("%w: terminal run status is invalid: %w", ErrInvalid, err)
	}
	if !terminalStatus(run.Status) {
		return Event{}, fmt.Errorf("%w: terminal run status is required", ErrInvalid)
	}
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	runDir, pathErr := s.checkedRunDir(run.ID)
	if pathErr != nil {
		return Event{}, pathErr
	}
	current, readErr := s.getRunUnlocked(run.ID)
	if readErr != nil {
		return Event{}, readErr
	}
	sequence, sequenceErr := lastEventSequence(filepath.Join(runDir, eventsFileName), run.ID)
	if sequenceErr != nil {
		return Event{}, sequenceErr
	}
	if terminalStatus(current.Status) {
		if err := s.syncRunStatusDirectory(runDir, "evaluation terminal run retry"); err != nil {
			return Event{}, err
		}
		return terminalEventForRun(current, sequence+1)
	}
	switch current.Status {
	case StatusRunning:
		if run.Status != StatusFailed && run.Status != StatusCancelled {
			return Event{}, fmt.Errorf("%w: run cannot transition from %s to %s", ErrConflict, current.Status, run.Status)
		}
	case StatusSealing:
		if run.Status != StatusCompleted && run.Status != StatusFailed {
			return Event{}, fmt.Errorf("%w: run cannot transition from %s to %s", ErrConflict, current.Status, run.Status)
		}
	default:
		return Event{}, fmt.Errorf("%w: run cannot transition from %s to %s", ErrConflict, current.Status, run.Status)
	}
	terminalEvent, eventErr := terminalEventForRun(run, sequence+1)
	if eventErr != nil {
		return Event{}, eventErr
	}
	if err := s.writeRunStatusDurably(runDir, run); err != nil {
		if durable, readErr := s.getRunUnlocked(run.ID); readErr == nil {
			s.runIndex.upsert(durable)
		}
		return Event{}, err
	}
	s.runIndex.upsert(run)
	return terminalEvent, nil
}
