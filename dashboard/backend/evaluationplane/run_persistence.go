package evaluationplane

import (
	"errors"
	"fmt"
	"os"
)

var errRunPublicationDurabilityUncertain = errors.New("evaluation run publication durability is uncertain")

// runNamespacePersistence is the narrow durability seam for publishing and
// deleting one ordinary run bundle, including its attestation cleanup barrier.
// Controlled-pair lifecycle changes have their own aggregate transaction and
// persistence contract.
type runNamespacePersistence interface {
	Rename(source, destination string) error
	RemoveAll(path string) error
	SyncDirectory(path, description string) error
}

type atomicRunNamespacePersistence struct{}

func (atomicRunNamespacePersistence) Rename(source, destination string) error {
	return os.Rename(source, destination)
}

func (atomicRunNamespacePersistence) RemoveAll(path string) error {
	return os.RemoveAll(path)
}

func (atomicRunNamespacePersistence) SyncDirectory(path, description string) error {
	return syncEvaluationDirectory(path, description)
}

func runPublicationProjectionIdentity(id string) string { return "run-create:" + id }

// beginRunPublicationDurability marks a rename as undecided before it becomes
// visible. The root-scoped projection prevents peer Stores and ledger-driven
// decisions from treating a visible bundle as committed until runsRoot has
// crossed its parent-directory fsync boundary.
func (s *Store) beginRunPublicationDurability(actor Actor, run Run) {
	s.lifecycle.runNamespaceMu.Lock()
	defer s.lifecycle.runNamespaceMu.Unlock()
	s.lifecycle.pendingRunPublications[run.ID] = pendingNamespacePublication{
		actorDigest:    actor.principalDigest,
		identityDigest: lifecycleDigest(run),
	}
	s.runIndex.markPendingChange(runPublicationProjectionIdentity(run.ID))
}

func (s *Store) abandonRunPublicationDurability(id string) {
	s.lifecycle.runNamespaceMu.Lock()
	defer s.lifecycle.runNamespaceMu.Unlock()
	delete(s.lifecycle.pendingRunPublications, id)
	s.runIndex.clearPendingChange(runPublicationProjectionIdentity(id))
}

func (s *Store) requireNoPendingRunPublications() error {
	s.lifecycle.runNamespaceMu.Lock()
	defer s.lifecycle.runNamespaceMu.Unlock()
	if len(s.lifecycle.pendingRunPublications) != 0 {
		return fmt.Errorf("%w: evaluation run publication recovery requires the startup owner or explicit create retry", ErrConflict)
	}
	return nil
}

func (s *Store) requireRunPublicationDurable(id string) error {
	s.lifecycle.runNamespaceMu.Lock()
	defer s.lifecycle.runNamespaceMu.Unlock()
	if _, pending := s.lifecycle.pendingRunPublications[id]; pending {
		return fmt.Errorf("%w: evaluation run publication requires an explicit matching create retry", ErrConflict)
	}
	return nil
}

// resolveRunPublicationDurabilityUnlocked is the keyed commit barrier for an
// ordinary Run create. The caller holds runIndex.coordinator so ListRuns can
// never observe the pending projection being cleared before the parent sync.
// A parent-directory fsync can commit every entry in that directory, so the
// pending set must contain only this exact actor/object before syncing.
func (s *Store) resolveRunPublicationDurabilityUnlocked(actor Actor, run Run) error {
	s.lifecycle.runNamespaceMu.Lock()
	defer s.lifecycle.runNamespaceMu.Unlock()
	if len(s.lifecycle.pendingRunPublications) == 0 {
		return nil
	}
	pending, exists := s.lifecycle.pendingRunPublications[run.ID]
	if !exists || len(s.lifecycle.pendingRunPublications) != 1 ||
		pending.actorDigest != actor.principalDigest || pending.identityDigest != lifecycleDigest(run) {
		return fmt.Errorf("%w: evaluation run create retry does not match the pending publication", ErrConflict)
	}
	if err := s.runPersistence.SyncDirectory(s.runsRoot, "evaluation run publication"); err != nil {
		return fmt.Errorf("%w: %w: %w", ErrConflict, errRunPublicationDurabilityUncertain, err)
	}
	delete(s.lifecycle.pendingRunPublications, run.ID)
	s.runIndex.clearPendingChange(runPublicationProjectionIdentity(run.ID))
	return nil
}

// recoverRunPublicationDurabilityUnlocked is startup-only. A new process has
// no pending projection from its predecessor, so it unconditionally syncs the
// namespace before rebuilding the run ledger and clears any failed initializer
// projection retained by a same-process startup takeover.
func (s *Store) recoverRunPublicationDurabilityUnlocked() error {
	s.lifecycle.runNamespaceMu.Lock()
	defer s.lifecycle.runNamespaceMu.Unlock()
	if err := s.runPersistence.SyncDirectory(s.runsRoot, "evaluation run publication recovery"); err != nil {
		return fmt.Errorf("%w: evaluation run publication durability is uncertain: %w", ErrConflict, err)
	}
	for runID := range s.lifecycle.pendingRunPublications {
		s.runIndex.clearPendingChange(runPublicationProjectionIdentity(runID))
	}
	clear(s.lifecycle.pendingRunPublications)
	return nil
}
