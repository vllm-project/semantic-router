package evaluationplane

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const deletingRunPrefix = ".deleting-evaluation-run-"

func runDeletionProjectionIdentity(id string) string { return "run-delete:" + id }

func runDeletionPath(root, id string) string {
	return filepath.Join(root, deletingRunPrefix+id)
}

func runDeletionID(name string) (string, bool) {
	if !strings.HasPrefix(name, deletingRunPrefix) {
		return "", false
	}
	id := strings.TrimPrefix(name, deletingRunPrefix)
	return id, validClientRequestID(id)
}

type runDeletionIntent struct {
	id       string
	path     string
	complete bool
}

// recoverRunDeletionsUnlocked commits every visible hide before any ledger,
// reference, quota, or GC scan can omit that Run. Cleanup happens only beyond
// the runsRoot commit cut, so a partial bundle is already authorized deletion
// garbage and can be reclaimed without reconstructing mutable owner state.
func (s *Store) recoverRunDeletionsUnlocked() error {
	intents, err := s.listRunDeletionIntentsUnlocked()
	if err != nil {
		return err
	}
	if len(intents) == 0 {
		return nil
	}
	for _, intent := range intents {
		if intent.complete {
			if _, _, err := readRunDeletionBundle(intent.path, intent.id); err != nil {
				return err
			}
		}
	}
	if err := s.runPersistence.SyncDirectory(s.runsRoot, "evaluation run deletion intent"); err != nil {
		return fmt.Errorf("%w: evaluation run deletion durability is uncertain: %w", ErrConflict, err)
	}
	ids := make([]string, 0, len(intents))
	for _, intent := range intents {
		ids = append(ids, intent.id)
	}
	s.runIndex.removeBatch(ids...)
	for _, id := range ids {
		s.runIndex.clearPendingChange(runDeletionProjectionIdentity(id))
		s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: id})
	}

	var attestationErr error
	for _, intent := range intents {
		if _, err := s.removeExecutionAttestationIfPresent(intent.id); err != nil {
			attestationErr = errors.Join(attestationErr, err)
		}
	}
	// Sync even when every file is already absent: a prior process can remove
	// an attestation visibly and fail its directory fsync before crashing.
	attestationSyncErr := s.runPersistence.SyncDirectory(
		s.attestationRoot, "evaluation run deletion attestation recovery",
	)
	if err := errors.Join(attestationErr, attestationSyncErr); err != nil {
		return fmt.Errorf("%w: evaluation run deletion attestation cleanup is uncertain: %w", ErrConflict, err)
	}
	for _, intent := range intents {
		if err := s.runPersistence.RemoveAll(intent.path); err != nil {
			return fmt.Errorf("%w: reclaim evaluation run deletion intent: %w", ErrConflict, err)
		}
	}
	if err := s.runPersistence.SyncDirectory(s.runsRoot, "evaluation run deletion recovery"); err != nil {
		// The deletion commit cut already succeeded. A cleanup entry may reappear
		// after power loss, but remains hidden and is safe to recover again.
		log.Printf("evaluationplane: committed run deletion cleanup sync deferred: %v", err)
	}
	return nil
}

// Generic reads must not roll a Service-owned deletion forward: only the
// explicit DeleteRun retry can also close shared event subscribers. Recovery is
// authorized only for the first Service opening an otherwise-unowned root.
func (s *Store) requireNoRunDeletionIntentsUnlocked() error {
	intents, err := s.listRunDeletionIntentsUnlocked()
	if err != nil {
		return err
	}
	if len(intents) != 0 {
		return fmt.Errorf("%w: evaluation run deletion recovery is required", ErrConflict)
	}
	return nil
}

func (s *Store) listRunDeletionIntentsUnlocked() ([]runDeletionIntent, error) {
	if err := requirePrivateDirectory(s.runsRoot); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(s.runsRoot)
	if err != nil {
		return nil, fmt.Errorf("list evaluation run deletions: %w", err)
	}
	intents := make([]runDeletionIntent, 0)
	for _, entry := range entries {
		id, recognized := runDeletionID(entry.Name())
		if !recognized {
			if strings.HasPrefix(entry.Name(), deletingRunPrefix) {
				return nil, fmt.Errorf("%w: evaluation run deletion intent is invalid", ErrInvalid)
			}
			continue
		}
		path := filepath.Join(s.runsRoot, entry.Name())
		if entry.Type()&os.ModeSymlink != 0 || !entry.IsDir() || requirePrivateDirectory(path) != nil {
			return nil, fmt.Errorf("%w: evaluation run deletion intent is invalid", ErrInvalid)
		}
		if _, liveErr := os.Lstat(filepath.Join(s.runsRoot, id)); liveErr == nil {
			return nil, fmt.Errorf("%w: evaluation run has both live and deleting identities", ErrInvalid)
		} else if !os.IsNotExist(liveErr) {
			return nil, fmt.Errorf("inspect live evaluation run during deletion recovery: %w", liveErr)
		}
		complete, inspectErr := inspectRunDeletionIdentityFiles(path)
		if inspectErr != nil {
			return nil, inspectErr
		}
		intents = append(intents, runDeletionIntent{id: id, path: path, complete: complete})
	}
	return intents, nil
}

func inspectRunDeletionIdentityFiles(path string) (bool, error) {
	present := 0
	for _, name := range []string{runFileName, lifecycleFileName} {
		info, err := os.Lstat(filepath.Join(path, name))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return false, err
		}
		if !info.Mode().IsRegular() || info.Mode().Perm() != 0o600 || info.Mode()&os.ModeSymlink != 0 {
			return false, fmt.Errorf("%w: evaluation run deletion bundle is invalid", ErrInvalid)
		}
		present++
	}
	return present == 2, nil
}

// resumeRunDeletionAsUnlocked preserves the owner boundary while a complete
// intent remains. A partial intent can only be produced after the namespace
// commit and is therefore completed by the common recovery barrier.
func (s *Store) resumeRunDeletionAsUnlocked(actor Actor, id string) (bool, error) {
	if err := validateActor(actor); err != nil {
		return false, err
	}
	if !validClientRequestID(id) {
		return false, fmt.Errorf("%w: run id must be a canonical UUID", ErrInvalid)
	}
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	intentPath := runDeletionPath(s.runsRoot, id)
	if _, err := os.Lstat(intentPath); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return true, fmt.Errorf("inspect evaluation run deletion intent: %w", err)
	}
	s.runIndex.markPendingChange(runDeletionProjectionIdentity(id))
	run, lifecycle, err := readRunDeletionBundle(intentPath, id)
	if err != nil {
		complete, inspectErr := inspectRunDeletionIdentityFiles(intentPath)
		if inspectErr != nil {
			return true, inspectErr
		}
		if complete {
			return true, err
		}
		return true, s.finishCommittedPartialRunDeletionUnlocked(id, intentPath)
	}
	if !actor.administrator && lifecycle.OwnerPrincipalDigest != actor.principalDigest {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceRun, "delete", "denied", "not_owner",
			id, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return true, auditErr
		}
		return true, fmt.Errorf("%w: run belongs to another evaluation principal", ErrForbidden)
	}
	if lifecycle.EvidenceHold || lifecycle.RetentionClass == RetentionProtected {
		return true, fmt.Errorf("%w: run deletion intent has invalid retention state", ErrConflict)
	}

	if err := s.runPersistence.SyncDirectory(s.runsRoot, "evaluation run deletion retry"); err != nil {
		return true, fmt.Errorf("%w: evaluation run deletion durability is uncertain: %w", ErrConflict, err)
	}
	s.runIndex.remove(id)
	s.runIndex.clearPendingChange(runDeletionProjectionIdentity(id))
	s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: id})
	return true, s.finishRunDeletionUnlocked(actor, run, intentPath)
}

func (s *Store) finishCommittedPartialRunDeletionUnlocked(id, intentPath string) error {
	s.runIndex.remove(id)
	s.runIndex.clearPendingChange(runDeletionProjectionIdentity(id))
	s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: id})
	if _, err := s.removeExecutionAttestationIfPresent(id); err != nil {
		return err
	}
	if err := s.runPersistence.SyncDirectory(
		s.attestationRoot, "evaluation run committed deletion attestation retry",
	); err != nil {
		return fmt.Errorf("%w: evaluation run deletion attestation cleanup is uncertain: %w", ErrConflict, err)
	}
	if err := s.runPersistence.RemoveAll(intentPath); err != nil {
		return fmt.Errorf("%w: reclaim committed evaluation run deletion: %w", ErrConflict, err)
	}
	if err := s.runPersistence.SyncDirectory(
		s.runsRoot, "evaluation run committed deletion cleanup",
	); err != nil {
		return fmt.Errorf("%w: evaluation run deletion cleanup is uncertain: %w", ErrConflict, err)
	}
	return nil
}

func readRunDeletionBundle(directory, id string) (Run, RunLifecycle, error) {
	if err := requirePrivateDirectory(directory); err != nil {
		return Run{}, RunLifecycle{}, err
	}
	var run Run
	if err := readJSON(filepath.Join(directory, runFileName), &run); err != nil {
		return Run{}, RunLifecycle{}, err
	}
	if err := validateStoredRun(id, run); err != nil {
		return Run{}, RunLifecycle{}, err
	}
	var lifecycle RunLifecycle
	if err := readJSON(filepath.Join(directory, lifecycleFileName), &lifecycle); err != nil {
		return Run{}, RunLifecycle{}, err
	}
	if err := validateRunLifecycle(run, lifecycle); err != nil {
		return Run{}, RunLifecycle{}, err
	}
	return run, lifecycle, nil
}

func (s *Store) publishRunDeletionUnlocked(actor Actor, run Run, runDir string) error {
	intentPath := runDeletionPath(s.runsRoot, run.ID)
	if _, err := os.Lstat(intentPath); err == nil {
		return fmt.Errorf("%w: run deletion is already in progress", ErrConflict)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("inspect evaluation run deletion destination: %w", err)
	}
	projectionIdentity := runDeletionProjectionIdentity(run.ID)
	s.runIndex.markPendingChange(projectionIdentity)
	if err := s.runPersistence.Rename(runDir, intentPath); err != nil {
		s.runIndex.clearPendingChange(projectionIdentity)
		return fmt.Errorf("begin evaluation run deletion: %w", err)
	}
	if err := s.runPersistence.SyncDirectory(s.runsRoot, "evaluation run deletion"); err != nil {
		return fmt.Errorf("%w: evaluation run deletion durability is uncertain: %w", ErrConflict, err)
	}
	s.runIndex.remove(run.ID)
	s.runIndex.clearPendingChange(projectionIdentity)
	s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID})
	return s.finishRunDeletionUnlocked(actor, run, intentPath)
}

func (s *Store) finishRunDeletionUnlocked(actor Actor, run Run, intentPath string) error {
	_, err := s.removeExecutionAttestationIfPresent(run.ID)
	if err != nil {
		return err
	}
	// Always sync: an earlier attempt may have removed the file before the
	// attestation directory sync failed.
	if err := s.runPersistence.SyncDirectory(s.attestationRoot, "evaluation run deletion attestation"); err != nil {
		return fmt.Errorf("%w: evaluation run deletion attestation durability is uncertain: %w", ErrConflict, err)
	}
	gcAuthorized := true
	if _, err := s.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceStore, "gc", "allowed", "delete_cascade", "", "",
	); err != nil {
		// The run deletion itself already crossed its durable namespace cut and
		// has its own allowed audit record. CAS collection is opportunistic: an
		// unavailable GC audit must retain objects, but must not strand the
		// committed run tombstone and block every ledger/reference operation.
		gcAuthorized = false
		log.Printf("evaluationplane: CAS collection audit deferred after run deletion: %v", err)
	}
	if err := s.runPersistence.RemoveAll(intentPath); err != nil {
		return fmt.Errorf("%w: reclaim evaluation run deletion intent: %w", ErrConflict, err)
	}
	if err := s.runPersistence.SyncDirectory(s.runsRoot, "evaluation run deletion cleanup"); err != nil {
		log.Printf("evaluationplane: committed run deletion cleanup sync deferred: %v", err)
	}
	if gcAuthorized {
		if err := s.sweepUnreferencedCASUnlocked(); err != nil {
			log.Printf("evaluationplane: CAS collection deferred after run deletion: %v", err)
		}
	}
	return nil
}
