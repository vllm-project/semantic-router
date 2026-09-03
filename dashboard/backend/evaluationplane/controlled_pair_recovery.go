package evaluationplane

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

func (s *Store) requireNoControlledPairRecoveryTransactionsUnlocked() error {
	entries, err := os.ReadDir(s.controlledPairRoot)
	if err != nil {
		return fmt.Errorf("list controlled pair transactions: %w", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return fmt.Errorf("%w: controlled pair transaction entry is invalid", ErrInvalid)
		}
		pair, err := s.readControlledPair(entry.Name())
		if err != nil {
			return err
		}
		switch pair.State {
		case controlledPairStatePublishing, controlledPairStateStarting,
			controlledPairStateCancelling, controlledPairStateDeleting:
			return fmt.Errorf("%w: controlled pair recovery requires the startup owner", ErrConflict)
		}
	}
	return nil
}

func (s *Store) validateStableControlledPairsUnlocked() error {
	entries, err := os.ReadDir(s.controlledPairRoot)
	if err != nil {
		return fmt.Errorf("list controlled pair transactions: %w", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return fmt.Errorf("%w: controlled pair transaction entry is invalid", ErrInvalid)
		}
		pair, err := s.readControlledPair(entry.Name())
		if err != nil {
			return err
		}
		switch pair.State {
		case controlledPairStatePending, controlledPairStateRunning, controlledPairStateTerminal:
			if err := s.validatePublishedControlledPair(pair); err != nil {
				return err
			}
		case controlledPairStateDeleted:
			// readControlledPair already validates the immutable identity tombstone.
		default:
			return fmt.Errorf("%w: controlled pair recovery requires the startup owner", ErrConflict)
		}
	}
	return nil
}

func (s *Store) recoverControlledPairTransactionsUnlocked() error {
	entries, err := os.ReadDir(s.controlledPairRoot)
	if err != nil {
		return fmt.Errorf("list controlled pair transactions: %w", err)
	}
	for _, entry := range entries {
		if !entry.IsDir() || !validClientRequestID(entry.Name()) {
			return fmt.Errorf("%w: controlled pair transaction entry is invalid", ErrInvalid)
		}
		dir := filepath.Join(s.controlledPairRoot, entry.Name())
		if err := requirePrivateDirectory(dir); err != nil {
			return err
		}
		hasManifest, hasTombstone, recoveryErr := cleanupControlledPairAtomicTemps(dir)
		if recoveryErr != nil {
			return recoveryErr
		}
		if !hasManifest {
			if hasTombstone {
				pair, readErr := s.readControlledPair(entry.Name())
				if readErr != nil || pair.State != controlledPairStateDeleted {
					return fmt.Errorf("%w: controlled pair tombstone is invalid", ErrInvalid)
				}
				if err := s.syncControlledPairDirectory(pair.PairID, "controlled pair tombstone recovery"); err != nil {
					return err
				}
				continue
			}
			if removeErr := os.Remove(dir); removeErr != nil {
				return removeErr
			}
			continue
		}
		pair, recoveryErr := s.readControlledPair(entry.Name())
		if recoveryErr != nil {
			return recoveryErr
		}
		if hasTombstone && pair.State != controlledPairStateDeleting {
			return fmt.Errorf("%w: controlled pair canonical inventory is inconsistent", ErrInvalid)
		}
		switch pair.State {
		case controlledPairStatePublishing:
			if err := s.recoverControlledPairPublication(pair); err != nil {
				return err
			}
		case controlledPairStateStarting:
			if err := s.recoverControlledPairStart(pair); err != nil {
				return err
			}
		case controlledPairStateCancelling:
			if err := s.recoverControlledPairCancellation(pair); err != nil {
				return err
			}
		case controlledPairStateDeleting:
			if err := s.finishControlledPairDeletion(pair); err != nil {
				return err
			}
		case controlledPairStateDeleted:
			// The identity tombstone is the committed end state.
			if err := s.syncControlledPairDirectory(pair.PairID, "controlled pair deletion recovery"); err != nil {
				return err
			}
		case controlledPairStatePending:
			if err := s.validatePublishedControlledPair(pair); err != nil {
				return err
			}
			if err := s.syncControlledPairDirectory(pair.PairID, "controlled pair pending recovery"); err != nil {
				return err
			}
		case controlledPairStateRunning:
			if err := s.validatePublishedControlledPair(pair); err != nil {
				return err
			}
			if err := s.syncControlledPairCommitCut(pair, "controlled pair running recovery"); err != nil {
				return err
			}
			if err := s.refreshControlledPairTerminalStateUnlocked(pair.BaselineRunID); err != nil {
				return err
			}
		case controlledPairStateTerminal:
			if err := s.validatePublishedControlledPair(pair); err != nil {
				return err
			}
			if err := s.syncControlledPairCommitCut(pair, "controlled pair terminal recovery"); err != nil {
				return err
			}
		}
	}
	return s.pairPersistence.SyncDirectory(s.controlledPairRoot, "controlled pair recovery")
}

func cleanupControlledPairAtomicTemps(dir string) (bool, bool, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false, false, fmt.Errorf("inspect controlled pair transaction: %w", err)
	}
	removed := false
	hasManifest, hasTombstone := false, false
	for _, entry := range entries {
		switch entry.Name() {
		case controlledPairManifestFile:
			hasManifest = true
		case controlledPairTombstoneFile:
			hasTombstone = true
		default:
			if entry.IsDir() || !strings.HasPrefix(entry.Name(), ".tmp-evaluation-") {
				return false, false, fmt.Errorf("%w: controlled pair transaction inventory is invalid", ErrInvalid)
			}
			info, statErr := entry.Info()
			if statErr != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Mode().Perm() != 0o600 {
				return false, false, fmt.Errorf("%w: controlled pair manifest temp is invalid", ErrInvalid)
			}
			if err := os.Remove(filepath.Join(dir, entry.Name())); err != nil {
				return false, false, err
			}
			removed = true
		}
	}
	if removed {
		if err := syncEvaluationDirectory(dir, "controlled pair manifest temp recovery"); err != nil {
			return false, false, err
		}
	}
	return hasManifest, hasTombstone, nil
}

func (s *Store) recoverControlledPairPublication(pair controlledPairManifest) error {
	items := []struct {
		runID, stageName, role string
	}{
		{pair.BaselineRunID, pair.BaselineStageName, controlledPairRoleBaseline},
		{pair.CandidateRunID, pair.CandidateStageName, controlledPairRoleCandidate},
	}
	complete := true
	for _, item := range items {
		destination := filepath.Join(s.runsRoot, item.runID)
		stage := filepath.Join(s.runsRoot, item.stageName)
		if _, statErr := os.Lstat(destination); statErr == nil {
			if _, stageErr := os.Lstat(stage); !os.IsNotExist(stageErr) {
				return fmt.Errorf("%w: controlled pair has both staged and published member", ErrInvalid)
			}
			if membershipErr := s.requireControlledPairMembership(destination, pair.PairID, item.runID, item.role); membershipErr != nil {
				return membershipErr
			}
			continue
		} else if !os.IsNotExist(statErr) {
			return statErr
		}
		if _, stageStatErr := os.Lstat(stage); stageStatErr == nil {
			if membershipErr := s.requireControlledPairMembership(stage, pair.PairID, item.runID, item.role); membershipErr != nil {
				return membershipErr
			}
			if renameErr := s.pairPersistence.Rename(stage, destination); renameErr != nil {
				return fmt.Errorf("roll forward controlled pair publication: %w", renameErr)
			}
			continue
		} else if !os.IsNotExist(stageStatErr) {
			return stageStatErr
		}
		complete = false
		break
	}
	if !complete {
		return s.rollbackControlledPairPublication(pair)
	}
	if err := s.pairPersistence.SyncDirectory(s.runsRoot, "controlled pair publication recovery"); err != nil {
		return err
	}
	pending := pair
	pending.State = controlledPairStatePending
	pending.BaselineStageName, pending.CandidateStageName = "", ""
	if err := s.validatePublishedControlledPair(pending); err != nil {
		if rollbackErr := s.rollbackControlledPairPublication(pair); rollbackErr != nil {
			return errors.Join(err, rollbackErr)
		}
		return nil
	}
	return s.writeControlledPairDurably(pending)
}

func (s *Store) rollbackControlledPairPublication(pair controlledPairManifest) error {
	for _, item := range []struct{ runID, stageName, role string }{
		{pair.BaselineRunID, pair.BaselineStageName, controlledPairRoleBaseline},
		{pair.CandidateRunID, pair.CandidateStageName, controlledPairRoleCandidate},
	} {
		for _, path := range []string{
			filepath.Join(s.runsRoot, item.runID), filepath.Join(s.runsRoot, item.stageName),
		} {
			if _, err := os.Lstat(path); os.IsNotExist(err) {
				continue
			}
			if err := s.requireControlledPairMembership(path, pair.PairID, item.runID, item.role); err != nil {
				return err
			}
			if err := os.RemoveAll(path); err != nil {
				return err
			}
		}
	}
	pairDir, _ := s.controlledPairDir(pair.PairID)
	if err := os.RemoveAll(pairDir); err != nil {
		return err
	}
	if err := s.pairPersistence.SyncDirectory(s.runsRoot, "controlled pair publication rollback"); err != nil {
		return err
	}
	return s.pairPersistence.SyncDirectory(s.controlledPairRoot, "controlled pair transaction rollback")
}

func (s *Store) requireControlledPairMembership(path, pairID, runID, role string) error {
	if err := requirePrivateDirectory(path); err != nil {
		return err
	}
	var membership controlledPairMembership
	if err := readJSON(filepath.Join(path, controlledPairMembershipFile), &membership); err != nil {
		return err
	}
	if membership.SchemaVersion != SchemaVersion || membership.PairID != pairID ||
		membership.RunID != runID || membership.Role != role {
		return fmt.Errorf("%w: controlled pair transaction destination is owned by another aggregate", ErrConflict)
	}
	return nil
}

func (s *Store) recoverControlledPairStart(pair controlledPairManifest) error {
	if err := s.validatePublishedControlledPair(pair); err != nil {
		return err
	}
	for index, run := range []Run{pair.BaselineRun, pair.CandidateRun} {
		runDir := filepath.Join(s.runsRoot, run.ID)
		role := controlledPairRoleBaseline
		if index == 1 {
			role = controlledPairRoleCandidate
		}
		if err := s.requireControlledPairMembership(runDir, pair.PairID, run.ID, role); err != nil {
			return err
		}
		if err := s.writeRunStatusDurably(runDir, run); err != nil {
			return err
		}
		if err := restoreInitialControlledPairEvent(filepath.Join(runDir, eventsFileName), run.ID); err != nil {
			return err
		}
	}
	pair.State = controlledPairStatePending
	pair.StartedAt, pair.StartReceiptDigest = nil, ""
	if err := s.validatePublishedControlledPair(pair); err != nil {
		return err
	}
	return s.writeControlledPairDurably(pair)
}

func restoreInitialControlledPairEvent(path, runID string) error {
	file, err := openBundleFile(path, os.O_RDONLY)
	if err != nil {
		return err
	}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 4*1024), maxWorkerEventLineBytes)
	if !scanner.Scan() {
		_ = file.Close()
		return fmt.Errorf("%w: controlled pair event history omits its initial snapshot", ErrInvalid)
	}
	first := append([]byte(nil), scanner.Bytes()...)
	if closeErr := file.Close(); closeErr != nil {
		return closeErr
	}
	event, err := decodeStoredEvent(first)
	if err != nil || event.ID != "1" || event.RunID != runID || event.Type != "snapshot" {
		return fmt.Errorf("%w: controlled pair initial event is invalid", ErrInvalid)
	}
	return writeBytesAtomic(path, append(first, '\n'))
}

func writeBytesAtomic(path string, data []byte) error {
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".tmp-evaluation-*")
	if err != nil {
		return err
	}
	tempName := temp.Name()
	defer func() { _ = os.Remove(tempName) }()
	if err := temp.Chmod(0o600); err != nil {
		_ = temp.Close()
		return err
	}
	if _, err := temp.Write(data); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tempName, path); err != nil {
		return err
	}
	return syncEvaluationDirectory(dir, "controlled pair event recovery")
}

func (s *Store) validatePublishedControlledPair(pair controlledPairManifest) error {
	var memberManifests [2]RunManifest
	for index, run := range []Run{pair.BaselineRun, pair.CandidateRun} {
		runDir := filepath.Join(s.runsRoot, run.ID)
		role := controlledPairRoleBaseline
		if index == 1 {
			role = controlledPairRoleCandidate
		}
		if err := s.requireControlledPairMembership(runDir, pair.PairID, run.ID, role); err != nil {
			return err
		}
		physical, err := s.getRunPhysical(run.ID)
		if err != nil {
			return err
		}
		var manifest RunManifest
		if err := readJSON(filepath.Join(runDir, manifestFileName), &manifest); err != nil {
			return err
		}
		if err := validateRunManifestContract(manifest); err != nil {
			return err
		}
		memberManifests[index] = manifest
		if err := validateRunManifestFrozenFields(physical, manifest); err != nil {
			return err
		}
		var lifecycle RunLifecycle
		if err := readJSON(filepath.Join(runDir, lifecycleFileName), &lifecycle); err != nil {
			return err
		}
		if err := validateRunLifecycle(physical, lifecycle); err != nil ||
			lifecycle.OwnerPrincipalDigest != pair.OwnerPrincipalDigest {
			return fmt.Errorf("%w: controlled pair ownership binding is invalid", ErrInvalid)
		}
		switch pair.State {
		case controlledPairStatePending, controlledPairStateTerminal:
			if !reflect.DeepEqual(physical, run) {
				return fmt.Errorf("%w: controlled pair member differs from its aggregate snapshot", ErrInvalid)
			}
		case controlledPairStateStarting:
			running := controlledPairRunningSnapshot(run, *pair.StartedAt)
			if !reflect.DeepEqual(physical, run) && !reflect.DeepEqual(physical, running) {
				return fmt.Errorf("%w: controlled pair starting member has invalid recovery state", ErrInvalid)
			}
		case controlledPairStateRunning, controlledPairStateCancelling:
			if err := validateControlledPairRuntimeMember(pair, run, physical); err != nil {
				return err
			}
		}
	}
	return s.validateControlledPairAuthoritativeIdentityUnlocked(pair, memberManifests[0], memberManifests[1])
}

func validateControlledPairRuntimeMember(pair controlledPairManifest, frozen, physical Run) error {
	expected := frozen
	expected.Status = physical.Status
	expected.Progress = physical.Progress
	expected.StartedAt = physical.StartedAt
	expected.CompletedAt = physical.CompletedAt
	expected.Error = physical.Error
	if !reflect.DeepEqual(expected, physical) {
		return fmt.Errorf("%w: controlled pair member immutable identity changed", ErrInvalid)
	}
	if pair.StartedAt == nil || physical.StartedAt == nil || !physical.StartedAt.Equal(*pair.StartedAt) {
		return fmt.Errorf("%w: controlled pair member start receipt changed", ErrInvalid)
	}
	if physical.Status != StatusRunning && physical.Status != StatusSealing && !terminalStatus(physical.Status) {
		return fmt.Errorf("%w: controlled pair member has an invalid runtime state", ErrInvalid)
	}
	return nil
}
