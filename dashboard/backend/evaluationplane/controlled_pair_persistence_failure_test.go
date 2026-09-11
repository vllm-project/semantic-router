package evaluationplane

import (
	"errors"
	"os"
	"path/filepath"
	"sync"
)

const injectedControlledPairPersistenceFailure = "injected controlled pair persistence failure"

type controlledPairPersistenceFailureCase struct {
	name    string
	install func(*Store, controlledPairManifest)
}

type failingControlledPairManifestPersistence struct {
	controlledPairPersistence
	state  string
	failed bool
}

func (p *failingControlledPairManifestPersistence) WriteManifest(
	path string,
	pair controlledPairManifest,
) error {
	if err := p.controlledPairPersistence.WriteManifest(path, pair); err != nil {
		return err
	}
	if pair.State == p.state && !p.failed {
		p.failed = true
		return errors.New(injectedControlledPairPersistenceFailure)
	}
	return nil
}

type failingControlledPairRenamePersistence struct {
	controlledPairPersistence
	runID  string
	failed bool
}

func (p *failingControlledPairRenamePersistence) Rename(source, destination string) error {
	if err := p.controlledPairPersistence.Rename(source, destination); err != nil {
		return err
	}
	if filepath.Base(destination) == p.runID && !p.failed {
		p.failed = true
		return errors.New(injectedControlledPairPersistenceFailure)
	}
	return nil
}

type failingControlledPairRemovalPersistence struct {
	controlledPairPersistence
	runID  string
	failed bool
}

func (p *failingControlledPairRemovalPersistence) RemoveAll(path string) error {
	if err := p.controlledPairPersistence.RemoveAll(path); err != nil {
		return err
	}
	if filepath.Base(path) == p.runID && !p.failed {
		p.failed = true
		return errors.New(injectedControlledPairPersistenceFailure)
	}
	return nil
}

type failingControlledPairDirectorySyncPersistence struct {
	controlledPairPersistence
	description string
	failed      bool
}

func (p *failingControlledPairDirectorySyncPersistence) SyncDirectory(path, description string) error {
	if err := p.controlledPairPersistence.SyncDirectory(path, description); err != nil {
		return err
	}
	if description == p.description && !p.failed {
		p.failed = true
		return errors.New(injectedControlledPairPersistenceFailure)
	}
	return nil
}

type failingControlledPairRunStatusPersistence struct {
	runStatusPersistence
	runID  string
	status RunStatus
	failed bool
}

func (p *failingControlledPairRunStatusPersistence) Write(path string, run Run) error {
	if err := p.runStatusPersistence.Write(path, run); err != nil {
		return err
	}
	if filepath.Base(filepath.Dir(path)) == p.runID && run.Status == p.status && !p.failed {
		p.failed = true
		return errors.New(injectedControlledPairPersistenceFailure)
	}
	return nil
}

type failingControlledPairRunEventPersistence struct {
	runEventPersistence
	runID  string
	failed bool
}

func (p *failingControlledPairRunEventPersistence) Append(path string, encoded []byte) error {
	if err := p.runEventPersistence.Append(path, encoded); err != nil {
		return err
	}
	if filepath.Base(filepath.Dir(path)) == p.runID && !p.failed {
		p.failed = true
		return errors.New(injectedControlledPairPersistenceFailure)
	}
	return nil
}

type pausingControlledPairManifestPersistence struct {
	controlledPairPersistence
	state   string
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (p *pausingControlledPairManifestPersistence) WriteManifest(
	path string,
	pair controlledPairManifest,
) error {
	if err := p.controlledPairPersistence.WriteManifest(path, pair); err != nil {
		return err
	}
	if pair.State == p.state {
		p.once.Do(func() {
			close(p.entered)
			<-p.release
		})
	}
	return nil
}

type exitingControlledPairRunStatusPersistence struct {
	runStatusPersistence
	runID    string
	status   RunStatus
	exitCode int
}

func (p *exitingControlledPairRunStatusPersistence) Write(path string, run Run) error {
	if err := p.runStatusPersistence.Write(path, run); err != nil {
		return err
	}
	if filepath.Base(filepath.Dir(path)) == p.runID && run.Status == p.status {
		os.Exit(p.exitCode)
	}
	return nil
}

func failAfterControlledPairManifestState(store *Store, state string) {
	store.pairPersistence = &failingControlledPairManifestPersistence{
		controlledPairPersistence: store.pairPersistence,
		state:                     state,
	}
}

func failAfterControlledPairRename(store *Store, runID string) {
	store.pairPersistence = &failingControlledPairRenamePersistence{
		controlledPairPersistence: store.pairPersistence,
		runID:                     runID,
	}
}

func failAfterControlledPairRemoval(store *Store, runID string) {
	store.pairPersistence = &failingControlledPairRemovalPersistence{
		controlledPairPersistence: store.pairPersistence,
		runID:                     runID,
	}
}

func failAfterControlledPairDirectorySync(store *Store, description string) {
	store.pairPersistence = &failingControlledPairDirectorySyncPersistence{
		controlledPairPersistence: store.pairPersistence,
		description:               description,
	}
}

func failAfterControlledPairRunStatus(store *Store, runID string, status RunStatus) {
	store.statusPersistence = &failingControlledPairRunStatusPersistence{
		runStatusPersistence: store.statusPersistence,
		runID:                runID,
		status:               status,
	}
}

func failAfterControlledPairRunEvent(store *Store, runID string) {
	store.eventPersistence = &failingControlledPairRunEventPersistence{
		runEventPersistence: store.eventPersistence,
		runID:               runID,
	}
}

func pauseAfterControlledPairManifestState(
	store *Store,
	state string,
	entered, release chan struct{},
) {
	store.pairPersistence = &pausingControlledPairManifestPersistence{
		controlledPairPersistence: store.pairPersistence,
		state:                     state, entered: entered, release: release,
	}
}

func controlledPairPublicationFailureCases() []controlledPairPersistenceFailureCase {
	return []controlledPairPersistenceFailureCase{
		{name: "publication_intent", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairManifestState(store, controlledPairStatePublishing)
		}},
		{name: "baseline_published", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRename(store, pair.BaselineRunID)
		}},
		{name: "candidate_published", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRename(store, pair.CandidateRunID)
		}},
		{name: "publication_committed", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairManifestState(store, controlledPairStatePending)
		}},
	}
}

func controlledPairStartFailureCases() []controlledPairPersistenceFailureCase {
	return []controlledPairPersistenceFailureCase{
		{name: "start_intent", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairManifestState(store, controlledPairStateStarting)
		}},
		{name: "baseline_running", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRunStatus(store, pair.BaselineRunID, StatusRunning)
		}},
		{name: "candidate_running", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRunStatus(store, pair.CandidateRunID, StatusRunning)
		}},
		{name: "baseline_start_event", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRunEvent(store, pair.BaselineRunID)
		}},
		{name: "candidate_start_event", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRunEvent(store, pair.CandidateRunID)
		}},
		{name: "start_committed", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairManifestState(store, controlledPairStateRunning)
		}},
	}
}

func controlledPairCancellationFailureCases() []controlledPairPersistenceFailureCase {
	return []controlledPairPersistenceFailureCase{
		{name: "cancel_intent", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairManifestState(store, controlledPairStateCancelling)
		}},
		{name: "baseline_cancelled", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRunStatus(store, pair.BaselineRunID, StatusCancelled)
		}},
		{name: "candidate_cancelled", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRunStatus(store, pair.CandidateRunID, StatusCancelled)
		}},
		{name: "cancel_committed", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairManifestState(store, controlledPairStateTerminal)
		}},
	}
}

func controlledPairDeletionFailureCases() []controlledPairPersistenceFailureCase {
	return []controlledPairPersistenceFailureCase{
		{name: "delete_intent", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairManifestState(store, controlledPairStateDeleting)
		}},
		{name: "baseline_namespace_removed", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRemoval(store, pair.BaselineRunID)
		}},
		{name: "candidate_namespace_removed", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRemoval(store, pair.CandidateRunID)
		}},
		{name: "member_cleanup_committed", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairDirectorySync(store, "controlled pair deletion")
		}},
		{name: "delete_committed", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairDirectorySync(store, "controlled pair tombstone publication")
		}},
	}
}
