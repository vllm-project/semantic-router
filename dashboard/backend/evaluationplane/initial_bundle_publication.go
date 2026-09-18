package evaluationplane

import (
	"fmt"
	"os"
	"path/filepath"
)

type initialBundleSpec struct {
	run       Run
	manifest  RunManifest
	lifecycle RunLifecycle
	decorate  func(string) error
}

type stagedInitialBundle struct {
	spec        initialBundleSpec
	stagedDir   string
	destination string
}

// prepareInitialBundlePublicationUnlocked is the shared preparation engine for
// both ordinary one-run publication and controlled-pair aggregate publication.
// Its caller owns the lifecycle, evidence, index, and Store locks.
func (s *Store) prepareInitialBundlePublicationUnlocked(
	actor Actor,
	specs []initialBundleSpec,
	aggregateReservationBytes int64,
) ([]stagedInitialBundle, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("%w: initial bundle publication is empty", ErrInvalid)
	}
	// A hidden ordinary-run deletion owns its UUID until an explicit DeleteRun
	// retry (or startup recovery) closes the parent-directory commit cut. No
	// ordinary or aggregate publication may bind a new live identity beside it.
	if err := s.requireNoRunDeletionIntentsUnlocked(); err != nil {
		return nil, err
	}
	runIDs := make([]string, 0, len(specs))
	for _, spec := range specs {
		runIDs = append(runIDs, spec.run.ID)
	}
	if err := s.requireUnreservedControlledPairMemberIDsUnlocked(actor, runIDs...); err != nil {
		return nil, err
	}
	prepared := make([]stagedInitialBundle, 0, len(specs))
	cleanup := func() {
		for _, item := range prepared {
			_ = os.RemoveAll(item.stagedDir)
		}
	}
	for _, spec := range specs {
		destination := filepath.Join(s.runsRoot, spec.run.ID)
		if err := s.requireAvailableRunDestinationUnlocked(actor, spec.run.ID, destination); err != nil {
			cleanup()
			return nil, err
		}
	}
	if reason, err := s.requireCreateQuotaUnlocked(actor, len(specs), aggregateReservationBytes); err != nil {
		for _, spec := range specs {
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceRun, "create", "denied", reason, spec.run.ID, actor.principalDigest,
			); auditErr != nil {
				cleanup()
				return nil, auditErr
			}
		}
		cleanup()
		return nil, err
	}
	for _, spec := range specs {
		stagedDir, err := s.stageInitialRunBundleUnlocked(
			actor, spec.run, spec.manifest, spec.lifecycle,
		)
		if err != nil {
			cleanup()
			return nil, err
		}
		if spec.decorate != nil {
			if err := spec.decorate(stagedDir); err != nil {
				_ = os.RemoveAll(stagedDir)
				cleanup()
				return nil, err
			}
			if err := syncEvaluationDirectory(stagedDir, "staged controlled pair member"); err != nil {
				_ = os.RemoveAll(stagedDir)
				cleanup()
				return nil, err
			}
		}
		prepared = append(prepared, stagedInitialBundle{
			spec: spec, stagedDir: stagedDir, destination: filepath.Join(s.runsRoot, spec.run.ID),
		})
	}
	// UUID collision is exceptional, but another Store instance can race this
	// process-local lock. Recheck the complete destination set before publishing.
	for _, item := range prepared {
		if err := s.requireAvailableRunDestinationUnlocked(actor, item.spec.run.ID, item.destination); err != nil {
			cleanup()
			return nil, err
		}
	}
	return prepared, nil
}

func cleanupStagedInitialBundles(items []stagedInitialBundle) {
	for _, item := range items {
		_ = os.RemoveAll(item.stagedDir)
	}
}

func publishStagedInitialBundle(item stagedInitialBundle, rename func(string, string) error) error {
	if err := rename(item.stagedDir, item.destination); err != nil {
		if _, statErr := os.Lstat(item.destination); statErr == nil {
			return fmt.Errorf("%w: run %s already exists", ErrConflict, item.spec.run.ID)
		}
		return fmt.Errorf("publish run bundle: %w", err)
	}
	return nil
}
