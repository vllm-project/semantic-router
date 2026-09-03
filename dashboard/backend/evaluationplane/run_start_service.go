package evaluationplane

import (
	"context"
	"fmt"
	"reflect"
)

func (s *Service) StartRunAs(ctx context.Context, actor Actor, id string) (Run, error) {
	release, err := s.beginOperation()
	if err != nil {
		return Run{}, err
	}
	defer release()
	return s.startRunAsInternal(ctx, actor, id)
}

func (s *Service) startRunAsInternal(ctx context.Context, actor Actor, id string) (Run, error) {
	if ctx == nil {
		return Run{}, fmt.Errorf("%w: evaluation start context is required", ErrInvalid)
	}
	if err := ctx.Err(); err != nil {
		return Run{}, err
	}
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	if err := s.store.authorizeRunActionUnlocked(actor, id, "start"); err != nil {
		return Run{}, err
	}
	return s.startRunInternal(ctx, id)
}

func (s *Service) startRunInternal(ctx context.Context, id string) (Run, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return Run{}, fmt.Errorf("%w: evaluation service is closed", ErrConflict)
	}
	if err := ctx.Err(); err != nil {
		return Run{}, err
	}
	run, err := s.store.GetRun(id)
	if err != nil {
		return Run{}, err
	}
	if run.Status == StatusSealing || terminalStatus(run.Status) {
		runDir, pathErr := s.store.checkedRunDir(id)
		if pathErr != nil {
			return Run{}, pathErr
		}
		if syncErr := s.store.syncRunStatusDirectory(runDir, "evaluation run start retry"); syncErr != nil {
			return Run{}, syncErr
		}
		return run, nil
	}
	return s.resumeOrdinaryRunLaunchLocked(ctx, run)
}

func (s *Service) validateRunStart(run Run) error {
	manifest, _, err := s.readDurableManifest(run.ID)
	if err != nil {
		return err
	}
	if manifest.CodeRevision != s.codeRevision {
		return fmt.Errorf("%w: pending run source revision does not match the active evaluation worker", ErrConflict)
	}
	registry, err := s.registrySnapshot()
	if err != nil {
		return err
	}
	executorID, singleExecutor := manifestExecutorIdentity(manifest)
	executor, registered := registry.executor(executorID)
	if !singleExecutor || !registered || executor.Mode != manifest.Mode {
		return fmt.Errorf("%w: pending run executor is not registered for its frozen mode", ErrInvalid)
	}
	if manifest.GateContractVersion != GateContractVersion ||
		!reflect.DeepEqual(manifest.SuiteRevisions, suiteRevisionSnapshot(registry, manifest.SuiteIDs)) ||
		!reflect.DeepEqual(manifest.SuiteExecutors, suiteExecutorSnapshot(registry, manifest.SuiteIDs, manifest.Mode)) {
		return fmt.Errorf("%w: pending run suite or change-profile contract revision does not match the active evaluation worker", ErrConflict)
	}
	_, currentTarget, err := s.validateCreateRequest(registry, CreateRunRequest{
		ClientRequestID: run.ClientRequestID,
		Name:            run.Name, Description: run.Description,
		SuiteIDs: run.SuiteIDs, TrackIDs: run.TrackIDs,
		Mode: run.Mode, TargetID: run.TargetID, ChangeProfile: run.ChangeProfile,
		SampleLimit: run.SampleLimit, Concurrency: run.Concurrency, Seed: run.Seed,
		CapacitySLO:          copyCapacitySLO(run.CapacitySLO),
		CapacityLoadProtocol: copyCapacityLoadProtocol(run.CapacityLoadProtocol),
		BaselineRunID:        run.BaselineRunID,
	})
	if err != nil {
		return fmt.Errorf("%w: run target is no longer supported", ErrConflict)
	}
	mixtureDrift := manifest.Mode == ModeLive && (currentTarget.Mixture == nil ||
		manifest.PolicySnapshotDigest != currentTarget.Mixture.RecipeDigest)
	if !manifestMatchesTargetDefinition(manifest.Target, currentTarget) || mixtureDrift {
		return fmt.Errorf("%w: pending run mixture no longer matches the active recipe, pool, or binding", ErrConflict)
	}
	if manifest.ConfigDigest != currentTarget.ConfigDigest {
		return fmt.Errorf("%w: pending run config digest no longer matches the active target", ErrConflict)
	}
	return nil
}

func suiteRevisionSnapshot(registry *Registry, suiteIDs []string) map[string]string {
	revisions := make(map[string]string, len(suiteIDs))
	for _, suiteID := range suiteIDs {
		if suite, ok := registry.suite(suiteID); ok {
			revisions[suiteID] = suite.Revision
		}
	}
	return revisions
}

func suiteExecutorSnapshot(registry *Registry, suiteIDs []string, mode Mode) map[string]string {
	executors := make(map[string]string, len(suiteIDs))
	for _, suiteID := range suiteIDs {
		if suite, ok := registry.suite(suiteID); ok {
			if executor, executable := suiteExecutorForMode(suite, mode); executable {
				executors[suiteID] = executor
			}
		}
	}
	return executors
}
