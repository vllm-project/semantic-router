package evaluationplane

import (
	"context"
	"fmt"
	"sync"
)

func (s *Service) acquireControlledPairLaunch(pairID string) (bool, chan struct{}) {
	coordinator := s.store.lifecycle
	coordinator.controlledPairLaunchMu.Lock()
	defer coordinator.controlledPairLaunchMu.Unlock()
	if done, exists := coordinator.controlledPairLaunches[pairID]; exists {
		return false, done
	}
	done := make(chan struct{})
	coordinator.controlledPairLaunches[pairID] = done
	return true, done
}

func (s *Service) releaseControlledPairLaunch(pairID string, done chan struct{}) {
	coordinator := s.store.lifecycle
	coordinator.controlledPairLaunchMu.Lock()
	defer coordinator.controlledPairLaunchMu.Unlock()
	if current, exists := coordinator.controlledPairLaunches[pairID]; exists && current == done {
		delete(coordinator.controlledPairLaunches, pairID)
		close(done)
	}
}

// beginControlledPairPrelaunch binds admission and credential freezing to both
// the caller and this Service's shutdown scope. Close first closes that scope
// and then waits for every registered prelaunch to leave, so a closed Service
// cannot publish a pair after Close returns.
func (s *Service) beginControlledPairPrelaunch(ctx context.Context) (context.Context, func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, nil, fmt.Errorf("%w: evaluation service is closed", ErrConflict)
	}
	s.prelaunches.Add(1)
	s.prelaunchCount++
	serviceContext := s.prelaunchContext
	s.mu.Unlock()

	prelaunchContext, cancel := context.WithCancel(serviceContext)
	stopCallerCancellation := context.AfterFunc(ctx, cancel)
	var once sync.Once
	return prelaunchContext, func() {
		once.Do(func() {
			stopCallerCancellation()
			cancel()
			s.mu.Lock()
			s.prelaunchCount--
			s.mu.Unlock()
			s.prelaunches.Done()
		})
	}, nil
}

func controlledPairPrelaunchErr(caller, prelaunch context.Context) error {
	if err := caller.Err(); err != nil {
		return err
	}
	return prelaunch.Err()
}

func (s *Service) reserveControlledPairWorkerSlots(caller context.Context) (func(), error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := caller.Err(); err != nil {
		return nil, err
	}
	if s.closed {
		return nil, fmt.Errorf("%w: evaluation service is closed", ErrConflict)
	}
	release, reserved := s.activity.reserveWorkerSlots(2)
	if !reserved {
		return nil, fmt.Errorf(
			"%w: two evaluation worker slots are required for controlled pairing", ErrConflict,
		)
	}
	return release, nil
}

func (s *Service) startControlledPairRunsAs(
	caller context.Context,
	actor Actor,
	pairID string,
	baselineContext, candidateContext *controlledPairRunContext,
) (Run, Run, bool, error) {
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	return s.startControlledPairRunsInternal(caller, actor, pairID, baselineContext, candidateContext)
}

func (s *Service) startControlledPairRunsInternal(
	caller context.Context,
	actor Actor,
	pairID string,
	baselineContext, candidateContext *controlledPairRunContext,
) (Run, Run, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := caller.Err(); err != nil {
		return Run{}, Run{}, false, err
	}
	if s.closed {
		return Run{}, Run{}, false, fmt.Errorf("%w: evaluation service is closed", ErrConflict)
	}
	pair, err := s.store.readControlledPair(pairID)
	if err != nil {
		return Run{}, Run{}, false, err
	}
	baselineManifest, _, baselineManifestErr := s.readDurableManifest(pair.BaselineRunID)
	if baselineManifestErr != nil {
		return Run{}, Run{}, false, baselineManifestErr
	}
	candidateManifest, _, candidateManifestErr := s.readDurableManifest(pair.CandidateRunID)
	if candidateManifestErr != nil {
		return Run{}, Run{}, false, candidateManifestErr
	}
	registry, registryErr := s.registrySnapshot()
	if registryErr != nil {
		return Run{}, Run{}, false, registryErr
	}
	if validationErr := validateControlledPairRegistryTargets(registry, baselineManifest, candidateManifest); validationErr != nil {
		return Run{}, Run{}, false, validationErr
	}
	baselinePath, err := s.store.ManifestPath(pair.BaselineRunID)
	if err != nil {
		return Run{}, Run{}, false, err
	}
	candidatePath, err := s.store.ManifestPath(pair.CandidateRunID)
	if err != nil {
		return Run{}, Run{}, false, err
	}
	start, startErr := s.store.startControlledPairAs(actor, pairID)
	if startErr != nil {
		return Run{}, Run{}, false, startErr
	}
	if start.Pair.State != controlledPairStateRunning || start.Baseline.Status != StatusRunning || start.Candidate.Status != StatusRunning {
		return Run{}, Run{}, false, fmt.Errorf("%w: controlled pair start did not commit one running aggregate", ErrConflict)
	}
	if !start.LaunchOwner {
		switch s.activity.countActiveRuns(start.Baseline.ID, start.Candidate.ID) {
		case 2:
			return start.Baseline, start.Candidate, false, nil
		case 1:
			return Run{}, Run{}, false, fmt.Errorf(
				"%w: controlled pair has partial worker ownership", ErrConflict,
			)
		}
	}
	if s.active[start.Baseline.ID] != nil || s.active[start.Candidate.ID] != nil {
		return Run{}, Run{}, false, fmt.Errorf("%w: controlled pair launch ownership overlaps active workers", ErrConflict)
	}
	baselineWorkerContext, baselineCancel := context.WithTimeout(context.Background(), s.workerTimeout)
	candidateWorkerContext, candidateCancel := context.WithTimeout(context.Background(), s.workerTimeout)
	if !s.activity.claim(
		[]string{start.Baseline.ID, start.Candidate.ID},
		[]context.CancelFunc{baselineCancel, candidateCancel},
	) {
		baselineCancel()
		candidateCancel()
		return Run{}, Run{}, false, fmt.Errorf("%w: controlled pair workers already have a live service owner", ErrConflict)
	}
	s.active[start.Baseline.ID], s.active[start.Candidate.ID] = baselineCancel, candidateCancel
	s.workerEvents[start.Baseline.ID], s.workerEvents[start.Candidate.ID] = 0, 0
	s.workers.Add(2)
	go s.execute(baselineWorkerContext, start.Baseline.ID, baselinePath, baselineContext)
	go s.execute(candidateWorkerContext, start.Candidate.ID, candidatePath, candidateContext)
	return start.Baseline, start.Candidate, true, nil
}
