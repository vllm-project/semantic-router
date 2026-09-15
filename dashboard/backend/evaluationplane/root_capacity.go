package evaluationplane

import (
	"fmt"
	"sync"
)

func (coordinator *evaluationRootCoordinator) validateWorkerCapacity(limit int) error {
	coordinator.capacityMu.Lock()
	defer coordinator.capacityMu.Unlock()
	if limit <= 0 {
		return fmt.Errorf("%w: evaluation worker capacity must be positive", ErrInvalid)
	}
	if coordinator.capacityLoaded && coordinator.workerCapacity != limit {
		return fmt.Errorf(
			"%w: evaluation worker capacity %d does not match the active root capacity %d",
			ErrConflict, limit, coordinator.workerCapacity,
		)
	}
	return nil
}

// commitWorkerCapacity installs the first fully initialized Service's worker
// budget. A failed opener cannot leave a configuration behind for its retry.
func (coordinator *evaluationRootCoordinator) commitWorkerCapacity(limit int) error {
	coordinator.capacityMu.Lock()
	defer coordinator.capacityMu.Unlock()
	if limit <= 0 {
		return fmt.Errorf("%w: evaluation worker capacity must be positive", ErrInvalid)
	}
	if coordinator.capacityLoaded {
		if coordinator.workerCapacity != limit {
			return fmt.Errorf(
				"%w: evaluation worker capacity %d does not match the active root capacity %d",
				ErrConflict, limit, coordinator.workerCapacity,
			)
		}
		return nil
	}
	coordinator.workerSlots = make(chan struct{}, limit)
	coordinator.workerCapacity = limit
	coordinator.capacityLoaded = true
	return nil
}

// reserveWorkerSlots is one atomic root-wide admission decision. In
// particular, a controlled pair either owns both slots or owns neither.
func (coordinator *evaluationRootCoordinator) reserveWorkerSlots(count int) (func(), bool) {
	coordinator.capacityMu.Lock()
	defer coordinator.capacityMu.Unlock()
	if !coordinator.capacityLoaded || count <= 0 ||
		count > coordinator.workerCapacity-len(coordinator.workerSlots) {
		return nil, false
	}
	for range count {
		coordinator.workerSlots <- struct{}{}
	}
	var once sync.Once
	return func() {
		once.Do(func() { coordinator.releaseWorkerSlots(count) })
	}, true
}

func (coordinator *evaluationRootCoordinator) releaseWorkerSlots(count int) {
	coordinator.capacityMu.Lock()
	defer coordinator.capacityMu.Unlock()
	for range count {
		<-coordinator.workerSlots
	}
}
