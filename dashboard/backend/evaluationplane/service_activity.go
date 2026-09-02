package evaluationplane

import "context"

// The root coordinator is shared by every Service using one durable root. A
// separate mutex keeps worker cleanup from inverting the lifecycle lock order.
func (c *evaluationRootCoordinator) claim(runIDs []string, cancellations []context.CancelFunc) bool {
	if len(runIDs) == 0 || len(runIDs) != len(cancellations) {
		return false
	}
	for _, cancel := range cancellations {
		if cancel == nil {
			return false
		}
	}
	seen := make(map[string]bool, len(runIDs))
	for _, runID := range runIDs {
		if runID == "" || seen[runID] {
			return false
		}
		seen[runID] = true
	}
	c.activityMu.Lock()
	defer c.activityMu.Unlock()
	for _, runID := range runIDs {
		if c.activeRuns[runID] != nil {
			return false
		}
	}
	for index, runID := range runIDs {
		c.activeRuns[runID] = cancellations[index]
	}
	return true
}

func (c *evaluationRootCoordinator) release(runID string) {
	c.activityMu.Lock()
	defer c.activityMu.Unlock()
	delete(c.activeRuns, runID)
}

func (c *evaluationRootCoordinator) contains(runID string) bool {
	c.activityMu.Lock()
	defer c.activityMu.Unlock()
	return c.activeRuns[runID] != nil
}

func (c *evaluationRootCoordinator) countActiveRuns(runIDs ...string) int {
	c.activityMu.Lock()
	defer c.activityMu.Unlock()
	count := 0
	for _, runID := range runIDs {
		if c.activeRuns[runID] != nil {
			count++
		}
	}
	return count
}

// requestCancel copies owner callbacks under the activity mutex and invokes
// them only after unlocking. The worker owner remains responsible for release,
// so remote cancellation cannot make deletion visible before process exit.
func (c *evaluationRootCoordinator) requestCancel(runIDs ...string) int {
	c.activityMu.Lock()
	cancellations := make([]context.CancelFunc, 0, len(runIDs))
	for _, runID := range runIDs {
		if cancel := c.activeRuns[runID]; cancel != nil {
			cancellations = append(cancellations, cancel)
		}
	}
	c.activityMu.Unlock()
	for _, cancel := range cancellations {
		cancel()
	}
	return len(cancellations)
}
