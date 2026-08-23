package backendinvoker

import (
	"context"
	"fmt"
	"time"
)

// ProcessLocalJournal is the explicit attempt-evidence strategy for managed
// routing with access enforcement disabled. One Invoke call owns the complete
// retry loop in one process, so no cross-replica quota settlement consumes its
// evidence. Access-enabled deployments must use AuthoritativeAttemptJournal.
type ProcessLocalJournal struct{}

var _ Journal = ProcessLocalJournal{}

func (ProcessLocalJournal) BeginDispatch(_ context.Context, plan Plan, deadline time.Time) error {
	if deadline.IsZero() || !deadline.After(time.Now()) {
		return fmt.Errorf("local dispatch deadline must be in the future")
	}
	return validatePlan(plan)
}

func (ProcessLocalJournal) BeginAttempt(_ context.Context, plan Plan, attempt Attempt) error {
	if attempt.ID == "" || attempt.Number < 1 || attempt.StartedAt.IsZero() {
		return fmt.Errorf("local attempt identity is incomplete")
	}
	_, err := attemptBackend(plan, attempt)
	return err
}

func (ProcessLocalJournal) FinishAttempt(_ context.Context, plan Plan, result AttemptResult) error {
	if result.CompletedAt.IsZero() || result.CompletedAt.Before(result.StartedAt) {
		return fmt.Errorf("local attempt completion is invalid")
	}
	if _, err := attemptBackend(plan, result.Attempt); err != nil {
		return err
	}
	_, err := runtimeAttemptState(result.State)
	return err
}
