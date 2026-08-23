package backendinvoker

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

// AuthoritativeAttemptJournal adapts BackendInvoker's execution seam to the
// cross-replica Valkey attempt journal. The Plan has already been authenticated
// by the request-bound dispatch capability; every store mutation independently
// revalidates its pending admission and immutable dispatch-plan digest.
type AuthoritativeAttemptJournal struct {
	engine quotaruntime.AttemptEvidenceEngine
}

var _ Journal = (*AuthoritativeAttemptJournal)(nil)

func NewAuthoritativeAttemptJournal(
	engine quotaruntime.AttemptEvidenceEngine,
) (*AuthoritativeAttemptJournal, error) {
	if engine == nil {
		return nil, fmt.Errorf("attempt evidence engine is required")
	}
	return &AuthoritativeAttemptJournal{engine: engine}, nil
}

func (j *AuthoritativeAttemptJournal) BeginDispatch(
	ctx context.Context,
	plan Plan,
	deadline time.Time,
) error {
	if j == nil || j.engine == nil {
		return fmt.Errorf("attempt evidence engine is required")
	}
	if plan.Ordinal < 0 || uint64(plan.Ordinal) > math.MaxUint32 ||
		plan.Execution.MaxRetries < 0 || plan.Execution.MaxRetries >= 6 {
		return fmt.Errorf("dispatch or attempt bound is outside the journal range")
	}
	_, err := j.engine.BeginDispatch(ctx, quotaruntime.BeginDispatchRequest{
		DispatchReference: journalDispatchReference(plan),
		DispatchType:      plan.DispatchType,
		Ordinal:           uint32(plan.Ordinal),
		Deadline:          deadline.UTC().Truncate(time.Millisecond),
		MaxAttempts:       uint32(plan.Execution.MaxRetries + 1),
	})
	if err != nil {
		return fmt.Errorf("begin authoritative dispatch evidence: %w", err)
	}
	return nil
}

func (j *AuthoritativeAttemptJournal) BeginAttempt(
	ctx context.Context,
	plan Plan,
	attempt Attempt,
) error {
	backend, err := attemptBackend(plan, attempt)
	if err != nil {
		return err
	}
	if attempt.Number < 1 || uint64(attempt.Number) > math.MaxUint32 {
		return fmt.Errorf("attempt number is outside the journal range")
	}
	_, err = j.engine.BeginAttempt(ctx, quotaruntime.BeginAttemptRequest{
		DispatchReference: journalDispatchReference(plan),
		AttemptID:         attempt.ID,
		AttemptNumber:     uint32(attempt.Number),
		BackendID:         backend.ID,
		ProviderID:        backend.ProviderID,
	})
	if err != nil {
		return fmt.Errorf("begin authoritative attempt evidence: %w", err)
	}
	return nil
}

func (j *AuthoritativeAttemptJournal) FinishAttempt(
	ctx context.Context,
	plan Plan,
	result AttemptResult,
) error {
	backend, err := attemptBackend(plan, result.Attempt)
	if err != nil {
		return err
	}
	if result.Number < 1 || uint64(result.Number) > math.MaxUint32 {
		return fmt.Errorf("attempt number is outside the journal range")
	}
	state, err := runtimeAttemptState(result.State)
	if err != nil {
		return err
	}
	_, err = j.engine.FinishAttempt(ctx, quotaruntime.FinishAttemptRequest{
		DispatchReference: journalDispatchReference(plan),
		AttemptID:         result.ID,
		AttemptNumber:     uint32(result.Number),
		BackendID:         backend.ID,
		ProviderID:        backend.ProviderID,
		State:             state,
		StatusCode:        result.StatusCode,
		ErrorCode:         result.ErrorCode,
	})
	if err != nil {
		return fmt.Errorf("finish authoritative attempt evidence: %w", err)
	}
	return nil
}

func journalDispatchReference(plan Plan) quotaruntime.DispatchReference {
	return quotaruntime.DispatchReference{
		Partition:          plan.QuotaPartition,
		AdmissionID:        plan.AdmissionID,
		AdmissionDigest:    plan.AdmissionDigest,
		DispatchID:         plan.DispatchID,
		DispatchPlanDigest: plan.DispatchPlanDigest,
		ModelID:            plan.ModelID,
		ModelRevision:      plan.ModelRevision,
		RequestDigest:      plan.RequestDigest,
	}
}

func attemptBackend(plan Plan, attempt Attempt) (Backend, error) {
	var selected Backend
	matches := 0
	for _, backend := range plan.Backends {
		if backend.ID == attempt.BackendID {
			selected = backend
			matches++
		}
	}
	if matches != 1 {
		return Backend{}, fmt.Errorf(
			"attempt backend %q must resolve to exactly one immutable backend",
			attempt.BackendID,
		)
	}
	return selected, nil
}

func runtimeAttemptState(state AttemptState) (quotaruntime.AttemptEvidenceState, error) {
	switch state {
	case AttemptKnownZero:
		return quotaruntime.AttemptEvidenceKnownZero, nil
	case AttemptResponseStarted:
		return quotaruntime.AttemptEvidenceResponseStarted, nil
	case AttemptUnknown:
		return quotaruntime.AttemptEvidenceUnknown, nil
	default:
		return "", fmt.Errorf("unsupported backend attempt state %q", state)
	}
}
