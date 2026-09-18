package fallback

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"
)

// Orchestrator manages the bounded execution and fallback lifecycle across model candidates.
type Orchestrator struct {
	policy         FallbackPolicy
	classifier     *Classifier
	circuitBreaker *BackendCircuitBreaker
	nowFunc        func() time.Time
}

// NewOrchestrator constructs a new Orchestrator instance.
func NewOrchestrator(policy FallbackPolicy, cb *BackendCircuitBreaker) *Orchestrator {
	if cb == nil {
		cb = NewBackendCircuitBreaker(policy.CircuitBreaker)
	}
	return &Orchestrator{
		policy:         policy,
		classifier:     NewClassifier(policy),
		circuitBreaker: cb,
		nowFunc:        time.Now,
	}
}

// Policy returns the configured fallback policy.
func (o *Orchestrator) Policy() FallbackPolicy {
	return o.policy
}

// CircuitBreaker returns the backend circuit breaker associated with this orchestrator.
func (o *Orchestrator) CircuitBreaker() *BackendCircuitBreaker {
	if o == nil {
		return nil
	}
	return o.circuitBreaker
}

// EvaluationResult contains the action directive resulting from an attempt evaluation.
type EvaluationResult struct {
	CanFallback  bool
	Succeeded    bool
	TriggerClass TriggerClass
	Reason       string
	Err          error
}

// NewExecutionRecord initializes a fresh execution record for a logical request.
func (o *Orchestrator) NewExecutionRecord(requestID, decisionName, initialModel string) *ExecutionRecord {
	return &ExecutionRecord{
		RequestID:     requestID,
		DecisionName:  decisionName,
		InitialModel:  initialModel,
		StartTime:     o.nowFunc(),
		Attempts:      make([]AttemptOutcome, 0, o.policy.MaxAttempts),
		VisitedModels: []string{initialModel},
	}
}

// EvaluateAttempt analyzes an attempt's outcome, updates the circuit breaker, and decides whether fallback is allowed.
func (o *Orchestrator) EvaluateAttempt(
	ctx context.Context,
	exec *ExecutionRecord,
	attempt AttemptOutcome,
	commit CommitState,
) EvaluationResult {
	if exec == nil {
		return EvaluationResult{
			CanFallback: false,
			Succeeded:   false,
			Reason:      "nil_execution_record",
			Err:         fmt.Errorf("execution record is required"),
		}
	}

	classification := o.classifier.Classify(attempt.Error, attempt.StatusCode)
	attempt.TriggerClass = classification.TriggerClass
	attempt.Retryable = classification.Retryable
	if attempt.ErrorMessage == "" && attempt.Error != nil {
		attempt.ErrorMessage = attempt.Error.Error()
	}

	exec.TotalDuration = o.nowFunc().Sub(exec.StartTime)

	// 1. Success case
	if classification.Reason == "success" && attempt.Error == nil {
		attempt.Discarded = false
		exec.Attempts = append(exec.Attempts, attempt)
		exec.SelectedModel = attempt.Model
		exec.FinalStatus = "succeeded"
		o.circuitBreaker.RecordSuccess(attempt.Backend)
		CorrelateExecution(exec)
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    true,
			TriggerClass: TriggerClassNone,
			Reason:       "success",
		}
	}

	// 2. Failure case
	attempt.Discarded = true
	exec.Attempts = append(exec.Attempts, attempt)

	// Only trip the backend circuit breaker on failures attributable to the upstream/infrastructure
	// (e.g. retryable failures, connection drops, timeouts, 5xx), NOT on client errors (4xx) or client cancellation.
	isClientCancellation := (ctx != nil && ctx.Err() != nil) || errors.Is(attempt.Error, context.Canceled)
	isClientError := attempt.StatusCode >= 400 && attempt.StatusCode < 500 && !slices.Contains(o.policy.RetryableStatusCodes, attempt.StatusCode)
	if !isClientCancellation && !isClientError {
		o.circuitBreaker.RecordFailure(attempt.Backend)
	}
	CorrelateExecution(exec)

	// Check context cancellation from client
	if ctx != nil && ctx.Err() != nil {
		exec.FinalStatus = "context_canceled"
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    false,
			TriggerClass: classification.TriggerClass,
			Reason:       "client_canceled",
			Err:          ctx.Err(),
		}
	}

	// Safety Gate A: Is fallback policy enabled?
	if !o.policy.Enabled {
		exec.FinalStatus = "fallback_disabled"
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    false,
			TriggerClass: classification.TriggerClass,
			Reason:       "fallback_disabled",
			Err:          ErrFallbackDisabled,
		}
	}

	// Safety Gate B: Is the outcome retryable?
	if !classification.Retryable {
		exec.FinalStatus = "non_retryable"
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    false,
			TriggerClass: classification.TriggerClass,
			Reason:       classification.Reason,
			Err:          ErrNonRetryableError,
		}
	}

	// Safety Gate C: Has response already been committed to the client?
	if commit.ResponseCommitted {
		exec.FinalStatus = "committed_failure"
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    false,
			TriggerClass: classification.TriggerClass,
			Reason:       "response_already_committed",
			Err:          ErrResponseAlreadyCommitted,
		}
	}

	// Safety Gate D: Did the attempt have non-idempotent tool side-effects?
	if commit.HasNonIdempotentSideEffects {
		exec.FinalStatus = "side_effect_failure"
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    false,
			TriggerClass: classification.TriggerClass,
			Reason:       "non_idempotent_side_effects",
			Err:          ErrNonIdempotentSideEffects,
		}
	}

	// Safety Gate E: Is the request body replayable?
	if !commit.BodyReplayable {
		exec.FinalStatus = "body_not_replayable"
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    false,
			TriggerClass: classification.TriggerClass,
			Reason:       "body_not_replayable",
			Err:          ErrBodyNotReplayable,
		}
	}

	// Safety Gate F: Attempt budget
	if len(exec.Attempts) >= o.policy.MaxAttempts {
		exec.FinalStatus = "max_attempts_exceeded"
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    false,
			TriggerClass: classification.TriggerClass,
			Reason:       "max_attempts_exceeded",
			Err:          ErrMaxAttemptsExceeded,
		}
	}

	// Safety Gate G: Total timeout budget
	if o.policy.TotalTimeout > 0 && exec.TotalDuration >= o.policy.TotalTimeout {
		exec.FinalStatus = "total_deadline_exceeded"
		return EvaluationResult{
			CanFallback:  false,
			Succeeded:    false,
			TriggerClass: classification.TriggerClass,
			Reason:       "total_deadline_exceeded",
			Err:          ErrTotalDeadlineExceeded,
		}
	}
	if ctx != nil {
		if deadline, hasDeadline := ctx.Deadline(); hasDeadline {
			now := time.Now()
			if o.nowFunc != nil {
				now = o.nowFunc()
			}
			if deadline.Sub(now) <= 0 {
				exec.FinalStatus = "total_deadline_exceeded"
				return EvaluationResult{
					CanFallback:  false,
					Succeeded:    false,
					TriggerClass: classification.TriggerClass,
					Reason:       "total_deadline_exceeded",
					Err:          ErrTotalDeadlineExceeded,
				}
			}
		}
	}

	return EvaluationResult{
		CanFallback:  true,
		Succeeded:    false,
		TriggerClass: classification.TriggerClass,
		Reason:       classification.Reason,
	}
}

// SelectNextCandidateIndex chooses the index of the next unvisited candidate strictly from the hard-eligible list.
func (o *Orchestrator) SelectNextCandidateIndex(
	exec *ExecutionRecord,
	numCandidates int,
	modelAt func(i int) string,
	backendResolver func(model string) (string, error),
) (int, error) {
	if exec == nil {
		return -1, fmt.Errorf("execution record is required")
	}

	sawCircuitOpen := false
	for i := 0; i < numCandidates; i++ {
		model := modelAt(i)
		if model == "" {
			continue
		}

		// Never revisit an already attempted model
		if slices.Contains(exec.VisitedModels, model) {
			continue
		}

		// Check circuit breaker for target backend if resolver provided
		if backendResolver != nil {
			backend, err := backendResolver(model)
			if err != nil {
				// Candidate backend cannot be resolved; skip this candidate
				continue
			}
			if backend != "" && !o.circuitBreaker.Allow(backend) {
				sawCircuitOpen = true
				continue
			}
		}

		// Found the next eligible candidate!
		exec.VisitedModels = append(exec.VisitedModels, model)
		return i, nil
	}

	if sawCircuitOpen {
		exec.FinalStatus = "circuit_open"
		return -1, ErrCircuitOpen
	}

	exec.FinalStatus = "candidates_exhausted"
	return -1, ErrNoEligibleCandidates
}

// SelectNextCandidate chooses the next unvisited candidate strictly from the decision's hard-eligible list.
func (o *Orchestrator) SelectNextCandidate(
	exec *ExecutionRecord,
	eligibleRefs []CandidateRef,
	backendResolver func(model string) (string, error),
) (*CandidateRef, error) {
	idx, err := o.SelectNextCandidateIndex(
		exec,
		len(eligibleRefs),
		func(i int) string { return eligibleRefs[i].Model },
		backendResolver,
	)
	if err != nil {
		return nil, err
	}
	return &eligibleRefs[idx], nil
}

// CorrelateExecution ensures request identity continuity, assigns attempt IDs and ordinals,
// and calculates non-duplicated token accounting summaries across all attempts.
func CorrelateExecution(exec *ExecutionRecord) *ExecutionRecord {
	if exec == nil {
		return nil
	}

	var billablePrompt, billableCompletion, billableTotal int
	var discardedPrompt, discardedCompletion, discardedTotal int

	for i := range exec.Attempts {
		attempt := &exec.Attempts[i]
		if attempt.AttemptID == "" {
			attempt.AttemptID = fmt.Sprintf("%s-attempt-%d", exec.RequestID, i+1)
		}
		attempt.Ordinal = i + 1

		total := attempt.TotalTokens
		if total == 0 && (attempt.PromptTokens > 0 || attempt.CompletionTokens > 0) {
			total = attempt.PromptTokens + attempt.CompletionTokens
		}

		if attempt.Discarded {
			discardedPrompt += attempt.PromptTokens
			discardedCompletion += attempt.CompletionTokens
			discardedTotal += total
		} else {
			billablePrompt += attempt.PromptTokens
			billableCompletion += attempt.CompletionTokens
			billableTotal += total
		}
	}

	exec.UsageSummary = TokenUsageSummary{
		BillablePromptTokens:      billablePrompt,
		BillableCompletionTokens:  billableCompletion,
		BillableTotalTokens:       billableTotal,
		DiscardedPromptTokens:     discardedPrompt,
		DiscardedCompletionTokens: discardedCompletion,
		DiscardedTotalTokens:      discardedTotal,
	}

	return exec
}

// RequestContext returns a parent context bounded by TotalTimeout if configured.
func (o *Orchestrator) RequestContext(parent context.Context) (context.Context, context.CancelFunc) {
	if parent == nil {
		parent = context.Background()
	}
	if o.policy.TotalTimeout > 0 {
		return context.WithTimeout(parent, o.policy.TotalTimeout)
	}
	return parent, func() {}
}

// AttemptContext derives a child context for an attempt bounded by the minimum of
// the remaining total budget (from the request context deadline) and PerAttemptTimeout.
func (o *Orchestrator) AttemptContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}

	timeout := o.policy.PerAttemptTimeout

	now := time.Now()
	if o.nowFunc != nil {
		now = o.nowFunc()
	}

	if deadline, hasDeadline := ctx.Deadline(); hasDeadline {
		remaining := deadline.Sub(now)
		if remaining <= 0 {
			// Total budget has already expired; return immediately expired context.
			return context.WithDeadline(ctx, deadline)
		}
		if timeout <= 0 || remaining < timeout {
			timeout = remaining
		}
	}

	if timeout > 0 {
		return context.WithTimeout(ctx, timeout)
	}
	return ctx, func() {}
}
