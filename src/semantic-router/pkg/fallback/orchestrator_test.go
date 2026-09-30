package fallback

import (
	"context"
	"errors"
	"fmt"
	"syscall"
	"testing"
	"time"
)

func TestOrchestrator(t *testing.T) {
	policy := DefaultEnabledPolicy()
	policy.MaxAttempts = 3
	policy.TotalTimeout = 5 * time.Second

	cb := NewBackendCircuitBreaker(policy.CircuitBreaker)
	orch := NewOrchestrator(policy, cb)

	simulatedTime := time.Date(2026, 9, 16, 12, 0, 0, 0, time.UTC)
	orch.nowFunc = func() time.Time { return simulatedTime }
	cb.nowFunc = func() time.Time { return simulatedTime }

	eligibleModels := []string{
		"model-primary",
		"model-fallback-1",
		"model-fallback-2",
	}

	safeCommit := CommitState{
		ResponseCommitted:           false,
		HasNonIdempotentSideEffects: false,
		BodyReplayable:              true,
	}

	if orch.Policy().MaxAttempts != 3 || orch.Policy().PerAttemptTimeout != 10*time.Second {
		t.Fatalf("unexpected orchestrator policy: %+v", orch.Policy())
	}

	t.Run("immediate success on primary model", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-1", "general-route", "model-primary")
		attempt := AttemptOutcome{
			AttemptID:        "req-1-attempt-1",
			Ordinal:          1,
			Model:            "model-primary",
			Backend:          "backend-primary",
			StatusCode:       200,
			Duration:         150 * time.Millisecond,
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		}

		res := orch.EvaluateAttempt(context.Background(), exec, attempt, safeCommit)
		if !res.Succeeded {
			t.Errorf("expected attempt to succeed")
		}
		if res.CanFallback {
			t.Errorf("successful attempt must not fallback")
		}
		if exec.SelectedModel != "model-primary" {
			t.Errorf("expected selected model model-primary, got %s", exec.SelectedModel)
		}
		if exec.FinalStatus != "succeeded" {
			t.Errorf("expected status succeeded, got %s", exec.FinalStatus)
		}
		if exec.UsageSummary.BillableTotalTokens != 150 || exec.UsageSummary.DiscardedTotalTokens != 0 {
			t.Errorf("unexpected usage summary: %+v", exec.UsageSummary)
		}
	})

	t.Run("successful fallback on attempt 2 after 503 on primary", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-2", "general-route", "model-primary")

		// Attempt 1: 503 Service Unavailable
		att1 := AttemptOutcome{
			AttemptID:        "req-2-attempt-1",
			Ordinal:          1,
			Model:            "model-primary",
			Backend:          "backend-primary",
			StatusCode:       503,
			Duration:         200 * time.Millisecond,
			PromptTokens:     80,
			CompletionTokens: 0,
			TotalTokens:      80,
		}

		eval1 := orch.EvaluateAttempt(context.Background(), exec, att1, safeCommit)
		if eval1.Succeeded {
			t.Errorf("attempt 1 should not succeed")
		}
		if !eval1.CanFallback {
			t.Errorf("attempt 1 with 503 should permit fallback")
		}
		if eval1.TriggerClass != TriggerClass5xx {
			t.Errorf("expected TriggerClass5xx, got %s", eval1.TriggerClass)
		}

		// Select next candidate
		candIdx, err := orch.SelectNextCandidateIndex(exec, len(eligibleModels), func(i int) string { return eligibleModels[i] }, nil)
		if err != nil {
			t.Fatalf("expected next candidate, got error: %v", err)
		}
		if eligibleModels[candIdx] != "model-fallback-1" {
			t.Errorf("expected model-fallback-1, got %s", eligibleModels[candIdx])
		}

		// Attempt 2: 200 OK
		att2 := AttemptOutcome{
			AttemptID:        "req-2-attempt-2",
			Ordinal:          2,
			Model:            "model-fallback-1",
			Backend:          "backend-fallback-1",
			StatusCode:       200,
			Duration:         300 * time.Millisecond,
			PromptTokens:     80,
			CompletionTokens: 60,
			TotalTokens:      140,
		}

		eval2 := orch.EvaluateAttempt(context.Background(), exec, att2, safeCommit)
		if !eval2.Succeeded {
			t.Errorf("attempt 2 should succeed")
		}
		if eval2.CanFallback {
			t.Errorf("attempt 2 should not fallback after success")
		}
		if exec.SelectedModel != "model-fallback-1" {
			t.Errorf("expected selected model model-fallback-1, got %s", exec.SelectedModel)
		}
		if exec.FinalStatus != "succeeded" {
			t.Errorf("expected status succeeded, got %s", exec.FinalStatus)
		}

		// Verify non-duplicated accounting
		if exec.UsageSummary.BillableTotalTokens != 140 {
			t.Errorf("expected billable tokens 140, got %d", exec.UsageSummary.BillableTotalTokens)
		}
		if exec.UsageSummary.DiscardedTotalTokens != 80 {
			t.Errorf("expected discarded tokens 80, got %d", exec.UsageSummary.DiscardedTotalTokens)
		}
	})

	t.Run("non-retryable client error halts fallback", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-3", "general-route", "model-primary")
		att := AttemptOutcome{
			AttemptID:  "req-3-attempt-1",
			Ordinal:    1,
			Model:      "model-primary",
			Backend:    "backend-primary",
			StatusCode: 400,
			Duration:   50 * time.Millisecond,
		}

		eval := orch.EvaluateAttempt(context.Background(), exec, att, safeCommit)
		if eval.CanFallback {
			t.Errorf("400 Bad Request must not permit fallback")
		}
		if !errors.Is(eval.Err, ErrNonRetryableError) {
			t.Errorf("expected ErrNonRetryableError, got %v", eval.Err)
		}
		if exec.FinalStatus != "non_retryable" {
			t.Errorf("expected status non_retryable, got %s", exec.FinalStatus)
		}
	})

	t.Run("reject fallback after response committed (streaming safety)", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-4", "streaming-route", "model-primary")
		committedState := CommitState{
			ResponseCommitted:           true,
			HasNonIdempotentSideEffects: false,
			BodyReplayable:              true,
		}
		att := AttemptOutcome{
			AttemptID:  "req-4-attempt-1",
			Ordinal:    1,
			Model:      "model-primary",
			Backend:    "backend-primary",
			StatusCode: 503,
		}

		eval := orch.EvaluateAttempt(context.Background(), exec, att, committedState)
		if eval.CanFallback {
			t.Errorf("fallback must be rejected when response is committed")
		}
		if !errors.Is(eval.Err, ErrResponseAlreadyCommitted) {
			t.Errorf("expected ErrResponseAlreadyCommitted, got %v", eval.Err)
		}
		if exec.FinalStatus != "committed_failure" {
			t.Errorf("expected status committed_failure, got %s", exec.FinalStatus)
		}
	})

	t.Run("reject fallback when non-idempotent tool side-effects occurred", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-5", "agent-route", "model-primary")
		sideEffectsState := CommitState{
			ResponseCommitted:           false,
			HasNonIdempotentSideEffects: true,
			BodyReplayable:              true,
		}
		att := AttemptOutcome{
			AttemptID:  "req-5-attempt-1",
			Ordinal:    1,
			Model:      "model-primary",
			Backend:    "backend-primary",
			StatusCode: 502,
		}

		eval := orch.EvaluateAttempt(context.Background(), exec, att, sideEffectsState)
		if eval.CanFallback {
			t.Errorf("fallback must be rejected when side-effects occurred")
		}
		if !errors.Is(eval.Err, ErrNonIdempotentSideEffects) {
			t.Errorf("expected ErrNonIdempotentSideEffects, got %v", eval.Err)
		}
		if exec.FinalStatus != "side_effect_failure" {
			t.Errorf("expected status side_effect_failure, got %s", exec.FinalStatus)
		}
	})

	t.Run("reject fallback when request body is not replayable", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-6", "general-route", "model-primary")
		nonReplayableState := CommitState{
			ResponseCommitted:           false,
			HasNonIdempotentSideEffects: false,
			BodyReplayable:              false,
		}
		att := AttemptOutcome{
			AttemptID:  "req-6-attempt-1",
			Ordinal:    1,
			Model:      "model-primary",
			Backend:    "backend-primary",
			StatusCode: 502,
		}

		eval := orch.EvaluateAttempt(context.Background(), exec, att, nonReplayableState)
		if eval.CanFallback {
			t.Errorf("fallback must be rejected when body is not replayable")
		}
		if !errors.Is(eval.Err, ErrBodyNotReplayable) {
			t.Errorf("expected ErrBodyNotReplayable, got %v", eval.Err)
		}
	})

	t.Run("budget limit: max attempts exceeded", func(t *testing.T) {
		tightPolicy := policy
		tightPolicy.MaxAttempts = 2
		tightOrch := NewOrchestrator(tightPolicy, nil)
		tightOrch.nowFunc = func() time.Time { return simulatedTime }

		exec := tightOrch.NewExecutionRecord("req-7", "general-route", "model-primary")

		att1 := AttemptOutcome{
			AttemptID:  "req-7-attempt-1",
			Ordinal:    1,
			Model:      "model-primary",
			StatusCode: 502,
		}
		eval1 := tightOrch.EvaluateAttempt(context.Background(), exec, att1, safeCommit)
		if !eval1.CanFallback {
			t.Fatalf("attempt 1 should permit fallback")
		}

		_, err := tightOrch.SelectNextCandidateIndex(exec, len(eligibleModels), func(i int) string { return eligibleModels[i] }, nil)
		if err != nil {
			t.Fatalf("unexpected select error: %v", err)
		}

		att2 := AttemptOutcome{
			AttemptID:  "req-7-attempt-2",
			Ordinal:    2,
			Model:      "model-fallback-1",
			StatusCode: 502,
		}
		eval2 := tightOrch.EvaluateAttempt(context.Background(), exec, att2, safeCommit)
		if eval2.CanFallback {
			t.Errorf("attempt 2 should exceed max attempts 2 and disallow fallback")
		}
		if !errors.Is(eval2.Err, ErrMaxAttemptsExceeded) {
			t.Errorf("expected ErrMaxAttemptsExceeded, got %v", eval2.Err)
		}
		if exec.FinalStatus != "max_attempts_exceeded" {
			t.Errorf("expected status max_attempts_exceeded, got %s", exec.FinalStatus)
		}
	})

	t.Run("budget limit: total timeout deadline exceeded", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-8", "general-route", "model-primary")
		att1 := AttemptOutcome{
			AttemptID:  "req-8-attempt-1",
			Ordinal:    1,
			Model:      "model-primary",
			StatusCode: 504,
		}

		// Simulate time jump exceeding TotalTimeout (5s)
		simulatedTime = simulatedTime.Add(6 * time.Second)

		eval := orch.EvaluateAttempt(context.Background(), exec, att1, safeCommit)
		if eval.CanFallback {
			t.Errorf("total timeout exceeded must disallow fallback")
		}
		if !errors.Is(eval.Err, ErrTotalDeadlineExceeded) {
			t.Errorf("expected ErrTotalDeadlineExceeded, got %v", eval.Err)
		}
		if exec.FinalStatus != "total_deadline_exceeded" {
			t.Errorf("expected status total_deadline_exceeded, got %s", exec.FinalStatus)
		}
	})

	t.Run("context cancellation halts fallback", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-9", "general-route", "model-primary")
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		att := AttemptOutcome{
			AttemptID:  "req-9-attempt-1",
			Ordinal:    1,
			Model:      "model-primary",
			StatusCode: 503,
		}

		eval := orch.EvaluateAttempt(ctx, exec, att, safeCommit)
		if eval.CanFallback {
			t.Errorf("canceled context must disallow fallback")
		}
		if !errors.Is(eval.Err, context.Canceled) {
			t.Errorf("expected context.Canceled error, got %v", eval.Err)
		}
	})

	t.Run("candidate exhaustion", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-10", "general-route", "model-primary")
		exec.VisitedModels = []string{"model-primary", "model-fallback-1", "model-fallback-2"}

		candIdx, err := orch.SelectNextCandidateIndex(exec, len(eligibleModels), func(i int) string { return eligibleModels[i] }, nil)
		if candIdx != -1 {
			t.Errorf("expected no candidate index, got %d", candIdx)
		}
		if !errors.Is(err, ErrNoEligibleCandidates) {
			t.Errorf("expected ErrNoEligibleCandidates, got %v", err)
		}
		if exec.FinalStatus != "candidates_exhausted" {
			t.Errorf("expected status candidates_exhausted, got %s", exec.FinalStatus)
		}
	})

	t.Run("circuit breaker integration: skip open backend candidate to next", func(t *testing.T) {
		cb := NewBackendCircuitBreaker(policy.CircuitBreaker)
		orchWithCB := NewOrchestrator(policy, cb)
		orchWithCB.nowFunc = func() time.Time { return simulatedTime }
		cb.nowFunc = func() time.Time { return simulatedTime }

		// Trip circuit breaker on backend-fallback-1
		cb.RecordFailure("backend-fallback-1")
		cb.RecordFailure("backend-fallback-1")
		cb.RecordFailure("backend-fallback-1")
		if cb.GetState("backend-fallback-1") != StateOpen {
			t.Fatalf("expected backend-fallback-1 to be open")
		}

		resolver := func(model string) (string, error) {
			switch model {
			case "model-primary":
				return "backend-primary", nil
			case "model-fallback-1":
				return "backend-fallback-1", nil
			case "model-fallback-2":
				return "backend-fallback-2", nil
			default:
				return "", nil
			}
		}

		exec := orchWithCB.NewExecutionRecord("req-11", "general-route", "model-primary")

		// Select candidate should skip model-fallback-1 (open circuit) and pick model-fallback-2!
		candIdx, err := orchWithCB.SelectNextCandidateIndex(exec, len(eligibleModels), func(i int) string { return eligibleModels[i] }, resolver)
		if err != nil {
			t.Fatalf("unexpected select error: %v", err)
		}
		if eligibleModels[candIdx] != "model-fallback-2" {
			t.Errorf("expected candidate model-fallback-2 (skipping open model-fallback-1), got %s", eligibleModels[candIdx])
		}
	})

	t.Run("connection failure triggers connection trigger class", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-12", "general-route", "model-primary")
		att := AttemptOutcome{
			AttemptID: "req-12-attempt-1",
			Ordinal:   1,
			Model:     "model-primary",
			Backend:   "backend-primary",
			Error:     syscall.ECONNREFUSED,
		}

		eval := orch.EvaluateAttempt(context.Background(), exec, att, safeCommit)
		if !eval.CanFallback {
			t.Errorf("connection refused must be retryable")
		}
		if eval.TriggerClass != TriggerClassConnection {
			t.Errorf("expected TriggerClassConnection, got %s", eval.TriggerClass)
		}
	})

	t.Run("client 4xx errors do not trip backend circuit breaker", func(t *testing.T) {
		cbFresh := NewBackendCircuitBreaker(policy.CircuitBreaker)
		orchFresh := NewOrchestrator(policy, cbFresh)
		testBackend := "robust-backend"

		for i := 0; i < 5; i++ {
			exec := orchFresh.NewExecutionRecord(fmt.Sprintf("req-4xx-%d", i), "route", "model-primary")
			att := AttemptOutcome{
				AttemptID:  fmt.Sprintf("req-4xx-%d-1", i),
				Ordinal:    1,
				Model:      "model-primary",
				Backend:    testBackend,
				StatusCode: 400,
			}
			eval := orchFresh.EvaluateAttempt(context.Background(), exec, att, safeCommit)
			if eval.CanFallback {
				t.Fatalf("400 must not allow fallback")
			}
		}

		if cbFresh.GetState(testBackend) != StateClosed {
			t.Errorf("expected circuit breaker to remain StateClosed after client 400s, got %s", cbFresh.GetState(testBackend))
		}
		if !cbFresh.Allow(testBackend) {
			t.Errorf("expected backend to still be allowed after client 400s")
		}
	})

	t.Run("client context cancellation does not trip backend circuit breaker", func(t *testing.T) {
		cbFresh := NewBackendCircuitBreaker(policy.CircuitBreaker)
		orchFresh := NewOrchestrator(policy, cbFresh)
		testBackend := "cancel-backend"

		for i := 0; i < 5; i++ {
			exec := orchFresh.NewExecutionRecord(fmt.Sprintf("req-cancel-%d", i), "route", "model-primary")
			att := AttemptOutcome{
				AttemptID: fmt.Sprintf("req-cancel-%d-1", i),
				Ordinal:   1,
				Model:     "model-primary",
				Backend:   testBackend,
				Error:     context.Canceled,
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			eval := orchFresh.EvaluateAttempt(ctx, exec, att, safeCommit)
			if eval.CanFallback {
				t.Fatalf("canceled context must not allow fallback")
			}
		}

		if cbFresh.GetState(testBackend) != StateClosed {
			t.Errorf("expected circuit breaker to remain StateClosed after client cancellations, got %s", cbFresh.GetState(testBackend))
		}
		if !cbFresh.Allow(testBackend) {
			t.Errorf("expected backend to still be allowed after client cancellations")
		}
	})

	t.Run("candidate is skipped when backendResolver returns error", func(t *testing.T) {
		exec := orch.NewExecutionRecord("req-skip-err", "general-route", "model-primary")
		errResolver := func(model string) (string, error) {
			if model == "model-fallback-1" {
				return "", errors.New("backend not found for model")
			}
			return "backend-fallback-2", nil
		}

		candIdx, err := orch.SelectNextCandidateIndex(exec, len(eligibleModels), func(i int) string { return eligibleModels[i] }, errResolver)
		if err != nil {
			t.Fatalf("unexpected select error: %v", err)
		}
		if eligibleModels[candIdx] != "model-fallback-2" {
			t.Errorf("expected model-fallback-2 (skipping model-fallback-1 on resolver error), got %s", eligibleModels[candIdx])
		}
	})

	t.Run("AttemptContext respects PerAttemptTimeout", func(t *testing.T) {
		ctx, cancel := orch.AttemptContext(context.Background())
		defer cancel()

		deadline, ok := ctx.Deadline()
		if !ok {
			t.Fatalf("expected context to have a deadline")
		}
		expectedDeadline := time.Now().Add(orch.Policy().PerAttemptTimeout)
		diff := deadline.Sub(expectedDeadline)
		if diff < -time.Second || diff > time.Second {
			t.Errorf("expected deadline close to %v, got %v", expectedDeadline, deadline)
		}
	})

	t.Run("AttemptContext clamps to remaining total deadline when tighter than PerAttemptTimeout", func(t *testing.T) {
		// Remaining total budget is 2s, but PerAttemptTimeout is 10s -> must clamp to 2s
		remainingTotal := 2 * time.Second
		reqCtx, reqCancel := context.WithTimeout(context.Background(), remainingTotal)
		defer reqCancel()

		attemptCtx, attemptCancel := orch.AttemptContext(reqCtx)
		defer attemptCancel()

		attemptDeadline, ok := attemptCtx.Deadline()
		if !ok {
			t.Fatalf("expected attempt context to have a deadline")
		}
		reqDeadline, _ := reqCtx.Deadline()
		diff := attemptDeadline.Sub(reqDeadline)
		if diff < -100*time.Millisecond || diff > 100*time.Millisecond {
			t.Errorf("expected attempt deadline to match request remaining deadline %v, got %v", reqDeadline, attemptDeadline)
		}
	})

	t.Run("AttemptContext respects PerAttemptTimeout when remaining total budget is larger", func(t *testing.T) {
		// Remaining total budget is 25s, and PerAttemptTimeout is 10s -> must use 10s
		reqCtx, reqCancel := context.WithTimeout(context.Background(), 25*time.Second)
		defer reqCancel()

		attemptCtx, attemptCancel := orch.AttemptContext(reqCtx)
		defer attemptCancel()

		attemptDeadline, ok := attemptCtx.Deadline()
		if !ok {
			t.Fatalf("expected attempt context to have a deadline")
		}
		expected := time.Now().Add(orch.Policy().PerAttemptTimeout)
		diff := attemptDeadline.Sub(expected)
		if diff < -time.Second || diff > time.Second {
			t.Errorf("expected attempt deadline close to %v, got %v", expected, attemptDeadline)
		}
	})

	t.Run("AttemptContext returns expired context when remaining total budget is exhausted", func(t *testing.T) {
		reqCtx, reqCancel := context.WithDeadline(context.Background(), time.Now().Add(-50*time.Millisecond))
		defer reqCancel()

		attemptCtx, attemptCancel := orch.AttemptContext(reqCtx)
		defer attemptCancel()

		if attemptCtx.Err() == nil {
			t.Fatalf("expected attempt context to be expired immediately")
		}
	})

	t.Run("nil execution record returns error", func(t *testing.T) {
		att := AttemptOutcome{Model: "model-primary", StatusCode: 200}
		eval := orch.EvaluateAttempt(context.Background(), nil, att, safeCommit)
		if eval.CanFallback || eval.Succeeded || eval.Err == nil {
			t.Errorf("expected nil exec to fail with error")
		}
	})
}

func TestCorrelateExecution(t *testing.T) {
	exec := &ExecutionRecord{
		RequestID:    "req-corr-1",
		DecisionName: "chat-decision",
		InitialModel: "model-a",
		Attempts: []AttemptOutcome{
			{
				Model:            "model-a",
				Backend:          "backend-a",
				StatusCode:       502,
				Discarded:        true,
				PromptTokens:     100,
				CompletionTokens: 0,
				TotalTokens:      100,
				Duration:         120 * time.Millisecond,
			},
			{
				Model:            "model-b",
				Backend:          "backend-b",
				StatusCode:       503,
				Discarded:        true,
				PromptTokens:     100,
				CompletionTokens: 10,
				TotalTokens:      110,
				Duration:         250 * time.Millisecond,
			},
			{
				Model:            "model-c",
				Backend:          "backend-c",
				StatusCode:       200,
				Discarded:        false,
				PromptTokens:     100,
				CompletionTokens: 80,
				TotalTokens:      0, // Test defensive sum when provider omits total_tokens
				Duration:         400 * time.Millisecond,
			},
		},
	}

	correlated := CorrelateExecution(exec)
	if correlated == nil {
		t.Fatalf("expected correlated record, got nil")
	}

	// Verify attempt identifiers generated with common request prefix
	if correlated.Attempts[0].AttemptID != "req-corr-1-attempt-1" {
		t.Errorf("unexpected attempt ID 1: %s", correlated.Attempts[0].AttemptID)
	}
	if correlated.Attempts[1].AttemptID != "req-corr-1-attempt-2" {
		t.Errorf("unexpected attempt ID 2: %s", correlated.Attempts[1].AttemptID)
	}
	if correlated.Attempts[2].AttemptID != "req-corr-1-attempt-3" {
		t.Errorf("unexpected attempt ID 3: %s", correlated.Attempts[2].AttemptID)
	}

	// Verify ordinals
	if correlated.Attempts[0].Ordinal != 1 || correlated.Attempts[1].Ordinal != 2 || correlated.Attempts[2].Ordinal != 3 {
		t.Errorf("ordinals not assigned properly")
	}

	// Verify non-duplicated accounting: only Attempt 3 is billable (and 100+80=180 computed dynamically)
	if correlated.UsageSummary.BillablePromptTokens != 100 {
		t.Errorf("expected billable prompt 100, got %d", correlated.UsageSummary.BillablePromptTokens)
	}
	if correlated.UsageSummary.BillableCompletionTokens != 80 {
		t.Errorf("expected billable completion 80, got %d", correlated.UsageSummary.BillableCompletionTokens)
	}
	if correlated.UsageSummary.BillableTotalTokens != 180 {
		t.Errorf("expected billable total 180 (defensively summed), got %d", correlated.UsageSummary.BillableTotalTokens)
	}

	// Attempts 1 and 2 discarded tokens
	if correlated.UsageSummary.DiscardedPromptTokens != 200 {
		t.Errorf("expected discarded prompt 200, got %d", correlated.UsageSummary.DiscardedPromptTokens)
	}
	if correlated.UsageSummary.DiscardedCompletionTokens != 10 {
		t.Errorf("expected discarded completion 10, got %d", correlated.UsageSummary.DiscardedCompletionTokens)
	}
	if correlated.UsageSummary.DiscardedTotalTokens != 210 {
		t.Errorf("expected discarded total 210, got %d", correlated.UsageSummary.DiscardedTotalTokens)
	}
}

func TestOrchestrator_EvaluateAttempt_2xxWithErrorIsFailedAttempt(t *testing.T) {
	policy := DefaultEnabledPolicy()
	cb := NewBackendCircuitBreaker(policy.CircuitBreaker)
	orch := NewOrchestrator(policy, cb)

	exec := orch.NewExecutionRecord("req-1", "general-route", "model-a")
	attempt := AttemptOutcome{
		Model:      "model-a",
		Backend:    "backend-a",
		StatusCode: 200,
		Error:      errors.New("connection reset during body streaming"),
	}

	result := orch.EvaluateAttempt(context.Background(), exec, attempt, CommitState{})
	if result.Succeeded {
		t.Fatalf("expected attempt with non-nil error to NOT succeed even if status is 200")
	}
	if exec.FinalStatus == "succeeded" {
		t.Fatalf("exec.FinalStatus must not be succeeded when attempt has an error")
	}
	if len(exec.Attempts) != 1 || !exec.Attempts[0].Discarded {
		t.Errorf("failed attempt must be marked Discarded")
	}
}
