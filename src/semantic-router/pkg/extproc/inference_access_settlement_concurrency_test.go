package extproc

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

type settlementContextValueKey struct{}

type settlementProbeAccess struct {
	*fakeInferenceAccess

	mu                 sync.Mutex
	evidenceReads      int
	settlementCalls    int
	firstSettleStarted chan struct{}
	releaseFirstSettle chan struct{}
	firstSettleOnce    sync.Once
	failFirstSettle    bool
	contextValue       string
	evidenceError      error
}

func (f *settlementProbeAccess) ReadAttemptEvidence(
	ctx context.Context,
	request accessruntime.AttemptEvidenceRequest,
) (accessruntime.AttemptEvidenceSnapshot, error) {
	if err := validateDetachedSettlementContext(ctx, f.contextValue); err != nil {
		return accessruntime.AttemptEvidenceSnapshot{}, err
	}
	f.mu.Lock()
	f.evidenceReads++
	readErr := f.evidenceError
	f.mu.Unlock()
	if readErr != nil {
		return accessruntime.AttemptEvidenceSnapshot{}, readErr
	}
	return responseStartedEvidence(request)
}

func (f *settlementProbeAccess) Settle(
	ctx context.Context,
	_ accessruntime.SettlementRequest,
) (quotaruntime.FinalizationResult, error) {
	if err := validateDetachedSettlementContext(ctx, f.contextValue); err != nil {
		return quotaruntime.FinalizationResult{}, err
	}
	f.mu.Lock()
	f.settlementCalls++
	call := f.settlementCalls
	f.mu.Unlock()
	if f.failFirstSettle && call == 1 {
		f.firstSettleOnce.Do(func() { close(f.firstSettleStarted) })
		select {
		case <-f.releaseFirstSettle:
			return quotaruntime.FinalizationResult{}, errors.New("private-settlement-canary")
		case <-ctx.Done():
			return quotaruntime.FinalizationResult{}, ctx.Err()
		}
	}
	return quotaruntime.FinalizationResult{}, nil
}

func (f *settlementProbeAccess) counts() (int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.evidenceReads, f.settlementCalls
}

func validateDetachedSettlementContext(ctx context.Context, wantValue string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("settlement inherited request cancellation: %w", err)
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		return errors.New("settlement context has no deadline")
	}
	remaining := time.Until(deadline)
	if remaining <= 0 || remaining > inferenceSettlementTimeout {
		return fmt.Errorf("settlement deadline is outside its bound: %s", remaining)
	}
	if got, _ := ctx.Value(settlementContextValueKey{}).(string); got != wantValue {
		return fmt.Errorf("settlement context value = %q, want %q", got, wantValue)
	}
	return nil
}

func waitForSettlementWaiters(t *testing.T, state *inferenceRequestAccess, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		state.mu.Lock()
		run := state.settlementRun
		waiters := 0
		if run != nil {
			waiters = run.waiters
		}
		state.mu.Unlock()
		if run != nil && waiters == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("settlement waiters = %d, want %d", waiters, want)
		}
		runtime.Gosched()
	}
}

func TestSettlementSurvivesCanceledRequestWithBoundedDetachedContext(t *testing.T) {
	const contextValue = "request-trace-value"
	access := &settlementProbeAccess{
		fakeInferenceAccess: &fakeInferenceAccess{},
		contextValue:        contextValue,
	}
	router := &OpenAIRouter{Config: inferenceTestConfig(t), InferenceAccess: access}
	request := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(request.TraceContext, request, "internal-model"); err != nil {
		t.Fatal(err)
	}
	applyPrimaryDispatchOutcome(t, request, backendinvoker.AttemptResponseStarted)

	traceContext := context.WithValue(request.TraceContext, settlementContextValueKey{}, contextValue)
	traceContext, cancel := context.WithCancel(traceContext)
	cancel()
	request.TraceContext = traceContext
	usage := responseUsageMetrics{
		promptTokens: 1, promptTokensReported: true,
		completionTokens: 1, completionTokensReported: true,
	}
	if err := router.completeAndSettlePrimaryInference(request, usage, 200); err != nil {
		t.Fatalf("settlement after request cancellation failed: %v", err)
	}
	if evidenceReads, settlementCalls := access.counts(); evidenceReads != 1 || settlementCalls != 1 {
		t.Fatalf("evidence reads/settlements = %d/%d, want 1/1", evidenceReads, settlementCalls)
	}
}

func TestSettlementRedactsAttemptEvidenceFailure(t *testing.T) {
	access := &settlementProbeAccess{
		fakeInferenceAccess: &fakeInferenceAccess{},
		evidenceError:       errors.New("private-evidence-canary"),
	}
	router := &OpenAIRouter{Config: inferenceTestConfig(t), InferenceAccess: access}
	request := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(request.TraceContext, request, "internal-model"); err != nil {
		t.Fatal(err)
	}
	applyPrimaryDispatchOutcome(t, request, backendinvoker.AttemptResponseStarted)
	usage := responseUsageMetrics{
		promptTokens: 1, promptTokensReported: true,
		completionTokens: 1, completionTokensReported: true,
	}
	err := router.completeAndSettlePrimaryInference(request, usage, 200)
	if !errors.Is(err, errInferenceSettlementUnavailable) {
		t.Fatalf("attempt-evidence error = %v, want unavailable result", err)
	}
	if strings.Contains(err.Error(), "private-evidence-canary") {
		t.Fatalf("attempt-evidence failure leaked private runtime error: %v", err)
	}
	if evidenceReads, settlementCalls := access.counts(); evidenceReads != 1 || settlementCalls != 0 {
		t.Fatalf("evidence reads/settlements = %d/%d, want 1/0", evidenceReads, settlementCalls)
	}
}

func TestConcurrentSettlementSharesFirstFailureAndAllowsRetry(t *testing.T) {
	access := &settlementProbeAccess{
		fakeInferenceAccess: &fakeInferenceAccess{},
		firstSettleStarted:  make(chan struct{}),
		releaseFirstSettle:  make(chan struct{}),
		failFirstSettle:     true,
	}
	router := &OpenAIRouter{Config: inferenceTestConfig(t), InferenceAccess: access}
	request := admittedInferenceTestContext("internal-model")
	if err := router.beginPrimaryInferenceDispatch(request.TraceContext, request, "internal-model"); err != nil {
		t.Fatal(err)
	}
	applyPrimaryDispatchOutcome(t, request, backendinvoker.AttemptResponseStarted)
	usage := responseUsageMetrics{
		promptTokens: 1, promptTokensReported: true,
		completionTokens: 1, completionTokensReported: true,
	}

	const callers = 8
	results := make(chan error, callers)
	go func() { results <- router.completeAndSettlePrimaryInference(request, usage, 200) }()
	select {
	case <-access.firstSettleStarted:
	case <-time.After(time.Second):
		t.Fatal("first settlement did not start")
	}
	var ready sync.WaitGroup
	ready.Add(callers - 1)
	for index := 1; index < callers; index++ {
		go func() {
			ready.Done()
			results <- router.completeAndSettlePrimaryInference(request, usage, 200)
		}()
	}
	ready.Wait()
	waitForSettlementWaiters(t, request.InferenceAccess, callers-1)
	close(access.releaseFirstSettle)

	for index := 0; index < callers; index++ {
		err := <-results
		if !errors.Is(err, errInferenceSettlementUnavailable) {
			t.Fatalf("concurrent settlement %d error = %v, want shared unavailable result", index, err)
		}
		if strings.Contains(err.Error(), "private-settlement-canary") {
			t.Fatalf("concurrent settlement %d leaked private runtime error: %v", index, err)
		}
	}
	if evidenceReads, settlementCalls := access.counts(); evidenceReads != 1 || settlementCalls != 1 {
		t.Fatalf("first-flight evidence reads/settlements = %d/%d, want 1/1", evidenceReads, settlementCalls)
	}

	if err := router.completeAndSettlePrimaryInference(request, usage, 200); err != nil {
		t.Fatalf("retry after shared failure = %v", err)
	}
	if evidenceReads, settlementCalls := access.counts(); evidenceReads != 1 || settlementCalls != 2 {
		t.Fatalf("retry evidence reads/settlements = %d/%d, want 1/2", evidenceReads, settlementCalls)
	}
}
