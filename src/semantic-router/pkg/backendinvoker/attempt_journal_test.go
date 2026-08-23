package backendinvoker

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

type attemptEvidenceEngineStub struct {
	dispatch *quotaruntime.BeginDispatchRequest
	attempt  *quotaruntime.BeginAttemptRequest
	finish   *quotaruntime.FinishAttemptRequest
}

func (s *attemptEvidenceEngineStub) BeginDispatch(
	_ context.Context,
	request quotaruntime.BeginDispatchRequest,
) (quotaruntime.BeginDispatchResult, error) {
	s.dispatch = &request
	return quotaruntime.BeginDispatchResult{}, nil
}

func (s *attemptEvidenceEngineStub) BeginAttempt(
	_ context.Context,
	request quotaruntime.BeginAttemptRequest,
) (quotaruntime.BeginAttemptResult, error) {
	s.attempt = &request
	return quotaruntime.BeginAttemptResult{}, nil
}

func (s *attemptEvidenceEngineStub) FinishAttempt(
	_ context.Context,
	request quotaruntime.FinishAttemptRequest,
) (quotaruntime.FinishAttemptResult, error) {
	s.finish = &request
	return quotaruntime.FinishAttemptResult{}, nil
}

func (s *attemptEvidenceEngineStub) ReadAttemptEvidence(
	context.Context,
	quotaruntime.ReadAttemptEvidenceRequest,
) (quotaruntime.ReadAttemptEvidenceResult, error) {
	return quotaruntime.ReadAttemptEvidenceResult{}, nil
}

func TestAuthoritativeAttemptJournalMapsImmutablePlanAndEvidence(t *testing.T) {
	engine := &attemptEvidenceEngineStub{}
	journal, err := NewAuthoritativeAttemptJournal(engine)
	if err != nil {
		t.Fatal(err)
	}
	plan := completeTestPlan(Plan{
		QuotaPartition: "partition-1", AdmissionID: "admission-1",
		AdmissionDigest: strings.Repeat("b", 64), DispatchID: "dispatch-1",
		DispatchType: "primary", Ordinal: 2, DispatchPlanDigest: strings.Repeat("a", 64),
		ModelID: "model-1", ModelRevision: 3, RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{}`)),
		Execution: Execution{MaxRetries: 2},
		Backends:  []Backend{{ID: "backend-1", ProviderID: "provider-1"}},
	})
	deadline := time.Date(2026, 8, 22, 1, 2, 3, 123456789, time.UTC)
	if err := journal.BeginDispatch(context.Background(), plan, deadline); err != nil {
		t.Fatalf("BeginDispatch() error = %v", err)
	}
	if engine.dispatch == nil || engine.dispatch.Partition != plan.QuotaPartition ||
		engine.dispatch.AdmissionDigest != plan.AdmissionDigest ||
		engine.dispatch.DispatchPlanDigest != plan.DispatchPlanDigest ||
		engine.dispatch.Ordinal != 2 || engine.dispatch.MaxAttempts != 3 ||
		engine.dispatch.Deadline.Nanosecond()%int(time.Millisecond) != 0 {
		t.Fatalf("dispatch request = %+v", engine.dispatch)
	}

	attempt := Attempt{ID: "dispatch-1:1", Number: 1, BackendID: "backend-1"}
	if err := journal.BeginAttempt(context.Background(), plan, attempt); err != nil {
		t.Fatalf("BeginAttempt() error = %v", err)
	}
	if engine.attempt == nil || engine.attempt.AttemptNumber != 1 ||
		engine.attempt.BackendID != "backend-1" || engine.attempt.ProviderID != "provider-1" {
		t.Fatalf("attempt request = %+v", engine.attempt)
	}
	if err := journal.FinishAttempt(context.Background(), plan, AttemptResult{
		Attempt: attempt, State: AttemptKnownZero, ErrorCode: "transport_error",
	}); err != nil {
		t.Fatalf("FinishAttempt() error = %v", err)
	}
	if engine.finish == nil || engine.finish.State != quotaruntime.AttemptEvidenceKnownZero ||
		engine.finish.StatusCode != 0 || engine.finish.ErrorCode != "transport_error" {
		t.Fatalf("finish request = %+v", engine.finish)
	}
}

func TestAuthoritativeAttemptJournalRejectsAmbiguousBackend(t *testing.T) {
	journal, err := NewAuthoritativeAttemptJournal(&attemptEvidenceEngineStub{})
	if err != nil {
		t.Fatal(err)
	}
	plan := Plan{Backends: []Backend{
		{ID: "duplicate", ProviderID: "provider-a"},
		{ID: "duplicate", ProviderID: "provider-b"},
	}}
	if err := journal.BeginAttempt(context.Background(), plan, Attempt{
		ID: "attempt-1", Number: 1, BackendID: "duplicate",
	}); err == nil {
		t.Fatal("BeginAttempt() accepted an ambiguous backend identity")
	}
}
