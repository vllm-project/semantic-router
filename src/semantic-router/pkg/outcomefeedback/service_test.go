package outcomefeedback

import (
	"context"
	"errors"
	"testing"
	"time"
)

type serviceRepositoryStub struct {
	receipt Receipt
	err     error
	calls   int
	caller  Caller
	key     string
	request Request
}

func (stub *serviceRepositoryStub) Record(_ context.Context, caller Caller, key string, request Request) (Receipt, error) {
	stub.calls++
	stub.caller, stub.key, stub.request = caller, key, request
	return stub.receipt, stub.err
}

type abuseLimiterStub struct {
	decision AbuseDecision
	err      error
	calls    int
}

func (stub *abuseLimiterStub) Allow(context.Context, Caller) (AbuseDecision, error) {
	stub.calls++
	return stub.decision, stub.err
}

func TestServiceSubmitsThroughDedicatedAbuseBudget(t *testing.T) {
	repository := &serviceRepositoryStub{receipt: Receipt{ID: "receipt-001", ReplayID: "replay-001", ProjectionRevision: 1}}
	limiter := &abuseLimiterStub{decision: AbuseDecision{Allowed: true}}
	service, err := NewService(ServiceOptions{Repository: repository, Limiter: limiter})
	if err != nil {
		t.Fatal(err)
	}
	caller := validCaller()
	request := Request{ReplayID: "replay-001", Target: TargetRoute, Verdict: VerdictGoodFit}
	receipt, err := service.Submit(context.Background(), caller, "outcome-001", request)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.ID != "receipt-001" || limiter.calls != 1 || repository.calls != 1 ||
		repository.caller != caller || repository.key != "outcome-001" {
		t.Fatalf("submission = receipt %+v, limiter calls %d, repository %+v", receipt, limiter.calls, repository)
	}
}

func TestServiceRejectsAtAbuseBoundaryWithoutRecording(t *testing.T) {
	repository := &serviceRepositoryStub{}
	limiter := &abuseLimiterStub{decision: AbuseDecision{Allowed: false, RetryAfter: 1500 * time.Millisecond}}
	service, err := NewService(ServiceOptions{Repository: repository, Limiter: limiter})
	if err != nil {
		t.Fatal(err)
	}
	_, err = service.Submit(context.Background(), validCaller(), "outcome-001", Request{
		ReplayID: "replay-001", Target: TargetRoute, Verdict: VerdictGoodFit,
	})
	var limited *RateLimitError
	if !errors.As(err, &limited) || limited.RetryAfter != 1500*time.Millisecond {
		t.Fatalf("Submit() error = %v, want RateLimitError", err)
	}
	if repository.calls != 0 {
		t.Fatalf("repository calls = %d, want zero", repository.calls)
	}
}

func TestServiceFailsClosedWhenAbuseStoreUnavailable(t *testing.T) {
	repository := &serviceRepositoryStub{}
	limiter := &abuseLimiterStub{err: errors.New("store unavailable")}
	service, err := NewService(ServiceOptions{Repository: repository, Limiter: limiter})
	if err != nil {
		t.Fatal(err)
	}
	_, err = service.Submit(context.Background(), validCaller(), "outcome-001", Request{
		ReplayID: "replay-001", Target: TargetRoute, Verdict: VerdictGoodFit,
	})
	if !errors.Is(err, ErrUnavailable) || repository.calls != 0 {
		t.Fatalf("Submit() = %v, repository calls %d", err, repository.calls)
	}
}

func validCaller() Caller {
	return Caller{
		NamespaceID: "00000000-0000-4000-8000-000000000001",
		APIKeyID:    "00000000-0000-4000-8000-000000000002",
		UserID:      "00000000-0000-4000-8000-000000000003",
		TeamID:      "00000000-0000-4000-8000-000000000004",
		Source:      SourceAPIKey,
	}
}
