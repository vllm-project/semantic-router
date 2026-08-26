package backendinvoker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type planResolverStub struct{ plans PlanChain }

func (s planResolverStub) ResolvePlans(context.Context, DispatchCapability) (PlanChain, error) {
	return s.plans, nil
}

type planResolverFunc func(context.Context, DispatchCapability) (PlanChain, error)

func (f planResolverFunc) ResolvePlans(ctx context.Context, capability DispatchCapability) (PlanChain, error) {
	return f(ctx, capability)
}

type observerStub struct{}

func (observerStub) Observe(_ context.Context, _ Plan, _ AttemptResult, response *http.Response) (io.ReadCloser, error) {
	return response.Body, nil
}

func TestHandlerRejectsRequestDigestMismatch(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	keyring := SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))}, MaxLifetime: time.Minute}
	capability := completeTestCapability(DispatchCapability{NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1, AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64), Candidates: []DispatchCandidate{testDispatchCandidate("dsp", "mdl", 1)}, RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", []byte(`{"ok":true}`)), Method: "POST", Path: "/v1/chat/completions", Audience: "backend-invoker", IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix()})
	token, err := keyring.Sign(capability, now)
	if err != nil {
		t.Fatal(err)
	}
	handler := &Handler{Audience: "backend-invoker", Keyring: keyring, Plans: planResolverStub{}, Invoker: &Invoker{}, Observer: observerStub{}, Now: func() time.Time { return now }}
	request := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"ok":false}`))
	request.Header.Set(DispatchCapabilityHeader, token)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("status=%d", response.Code)
	}
	outcome, err := keyring.VerifyOutcome(response.Header().Get(DispatchOutcomeHeader), capability.Audience, now)
	if err != nil {
		t.Fatalf("verify pre-dispatch outcome: %v", err)
	}
	if outcome.RequestID != capability.RequestID || len(outcome.Attempted) != 0 {
		t.Fatalf("pre-dispatch outcome = %+v", outcome)
	}
}

func TestHandlerDoesNotSignOutcomeForUnverifiedCapability(t *testing.T) {
	handler := &Handler{
		Audience: "backend-invoker",
		Plans:    planResolverStub{}, Invoker: &Invoker{}, Observer: observerStub{},
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	request.Header.Set(DispatchCapabilityHeader, "not-a-capability")
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusUnauthorized)
	}
	if values := response.Header().Values(DispatchOutcomeHeader); len(values) != 0 {
		t.Fatalf("unverified request received dispatch outcome headers %#v", values)
	}
}

func TestHandlerInvokesResolvedPinnedPlan(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	body := []byte(`{"model":"vllm-sr/blend","messages":[{"role":"user","content":"hello"}]}`)
	keyring := SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))}, MaxLifetime: time.Minute}
	capability := completeTestCapability(DispatchCapability{NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1, AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64), Candidates: []DispatchCandidate{testDispatchCandidate("dsp", "mdl", 1)}, RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", body), Method: "POST", Path: "/v1/chat/completions", Audience: "backend-invoker", IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix()})
	token, err := keyring.Sign(capability, now)
	if err != nil {
		t.Fatal(err)
	}
	journal := &journalStub{}
	plan := testPlan()
	plan.RoutingRevision = capability.RoutingRevision
	plan.Ordinal = 0
	plan.QuotaPartition = capability.QuotaPartition
	plan.DispatchPlanDigest = capability.Candidates[0].DispatchPlanDigest
	invoker := &Invoker{Journal: journal, Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{
			"Content-Type":         {"application/json"},
			DispatchOutcomeHeader:  {"provider-forged"},
			"X-VSR-Internal-State": {"provider-forged"},
			"X-Vllm-Sr-User-ID":    {"provider-forged"},
			"X-Authz-User-ID":      {"provider-forged"},
		}, Body: io.NopCloser(strings.NewReader(testChatResponseBody))}, nil
	})}
	handler := &Handler{Audience: "backend-invoker", Keyring: keyring, Plans: planResolverStub{PlanChain{Candidates: []Plan{plan}}}, Invoker: invoker, Observer: observerStub{}, Now: func() time.Time { return now }}
	request := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(body)))
	request.Header.Set(DispatchCapabilityHeader, token)
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"content":"ok"`) {
		t.Fatalf("response=%d %s", response.Code, response.Body.String())
	}
	if _, err := keyring.VerifyOutcome(response.Header().Get(DispatchOutcomeHeader), "backend-invoker", now); err != nil {
		t.Fatalf("dispatch outcome did not verify: %v", err)
	}
	for _, name := range []string{"X-VSR-Internal-State", "X-Vllm-Sr-User-ID", "X-Authz-User-ID"} {
		if value := response.Header().Get(name); value != "" {
			t.Fatalf("reserved provider header %q leaked as %q", name, value)
		}
	}
}

func TestHandlerFailureReturnsOneSignedDispatchOutcome(t *testing.T) {
	now := time.Unix(1_700_000_000, 0).UTC()
	body := []byte(`{"model":"vllm-sr/blend","messages":[{"role":"user","content":"hello"}]}`)
	keyring := SigningKeyring{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
		MaxLifetime: time.Minute,
	}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1,
		AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64),
		Candidates:    []DispatchCandidate{testDispatchCandidate("dsp", "mdl", 1)},
		RequestDigest: RequestDigest(http.MethodPost, "/v1/chat/completions", "", body),
		Method:        http.MethodPost, Path: "/v1/chat/completions", Audience: "backend-invoker",
		IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix(),
	})
	token, err := keyring.Sign(capability, now)
	if err != nil {
		t.Fatal(err)
	}
	plan := testPlan()
	plan.Execution.MaxRetries = 0
	plan.Execution.RetryOn = nil
	invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
		return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
	})}
	handler := &Handler{
		Audience: "backend-invoker", Keyring: keyring,
		Plans:   planResolverStub{PlanChain{Candidates: []Plan{plan}}},
		Invoker: invoker, Observer: observerStub{}, Now: func() time.Time { return now },
	}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
	request.Header.Set(DispatchCapabilityHeader, token)
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d: %s", response.Code, http.StatusBadGateway, response.Body.String())
	}
	values := response.Header().Values(DispatchOutcomeHeader)
	if len(values) != 1 {
		t.Fatalf("dispatch outcome headers = %#v, want exactly one", values)
	}
	outcome, err := keyring.VerifyOutcome(values[0], capability.Audience, now)
	if err != nil {
		t.Fatalf("verify dispatch outcome: %v", err)
	}
	if outcome.SelectedDispatchID != "" || len(outcome.Attempted) != 1 || outcome.Attempted[0].State != AttemptKnownZero {
		t.Fatalf("dispatch outcome = %+v", outcome)
	}
}

func TestHandlerRejectsResolvedCandidateSubstitutionBeforeDispatch(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	body := []byte(`{}`)
	keyring := SigningKeyring{ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))}, MaxLifetime: time.Minute}
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1,
		AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64),
		Candidates:    []DispatchCandidate{testDispatchCandidate("dsp", "mdl", 1)},
		RequestDigest: RequestDigest("POST", "/v1/chat/completions", "", body),
		Method:        "POST", Path: "/v1/chat/completions", Audience: "backend-invoker",
		IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix(),
	})
	token, err := keyring.Sign(capability, now)
	if err != nil {
		t.Fatal(err)
	}
	plan := testPlan()
	plan.ModelID = "caller-substituted-model"
	journal := &journalStub{}
	handler := &Handler{
		Audience: "backend-invoker", Keyring: keyring,
		Plans:   planResolverStub{PlanChain{Candidates: []Plan{plan}}},
		Invoker: &Invoker{Journal: journal}, Observer: observerStub{}, Now: func() time.Time { return now },
	}
	request := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(string(body)))
	request.Header.Set(DispatchCapabilityHeader, token)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusUnauthorized || journal.dispatches != 0 {
		t.Fatalf("status=%d dispatches=%d", response.Code, journal.dispatches)
	}
}

func TestHandlerConcurrentSnapshotGapReturnsIsolatedKnownZeroOutcome(t *testing.T) {
	const requestCount = 32
	now := time.Unix(1_700_000_000, 0).UTC()
	body := []byte(`{"model":"vllm-sr/blend","messages":[{"role":"user","content":"hello"}]}`)
	keyring := SigningKeyring{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
		MaxLifetime: time.Minute,
	}
	ready := make(chan struct{}, requestCount)
	release := make(chan struct{})
	resolver := planResolverFunc(func(ctx context.Context, capability DispatchCapability) (PlanChain, error) {
		ready <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
			return PlanChain{}, ctx.Err()
		}
		if capability.RequestID == "request-gap" {
			return PlanChain{}, errors.New("exact routing snapshot is not available on this replica")
		}
		return PlanChain{Candidates: []Plan{planForCapability(capability)}}, nil
	})
	journal := &concurrentJournal{}
	var physicalAttempts atomic.Int64
	handler := &Handler{
		Audience: "backend-invoker", Keyring: keyring, Plans: resolver,
		Invoker: &Invoker{
			Journal: journal,
			Transport: transportFunc(func(*http.Request) (*http.Response, error) {
				physicalAttempts.Add(1)
				return &http.Response{
					StatusCode: http.StatusOK, Header: make(http.Header),
					Body: io.NopCloser(strings.NewReader(testChatResponseBody)),
				}, nil
			}),
		},
		Observer: observerStub{}, Now: func() time.Time { return now },
	}
	results := make(chan concurrentHandlerResult, requestCount)
	var workers sync.WaitGroup
	for index := 0; index < requestCount; index++ {
		requestID := fmt.Sprintf("request-%02d", index)
		if index == 0 {
			requestID = "request-gap"
		}
		capability := completeTestCapability(DispatchCapability{
			NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1,
			AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64),
			Candidates: []DispatchCandidate{testDispatchCandidate(
				fmt.Sprintf("dispatch-%02d", index), "mdl", 1,
			)},
			RequestDigest: RequestDigest(http.MethodPost, "/v1/chat/completions", "", body),
			Method:        http.MethodPost, Path: "/v1/chat/completions", Audience: "backend-invoker",
			IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix(),
		})
		capability.RequestID = requestID
		token, err := keyring.Sign(capability, now)
		if err != nil {
			t.Fatal(err)
		}
		workers.Add(1)
		go func() {
			defer workers.Done()
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))
			request.Header.Set(DispatchCapabilityHeader, token)
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			results <- concurrentHandlerResult{requestID: requestID, response: response}
		}()
	}
	for index := 0; index < requestCount; index++ {
		<-ready
	}
	close(release)
	workers.Wait()
	close(results)

	assertConcurrentSnapshotOutcomes(t, results, keyring, now)
	if got := physicalAttempts.Load(); got != requestCount-1 {
		t.Fatalf("physical attempts = %d, want %d", got, requestCount-1)
	}
	if got := journal.dispatches.Load(); got != requestCount-1 {
		t.Fatalf("journaled dispatches = %d, want %d", got, requestCount-1)
	}
}

type concurrentHandlerResult struct {
	requestID string
	response  *httptest.ResponseRecorder
}

func assertConcurrentSnapshotOutcomes(
	t *testing.T,
	results <-chan concurrentHandlerResult,
	keyring SigningKeyring,
	now time.Time,
) {
	t.Helper()
	for completed := range results {
		values := completed.response.Header().Values(DispatchOutcomeHeader)
		if len(values) != 1 {
			t.Fatalf("request %s outcome headers = %#v, want exactly one", completed.requestID, values)
		}
		outcome, err := keyring.VerifyOutcome(values[0], "backend-invoker", now)
		if err != nil {
			t.Fatalf("request %s outcome verification: %v", completed.requestID, err)
		}
		if outcome.RequestID != completed.requestID {
			t.Fatalf("request %s received outcome for %s", completed.requestID, outcome.RequestID)
		}
		if completed.requestID == "request-gap" {
			if completed.response.Code != http.StatusNotFound || len(outcome.Attempted) != 0 {
				t.Fatalf("snapshot-gap response = %d, outcome = %+v", completed.response.Code, outcome)
			}
			continue
		}
		if completed.response.Code != http.StatusOK || len(outcome.Attempted) != 1 ||
			outcome.SelectedDispatchID == "" || outcome.Attempted[0].State != AttemptResponseStarted {
			t.Fatalf("request %s response = %d, outcome = %+v", completed.requestID, completed.response.Code, outcome)
		}
	}
}

type concurrentJournal struct {
	dispatches atomic.Int64
}

func (journal *concurrentJournal) BeginDispatch(context.Context, Plan, time.Time) error {
	journal.dispatches.Add(1)
	return nil
}

func (*concurrentJournal) BeginAttempt(context.Context, Plan, Attempt) error { return nil }

func (*concurrentJournal) FinishAttempt(context.Context, Plan, AttemptResult) error { return nil }

func planForCapability(capability DispatchCapability) Plan {
	plan := testPlan()
	candidate := capability.Candidates[0]
	plan.NamespaceID = capability.NamespaceID
	plan.QuotaPartition = capability.QuotaPartition
	plan.PublicationID = capability.PublicationID
	plan.RuntimeEpoch = capability.RuntimeEpoch
	plan.RoutingRevision = capability.RoutingRevision
	plan.RoutingDigest = capability.RoutingDigest
	plan.AdmissionID = capability.AdmissionID
	plan.AdmissionDigest = capability.AdmissionDigest
	plan.RequestID = capability.RequestID
	plan.DispatchID = candidate.DispatchID
	plan.DispatchType = candidate.DispatchType
	plan.Ordinal = candidate.Ordinal
	plan.Priority = candidate.Priority
	plan.DispatchPlanDigest = candidate.DispatchPlanDigest
	plan.ModelID = candidate.ModelID
	plan.ModelRevision = candidate.ModelRevision
	plan.Execution.MaxRetries = 0
	plan.Execution.RetryOn = nil
	return plan
}
