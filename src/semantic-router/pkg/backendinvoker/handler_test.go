package backendinvoker

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

type planResolverStub struct{ plans PlanChain }

func (s planResolverStub) ResolvePlans(context.Context, DispatchCapability) (PlanChain, error) {
	return s.plans, nil
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
