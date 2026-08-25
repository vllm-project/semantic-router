package backendinvoker

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptrace"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type journalStub struct {
	dispatches int
	plans      []Plan
	attempts   []AttemptResult
}

func (j *journalStub) BeginDispatch(_ context.Context, plan Plan, _ time.Time) error {
	j.dispatches++
	j.plans = append(j.plans, plan)
	return nil
}
func (j *journalStub) BeginAttempt(context.Context, Plan, Attempt) error { return nil }
func (j *journalStub) FinishAttempt(_ context.Context, _ Plan, result AttemptResult) error {
	j.attempts = append(j.attempts, result)
	return nil
}

type transportFunc func(*http.Request) (*http.Response, error)

func (f transportFunc) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

type closeCountingBody struct {
	io.Reader
	closes int
}

func (body *closeCountingBody) Close() error {
	body.closes++
	return nil
}

func testPlan() Plan {
	body := []byte(`{"model":"public-model","messages":[{"role":"user","content":"hello"}]}`)
	return completeTestPlan(Plan{
		NamespaceID: "ns", QuotaPartition: "partition", RoutingRevision: 1,
		AdmissionID: "adm", AdmissionDigest: strings.Repeat("b", 64),
		DispatchID: "dsp", DispatchType: "primary", Ordinal: 0,
		DispatchPlanDigest: strings.Repeat("a", 64), ModelID: "mdl", ModelRevision: 1,
		Method: http.MethodPost, Path: "/v1/chat/completions", Headers: http.Header{"Authorization": {"Bearer caller"}}, Body: body,
		Execution: Execution{
			MaxRetries: 1, RetryOn: []FallbackTrigger{FallbackUnavailable},
			RequestTimeout: time.Minute, StreamTimeout: time.Minute,
		},
		RequestDigest: RequestDigest(http.MethodPost, "/v1/chat/completions", "", body),
		Backends: []Backend{{
			ID: "be", Origin: "https://models.example", ProviderID: "openai",
			WireFormat: "openai.chat.v1", ProviderModelID: "m",
			Connection: Connection{Path: "/v1/chat/completions"}, Weight: 1,
		}},
	})
}

func TestValidatePlanRequiresWireFormat(t *testing.T) {
	plan := testPlan()
	plan.Backends[0].WireFormat = ""
	if err := validatePlan(plan); err == nil || !strings.Contains(err.Error(), "wire format") {
		t.Fatalf("validatePlan() error = %v, want missing wire format", err)
	}
}

func TestSanitizedHeadersStripsEveryRouterOwnedPrefix(t *testing.T) {
	headers := sanitizedHeaders(http.Header{
		"X-Vsr-Future-Internal":  {"secret-context"},
		"X-Vllm-Sr-Future-Claim": {"identity-context"},
		"X-Authz-Future-Claim":   {"authorization-context"},
		"X-User-OpenAI-Key":      {"caller-provider-secret"},
		"X-Goog-Api-Key":         {"caller-google-secret"},
		"X-Request-ID":           {"request-id"},
	})
	if got := headers.Get("X-Request-ID"); got != "request-id" {
		t.Fatalf("safe header = %q", got)
	}
	for _, key := range []string{
		"X-Vsr-Future-Internal",
		"X-Vllm-Sr-Future-Claim",
		"X-Authz-Future-Claim",
		"X-User-OpenAI-Key",
		"X-Goog-Api-Key",
	} {
		if got := headers.Get(key); got != "" {
			t.Fatalf("Router-owned header %q leaked as %q", key, got)
		}
	}
}

func TestInvokerRetriesOnlyKnownZeroTransportFailure(t *testing.T) {
	journal := &journalStub{}
	calls := 0
	invoker := &Invoker{Journal: journal, Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
		calls++
		if request.Header.Get("Authorization") != "" {
			t.Fatal("caller authorization leaked")
		}
		if calls == 1 {
			return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(testChatResponseBody))}, nil
	})}
	result, err := invoker.Invoke(context.Background(), testPlan())
	if err != nil {
		t.Fatal(err)
	}
	defer result.Response.Body.Close()
	if calls != 2 || len(journal.attempts) != 2 {
		t.Fatalf("calls=%d attempts=%d", calls, len(journal.attempts))
	}
	if journal.attempts[0].State != AttemptKnownZero || journal.attempts[1].State != AttemptResponseStarted {
		t.Fatalf("unexpected evidence: %#v", journal.attempts)
	}
}

func TestInvokerDoesNotRetryKnownZeroFailureForDisabledTrigger(t *testing.T) {
	journal := &journalStub{}
	calls := 0
	plan := testPlan()
	plan.Execution.RetryOn = []FallbackTrigger{FallbackTimeout}
	invoker := &Invoker{Journal: journal, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
		calls++
		return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
	})}

	result, err := invoker.Invoke(context.Background(), plan)
	if err == nil {
		t.Fatal("Invoke() unexpectedly retried a disabled failure trigger")
	}
	if calls != 1 || len(journal.attempts) != 1 || len(result.Outcomes) != 1 ||
		result.Outcomes[0].State != AttemptKnownZero {
		t.Fatalf("calls=%d attempts=%+v outcomes=%+v", calls, journal.attempts, result.Outcomes)
	}
}

func TestInvokerResponseBodyOwnership(t *testing.T) {
	t.Run("transform failure closes source", func(t *testing.T) {
		source := &closeCountingBody{Reader: strings.NewReader(`{"choices":`)}
		invoker := &Invoker{
			Journal: &journalStub{},
			Transport: transportFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: source}, nil
			}),
		}
		if _, err := invoker.Invoke(context.Background(), testPlan()); err == nil {
			t.Fatal("Invoke() unexpectedly accepted a malformed backend response")
		}
		if source.closes != 1 {
			t.Fatalf("source closes = %d, want 1", source.closes)
		}
	})

	t.Run("non-2xx closes source and transfers safe response", func(t *testing.T) {
		source := &closeCountingBody{Reader: strings.NewReader(`{"error":"busy"}`)}
		invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusServiceUnavailable, Header: make(http.Header), Body: source}, nil
		})}
		result, err := invoker.Invoke(context.Background(), testPlan())
		if err != nil {
			t.Fatal(err)
		}
		if source.closes != 1 {
			t.Fatalf("source closes before returned body close = %d, want 1", source.closes)
		}
		if err := result.Response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		if source.closes != 1 {
			t.Fatalf("source closes after returned body close = %d, want 1", source.closes)
		}
	})

	t.Run("streaming transfers source until caller close", func(t *testing.T) {
		source := &closeCountingBody{Reader: strings.NewReader(chatStreamFixture())}
		plan := testPlan()
		plan.Streaming = true
		invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       source,
			}, nil
		})}
		result, err := invoker.Invoke(context.Background(), plan)
		if err != nil {
			t.Fatal(err)
		}
		if source.closes != 0 {
			t.Fatalf("source closes before caller close = %d, want 0", source.closes)
		}
		if err := result.Response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		if source.closes != 1 {
			t.Fatalf("source closes after caller close = %d, want 1", source.closes)
		}
	})

	t.Run("retry transfers successful response", func(t *testing.T) {
		source := &closeCountingBody{Reader: strings.NewReader(testChatResponseBody)}
		calls := 0
		invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
			calls++
			if calls == 1 {
				return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
			}
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: source}, nil
		})}
		result, err := invoker.Invoke(context.Background(), testPlan())
		if err != nil {
			t.Fatal(err)
		}
		if calls != 2 || source.closes != 1 {
			t.Fatalf("calls = %d, source closes before caller close = %d, want 2 and 1", calls, source.closes)
		}
		if err := result.Response.Body.Close(); err != nil {
			t.Fatal(err)
		}
		if source.closes != 1 {
			t.Fatalf("source closes after caller close = %d, want 1", source.closes)
		}
	})
}

func TestInvokerDoesNotRetryAfterRequestWasWritten(t *testing.T) {
	journal := &journalStub{}
	calls := 0
	invoker := &Invoker{Journal: journal, Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
		calls++
		trace := httptrace.ContextClientTrace(request.Context())
		trace.WroteRequest(httptrace.WroteRequestInfo{})
		return nil, errors.New("connection reset")
	})}
	_, err := invoker.Invoke(context.Background(), testPlan())
	if err == nil {
		t.Fatal("expected transport failure")
	}
	if calls != 1 || len(journal.attempts) != 1 || journal.attempts[0].State != AttemptUnknown {
		t.Fatalf("unsafe retry or evidence: calls=%d attempts=%#v", calls, journal.attempts)
	}
}

func TestInvokerDoesNotInferKnownZeroFromMissingTraceCallbacks(t *testing.T) {
	chain := fallbackTestChain(2)
	chain.Fallback.On = []FallbackTrigger{FallbackUnavailable}
	calls := 0
	invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
		calls++
		return nil, errors.New("ambiguous transport failure")
	})}
	result, err := invoker.InvokeChain(context.Background(), chain)
	if err == nil || calls != 1 || len(result.Outcomes) != 1 ||
		result.Outcomes[0].State != AttemptUnknown || result.Outcomes[0].FallbackTrigger != "" {
		t.Fatalf("error=%v calls=%d outcomes=%+v", err, calls, result.Outcomes)
	}
}

type timeoutTransportError struct{}

func (timeoutTransportError) Error() string   { return "transport timeout" }
func (timeoutTransportError) Timeout() bool   { return true }
func (timeoutTransportError) Temporary() bool { return true }

func TestInvokerDoesNotFallbackAfterUnsafeTimeoutOrPartialResponse(t *testing.T) {
	for name, traceAttempt := range map[string]func(*httptrace.ClientTrace){
		"request written": func(trace *httptrace.ClientTrace) {
			trace.WroteRequest(httptrace.WroteRequestInfo{})
		},
		"response started": func(trace *httptrace.ClientTrace) {
			trace.GotFirstResponseByte()
		},
	} {
		t.Run(name, func(t *testing.T) {
			journal := &journalStub{}
			calls := 0
			chain := fallbackTestChain(2)
			chain.Fallback.On = []FallbackTrigger{FallbackTimeout}
			invoker := &Invoker{Journal: journal, Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
				calls++
				traceAttempt(httptrace.ContextClientTrace(request.Context()))
				return nil, NewKnownZeroTransportFailure(FallbackTimeout, timeoutTransportError{})
			})}
			result, err := invoker.InvokeChain(context.Background(), chain)
			if err == nil {
				t.Fatal("unsafe timeout unexpectedly succeeded")
			}
			if calls != 1 || len(result.Outcomes) != 1 || result.Outcomes[0].State != AttemptUnknown {
				t.Fatalf("calls=%d outcomes=%+v", calls, result.Outcomes)
			}
		})
	}
}

func TestInvokerExhaustsSameModelRetriesBeforeFallbackAndReportsOnlyAttempts(t *testing.T) {
	journal := &journalStub{}
	chain := fallbackTestChain(3)
	chain.Candidates[0].Execution.MaxRetries = 1
	chain.Candidates[0].Execution.RetryOn = []FallbackTrigger{FallbackUnavailable}
	chain.Fallback.On = []FallbackTrigger{FallbackUnavailable}
	calls := make([]string, 0, 3)
	invoker := &Invoker{Journal: journal, Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
		calls = append(calls, request.URL.Host)
		if len(calls) <= 2 {
			return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(testChatResponseBody))}, nil
	})}
	result, err := invoker.InvokeChain(context.Background(), chain)
	if err != nil {
		t.Fatal(err)
	}
	defer result.Response.Body.Close()
	if len(calls) != 3 || calls[0] != "model-0.example" || calls[1] != "model-0.example" || calls[2] != "model-1.example" {
		t.Fatalf("call order = %+v", calls)
	}
	if len(result.Outcomes) != 2 || len(result.Outcomes[0].Attempts) != 2 || len(result.Outcomes[1].Attempts) != 1 {
		t.Fatalf("attempted outcomes = %+v", result.Outcomes)
	}
	if result.Selected == nil || result.Selected.ModelID != "model-1" {
		t.Fatalf("selected plan = %+v", result.Selected)
	}
	if journal.dispatches != 2 {
		t.Fatalf("journaled dispatches = %d, want only two attempted candidates", journal.dispatches)
	}
}

func TestInvokerDoesNotFallbackWhenTriggerDisabled(t *testing.T) {
	chain := fallbackTestChain(2)
	calls := 0
	invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
		calls++
		return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
	})}
	result, err := invoker.InvokeChain(context.Background(), chain)
	if err == nil || calls != 1 || len(result.Outcomes) != 1 {
		t.Fatalf("error=%v calls=%d outcomes=%+v", err, calls, result.Outcomes)
	}
}

func TestInvokerKnownZeroFailureHasNoSelectedPlanAndProducesSignedOutcome(t *testing.T) {
	now := time.Unix(1_800_000_000, 0).UTC()
	chain := fallbackTestChain(1)
	invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
		return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
	})}

	result, err := invoker.InvokeChain(context.Background(), chain)
	if err == nil {
		t.Fatal("expected transport failure")
	}
	if result.Selected != nil {
		t.Fatalf("selected plan = %+v, want nil for a known-zero failure", result.Selected)
	}
	if len(result.Outcomes) != 1 || result.Outcomes[0].State != AttemptKnownZero {
		t.Fatalf("outcomes = %+v", result.Outcomes)
	}

	plan := chain.Candidates[0]
	capability := completeTestCapability(DispatchCapability{
		NamespaceID: plan.NamespaceID, QuotaPartition: plan.QuotaPartition,
		RoutingRevision: plan.RoutingRevision, AdmissionID: plan.AdmissionID,
		AdmissionDigest: plan.AdmissionDigest, Candidates: []DispatchCandidate{candidateFromPlan(plan)},
		RequestDigest: plan.RequestDigest, Method: plan.Method, Path: plan.Path,
		Audience: "backend-invoker", IssuedAt: now.Unix(), ExpiresAt: now.Add(time.Minute).Unix(),
	})
	keyring := SigningKeyring{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
		MaxLifetime: time.Minute,
	}
	outcome, err := outcomeForResult(capability, result, now, keyring.MaxLifetime)
	if err != nil {
		t.Fatalf("build dispatch outcome: %v", err)
	}
	token, err := keyring.SignOutcome(outcome, now)
	if err != nil {
		t.Fatalf("sign dispatch outcome: %v", err)
	}
	verified, err := keyring.VerifyOutcome(token, capability.Audience, now)
	if err != nil {
		t.Fatalf("verify dispatch outcome: %v", err)
	}
	if verified.SelectedDispatchID != "" || len(verified.Attempted) != 1 || verified.Attempted[0].State != AttemptKnownZero {
		t.Fatalf("verified outcome = %+v", verified)
	}
}

func TestInvokerRequestBuildFailureHasNoSelectedPlan(t *testing.T) {
	plan := testPlan()
	plan.Body = []byte(`{"messages":`)
	plan.RequestDigest = RequestDigest(plan.Method, plan.Path, plan.Query, plan.Body)
	invoker := &Invoker{Journal: &journalStub{}}

	result, err := invoker.Invoke(context.Background(), plan)
	if err == nil {
		t.Fatal("expected request translation failure")
	}
	if result.Selected != nil {
		t.Fatalf("selected plan = %+v, want nil before any response starts", result.Selected)
	}
	if len(result.Outcomes) != 1 || result.Outcomes[0].State != AttemptKnownZero || result.Attempt.ErrorCode != "request_build_failed" {
		t.Fatalf("result = %+v", result)
	}
}

func TestInvokerNeverTreatsHTTPFailureResponseAsKnownZero(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, http.StatusServiceUnavailable} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			chain := fallbackTestChain(2)
			chain.Fallback.On = []FallbackTrigger{FallbackUnavailable}
			calls := 0
			invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(*http.Request) (*http.Response, error) {
				calls++
				return &http.Response{StatusCode: status, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"error":"busy"}`))}, nil
			})}
			result, err := invoker.InvokeChain(context.Background(), chain)
			if err != nil {
				t.Fatal(err)
			}
			defer result.Response.Body.Close()
			if calls != 1 || len(result.Outcomes) != 1 || result.Outcomes[0].State != AttemptResponseStarted || result.Selected.ModelID != "model-0" {
				t.Fatalf("calls=%d result=%+v", calls, result)
			}
			if result.Attempt.StatusCode != status || result.Attempt.ErrorCode != "" {
				t.Fatalf("attempt evidence = %+v, want authoritative status without a transport error code", result.Attempt)
			}
		})
	}
}

func TestInvokerUsesOneSharedDeadlineAcrossFallbackCandidates(t *testing.T) {
	chain := fallbackTestChain(2)
	chain.Fallback.On = []FallbackTrigger{FallbackUnavailable}
	chain.Candidates[0].Execution.RequestTimeout = time.Minute
	chain.Candidates[1].Execution.RequestTimeout = 10 * time.Minute
	deadlines := make([]time.Time, 0, 2)
	invoker := &Invoker{Journal: &journalStub{}, Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
		deadline, ok := request.Context().Deadline()
		if !ok {
			t.Fatal("backend request has no deadline")
		}
		deadlines = append(deadlines, deadline)
		if len(deadlines) == 1 {
			return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(testChatResponseBody))}, nil
	})}
	result, err := invoker.InvokeChain(context.Background(), chain)
	if err != nil {
		t.Fatal(err)
	}
	defer result.Response.Body.Close()
	if len(deadlines) != 2 || !deadlines[0].Equal(deadlines[1]) {
		t.Fatalf("candidate deadlines = %+v", deadlines)
	}
}

type isolatedCredentialResolver struct {
	pinned []string
}

func (resolver *isolatedCredentialResolver) Pin(_ context.Context, _ CredentialPublication, id, _, _ string) (string, error) {
	resolver.pinned = append(resolver.pinned, id)
	return "version-" + id, nil
}

func (*isolatedCredentialResolver) ResolvePinned(_ context.Context, _ CredentialPublication, id, version, _, _ string) (Credential, error) {
	return Credential{Header: "Authorization", Prefix: "Bearer ", Secret: "secret-" + id, Version: version}, nil
}

func TestInvokerPinsCredentialsOnlyForAttemptedCandidate(t *testing.T) {
	chain := fallbackTestChain(3)
	chain.Fallback.On = []FallbackTrigger{FallbackUnavailable}
	for index := range chain.Candidates {
		chain.Candidates[index].Backends[0].ProviderCredentialID = "credential-" + strconv.Itoa(index)
	}
	resolver := &isolatedCredentialResolver{}
	calls := 0
	invoker := &Invoker{Journal: &journalStub{}, Credentials: resolver, Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
		calls++
		want := "Bearer secret-credential-" + strconv.Itoa(calls-1)
		if request.Header.Get("Authorization") != want {
			t.Fatalf("credential = %q, want %q", request.Header.Get("Authorization"), want)
		}
		if calls == 1 {
			return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
		}
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(testChatResponseBody))}, nil
	})}
	result, err := invoker.InvokeChain(context.Background(), chain)
	if err != nil {
		t.Fatal(err)
	}
	defer result.Response.Body.Close()
	if strings.Join(resolver.pinned, ",") != "credential-0,credential-1" {
		t.Fatalf("pinned credentials = %+v", resolver.pinned)
	}
}

func fallbackTestChain(count int) PlanChain {
	base := testPlan()
	base.Execution.MaxRetries = 0
	base.Execution.RetryOn = nil
	chain := PlanChain{Candidates: make([]Plan, 0, count)}
	for index := 0; index < count; index++ {
		plan := base
		plan.DispatchID = "dispatch-" + strconv.Itoa(index)
		plan.Ordinal = index
		plan.Priority = index
		plan.DispatchPlanDigest = strings.Repeat(string(rune('a'+index)), 64)
		plan.ModelID = "model-" + strconv.Itoa(index)
		plan.ModelRevision = int64(index + 1)
		plan.Backends = append([]Backend(nil), base.Backends...)
		plan.Backends[0].ID = "backend-" + strconv.Itoa(index)
		plan.Backends[0].Origin = "https://model-" + strconv.Itoa(index) + ".example"
		chain.Candidates = append(chain.Candidates, plan)
	}
	return chain
}

func TestInvokerInjectsOnlyProviderCredential(t *testing.T) {
	journal := &journalStub{}
	plan := testPlan()
	plan.Backends[0].ProviderCredentialID = "cred"
	resolver := &credentialResolverStub{
		pinVersion: "version-1",
		credential: Credential{
			Header: "Authorization", Prefix: "Bearer ", Secret: "provider", Version: "version-1",
			Extra: http.Header{"X-Provider-Account": {"operator-account"}},
		},
	}
	plan.Headers.Set("X-Provider-Account", "caller-account")
	invoker := &Invoker{
		Journal:     journal,
		Credentials: resolver,
		Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
			if request.Header.Get("Authorization") != "Bearer provider" {
				t.Fatalf("authorization = %q", request.Header.Get("Authorization"))
			}
			if values := request.Header.Values("X-Provider-Account"); len(values) != 1 || values[0] != "operator-account" {
				t.Fatalf("provider account header = %#v", values)
			}
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(testChatResponseBody))}, nil
		}),
	}
	result, err := invoker.Invoke(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	result.Response.Body.Close()
	if resolver.pinCalls != 1 || resolver.resolveCalls != 1 {
		t.Fatalf("pin calls=%d resolve calls=%d", resolver.pinCalls, resolver.resolveCalls)
	}
	if len(journal.plans) != 1 || journal.plans[0].Backends[0].ProviderCredentialVersion != "version-1" {
		t.Fatalf("dispatch was not journaled with its pinned credential: %#v", journal.plans)
	}
}

func TestInvokerPinsCredentialVersionAcrossRetries(t *testing.T) {
	journal := &journalStub{}
	plan := testPlan()
	plan.Backends[0].ProviderCredentialID = "cred"
	resolver := &credentialResolverStub{
		pinVersion: "version-before-rotation",
		credential: Credential{
			Header: "Authorization", Prefix: "Bearer ", Secret: "provider", Version: "version-before-rotation",
		},
	}
	calls := 0
	invoker := &Invoker{
		Journal:     journal,
		Credentials: resolver,
		Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
			calls++
			if calls == 1 {
				return nil, NewKnownZeroTransportFailure(FallbackUnavailable, errors.New("dial failed"))
			}
			return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(testChatResponseBody))}, nil
		}),
	}
	result, err := invoker.Invoke(context.Background(), plan)
	if err != nil {
		t.Fatal(err)
	}
	result.Response.Body.Close()
	if resolver.pinCalls != 1 || resolver.resolveCalls != 2 {
		t.Fatalf("pin calls=%d resolve calls=%d", resolver.pinCalls, resolver.resolveCalls)
	}
	for _, version := range resolver.resolvedVersions {
		if version != "version-before-rotation" {
			t.Fatalf("retry resolved version %q", version)
		}
	}
	if len(journal.plans) != 1 || journal.plans[0].Backends[0].ProviderCredentialVersion != "version-before-rotation" {
		t.Fatalf("journal plan = %#v", journal.plans)
	}
}

func TestInvokerRejectsCredentialVersionSubstitution(t *testing.T) {
	journal := &journalStub{}
	plan := testPlan()
	plan.Backends[0].ProviderCredentialID = "cred"
	resolver := &credentialResolverStub{
		pinVersion: "pinned-version",
		credential: Credential{
			Header: "Authorization", Prefix: "Bearer ", Secret: "provider", Version: "different-version",
		},
	}
	invoker := &Invoker{
		Journal:     journal,
		Credentials: resolver,
		Transport: transportFunc(func(*http.Request) (*http.Response, error) {
			t.Fatal("transport must not receive a substituted credential")
			return nil, nil
		}),
	}
	if _, err := invoker.Invoke(context.Background(), plan); err == nil || !strings.Contains(err.Error(), "different version") {
		t.Fatalf("substitution error = %v", err)
	}
}

func TestInvokerDoesNotJournalAnUnpinnableDispatch(t *testing.T) {
	journal := &journalStub{}
	plan := testPlan()
	plan.Backends[0].ProviderCredentialID = "cred"
	resolver := &credentialResolverStub{pinErr: errors.New("disabled")}
	invoker := &Invoker{Journal: journal, Credentials: resolver}
	if _, err := invoker.Invoke(context.Background(), plan); err == nil || !strings.Contains(err.Error(), "pin provider credential") {
		t.Fatalf("pin error = %v", err)
	}
	if journal.dispatches != 0 {
		t.Fatalf("journaled unpinnable dispatch %d times", journal.dispatches)
	}
}

func TestValidatePlanRejectsCredentialVersionWithoutCredential(t *testing.T) {
	plan := testPlan()
	plan.Backends[0].ProviderCredentialVersion = "version"
	if err := validatePlan(plan); err == nil {
		t.Fatal("accepted credential version without credential identity")
	}
}

func TestValidatePlanRejectsBackendWeightOverflow(t *testing.T) {
	plan := testPlan()
	plan.Backends = []Backend{
		{ID: "one", Origin: "https://one.example", ProviderID: "openai", WireFormat: llmprotocol.OpenAIChatV1, ProviderModelID: "m", Connection: Connection{Path: "/v1/chat/completions"}, Weight: ^uint64(0)},
		{ID: "two", Origin: "https://two.example", ProviderID: "openai", WireFormat: llmprotocol.OpenAIChatV1, ProviderModelID: "m", Connection: Connection{Path: "/v1/chat/completions"}, Weight: 1},
	}
	if err := validatePlan(plan); err == nil {
		t.Fatal("accepted overflowing backend weights")
	}
}

type credentialResolverStub struct {
	credential       Credential
	pinVersion       string
	pinErr           error
	resolveErr       error
	pinCalls         int
	resolveCalls     int
	resolvedVersions []string
	providers        []string
	origins          []string
}

func (s *credentialResolverStub) Pin(_ context.Context, _ CredentialPublication, _, provider, origin string) (string, error) {
	s.pinCalls++
	s.providers = append(s.providers, provider)
	s.origins = append(s.origins, origin)
	if s.pinErr != nil {
		return "", s.pinErr
	}
	return s.pinVersion, nil
}

func (s *credentialResolverStub) ResolvePinned(_ context.Context, _ CredentialPublication, _, version, provider, origin string) (Credential, error) {
	s.resolveCalls++
	s.resolvedVersions = append(s.resolvedVersions, version)
	s.providers = append(s.providers, provider)
	s.origins = append(s.origins, origin)
	if s.resolveErr != nil {
		return Credential{}, s.resolveErr
	}
	return s.credential, nil
}
